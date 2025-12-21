//! I/O Reactor - Event loop abstraction over mio
//!
//! Provides non-blocking I/O for the fiber scheduler using mio's Poll,
//! which abstracts over epoll (Linux) and kqueue (macOS).

use mio::event::Source;
use mio::{Events, Interest, Poll, Token};
use std::collections::HashMap;
use std::io;
use std::time::Duration;

use crate::runtime::Pid;

/// Unique token for tracking I/O sources
/// We use the fiber's Pid as the token value for simple mapping
pub type IoToken = Token;

/// Interest flags for I/O operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoInterest {
    /// Interested in read readiness
    Read,
    /// Interested in write readiness
    Write,
    /// Interested in both read and write
    ReadWrite,
}

impl From<IoInterest> for Interest {
    fn from(interest: IoInterest) -> Self {
        match interest {
            IoInterest::Read => Interest::READABLE,
            IoInterest::Write => Interest::WRITABLE,
            IoInterest::ReadWrite => Interest::READABLE | Interest::WRITABLE,
        }
    }
}

/// Result of polling the reactor
#[derive(Debug)]
pub struct IoEvent {
    /// The fiber that should be woken
    pub fiber_id: Pid,
    /// Whether the source is readable
    pub readable: bool,
    /// Whether the source is writable
    pub writable: bool,
    /// Whether an error occurred
    pub error: bool,
}

/// Registration info stored for each token
#[derive(Debug)]
struct Registration {
    fiber_id: Pid,
    interest: IoInterest,
}

/// I/O Reactor wrapping mio::Poll
///
/// Maps mio tokens to fiber IDs so the scheduler knows which
/// fiber to wake when I/O is ready.
pub struct IoReactor {
    /// The mio poll instance
    poll: Poll,
    /// Reusable events buffer
    events: Events,
    /// Token -> fiber mapping
    registrations: HashMap<usize, Registration>,
    /// Next token value to assign
    next_token: usize,
}

impl IoReactor {
    /// Create a new I/O reactor
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            poll: Poll::new()?,
            events: Events::with_capacity(1024),
            registrations: HashMap::new(),
            next_token: 0,
        })
    }

    /// Register an I/O source with the reactor
    ///
    /// Returns a token that can be used to deregister later.
    /// The fiber_id is used to map events back to the waiting fiber.
    pub fn register<S: Source>(
        &mut self,
        source: &mut S,
        interest: IoInterest,
        fiber_id: Pid,
    ) -> io::Result<IoToken> {
        let token_val = self.next_token;
        self.next_token += 1;
        let token = Token(token_val);

        self.poll
            .registry()
            .register(source, token, interest.into())?;

        self.registrations.insert(
            token_val,
            Registration {
                fiber_id,
                interest,
            },
        );

        Ok(token)
    }

    /// Re-register an I/O source with new interest
    pub fn reregister<S: Source>(
        &mut self,
        source: &mut S,
        token: IoToken,
        interest: IoInterest,
    ) -> io::Result<()> {
        self.poll
            .registry()
            .reregister(source, token, interest.into())?;

        if let Some(reg) = self.registrations.get_mut(&token.0) {
            reg.interest = interest;
        }

        Ok(())
    }

    /// Deregister an I/O source from the reactor
    pub fn deregister<S: Source>(&mut self, source: &mut S, token: IoToken) -> io::Result<()> {
        self.poll.registry().deregister(source)?;
        self.registrations.remove(&token.0);
        Ok(())
    }

    /// Poll for I/O events with the given timeout
    ///
    /// Returns a list of events indicating which fibers should be woken
    /// and why (readable, writable, or error).
    ///
    /// A timeout of None blocks indefinitely.
    /// A timeout of Some(Duration::ZERO) returns immediately (non-blocking check).
    pub fn poll(&mut self, timeout: Option<Duration>) -> io::Result<Vec<IoEvent>> {
        self.events.clear();
        self.poll.poll(&mut self.events, timeout)?;

        let mut results = Vec::with_capacity(self.events.iter().count());

        for event in self.events.iter() {
            let token_val = event.token().0;
            if let Some(reg) = self.registrations.get(&token_val) {
                results.push(IoEvent {
                    fiber_id: reg.fiber_id,
                    readable: event.is_readable(),
                    writable: event.is_writable(),
                    error: event.is_error() || event.is_read_closed() || event.is_write_closed(),
                });
            }
        }

        Ok(results)
    }

    /// Check if there are any registered sources
    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }

    /// Get the number of registered sources
    pub fn len(&self) -> usize {
        self.registrations.len()
    }

    /// Get the fiber ID associated with a token
    pub fn get_fiber(&self, token: IoToken) -> Option<Pid> {
        self.registrations.get(&token.0).map(|r| r.fiber_id)
    }
}

impl Default for IoReactor {
    fn default() -> Self {
        Self::new().expect("failed to create IoReactor")
    }
}

impl std::fmt::Debug for IoReactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoReactor")
            .field("registrations", &self.registrations.len())
            .field("next_token", &self.next_token)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mio::net::TcpListener;
    use std::net::SocketAddr;

    #[test]
    fn test_reactor_new() {
        let reactor = IoReactor::new().expect("should create reactor");
        assert!(reactor.is_empty());
        assert_eq!(reactor.len(), 0);
    }

    #[test]
    fn test_register_and_deregister() {
        let mut reactor = IoReactor::new().expect("should create reactor");

        // Create a TCP listener to register
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut listener = TcpListener::bind(addr).expect("should bind");

        // Register it
        let token = reactor
            .register(&mut listener, IoInterest::Read, 42)
            .expect("should register");

        assert_eq!(reactor.len(), 1);
        assert_eq!(reactor.get_fiber(token), Some(42));

        // Deregister it
        reactor
            .deregister(&mut listener, token)
            .expect("should deregister");

        assert!(reactor.is_empty());
        assert_eq!(reactor.get_fiber(token), None);
    }

    #[test]
    fn test_poll_no_events() {
        let mut reactor = IoReactor::new().expect("should create reactor");

        // Poll with zero timeout should return immediately with no events
        let events = reactor
            .poll(Some(Duration::ZERO))
            .expect("should poll");

        assert!(events.is_empty());
    }

    #[test]
    fn test_poll_with_ready_listener() {
        let mut reactor = IoReactor::new().expect("should create reactor");

        // Create and register a listener
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut listener = TcpListener::bind(addr).expect("should bind");
        let local_addr = listener.local_addr().expect("should get addr");

        let _token = reactor
            .register(&mut listener, IoInterest::Read, 123)
            .expect("should register");

        // Connect to the listener to make it readable
        let _client = std::net::TcpStream::connect(local_addr).expect("should connect");

        // Give the OS a moment to process
        std::thread::sleep(Duration::from_millis(10));

        // Poll should now return an event
        let events = reactor
            .poll(Some(Duration::from_millis(100)))
            .expect("should poll");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].fiber_id, 123);
        assert!(events[0].readable);
    }
}
