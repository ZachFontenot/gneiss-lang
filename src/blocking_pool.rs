//! Blocking Thread Pool for I/O Operations
//!
//! Handles blocking operations (DNS resolution, blocking file I/O) that
//! would stall the fiber scheduler if run on the main thread.
//!
//! Uses a fixed pool of worker threads that communicate results back
//! to the scheduler via mpsc channels.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::eval::{IoOp, OpenMode};
use crate::runtime::Pid;

/// OS resources that can be stored and accessed by handle ID
pub enum IoResource {
    File(std::fs::File),
    TcpSocket(std::net::TcpStream),
    TcpListener(std::net::TcpListener),
}

/// A task to be executed on the blocking pool
#[derive(Debug)]
pub struct BlockingTask {
    /// The fiber waiting for this operation
    pub fiber_id: Pid,
    /// The I/O operation to perform
    pub op: IoOp,
}

/// Result of a blocking operation
#[derive(Debug)]
pub struct BlockingResult {
    /// The fiber that should be woken
    pub fiber_id: Pid,
    /// The result of the operation
    pub result: BlockingOpResult,
}

/// The actual result of a blocking I/O operation
#[derive(Debug)]
pub enum BlockingOpResult {
    /// File opened successfully, returns handle ID
    FileOpened(u64),
    /// Read completed, returns bytes read
    Read(Vec<u8>),
    /// Write completed, returns bytes written
    Written(usize),
    /// Close completed
    Closed,
    /// TCP connection established, returns socket ID
    TcpConnected(u64),
    /// TCP listener created, returns listener ID
    TcpListening(u64),
    /// TCP connection accepted, returns socket ID
    TcpAccepted(u64),
    /// Sleep completed
    Slept,
    /// Operation failed with an error
    Error(std::io::Error),
}

/// Message sent to worker threads
enum WorkerMessage {
    Task(BlockingTask),
    Shutdown,
}

/// Thread-safe handle registry for files and sockets
struct HandleRegistry {
    next_id: u64,
    resources: HashMap<u64, IoResource>,
}

impl HandleRegistry {
    fn new() -> Self {
        Self {
            next_id: 1,
            resources: HashMap::new(),
        }
    }

    /// Insert a resource and return its ID
    fn insert(&mut self, resource: IoResource) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.resources.insert(id, resource);
        id
    }

    /// Get a mutable reference to a resource by ID
    fn get_mut(&mut self, id: u64) -> Option<&mut IoResource> {
        self.resources.get_mut(&id)
    }

    /// Remove a resource by ID, returning it if it existed
    fn remove(&mut self, id: u64) -> Option<IoResource> {
        self.resources.remove(&id)
    }
}

/// A pool of worker threads for blocking I/O operations
pub struct BlockingPool {
    /// Sender for submitting tasks to workers
    task_sender: Sender<WorkerMessage>,
    /// Receiver for getting results back
    result_receiver: Receiver<BlockingResult>,
    /// Worker thread handles
    workers: Vec<JoinHandle<()>>,
    /// Number of workers
    worker_count: usize,
}

impl BlockingPool {
    /// Create a new blocking pool with the default number of workers
    /// (equal to the number of CPU cores)
    pub fn new() -> Self {
        Self::with_workers(num_cpus::get())
    }

    /// Create a blocking pool with a specific number of workers
    pub fn with_workers(count: usize) -> Self {
        let count = count.max(1); // At least one worker

        let (task_sender, task_receiver) = mpsc::channel::<WorkerMessage>();
        let (result_sender, result_receiver) = mpsc::channel::<BlockingResult>();

        // Wrap receiver in Arc<Mutex> so workers can share it
        let task_receiver = Arc::new(Mutex::new(task_receiver));

        // Shared handle registry for storing actual OS resources
        let handle_registry = Arc::new(Mutex::new(HandleRegistry::new()));

        let mut workers = Vec::with_capacity(count);

        for id in 0..count {
            let receiver = Arc::clone(&task_receiver);
            let sender = result_sender.clone();
            let registry = Arc::clone(&handle_registry);

            let handle = thread::Builder::new()
                .name(format!("blocking-pool-{}", id))
                .spawn(move || {
                    worker_loop(receiver, sender, registry);
                })
                .expect("failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            task_sender,
            result_receiver,
            workers,
            worker_count: count,
        }
    }

    /// Submit a blocking task to the pool
    pub fn submit(&self, task: BlockingTask) -> Result<(), BlockingTask> {
        self.task_sender
            .send(WorkerMessage::Task(task))
            .map_err(|e| match e.0 {
                WorkerMessage::Task(t) => t,
                WorkerMessage::Shutdown => unreachable!(),
            })
    }

    /// Try to receive a completed result (non-blocking)
    pub fn try_recv(&self) -> Option<BlockingResult> {
        self.result_receiver.try_recv().ok()
    }

    /// Receive all available results (non-blocking)
    pub fn drain_results(&self) -> Vec<BlockingResult> {
        let mut results = Vec::new();
        while let Some(result) = self.try_recv() {
            results.push(result);
        }
        results
    }

    /// Get the number of workers in the pool
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Shutdown the pool, waiting for all workers to finish
    pub fn shutdown(self) {
        // Send shutdown message to all workers
        for _ in 0..self.worker_count {
            let _ = self.task_sender.send(WorkerMessage::Shutdown);
        }

        // Wait for all workers to finish
        for worker in self.workers {
            let _ = worker.join();
        }
    }
}

impl Default for BlockingPool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BlockingPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockingPool")
            .field("worker_count", &self.worker_count)
            .finish()
    }
}

/// Worker thread main loop
fn worker_loop(
    receiver: Arc<Mutex<Receiver<WorkerMessage>>>,
    sender: Sender<BlockingResult>,
    registry: Arc<Mutex<HandleRegistry>>,
) {
    loop {
        // Get next task
        let message = {
            let lock = receiver.lock().expect("worker receiver lock poisoned");
            lock.recv()
        };

        match message {
            Ok(WorkerMessage::Task(task)) => {
                let result = execute_blocking_op(&task.op, &registry);
                let _ = sender.send(BlockingResult {
                    fiber_id: task.fiber_id,
                    result,
                });
            }
            Ok(WorkerMessage::Shutdown) | Err(_) => {
                // Channel closed or shutdown requested
                break;
            }
        }
    }
}

/// Execute a blocking I/O operation
fn execute_blocking_op(
    op: &IoOp,
    registry: &Arc<Mutex<HandleRegistry>>,
) -> BlockingOpResult {
    match op {
        IoOp::FileOpen { path, mode } => {
            use std::fs::OpenOptions;

            let result = match mode {
                OpenMode::Read => OpenOptions::new().read(true).open(path),
                OpenMode::Write => OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path),
                OpenMode::Append => OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path),
                OpenMode::ReadWrite => OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(path),
            };

            match result {
                Ok(file) => {
                    let id = registry.lock().unwrap().insert(IoResource::File(file));
                    BlockingOpResult::FileOpened(id)
                }
                Err(e) => BlockingOpResult::Error(e),
            }
        }

        IoOp::TcpConnect { host, port } => {
            use std::net::TcpStream;

            let addr = format!("{}:{}", host, port);
            match TcpStream::connect(&addr) {
                Ok(stream) => {
                    let id = registry.lock().unwrap().insert(IoResource::TcpSocket(stream));
                    BlockingOpResult::TcpConnected(id)
                }
                Err(e) => BlockingOpResult::Error(e),
            }
        }

        IoOp::TcpListen { host, port } => {
            use std::net::TcpListener as StdTcpListener;

            let addr = format!("{}:{}", host, port);
            match StdTcpListener::bind(&addr) {
                Ok(listener) => {
                    let id = registry.lock().unwrap().insert(IoResource::TcpListener(listener));
                    BlockingOpResult::TcpListening(id)
                }
                Err(e) => BlockingOpResult::Error(e),
            }
        }

        IoOp::TcpAccept { listener } => {
            // Clone the listener so we can accept without holding the registry lock
            // This is important because accept() is blocking and we don't want to
            // block all other I/O operations while waiting for a connection
            let listener_clone = {
                let mut reg = registry.lock().unwrap();
                match reg.get_mut(*listener) {
                    Some(IoResource::TcpListener(l)) => {
                        // try_clone() returns a new TcpListener that shares the same
                        // underlying OS socket
                        l.try_clone()
                    }
                    Some(_) => Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "handle is not a TcpListener",
                    )),
                    None => Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "listener handle not found",
                    )),
                }
            };

            // Now do the blocking accept without holding the lock
            let accept_result = match listener_clone {
                Ok(l) => l.accept().map(|(stream, _addr)| stream),
                Err(e) => Err(e),
            };

            // Now insert the new socket (if accept succeeded)
            match accept_result {
                Ok(stream) => {
                    let id = registry.lock().unwrap().insert(IoResource::TcpSocket(stream));
                    BlockingOpResult::TcpAccepted(id)
                }
                Err(e) => BlockingOpResult::Error(e),
            }
        }

        IoOp::Read { handle, count } => {
            let mut reg = registry.lock().unwrap();
            match reg.get_mut(*handle) {
                Some(IoResource::File(f)) => {
                    let mut buf = vec![0u8; *count];
                    match f.read(&mut buf) {
                        Ok(n) => {
                            buf.truncate(n);
                            BlockingOpResult::Read(buf)
                        }
                        Err(e) => BlockingOpResult::Error(e),
                    }
                }
                Some(IoResource::TcpSocket(s)) => {
                    let mut buf = vec![0u8; *count];
                    match s.read(&mut buf) {
                        Ok(n) => {
                            buf.truncate(n);
                            BlockingOpResult::Read(buf)
                        }
                        Err(e) => BlockingOpResult::Error(e),
                    }
                }
                Some(IoResource::TcpListener(_)) => BlockingOpResult::Error(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "cannot read from a TcpListener",
                )),
                None => BlockingOpResult::Error(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "handle not found",
                )),
            }
        }

        IoOp::Write { handle, data } => {
            let mut reg = registry.lock().unwrap();
            match reg.get_mut(*handle) {
                Some(IoResource::File(f)) => match f.write(data) {
                    Ok(n) => BlockingOpResult::Written(n),
                    Err(e) => BlockingOpResult::Error(e),
                },
                Some(IoResource::TcpSocket(s)) => match s.write(data) {
                    Ok(n) => BlockingOpResult::Written(n),
                    Err(e) => BlockingOpResult::Error(e),
                },
                Some(IoResource::TcpListener(_)) => BlockingOpResult::Error(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "cannot write to a TcpListener",
                )),
                None => BlockingOpResult::Error(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "handle not found",
                )),
            }
        }

        IoOp::Close { handle } => {
            let mut reg = registry.lock().unwrap();
            match reg.remove(*handle) {
                Some(_resource) => {
                    // Resource is dropped here, which closes the file/socket
                    BlockingOpResult::Closed
                }
                None => BlockingOpResult::Error(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "handle not found",
                )),
            }
        }

        IoOp::Sleep { duration_ms } => {
            std::thread::sleep(std::time::Duration::from_millis(*duration_ms));
            BlockingOpResult::Slept
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_pool_creation() {
        let pool = BlockingPool::with_workers(2);
        assert_eq!(pool.worker_count(), 2);
        pool.shutdown();
    }

    #[test]
    fn test_pool_default() {
        let pool = BlockingPool::new();
        assert!(pool.worker_count() >= 1);
        pool.shutdown();
    }

    #[test]
    fn test_sleep_operation() {
        let pool = BlockingPool::with_workers(1);

        let task = BlockingTask {
            fiber_id: 42,
            op: IoOp::Sleep { duration_ms: 10 },
        };

        pool.submit(task).expect("should submit");

        // Wait for result
        std::thread::sleep(Duration::from_millis(50));

        let results = pool.drain_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].fiber_id, 42);
        assert!(matches!(results[0].result, BlockingOpResult::Slept));

        pool.shutdown();
    }

    #[test]
    fn test_file_open_nonexistent() {
        let pool = BlockingPool::with_workers(1);

        let task = BlockingTask {
            fiber_id: 123,
            op: IoOp::FileOpen {
                path: "/nonexistent/path/to/file.txt".to_string(),
                mode: OpenMode::Read,
            },
        };

        pool.submit(task).expect("should submit");

        // Wait for result
        std::thread::sleep(Duration::from_millis(50));

        let results = pool.drain_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].fiber_id, 123);
        assert!(matches!(results[0].result, BlockingOpResult::Error(_)));

        pool.shutdown();
    }

    #[test]
    fn test_multiple_tasks() {
        let pool = BlockingPool::with_workers(2);

        // Submit multiple sleep tasks
        for i in 0..5 {
            let task = BlockingTask {
                fiber_id: i,
                op: IoOp::Sleep { duration_ms: 5 },
            };
            pool.submit(task).expect("should submit");
        }

        // Wait for all results
        std::thread::sleep(Duration::from_millis(100));

        let results = pool.drain_results();
        assert_eq!(results.len(), 5);

        // All should be Slept
        for result in &results {
            assert!(matches!(result.result, BlockingOpResult::Slept));
        }

        pool.shutdown();
    }

    #[test]
    fn test_tcp_connect_failure() {
        let pool = BlockingPool::with_workers(1);

        let task = BlockingTask {
            fiber_id: 999,
            op: IoOp::TcpConnect {
                host: "127.0.0.1".to_string(),
                port: 1, // Unlikely to be listening
            },
        };

        pool.submit(task).expect("should submit");

        // Wait for result (connection refused should be quick)
        std::thread::sleep(Duration::from_millis(100));

        let results = pool.drain_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].fiber_id, 999);
        assert!(matches!(results[0].result, BlockingOpResult::Error(_)));

        pool.shutdown();
    }
}
