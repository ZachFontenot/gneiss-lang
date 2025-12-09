//! Runtime: processes, channels, and scheduler

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

use crate::eval::{Cont, Value};

/// Unique process identifier
pub type Pid = u64;

/// Unique channel identifier
pub type ChannelId = u64;

/// A channel for inter-process communication
#[derive(Debug)]
pub struct Channel {
    pub id: ChannelId,
    /// Processes waiting to send (with their values)
    pub senders: VecDeque<(Pid, Value)>,
    /// Processes waiting to receive
    pub receivers: VecDeque<Pid>,
}

impl Channel {
    pub fn new(id: ChannelId) -> Self {
        Self {
            id,
            senders: VecDeque::new(),
            receivers: VecDeque::new(),
        }
    }
}

/// Process state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    /// Ready to run
    Ready,
    /// Blocked waiting to send on a channel
    BlockedSend(ChannelId),
    /// Blocked waiting to receive from a channel
    BlockedRecv(ChannelId),
    /// Blocked waiting on multiple channels (select)
    BlockedSelect(Vec<ChannelId>),
    /// Terminated
    Done,
}

/// A lightweight process
pub struct Process {
    pub pid: Pid,
    pub state: ProcessState,
    /// The continuation (thunk to call when resumed)
    pub continuation: Option<ProcessContinuation>,
    /// Saved continuation stack when blocked (for resumption)
    pub saved_cont: Option<Cont>,
    /// Value received from a channel (set when unblocked)
    pub received_value: Option<Value>,
    /// Channel that fired when resuming from a select
    pub select_fired_channel: Option<ChannelId>,
}

/// What a process should do when it resumes
pub enum ProcessContinuation {
    /// Start executing a thunk (initial spawn)
    Start(Value),
    /// Resume after receiving a value - continue with saved_cont
    ResumeAfterRecv,
    /// Resume after sending - continue with saved_cont
    ResumeAfterSend,
    /// Resume after select received a value - SelectReady frame handles the value
    ResumeAfterSelect,
}

/// The runtime scheduler
pub struct Runtime {
    /// All processes
    processes: HashMap<Pid, RefCell<Process>>,
    /// Ready queue
    ready_queue: VecDeque<Pid>,
    /// All channels
    channels: HashMap<ChannelId, RefCell<Channel>>,
    /// Next PID to allocate
    next_pid: Pid,
    /// Next channel ID to allocate
    next_channel_id: ChannelId,
    /// Currently running process
    current_pid: Option<Pid>,
    /// The main process PID (first spawned)
    main_pid: Option<Pid>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            ready_queue: VecDeque::new(),
            channels: HashMap::new(),
            next_pid: 1,
            next_channel_id: 1,
            current_pid: None,
            main_pid: None,
        }
    }

    /// Spawn a new process running the given thunk
    pub fn spawn(&mut self, thunk: Value) -> Pid {
        let pid = self.next_pid;
        self.next_pid += 1;

        let process = Process {
            pid,
            state: ProcessState::Ready,
            continuation: Some(ProcessContinuation::Start(thunk)),
            saved_cont: None,
            received_value: None,
            select_fired_channel: None,
        };

        self.processes.insert(pid, RefCell::new(process));
        self.ready_queue.push_back(pid);

        // First spawned process is main
        if self.main_pid.is_none() {
            self.main_pid = Some(pid);
        }

        pid
    }

    /// Create a new channel
    pub fn new_channel(&mut self) -> ChannelId {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        self.channels.insert(id, RefCell::new(Channel::new(id)));
        id
    }

    /// Get the current process ID
    pub fn current_pid(&self) -> Option<Pid> {
        self.current_pid
    }

    /// Attempt to send a value on a channel (synchronous rendezvous)
    /// Returns true if the send completed immediately, false if blocked
    pub fn send(&mut self, channel_id: ChannelId, value: Value) -> bool {
        let pid = self.current_pid.expect("send called outside of process");

        let mut channel = self
            .channels
            .get(&channel_id)
            .expect("invalid channel")
            .borrow_mut();

        // Check if there's a waiting receiver
        if let Some(receiver_pid) = channel.receivers.pop_front() {
            // Direct handoff - wake up the receiver with the value
            drop(channel);

            let mut receiver = self.processes.get(&receiver_pid).unwrap().borrow_mut();

            // If receiver was in a select, unregister from other channels and record which channel fired
            if let ProcessState::BlockedSelect(ref select_channels) = receiver.state {
                let other_channels: Vec<_> = select_channels
                    .iter()
                    .filter(|&&ch| ch != channel_id)
                    .copied()
                    .collect();
                drop(receiver);

                // Remove from other channels' receiver lists
                for other_ch in other_channels {
                    if let Some(ch) = self.channels.get(&other_ch) {
                        ch.borrow_mut().receivers.retain(|&p| p != receiver_pid);
                    }
                }

                let mut receiver = self.processes.get(&receiver_pid).unwrap().borrow_mut();
                receiver.received_value = Some(value);
                receiver.select_fired_channel = Some(channel_id); // Record which channel fired
                receiver.state = ProcessState::Ready;
                receiver.continuation = Some(ProcessContinuation::ResumeAfterSelect);
            } else {
                receiver.received_value = Some(value);
                receiver.state = ProcessState::Ready;
                receiver.continuation = Some(ProcessContinuation::ResumeAfterRecv);
            }

            self.ready_queue.push_back(receiver_pid);
            true
        } else {
            // No receiver - block the sender
            channel.senders.push_back((pid, value));
            drop(channel);

            let mut process = self.processes.get(&pid).unwrap().borrow_mut();
            process.state = ProcessState::BlockedSend(channel_id);
            false
        }
    }

    /// Attempt to receive from a channel (synchronous rendezvous)
    /// Returns Some(value) if received immediately, None if blocked
    pub fn recv(&mut self, channel_id: ChannelId) -> Option<Value> {
        let pid = self.current_pid.expect("recv called outside of process");

        let mut channel = self
            .channels
            .get(&channel_id)
            .expect("invalid channel")
            .borrow_mut();

        // Check if there's a waiting sender
        if let Some((sender_pid, value)) = channel.senders.pop_front() {
            // Direct handoff - wake up the sender
            drop(channel);

            let mut sender = self.processes.get(&sender_pid).unwrap().borrow_mut();
            sender.state = ProcessState::Ready;
            sender.continuation = Some(ProcessContinuation::ResumeAfterSend);
            drop(sender);

            self.ready_queue.push_back(sender_pid);
            Some(value)
        } else {
            // No sender - block the receiver
            channel.receivers.push_back(pid);
            drop(channel);

            let mut process = self.processes.get(&pid).unwrap().borrow_mut();
            process.state = ProcessState::BlockedRecv(channel_id);
            None
        }
    }

    /// Non-blocking receive: check if sender waiting, don't block if not.
    /// Does NOT register as a waiter. Used for select.
    pub fn try_recv(&mut self, channel_id: ChannelId) -> Option<Value> {
        let mut channel = self
            .channels
            .get(&channel_id)?
            .borrow_mut();

        if let Some((sender_pid, value)) = channel.senders.pop_front() {
            drop(channel);

            // Wake up the sender
            let mut sender = self.processes.get(&sender_pid).unwrap().borrow_mut();
            sender.state = ProcessState::Ready;
            sender.continuation = Some(ProcessContinuation::ResumeAfterSend);
            drop(sender);

            self.ready_queue.push_back(sender_pid);
            Some(value)
        } else {
            None
        }
    }

    /// Block current process waiting on any of the given channels (for select)
    pub fn block_on_select(&mut self, channels: &[ChannelId]) {
        let pid = self.current_pid.expect("block_on_select called outside process");

        // Register as receiver on all channels
        for &channel_id in channels {
            if let Some(channel) = self.channels.get(&channel_id) {
                channel.borrow_mut().receivers.push_back(pid);
            }
        }

        // Mark process as blocked on select
        let mut process = self.processes.get(&pid).unwrap().borrow_mut();
        process.state = ProcessState::BlockedSelect(channels.to_vec());
    }

    /// Get the next ready process to run
    pub fn next_ready(&mut self) -> Option<Pid> {
        self.ready_queue.pop_front()
    }

    /// Set the current process
    pub fn set_current(&mut self, pid: Option<Pid>) {
        self.current_pid = pid;
    }

    /// Mark the current process as ready and add back to queue
    pub fn yield_current(&mut self) {
        if let Some(pid) = self.current_pid {
            let process = self.processes.get(&pid).unwrap().borrow();
            if process.state == ProcessState::Ready {
                drop(process);
                self.ready_queue.push_back(pid);
            }
        }
    }

    /// Mark a process as done
    pub fn mark_done(&mut self, pid: Pid) {
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::Done;
        }
    }

    /// Get a process's continuation
    pub fn take_continuation(&mut self, pid: Pid) -> Option<ProcessContinuation> {
        self.processes
            .get(&pid)
            .and_then(|p| p.borrow_mut().continuation.take())
    }

    /// Get the received value for a process
    pub fn take_received_value(&mut self, pid: Pid) -> Option<Value> {
        self.processes
            .get(&pid)
            .and_then(|p| p.borrow_mut().received_value.take())
    }

    /// Save a continuation for a blocked process
    pub fn save_cont(&mut self, pid: Pid, cont: Cont) {
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().saved_cont = Some(cont);
        }
    }

    /// Take the saved continuation for a process
    pub fn take_saved_cont(&mut self, pid: Pid) -> Option<Cont> {
        self.processes
            .get(&pid)
            .and_then(|p| p.borrow_mut().saved_cont.take())
    }

    /// Take the select_fired_channel for a process (used when resuming from select)
    pub fn take_select_fired_channel(&mut self, pid: Pid) -> Option<ChannelId> {
        self.processes
            .get(&pid)
            .and_then(|p| p.borrow_mut().select_fired_channel.take())
    }

    /// Check if we're in a true deadlock situation
    /// A deadlock is when the main process is blocked (or there are only blocked processes left)
    /// Spawned processes being blocked after main completes is not deadlock
    pub fn is_deadlocked(&self) -> bool {
        if !self.ready_queue.is_empty() {
            return false; // Still have work to do
        }

        // Check if main is blocked
        if let Some(main_pid) = self.main_pid {
            if let Some(main_process) = self.processes.get(&main_pid) {
                let state = &main_process.borrow().state;
                if matches!(
                    state,
                    ProcessState::BlockedSend(_)
                        | ProcessState::BlockedRecv(_)
                        | ProcessState::BlockedSelect(_)
                ) {
                    return true; // Main is blocked = deadlock
                }
                // Main is Done or Ready - not deadlock
                return false;
            }
        }

        // No main process tracked - fall back to original behavior
        self.processes.values().any(|p| {
            let state = &p.borrow().state;
            matches!(
                state,
                ProcessState::BlockedSend(_)
                    | ProcessState::BlockedRecv(_)
                    | ProcessState::BlockedSelect(_)
            )
        })
    }

    /// Check if all processes are done
    pub fn is_done(&self) -> bool {
        self.ready_queue.is_empty()
            && self
                .processes
                .values()
                .all(|p| p.borrow().state == ProcessState::Done)
    }

    /// Get number of ready processes
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Get number of blocked processes
    pub fn blocked_count(&self) -> usize {
        self.processes
            .values()
            .filter(|p| {
                let state = &p.borrow().state;
                matches!(
                    state,
                    ProcessState::BlockedSend(_)
                        | ProcessState::BlockedRecv(_)
                        | ProcessState::BlockedSelect(_)
                )
            })
            .count()
    }

    /// Unregister a process from channels' receiver lists (used when select completes)
    pub fn unregister_from_channels(&mut self, pid: Pid, channels: &[ChannelId], except: ChannelId) {
        for &channel_id in channels {
            if channel_id != except {
                if let Some(ch) = self.channels.get(&channel_id) {
                    ch.borrow_mut().receivers.retain(|&p| p != pid);
                }
            }
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn() {
        let mut rt = Runtime::new();
        let pid = rt.spawn(Value::Unit);
        assert_eq!(pid, 1);
        assert_eq!(rt.ready_count(), 1);
    }

    #[test]
    fn test_channel() {
        let rt = Runtime::new();
        // Basic channel creation test would go here
        assert_eq!(rt.ready_count(), 0);
    }
}
