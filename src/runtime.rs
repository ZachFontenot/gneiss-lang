//! Runtime: processes, channels, and scheduler

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::blocking_pool::{BlockingPool, BlockingTask, BlockingOpResult};
use crate::eval::{Cont, FiberId, IoOp, SelectEffectArm, Value};
use crate::io_reactor::{IoReactor, IoToken};

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
    /// Blocked waiting for another fiber to complete (Phase 7)
    BlockedJoin(FiberId),
    /// Blocked waiting for I/O (reactor will wake when ready)
    BlockedIo(IoToken),
    /// Blocked waiting for a timer to expire
    BlockedSleep(Instant),
    /// Blocked waiting for a blocking pool operation
    BlockedBlocking,
    /// Terminated
    Done,
}

/// A lightweight process (fiber)
pub struct Process {
    pub pid: Pid,
    pub state: ProcessState,
    /// The continuation (thunk to call when resumed)
    pub continuation: Option<ProcessContinuation>,
    /// Fiber continuation for effect resumption
    pub fiber_cont: Option<Cont>,
    /// Result value when fiber completes (for join)
    pub result: Option<Value>,
    /// Fibers waiting to join this one
    pub joiners: Vec<Pid>,
    /// Select arms (stored when blocked on select)
    pub select_arms: Option<Vec<SelectEffectArm>>,
}

/// What a process should do when it resumes
pub enum ProcessContinuation {
    /// Start executing a thunk (initial spawn)
    Start(Value),
    /// Resume fiber with a value
    ResumeFiber(Value),
    /// Resume fiber from select with channel and value
    ResumeSelect { channel: ChannelId, value: Value },
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
    /// I/O reactor for non-blocking I/O (optional, created on first use)
    io_reactor: Option<IoReactor>,
    /// Blocking thread pool for blocking operations (optional, created on first use)
    blocking_pool: Option<BlockingPool>,
    /// Timers: (deadline, fiber_id) sorted by deadline
    timers: Vec<(Instant, Pid)>,
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
            io_reactor: None,
            blocking_pool: None,
            timers: Vec::new(),
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
            fiber_cont: None,
            result: None,
            joiners: Vec::new(),
            select_arms: None,
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

    /// Check if we're in a true deadlock situation
    /// A deadlock is when the main process is blocked on channels/joins (not I/O)
    /// Spawned processes being blocked after main completes is not deadlock
    /// Note: BlockedIo, BlockedSleep, BlockedBlocking are NOT deadlock - they will complete
    pub fn is_deadlocked(&self) -> bool {
        if !self.ready_queue.is_empty() {
            return false; // Still have work to do
        }

        // If there are pending timers or I/O, not deadlocked
        if !self.timers.is_empty() {
            return false;
        }
        if let Some(ref reactor) = self.io_reactor {
            if !reactor.is_empty() {
                return false;
            }
        }

        // Check if main is blocked on channels/joins (real deadlock)
        if let Some(main_pid) = self.main_pid {
            if let Some(main_process) = self.processes.get(&main_pid) {
                let state = &main_process.borrow().state;
                if matches!(
                    state,
                    ProcessState::BlockedSend(_)
                        | ProcessState::BlockedRecv(_)
                        | ProcessState::BlockedSelect(_)
                        | ProcessState::BlockedJoin(_)
                ) {
                    return true; // Main is blocked on channels = deadlock
                }
                // Main is Done, Ready, or blocked on I/O - not deadlock
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
                    | ProcessState::BlockedJoin(_)
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
                        | ProcessState::BlockedJoin(_)
                        | ProcessState::BlockedIo(_)
                        | ProcessState::BlockedSleep(_)
                        | ProcessState::BlockedBlocking
                )
            })
            .count()
    }

    /// Unregister a process from channels' receiver lists (used when select completes)
    pub fn unregister_from_channels(
        &mut self,
        pid: Pid,
        channels: &[ChannelId],
        except: ChannelId,
    ) {
        for &channel_id in channels {
            if channel_id != except {
                if let Some(ch) = self.channels.get(&channel_id) {
                    ch.borrow_mut().receivers.retain(|&p| p != pid);
                }
            }
        }
    }

    // ========================================================================
    // Phase 7: Fiber Effect-Based Methods
    // ========================================================================

    /// Store fiber's continuation for later resumption (Phase 7)
    pub fn store_fiber_cont(&mut self, pid: Pid, cont: Cont) {
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().fiber_cont = Some(cont);
        }
    }

    /// Take the fiber continuation for a process (Phase 7)
    pub fn take_fiber_cont(&mut self, pid: Pid) -> Option<Cont> {
        self.processes
            .get(&pid)
            .and_then(|p| p.borrow_mut().fiber_cont.take())
    }

    /// Resume fiber with a value - set continuation and re-queue as ready (Phase 7)
    pub fn resume_fiber_with(&mut self, pid: Pid, value: Value) {
        if let Some(process) = self.processes.get(&pid) {
            let mut p = process.borrow_mut();
            p.state = ProcessState::Ready;
            p.continuation = Some(ProcessContinuation::ResumeFiber(value));
            drop(p);
            self.ready_queue.push_back(pid);
        }
    }

    /// Complete a fiber and wake any joiners (Phase 7)
    pub fn complete_fiber(&mut self, pid: Pid, result: Value) {
        if let Some(process) = self.processes.get(&pid) {
            let mut p = process.borrow_mut();
            p.state = ProcessState::Done;
            p.result = Some(result.clone());

            // Wake all joiners with the result
            let joiners = std::mem::take(&mut p.joiners);
            drop(p);

            for joiner_pid in joiners {
                self.resume_fiber_with(joiner_pid, result.clone());
            }
        }
    }

    /// Yield fiber - store continuation and re-queue at back of ready queue (Phase 7)
    pub fn yield_fiber(&mut self, pid: Pid) {
        if let Some(process) = self.processes.get(&pid) {
            let mut p = process.borrow_mut();
            p.state = ProcessState::Ready;
            p.continuation = Some(ProcessContinuation::ResumeFiber(Value::Unit));
            drop(p);
            self.ready_queue.push_back(pid);
        }
    }

    /// Block fiber waiting to send - try immediate handoff first
    pub fn block_fiber_send(&mut self, pid: Pid, channel_id: ChannelId, value: Value, cont: Option<Cont>) {
        if let Some(c) = cont {
            self.store_fiber_cont(pid, c);
        }

        let mut channel = match self.channels.get(&channel_id) {
            Some(ch) => ch.borrow_mut(),
            None => return,
        };

        if let Some(receiver_pid) = channel.receivers.pop_front() {
            drop(channel);

            let receiver = self.processes.get(&receiver_pid).unwrap().borrow();
            let is_select = matches!(receiver.state, ProcessState::BlockedSelect(_));
            let select_channels: Vec<ChannelId> = if let ProcessState::BlockedSelect(ref chs) = receiver.state {
                chs.iter().filter(|&&ch| ch != channel_id).copied().collect()
            } else {
                Vec::new()
            };
            drop(receiver);

            if is_select {
                // Remove from other channels' receiver lists
                for other_ch in select_channels {
                    if let Some(ch) = self.channels.get(&other_ch) {
                        ch.borrow_mut().receivers.retain(|&p| p != receiver_pid);
                    }
                }
                let mut receiver = self.processes.get(&receiver_pid).unwrap().borrow_mut();
                receiver.state = ProcessState::Ready;
                receiver.continuation = Some(ProcessContinuation::ResumeSelect { channel: channel_id, value });
            } else {
                let mut receiver = self.processes.get(&receiver_pid).unwrap().borrow_mut();
                receiver.state = ProcessState::Ready;
                receiver.continuation = Some(ProcessContinuation::ResumeFiber(value));
            }
            self.ready_queue.push_back(receiver_pid);
            self.resume_fiber_with(pid, Value::Unit);
        } else {
            channel.senders.push_back((pid, value));
            drop(channel);
            let mut process = self.processes.get(&pid).unwrap().borrow_mut();
            process.state = ProcessState::BlockedSend(channel_id);
        }
    }

    /// Block fiber waiting to receive - try immediate handoff first
    pub fn block_fiber_recv(&mut self, pid: Pid, channel_id: ChannelId, cont: Option<Cont>) {
        if let Some(c) = cont {
            self.store_fiber_cont(pid, c);
        }

        let mut channel = match self.channels.get(&channel_id) {
            Some(ch) => ch.borrow_mut(),
            None => return,
        };

        if let Some((sender_pid, value)) = channel.senders.pop_front() {
            drop(channel);

            // Wake sender with Unit
            let mut sender = self.processes.get(&sender_pid).unwrap().borrow_mut();
            sender.state = ProcessState::Ready;
            sender.continuation = Some(ProcessContinuation::ResumeFiber(Value::Unit));
            drop(sender);
            self.ready_queue.push_back(sender_pid);

            // Resume receiver with value
            self.resume_fiber_with(pid, value);
        } else {
            channel.receivers.push_back(pid);
            drop(channel);
            let mut process = self.processes.get(&pid).unwrap().borrow_mut();
            process.state = ProcessState::BlockedRecv(channel_id);
        }
    }

    /// Block fiber waiting to join another fiber (Phase 7)
    pub fn block_fiber_join(&mut self, pid: Pid, target_id: FiberId, cont: Option<Cont>) {
        // Store continuation first
        if let Some(c) = cont {
            self.store_fiber_cont(pid, c);
        }

        // Check if target already completed
        if let Some(target) = self.processes.get(&target_id) {
            let target_ref = target.borrow();
            if target_ref.state == ProcessState::Done {
                // Already done - resume with result immediately
                if let Some(result) = target_ref.result.clone() {
                    drop(target_ref);
                    self.resume_fiber_with(pid, result);
                    return;
                }
            }
            drop(target_ref);

            // Not done yet - add to joiners list
            target.borrow_mut().joiners.push(pid);
        }

        // Mark as blocked
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::BlockedJoin(target_id);
        }
    }

    /// Block fiber on select - register on all channels
    pub fn block_fiber_select(&mut self, pid: Pid, arms: Vec<SelectEffectArm>, cont: Option<Cont>) {
        if let Some(c) = cont {
            self.store_fiber_cont(pid, c);
        }

        let channels: Vec<ChannelId> = arms.iter().map(|a| a.channel).collect();

        // Try each channel for immediate handoff
        for i in 0..arms.len() {
            let arm_channel = arms[i].channel;
            if let Some(channel) = self.channels.get(&arm_channel) {
                let mut ch = channel.borrow_mut();
                if let Some((sender_pid, value)) = ch.senders.pop_front() {
                    drop(ch);

                    // Wake sender with Unit
                    let mut sender = self.processes.get(&sender_pid).unwrap().borrow_mut();
                    sender.state = ProcessState::Ready;
                    sender.continuation = Some(ProcessContinuation::ResumeFiber(Value::Unit));
                    drop(sender);
                    self.ready_queue.push_back(sender_pid);

                    // Store arms and resume with select
                    if let Some(process) = self.processes.get(&pid) {
                        let mut p = process.borrow_mut();
                        p.select_arms = Some(arms);
                        p.state = ProcessState::Ready;
                        p.continuation = Some(ProcessContinuation::ResumeSelect { channel: arm_channel, value });
                        drop(p);
                        self.ready_queue.push_back(pid);
                    }
                    return;
                }
            }
        }

        // No immediate match - store arms and block on all channels
        if let Some(process) = self.processes.get(&pid) {
            let mut p = process.borrow_mut();
            p.select_arms = Some(arms);
            p.state = ProcessState::BlockedSelect(channels.clone());
        }

        for &channel_id in &channels {
            if let Some(channel) = self.channels.get(&channel_id) {
                channel.borrow_mut().receivers.push_back(pid);
            }
        }
    }

    /// Take the select arms from a process
    pub fn take_select_arms(&mut self, pid: Pid) -> Option<Vec<SelectEffectArm>> {
        self.processes
            .get(&pid)
            .and_then(|p| p.borrow_mut().select_arms.take())
    }

    // ========================================================================
    // I/O Integration
    // ========================================================================

    /// Get or create the I/O reactor
    fn ensure_reactor(&mut self) -> &mut IoReactor {
        if self.io_reactor.is_none() {
            self.io_reactor = Some(IoReactor::new().expect("failed to create I/O reactor"));
        }
        self.io_reactor.as_mut().unwrap()
    }

    /// Get or create the blocking pool
    fn ensure_blocking_pool(&mut self) -> &mut BlockingPool {
        if self.blocking_pool.is_none() {
            self.blocking_pool = Some(BlockingPool::new());
        }
        self.blocking_pool.as_mut().unwrap()
    }

    /// Block a fiber waiting for a sleep timer
    pub fn block_fiber_sleep(&mut self, pid: Pid, duration_ms: u64, cont: Option<Cont>) {
        if let Some(c) = cont {
            self.store_fiber_cont(pid, c);
        }

        let deadline = Instant::now() + Duration::from_millis(duration_ms);

        // Insert timer sorted by deadline
        let pos = self.timers.iter().position(|(d, _)| *d > deadline).unwrap_or(self.timers.len());
        self.timers.insert(pos, (deadline, pid));

        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::BlockedSleep(deadline);
        }
    }

    /// Block a fiber waiting for a blocking pool operation
    pub fn block_fiber_blocking(&mut self, pid: Pid, op: IoOp, cont: Option<Cont>) {
        if let Some(c) = cont {
            self.store_fiber_cont(pid, c);
        }

        // Submit to blocking pool
        let pool = self.ensure_blocking_pool();
        let task = BlockingTask { fiber_id: pid, op };
        let _ = pool.submit(task);

        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::BlockedBlocking;
        }
    }

    /// Check timers and wake any fibers whose deadlines have passed
    pub fn check_timers(&mut self) {
        let now = Instant::now();

        // Collect expired timers
        let mut expired = Vec::new();
        while let Some(&(deadline, pid)) = self.timers.first() {
            if deadline <= now {
                expired.push(pid);
                self.timers.remove(0);
            } else {
                break;
            }
        }

        // Wake expired fibers
        for pid in expired {
            self.resume_fiber_with(pid, Value::Unit);
        }
    }

    /// Check the blocking pool for completed operations and wake fibers
    pub fn check_blocking_pool(&mut self) {
        if let Some(ref pool) = self.blocking_pool {
            let results = pool.drain_results();
            for result in results {
                let value = self.blocking_result_to_value(result.result);
                self.resume_fiber_with(result.fiber_id, value);
            }
        }
    }

    /// Convert a BlockingOpResult to a Value (Result type)
    fn blocking_result_to_value(&self, result: BlockingOpResult) -> Value {
        match result {
            BlockingOpResult::FileOpened(id) => {
                Value::ok(Value::FileHandle(id))
            }
            BlockingOpResult::TcpConnected(id) => {
                Value::ok(Value::TcpSocket(id))
            }
            BlockingOpResult::TcpListening(id) => {
                Value::ok(Value::TcpListener(id))
            }
            BlockingOpResult::TcpAccepted(id) => {
                Value::ok(Value::TcpSocket(id))
            }
            BlockingOpResult::Read(bytes) => {
                Value::ok(Value::Bytes(bytes))
            }
            BlockingOpResult::Written(count) => {
                Value::ok(Value::Int(count as i64))
            }
            BlockingOpResult::Closed => {
                Value::ok(Value::Unit)
            }
            BlockingOpResult::Slept => {
                Value::Unit
            }
            BlockingOpResult::Error(e) => {
                Value::from_io_result::<(), _>(Err(e), |_| Value::Unit)
            }
        }
    }

    /// Get time until next timer fires (for poll timeout)
    pub fn time_until_next_timer(&self) -> Option<Duration> {
        self.timers.first().map(|(deadline, _)| {
            let now = Instant::now();
            if *deadline <= now {
                Duration::ZERO
            } else {
                *deadline - now
            }
        })
    }

    /// Check if there's any I/O or blocking work pending
    pub fn has_io_pending(&self) -> bool {
        // Check if there are any fibers blocked on I/O, sleep, or blocking pool
        self.processes.values().any(|p| {
            let state = &p.borrow().state;
            matches!(
                state,
                ProcessState::BlockedIo(_)
                    | ProcessState::BlockedSleep(_)
                    | ProcessState::BlockedBlocking
            )
        })
    }

    /// Poll I/O and timers, waking any ready fibers
    /// Returns true if any fibers were woken
    pub fn poll_io(&mut self, timeout: Option<Duration>) -> bool {
        let mut woke_any = false;

        // Check timers first
        self.check_timers();
        if !self.ready_queue.is_empty() {
            woke_any = true;
        }

        // Check blocking pool
        self.check_blocking_pool();
        if !self.ready_queue.is_empty() {
            woke_any = true;
        }

        // If nothing is ready yet and we have a timeout, wait before checking again
        // This handles the case where we're waiting for timers but have no reactor
        if !woke_any {
            if let Some(ref mut reactor) = self.io_reactor {
                // Poll I/O reactor with the timeout
                if let Ok(events) = reactor.poll(timeout) {
                    for event in events {
                        // For now, just wake the fiber - actual I/O handling
                        // will be done when the builtin is implemented
                        self.resume_fiber_with(event.fiber_id, Value::Unit);
                        woke_any = true;
                    }
                }
            } else if let Some(duration) = timeout {
                // No reactor but we have a timeout - sleep to wait for timers
                std::thread::sleep(duration);
                // Check timers again after sleeping
                self.check_timers();
                if !self.ready_queue.is_empty() {
                    woke_any = true;
                }
            }
        }

        woke_any
    }

    /// Dispatch an I/O operation to the appropriate handler
    /// Called from eval.rs when handling FiberEffect::Io
    pub fn dispatch_io(&mut self, pid: Pid, op: IoOp, cont: Option<Cont>) {
        match &op {
            IoOp::Sleep { duration_ms } => {
                self.block_fiber_sleep(pid, *duration_ms, cont);
            }
            // All other ops go to blocking pool for now
            // Later, socket ops will use the reactor for non-blocking I/O
            _ => {
                self.block_fiber_blocking(pid, op, cont);
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
