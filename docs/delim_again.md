# Direct-Style Structured Concurrency via Delimited Continuations

## Design Document for Gneiss Language Runtime

### Version 1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Motivation](#2-background-and-motivation)
3. [Architecture Overview](#3-architecture-overview)
4. [Implementation Plan](#4-implementation-plan)
5. [Detailed Code Changes](#5-detailed-code-changes)
6. [User-Facing API](#6-user-facing-api)
7. [Testing Strategy](#7-testing-strategy)
8. [Migration Guide](#8-migration-guide)
9. [Future Extensions](#9-future-extensions)

---

## 1. Executive Summary

### Goal

Implement a fiber-based runtime for Gneiss that enables direct-style structured concurrency using delimited continuations as the suspension mechanism. This allows users to write concurrent code that looks like normal sequential code—no monads, no HKT, no special syntax.

### Key Insight

The existing runtime scheduler IS an effect handler. Channel operations and fiber management should be unified as effects that capture continuations and return control to the scheduler for interpretation.

### Before (Current)

```
-- Channel ops are special-cased in the evaluator
-- Two separate systems: shift/reset vs runtime blocking
ExprKind::ChanRecv → special handling → StepResult::Blocked
```

### After (Target)

```
-- All fiber effects go through the same mechanism
recv ch → shift captures continuation → FiberEffect::Recv → scheduler interprets
```

### Benefits

- **Direct style**: No monadic wrappers, `>>=`, or `IO` types needed
- **Unified model**: One system for all fiber effects
- **Simpler runtime**: Scheduler interprets effect ADT, not special cases
- **Composable**: Easy to build structured concurrency primitives as library functions

---

## 2. Background and Motivation

### 2.1 Current Architecture

The existing implementation has two parallel systems:

**System 1: User-level shift/reset**
- `ExprKind::Reset` pushes `Frame::Prompt`
- `ExprKind::Shift` captures frames up to `Prompt`
- Used for user-defined effects (the Freer monad pattern)

**System 2: Runtime concurrency**
- Channel operations return `StepResult::Blocked`
- Runtime saves continuation in `Process.saved_cont`
- Scheduler resumes blocked processes

These systems are conceptually the same but implemented differently.

### 2.2 Problem Statement

1. **Duplication**: Channel ops have special-case handling separate from shift/reset
2. **Inflexibility**: Can't easily add new fiber effects (timeouts, cancellation)
3. **Complexity**: Two mental models for the same underlying mechanism

### 2.3 Inspiration

This design follows the approach of:
- **OCaml 5 Eio**: Effect handlers for direct-style concurrency
- **Scala Cats-Effect**: Fiber-based structured concurrency
- **Kotlin Coroutines**: Suspension points with continuation capture

### 2.4 Design Principles

1. **One-shot continuations only**: Each captured continuation resumes exactly once
2. **Single prompt per fiber**: `FiberBoundary` is the implicit delimiter
3. **No multi-prompt needed**: All fiber effects go to the runtime scheduler
4. **User shift/reset preserved**: Separate system for user-defined effects

---

## 3. Architecture Overview

### 3.1 Conceptual Model

```
┌─────────────────────────────────────────────────────┐
│                    User Code                         │
│  let x = recv ch in ...   (looks like normal code)  │
└─────────────────────┬───────────────────────────────┘
                      │ Fiber op triggers capture
                      ▼
┌─────────────────────────────────────────────────────┐
│               Continuation Capture                   │
│  Frames captured up to FiberBoundary                │
└─────────────────────┬───────────────────────────────┘
                      │ Returns FiberEffect value
                      ▼
┌─────────────────────────────────────────────────────┐
│                  FiberEffect ADT                     │
│  Recv { channel, continuation }                     │
└─────────────────────┬───────────────────────────────┘
                      │ Scheduler pattern matches
                      ▼
┌─────────────────────────────────────────────────────┐
│                Runtime Scheduler                     │
│  Interprets effects, manages fibers, handles I/O    │
└─────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `FiberEffect` enum | Represents suspended fiber operations |
| `Frame::FiberBoundary` | Implicit delimiter for each fiber |
| `capture_to_fiber_boundary()` | Captures continuation up to boundary |
| Scheduler | Interprets effects, manages fiber lifecycle |
| Channel subsystem | Handles rendezvous (unchanged conceptually) |

### 3.3 Fiber Lifecycle

```
                    ┌─────────┐
         spawn()    │ Created │
            │       └────┬────┘
            ▼            │
       ┌────────┐        │ start
       │ Ready  │◄───────┘
       └───┬────┘
           │ scheduled
           ▼
       ┌────────┐
       │Running │──────────────┐
       └───┬────┘              │
           │                   │
     ┌─────┴─────┐        FiberEffect
     │           │         returned
   Done     FiberEffect        │
     │       returned          │
     ▼           │             ▼
┌─────────┐      │      ┌───────────┐
│Completed│      └─────►│  Blocked  │
└─────────┘             └─────┬─────┘
                              │ wakeup
                              ▼
                         ┌────────┐
                         │ Ready  │
                         └────────┘
```

---

## 4. Implementation Plan

### 4.1 Phase Overview

| Phase | Description | Files Modified |
|-------|-------------|----------------|
| 1 | Define FiberEffect ADT | `eval.rs` |
| 2 | Add FiberBoundary frame | `eval.rs` |
| 3 | Implement continuation capture | `eval.rs` |
| 4 | Convert channel ops to effects | `eval.rs` |
| 5 | Convert spawn/join to effects | `eval.rs` |
| 6 | Unify scheduler loop | `runtime.rs` |
| 7 | Clean up old blocking code | `eval.rs`, `runtime.rs` |
| 8 | Add new primitives (yield, etc.) | `eval.rs`, `ast.rs` |

### 4.2 Phase 1: Define FiberEffect ADT

**File**: `eval.rs`

**Task**: Add new enum representing fiber effects.

```rust
/// Effects that fibers can perform, requiring runtime intervention.
/// Each variant captures the continuation to resume after the effect is handled.
#[derive(Debug, Clone)]
pub enum FiberEffect {
    /// Fiber completed with a value
    Done(Box<Value>),
    
    /// Fork a new fiber
    /// - `thunk`: The computation to run in the new fiber (a closure)
    /// - `cont`: Continuation expecting the child's Pid
    Fork {
        thunk: Box<Value>,
        cont: Box<Cont>,
    },
    
    /// Yield control to scheduler (cooperative multitasking)
    /// - `cont`: Continuation to resume with Unit
    Yield {
        cont: Box<Cont>,
    },
    
    /// Create a new channel
    /// - `cont`: Continuation expecting the new Channel
    NewChan {
        cont: Box<Cont>,
    },
    
    /// Send a value on a channel (blocks until receiver ready)
    /// - `channel`: Target channel ID
    /// - `value`: Value to send
    /// - `cont`: Continuation to resume with Unit after send completes
    Send {
        channel: ChannelId,
        value: Box<Value>,
        cont: Box<Cont>,
    },
    
    /// Receive a value from a channel (blocks until sender ready)
    /// - `channel`: Source channel ID
    /// - `cont`: Continuation expecting the received Value
    Recv {
        channel: ChannelId,
        cont: Box<Cont>,
    },
    
    /// Wait for a fiber to complete
    /// - `pid`: The fiber to wait for
    /// - `cont`: Continuation expecting the fiber's result Value
    Join {
        pid: Pid,
        cont: Box<Cont>,
    },
    
    /// Select on multiple channels (blocks until one ready)
    /// - `arms`: Channel IDs with their patterns and body expressions
    /// - `cont`: Continuation (used after pattern binding and body eval setup)
    Select {
        arms: Vec<SelectEffectArm>,
        cont: Box<Cont>,
    },
}

/// A select arm for the FiberEffect::Select variant
#[derive(Debug, Clone)]
pub struct SelectEffectArm {
    pub channel: ChannelId,
    pub pattern: Pattern,
    pub body: Rc<Expr>,
    pub env: Env,
}
```

**Add to Value enum**:

```rust
pub enum Value {
    // ... existing variants ...
    
    /// A suspended fiber effect awaiting runtime handling
    FiberEffect(FiberEffect),
}
```

### 4.3 Phase 2: Add FiberBoundary Frame

**File**: `eval.rs`

**Task**: Add new frame type that acts as the implicit `reset` for each fiber.

```rust
pub enum Frame {
    // ... existing frames ...
    
    /// Fiber boundary - the implicit delimiter for fiber continuations.
    /// When a fiber effect captures its continuation, it stops here.
    /// When a fiber completes normally, this wraps the result in FiberEffect::Done.
    FiberBoundary,
    
    // New frames for fiber operations (evaluated before capture):
    
    /// After evaluating channel expr in recv, capture and return Recv effect
    FiberRecv,
    
    /// After evaluating channel expr in send, evaluate the value
    FiberSendValue { 
        value_expr: Rc<Expr>, 
        env: Env 
    },
    
    /// After evaluating value in send, capture and return Send effect
    FiberSendReady { 
        channel: ChannelId 
    },
    
    /// After evaluating thunk expr in spawn, capture and return Fork effect
    FiberFork,
    
    /// After evaluating pid expr in join, capture and return Join effect
    FiberJoin,
    
    /// Collecting channel values for select
    FiberSelectChans {
        patterns: Vec<Pattern>,
        bodies: Vec<Rc<Expr>>,
        remaining_chan_exprs: Vec<Expr>,
        collected_channels: Vec<ChannelId>,
        env: Env,
    },
    
    /// All select channels evaluated, ready to capture and return Select effect
    FiberSelectReady {
        channels: Vec<ChannelId>,
        patterns: Vec<Pattern>,
        bodies: Vec<Rc<Expr>>,
        env: Env,
    },
}
```

### 4.4 Phase 3: Implement Continuation Capture

**File**: `eval.rs`

**Task**: Add helper function to capture frames up to FiberBoundary.

```rust
impl Interpreter {
    /// Capture all frames up to (but not including) FiberBoundary.
    /// Returns the captured continuation.
    /// 
    /// The FiberBoundary frame remains on the continuation stack.
    /// Captured frames are reversed so outermost is first (for proper restoration).
    /// 
    /// # Panics
    /// Panics if no FiberBoundary is found (indicates a bug - all fibers should
    /// have a boundary installed at spawn time).
    fn capture_to_fiber_boundary(&self, cont: &mut Cont) -> Cont {
        let mut captured_frames = Vec::new();
        
        loop {
            match cont.pop() {
                None => {
                    // No FiberBoundary found - this is a bug
                    // In production, you might want to return an error instead
                    panic!("capture_to_fiber_boundary: no FiberBoundary found on stack");
                }
                Some(Frame::FiberBoundary) => {
                    // Found the boundary - put it back, we're done capturing
                    cont.push(Frame::FiberBoundary);
                    break;
                }
                Some(frame) => {
                    // Capture this frame
                    captured_frames.push(frame);
                }
            }
        }
        
        // Reverse so outermost frame is first
        // This makes restoration simple: just push in order
        captured_frames.reverse();
        
        Cont { frames: captured_frames }
    }
    
    /// Create a FiberEffect value and prepare to return it.
    /// The effect will propagate up to the FiberBoundary frame.
    fn return_fiber_effect(&self, effect: FiberEffect, cont: Cont) -> StepResult {
        StepResult::Continue(State::Apply {
            value: Value::FiberEffect(effect),
            cont,
        })
    }
}
```

### 4.5 Phase 4: Convert Channel Operations to Effects

**File**: `eval.rs`

**Task**: Modify channel operation handling in `step_eval` and `step_apply`.

#### 4.5.1 Modify `step_eval` for Channel Operations

```rust
// In step_eval match on ExprKind:

ExprKind::NewChannel => {
    // Capture continuation and return NewChan effect
    let captured = self.capture_to_fiber_boundary(&mut cont);
    self.return_fiber_effect(
        FiberEffect::NewChan { cont: Box::new(captured) },
        cont,
    )
}

ExprKind::ChanRecv(channel_expr) => {
    // First evaluate the channel expression, then capture
    cont.push(Frame::FiberRecv);
    StepResult::Continue(State::Eval {
        expr: channel_expr.clone(),
        env,
        cont,
    })
}

ExprKind::ChanSend { channel, value } => {
    // Evaluate channel, then value, then capture
    cont.push(Frame::FiberSendValue { 
        value_expr: value.clone(), 
        env: env.clone() 
    });
    StepResult::Continue(State::Eval {
        expr: channel.clone(),
        env,
        cont,
    })
}
```

#### 4.5.2 Modify `step_apply` for Channel Frames

```rust
// In step_apply match on Frame:

Some(Frame::FiberRecv) => {
    match value {
        Value::Channel(channel_id) => {
            let captured = self.capture_to_fiber_boundary(&mut cont);
            self.return_fiber_effect(
                FiberEffect::Recv {
                    channel: channel_id,
                    cont: Box::new(captured),
                },
                cont,
            )
        }
        _ => StepResult::Error(EvalError::TypeError(
            "recv expects a channel".into()
        )),
    }
}

Some(Frame::FiberSendValue { value_expr, env }) => {
    match value {
        Value::Channel(channel_id) => {
            // Got the channel, now evaluate the value to send
            cont.push(Frame::FiberSendReady { channel: channel_id });
            StepResult::Continue(State::Eval {
                expr: value_expr,
                env,
                cont,
            })
        }
        _ => StepResult::Error(EvalError::TypeError(
            "send expects a channel".into()
        )),
    }
}

Some(Frame::FiberSendReady { channel }) => {
    // Got both channel and value, capture and return effect
    let captured = self.capture_to_fiber_boundary(&mut cont);
    self.return_fiber_effect(
        FiberEffect::Send {
            channel,
            value: Box::new(value),
            cont: Box::new(captured),
        },
        cont,
    )
}
```

### 4.6 Phase 5: Convert Spawn/Join to Effects

**File**: `eval.rs`

#### 4.6.1 Add Join to AST (if not present)

**File**: `ast.rs`

```rust
pub enum ExprKind {
    // ... existing variants ...
    
    /// Join a fiber: join pid
    Join(Rc<Expr>),
    
    /// Yield control: yield
    Yield,
}
```

#### 4.6.2 Modify `step_eval` for Spawn/Join/Yield

```rust
ExprKind::Spawn(thunk_expr) => {
    // Evaluate the thunk expression, then capture for Fork
    cont.push(Frame::FiberFork);
    StepResult::Continue(State::Eval {
        expr: thunk_expr.clone(),
        env,
        cont,
    })
}

ExprKind::Join(pid_expr) => {
    // Evaluate the pid expression, then capture for Join
    cont.push(Frame::FiberJoin);
    StepResult::Continue(State::Eval {
        expr: pid_expr.clone(),
        env,
        cont,
    })
}

ExprKind::Yield => {
    // Capture immediately and return Yield effect
    let captured = self.capture_to_fiber_boundary(&mut cont);
    self.return_fiber_effect(
        FiberEffect::Yield { cont: Box::new(captured) },
        cont,
    )
}
```

#### 4.6.3 Modify `step_apply` for Spawn/Join Frames

```rust
Some(Frame::FiberFork) => {
    // value is the thunk to fork
    let captured = self.capture_to_fiber_boundary(&mut cont);
    self.return_fiber_effect(
        FiberEffect::Fork {
            thunk: Box::new(value),
            cont: Box::new(captured),
        },
        cont,
    )
}

Some(Frame::FiberJoin) => {
    match value {
        Value::Pid(pid) => {
            let captured = self.capture_to_fiber_boundary(&mut cont);
            self.return_fiber_effect(
                FiberEffect::Join {
                    pid,
                    cont: Box::new(captured),
                },
                cont,
            )
        }
        _ => StepResult::Error(EvalError::TypeError(
            "join expects a pid".into()
        )),
    }
}
```

### 4.7 Phase 6: Handle FiberBoundary Completion

**File**: `eval.rs`

When a fiber completes normally (value reaches FiberBoundary), wrap in Done:

```rust
// In step_apply match on Frame:

Some(Frame::FiberBoundary) => {
    // Fiber completed normally - wrap result in Done effect
    StepResult::Continue(State::Apply {
        value: Value::FiberEffect(FiberEffect::Done(Box::new(value))),
        cont,
    })
}
```

### 4.8 Phase 7: Unify Scheduler Loop

**File**: `runtime.rs`

#### 4.8.1 Simplified Process State

```rust
pub enum ProcessState {
    /// Ready to run
    Ready,
    /// Blocked on a channel send
    BlockedSend { 
        channel: ChannelId, 
        value: Value, 
        cont: Cont 
    },
    /// Blocked on a channel receive
    BlockedRecv { 
        channel: ChannelId, 
        cont: Cont 
    },
    /// Blocked waiting for another fiber to complete
    BlockedJoin { 
        waiting_for: Pid, 
        cont: Cont 
    },
    /// Blocked on select
    BlockedSelect { 
        arms: Vec<SelectEffectArm>, 
        cont: Cont 
    },
    /// Terminated with a result
    Done(Option<Value>),
}

pub struct Process {
    pub pid: Pid,
    pub state: ProcessState,
    /// For Ready state: how to start/resume the fiber
    pub continuation: Option<ProcessContinuation>,
}

pub enum ProcessContinuation {
    /// Start executing a thunk (initial spawn)
    Start(Value),
    /// Resume with a value fed into the continuation
    Resume { cont: Cont, value: Value },
}
```

#### 4.8.2 New Scheduler Methods

```rust
impl Runtime {
    /// Run a fiber until it produces a FiberEffect.
    /// 
    /// This repeatedly steps the interpreter until:
    /// - A FiberEffect value is produced (returned)
    /// - An error occurs (returned as Err)
    pub fn run_fiber_until_effect(
        &mut self,
        interp: &mut Interpreter,
        pid: Pid,
    ) -> Result<FiberEffect, EvalError> {
        let process = self.processes.get(&pid)
            .ok_or_else(|| EvalError::RuntimeError("unknown pid".into()))?;
        
        let initial_state = match process.borrow_mut().continuation.take() {
            Some(ProcessContinuation::Start(thunk)) => {
                // Starting a new fiber - set up initial state with FiberBoundary
                let mut cont = Cont::new();
                cont.push(Frame::FiberBoundary);
                
                // Apply thunk to unit: thunk ()
                cont.push(Frame::AppArg { func: thunk });
                State::Apply { value: Value::Unit, cont }
            }
            Some(ProcessContinuation::Resume { cont, value }) => {
                // Resuming a suspended fiber
                State::Apply { value, cont }
            }
            None => {
                return Err(EvalError::RuntimeError(
                    "fiber has no continuation".into()
                ));
            }
        };
        
        // Step until we get a FiberEffect
        let mut state = initial_state;
        loop {
            match interp.step(state) {
                StepResult::Continue(next_state) => {
                    state = next_state;
                }
                StepResult::Done(value) => {
                    // Shouldn't happen if FiberBoundary is set up correctly
                    return Ok(FiberEffect::Done(Box::new(value)));
                }
                StepResult::Error(e) => {
                    return Err(e);
                }
            }
            
            // Check if current state is returning a FiberEffect
            if let State::Apply { value: Value::FiberEffect(effect), .. } = &state {
                return Ok(effect.clone());
            }
        }
    }
    
    /// Resume a fiber by giving it a continuation and value, marking it ready.
    pub fn resume_fiber(&mut self, pid: Pid, cont: Cont, value: Value) {
        if let Some(process) = self.processes.get(&pid) {
            let mut p = process.borrow_mut();
            p.state = ProcessState::Ready;
            p.continuation = Some(ProcessContinuation::Resume { cont, value });
        }
        self.ready_queue.push_back(pid);
    }
    
    /// Mark a fiber as completed with a result value.
    /// Wakes up any fibers waiting to join on this one.
    pub fn complete_fiber(&mut self, pid: Pid, result: Value) {
        // Store the result
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::Done(Some(result.clone()));
        }
        
        // Wake up any joiners
        let waiters: Vec<_> = self.processes.iter()
            .filter_map(|(&waiter_pid, proc)| {
                let p = proc.borrow();
                if let ProcessState::BlockedJoin { waiting_for, .. } = &p.state {
                    if *waiting_for == pid {
                        return Some(waiter_pid);
                    }
                }
                None
            })
            .collect();
        
        for waiter_pid in waiters {
            if let Some(proc) = self.processes.get(&waiter_pid) {
                let mut p = proc.borrow_mut();
                if let ProcessState::BlockedJoin { cont, .. } = 
                    std::mem::replace(&mut p.state, ProcessState::Ready) 
                {
                    p.continuation = Some(ProcessContinuation::Resume { 
                        cont, 
                        value: result.clone() 
                    });
                    drop(p);
                    self.ready_queue.push_back(waiter_pid);
                }
            }
        }
    }
    
    /// Main scheduler loop - runs until main fiber completes or deadlock.
    pub fn run(&mut self, interp: &mut Interpreter) -> Result<Value, EvalError> {
        while let Some(pid) = self.ready_queue.pop_front() {
            self.current_pid = Some(pid);
            
            let effect = self.run_fiber_until_effect(interp, pid)?;
            
            match effect {
                FiberEffect::Done(value) => {
                    self.complete_fiber(pid, *value);
                    
                    // Check if this was the main fiber
                    if Some(pid) == self.main_pid {
                        if let Some(proc) = self.processes.get(&pid) {
                            if let ProcessState::Done(Some(result)) = 
                                &proc.borrow().state 
                            {
                                return Ok(result.clone());
                            }
                        }
                    }
                }
                
                FiberEffect::Fork { thunk, cont } => {
                    // Spawn child fiber
                    let child_pid = self.spawn_internal(*thunk);
                    
                    // Resume parent with child's pid
                    self.resume_fiber(pid, *cont, Value::Pid(child_pid));
                }
                
                FiberEffect::Yield { cont } => {
                    // Put back in ready queue (at the end for fairness)
                    self.resume_fiber(pid, *cont, Value::Unit);
                }
                
                FiberEffect::NewChan { cont } => {
                    let channel_id = self.new_channel();
                    self.resume_fiber(pid, *cont, Value::Channel(channel_id));
                }
                
                FiberEffect::Send { channel, value, cont } => {
                    if let Some(receiver_pid) = self.try_complete_send(channel, &value) {
                        // Rendezvous succeeded - both fibers continue
                        // Sender resumes with Unit
                        self.resume_fiber(pid, *cont, Value::Unit);
                        // Receiver was already woken in try_complete_send
                    } else {
                        // No receiver ready - block sender
                        self.block_on_send(pid, channel, *value, *cont);
                    }
                }
                
                FiberEffect::Recv { channel, cont } => {
                    if let Some(value) = self.try_complete_recv(channel) {
                        // Rendezvous succeeded - resume with received value
                        self.resume_fiber(pid, *cont, value);
                    } else {
                        // No sender ready - block receiver
                        self.block_on_recv(pid, channel, *cont);
                    }
                }
                
                FiberEffect::Join { pid: child_pid, cont } => {
                    // Check if child already completed
                    if let Some(result) = self.get_fiber_result(child_pid) {
                        self.resume_fiber(pid, *cont, result);
                    } else {
                        // Block until child completes
                        self.block_on_join(pid, child_pid, *cont);
                    }
                }
                
                FiberEffect::Select { arms, cont } => {
                    // Try each channel to see if any sender is waiting
                    let mut found = None;
                    for (i, arm) in arms.iter().enumerate() {
                        if let Some(value) = self.try_complete_recv(arm.channel) {
                            found = Some((i, value));
                            break;
                        }
                    }
                    
                    if let Some((arm_index, value)) = found {
                        // One channel was ready - set up to evaluate that arm's body
                        self.handle_select_ready(pid, *cont, arm_index, value, arms);
                    } else {
                        // None ready - block on all channels
                        self.block_on_select(pid, arms, *cont);
                    }
                }
            }
            
            self.current_pid = None;
        }
        
        // No more ready fibers
        if self.is_deadlocked() {
            Err(EvalError::Deadlock)
        } else if let Some(main_pid) = self.main_pid {
            // Return main fiber's result
            self.get_fiber_result(main_pid)
                .ok_or_else(|| EvalError::RuntimeError("main fiber has no result".into()))
        } else {
            Ok(Value::Unit)
        }
    }
}
```

#### 4.8.3 Helper Methods for Blocking

```rust
impl Runtime {
    fn block_on_send(&mut self, pid: Pid, channel: ChannelId, value: Value, cont: Cont) {
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::BlockedSend { 
                channel, value, cont 
            };
        }
        // Register with channel
        if let Some(ch) = self.channels.get(&channel) {
            ch.borrow_mut().senders.push_back(pid);
        }
    }
    
    fn block_on_recv(&mut self, pid: Pid, channel: ChannelId, cont: Cont) {
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::BlockedRecv { 
                channel, cont 
            };
        }
        if let Some(ch) = self.channels.get(&channel) {
            ch.borrow_mut().receivers.push_back(pid);
        }
    }
    
    fn block_on_join(&mut self, pid: Pid, waiting_for: Pid, cont: Cont) {
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::BlockedJoin { 
                waiting_for, cont 
            };
        }
    }
    
    fn block_on_select(&mut self, pid: Pid, arms: Vec<SelectEffectArm>, cont: Cont) {
        // Register as receiver on all channels
        for arm in &arms {
            if let Some(ch) = self.channels.get(&arm.channel) {
                ch.borrow_mut().receivers.push_back(pid);
            }
        }
        
        if let Some(process) = self.processes.get(&pid) {
            process.borrow_mut().state = ProcessState::BlockedSelect { arms, cont };
        }
    }
    
    /// Try to complete a send by finding a waiting receiver.
    /// Returns the receiver's pid if successful (and wakes them up).
    fn try_complete_send(&mut self, channel: ChannelId, value: &Value) -> Option<Pid> {
        let receiver_pid = {
            let ch = self.channels.get(&channel)?;
            ch.borrow_mut().receivers.pop_front()
        }?;
        
        // Wake up receiver with the value
        if let Some(proc) = self.processes.get(&receiver_pid) {
            let mut p = proc.borrow_mut();
            match std::mem::replace(&mut p.state, ProcessState::Ready) {
                ProcessState::BlockedRecv { cont, .. } => {
                    p.continuation = Some(ProcessContinuation::Resume { 
                        cont, 
                        value: value.clone() 
                    });
                }
                ProcessState::BlockedSelect { arms, cont } => {
                    // Find which arm matched and handle it
                    drop(p);
                    self.handle_select_wakeup(receiver_pid, channel, value.clone(), arms, cont);
                    return Some(receiver_pid);
                }
                other => {
                    p.state = other; // Restore - shouldn't happen
                    return None;
                }
            }
        }
        
        self.ready_queue.push_back(receiver_pid);
        Some(receiver_pid)
    }
    
    /// Try to complete a receive by finding a waiting sender.
    /// Returns the value if successful (and wakes up the sender).
    fn try_complete_recv(&mut self, channel: ChannelId) -> Option<Value> {
        let (sender_pid, value) = {
            let ch = self.channels.get(&channel)?;
            let mut ch = ch.borrow_mut();
            
            // Look for a sender
            let sender_pid = ch.senders.pop_front()?;
            
            // Get the value from the sender's blocked state
            let proc = self.processes.get(&sender_pid)?;
            let p = proc.borrow();
            if let ProcessState::BlockedSend { value, .. } = &p.state {
                (sender_pid, value.clone())
            } else {
                return None;
            }
        };
        
        // Wake up sender
        if let Some(proc) = self.processes.get(&sender_pid) {
            let mut p = proc.borrow_mut();
            if let ProcessState::BlockedSend { cont, .. } = 
                std::mem::replace(&mut p.state, ProcessState::Ready) 
            {
                p.continuation = Some(ProcessContinuation::Resume { 
                    cont, 
                    value: Value::Unit 
                });
            }
        }
        self.ready_queue.push_back(sender_pid);
        
        Some(value)
    }
    
    fn get_fiber_result(&self, pid: Pid) -> Option<Value> {
        let proc = self.processes.get(&pid)?;
        if let ProcessState::Done(Some(value)) = &proc.borrow().state {
            Some(value.clone())
        } else {
            None
        }
    }
}
```

---

## 5. Detailed Code Changes

### 5.1 Files to Modify

| File | Changes |
|------|---------|
| `ast.rs` | Add `ExprKind::Join`, `ExprKind::Yield` (if not present) |
| `eval.rs` | Add `FiberEffect`, new `Frame` variants, capture logic, modify step functions |
| `runtime.rs` | New `ProcessState`, unified scheduler loop, helper methods |
| `parser.rs` | Parse `join` and `yield` keywords (if adding new syntax) |
| `lexer.rs` | Tokenize `join` and `yield` keywords (if adding new syntax) |
| `infer.rs` | Type rules for `join` and `yield` |

### 5.2 Removal Checklist

After migration, remove:

- [ ] `StepResult::Blocked` variant
- [ ] `BlockReason` enum
- [ ] Old `Frame::Recv`, `Frame::SendChan`, `Frame::SendVal`, etc.
- [ ] `Process.saved_cont` field
- [ ] `Process.received_value` field
- [ ] `Process.select_fired_channel` field
- [ ] `ProcessContinuation::ResumeAfterRecv`, `ResumeAfterSend`, `ResumeAfterSelect`
- [ ] Old blocking/wakeup logic in `step_apply`

### 5.3 Testing During Migration

After each phase, ensure existing tests pass:

```bash
cargo test
```

Key test files to verify:
- Channel communication tests
- Spawn tests
- Select tests
- Deadlock detection tests

---

## 6. User-Facing API

### 6.1 Primitives

```
-- Fiber creation
spawn : (() -> a) -> Pid
join : Pid -> a

-- Cooperative scheduling
yield : ()

-- Channels
new_channel : () -> Channel a
send : Channel a -> a -> ()
recv : Channel a -> a

-- Select (existing syntax)
select
| x <- ch1 -> expr1
| y <- ch2 -> expr2
```

### 6.2 Library Functions (User-Definable)

```
-- Run two computations in parallel, return both results
let parallel a b =
    let ch_a = new_channel () in
    let ch_b = new_channel () in
    
    let _ = spawn (fun () -> send ch_a (a ())) in
    let _ = spawn (fun () -> send ch_b (b ())) in
    
    let result_a = recv ch_a in
    let result_b = recv ch_b in
    (result_a, result_b)

-- Race two computations, return first to complete
let race a b =
    let ch = new_channel () in
    let _ = spawn (fun () -> send ch (Left (a ()))) in
    let _ = spawn (fun () -> send ch (Right (b ()))) in
    recv ch

-- Fork-join pattern
let fork_join tasks =
    let pids = map (fun task -> spawn task) tasks in
    map join pids

-- Worker pool
let worker_pool num_workers work_fn job_list =
    let job_ch = new_channel () in
    let result_ch = new_channel () in
    
    -- Spawn workers
    let rec spawn_workers n =
        if n <= 0 then ()
        else (
            spawn (fun () ->
                let rec loop () =
                    let job = recv job_ch in
                    send result_ch (work_fn job);
                    loop ()
                in loop ()
            );
            spawn_workers (n - 1)
        )
    in
    spawn_workers num_workers;
    
    -- Submit jobs
    let rec submit jobs =
        match jobs with
        | [] -> ()
        | j :: rest -> send job_ch j; submit rest
    in
    submit job_list;
    
    -- Collect results
    let rec collect n acc =
        if n <= 0 then acc
        else collect (n - 1) (recv result_ch :: acc)
    in
    collect (length job_list) []
```

### 6.3 Example Programs

#### Producer-Consumer

```
let producer_consumer () =
    let buffer = new_channel () in
    
    let producer = spawn (fun () ->
        let rec produce n =
            if n > 10 then ()
            else (
                send buffer n;
                produce (n + 1)
            )
        in
        produce 1
    ) in
    
    let consumer = spawn (fun () ->
        let rec consume total count =
            if count >= 10 then total
            else
                let x = recv buffer in
                consume (total + x) (count + 1)
        in
        consume 0 0
    ) in
    
    join producer;
    let sum = join consumer in
    print ("Sum: " ++ int_to_string sum)  -- Sum: 55
```

#### Ping-Pong

```
let ping_pong () =
    let ping_ch = new_channel () in
    let pong_ch = new_channel () in
    
    let pinger = spawn (fun () ->
        let rec ping n =
            if n <= 0 then ()
            else (
                send ping_ch "ping";
                let _ = recv pong_ch in
                print ("ping " ++ int_to_string n);
                ping (n - 1)
            )
        in
        ping 5
    ) in
    
    let ponger = spawn (fun () ->
        let rec pong () =
            select
            | msg <- ping_ch ->
                print ("received: " ++ msg);
                send pong_ch "pong";
                pong ()
        in
        pong ()
    ) in
    
    join pinger;
    print "done"
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

#### Phase 1-2: FiberEffect and FiberBoundary

```rust
#[test]
fn test_fiber_boundary_captures_done() {
    // A simple expression that completes should produce FiberEffect::Done
    let mut interp = Interpreter::new();
    let mut cont = Cont::new();
    cont.push(Frame::FiberBoundary);
    
    let state = State::Apply { value: Value::Int(42), cont };
    
    // Step should produce FiberEffect::Done(42)
    match interp.step(state) {
        StepResult::Continue(State::Apply { 
            value: Value::FiberEffect(FiberEffect::Done(v)), .. 
        }) => {
            assert_eq!(*v, Value::Int(42));
        }
        _ => panic!("expected FiberEffect::Done"),
    }
}
```

#### Phase 4-5: Channel and Spawn Effects

```rust
#[test]
fn test_recv_produces_effect() {
    // recv ch should capture continuation and produce FiberEffect::Recv
}

#[test]
fn test_send_produces_effect() {
    // send ch v should capture and produce FiberEffect::Send
}

#[test]
fn test_spawn_produces_fork_effect() {
    // spawn thunk should produce FiberEffect::Fork
}
```

#### Phase 6: Scheduler Integration

```rust
#[test]
fn test_scheduler_handles_fork() {
    // Fork should create new fiber, resume parent with child pid
}

#[test]
fn test_scheduler_handles_rendezvous() {
    // Send+Recv should complete rendezvous and wake both fibers
}

#[test]
fn test_scheduler_handles_join() {
    // Join should block until child completes, then resume with result
}
```

### 7.2 Integration Tests

```rust
#[test]
fn test_producer_consumer() {
    let program = r#"
        let main () =
            let ch = new_channel () in
            let p = spawn (fun () -> send ch 42) in
            let x = recv ch in
            join p;
            x
    "#;
    let result = run_program(program);
    assert_eq!(result, Ok(Value::Int(42)));
}

#[test]
fn test_parallel_execution() {
    let program = r#"
        let main () =
            let ch1 = new_channel () in
            let ch2 = new_channel () in
            let _ = spawn (fun () -> send ch1 1) in
            let _ = spawn (fun () -> send ch2 2) in
            let a = recv ch1 in
            let b = recv ch2 in
            a + b
    "#;
    let result = run_program(program);
    assert_eq!(result, Ok(Value::Int(3)));
}

#[test]
fn test_join_waits_for_completion() {
    let program = r#"
        let main () =
            let child = spawn (fun () ->
                yield ();
                yield ();
                42
            ) in
            join child
    "#;
    let result = run_program(program);
    assert_eq!(result, Ok(Value::Int(42)));
}

#[test]
fn test_deadlock_detection() {
    let program = r#"
        let main () =
            let ch = new_channel () in
            recv ch  -- No sender, deadlock!
    "#;
    let result = run_program(program);
    assert!(matches!(result, Err(EvalError::Deadlock)));
}
```

### 7.3 Regression Tests

Ensure all existing tests continue to pass:

- `test_channel_basic`
- `test_spawn_basic`
- `test_select_basic`
- `test_rendezvous_synchronization`
- All shift/reset tests (should be unaffected)

---

## 8. Migration Guide

### 8.1 Step-by-Step Migration

1. **Create new types** (Phase 1-2)
   - Add `FiberEffect` enum
   - Add new `Frame` variants
   - Don't remove old code yet

2. **Add capture function** (Phase 3)
   - Implement `capture_to_fiber_boundary`
   - Test in isolation

3. **Parallel implementation** (Phase 4-5)
   - Add new handlers alongside old ones
   - Use feature flag or config to switch

4. **Scheduler migration** (Phase 6)
   - Update scheduler to handle `FiberEffect`
   - Test with new code paths

5. **Cleanup** (Phase 7)
   - Remove old `StepResult::Blocked`
   - Remove old frame types
   - Remove old process state fields

### 8.2 Rollback Plan

Keep old code behind feature flag during migration:

```rust
#[cfg(feature = "new_fiber_runtime")]
fn handle_recv(...) { /* new implementation */ }

#[cfg(not(feature = "new_fiber_runtime"))]
fn handle_recv(...) { /* old implementation */ }
```

### 8.3 Breaking Changes

**None for user code** - the API remains the same:
- `spawn`, `send`, `recv`, `select` work identically
- Only internal implementation changes

---

## 9. Future Extensions

### 9.1 Cancellation

```rust
FiberEffect::Cancel {
    pid: Pid,
    cont: Box<Cont>,
}
```

```
cancel : Pid -> ()
```

### 9.2 Timeouts

```rust
FiberEffect::Sleep {
    duration: Duration,
    cont: Box<Cont>,
}
```

```
sleep : Int -> ()  -- milliseconds

let with_timeout duration action =
    race action (fun () -> sleep duration; None)
```

### 9.3 Error Handling / Supervision

```rust
FiberEffect::Supervise {
    child_thunk: Box<Value>,
    cont: Box<Cont>,  // Receives Result a Error
}
```

```
supervise : (() -> a) -> Result a Error
```

### 9.4 Resource Management

```
bracket : (() -> r) -> (r -> ()) -> (r -> a) -> a
```

Implemented as library function with proper cleanup on fiber cancellation.

### 9.5 Async I/O Integration

The scheduler can be extended to integrate with OS async I/O:

```rust
FiberEffect::AsyncRead {
    fd: FileDescriptor,
    buffer: Buffer,
    cont: Box<Cont>,
}
```

This would allow the scheduler to use `epoll`/`kqueue` for efficient I/O multiplexing.

---

## Appendix A: Type Inference Changes

### A.1 New Expression Types

```rust
// In infer.rs

ExprKind::Join(pid_expr) => {
    let pid_ty = self.infer_expr(env, pid_expr)?;
    self.unify_at(&pid_ty, &Type::Pid, &pid_expr.span)?;
    
    // Join returns the fiber's result type, which we don't know statically
    // For now, return a fresh type variable
    let result_ty = self.fresh_var();
    Ok(InferResult::pure(result_ty, ans))
}

ExprKind::Yield => {
    Ok(InferResult::pure(Type::Unit, ans))
}
```

### A.2 Note on Fiber Result Types

The type of `join` is challenging because we don't track fiber result types through `Pid`. Options:

1. **Dynamic typing for join results** - return `Any` or fresh var
2. **Typed fiber references** - `Fiber a` instead of `Pid`
3. **Effect type tracking** - requires more complex type system

For now, option 1 is simplest and matches how many languages handle this.

---

## Appendix B: Preserving User shift/reset

The fiber effect system should NOT interfere with user-defined effects via shift/reset.

### B.1 Key Invariant

- `Frame::Prompt` is for user-level shift/reset
- `Frame::FiberBoundary` is for fiber effects
- They use different capture mechanisms:
  - `shift` captures to nearest `Prompt`
  - Fiber ops capture to nearest `FiberBoundary`

### B.2 Nesting Behavior

```
reset (                      -- Pushes Prompt
    spawn (fun () ->         -- This is in a FiberBoundary context
        reset (              -- Inner Prompt
            shift (fun k ->  -- Captures to inner Prompt
                k 1 + k 2
            )
        )
    )
)
```

This should work correctly:
- `shift` captures to the inner `Prompt`, not to `FiberBoundary`
- The spawn still works because it captures to `FiberBoundary`

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Fiber** | A lightweight, cooperatively scheduled thread of execution |
| **Continuation** | The "rest of the computation" after a suspension point |
| **FiberBoundary** | Implicit delimiter marking the extent of a fiber's continuation |
| **Rendezvous** | Synchronous communication where sender and receiver meet |
| **Effect** | An operation that suspends a fiber awaiting runtime handling |
| **Scheduler** | The runtime component that decides which fiber runs next |

---

*End of Design Document*
