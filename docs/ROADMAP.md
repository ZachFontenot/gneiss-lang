# Gneiss Development Roadmap

## Current State (v0.1-alpha)

**Working:**
- Lexer and parser (handwritten recursive descent)
- Hindley-Milner type inference with let-polymorphism
- Tree-walking interpreter
- Pattern matching (literals, variables, tuples, lists, cons, constructors)
- ADTs (algebraic data types)
- Recursion
- Basic REPL

**Scaffolded but incomplete:**
- Runtime scheduler (processes created but blocking/resumption not wired up)
- Channels (parse and typecheck, runtime handoff incomplete)

---

## Phase 1: Complete the Concurrency Story (v0.1)

**Goal:** Two processes communicating over a channel, end-to-end.

### 1.1 Fix the Scheduler Loop
The current interpreter runs `main`, spawns processes, but doesn't interleave execution properly.

Tasks:
- [ ] Refactor `eval` to be resumable (either CPS transform or explicit continuation stack)
- [ ] Implement proper yield points at `Channel.send` and `Channel.recv`
- [ ] Test: ping-pong between two processes

### 1.2 Blocking Semantics
Currently `send`/`recv` return immediately. Need:
- [ ] `send` blocks until a receiver is ready (rendezvous)
- [ ] `recv` blocks until a sender is ready
- [ ] Process state machine: Ready → Blocked → Ready → Done

### 1.3 Basic Select
- [ ] `select` over multiple receive operations
- [ ] Non-deterministic choice when multiple channels ready

### 1.4 Tests
- [ ] Unit tests for scheduler
- [ ] Integration test: producer-consumer
- [ ] Integration test: multiple channels

**Deliverable:** Can write and run concurrent Gneiss programs with channels.

---

## Phase 2: Error Handling & Polish (v0.2)

### 2.1 Better Error Messages
- [ ] Track source spans through inference
- [ ] Pretty-print type errors with source location
- [ ] Suggest fixes for common mistakes

### 2.2 Process Failure
- [ ] Processes can crash (panic)
- [ ] Option: linked processes (Erlang-style)
- [ ] Option: supervised restart

### 2.3 Timeouts
- [ ] `recv_timeout : Channel a -> Int -> Option a`
- [ ] Or: timeout as part of `select`

### 2.4 Standard Library Builtins
- [ ] List functions: `map`, `filter`, `fold`, `length`, `reverse`
- [ ] String functions: `concat`, `split`, `chars`
- [ ] Integer functions: `abs`, `min`, `max`
- [ ] IO: `read_line`, `print_line`

### 2.5 REPL Improvements
- [ ] Multi-line input
- [ ] History (readline/rustyline)
- [ ] `:load` command to load files
- [ ] Show process state

---

## Phase 3: Type System Extensions (v0.3)

Pick **one** of these initially:

### Option A: Typeclasses (Recommended)
```
trait Show a =
  show : a -> String

impl Show Int =
  show n = int_to_string n

impl Show a => Show (List a) =
  show xs = "[" ++ join ", " (map show xs) ++ "]"
```

Tasks:
- [ ] Parse trait/impl declarations
- [ ] Implement dictionary-passing or monomorphization
- [ ] Constrained type inference
- [ ] Useful traits: `Show`, `Eq`, `Ord`

### Option B: Row Polymorphism (Records)
```
let get_name r = r.name

-- Inferred: { name : a | r } -> a
```

Tasks:
- [ ] Record syntax in parser
- [ ] Row types in inference
- [ ] Field access and update

### Option C: Effect Tracking (Advanced)
```
let read_file : String -> IO String
let pure_fn : Int -> Int  -- no effects
```

Probably too ambitious for v0.3, but worth considering.

---

## Phase 4: Modules & Imports (v0.4)

### 4.1 Basic Modules
```
-- File: List.gn
let map f xs = ...
let filter p xs = ...

-- File: Main.gn  
import List

let main _ = List.map double [1, 2, 3]
```

Tasks:
- [ ] Module = file (no in-language module syntax initially)
- [ ] `import Module` brings names into scope
- [ ] `import Module (foo, bar)` selective import
- [ ] Circular import detection

### 4.2 Visibility
- [ ] Public by default, or `private let`?
- [ ] Or: export list at top of file

### 4.3 Package/Project Structure
- [ ] `gneiss.toml` manifest
- [ ] Source directories
- [ ] Dependencies (future)

---

## Phase 5: Performance (v0.5)

### 5.1 Bytecode Compiler
Replace tree-walking interpreter with:
- [ ] Design bytecode instruction set
- [ ] Compile AST → bytecode
- [ ] Stack-based VM

### 5.2 Tail Call Optimization
- [ ] Detect tail position
- [ ] Emit tail call instruction
- [ ] Critical for recursive process loops

### 5.3 Efficient Closures
- [ ] Closure conversion
- [ ] Flat closure representation
- [ ] Escape analysis (optional)

### 5.4 Better Scheduler
- [ ] Work-stealing (if multi-threaded)
- [ ] Process priorities
- [ ] Fairness guarantees

---

## Phase 6: Compilation Target (v1.0)

Choose one:

### Option A: BEAM (Erlang VM)
- Excellent concurrency runtime
- Battle-tested scheduler
- Hot code loading
- Challenge: semantic gap (async mailboxes vs sync channels)

### Option B: Native (via LLVM or Cranelift)
- Best performance
- Full control
- Challenge: must build GC and scheduler

### Option C: WebAssembly
- Browser and edge deployment
- Growing ecosystem
- Challenge: concurrency model (threads vs async)

---

## Recommended Next Weekend

If you have a weekend to hack:

1. **Day 1 Morning:** Refactor eval to use explicit continuation stack
2. **Day 1 Afternoon:** Wire up scheduler to actually interleave processes
3. **Day 1 Evening:** Get ping-pong working (two processes, one channel)

4. **Day 2 Morning:** Add `select` for multiple channels
5. **Day 2 Afternoon:** Write a few real examples (counter actor, worker pool)
6. **Day 2 Evening:** Clean up, add tests, document

This gets you to a **demo-able v0.1** where you can show off the core idea: ML-style types + Go-style channels.

---

## Reading List

**Concurrency:**
- "Concurrent ML" (Reppy) - the original CML paper
- "Erlang/OTP in Action" - practical actor patterns
- Go channel implementation in runtime source

**Type Systems:**
- "Types and Programming Languages" (Pierce) - chapters on HM
- "Typing Haskell in Haskell" - typeclasses implementation
- "Complete and Easy Bidirectional Typechecking" - if you want to go beyond HM

**Implementation:**
- "Crafting Interpreters" (Nystrom) - bytecode VM chapters
- "Programming Language Pragmatics" - compilation techniques
- "The Implementation of Functional Programming Languages" (SPJ) - STG machine

---

## Decision Log

Decisions made so far (for future reference):

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Concurrency model | CML channels, not Erlang mailboxes | Matches Go-style, typed channels |
| Type system | Hindley-Milner first | Simple, well-understood, typeclasses later |
| Parser | Handwritten recursive descent | Control over error messages, Chumsky was problematic |
| Runtime | Single-threaded cooperative | Simpler for v0.1, can add threads later |
| Syntax | OCaml-inspired | Familiar to ML users, minimal noise |
| Sendable constraint | Deferred | Runtime copies everything, no shared memory |
