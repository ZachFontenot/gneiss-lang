# Gneiss Development Roadmap

## Completed Work

The following features have been implemented and are working:

### Concurrency (Original Phase 1) ✓
- **CPS Interpreter**: Defunctionalized continuation-passing style with explicit frame stack
- **Channels**: Synchronous rendezvous channels with proper blocking/resumption
- **Select**: Multi-arm select over multiple channels with non-deterministic choice
- **Scheduler**: Single-threaded cooperative scheduler with round-robin ready queue
- **Deadlock Detection**: Detects when all processes are blocked
- **Process Communication**: Two-way ping-pong, producer-consumer patterns working

### Typeclasses ✓
- **Trait Declarations**: `trait Show a = val show : a -> String end`
- **Instance Declarations**: Basic and constrained (`impl Show for (List a) where a : Show`)
- **Dictionary Passing**: Runtime method dispatch via dictionaries
- **Instance Resolution**: Proper constraint propagation and overlap detection
- **Arbitrary User-Defined Traits**: Not limited to built-ins

### Delimited Continuations ✓
- **Reset/Shift**: Full implementation of delimited continuations
- **Multi-invocation**: Captured continuations can be called multiple times
- **Nested Prompts**: Proper nesting of reset boundaries

### Core Language ✓
- Hindley-Milner type inference with let-polymorphism
- ADTs (algebraic data types) with pattern matching
- Local recursive functions (`let f x = ... f ... in body`)
- Value restriction for sound polymorphism

---

## Phase 2: Error Infrastructure

**Goal:** Better error messages with source location and context.

### 2.1 Source Location Framework
- [ ] Add line:column position tracking (convert byte offset to line/col)
- [ ] Create source location infrastructure for error reporting
- [ ] Store source text for context printing

### 2.2 Error Pretty-Printing
- [ ] Print source line with error
- [ ] Show caret pointing to error location
- [ ] Multi-line error context for complex errors

### 2.3 Update Error Types
- [ ] Update ParseError to use new framework
- [ ] Update TypeError to use new framework
- [ ] Update EvalError to use new framework

### 2.4 Structured Output
- [ ] JSON error format option (for tooling integration)
- [ ] Consistent error format across all error types

---

## Phase 3: Module System

**Goal:** Multi-file projects with imports and exports.

### 3.1 Syntax
```gneiss
-- File: List.gn
module List

let map f xs = ...
let filter p xs = ...

-- File: Main.gn
module Main

import List
import List (map, filter)  -- selective import
import List as L           -- qualified import

let main () = List.map double [1, 2, 3]
```

### 3.2 Tasks
- [ ] Add tokens: `module`, `import`, `export`, `as`
- [ ] Add AST nodes for module/import/export declarations
- [ ] Parse module declarations and imports
- [ ] Implement module name resolution (file→module mapping)
- [ ] Implement import resolution and name binding
- [ ] Add circular dependency detection
- [ ] Add visibility controls (public/private)

### 3.3 Visibility
- [ ] Default: public exports
- [ ] `private let` for module-internal bindings
- [ ] Or: explicit export list

---

## Phase 4: REPL & Tooling

**Goal:** Improved developer experience.

### 4.1 REPL Improvements
- [ ] History and line editing support
- [ ] Multi-line input mode
- [ ] `:load` command to load .gn files
- [ ] `:env` command to show current bindings
- [ ] `:type` expression type inspection (basic version exists)

### 4.2 Build System
- [ ] `gneiss.toml` project manifest
- [ ] Source directories configuration
- [ ] Build command for multi-file projects

---

## Phase 5: Performance

### 5.1 Bytecode Compiler
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

## Phase 6: Compilation Target

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

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Concurrency model | CML channels, not Erlang mailboxes | Matches Go-style, typed channels |
| Type system | Hindley-Milner | Simple, well-understood |
| Typeclasses | Dictionary passing | Simpler than monomorphization, works with separate compilation |
| Continuations | Delimited (shift/reset) | Composable, typed, enables effects |
| Parser | Handwritten recursive descent | Control over error messages |
| Runtime | Single-threaded cooperative | Simpler for v0.1, can add threads later |
| Syntax | OCaml-inspired | Familiar to ML users, minimal noise |
| Module syntax | Explicit `module` keyword | Clear file structure, not implicit from filename |
| Sendable constraint | Deferred | Runtime copies everything, no shared memory |

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
