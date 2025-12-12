# Gneiss Design Document

This document captures the design rationale, key decisions, and invariants for the Gneiss programming language. It is intended to be loaded into context by AI assistants and human contributors.

## Overview

Gneiss is a statically-typed functional programming language with:
- **ML-family syntax** (inspired by OCaml)
- **Hindley-Milner type inference** with let-polymorphism
- **CSP-style concurrency** with synchronous (rendezvous) channels

The goal is a language where concurrent programs are safe by construction: no data races, no shared mutable state, typed channels that prevent mixed-type sends.

## Core Design Decisions

### 1. Concurrency Model: CML, not Erlang

**Decision:** Synchronous rendezvous channels (Concurrent ML style), not async mailboxes (Erlang style).

**Rationale:**
- Go popularized CSP channels; this model is well-understood
- Synchronous semantics are simpler to reason about
- Typed channels (vs. untyped mailboxes) catch errors at compile time
- No need for selective receive pattern matching

**Implications:**
- `Channel.send` blocks until a receiver is ready
- `Channel.recv` blocks until a sender is ready
- Single-process send-then-recv deadlocks (this is correct behavior)
- Two processes must rendezvous for communication to occur

**Example of correct blocking:**
```
-- This DEADLOCKS (correctly!)
let main () =
  let ch = Channel.new in
  Channel.send ch 42;   -- blocks forever, no receiver
  Channel.recv ch       -- never reached
```

**Example of correct communication:**
```
-- This WORKS: two processes rendezvous
let main () =
  let ch = Channel.new in
  spawn (fun () -> Channel.send ch 42);
  Channel.recv ch  -- receives 42
```

### 2. Type System: Hindley-Milner Only

**Decision:** No typeclasses, no HKT, no traits, no row polymorphism (for now).

**Rationale:**
- HM is well-understood and sufficient for v0.1
- Typeclasses add significant implementation complexity
- Can be added later without breaking existing code
- Focus on getting concurrency semantics right first

**What we have:**
- Parametric polymorphism (`let id x = x` works at any type)
- Let-polymorphism (polymorphic let bindings)
- ADTs (algebraic data types) with pattern matching
- Type inference (almost no annotations needed)

**What we don't have (intentionally deferred):**
- Typeclasses / traits
- Higher-kinded types
- GADTs
- Row polymorphism / extensible records
- Type annotations on expressions (only on top-level bindings, optionally)

### 3. Value Restriction for Channels

**Decision:** Apply ML-style value restriction to prevent unsound polymorphic channels.

**The problem:**
```
let ch = Channel.new        -- What type? Channel 'a (polymorphic)?
let _ = spawn (fun () -> Channel.send ch 42)
let _ = spawn (fun () -> Channel.send ch "hello")  -- Unsound!
```

If `Channel.new` gets type `forall a. Channel a`, we can send different types through the same channel.

**The solution:** Only generalize *syntactic values* in let bindings.

Syntactic values are:
- Literals (`42`, `true`, `"hello"`)
- Variables
- Lambdas (`fun x -> ...`)
- Constructors applied to values
- Tuples/lists of values

`Channel.new` is NOT a syntactic value (it's an effectful expression), so:
```
let ch = Channel.new  -- NOT generalized, stays monomorphic
```

The channel's type is fixed at its first use, catching mixed-type sends at compile time.

**Implementation:** See `is_syntactic_value()` in `infer.rs`.

### 4. Interpreter Architecture: Defunctionalized CPS

**Decision:** CPS interpreter with explicit continuation stack, not direct recursion.

**Rationale:**
- Processes must be suspendable at channel operations
- Direct recursive `eval()` can't yield mid-expression
- CPS allows saving continuation and resuming later

**Key types:**
```rust
enum State {
    Eval { expr, env, cont },   // Evaluate an expression
    Apply { value, cont },       // Return a value to continuation
}

enum Frame {
    AppFunc { arg, env },        // Evaluating func, arg next
    AppArg { func },             // Have func, evaluating arg
    Let { pattern, body, env },  // Evaluated value, bind and continue
    // ... etc for each expression form
}

struct Cont {
    frames: Vec<Frame>,          // The continuation stack
}
```

**Blocking:**
```rust
enum StepResult {
    Continue(State),                              // More work
    Done(Value),                                  // Finished
    Blocked { reason: BlockReason, state: State }, // Suspended
}
```

When a process hits `Channel.send` with no receiver, it returns `Blocked` with its continuation saved. The scheduler picks another process. When a receiver arrives, the sender is resumed with its saved continuation.

### 5. Scheduler: Cooperative, Single-Threaded

**Decision:** Single-threaded cooperative scheduler for v0.1.

**Rationale:**
- Simpler to implement and debug
- No need for synchronization primitives
- Sufficient for demonstrating semantics
- Can add preemption/parallelism later

**Behavior:**
- Round-robin ready queue
- Processes yield at channel operations
- Deadlock detected when all processes blocked and ready queue empty

### 6. No Async/Buffered Channels

**Decision:** Channels are strictly synchronous with no buffering.

**Rationale:**
- Simpler semantics
- Forces explicit synchronization
- Buffered channels can be built on top if needed
- Matches CML semantics

**NOT supported:**
```
-- NO buffered channels
let ch = Channel.new_buffered 10  -- doesn't exist

-- NO non-blocking operations
let result = Channel.try_send ch x  -- doesn't exist
```

### 7. Typeclasses: Dictionary Passing

**Decision:** Implement typeclasses using dictionary passing at runtime.

**Rationale:**
- Simpler than monomorphization (no code duplication)
- Works with separate compilation
- Runtime overhead acceptable for interpreted language
- Clear semantics for constrained instances

**Syntax:**
```gneiss
trait Show a =
  val show : a -> String
end

impl Show for Int =
  let show n = int_to_string n
end

impl Show for (List a) where a : Show =
  let show xs = "[" ++ join ", " (map show xs) ++ "]"
end
```

**How it works:**
1. Type inference collects predicates (constraints) during inference
2. Instance resolution finds matching instances for each predicate
3. At runtime, dictionaries are constructed containing method implementations
4. Method calls dispatch through the dictionary

**Constraints:**
- Single-parameter typeclasses only (no multi-param or HKT)
- Overlapping instances are detected and rejected
- Orphan instances allowed (no coherence checking yet)

### 8. Delimited Continuations

**Decision:** Implement shift/reset style delimited continuations.

**Rationale:**
- Composable control flow primitive
- Can express exceptions, generators, coroutines, etc.
- Well-studied semantics
- Complements channel-based concurrency

**Syntax:**
```gneiss
-- reset delimits the continuation scope
-- shift captures the continuation up to the enclosing reset
let result = reset (
  1 + shift (fun k -> k (k 10))
)
-- result = 1 + (1 + 10) = 12
```

**Semantics:**
- `reset e` evaluates `e` with a prompt (delimiter)
- `shift (fun k -> body)` captures the continuation up to the nearest reset
- The captured continuation `k` can be called zero, once, or multiple times
- Calling `k v` resumes computation with `v` as the result of the shift

**Implementation:**
- Uses the CPS interpreter's frame stack
- `Frame::Prompt` marks reset boundaries
- `Value::Continuation` holds captured frames
- Continuation application splices frames back into the stack

## Invariants

These properties should always hold. Violating them is a bug.

### Type Safety Invariants

1. **Well-typed programs don't get runtime type errors.** If it passes type inference, evaluation won't produce `TypeError`.

2. **Channel type consistency.** Every value sent through a channel `ch : Channel T` has type `T`.

3. **Value restriction prevents polymorphic effectful bindings.** `Channel.new`, `spawn`, etc. are never generalized.

4. **Occurs check prevents infinite types.** `fun x -> x x` is rejected.

### Concurrency Invariants

5. **Rendezvous semantics.** Send and receive both block until a partner is available.

6. **No data races.** Processes share nothing; all communication is through channels.

7. **Deadlock is detectable.** When all processes are blocked and ready queue is empty, we report deadlock (not hang silently).

8. **Process isolation.** A process cannot access another process's local variables or continuation.

## File Structure

```
src/
  ast.rs      -- AST node definitions with Span for error reporting
  lexer.rs    -- Hand-written tokenizer (no parser generators)
  parser.rs   -- Recursive descent parser
  types.rs    -- Internal type representation (Type, TypeVar, Scheme)
  infer.rs    -- Hindley-Milner inference with value restriction
  eval.rs     -- CPS interpreter with Frame/Cont/State
  runtime.rs  -- Process scheduler, Channel, ready queue
  main.rs     -- REPL and file execution
  lib.rs      -- Public exports

tests/
  properties.rs  -- Property-based tests for soundness

examples/
  *.gn        -- Example programs

docs/
  SYNTAX.md   -- Full syntax reference
  ROADMAP.md  -- Development phases
```

## Testing Strategy

### Unit Tests (in each module)
- Lexer: token sequences
- Parser: AST structure
- Inference: type of expressions
- Eval: value of expressions
- Runtime: scheduler behavior

### Property Tests (`tests/properties.rs`)
- **Type preservation:** well-typed expr → well-typed value
- **Progress:** well-typed expr doesn't panic
- **Determinism:** same expr → same type/value
- **Occurs check:** self-application rejected
- **Let-polymorphism:** identity usable at multiple types

### Concurrency Tests (in `eval.rs`)
- Ping-pong communication
- Multiple messages through channel
- Deadlock detection
- Interleaving requirement (programs that need scheduler to not deadlock)

### Soundness Canaries
These should always be **type errors**:
```
-- Mixed types through same channel
let ch = Channel.new in Channel.send ch 42; Channel.send ch true

-- Via spawn
let ch = Channel.new in 
spawn (fun () -> Channel.send ch 42);
Channel.send ch true

-- Via function
let f ch = Channel.send ch true
let ch = Channel.new in Channel.send ch 42; f ch
```

## What NOT To Do

When working on this codebase, avoid these anti-patterns:

1. **Don't make channels async/buffered.** Synchronous rendezvous is intentional.

2. **Don't use parser generator libraries.** Hand-written parser gives better errors and was chosen after Chumsky failed.

3. **Don't add `receive` syntax (Erlang-style).** We use CML channels, not mailboxes.

4. **Don't add implicit parallelism.** Single-threaded cooperative scheduler is intentional.

5. **Don't generalize non-values.** Value restriction is load-bearing for soundness.

6. **Don't add multi-parameter typeclasses or HKT.** Single-parameter typeclasses are sufficient for now.

## Future Directions

Next phases of development (see ROADMAP.md for details):

- **Phase 2:** Error infrastructure - line:column tracking, source context in errors
- **Phase 3:** Module system - explicit `module` keyword, imports/exports
- **Phase 4:** REPL & tooling - history, multi-line, :load command
- **Phase 5:** Bytecode compiler, tail call optimization
- **Phase 6:** Real backend (BEAM/WASM/native)

## References

- *Concurrent ML* (Reppy) - channel semantics
- *Types and Programming Languages* (Pierce) - HM inference
- *Crafting Interpreters* (Nystrom) - interpreter patterns
- Standard ML value restriction - preventing unsound polymorphism
