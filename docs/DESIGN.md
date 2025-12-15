# Gneiss Design Document

This document captures the design rationale, key decisions, and invariants for the Gneiss programming language.

## Overview

Gneiss is a statically-typed functional programming language with:
- **ML-family syntax** (inspired by OCaml)
- **Hindley-Milner type inference** with let-polymorphism
- **Fiber-based concurrency** built on delimited continuations
- **Typeclasses** with dictionary passing

The goal is a language where concurrent programs are safe by construction: no data races, no shared mutable state, typed channels that prevent mixed-type communication.

## Core Design Decisions

### 1. Concurrency Model: Fibers + Delimited Continuations

**Decision:** Lightweight fibers with synchronous channels, implemented via delimited continuations.

**Rationale:**
- Delimited continuations provide a clean foundation for suspendable computation
- Fibers are first-class values that can be spawned, joined, and yielded
- Synchronous (rendezvous) channels force explicit synchronization points
- Typed channels catch communication errors at compile time
- Single unified mechanism for all blocking operations

**Key primitives:**
```gneiss
-- Fiber operations
let fiber = Fiber.spawn (fun () -> computation)
let result = Fiber.join fiber
let _ = Fiber.yield ()

-- Channel operations
let ch = Channel.new
Channel.send ch value
let x = Channel.recv ch

-- Select over multiple channels
select
| x <- ch1 -> handle_x x
| y <- ch2 -> handle_y y
end
```

**How it works:**
1. All blocking operations produce `FiberEffect` values
2. Effects bubble up to a `FiberBoundary` frame (implicit delimiter)
3. The scheduler pattern-matches on effects and handles them uniformly
4. Continuations are captured and stored for later resumption

**Example:**
```gneiss
let main () =
  let ch = Channel.new in
  let worker = Fiber.spawn (fun () ->
    Channel.send ch 42
  ) in
  let result = Channel.recv ch in
  let _ = Fiber.join worker in
  result
```

### 2. Type System: Hindley-Milner + Typeclasses

**Decision:** HM type inference with single-parameter typeclasses.

**What we have:**
- Parametric polymorphism (`let id x = x` works at any type)
- Let-polymorphism (polymorphic let bindings)
- ADTs (algebraic data types) with pattern matching
- Type inference (almost no annotations needed)
- Typeclasses with dictionary passing

**Typeclass syntax:**
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

**Constraints:**
- Single-parameter typeclasses only (no multi-param or functional dependencies)
- Overlapping instances are detected and rejected
- Orphan instances allowed (no coherence checking yet)

### 3. Value Restriction for Soundness

**Decision:** Apply ML-style value restriction to prevent unsound polymorphic effects.

**The problem:**
```gneiss
let ch = Channel.new        -- What type? Channel 'a (polymorphic)?
let _ = Fiber.spawn (fun () -> Channel.send ch 42)
let _ = Fiber.spawn (fun () -> Channel.send ch "hello")  -- Unsound!
```

**The solution:** Only generalize *syntactic values* in let bindings.

Syntactic values are: literals, variables, lambdas, constructors applied to values, tuples/lists of values.

`Channel.new` is NOT a syntactic value (it's effectful), so:
```gneiss
let ch = Channel.new  -- NOT generalized, stays monomorphic
```

The channel's type is fixed at its first use, catching mixed-type sends at compile time.

### 4. Interpreter Architecture: Effect-Based CPS

**Decision:** CPS interpreter where all blocking operations produce effects handled by a unified scheduler.

**Key types:**
```rust
enum State {
    Eval { expr, env, cont },   // Evaluate an expression
    Apply { value, cont },       // Return a value to continuation
}

enum Frame {
    AppFunc { arg, env },        // Evaluating func, arg next
    AppArg { func },             // Have func, evaluating arg
    FiberBoundary,               // Implicit delimiter for fiber effects
    FiberRecv,                   // Waiting for channel value
    // ... etc
}

enum FiberEffect {
    Done(Value),                 // Fiber completed
    Fork { thunk, cont },        // Spawn new fiber
    Yield { cont },              // Yield to scheduler
    Send { channel, value, cont },
    Recv { channel, cont },
    Join { fiber_id, cont },
    Select { arms, cont },
}
```

**Effect flow:**
1. Fiber operation (e.g., `Channel.recv ch`) pushes appropriate frame
2. Frame captures continuation up to `FiberBoundary`
3. `FiberEffect` value bubbles up to boundary
4. Boundary returns `StepResult::Done(Value::FiberEffect(effect))`
5. Scheduler handles effect, stores continuation, manages ready queue

### 5. Scheduler: Cooperative, Single-Threaded

**Decision:** Single-threaded cooperative scheduler.

**Rationale:**
- Simpler to implement and debug
- No need for synchronization primitives
- Sufficient for demonstrating semantics
- Can add parallelism later without changing the model

**Behavior:**
- Round-robin ready queue
- Fibers yield at channel operations, explicit yield, or join
- Deadlock detected when all fibers blocked and ready queue empty

### 6. Delimited Continuations: shift/reset

**Decision:** First-class delimited continuations as a user-facing feature.

**Syntax:**
```gneiss
let result = reset (
  1 + shift (fun k -> k (k 10))
)
-- result = 1 + (1 + 10) = 12
```

**Semantics:**
- `reset e` evaluates `e` with a prompt (delimiter)
- `shift (fun k -> body)` captures continuation up to nearest reset
- Captured continuation `k` can be called zero, once, or multiple times
- Fiber effects use the same underlying mechanism (FiberBoundary as implicit prompt)

### 7. Channels: Synchronous Rendezvous

**Decision:** Channels are strictly synchronous with no buffering.

**Rationale:**
- Simpler semantics
- Forces explicit synchronization
- Predictable behavior (send blocks until recv ready, and vice versa)

**NOT supported:**
```gneiss
-- NO buffered channels
let ch = Channel.new_buffered 10  -- doesn't exist

-- NO non-blocking operations
let result = Channel.try_send ch x  -- doesn't exist
```

## Invariants

These properties should always hold. Violating them is a bug.

### Type Safety Invariants

1. **Well-typed programs don't get runtime type errors.** If it passes type inference, evaluation won't produce `TypeError`.

2. **Channel type consistency.** Every value sent through a channel `ch : Channel T` has type `T`.

3. **Value restriction prevents polymorphic effectful bindings.** `Channel.new`, `Fiber.spawn`, etc. are never generalized.

4. **Occurs check prevents infinite types.** `fun x -> x x` is rejected.

### Concurrency Invariants

5. **Rendezvous semantics.** Send and receive both block until a partner is available.

6. **No data races.** Fibers share nothing; all communication is through channels.

7. **Deadlock is detectable.** When all fibers are blocked and ready queue is empty, we report deadlock.

8. **Fiber isolation.** A fiber cannot access another fiber's local variables or continuation.

9. **Effect uniformity.** All blocking operations go through the same effect mechanism.

## File Structure

```
src/
  ast.rs      -- AST node definitions with Span for error reporting
  lexer.rs    -- Hand-written tokenizer
  parser.rs   -- Recursive descent parser
  types.rs    -- Internal type representation (Type, TypeVar, Scheme)
  infer.rs    -- Hindley-Milner inference with typeclasses
  eval.rs     -- CPS interpreter with fiber effects
  runtime.rs  -- Fiber scheduler, channels, ready queue
  main.rs     -- REPL and file execution
  lib.rs      -- Public exports

tests/
  fiber_effects.rs  -- Fiber and channel tests
  continuations.rs  -- Delimited continuation tests
  typeclasses.rs    -- Typeclass tests
  properties.rs     -- Property-based soundness tests

examples/
  *.gn        -- Example programs

docs/
  SYNTAX.md   -- Full syntax reference
  ROADMAP.md  -- Development roadmap
  DESIGN.md   -- This file
```

## Testing Strategy

### Unit Tests
- Lexer: token sequences
- Parser: AST structure
- Inference: type of expressions, typeclass resolution
- Eval: value of expressions
- Runtime: scheduler behavior, effect handling

### Fiber/Concurrency Tests
- Spawn and join
- Channel send/recv
- Select over multiple channels
- Deadlock detection
- Interleaving (programs that need scheduler cooperation)

### Soundness Canaries
These should always be **type errors**:
```gneiss
-- Mixed types through same channel
let ch = Channel.new in Channel.send ch 42; Channel.send ch true

-- Via spawn
let ch = Channel.new in
Fiber.spawn (fun () -> Channel.send ch 42);
Channel.send ch true
```

## What NOT To Do

When working on this codebase, avoid these anti-patterns:

1. **Don't make channels async/buffered.** Synchronous rendezvous is intentional.

2. **Don't use parser generator libraries.** Hand-written parser gives better errors.

3. **Don't add implicit parallelism.** Single-threaded cooperative scheduler is intentional for now.

4. **Don't generalize non-values.** Value restriction is load-bearing for soundness.

5. **Don't bypass the effect system.** All blocking operations should produce FiberEffects.

6. **Don't add multiple code paths for blocking.** The unified fiber effect system replaced the old dual-path architecture.

## Future Directions

Development is driven by a **dogfooding goal**: build a web server in Gneiss.

This goal identifies what language features are actually needed:
- **Module system** - Can't build real projects without imports
- **Record types** - Structured data (Request, Response, Config)
- **Standard library** - String ops, data structures, Result/Option
- **I/O primitives** - Files, sockets, read/write
- **Async I/O integration** - Hook scheduler into epoll/kqueue/io_uring

Type system opportunities:
- **Effect tracking** - Distinguish pure functions from I/O
- **Resource types** - Ensure handles are closed (linear/affine types)

See ROADMAP.md for detailed plans.

## References

- *Delimcc* - Delimited continuation implementations
- *Types and Programming Languages* (Pierce) - HM inference
- *Typing Haskell in Haskell* - Typeclass implementation
- *Crafting Interpreters* (Nystrom) - Interpreter patterns
- Standard ML value restriction - Preventing unsound polymorphism
