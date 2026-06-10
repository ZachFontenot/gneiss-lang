# Gneiss Development Roadmap

*Updated 2026-06-10 — baseline reset after scrapping algebraic effects and the C backend.*

## Where We Are

The tree-walking CPS interpreter is the language. Commit `a6f33a4` removed
algebraic effects, user-facing delimited continuations, and the C codegen
pipeline after both failed to stabilize. The commits that followed hardened
what remains: value restriction enforcement, per-binding predicate discharge,
constructor pattern well-formedness, and Maranget exhaustiveness checking.
The full test suite (~600 tests) is green. This is the baseline we grow from.

**Strategy: harden the floor, then grow.** New semantics (effects, native
compilation, mutable state) only return as fresh designs on top of a stable,
well-tested core — see "Deliberately Not Doing" below and the "Don't propose"
section of CLAUDE.md.

---

## Completed Work

### Core Language ✓
- Hindley-Milner type inference with let-polymorphism
- ADTs (algebraic data types) with pattern matching
- Local recursive functions (`let f x = ... f ... in body`)
- Value restriction for sound polymorphism (incl. constructor/record values)
- Pattern-match exhaustiveness checking (Maranget usefulness algorithm)
- Typed AST elaboration with resolved types and dictionary parameters

### Fiber-Based Concurrency ✓
- **Fiber Effects**: All blocking operations produce `FiberEffect` values
  (internal scheduler mechanism — unrelated to the scrapped user-facing effects)
- **Unified Scheduler**: Handles Fork, Join, Yield, Send, Recv, Select uniformly
- **Fiber API**: `Fiber.spawn`, `Fiber.join`, `Fiber.yield`
- **Synchronous Channels**: Rendezvous semantics with typed communication
- **Select**: Multi-arm select over multiple channels
- **Deadlock Detection**: Reports when all fibers are blocked
- **Tail Calls**: The CPS interpreter is inherently tail-call optimized;
  verified by `tests/tail_call_optimization.rs` (5M-iteration loops in
  constant space)

### Typeclasses ✓
- **Trait Declarations**: `trait Show a = val show : a -> String end`
- **Instance Declarations**: Basic and constrained instances
- **Dictionary Passing**: Runtime method dispatch
- **Instance Resolution**: Constraint propagation and overlap detection
- **Predicate Discharge**: Checked after each top-level binding

### Module System ✓
- **Module Declarations**: `module List`
- **Import Statements**: `import List`, `import List as L`, `import List (foo, bar)`
- **Module Resolution**: File discovery and path mapping
- **Dependency Graph**: Topological sort, circular dependency detection
- **Multi-Module Type Checking**: Shared TypeEnv across modules

### Record Types ✓
- **Type Declarations**: `type Person = { name : String, age : Int }`
- **Record Literals**: `Person { name = "Alice", age = 30 }`
- **Field Access**: `person.name`, `person.age`
- **Record Update**: `{ person with age = 31 }`

### I/O and Async ✓
- **File Operations**: `file_open`, `file_read_line`, `file_read_all`, `file_write`, `file_close`
- **TCP Sockets**: `tcp_connect`, `tcp_listen`, `tcp_accept`, `tcp_send`, `tcp_recv`, `tcp_close`
- **Bytes Type**: Binary data type for protocol parsing
- **Async Integration**: Non-blocking I/O with mio event loop + blocking pool
- **Handle Registry**: Resource management across the Rust/Gneiss boundary

### Web Server Dogfooding ✓
The original dogfooding goal — a functional web server in Gneiss — was reached
(bd epic `gneiss-lang-aeb`): `examples/hello_server.gn`, `routing_server.gn`,
`json_api.gn`, `rest_api.gn`, backed by `stdlib/{http,json,router,server,
request,response,html}.gn`. It surfaced the resource/concurrency questions
that now drive the next design phase.

---

## Near Term: Harden the Baseline

Make "baseline" mean a defended core, not a snapshot that happened to be green.

1. **Fix prelude type shadowing** (`gneiss-lang-1up6`) — user `type`
   declarations can silently replace prelude types and break prelude
   instances. Likely fix: forbid redefining an in-scope type name.
2. **Fix record field access in mutual recursion** (`gneiss-lang-51e`) —
   needs deferred `HasField` constraints instead of erroring on unresolved
   type variables. Root-cause analysis is in the issue.
3. **Examples smoke test** — run every `examples/*.gn` in CI so examples and
   prelude can't silently rot.
4. **Property tests** already filed: exhaustiveness (`3ik`), generalization
   levels (`0e9`), channel type safety (`g5a`), unification idempotence (`4ya`).
5. **Runtime error spans** (`vhf`) — EvalError should point at source.

Rejection tests (`tests/type_rejection.rs`) remain the canonical soundness
canaries; every fix above lands with them. See `docs/TESTING.md`.

---

## Next Design Conversation: Concurrency & Resources

Tracked as epic `gneiss-lang-0ahm`. **No design is committed yet.** The open
questions, in rough order of pain:

1. **Resource lifecycle** — file/socket handles can leak; no `with_resource`
   or close-on-scope-exit story. Related: linear/affine resource types (`s5k`).
2. **Fiber lifetime structure** — Go-style (main controls everything) vs
   Erlang-style (independent processes). Research notes in `b2n`.
3. **Channel backpressure** — pure rendezvous channels are brittle under
   load; consider optional bounded buffers.
4. **HTTP server drops connections at 200+ concurrent** (`wwk`) — partly a
   symptom of 1 and 3; scheduler itself was ruled out.

---

## Ongoing Tracks

- **REPL/script mode split** (epic `46s6`) and REPL usability (epic `zex`):
  rustyline-style editing, history, multi-line input, `:load`, `:env`.
- **Stdlib expansion** (epic `nrnw`): Map/Set/Queue, string ops, path/process,
  time. Expand incrementally as real programs demand.
- **Developer tooling** (epic `6ynh`): LSP, formatter, linter — deferred until
  the core is stable.

---

## Deliberately Not Doing

Scrapped in `a6f33a4` (2026-04); none return without a fresh design document
against a frozen, well-tested semantics:

- **Algebraic effects** (`effect`/`perform`/`handle`, effect rows, handler stacks)
- **User-facing shift/reset** and delimited continuations
- **C code generation** (mono → ANF → CPS → closure-conv → flat IR → C);
  the Perceus epic `gsvz` is closed as deferred indefinitely
- **Mutable references** (`ref`, `!r`, `r := v`) — motivation was effect
  handler state threading; closed with the effects scrap (`zh5`)
- **Row polymorphism**

If/when native compilation is revisited, the prerequisite is exactly what the
near-term work produces: stable semantics pinned by property and rejection
tests.

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Concurrency | Fibers + channels | Clean foundation, single mechanism |
| Channels | Synchronous rendezvous | Simple semantics, forces explicit sync |
| Type inference | Hindley-Milner | Well-understood, sufficient |
| Typeclasses | Dictionary passing | Works with separate compilation |
| Parser | Handwritten recursive descent | Control over error messages |
| Scheduler | Single-threaded cooperative | Simple for v0.1, parallelism later |
| Syntax | OCaml-inspired | Familiar to ML users |
| Development | Web server dogfooding | Drives real feature needs |
| Effects & C backend | Scrapped (a6f33a4, 2026-04) | Semantics never stabilized; interpreter-first instead |
| Baseline reset | Merged to main (2026-06) | Harden the floor before growing the language |

---

## References

**Type Systems:**
- Pierce - "Types and Programming Languages"
- "Typing Haskell in Haskell" - typeclass implementation

**Implementation:**
- Nystrom - "Crafting Interpreters"
- Appel - "Compiling with Continuations"
- SPJ - "The Implementation of Functional Programming Languages"
