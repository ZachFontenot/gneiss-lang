# Gneiss Development Roadmap

## Completed Work

### Core Language ✓
- Hindley-Milner type inference with let-polymorphism
- ADTs (algebraic data types) with pattern matching
- Local recursive functions (`let f x = ... f ... in body`)
- Value restriction for sound polymorphism

### Fiber-Based Concurrency ✓
- **Fiber Effects**: All blocking operations produce `FiberEffect` values
- **Unified Scheduler**: Handles Fork, Join, Yield, Send, Recv, Select uniformly
- **Fiber API**: `Fiber.spawn`, `Fiber.join`, `Fiber.yield`
- **Synchronous Channels**: Rendezvous semantics with typed communication
- **Select**: Multi-arm select over multiple channels
- **Deadlock Detection**: Reports when all fibers are blocked

### Delimited Continuations ✓
- **Reset/Shift**: Full implementation of delimited continuations
- **Multi-invocation**: Captured continuations callable multiple times
- **Foundation for Fibers**: FiberBoundary acts as implicit prompt

### Typeclasses ✓
- **Trait Declarations**: `trait Show a = val show : a -> String end`
- **Instance Declarations**: Basic and constrained instances
- **Dictionary Passing**: Runtime method dispatch
- **Instance Resolution**: Constraint propagation and overlap detection

---

## Dogfooding Goal: Web Server

**Target:** Build a simple but functional web server in Gneiss.

This drives language development by identifying what's actually needed for real programs.

```gneiss
-- Target API
let handler request =
    match request.path with
    | "/" -> Response.html 200 "<h1>Hello</h1>"
    | "/api/users" -> Response.json 200 (User.all () |> Json.encode)
    | _ -> Response.text 404 "Not found"

let main () =
    Server.listen 8080 handler
```

### Required Features (Priority Order)

| Priority | Feature | Why |
|----------|---------|-----|
| P1 | Module System | Can't build real projects without imports |
| P1 | Record Types | Request/Response/Config structs |
| P1 | Standard Library | String ops, Option, Result, Map |
| P2 | I/O Primitives | File handles, sockets, read/write |
| P2 | Bytes Type | Binary protocol parsing |
| P2 | Async I/O | Hook scheduler into epoll/kqueue |
| P3 | String Interpolation | Response generation, logging |
| P3 | JSON | Data interchange |

### Type System Opportunities

- **Effect tracking** - Distinguish pure functions from I/O
- **Resource types** - Ensure handles are closed (linear/affine)
- **Validated strings** - URLs, HTML-escaped text
- **Protocol state machines** - Type-safe HTTP parsing

---

## Next Up: Module System

**Goal:** Multi-file projects with imports and exports.

```gneiss
-- File: List.gn
module List

let map f xs = ...
let filter p xs = ...

-- File: Main.gn
module Main

import List
import List (map, filter)  -- selective
import List as L           -- qualified

let main () = List.map double [1, 2, 3]
```

### Tasks
- [ ] Tokens: `module`, `import`, `export`, `as`
- [ ] AST nodes for module/import declarations
- [ ] Module name resolution (file → module)
- [ ] Import resolution and name binding
- [ ] Circular dependency detection
- [ ] Visibility controls (public/private)

---

## Then: Record Types

**Goal:** Named product types with field access.

```gneiss
type Request = {
    method : String,
    path : String,
    headers : List (String, String),
    body : Bytes
}

let handler req =
    match req.method with
    | "GET" -> handle_get req.path
    | "POST" -> handle_post req.body
    | _ -> error "unsupported"
```

### Design Questions
- Structural vs nominal typing?
- Row polymorphism for extensible records?
- Field access syntax: `req.path` vs `path req`?
- Record update syntax?

---

## Future: I/O and Runtime

### I/O Primitives
- File operations (open, read, write, close)
- Socket operations (listen, accept, connect)
- Integration with Result type for errors

### Async I/O Integration
- Hook fiber scheduler into system event loop
- `FiberEffect::IO` for non-blocking operations
- epoll (Linux), kqueue (macOS), io_uring (modern Linux)

### Standard Library
- Core: Option, Result, List, Map, Set
- String: split, join, trim, format
- Bytes: slicing, parsing combinators
- JSON: encode/decode with typeclass derivation

---

## Performance (When Needed)

### Bytecode Compiler
- Design instruction set
- Compile AST → bytecode
- Stack-based VM

### Tail Call Optimization
- Detect tail position
- Critical for recursive fiber loops

### Scheduler Improvements
- Work-stealing (if multi-threaded)
- Fiber priorities
- Fairness guarantees

---

## Long-Term Goals

### REPL Improvements
- History and line editing (rustyline or similar)
- Multi-line input mode
- `:load` command to load .gn files
- `:type` for expression inspection
- `:env` to show current bindings
- Hot reloading of modules

### Full Compiler
- Move from tree-walking interpreter to compiled output
- Bytecode VM as intermediate step
- Native compilation via Cranelift or LLVM
- WebAssembly target for browser/edge deployment

### Advanced Type System
- **Effect tracking** - Static guarantees about I/O, state, exceptions
- **Resource types** - Linear/affine types for safe resource management
- **Refinement types** - Dependent-ish types for tighter constraints
- **Row polymorphism** - Extensible records without boilerplate

### Perceus Reference Counting (Long Shot)
Implement the Perceus algorithm for functional-but-in-place (FBIP) semantics:
- Precise reference counting with reuse analysis
- In-place mutation of uniquely-owned data
- No GC pauses, predictable performance
- Enables functional style with imperative efficiency

This would make Gneiss competitive for systems programming while keeping the pure functional model.

---

## Compilation Targets

| Target | Pros | Cons |
|--------|------|------|
| Native (Cranelift) | Best performance, full control | Must build GC/RC, scheduler |
| WebAssembly | Browser deployment, growing ecosystem | Concurrency model challenges |

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Concurrency | Fibers + delimited continuations | Clean foundation, single mechanism |
| Channels | Synchronous rendezvous | Simple semantics, forces explicit sync |
| Type inference | Hindley-Milner | Well-understood, sufficient |
| Typeclasses | Dictionary passing | Works with separate compilation |
| Continuations | shift/reset | Composable, typed, underlies fibers |
| Parser | Handwritten recursive descent | Control over error messages |
| Scheduler | Single-threaded cooperative | Simple for v0.1, parallelism later |
| Syntax | OCaml-inspired | Familiar to ML users |
| Development | Web server dogfooding | Drives real feature needs |

---

## References

**Continuations & Effects:**
- Filinski - "Representing Monads" (shift/reset)
- Kiselyov - delimcc library and papers
- Dolan et al - "Concurrent System Programming with Effect Handlers"

**Type Systems:**
- Pierce - "Types and Programming Languages"
- "Typing Haskell in Haskell" - typeclass implementation
- Leijen - "Extensible Records with Scoped Labels"

**Implementation:**
- Nystrom - "Crafting Interpreters"
- Appel - "Compiling with Continuations"
- SPJ - "The Implementation of Functional Programming Languages"
