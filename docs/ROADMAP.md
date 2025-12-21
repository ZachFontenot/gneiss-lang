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
- **Structural Typing**: Records unify based on field types

### I/O and Async ✓
- **File Operations**: `file_open`, `file_read_line`, `file_read_all`, `file_write`, `file_close`
- **TCP Sockets**: `tcp_connect`, `tcp_listen`, `tcp_accept`, `tcp_send`, `tcp_recv`, `tcp_close`
- **Bytes Type**: Binary data type for protocol parsing
- **IoError Type**: Error handling for I/O operations
- **Async Integration**: Non-blocking I/O with mio event loop
- **Handle Registry**: Safe resource management across Rust/Gneiss boundary

### Standard Library (Partial) ✓
- **List Operations**: `List.map`, `List.filter`, `List.fold_left`, `List.length`, `List.concat`, etc.
- **String Operations**: `String.split`, `String.concat`, `string_to_chars`, `chars_to_string`, etc.
- **Option/Result**: Basic pattern matching support

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

| Priority | Feature | Status |
|----------|---------|--------|
| P1 | Module System | ✓ Complete |
| P1 | Record Types | ✓ Complete |
| P1 | Standard Library | Partial - need Map/Dict |
| P2 | I/O Primitives | ✓ Complete |
| P2 | Bytes Type | ✓ Complete |
| P2 | Async I/O | ✓ Complete |
| P3 | Map/Dict Type | **Next** - key-value storage for headers, routing |
| P3 | JSON | Needed for API responses |
| P3 | HTTP Parsing | Request/response parsing over TCP |
| P4 | String Interpolation | Nice-to-have for response generation |

### Type System Opportunities

- **Effect tracking** - Distinguish pure functions from I/O
- **Resource types** - Ensure handles are closed (linear/affine)
- **Validated strings** - URLs, HTML-escaped text
- **Protocol state machines** - Type-safe HTTP parsing

---

## Next Up: Map/Dict Type

**Goal:** Key-value data structure for headers, routing tables, caches.

```gneiss
-- Create and use maps
let headers = Map.empty ()
    |> Map.insert "Content-Type" "text/html"
    |> Map.insert "X-Request-Id" "abc123"

let content_type = Map.get "Content-Type" headers  -- Some "text/html"

-- Use in HTTP handling
type Request = {
    method : String,
    path : String,
    headers : Map String String,
    body : String
}
```

### Implementation Options
1. **Hash Map** - O(1) average, needs hash function
2. **Tree Map** - O(log n), needs Ord typeclass
3. **Association List** - Simple, O(n) lookup

### Tasks
- [ ] Decide on implementation (tree map for now?)
- [ ] Add Map type to type system
- [ ] Implement core operations: empty, insert, get, remove, contains
- [ ] Add iteration: keys, values, entries, fold
- [ ] Consider Ord typeclass for tree map ordering

---

## Then: JSON Support

**Goal:** Parse and serialize JSON for API data interchange.

```gneiss
-- Parse JSON
let data = Json.parse "{\"name\": \"Alice\", \"age\": 30}"

-- Access fields
match data with
| JsonObject obj -> Map.get "name" obj
| _ -> None

-- Serialize
let response = Json.object [
    ("status", Json.string "ok"),
    ("count", Json.int 42)
]
print (Json.encode response)  -- {"status":"ok","count":42}
```

### Tasks
- [ ] Define Json ADT (JsonNull, JsonBool, JsonInt, JsonFloat, JsonString, JsonArray, JsonObject)
- [ ] Implement JSON parser
- [ ] Implement JSON encoder
- [ ] Consider typeclass-based encoding (ToJson, FromJson)

---

## Then: HTTP Protocol

**Goal:** Parse HTTP requests and generate responses over TCP.

```gneiss
-- Low-level: parse raw HTTP
let parse_request raw_bytes =
    let lines = String.split "\r\n" raw_bytes in
    let request_line = List.head lines in
    -- Parse "GET /path HTTP/1.1"
    ...

-- High-level: server abstraction
let handler request =
    Response.html 200 "<h1>Hello</h1>"

let main () =
    Server.listen 8080 handler
```

### Tasks
- [ ] HTTP request parser (method, path, headers, body)
- [ ] HTTP response builder
- [ ] Content-Type handling
- [ ] Chunked transfer encoding (optional)
- [ ] High-level Server.listen abstraction

---

## Remaining Standard Library Work

### Core Data Structures
- Map/Dict (priority - see above)
- Set (can build on Map)

### Bytes Operations
- Slicing and indexing
- Parsing combinators for binary protocols
- Conversion to/from String (UTF-8)

### Additional String Operations
- `String.trim`, `String.replace`
- `String.starts_with`, `String.ends_with`
- Format/interpolation (P4)

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
