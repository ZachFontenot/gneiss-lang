# Gneiss Future Work Roadmap

Strategic plan for evolving Gneiss toward a production-ready functional language with predictable performance, algebraic effects, and batteries-included stdlib.

*Created: 2025-12-27*

## Vision Summary

- **Safety + Performance**: Haskell/OCaml/Rust-level abstractions with predictable, GC-free performance via Perceus
- **Pure FP by default**: IO under algebraic effects, but ergonomic (implicit in REPL/run mode)
- **Batteries included**: Node.js/Go-level stdlib completeness
- **Koka-style compilation**: Perceus reference counting → C output

---

## Phase 1: IO Under Effects (Foundation)

**Goal**: Move from free side-effecting to IO-as-effect while maintaining ergonomics.

**Beads Epic**: `gneiss-lang-ve4m`

### 1.1 Type System Changes
- **File**: `src/infer.rs` (lines 3199-3208)
- Change `print` type signature from `Row::Empty` to `Row::Extend { effect: IO, ... }`
- Add IO effect to function signature propagation
- Same for `get_args`, debug output functions

### 1.2 Runtime Changes
- **File**: `src/eval.rs` (lines 3625-3628)
- Change `print` from immediate execution to `BuiltinResult::Effect(EffectStub::Io)`
- Add `IoOp::Print { value }` variant
- Console I/O joins file/network I/O pathway through scheduler

### 1.3 Implicit IO Environments
- **REPL mode**: Wrap each input in implicit IO handler (like Haskell's GHCi)
- **Run mode**: Wrap `main()` in implicit IO+Async handlers
- **Library code**: Must declare IO effect in signatures
- Result: Convenient scripting, type-safe libraries

### 1.4 Debug Escape Hatch
- Add `--debug` Cargo feature flag
- When enabled: `debug_print` builtin works without effect tracking
- Production builds: all IO tracked

### Files to modify:
- `src/infer.rs` - type signatures
- `src/eval.rs` - runtime behavior
- `src/main.rs` - implicit handlers for REPL/run
- `stdlib/effect/io.gn` - ensure complete

---

## Phase 2: TCO Architecture

**Goal**: Design tail-call optimization that works for both interpreter and future Perceus codegen.

**Beads Epic**: `gneiss-lang-5exh`

### 2.1 Detect Tail Position
- **File**: `src/eval.rs`
- Add `is_tail_position` flag to evaluation context
- Identify tail calls in: function bodies, if/match branches, let bodies
- Mark tail calls during AST traversal or evaluation

### 2.2 Interpreter TCO
- In tail position: reuse current `Env` and `Cont` stack space
- Replace `AppFunc` → `AppArg` → return chain with direct jump
- Critical for: recursive fiber loops, parser combinators, fold operations

### 2.3 TCO-Aware IR (for Phase 5)
- Design intermediate representation with explicit tail call annotation
- Will feed into Perceus transformation and C codegen
- Tail calls → direct jumps in C output

### Design consideration:
The CPS interpreter already has continuation structure. TCO means not pushing continuation frames for tail calls - return directly to caller's continuation.

---

## Phase 3: REPL/Script Mode Split

**Goal**: Clean separation between interactive REPL and script execution.

**Beads Epic**: `gneiss-lang-46s6`

### 3.1 Execution Modes
```rust
enum ExecutionMode {
    Repl,      // Interactive, implicit IO, accumulated env
    Script,    // File execution, implicit IO+Async, fresh per run
}
```

### 3.2 REPL Mode Behavior
- Implicit IO handler wrapping each input
- Accumulated TypeEnv and global Env across inputs
- `:type`, `:load`, `:env` commands
- History and line editing (rustyline)
- Multi-line input support

### 3.3 Script/Run Mode Behavior
- `gneiss run <file>` or `gneiss <file>`
- Implicit IO+Async handlers at top level
- `main()` required (or last expression is result)
- Module system fully active
- Optimizations enabled
- Clean error output (no REPL noise)

### 3.4 Shared Infrastructure
- Same interpreter core (`eval.rs`)
- Same type checker (`infer.rs`)
- Mode flag changes implicit handler behavior

### Files to modify:
- `src/main.rs` - command parsing, mode dispatch
- `src/eval.rs` - add ExecutionMode to Interpreter
- New `src/repl.rs` - extract REPL logic

---

## Phase 4: Stdlib Expansion

**Goal**: Batteries-included stdlib comparable to Go/Node.js.

**Beads Epic**: `gneiss-lang-nrnw`

*Parallel track - expand incrementally alongside other phases.*

### 4.1 Core Data Structures
- **Map/Dict** (priority - already in roadmap)
- **Set** (build on Map)
- **Queue/Deque**
- **Priority Queue**

### 4.2 String/Text
- Full Unicode support
- Regex (basic pattern matching)
- StringBuilder for efficient concatenation
- Encoding/decoding (UTF-8, Base64)

### 4.3 I/O Ecosystem
- **Path** module - cross-platform path manipulation
- **Directory** operations - list, create, remove
- **Process** - spawn, exec, pipes
- **Environment** - env vars, working dir

### 4.4 Networking
- **HTTP client** - requests, responses, headers
- **URL parsing**
- **DNS resolution**
- HTTP server already exists in examples

### 4.5 Data Formats
- **JSON** (already partially done)
- **YAML/TOML** (config files)
- **CSV**

### 4.6 Time/Date
- Timestamps, durations
- Parsing and formatting
- Timezones (defer full tz database)

### Organization:
```
stdlib/
  core/       -- Option, Result, List, Tuple
  data/       -- Map, Set, Queue, Array
  text/       -- String, Regex, Encoding
  io/         -- File, Path, Directory, Process
  net/        -- Http, Tcp, Url
  format/     -- Json, Yaml, Csv
  time/       -- DateTime, Duration
  effect/     -- IO, State, Reader, Writer, etc.
```

---

## Phase 5: Perceus Codegen

**Goal**: Compile Gneiss to C via Perceus reference counting for predictable, GC-free performance.

**Beads Epic**: `gneiss-lang-gsvz`

### 5.1 Study Phase
- Deep dive into Koka papers and implementation
- Understand: Perceus algorithm, reuse analysis, drop specialization
- Key papers:
  - "Perceus: Garbage Free Reference Counting with Reuse"
  - "Reference Counting with Frame Limited Reuse"

### 5.2 IR Design
- Design typed intermediate representation
- Explicit allocations and drops
- Tail call annotations (from Phase 2)
- Effect handlers → control flow

### 5.3 Perceus Transformation
- Insert reference count operations
- Reuse analysis for in-place mutation of unique values
- Drop specialization to avoid redundant decrements

### 5.4 C Code Generation
- Generate readable C code (like Koka does)
- Runtime library: tagged unions, closures, RC primitives
- Effect handler → setjmp/longjmp or explicit state machine

### 5.5 Build Integration
- `gneiss compile <file>` → generates C
- Invoke C compiler (gcc/clang)
- Link with runtime library
- Produce native executable

### Milestones:
1. IR design and printer
2. Simple expressions compile (no effects)
3. Closures and allocations work
4. Reference counting correct
5. Reuse analysis optimization
6. Effect handlers compile
7. Full language support

---

## Phase 6: Test Audit

**Goal**: Clean, maintainable test suite with clear purpose for each test.

**Beads Epic**: `gneiss-lang-9urh`

*Parallel track - do alongside other work, clean as you go.*

### 6.1 Current State (~463 tests)
- 11 integration test files in `tests/`
- 159 inline unit tests in `src/`
- Property-based tests with proptest
- Snapshot tests for error messages

### 6.2 Audit Process
1. **Categorize each test**: unit, integration, property, stress, regression
2. **Identify redundancy**: overlapping coverage between files
3. **Assess stress tests**: Do all 18+ scheduler stress tests add value?
4. **Check legacy tests**: "Phase 7" comments suggest evolution - prune obsolete
5. **Document purpose**: Each test file gets header explaining what it covers

### 6.3 Target Organization
```
tests/
  unit/           -- Fast, isolated component tests
  integration/    -- Full pipeline tests
  property/       -- Proptest invariant checks
  regression/     -- Bug fix verification
  stress/         -- Performance/load tests (run separately)
  snapshots/      -- Error message verification
```

### 6.4 Test Guidelines
- Unit tests in `src/` modules stay (fast, focused)
- Integration tests move to appropriate category
- Property tests stay but review coverage overlap
- Add `#[ignore]` for slow stress tests, run in CI

---

## Phase 7: Developer Tooling (Deferred)

**Goal**: Full development environment support. Prioritize after core is stable.

**Beads Epic**: `gneiss-lang-6ynh`

### 7.1 Language Server (LSP)
- Go-to-definition
- Find references
- Autocomplete
- Inline type errors
- Hover for type info

### 7.2 Formatter
- Canonical formatting (like gofmt)
- `gneiss fmt` command
- Editor integration

### 7.3 Linter
- Unused bindings
- Shadowing warnings
- Effect leakage detection
- Style suggestions

### 7.4 Debugger
- Breakpoints
- Step evaluation
- Inspect environments
- Fiber/effect stack visualization

### Priority:
These are deferred until core language work (Phases 1-5) is solid.

---

## Implementation Priority

### Primary Track (sequential):
1. **Phase 1: IO Under Effects** - Foundation for pure FP
2. **Phase 2: TCO** - Needed for idiomatic FP, feeds into codegen
3. **Phase 3: REPL/Script Split** - Builds on Phase 1
4. **Phase 5: Perceus Codegen** - Major undertaking, after foundations solid
5. **Phase 7: Tooling** - After language stable

### Parallel Tracks:
- **Phase 4: Stdlib** - Expand incrementally alongside all phases
- **Phase 6: Test Audit** - Do alongside other work, clean as you go

---

## Dependency Graph

```
Phase 1 (IO) ─┬─→ Phase 2 (TCO) ──→ Phase 5 (Perceus) ──→ Phase 7 (Tooling)
              └─→ Phase 3 (REPL Split)

Phase 4 (Stdlib) ──→ (parallel, no dependencies)
Phase 6 (Test Audit) ──→ (parallel, no dependencies)
```

---

## Tracking

All phases are tracked as beads epics with dependencies enforced.

Use `bd ready` to see what's available to work on.
Use `bd show <id>` to see epic details and subtasks.
