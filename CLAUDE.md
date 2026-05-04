# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking, invoked via `mcp__beads__*` MCP tools (not the CLI). Use them instead of markdown TODOs.

- READ AGENTS.md for the bd workflow
- READ docs/DESIGN.md for general ideas of the project
- READ docs/ROADMAP.md for the overarching plans
- READ docs/TESTING.md for the testing philosophy

## Project Overview

Gneiss is a statically-typed functional language with HM type inference, ADTs, records, typeclasses (dictionary passing), modules, fibers, and synchronous (rendezvous) channels. Implemented in Rust as a tree-walking CPS interpreter.

## Build & Test Commands

```bash
cargo build          # Build the project
cargo test           # Run all tests
cargo test <name>    # Run a specific test (e.g., cargo test test_lambda)
cargo run            # Start the REPL
cargo run -- <file>  # Execute a .gn source file
```

## Architecture

Pipeline: **Source → Lexer → Parser → Type Inference → Elaboration → Interpreter**

### Module structure (`src/`)

- `lexer.rs` — hand-written tokenizer producing `SpannedToken` with source spans
- `parser/` — recursive-descent parser (mod, types, pattern, record, cursor, error, combinators)
- `ast.rs` — AST node definitions; nodes wrapped in `Spanned<T>`
- `types.rs` — `Type`, `Scheme`, `TypeEnv`, `Pred`, plus the `UnionFind` for type-variable unification
- `infer.rs` — Hindley-Milner inference with let-polymorphism, value restriction, predicate discharge, exhaustiveness hookup
- `exhaustiveness.rs` — Maranget usefulness algorithm for pattern-match coverage
- `elaborate.rs` — produces a typed AST (`typed_ast.rs`) with resolved types and dictionary parameters
- `eval.rs` — CPS interpreter with defunctionalized continuations; runtime fiber dispatch
- `runtime.rs` — cooperative scheduler and synchronous channels
- `io_reactor.rs`, `blocking_pool.rs` — non-blocking I/O via mio + a worker pool for blocking ops
- `module.rs` — module resolution and dependency graph
- `prelude.rs`, `operators.rs` — prelude loader, operator-precedence parsing
- `errors.rs` — error formatting (snippets, suggestions, distance)
- `test_support.rs` — `parse_expr`, `typecheck_expr`, `assert_type`, `run_program_ok`, etc.
- `main.rs` — REPL and file execution entry point

### Key design patterns

- **Spanned AST.** Every AST node is `Spanned<T>` so errors point at source. Expressions use `Rc<Expr>` for sharing in closures.
- **Union-Find for type vars.** `UnionFind` uses a `Cell`-backed parent vec with path compression and union-by-rank. Type variables track levels for let-polymorphism; `update_levels` lowers vars when generalize doesn't fire.
- **Environments.** `TypeEnv` and runtime `Env` use parent-pointer chains for lexical scoping.
- **Concurrency.** CSP-style. `Fiber.spawn (fun () -> ...)` schedules a fiber; `Channel.new`, `Channel.send`, `Channel.recv`, and `select` operate on rendezvous channels. Bare `spawn` is a back-compat builtin.

## Language syntax

```
-- Line comment
{- Block comment -}

let x = 42
let add x y = x + y
let rec fact n = if n == 0 then 1 else n * fact (n - 1)

fun x y -> x + y                          -- lambda

match expr with
| Some x -> x
| None -> 0
end

type Option a = | Some a | None
type Person = { name : String, age : Int }

trait Show a = val show : a -> String end
impl Show for Int = let show n = int_to_string n end

-- Operators: + - * / % == != < > <= >= && || :: ++ |> <| >> <<
-- Pipe: x |> f  is  f x
```

## Built-in functions (selected)

- `print : Show a => a -> ()` — print any showable value
- `int_to_string : Int -> String`, `string_length : String -> Int`
- `Channel.new : () -> Channel a`, `Channel.send`, `Channel.recv`
- `Fiber.spawn : (() -> a) -> Fiber a`, `Fiber.join`, `Fiber.yield`

Full prelude: `stdlib/prelude.gn`. Other stdlib modules: `stdlib/{html,http,json,request,response,router,server}.gn`.

## Don't propose

These were intentionally scrapped (commit a6f33a4) and won't come back without a fresh design:
- Algebraic effects (`effect`/`perform`/`handle`/`return` keywords, `Row`, `EffectEnv`, handler stacks)
- Shift/reset and delimited continuations (user-facing)
- C code generation (the mono → ANF → CPS → closure-conv → flat IR → C pipeline)
- Mutable references (`ref`, `!r`, `r := v`) — never implemented; tracked in bd as `gneiss-lang-zh5` but the direction is unclear
- Row polymorphism

## Testing

All code changes follow `docs/TESTING.md`:

1. **Test the process, not just the output.** Verify AST structure and inferred types.
2. **Layered.** Parser → Rejection → Type → Runtime → Output.
3. **Rejection tests are mandatory.** `tests/type_rejection.rs` is the canonical home for soundness canaries (value restriction, predicate discharge, constructor arity, val-without-let, exhaustiveness, etc.).
4. **Use `test_support.rs`.** Prefer `parse_expr()`, `typecheck_expr()`, `assert_type()`, `assert_type_error()` over `run_program_ok()`.

```rust
assert_type("fun x -> x + 1", "Int -> Int");
assert_type_error("1 + true");
assert_eval_int("1 + 2", 3);  // acceptable after semantic tests exist
```
