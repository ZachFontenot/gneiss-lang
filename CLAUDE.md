# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads)
for issue tracking. Use `bd` commands instead of markdown TODOs.
READ AGENTS.md for workflow details.
READ docs/DESIGN.md for general ideas of the project
READ docs/ROADMAP.md for the overarching plans

## Project Overview

Gneiss is a statically-typed functional programming language with actors and channels, implemented in Rust. It features Hindley-Milner type inference and a tree-walking interpreter.

## Build & Test Commands

```bash
cargo build          # Build the project
cargo test           # Run all tests
cargo test <name>    # Run a specific test (e.g., cargo test test_lambda)
cargo run            # Start the REPL
cargo run -- <file>  # Execute a .gneiss source file
```

## Architecture

The compiler pipeline flows: **Source � Lexer � Parser � Type Inference � Interpreter**

### Module Structure

- `lexer.rs` - Hand-written tokenizer producing `SpannedToken` with source locations
- `parser.rs` - Recursive descent parser producing an AST
- `ast.rs` - AST node definitions with span information for error reporting
- `types.rs` - Internal type representation (`Type`, `TypeVar`, `Scheme`, `TypeEnv`)
- `infer.rs` - Hindley-Milner type inference with let-polymorphism
- `eval.rs` - CPS interpreter with defunctionalized continuations
- `runtime.rs` - Lightweight process scheduler and synchronous channels
- `main.rs` - REPL and file execution entry point

### Key Design Patterns

**AST Nodes**: All AST nodes are wrapped in `Spanned<T>` providing source location tracking. Expressions use `Rc<Expr>` for sharing in closures.

**Type Variables**: Uses mutable reference cells (`Rc<RefCell<TypeVar>>`) with a union-find style linking for unification. Type variables track levels for let-polymorphism.

**Environments**: Both type environments (`TypeEnv`) and runtime environments (`Env`) use parent-pointer chains for lexical scoping.

**Concurrency**: CSP-style with synchronous (rendezvous) channels. Processes are spawned with `spawn (fun () -> ...)` and communicate via `Channel.new`, `Channel.send`, `Channel.recv`.

## Language Syntax

```
-- Line comment
{- Block comment -}

-- Let bindings
let x = 42
let add x y = x + y

-- Lambdas
fun x y -> x + y

-- Pattern matching
match expr with
| Some x -> x
| None -> 0

-- Type declarations
type Option a = | Some a | None

-- Operators: + - * / % == != < > <= >= && || :: ++ |> <| >> <<
-- Pipe: x |> f (equivalent to f x)
```

## Built-in Functions

- `print : a -> ()` - Print any value
- `int_to_string : Int -> String`
- `string_length : String -> Int`
