# GitHub Copilot Instructions for Gneiss

## Project Overview

Gneiss is a statically-typed functional programming language with actors and channels, implemented in Rust. It features Hindley-Milner type inference and a tree-walking interpreter.

## Tech Stack

- **Language**: Rust (Edition 2021)
- **Dependencies**: thiserror for error handling
- **Testing**: Rust standard testing

## Issue Tracking with bd

**CRITICAL**: This project uses **bd (beads)** for ALL task tracking. Do NOT create markdown TODO lists.

### Essential Commands

```bash
# Find work
bd ready --json                    # Unblocked issues

# Create and manage
bd create "Title" -t bug|feature|task -p 0-4 --json
bd create "Subtask" --parent <epic-id> --json  # Hierarchical subtask
bd update <id> --status in_progress --json
bd close <id> --reason "Done" --json
```

### Workflow

1. **Check ready work**: `bd ready --json`
2. **Claim task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** `bd create "Found bug" -p 1 --deps discovered-from:<parent-id> --json`
5. **Complete**: `bd close <id> --reason "Done" --json`

## Build & Test Commands

```bash
cargo build          # Build the project
cargo test           # Run all tests
cargo test <name>    # Run a specific test
cargo run            # Start the REPL
cargo run -- <file>  # Execute a .gneiss source file
```

## Project Structure

```
gneiss-lang/
├── src/
│   ├── main.rs      # REPL and file execution
│   ├── lib.rs       # Public exports
│   ├── lexer.rs     # Tokenizer
│   ├── parser.rs    # Recursive descent parser
│   ├── ast.rs       # AST definitions
│   ├── types.rs     # Type representation
│   ├── infer.rs     # Type inference
│   ├── eval.rs      # Interpreter
│   └── runtime.rs   # Process scheduler
└── .beads/          # Issue tracking database
```

## Important Rules

- Use bd for ALL task tracking
- Always use `--json` flag for programmatic use
- Link discovered work with `discovered-from` dependencies
- Do NOT create markdown TODO lists

---

**For detailed workflows, see [AGENTS.md](../AGENTS.md)**
