# Session Start - Load Project Context

Initialize session with project context and current work status.

## Instructions

### 1. Set Beads Context
```
bd set_context --workspace-root <current directory>
```

### 2. Read Core Design Docs
- Read `docs/DESIGN.md` - design decisions and invariants
- Read `docs/ROADMAP.md` - development phases

Extract key constraints (the "don't do" list).

### 3. Check Previous Session State

Look for recently updated beads:
```
bd list --limit 5  # recently updated
```

Check if anything was left in_progress (indicates interrupted work).

### 4. Scan for Untracked Debt

Quick scan for debt markers not in beads:
```
grep -rn "TODO\|FIXME\|unimplemented!\|todo!" src/ tests/
```

If any found, note them - they may need beads.

### 5. Get Current Work Status
```
bd ready      # available work
bd blocked    # blocked issues
bd stats      # overview
```

### 6. Run Tests (Quick Health Check)
```
cargo test --quiet 2>&1 | tail -5
```

Note if tests are failing or any are ignored.

## Output Format

```
## Gneiss - Session Start

**Project:** ML-family language with CSP channels, Rust implementation.
**Phase:** 1 - Complete the Concurrency Story

### Key Constraints (Do NOT)
- Add typeclasses/traits (deferred to v0.3)
- Make channels async/buffered (sync rendezvous is intentional)
- Use parser generators (hand-written for better errors)
- Add Erlang-style receive (CML channels, not mailboxes)
- Generalize non-values (value restriction is soundness-critical)

### Previous Session
- [Any in_progress beads or recent updates]

### Test Status
- Passing: X, Failing: Y, Ignored: Z

### Untracked Debt
- [Any TODO/FIXME not in beads, or "None found"]

### Ready Work (by priority)
P1: [high priority beads]
P2: [medium priority beads]
P3: [low priority beads]

### Blocked
- [blocked beads and why, or "None"]

### Stats
Open: X | In Progress: Y | Blocked: Z | Closed: W
```

Keep output concise - this runs at every session start.
