# Session End - Honest Status Check

Before closing this session, perform a rigorous audit to ensure work is accurately tracked.

## Instructions

### 1. Scan for Deferred Work

Search the codebase for debt markers:
```
grep -rn "TODO\|FIXME\|unimplemented!\|todo!\|#\[ignore" src/ tests/
```

For each finding:
- Is there a bead tracking it? If not, create one.
- If it was added THIS session, explicitly note it.

### 2. Check for Risky Patterns

Count and review:
```
grep -c "\.unwrap()" src/*.rs      # Potential panic points
grep -c "panic!" src/*.rs          # Explicit panics
grep -c "unreachable!" src/*.rs    # Assumed-dead paths
```

If any were ADDED this session, either:
- Justify why it's safe (document in bead notes), OR
- Create a bead to handle the error properly

### 3. Audit In-Progress Beads

For each bead that was worked on this session:
- Run `bd show <id>` to see current state
- Update with HONEST notes:
  - What's actually done vs what was attempted
  - What tests exist for the completed work
  - What's NOT covered / NOT working
  - Any workarounds or hacks introduced

### 4. Audit "Completed" Beads

For any bead closed this session:
- Verify the acceptance criteria are actually met
- If tests were supposed to pass, run them: `cargo test`
- If any test is ignored or failing, REOPEN the bead or create a follow-up

### 5. Check for Undiscovered Work

Review the git diff for this session:
```
git diff --stat
git diff  # full diff
```

Look for:
- New error paths that need handling
- New features that need tests
- Assumptions that should be documented
- Anything you thought "I'll fix this later"

Create beads for each discovery.

### 6. Create Session Summary

Update any worked-on beads with session notes using:
```
bd update <id> --notes "Session YYYY-MM-DD: <what happened>"
```

### 7. Final Git Status

```
git status
```

- Uncommitted changes? Either commit or stash with clear message.
- For commits, message should reflect ACTUAL state, not aspirational state.

## Output Format

Provide a summary:

```
## Session End Report

### Worked On
- [bead-id]: [status - what's actually done]

### Discovered Issues (new beads created)
- [new-bead-id]: [title]

### Deferred Work Found
- [file:line]: [marker] - [tracked by bead-id / UNTRACKED]

### Honest Assessment
- Tests passing: yes/no (X ignored)
- New unwrap()s added: N
- Incomplete features: [list]

### Git State
- Branch: X
- Uncommitted changes: yes/no
- Ready to commit: yes/no
```

## Critical Rule

**DO NOT say "done" or "complete" unless:**
1. Tests pass (none ignored for convenience)
2. All acceptance criteria verifiably met
3. No untracked TODO/FIXME added
4. Beads updated with honest notes

If something is partially done, say "partial" and explain what remains.
