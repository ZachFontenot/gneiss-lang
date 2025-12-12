# Gneiss Delimited Continuations: Implementation Validation & Fixes

## Executive Summary

This document provides findings from validating the Gneiss programming language's delimited continuations (`shift`/`reset`) implementation against established academic literature and reference implementations. **One critical fix is required**: continuation invocation must wrap restored frames in a fresh `Prompt` delimiter.

---

## 1. Critical Issue: Missing Reset Wrap on Continuation Invocation

### Problem

The current implementation applies captured continuations without wrapping them in a fresh delimiter. This violates the canonical `shift/reset` semantics.

### Canonical Reduction Rule

```
reset E[shift k body] → reset ((λk. body) (λx. reset E[x]))
```

The key insight: when `k` is invoked with argument `x`, the evaluation context `E[x]` is wrapped in a fresh `reset`. This is what distinguishes `shift/reset` from `prompt/control`.

### Current Behavior (Incorrect)

In `eval.rs`, continuation application:

```rust
Value::Continuation { frames } => {
    for frame in frames.into_iter().rev() {
        cont.push(frame);
    }
    StepResult::Continue(State::Apply { value: arg, cont })
}
```

This splices frames directly without a delimiter, implementing `prompt/control` semantics instead of `shift/reset`.

### Required Fix

Add a `Frame::Prompt` before splicing the captured frames:

```rust
Value::Continuation { frames } => {
    // CRITICAL: Push a fresh Prompt - this is the "reset" that wraps k's invocation
    // This implements: k x ≡ reset E[x]
    cont.push(Frame::Prompt);
    
    // Then splice the captured frames
    for frame in frames.into_iter().rev() {
        cont.push(frame);
    }
    StepResult::Continue(State::Apply { value: arg, cont })
}
```

### Why This Matters

Without this fix:
1. Nested `shift` operations capture incorrect contexts
2. Multiple continuation invocations may interfere
3. Programs relying on the delimiter for control flow isolation will behave incorrectly

---

## 2. Test Cases for Validation

### Test 2.1: Shift in Continuation Argument

This test verifies that continuation invocation is properly delimited.

```rust
#[test]
fn test_shift_in_continuation_argument() {
    let result = eval(r#"
        reset (
            shift (fun k1 -> k1 (shift (fun k2 -> k2 100))) + 1
        )
    "#).unwrap();
    
    // Execution trace with CORRECT semantics:
    // 1. Outer shift captures [□ + 1], binds to k1
    // 2. Evaluate body: k1 (shift (fun k2 -> k2 100))
    // 3. To apply k1, first evaluate argument
    // 4. Inner shift looks for nearest reset
    // 5. k1's invocation WRAPS IN RESET, so inner shift captures empty context []
    // 6. Inner shift returns: k2 100 = 100
    // 7. k1 100 = reset (100 + 1) = 101
    
    assert_eq!(result, Value::Int(101));
}
```

**Without the fix**: Inner `shift` may capture `k1`'s context, producing wrong results.

### Test 2.2: Multiple Continuation Invocations

```rust
#[test]
fn test_multiple_continuation_calls() {
    let result = eval(r#"
        reset (
            let x = shift (fun k -> k 1 + k 10 + k 100) in
            x * 2
        )
    "#).unwrap();
    
    // k 1 = reset (1 * 2) = 2
    // k 10 = reset (10 * 2) = 20  
    // k 100 = reset (100 * 2) = 200
    // Result: 2 + 20 + 200 = 222
    
    assert_eq!(result, Value::Int(222));
}
```

### Test 2.3: Discarded Continuation

```rust
#[test]
fn test_discarded_continuation() {
    let result = eval(r#"
        reset (
            1 + shift (fun k -> 42)
        )
    "#).unwrap();
    
    // k is never called, result is just 42
    assert_eq!(result, Value::Int(42));
}
```

### Test 2.4: Nested Resets

```rust
#[test]
fn test_nested_resets() {
    let result = eval(r#"
        reset (
            1 + reset (
                2 + shift (fun k -> k (k 10))
            )
        )
    "#).unwrap();
    
    // Inner shift captures [2 + □] only (stopped at inner reset)
    // k 10 = reset (2 + 10) = 12
    // k 12 = reset (2 + 12) = 14
    // Outer: 1 + 14 = 15
    
    assert_eq!(result, Value::Int(15));
}
```

### Test 2.5: Early Exit Pattern

```rust
#[test]
fn test_early_exit() {
    let result = eval(r#"
        reset (
            let escape = shift (fun k -> k) in
            escape 42;
            999
        )
    "#).unwrap();
    
    // This tests that shift can be used for early exit
    // escape becomes the continuation, calling escape 42 jumps out
    // The 999 is never reached
    
    assert_eq!(result, Value::Int(42));
}
```

---

## 3. Reference Semantics

### 3.1 Formal Operational Semantics

From Danvy & Filinski "Abstracting Control" (1990):

```
Evaluation contexts:
E ::= □ | E e | v E | op(v..., E, e...) | ...

Reduction rules:
⟨v⟩ → v                                           (reset-value)
⟨E[Sk.e]⟩ → ⟨(λk.e) (λx.⟨E[x]⟩)⟩                  (shift)
```

Key points:
- `⟨...⟩` denotes `reset`
- `Sk.e` denotes `shift k. e`
- The captured continuation `λx.⟨E[x]⟩` includes a reset wrapper

### 3.2 Comparison: shift/reset vs control/prompt

| Operator | Continuation Wrapper | Semantics |
|----------|---------------------|-----------|
| `shift/reset` | `λx. reset E[x]` | Static (wrapped) |
| `control/prompt` | `λx. E[x]` | Dynamic (unwrapped) |

Your current implementation behaves like `control/prompt`. The fix converts it to proper `shift/reset`.

### 3.3 CPS Transformation

The CPS transform makes the delimiter explicit:

```
⟦reset e⟧ = λk. k (⟦e⟧ (λx.x))
⟦shift f. e⟧ = λk. ⟦e⟧[f := λx.λk'. k' (k x)] (λx.x)
```

Note: `k` is applied inside a fresh `(λx.x)` reset.

---

## 4. Canonical Examples to Implement

### 4.1 State Monad

This is a classic test of delimited continuations:

```
let get () = shift (fun k -> fun s -> k s s)
let put v = shift (fun k -> fun _ -> k () v)

let run_state init comp = 
    (reset (
        let result = comp () in
        fun s -> (result, s)
    )) init

-- Test:
run_state 0 (fun () ->
    let x = get () in
    put (x + 1);
    let y = get () in
    put (y + 1);
    get ()
)
-- Expected: (2, 2)
```

**How it works:**
- `get` captures continuation, threads state through
- `put` captures continuation, replaces state
- `reset` delimits the stateful computation
- Result is a function from initial state to (result, final_state)

### 4.2 Nondeterminism / Backtracking

```
let fail () = shift (fun k -> [])
let choose xs = shift (fun k -> concat_map k xs)

let pythagorean_triples n =
    reset (
        let a = choose (range 1 n) in
        let b = choose (range a n) in
        let c = choose (range b n) in
        if a*a + b*b == c*c then
            [(a, b, c)]
        else
            fail ()
    )

-- pythagorean_triples 20 returns all triples up to 20
```

**How it works:**
- `choose` invokes continuation once per element
- `fail` discards continuation, returns empty
- Results accumulate via list concatenation

### 4.3 Generators / Yield

```
type 'a generator = Done | Yield of 'a * (unit -> 'a generator)

let yield x = shift (fun k -> Yield (x, k))

let tree_elements tree =
    reset (
        let rec walk t = match t with
            | Leaf x -> yield x
            | Node (l, r) -> walk l; walk r
        in
        walk tree;
        Done
    )

-- Lazy traversal: each yield suspends, k resumes
```

### 4.4 Printf (Answer Type Modification)

This example requires answer type modification in the type system:

```
let printf_int () = shift (fun k -> fun n -> k (string_of_int n))
let printf_str () = shift (fun k -> fun s -> k s)
let printf_lit s = s

-- Usage:
reset (printf_lit "x = " ^ printf_int () ^ printf_lit ", y = " ^ printf_int ())
-- Type: int -> int -> string
-- Calling with 5 10 produces "x = 5, y = 10"
```

**Note:** This requires answer-type polymorphism. Your current simplified typing (`k : a -> a`) rejects this. See Section 5.

---

## 5. Type System Considerations

### 5.1 Current Simplified Typing

```rust
// Current: k : τ → τ (answer type preserved)
let captured_ty = self.fresh_var();
let cont_ty = Type::arrow(captured_ty.clone(), captured_ty.clone());
```

This restricts `shift` to cases where the continuation's input and output types match.

### 5.2 Full Answer-Type Modification

The complete typing rule tracks answer type changes:

```
Γ ⊢ e : τ, α/β    means: e has type τ, changes answer type from α to β

Typing rules:
─────────────────────────────────
Γ ⊢ reset e : τ    (where Γ ⊢ e : τ, τ/τ)

Γ, k : (τ → α) → β ⊢ body : β, β/β
───────────────────────────────────
Γ ⊢ shift k. body : τ, α/β
```

### 5.3 Recommendation

For initial implementation, the simplified typing is acceptable. Document the limitation:

```
-- LIMITATION: Gneiss requires shift continuations to preserve answer type
-- k : τ → τ, not the full k : τ → α
-- This rejects some valid programs (e.g., typed printf)
```

Future enhancement: implement answer-type polymorphism following Asai & Kiselyov's type system.

---

## 6. Implementation Checklist

### Immediate Fix (Required)

- [ ] Add `cont.push(Frame::Prompt)` in continuation application
- [ ] Location: `eval.rs`, in the `Value::Continuation` match arm

### Validation Tests (Required)

- [ ] Test: shift in continuation argument (Section 2.1)
- [ ] Test: multiple continuation invocations (Section 2.2)
- [ ] Test: discarded continuation (Section 2.3)
- [ ] Test: nested resets (Section 2.4)
- [ ] Test: early exit pattern (Section 2.5)

### Example Programs (Recommended)

- [ ] State monad example working
- [ ] Simple nondeterminism (choose from list)
- [ ] Generator/yield pattern

### Documentation (Recommended)

- [ ] Document shift/reset semantics in language guide
- [ ] Note answer-type limitation
- [ ] Provide canonical examples

---

## 7. References

1. **Danvy & Filinski, "Abstracting Control" (1990)**
   - Seminal paper introducing shift/reset
   - Foundational reduction semantics

2. **Asai & Kiselyov, "Introduction to Programming with Shift and Reset" (2011)**
   - Tutorial: http://pllab.is.ocha.ac.jp/~asai/cw2011tutorial/main-e.pdf
   - Comprehensive examples and typing rules

3. **Tübingen PL1 Lecture Notes**
   - CPS interpreter implementation
   - https://ps-tuebingen-courses.github.io/pl1-lecture-notes/19-shift-reset/shift-reset.html

4. **Oleg Kiselyov's Continuations Page**
   - https://okmij.org/ftp/continuations/index.html
   - Advanced examples and implementations

---

## Appendix A: Quick Reference

### Reduction Rule
```
reset E[shift k body] → reset ((λk. body) (λx. reset E[x]))
                                              ↑↑↑↑↑
                                        THIS IS THE KEY
```

### The Fix (Copy-Paste Ready)
```rust
Value::Continuation { frames } => {
    // Wrap continuation invocation in reset (the defining feature of shift/reset)
    cont.push(Frame::Prompt);
    
    for frame in frames.into_iter().rev() {
        cont.push(frame);
    }
    StepResult::Continue(State::Apply { value: arg, cont })
}
```

### Mental Model
- `reset` = "I'm setting a checkpoint here"
- `shift` = "Capture everything back to the checkpoint, give it to me as a function"
- Calling `k` = "Run that captured computation, BUT set a new checkpoint first"
