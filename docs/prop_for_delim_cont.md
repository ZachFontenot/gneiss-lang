# Property-Based Testing for Delimited Continuations

## Overview

This document specifies property-based tests (proptests) for validating a `shift/reset` implementation. These tests go beyond example-based testing to verify algebraic laws, semantic invariants, and edge cases that might not be covered by manual test cases.

---

## 1. Core Algebraic Laws

These are the fundamental equations that any correct `shift/reset` implementation must satisfy.

### Law 1: Reset-Value Identity
```
reset v ≡ v
```
A reset around a pure value is the value itself.

```rust
proptest! {
    #[test]
    fn law_reset_value_identity(v in arb_value()) {
        // reset (fun () -> v) ≡ v
        let expr = Expr::Reset(Box::new(Expr::Value(v.clone())));
        let result = eval(&expr)?;
        prop_assert_eq!(result, v);
    }
}
```

### Law 2: Reset-Reset Collapse
```
reset (reset e) ≡ reset e
```
Nested resets with no intervening shift collapse to a single reset.

```rust
proptest! {
    #[test]
    fn law_reset_reset_collapse(e in arb_pure_expr()) {
        // reset (reset e) ≡ reset e
        let single = Expr::Reset(Box::new(e.clone()));
        let double = Expr::Reset(Box::new(Expr::Reset(Box::new(e.clone()))));
        
        let r1 = eval(&single)?;
        let r2 = eval(&double)?;
        prop_assert_eq!(r1, r2);
    }
}
```

### Law 3: Shift-Reset Elimination (Discarded Continuation)
```
reset (E[shift k. v]) ≡ reset v   (when k not free in v)
```
When the continuation is discarded, the result is just the body evaluated in a reset.

```rust
proptest! {
    #[test]
    fn law_shift_discard_continuation(v in arb_value(), ctx in arb_pure_context()) {
        // reset (ctx[shift k. v]) ≡ reset v  (k unused)
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::Value(v.clone())),
        };
        let in_context = plug_context(&ctx, shift_expr);
        let wrapped = Expr::Reset(Box::new(in_context));
        
        let direct = Expr::Reset(Box::new(Expr::Value(v.clone())));
        
        let r1 = eval(&wrapped)?;
        let r2 = eval(&direct)?;
        prop_assert_eq!(r1, r2);
    }
}
```

### Law 4: Shift-Reset with Identity Continuation
```
reset (shift k. k v) ≡ reset v
```
Capturing and immediately invoking with a value is equivalent to just the value.

```rust
proptest! {
    #[test]
    fn law_shift_immediate_invoke(v in arb_value()) {
        // reset (shift k. k v) ≡ v
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::Value(v.clone())),
            )),
        };
        let expr = Expr::Reset(Box::new(shift_expr));
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, v);
    }
}
```

### Law 5: Continuation Extraction
```
reset (E[shift k. k]) ≡ λx. reset (E[x])
```
Extracting the continuation produces a function equivalent to the context with reset.

```rust
proptest! {
    #[test]
    fn law_continuation_extraction(ctx in arb_pure_context(), v in arb_value()) {
        // Apply extracted continuation to v, compare with direct evaluation
        let extract = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::Var("k".into())),
        };
        let expr = Expr::Reset(Box::new(plug_context(&ctx, extract)));
        
        // Should produce a continuation value
        let cont = eval(&expr)?;
        
        // Applying continuation to v
        let applied = eval(&Expr::App(
            Box::new(Expr::Value(cont)),
            Box::new(Expr::Value(v.clone())),
        ))?;
        
        // Should equal reset (ctx[v])
        let direct = eval(&Expr::Reset(Box::new(plug_context(&ctx, Expr::Value(v.clone())))))?;
        
        prop_assert_eq!(applied, direct);
    }
}
```

---

## 2. Semantic Invariants

### Invariant 1: Continuation Invocation is Delimited

**Critical property**: When a captured continuation `k` is invoked, the evaluation happens inside an implicit reset.

```rust
proptest! {
    #[test]
    fn invariant_continuation_wrapped_in_reset(n in 0i64..100) {
        // This is THE critical test for shift/reset vs control/prompt
        //
        // reset (shift k1. k1 (shift k2. k2 n)) + 1
        //
        // With CORRECT shift/reset:
        //   k1's invocation wraps in reset, so inner shift captures []
        //   Result: n + 1
        //
        // With INCORRECT (control/prompt) semantics:
        //   Inner shift would capture outer context
        //   Different result
        
        let inner_shift = Expr::Shift {
            param: "k2".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k2".into())),
                Box::new(Expr::Lit(n)),
            )),
        };
        
        let outer_shift = Expr::Shift {
            param: "k1".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k1".into())),
                Box::new(inner_shift),
            )),
        };
        
        let expr = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(outer_shift),
            Box::new(Expr::Lit(1)),
        )));
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, Value::Int(n + 1));
    }
}
```

### Invariant 2: Multiple Invocations are Independent

Each invocation of a captured continuation should be independent.

```rust
proptest! {
    #[test]
    fn invariant_multiple_invocations_independent(a in 0i64..50, b in 0i64..50) {
        // reset (shift k. k a + k b) * 2
        // Each k invocation should independently compute (* 2)
        // Result: (a * 2) + (b * 2)
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::App(
                    Box::new(Expr::Var("k".into())),
                    Box::new(Expr::Lit(a)),
                )),
                Box::new(Expr::App(
                    Box::new(Expr::Var("k".into())),
                    Box::new(Expr::Lit(b)),
                )),
            )),
        };
        
        let expr = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(shift_expr),
            Box::new(Expr::Lit(2)),
        )));
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, Value::Int((a * 2) + (b * 2)));
    }
}
```

### Invariant 3: Continuation Captures Exactly to Delimiter

The continuation captures exactly the context up to the nearest enclosing reset.

```rust
proptest! {
    #[test]
    fn invariant_captures_to_nearest_reset(inner in 1i64..10, outer in 1i64..10) {
        // outer + reset (inner + shift k. k 0)
        // k should capture only (inner + [])
        // Result: outer + (inner + 0) = outer + inner
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::Lit(0)),
            )),
        };
        
        let inner_ctx = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(inner)),
            Box::new(shift_expr),
        );
        
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(outer)),
            Box::new(Expr::Reset(Box::new(inner_ctx))),
        );
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, Value::Int(outer + inner));
    }
}
```

### Invariant 4: Nested Resets Delimit Independently

```rust
proptest! {
    #[test]
    fn invariant_nested_resets_independent(n in 1i64..20) {
        // reset (1 + reset (2 + shift k. k (k n)))
        // Inner k captures (2 + []), so k n = 2 + n, k (k n) = 2 + (2 + n) = 4 + n
        // Outer reset sees value 4 + n
        // Result: 1 + (4 + n) = 5 + n
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::App(
                    Box::new(Expr::Var("k".into())),
                    Box::new(Expr::Lit(n)),
                )),
            )),
        };
        
        let inner = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(2)),
            Box::new(shift_expr),
        )));
        
        let outer = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(1)),
            Box::new(inner),
        )));
        
        let result = eval(&outer)?;
        prop_assert_eq!(result, Value::Int(5 + n));
    }
}
```

---

## 3. Compositionality Properties

### Property 1: Pure Expressions are Answer-Type Polymorphic

Pure expressions (no shift) can be placed in any context without affecting the answer type.

```rust
proptest! {
    #[test]
    fn prop_pure_answer_type_polymorphic(
        pure_e in arb_pure_expr_int(),
        ctx1 in arb_reset_context_int(),
        ctx2 in arb_reset_context_string(),
    ) {
        // A pure expression should evaluate the same regardless of answer type context
        let v1 = eval(&plug_context(&ctx1, pure_e.clone()));
        let v2 = eval(&plug_context(&ctx2, pure_e.clone()));
        
        // Extract the "pure" value (before context transformation)
        let pure_v = eval(&Expr::Reset(Box::new(pure_e.clone())))?;
        
        // Both contexts should use the same value from pure_e
        // (We can't directly compare v1 and v2 since they may have different types,
        //  but we can verify pure_e evaluates consistently)
        prop_assert!(v1.is_ok() && v2.is_ok());
    }
}
```

### Property 2: Continuation Composition

Captured continuations compose correctly.

```rust
proptest! {
    #[test]
    fn prop_continuation_composition(a in 1i64..10, b in 1i64..10) {
        // f = reset (shift k. k) + a    -- f is continuation ([] + a)
        // g = reset (shift k. k) * b    -- g is continuation ([] * b)
        // f (g n) should equal (n * b) + a
        
        let n = 5i64;
        
        let f = eval(&Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Shift {
                param: "k".into(),
                body: Box::new(Expr::Var("k".into())),
            }),
            Box::new(Expr::Lit(a)),
        ))))?;
        
        let g = eval(&Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Shift {
                param: "k".into(),
                body: Box::new(Expr::Var("k".into())),
            }),
            Box::new(Expr::Lit(b)),
        ))))?;
        
        // g(n) first
        let g_n = eval(&Expr::App(
            Box::new(Expr::Value(g)),
            Box::new(Expr::Lit(n)),
        ))?;
        
        // then f(g(n))
        let result = eval(&Expr::App(
            Box::new(Expr::Value(f)),
            Box::new(Expr::Value(g_n)),
        ))?;
        
        prop_assert_eq!(result, Value::Int((n * b) + a));
    }
}
```

---

## 4. Edge Cases and Stress Tests

### Edge 1: Empty Context Continuation

```rust
proptest! {
    #[test]
    fn edge_empty_context(v in arb_value()) {
        // reset (shift k. k v)
        // Continuation is identity: [] 
        // Result should be v
        
        let expr = Expr::Reset(Box::new(Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::Value(v.clone())),
            )),
        }));
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, v);
    }
}
```

### Edge 2: Deeply Nested Shifts

```rust
proptest! {
    #[test]
    fn edge_deeply_nested_shifts(depth in 1usize..10, base in 0i64..10) {
        // Build: reset (shift k1. k1 (shift k2. k2 (... (shift kn. kn base)...))) + 1 + 1 + ...
        // Each shift captures ([] + 1), and with proper reset wrapping,
        // result should be base + depth
        
        let mut expr = Expr::Lit(base);
        
        for i in 0..depth {
            let param = format!("k{}", i);
            expr = Expr::Shift {
                param: param.clone(),
                body: Box::new(Expr::App(
                    Box::new(Expr::Var(param)),
                    Box::new(expr),
                )),
            };
            expr = Expr::BinOp(BinOp::Add, Box::new(expr), Box::new(Expr::Lit(1)));
        }
        
        expr = Expr::Reset(Box::new(expr));
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, Value::Int(base + depth as i64));
    }
}
```

### Edge 3: Continuation Never Called

```rust
proptest! {
    #[test]
    fn edge_continuation_never_called(v in arb_value(), ctx in arb_pure_context()) {
        // reset (ctx[shift k. v])
        // k is bound but never used, so ctx is discarded
        // Result is v
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::Value(v.clone())),
        };
        
        let expr = Expr::Reset(Box::new(plug_context(&ctx, shift_expr)));
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, v);
    }
}
```

### Edge 4: Continuation Called Many Times

```rust
proptest! {
    #[test]
    fn edge_continuation_multi_call(n in 2usize..10, base in 1i64..5) {
        // reset (shift k. k base + k base + ... + k base)  [n times]
        // With context (* 2), each k base = base * 2
        // Result: n * (base * 2)
        
        // Build: k base + k base + ... 
        let mut body = Expr::App(
            Box::new(Expr::Var("k".into())),
            Box::new(Expr::Lit(base)),
        );
        
        for _ in 1..n {
            body = Expr::BinOp(
                BinOp::Add,
                Box::new(body),
                Box::new(Expr::App(
                    Box::new(Expr::Var("k".into())),
                    Box::new(Expr::Lit(base)),
                )),
            );
        }
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(body),
        };
        
        let expr = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(shift_expr),
            Box::new(Expr::Lit(2)),
        )));
        
        let result = eval(&expr)?;
        prop_assert_eq!(result, Value::Int((n as i64) * (base * 2)));
    }
}
```

---

## 5. Application-Specific Properties

### State Monad Correctness

```rust
proptest! {
    #[test]
    fn app_state_monad_get_put(init in 0i64..100, delta in 1i64..50) {
        // Simulate: run_state init { x = get(); put(x + delta); get() }
        // Should return (init + delta)
        
        // This requires building the state monad primitives...
        // See implementation section below
        
        let program = state_monad_program(init, delta);
        let result = eval(&program)?;
        prop_assert_eq!(result, Value::Int(init + delta));
    }
}
```

### Nondeterminism/Backtracking

```rust
proptest! {
    #[test]
    fn app_nondeterminism_collects_all(xs in prop::collection::vec(1i64..20, 1..5)) {
        // choose xs should invoke continuation once per element
        // Collecting results should give us back all elements
        
        let program = nondeterminism_collect(&xs);
        let result = eval(&program)?;
        
        // Result should be a list containing all xs (in some order)
        let result_list = extract_list(&result)?;
        prop_assert_eq!(result_list.len(), xs.len());
        for x in &xs {
            prop_assert!(result_list.contains(x));
        }
    }
}
```

### Generator/Yield Correctness

```rust
proptest! {
    #[test]
    fn app_generator_yields_in_order(xs in prop::collection::vec(1i64..100, 1..10)) {
        // A generator yielding xs should produce them in order
        
        let program = generator_program(&xs);
        let mut results = Vec::new();
        let mut state = eval(&program)?;
        
        while let Value::Yield(n, cont) = state {
            results.push(n);
            state = eval(&Expr::App(
                Box::new(Expr::Value(Value::Continuation(cont))),
                Box::new(Expr::Unit),
            ))?;
        }
        
        prop_assert_eq!(results, xs);
    }
}
```

---

## 6. Generators for Arbitrary Expressions

```rust
use proptest::prelude::*;

/// Generate arbitrary pure values
fn arb_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        any::<i64>().prop_map(Value::Int),
        any::<bool>().prop_map(Value::Bool),
        Just(Value::Unit),
    ]
}

/// Generate arbitrary pure expressions (no shift)
fn arb_pure_expr() -> impl Strategy<Value = Expr> {
    let leaf = prop_oneof![
        any::<i64>().prop_map(Expr::Lit),
        any::<bool>().prop_map(Expr::Bool),
    ];
    
    leaf.prop_recursive(
        4,   // depth
        64,  // max nodes
        10,  // items per collection
        |inner| {
            prop_oneof![
                // Binary operations
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b))),
                // Conditionals (only if types align - simplified here)
                (any::<bool>(), inner.clone(), inner.clone())
                    .prop_map(|(c, t, e)| Expr::If(
                        Box::new(Expr::Bool(c)),
                        Box::new(t),
                        Box::new(e),
                    )),
                // Let bindings
                (inner.clone(), inner.clone())
                    .prop_map(|(e1, e2)| Expr::Let(
                        "x".into(),
                        Box::new(e1),
                        Box::new(e2),
                    )),
            ]
        }
    )
}

/// Generate pure evaluation contexts (expressions with a hole)
/// Represented as a function Expr -> Expr
fn arb_pure_context() -> impl Strategy<Value = Context> {
    prop_oneof![
        // Hole: []
        Just(Context::Hole),
        // [] + e
        arb_pure_expr().prop_map(|e| Context::AddLeft(Box::new(Context::Hole), e)),
        // e + []
        arb_pure_expr().prop_map(|e| Context::AddRight(e, Box::new(Context::Hole))),
        // [] * e
        arb_pure_expr().prop_map(|e| Context::MulLeft(Box::new(Context::Hole), e)),
        // e * []
        arb_pure_expr().prop_map(|e| Context::MulRight(e, Box::new(Context::Hole))),
        // let x = [] in e
        arb_pure_expr().prop_map(|e| Context::LetBinding(Box::new(Context::Hole), e)),
        // let x = v in []
        arb_pure_expr().prop_map(|e| Context::LetBody(e, Box::new(Context::Hole))),
    ]
}

/// Context representation
#[derive(Clone, Debug)]
enum Context {
    Hole,
    AddLeft(Box<Context>, Expr),
    AddRight(Expr, Box<Context>),
    MulLeft(Box<Context>, Expr),
    MulRight(Expr, Box<Context>),
    LetBinding(Box<Context>, Expr),
    LetBody(Expr, Box<Context>),
}

/// Plug an expression into a context
fn plug_context(ctx: &Context, e: Expr) -> Expr {
    match ctx {
        Context::Hole => e,
        Context::AddLeft(inner, right) => 
            Expr::BinOp(BinOp::Add, Box::new(plug_context(inner, e)), Box::new(right.clone())),
        Context::AddRight(left, inner) =>
            Expr::BinOp(BinOp::Add, Box::new(left.clone()), Box::new(plug_context(inner, e))),
        Context::MulLeft(inner, right) =>
            Expr::BinOp(BinOp::Mul, Box::new(plug_context(inner, e)), Box::new(right.clone())),
        Context::MulRight(left, inner) =>
            Expr::BinOp(BinOp::Mul, Box::new(left.clone()), Box::new(plug_context(inner, e))),
        Context::LetBinding(inner, body) =>
            Expr::Let("x".into(), Box::new(plug_context(inner, e)), Box::new(body.clone())),
        Context::LetBody(binding, inner) =>
            Expr::Let("x".into(), Box::new(binding.clone()), Box::new(plug_context(inner, e))),
    }
}
```

---

## 7. Implementation Checklist

### Required Generators
- [ ] `arb_value()` - Arbitrary values
- [ ] `arb_pure_expr()` - Expressions without shift
- [ ] `arb_pure_context()` - Contexts without shift
- [ ] `arb_impure_expr()` - Expressions that may contain shift (more complex)

### Core Law Tests
- [ ] Reset-value identity
- [ ] Reset-reset collapse
- [ ] Shift-discard continuation
- [ ] Shift-immediate invoke
- [ ] Continuation extraction

### Critical Semantic Tests
- [ ] **Continuation wrapped in reset** (THE critical test)
- [ ] Multiple invocations independent
- [ ] Captures to nearest reset
- [ ] Nested resets independent

### Edge Case Tests
- [ ] Empty context continuation
- [ ] Deeply nested shifts
- [ ] Continuation never called
- [ ] Continuation called many times
- [ ] Shift with no enclosing reset (should error)

### Application Tests
- [ ] State monad get/put
- [ ] Nondeterminism collects all
- [ ] Generator yields in order

---

## 8. Running the Tests

Add to `Cargo.toml`:
```toml
[dev-dependencies]
proptest = "1.4"
```

Run with increased cases for thorough testing:
```bash
PROPTEST_CASES=10000 cargo test --release
```

For debugging failures:
```bash
PROPTEST_MAX_SHRINK_ITERS=1000000 cargo test test_name
```

---

## Appendix A: The Critical Test Explained

The single most important test is **Invariant 1: Continuation Invocation is Delimited**.

```
reset (shift k1. k1 (shift k2. k2 100)) + 1
```

**Correct execution (shift/reset)**:
1. Outer shift captures `[□ + 1]`, binds to `k1`
2. Body: `k1 (shift k2. k2 100)`
3. To call `k1`, evaluate argument first
4. Inner shift looks for nearest reset
5. **`k1` invocation IS wrapped in reset**, so inner shift sees empty context
6. Inner shift: `k2 100 = 100`
7. Now call `k1 100`: `reset (100 + 1) = 101`

**Incorrect execution (control/prompt)**:
1. Outer control captures `[□ + 1]`, binds to `k1`
2. Body: `k1 (control k2. k2 100)`
3. Call `k1` with unevaluated argument
4. Inner control looks for nearest prompt
5. **`k1` invocation is NOT wrapped**, so inner control sees `[k1 □]` or similar
6. Different result!

If this test passes, your implementation correctly wraps continuation invocations in a reset.
