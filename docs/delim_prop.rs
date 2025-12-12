//! Property-based tests for delimited continuations (shift/reset)
//!
//! Add to Cargo.toml:
//! ```toml
//! [dev-dependencies]
//! proptest = "1.4"
//! ```

use proptest::prelude::*;

// =============================================================================
// AST DEFINITIONS (adjust to match your actual AST)
// =============================================================================

/// Placeholder - replace with your actual Expr type
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Lit(i64),
    Bool(bool),
    Unit,
    Var(String),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Let(String, Box<Expr>, Box<Expr>),
    Lambda(String, Box<Expr>),
    App(Box<Expr>, Box<Expr>),
    Reset(Box<Expr>),
    Shift { param: String, body: Box<Expr> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Lt,
}

/// Placeholder - replace with your actual Value type
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Unit,
    Closure(/* ... */),
    Continuation(/* ... */),
}

/// Placeholder - replace with your actual eval function
fn eval(expr: &Expr) -> Result<Value, String> {
    todo!("Connect to your evaluator")
}

// =============================================================================
// CONTEXT REPRESENTATION
// =============================================================================

/// Evaluation context with a hole
#[derive(Clone, Debug)]
pub enum Context {
    Hole,
    AddLeft(Box<Context>, Expr),
    AddRight(Expr, Box<Context>),
    MulLeft(Box<Context>, Expr),
    MulRight(Expr, Box<Context>),
    SubLeft(Box<Context>, Expr),
    SubRight(Expr, Box<Context>),
    LetBinding(String, Box<Context>, Expr),
    LetBody(String, Expr, Box<Context>),
    AppFun(Box<Context>, Expr),
    AppArg(Expr, Box<Context>),
    IfCond(Box<Context>, Expr, Expr),
}

impl Context {
    /// Plug an expression into the context hole
    pub fn plug(&self, e: Expr) -> Expr {
        match self {
            Context::Hole => e,
            Context::AddLeft(ctx, right) => {
                Expr::BinOp(BinOp::Add, Box::new(ctx.plug(e)), Box::new(right.clone()))
            }
            Context::AddRight(left, ctx) => {
                Expr::BinOp(BinOp::Add, Box::new(left.clone()), Box::new(ctx.plug(e)))
            }
            Context::MulLeft(ctx, right) => {
                Expr::BinOp(BinOp::Mul, Box::new(ctx.plug(e)), Box::new(right.clone()))
            }
            Context::MulRight(left, ctx) => {
                Expr::BinOp(BinOp::Mul, Box::new(left.clone()), Box::new(ctx.plug(e)))
            }
            Context::SubLeft(ctx, right) => {
                Expr::BinOp(BinOp::Sub, Box::new(ctx.plug(e)), Box::new(right.clone()))
            }
            Context::SubRight(left, ctx) => {
                Expr::BinOp(BinOp::Sub, Box::new(left.clone()), Box::new(ctx.plug(e)))
            }
            Context::LetBinding(name, ctx, body) => {
                Expr::Let(name.clone(), Box::new(ctx.plug(e)), Box::new(body.clone()))
            }
            Context::LetBody(name, binding, ctx) => {
                Expr::Let(name.clone(), Box::new(binding.clone()), Box::new(ctx.plug(e)))
            }
            Context::AppFun(ctx, arg) => {
                Expr::App(Box::new(ctx.plug(e)), Box::new(arg.clone()))
            }
            Context::AppArg(fun, ctx) => {
                Expr::App(Box::new(fun.clone()), Box::new(ctx.plug(e)))
            }
            Context::IfCond(ctx, then_e, else_e) => {
                Expr::If(Box::new(ctx.plug(e)), Box::new(then_e.clone()), Box::new(else_e.clone()))
            }
        }
    }
    
    /// Compose two contexts
    pub fn compose(&self, inner: &Context) -> Context {
        match self {
            Context::Hole => inner.clone(),
            Context::AddLeft(ctx, right) => {
                Context::AddLeft(Box::new(ctx.compose(inner)), right.clone())
            }
            Context::AddRight(left, ctx) => {
                Context::AddRight(left.clone(), Box::new(ctx.compose(inner)))
            }
            // ... implement for other variants
            _ => todo!("Implement compose for all context variants")
        }
    }
}

// =============================================================================
// PROPTEST GENERATORS
// =============================================================================

/// Generate arbitrary integer values in a reasonable range
fn arb_int() -> impl Strategy<Value = i64> {
    -1000i64..1000i64
}

/// Generate small positive integers (useful for avoiding overflow)
fn arb_small_positive() -> impl Strategy<Value = i64> {
    1i64..20i64
}

/// Generate arbitrary pure values (no closures/continuations)
fn arb_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        arb_int().prop_map(Value::Int),
        any::<bool>().prop_map(Value::Bool),
        Just(Value::Unit),
    ]
}

/// Generate integer literals
fn arb_int_lit() -> impl Strategy<Value = Expr> {
    arb_int().prop_map(Expr::Lit)
}

/// Generate pure expressions (no shift) that evaluate to integers
fn arb_pure_int_expr() -> impl Strategy<Value = Expr> {
    let leaf = arb_int_lit();
    
    leaf.prop_recursive(
        3,   // max depth
        32,  // max nodes
        8,   // items per collection
        |inner| {
            prop_oneof![
                // Addition
                (inner.clone(), inner.clone()).prop_map(|(a, b)| {
                    Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b))
                }),
                // Multiplication (use smaller values to avoid overflow)
                (arb_small_positive().prop_map(Expr::Lit), arb_small_positive().prop_map(Expr::Lit))
                    .prop_map(|(a, b)| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b))),
                // Let binding
                (inner.clone(), inner.clone()).prop_map(|(binding, body)| {
                    Expr::Let("x".into(), Box::new(binding), Box::new(body))
                }),
            ]
        }
    )
}

/// Generate pure evaluation contexts (hole produces int, result is int)
fn arb_int_context() -> impl Strategy<Value = Context> {
    let leaf = Just(Context::Hole);
    
    leaf.prop_recursive(
        2,   // max depth
        8,   // max nodes  
        4,   // items per collection
        |inner| {
            prop_oneof![
                // [] + n
                (inner.clone(), arb_int_lit()).prop_map(|(ctx, e)| {
                    Context::AddLeft(Box::new(ctx), e)
                }),
                // n + []
                (arb_int_lit(), inner.clone()).prop_map(|(e, ctx)| {
                    Context::AddRight(e, Box::new(ctx))
                }),
                // [] * n (small n)
                (inner.clone(), arb_small_positive().prop_map(Expr::Lit)).prop_map(|(ctx, e)| {
                    Context::MulLeft(Box::new(ctx), e)
                }),
                // n * [] (small n)
                (arb_small_positive().prop_map(Expr::Lit), inner.clone()).prop_map(|(e, ctx)| {
                    Context::MulRight(e, Box::new(ctx))
                }),
                // [] - n
                (inner.clone(), arb_int_lit()).prop_map(|(ctx, e)| {
                    Context::SubLeft(Box::new(ctx), e)
                }),
            ]
        }
    )
}

// =============================================================================
// ALGEBRAIC LAW TESTS
// =============================================================================

proptest! {
    /// Law: reset v ≡ v
    #[test]
    fn law_reset_value_identity(n in arb_int()) {
        let expr = Expr::Reset(Box::new(Expr::Lit(n)));
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int(n));
    }
    
    /// Law: reset (reset e) ≡ reset e (for pure e)
    #[test]
    fn law_reset_reset_collapse(n in arb_int()) {
        let inner = Expr::Lit(n);
        let single = Expr::Reset(Box::new(inner.clone()));
        let double = Expr::Reset(Box::new(Expr::Reset(Box::new(inner))));
        
        let r1 = eval(&single).unwrap();
        let r2 = eval(&double).unwrap();
        prop_assert_eq!(r1, r2);
    }
    
    /// Law: reset (shift k. v) ≡ v (k unused, discards continuation)
    #[test]
    fn law_shift_discard_continuation(
        n in arb_int(),
        ctx in arb_int_context()
    ) {
        // reset (ctx[shift k. n])
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::Lit(n)),
        };
        let in_ctx = ctx.plug(shift_expr);
        let expr = Expr::Reset(Box::new(in_ctx));
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int(n));
    }
    
    /// Law: reset (shift k. k v) ≡ v
    #[test]
    fn law_shift_immediate_invoke(n in arb_int()) {
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::Lit(n)),
            )),
        };
        let expr = Expr::Reset(Box::new(shift_expr));
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int(n));
    }
    
    /// Law: Extracted continuation behaves like context
    /// reset (ctx[shift k. k]) applied to v ≡ reset (ctx[v])
    #[test]
    fn law_continuation_extraction(
        ctx in arb_int_context(),
        v in arb_small_positive()
    ) {
        // Extract continuation: reset (ctx[shift k. k])
        let extract = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::Var("k".into())),
        };
        let extract_expr = Expr::Reset(Box::new(ctx.plug(extract)));
        let cont = eval(&extract_expr).unwrap();
        
        // Apply to value
        let applied = eval(&Expr::App(
            Box::new(Expr::Lit(0)), // placeholder, replace with Value injection
            Box::new(Expr::Lit(v)),
        ));
        
        // Direct evaluation: reset (ctx[v])
        let direct = eval(&Expr::Reset(Box::new(ctx.plug(Expr::Lit(v))))).unwrap();
        
        // Note: This test needs adjustment to inject Value back into Expr
        // The concept is: apply cont to v should equal direct
    }
}

// =============================================================================
// CRITICAL SEMANTIC TESTS
// =============================================================================

proptest! {
    /// CRITICAL: Continuation invocation is wrapped in reset
    /// This is THE distinguishing property of shift/reset vs control/prompt
    #[test]
    fn critical_continuation_wrapped_in_reset(n in 0i64..100) {
        // reset (shift k1. k1 (shift k2. k2 n)) + 1
        //
        // With correct shift/reset semantics:
        // 1. Outer shift captures [□ + 1], binds to k1
        // 2. Evaluate: k1 (shift k2. k2 n)
        // 3. To call k1, first evaluate the argument
        // 4. Inner shift looks for nearest reset
        // 5. k1's invocation IS wrapped in reset, so inner shift captures []
        // 6. Inner shift: k2 n = n (identity continuation)
        // 7. k1 n = reset (n + 1) = n + 1
        //
        // Result: n + 1
        
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
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int(n + 1));
    }
    
    /// Multiple invocations are independent
    #[test]
    fn invariant_multiple_invocations_independent(a in 1i64..20, b in 1i64..20) {
        // reset (shift k. k a + k b) * 2
        // k captures [□ * 2]
        // k a = a * 2
        // k b = b * 2
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
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int((a * 2) + (b * 2)));
    }
    
    /// Captures exactly to nearest reset
    #[test]
    fn invariant_captures_to_nearest_reset(inner in 1i64..10, outer in 1i64..10) {
        // outer + reset (inner + shift k. k 0)
        // k captures only [inner + □]
        // Result: outer + (inner + 0) = outer + inner
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::Lit(0)),
            )),
        };
        
        let inner_expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(inner)),
            Box::new(shift_expr),
        );
        
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(outer)),
            Box::new(Expr::Reset(Box::new(inner_expr))),
        );
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int(outer + inner));
    }
    
    /// Nested resets delimit independently
    #[test]
    fn invariant_nested_resets_independent(n in 1i64..10) {
        // reset (1 + reset (2 + shift k. k (k n)))
        // Inner k captures [2 + □]
        // k n = 2 + n
        // k (k n) = k (2 + n) = 2 + (2 + n) = 4 + n
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
        
        let result = eval(&outer).unwrap();
        prop_assert_eq!(result, Value::Int(5 + n));
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

proptest! {
    /// Empty context continuation (identity)
    #[test]
    fn edge_empty_context(n in arb_int()) {
        // reset (shift k. k n)
        // k is identity continuation []
        // Result: n
        
        let expr = Expr::Reset(Box::new(Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::Lit(n)),
            )),
        }));
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int(n));
    }
    
    /// Deeply nested shifts
    #[test]
    fn edge_deeply_nested_shifts(depth in 1usize..6, base in 0i64..10) {
        // Each level: shift k. k [...] + 1
        // With proper reset wrapping, result = base + depth
        
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
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int(base + depth as i64));
    }
    
    /// Continuation called many times
    #[test]
    fn edge_multi_call(n in 2usize..6, base in 1i64..5) {
        // reset (shift k. k base + k base + ... ) * 2
        // n calls to k, each k base = base * 2
        // Result: n * (base * 2)
        
        // Build: k base + k base + ...
        let call_k = || Expr::App(
            Box::new(Expr::Var("k".into())),
            Box::new(Expr::Lit(base)),
        );
        
        let mut body = call_k();
        for _ in 1..n {
            body = Expr::BinOp(BinOp::Add, Box::new(body), Box::new(call_k()));
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
        
        let result = eval(&expr).unwrap();
        prop_assert_eq!(result, Value::Int((n as i64) * (base * 2)));
    }
}

// =============================================================================
// STRESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    /// Random pure expressions evaluate consistently in reset
    #[test]
    fn stress_pure_in_reset(e in arb_pure_int_expr()) {
        // reset (reset e) == reset e for pure e
        let single = Expr::Reset(Box::new(e.clone()));
        let double = Expr::Reset(Box::new(Expr::Reset(Box::new(e))));
        
        let r1 = eval(&single);
        let r2 = eval(&double);
        
        // Both should succeed or both fail the same way
        match (r1, r2) {
            (Ok(v1), Ok(v2)) => prop_assert_eq!(v1, v2),
            (Err(_), Err(_)) => {} // Both error is fine (e.g., div by zero)
            _ => prop_assert!(false, "Inconsistent evaluation"),
        }
    }
    
    /// Arithmetic with extracted continuations
    #[test]
    fn stress_continuation_arithmetic(
        a in arb_small_positive(),
        b in arb_small_positive(),
        c in arb_small_positive()
    ) {
        // reset (shift k. k a * k b + k c) + 1
        // k captures [□ + 1]
        // k x = x + 1
        // Result: (a + 1) * (b + 1) + (c + 1)
        
        let call_k = |n: i64| Expr::App(
            Box::new(Expr::Var("k".into())),
            Box::new(Expr::Lit(n)),
        );
        
        let body = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(call_k(a)),
                Box::new(call_k(b)),
            )),
            Box::new(call_k(c)),
        );
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(body),
        };
        
        let expr = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(shift_expr),
            Box::new(Expr::Lit(1)),
        )));
        
        let result = eval(&expr).unwrap();
        let expected = (a + 1) * (b + 1) + (c + 1);
        prop_assert_eq!(result, Value::Int(expected));
    }
}

// =============================================================================
// TEST FOR COMMON IMPLEMENTATION BUGS
// =============================================================================

#[cfg(test)]
mod bug_regression_tests {
    use super::*;
    
    /// Bug: Continuation not wrapped in reset
    /// This is the most common bug in shift/reset implementations
    #[test]
    fn regression_continuation_must_wrap_reset() {
        // This test MUST pass for correct shift/reset
        // It will FAIL if continuations are not wrapped (control/prompt behavior)
        
        // reset (shift k1. k1 (shift k2. k2 42)) + 1
        let inner = Expr::Shift {
            param: "k2".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k2".into())),
                Box::new(Expr::Lit(42)),
            )),
        };
        
        let outer = Expr::Shift {
            param: "k1".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k1".into())),
                Box::new(inner),
            )),
        };
        
        let expr = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(outer),
            Box::new(Expr::Lit(1)),
        )));
        
        let result = eval(&expr).unwrap();
        assert_eq!(result, Value::Int(43), 
            "Continuation invocation must be wrapped in reset! \
             Expected 43, got {:?}. \
             This indicates your implementation has control/prompt semantics \
             instead of shift/reset semantics.", result);
    }
    
    /// Bug: Captured frames not reversed correctly
    #[test]
    fn regression_frame_order() {
        // reset ((shift k. k 1) + 2) * 3
        // k captures [□ + 2] then [... * 3]
        // Order matters: should be ((1 + 2) * 3) = 9
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::App(
                Box::new(Expr::Var("k".into())),
                Box::new(Expr::Lit(1)),
            )),
        };
        
        let expr = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(shift_expr),
                Box::new(Expr::Lit(2)),
            )),
            Box::new(Expr::Lit(3)),
        )));
        
        let result = eval(&expr).unwrap();
        assert_eq!(result, Value::Int(9),
            "Frame capture/restore order is incorrect");
    }
    
    /// Bug: Multiple invocations share mutable state
    #[test]
    fn regression_invocations_isolated() {
        // reset (shift k. (k 1) + (k 2)) * 2
        // Each k invocation should be independent
        // k 1 = 1 * 2 = 2
        // k 2 = 2 * 2 = 4
        // Result: 2 + 4 = 6
        
        let shift_expr = Expr::Shift {
            param: "k".into(),
            body: Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::App(
                    Box::new(Expr::Var("k".into())),
                    Box::new(Expr::Lit(1)),
                )),
                Box::new(Expr::App(
                    Box::new(Expr::Var("k".into())),
                    Box::new(Expr::Lit(2)),
                )),
            )),
        };
        
        let expr = Expr::Reset(Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(shift_expr),
            Box::new(Expr::Lit(2)),
        )));
        
        let result = eval(&expr).unwrap();
        assert_eq!(result, Value::Int(6),
            "Multiple continuation invocations are not isolated");
    }
}
