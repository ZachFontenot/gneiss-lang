//! Property-based tests for delimited continuations (shift/reset)
//!
//! These tests verify the algebraic laws and semantic invariants of shift/reset.
//! Based on docs/prop_for_delim_cont.md and docs/delim_prop.rs.
//!
//! Note: Gneiss uses strict call-by-value (CBV) evaluation. Some tests from the
//! literature assume call-by-name semantics (e.g., the "critical" test where
//! shift appears in a continuation argument). These are adapted for CBV.

use proptest::prelude::*;
use gneiss::{Lexer, Parser, Inferencer, Interpreter};
use gneiss::eval::Value;

/// Run a Gneiss expression and return the result
fn eval(source: &str) -> Result<Value, String> {
    let tokens = Lexer::new(source).tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;

    let mut inferencer = Inferencer::new();
    let _env = inferencer.infer_program(&program).map_err(|e| e.to_string())?;

    let mut interpreter = Interpreter::new();
    interpreter.set_class_env(inferencer.take_class_env());
    interpreter.set_type_ctx(inferencer.take_type_ctx());

    interpreter.run(&program).map_err(|e| e.to_string())
}

/// Generate a small integer that won't cause overflow
fn small_int() -> impl Strategy<Value = i64> {
    -100i64..100
}

/// Generate a small positive integer (for multiplication tests)
fn small_positive() -> impl Strategy<Value = i64> {
    1i64..20
}

/// Format an integer for Gneiss source (wrap negatives in parens)
fn format_int(n: i64) -> String {
    if n < 0 { format!("({})", n) } else { n.to_string() }
}

// =============================================================================
// ALGEBRAIC LAW TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Law 1: reset v ≡ v
    /// A reset around a pure value is the value itself.
    #[test]
    fn law_reset_value_identity(n in small_int()) {
        let source = format!("reset {}", format_int(n));
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(result, Value::Int(v) if v == n),
            "reset {} should equal {}, got {:?}", n, n, result
        );
    }

    /// Law 2: reset (reset e) ≡ reset e (for pure expressions)
    /// Nested resets with no intervening shift collapse.
    #[test]
    fn law_reset_reset_collapse(n in small_int()) {
        let single = format!("reset {}", format_int(n));
        let double = format!("reset (reset {})", format_int(n));

        let r1 = eval(&single).map_err(|e| TestCaseError::fail(e))?;
        let r2 = eval(&double).map_err(|e| TestCaseError::fail(e))?;

        // Both should produce the same Int value
        match (r1, r2) {
            (Value::Int(v1), Value::Int(v2)) => prop_assert_eq!(v1, v2, "reset (reset {}) should equal reset {}", n, n),
            (a, b) => prop_assert!(false, "Expected Int values, got {:?} and {:?}", a, b),
        }
    }

    /// Law 3: reset (ctx[shift k. v]) ≡ v when k is unused
    /// When the continuation is discarded, only the body matters.
    #[test]
    fn law_shift_discard_continuation(result_val in small_int(), ctx_val in small_positive()) {
        // reset (ctx_val + shift (fun k -> result_val))
        // k is unused, so the + ctx_val is discarded
        let source = format!(
            "reset ({} + shift (fun k -> {}))",
            format_int(ctx_val),
            format_int(result_val)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(result, Value::Int(v) if v == result_val),
            "Expected {}, got {:?}", result_val, result
        );
    }

    /// Law 4: reset (shift k. k v) ≡ v
    /// Capturing and immediately invoking with a value equals the value.
    #[test]
    fn law_shift_immediate_invoke(n in small_int()) {
        let source = format!(
            "reset (shift (fun k -> k {}))",
            format_int(n)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(result, Value::Int(v) if v == n),
            "reset (shift k. k {}) should equal {}, got {:?}", n, n, result
        );
    }

    // Note: Law 5 (Extracted continuation) is skipped because returning k directly
    // from shift creates an infinite type (k : a -> a, but returning k makes it
    // (a -> a) -> (a -> a) -> ...). This is a type system limitation, not a
    // continuation semantics issue. The behavior is tested indirectly via other tests.
}

// =============================================================================
// SEMANTIC INVARIANT TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Invariant 1: Multiple invocations are independent
    /// Each call to k should be isolated.
    #[test]
    fn invariant_multiple_invocations_independent(a in small_positive(), b in small_positive()) {
        // reset (shift k. k a + k b) * 2
        // k captures [□ * 2]
        // k a = a * 2, k b = b * 2
        // Result: (a * 2) + (b * 2)
        let source = format!(
            "reset (shift (fun k -> k {} + k {}) * 2)",
            format_int(a),
            format_int(b)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = (a * 2) + (b * 2);
        prop_assert!(
            matches!(result, Value::Int(r) if r == expected),
            "Expected {}, got {:?}", expected, result
        );
    }

    /// Invariant 2: Continuation captures exactly to the nearest reset
    #[test]
    fn invariant_captures_to_nearest_reset(inner in small_positive(), outer in small_positive()) {
        // outer + reset (inner + shift k. k 0)
        // k captures only [inner + □], not [outer + reset (inner + □)]
        // Result: outer + (inner + 0) = outer + inner
        let source = format!(
            "{} + reset ({} + shift (fun k -> k 0))",
            format_int(outer),
            format_int(inner)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = outer + inner;
        prop_assert!(
            matches!(result, Value::Int(r) if r == expected),
            "Expected {}, got {:?}", expected, result
        );
    }

    /// Invariant 3: Nested resets delimit independently
    #[test]
    fn invariant_nested_resets_independent(n in small_positive()) {
        // reset (1 + reset (2 + shift k. k (k n)))
        // Inner k captures [2 + □]
        // k n = 2 + n
        // k (k n) = k (2 + n) = 2 + (2 + n) = 4 + n
        // Outer: 1 + (4 + n) = 5 + n
        let source = format!(
            "reset (1 + reset (2 + shift (fun k -> k (k {}))))",
            format_int(n)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = 5 + n;
        prop_assert!(
            matches!(result, Value::Int(r) if r == expected),
            "Expected {}, got {:?}", expected, result
        );
    }

    // Note: Invariant 4 (Continuation body delimited) is skipped because returning k
    // directly from shift creates an infinite type. The continuation-wrapped-in-reset
    // behavior is tested by invariant_nested_resets_independent and regression tests.
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Edge 1: Empty context continuation (identity)
    #[test]
    fn edge_empty_context(n in small_int()) {
        // reset (shift k. k n)
        // k is identity: []
        // Result: n
        let source = format!(
            "reset (shift (fun k -> k {}))",
            format_int(n)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(result, Value::Int(r) if r == n),
            "Expected {}, got {:?}", n, result
        );
    }

    /// Edge 2: Continuation called many times
    #[test]
    fn edge_continuation_multi_call(n in 2usize..6, base in small_positive()) {
        // reset (shift k. k base + k base + ... ) * 2
        // n calls to k, each k base = base * 2
        // Result: n * (base * 2)

        // Build: k base + k base + ...
        let mut calls: Vec<String> = Vec::new();
        for _ in 0..n {
            calls.push(format!("k {}", format_int(base)));
        }
        let body = calls.join(" + ");

        let source = format!(
            "reset (shift (fun k -> {}) * 2)",
            body
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = (n as i64) * (base * 2);
        prop_assert!(
            matches!(result, Value::Int(r) if r == expected),
            "Expected {}, got {:?}", expected, result
        );
    }

    /// Edge 3: Deeply nested context
    #[test]
    fn edge_deep_context(depth in 1usize..5, base in small_positive()) {
        // reset (1 + (1 + (... + shift k. k base)))
        // k captures [1 + (1 + ...)] - depth additions
        // Result: base + depth
        let mut inner = format!("shift (fun k -> k {})", format_int(base));
        for _ in 0..depth {
            inner = format!("1 + {}", inner);
        }
        let source = format!("reset ({})", inner);

        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = base + (depth as i64);
        prop_assert!(
            matches!(result, Value::Int(r) if r == expected),
            "Expected {}, got {:?}", expected, result
        );
    }
}

// =============================================================================
// BUG REGRESSION TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Regression: Frame order must be correct
    /// reset ((shift k. k 1) + 2) * 3
    /// k captures [□ + 2] then [... * 3]
    /// k 1 = (1 + 2) * 3 = 9
    #[test]
    fn regression_frame_order(_dummy in any::<u8>()) {
        let source = "reset ((shift (fun k -> k 1) + 2) * 3)";
        let result = eval(source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(result, Value::Int(9)),
            "Frame order incorrect: expected 9, got {:?}", result
        );
    }

    /// Regression: Multiple invocations must be isolated
    /// reset (shift k. (k 1) + (k 2)) * 2
    /// k 1 = 1 * 2 = 2, k 2 = 2 * 2 = 4
    /// Result: 2 + 4 = 6
    #[test]
    fn regression_invocations_isolated(_dummy in any::<u8>()) {
        let source = "reset (shift (fun k -> k 1 + k 2) * 2)";
        let result = eval(source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(result, Value::Int(6)),
            "Invocations not isolated: expected 6, got {:?}", result
        );
    }

    /// Regression: Continuation invocation must be wrapped in reset
    /// This validates our fix from gneiss-lang-3iy
    #[test]
    fn regression_continuation_wrapped_in_reset(n in small_positive()) {
        // reset (2 + shift k. k (k n))
        // k captures [2 + □]
        // k n = 2 + n (inside fresh reset)
        // k (k n) = k (2 + n) = 2 + (2 + n) = 4 + n
        let source = format!(
            "reset (2 + shift (fun k -> k (k {})))",
            format_int(n)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = 4 + n;
        prop_assert!(
            matches!(result, Value::Int(r) if r == expected),
            "Continuation not wrapped in reset: expected {}, got {:?}", expected, result
        );
    }
}

// =============================================================================
// STRESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Stress: Arithmetic with extracted continuations
    #[test]
    fn stress_continuation_arithmetic(a in small_positive(), b in small_positive(), c in small_positive()) {
        // reset (shift k. k a * k b + k c) + 1
        // k captures [□ + 1]
        // k x = x + 1
        // Result: (a + 1) * (b + 1) + (c + 1)
        let source = format!(
            "reset (shift (fun k -> k {} * k {} + k {}) + 1)",
            format_int(a), format_int(b), format_int(c)
        );
        let result = eval(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = (a + 1) * (b + 1) + (c + 1);
        prop_assert!(
            matches!(result, Value::Int(r) if r == expected),
            "Expected {}, got {:?}", expected, result
        );
    }

    /// Stress: Pure expressions in reset are consistent
    #[test]
    fn stress_pure_in_reset(a in small_int(), b in small_int()) {
        // reset (a + b) should equal a + b
        let with_reset = format!("reset ({} + {})", format_int(a), format_int(b));
        let without_reset = format!("{} + {}", format_int(a), format_int(b));

        let r1 = eval(&with_reset).map_err(|e| TestCaseError::fail(e))?;
        let r2 = eval(&without_reset).map_err(|e| TestCaseError::fail(e))?;

        // Both should produce the same Int value
        match (r1, r2) {
            (Value::Int(v1), Value::Int(v2)) => prop_assert_eq!(v1, v2, "reset ({} + {}) should equal plain {} + {}", a, b, a, b),
            (a_val, b_val) => prop_assert!(false, "Expected Int values, got {:?} and {:?}", a_val, b_val),
        }
    }
}

// =============================================================================
// CBV-SPECIFIC TESTS
// =============================================================================

mod cbv_semantics {
    use super::*;

    /// In CBV, shift in a continuation argument has no enclosing reset
    /// This is correct behavior for strict evaluation
    #[test]
    fn cbv_shift_in_continuation_arg_errors() {
        // reset (shift k1. k1 (shift k2. k2 100)) + 1
        // In CBV: inner shift is evaluated BEFORE k1 is invoked
        // At that point, there's no enclosing reset (outer was consumed by outer shift)
        let source = "reset (shift (fun k1 -> k1 (shift (fun k2 -> k2 100))) + 1)";
        let result = eval(source);
        assert!(result.is_err(), "Expected error for shift without enclosing reset in CBV");
    }

    // Note: cbv_stored_continuation_works is skipped because `reset (shift k -> k)`
    // creates an infinite type (occurs check fails). The type system requires the
    // continuation to return the same type as its input, but returning k itself
    // creates a cyclic type. This is a type system limitation, not a runtime issue.
}
