//! Property-based tests for effect system and purity tracking
//!
//! These tests verify the type system properties of delimited continuations
//! and algebraic effects, specifically testing that effect tracking is correct.
//! The runtime tests are in proptest_continuations.rs; these test the TYPE SYSTEM.

use gneiss::types::{InferResult, Row, Type, TypeEnv, TypeVar};
use gneiss::{Inferencer, Lexer, Parser};
use proptest::prelude::*;
use std::rc::Rc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse and infer a source string, returning InferResult with answer types
fn infer_full(source: &str) -> Result<InferResult, String> {
    let tokens = Lexer::new(source).tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr().map_err(|e| e.to_string())?;

    let mut inferencer = Inferencer::new();
    let env = TypeEnv::new();
    inferencer
        .infer_expr_full(&env, &expr)
        .map_err(|e| e.to_string())
}

/// Parse and infer, returning just the type
fn infer_type(source: &str) -> Result<Type, String> {
    infer_full(source).map(|r| r.ty)
}

/// Check if two types are the same after resolution (for type variables)
fn types_same(t1: &Type, t2: &Type) -> bool {
    let t1 = t1.resolve();
    let t2 = t2.resolve();
    match (&t1, &t2) {
        (Type::Var(v1), Type::Var(v2)) => {
            // Check if same variable by comparing Rc pointers or IDs
            if Rc::ptr_eq(v1, v2) {
                return true;
            }
            match (&*v1.borrow(), &*v2.borrow()) {
                (TypeVar::Unbound { id: id1, .. }, TypeVar::Unbound { id: id2, .. }) => id1 == id2,
                (TypeVar::Generic(id1), TypeVar::Generic(id2)) => id1 == id2,
                _ => false,
            }
        }
        (Type::Int, Type::Int) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::String, Type::String) => true,
        (Type::Unit, Type::Unit) => true,
        _ => false,
    }
}

/// Check if an expression is pure (empty effect row)
fn is_pure(result: &InferResult) -> bool {
    matches!(result.effects.resolve(), Row::Empty)
}

/// Generate small integers that won't overflow
fn small_int() -> impl Strategy<Value = i64> {
    -100i64..100
}

/// Generate small positive integers
fn small_positive() -> impl Strategy<Value = i64> {
    1i64..20
}

/// Format an integer for Gneiss source (wrap negatives in parens)
fn format_int(n: i64) -> String {
    if n < 0 {
        format!("({})", n)
    } else {
        n.to_string()
    }
}

// ============================================================================
// Category 1: Purity Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property 1.1: Integer literals are pure
    #[test]
    fn purity_int_literal(n in small_int()) {
        let source = format_int(n);
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            is_pure(&result),
            "Int literal {} should be pure", n
        );
    }

    /// Property 1.1: Boolean literals are pure
    #[test]
    fn purity_bool_literal(b in any::<bool>()) {
        let source = b.to_string();
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            is_pure(&result),
            "Bool literal {} should be pure", b
        );
    }

    /// Property 1.1: Unit literal is pure
    #[test]
    fn purity_unit_literal(_dummy in any::<u8>()) {
        let result = infer_full("()").map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Unit literal should be pure");
    }

    /// Property 1.3: Lambda expressions are pure (building closure doesn't run body)
    #[test]
    fn purity_lambda(_dummy in any::<u8>()) {
        let result = infer_full("fun x -> x").map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Lambda expression should be pure");
    }

    /// Property 1.3: Lambda with computation in body is still pure to create
    #[test]
    fn purity_lambda_with_body(n in small_int()) {
        let source = format!("fun x -> x + {}", format_int(n));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Lambda with arithmetic body should be pure");
    }

    /// Property 1.4: Pure binary operations preserve purity
    #[test]
    fn purity_binop_preserves(a in small_int(), b in small_int()) {
        let source = format!("{} + {}", format_int(a), format_int(b));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            is_pure(&result),
            "Pure addition {} + {} should be pure", a, b
        );
    }

    /// Property 1.5: Reset makes ANY expression pure from outside
    #[test]
    fn purity_reset_makes_pure(n in small_int()) {
        let source = format!("reset {}", format_int(n));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "reset should make expression pure");
    }

    /// Property 1.5: Reset makes shift expression pure from outside
    #[test]
    fn purity_reset_shift(_dummy in any::<u8>()) {
        let source = "reset (shift (fun k -> k 42))";
        let result = infer_full(source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "reset (shift ...) should be pure from outside");
    }

    /// Property 1.5: Even reset of discarded continuation is pure
    #[test]
    fn purity_reset_discarded_k(_dummy in any::<u8>()) {
        let source = r#"reset (shift (fun k -> "hello"))"#;
        let result = infer_full(source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "reset (shift ...) with discarded k should be pure");
    }
}

// ============================================================================
// Category 2: Answer-Type Threading
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property 2.1: BinOp threads answer types left-to-right - pure case
    #[test]
    fn threading_binop_pure_pure(a in small_int(), b in small_int()) {
        let source = format!("{} + {}", format_int(a), format_int(b));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "pure + pure should be pure");
    }

    /// Property 2.2: Let threads answer types sequentially
    #[test]
    fn threading_let_sequential(n in small_int()) {
        let source = format!("let x = {} in x + 1", format_int(n));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "let with pure binding and body should be pure");
    }

    /// Property 2.3: If-then-else unifies branch answer types
    #[test]
    fn threading_if_branches_unify(c in any::<bool>(), t in small_int(), e in small_int()) {
        let source = format!("if {} then {} else {}", c, format_int(t), format_int(e));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "if with pure branches should be pure");
    }

    /// Property 2.4: Application threads through function and argument
    /// NOTE: Temporarily ignored during effect system migration (Arrow type changed)
    #[test]
    #[ignore]
    fn threading_application_pure(n in small_int()) {
        let source = format!("(fun x -> x + 1) {}", format_int(n));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Pure application should be pure");
    }
}

// ============================================================================
// Category 3: Reset/Shift Rules - THE KEY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property 3.1: Reset result type equals body's answer_after
    #[test]
    fn reset_result_type(n in small_int()) {
        let source = format!("reset {}", format_int(n));
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(result.ty.resolve(), Type::Int),
            "reset {} should have type Int", n
        );
    }

    /// CRITICAL Property 3.4: Answer type modification changes result type
    /// This is the key test for answer-type polymorphism!
    /// NOTE: Ignored during effect system migration - will be reimplemented with handlers
    #[test]
    #[ignore]
    fn answer_type_modification_string(_dummy in any::<u8>()) {
        // reset (1 + shift (fun k -> "hello"))
        // The shift body returns String, discarding the continuation
        // So the whole thing should type as String, NOT Int
        let source = r#"reset (1 + shift (fun k -> "hello"))"#;
        let result = infer_full(source).map_err(|e| TestCaseError::fail(e))?;

        prop_assert!(
            matches!(result.ty.resolve(), Type::String),
            "Answer type modification should produce String, got {:?}", result.ty
        );
    }

    /// Property 3.3: Continuation k is pure and can be called multiple times
    #[test]
    fn continuation_multiple_calls(a in small_positive(), b in small_positive()) {
        let source = format!(
            "reset (shift (fun k -> k {} + k {}) * 2)",
            a, b
        );

        let result = infer_full(&source);
        prop_assert!(
            result.is_ok(),
            "Multiple calls to k should type-check: {:?}", result
        );

        let r = result.unwrap();
        prop_assert!(
            matches!(r.ty.resolve(), Type::Int),
            "Result should be Int"
        );
    }

    /// Answer type modification with nested computation
    /// NOTE: Ignored during effect system migration - will be reimplemented with handlers
    #[test]
    #[ignore]
    fn answer_type_in_context(n in small_positive()) {
        // reset (n + shift (fun k -> "changed"))
        // k is discarded, so result is String
        let source = format!(r#"reset ({} + shift (fun k -> "changed"))"#, n);
        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;

        prop_assert!(
            matches!(result.ty.resolve(), Type::String),
            "Should be String regardless of context, got {:?}", result.ty
        );
    }
}

// ============================================================================
// Category 4: Polymorphism and Purity
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property 4.1: Pure let bindings can be polymorphic
    #[test]
    fn polymorphism_pure_let(_dummy in any::<u8>()) {
        let source = "let id = fun x -> x in (id 1, id true)";
        let result = infer_type(source);
        prop_assert!(
            result.is_ok(),
            "Pure polymorphic id should work: {:?}", result
        );
    }
}

// ============================================================================
// Category 5: Error Detection
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Continuation receives wrong type
    #[test]
    fn error_continuation_type_mismatch(_dummy in any::<u8>()) {
        // In reset ([] + 2), the hole expects Int
        // If we pass Bool to k, it should fail
        let source = "reset ((shift (fun k -> k true)) + 2)";
        let result = infer_type(source);
        prop_assert!(
            result.is_err(),
            "Should fail: continuation expects Int but got Bool"
        );
    }

    /// Type mismatch in BinOp operands
    #[test]
    fn error_binop_type_mismatch(_dummy in any::<u8>()) {
        let source = "reset ((shift (fun k -> k 1)) + true)";
        let result = infer_type(source);
        prop_assert!(
            result.is_err(),
            "Should fail: trying to add Int to Bool"
        );
    }
}

// ============================================================================
// Category 6: Regression Tests
// ============================================================================

#[cfg(test)]
mod regression {
    use super::*;

    /// Bug: Wrong threading direction would cause effects to flow backwards
    #[test]
    fn regression_threading_direction() {
        let source = "reset (reset (shift (fun k -> k 1)) + reset (shift (fun k -> k 2)))";
        let result = infer_full(source);
        assert!(result.is_ok(), "Threading direction test: {:?}", result);
        assert!(matches!(result.unwrap().ty.resolve(), Type::Int));
    }

    /// Bug: Missing threading in Let expression
    #[test]
    fn regression_missing_threading_let() {
        let source = "reset (let x = shift (fun k -> k 5) in x + 1)";
        let result = infer_full(source);
        assert!(result.is_ok(), "Let threading: {:?}", result);
        assert!(matches!(result.unwrap().ty.resolve(), Type::Int));
    }

    /// Bug: Missing threading in If expression
    #[test]
    fn regression_missing_threading_if() {
        let source = "reset (if true then shift (fun k -> k 1) else shift (fun k -> k 2))";
        let result = infer_full(source);
        assert!(result.is_ok(), "If threading: {:?}", result);
    }

    /// Bug: Reset not constraining body properly - KEY TEST
    /// NOTE: Ignored during effect system migration - will be reimplemented with handlers
    #[test]
    #[ignore]
    fn regression_reset_constraint() {
        let source = r#"reset (shift (fun k -> "changed"))"#;
        let result = infer_full(source);
        assert!(result.is_ok());
        // Result should be String, not whatever the hole type was
        assert!(
            matches!(result.unwrap().ty.resolve(), Type::String),
            "Answer type modification must work"
        );
    }

    /// Bug: Answer type modification through BinOp
    /// NOTE: Ignored during effect system migration - will be reimplemented with handlers
    #[test]
    #[ignore]
    fn regression_binop_answer_modification() {
        // This was the original soundness bug!
        // reset (1 + shift (fun k -> "hello")) should be String, not Int
        let source = r#"reset (1 + shift (fun k -> "hello"))"#;
        let result = infer_full(source);
        assert!(result.is_ok(), "Should type-check: {:?}", result);
        let ty = result.unwrap().ty.resolve();
        assert!(
            matches!(ty, Type::String),
            "CRITICAL: Answer type modification through BinOp must produce String, got {:?}",
            ty
        );
    }

    /// Bug: Application threading
    #[test]
    fn regression_app_threading() {
        let source = "reset ((fun x -> x + 1) (shift (fun k -> k 10)))";
        let result = infer_full(source);
        assert!(result.is_ok(), "App threading: {:?}", result);
    }

    /// Continuation can be called multiple times
    #[test]
    fn regression_continuation_multi_call() {
        let source = "reset (shift (fun k -> k 1 + k 2 + k 3) * 10)";
        let result = infer_full(source);
        assert!(result.is_ok(), "Continuation multiple use: {:?}", result);
    }
}

// ============================================================================
// Category 7: Stress Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Stress: Deeply nested pure expressions stay pure
    #[test]
    fn stress_nested_pure(depth in 1usize..5, base in small_int()) {
        let mut source = format_int(base);
        for _ in 0..depth {
            source = format!("({} + 1)", source);
        }

        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Nested pure additions should be pure");
    }

    /// Stress: Multiple resets are all pure
    #[test]
    fn stress_multiple_resets(n in 1usize..5, base in small_int()) {
        let mut source = format_int(base);
        for _ in 0..n {
            source = format!("reset ({})", source);
        }

        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Multiple resets should be pure");
    }

    /// Stress: Deep context with shift
    #[test]
    fn stress_deep_context_shift(depth in 1usize..4, base in small_positive()) {
        let mut source = format!("shift (fun k -> k {})", base);
        for _ in 0..depth {
            source = format!("1 + {}", source);
        }
        source = format!("reset ({})", source);

        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Reset should make deep context pure");
        prop_assert!(matches!(result.ty.resolve(), Type::Int));
    }

    /// Stress: Pure functions applied to pure arguments
    /// NOTE: Temporarily ignored during effect system migration (Arrow type changed)
    #[test]
    #[ignore]
    fn stress_pure_application_chain(n in 1usize..4, base in small_int()) {
        let mut source = format_int(base);
        for _ in 0..n {
            source = format!("(fun x -> x + 1) ({})", source);
        }

        let result = infer_full(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(is_pure(&result), "Chain of pure applications should be pure");
    }
}

// ============================================================================
// Category 8: Soundness - Type Preservation
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Well-typed shift/reset expressions should all type-check
    #[test]
    fn soundness_various_expressions(n in 1i64..20) {
        let cases = vec![
            format!("reset {}", n),
            format!("reset (shift (fun k -> k {}))", n),
            format!("reset (shift (fun k -> k {} + k {}))", n, n + 1),
            format!("reset ({} + shift (fun k -> k 1))", n),
            format!("reset (1 + reset ({} + shift (fun k -> k 2)))", n),
        ];

        for source in cases {
            let ty_result = infer_type(&source);
            prop_assert!(ty_result.is_ok(), "Should type-check: {} -> {:?}", source, ty_result);
        }
    }
}
