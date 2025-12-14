//! Property-based tests for Gneiss type system soundness
//!
//! These tests verify fundamental properties of the Hindley-Milner type inference:
//! - Unification symmetry and idempotence
//! - Occurs check correctness (prevents infinite types)
//! - Type preservation (well-typed programs evaluate to correctly-typed values)
//! - Progress (well-typed programs don't get stuck)
//! - Let-polymorphism correctness

use proptest::prelude::*;
use std::rc::Rc;

use gneiss::ast::*;
use gneiss::types::*;
use gneiss::infer::Inferencer;
use gneiss::{Lexer, Parser, Interpreter, Value};

// ============================================================================
// Type Generators
// ============================================================================

/// Generate a ground (monomorphic, no type variables) type
fn arb_ground_type(depth: usize) -> BoxedStrategy<Type> {
    if depth == 0 {
        prop_oneof![
            Just(Type::Int),
            Just(Type::Bool),
            Just(Type::String),
            Just(Type::Unit),
            Just(Type::Float),
            Just(Type::Char),
        ]
        .boxed()
    } else {
        prop_oneof![
            // Base types (weighted higher to avoid explosion)
            3 => Just(Type::Int),
            3 => Just(Type::Bool),
            3 => Just(Type::String),
            3 => Just(Type::Unit),
            // Compound types
            1 => arb_ground_type(depth - 1).prop_map(|t| Type::list(t)),
            1 => (arb_ground_type(depth - 1), arb_ground_type(depth - 1))
                .prop_map(|(a, b)| {
                    // Create a pure arrow type (same answer type for both)
                    let ans = Type::new_var(9999, 0);
                    Type::Arrow {
                        arg: Rc::new(a),
                        ret: Rc::new(b),
                        ans_in: Rc::new(ans.clone()),
                        ans_out: Rc::new(ans),
                    }
                }),
            1 => prop::collection::vec(arb_ground_type(depth - 1), 2..=3)
                .prop_map(Type::Tuple),
        ]
        .boxed()
    }
}

/// Generate a type that may contain fresh type variables using proptest strategies
/// Reserved for future unification tests with polymorphic types
#[allow(dead_code)]
fn arb_type_with_vars(depth: usize) -> BoxedStrategy<Type> {
    if depth == 0 {
        prop_oneof![
            Just(Type::Int),
            Just(Type::Bool),
            Just(Type::String),
            Just(Type::Unit),
            (0u32..100).prop_map(|id| Type::new_var(id, 0)),
        ]
        .boxed()
    } else {
        prop_oneof![
            // Higher weight for base types
            3 => Just(Type::Int),
            3 => Just(Type::Bool),
            2 => (0u32..100).prop_map(|id| Type::new_var(id, 0)),
            // Compound types
            1 => arb_type_with_vars(depth - 1).prop_map(|t| Type::list(t)),
            1 => (arb_type_with_vars(depth - 1), arb_type_with_vars(depth - 1))
                .prop_map(|(a, b)| {
                    // Create a pure arrow type (same answer type for both)
                    let ans = Type::new_var(9999, 0);
                    Type::Arrow {
                        arg: Rc::new(a),
                        ret: Rc::new(b),
                        ans_in: Rc::new(ans.clone()),
                        ans_out: Rc::new(ans),
                    }
                }),
        ]
        .boxed()
    }
}

// ============================================================================
// AST Generators
// ============================================================================

/// Generate an arbitrary literal
/// Reserved for future AST-based property tests
#[allow(dead_code)]
fn arb_literal() -> BoxedStrategy<Literal> {
    prop_oneof![
        any::<i64>().prop_map(Literal::Int),
        any::<bool>().prop_map(Literal::Bool),
        "[a-zA-Z0-9 ]{0,10}".prop_map(Literal::String),
        Just(Literal::Unit),
    ]
    .boxed()
}

/// Create a spanned AST node with a dummy span
/// Reserved for future AST-based property tests
#[allow(dead_code)]
fn spanned<T>(node: T) -> Spanned<T> {
    Spanned::new(node, Span::default())
}

/// Generate well-scoped expressions
/// The `env` parameter contains variables that are currently in scope
/// Reserved for future AST-based property tests
#[allow(dead_code)]
fn arb_expr(env: Vec<String>, depth: usize) -> BoxedStrategy<Expr> {
    if depth == 0 {
        // Base case: literals or variables in scope
        if env.is_empty() {
            arb_literal()
                .prop_map(|lit| spanned(ExprKind::Lit(lit)))
                .boxed()
        } else {
            prop_oneof![
                arb_literal().prop_map(|lit| spanned(ExprKind::Lit(lit))),
                proptest::sample::select(env.clone())
                    .prop_map(|name| spanned(ExprKind::Var(name))),
            ]
            .boxed()
        }
    } else {
        let env_clone = env.clone();
        let env_clone2 = env.clone();
        let env_clone3 = env.clone();
        let env_clone4 = env.clone();

        prop_oneof![
            // Literal (base case, higher weight)
            3 => arb_literal().prop_map(|lit| spanned(ExprKind::Lit(lit))),

            // Variable reference (if any in scope)
            2 => {
                if env.is_empty() {
                    arb_literal().prop_map(|lit| spanned(ExprKind::Lit(lit))).boxed()
                } else {
                    proptest::sample::select(env.clone())
                        .prop_map(|name| spanned(ExprKind::Var(name)))
                        .boxed()
                }
            },

            // Lambda - introduces a new binding
            1 => {
                let param_name = format!("x{}", depth);
                let mut new_env = env_clone.clone();
                new_env.push(param_name.clone());
                arb_expr(new_env, depth - 1).prop_map(move |body| {
                    spanned(ExprKind::Lambda {
                        params: vec![spanned(PatternKind::Var(param_name.clone()))],
                        body: Rc::new(body),
                    })
                })
            },

            // Application
            1 => {
                (arb_expr(env_clone2.clone(), depth - 1), arb_expr(env_clone2, depth - 1))
                    .prop_map(|(func, arg)| {
                        spanned(ExprKind::App {
                            func: Rc::new(func),
                            arg: Rc::new(arg),
                        })
                    })
            },

            // Let binding - introduces a new binding for the body
            1 => {
                let bind_name = format!("v{}", depth);
                let mut new_env = env_clone3.clone();
                new_env.push(bind_name.clone());
                (arb_expr(env_clone3, depth - 1), arb_expr(new_env, depth - 1))
                    .prop_map(move |(value, body)| {
                        spanned(ExprKind::Let {
                            pattern: spanned(PatternKind::Var(bind_name.clone())),
                            value: Rc::new(value),
                            body: Some(Rc::new(body)),
                        })
                    })
            },

            // If expression
            1 => {
                (
                    arb_expr(env_clone4.clone(), depth - 1),
                    arb_expr(env_clone4.clone(), depth - 1),
                    arb_expr(env_clone4, depth - 1),
                )
                    .prop_map(|(cond, then_branch, else_branch)| {
                        spanned(ExprKind::If {
                            cond: Rc::new(cond),
                            then_branch: Rc::new(then_branch),
                            else_branch: Rc::new(else_branch),
                        })
                    })
            },

            // Binary operations (arithmetic)
            1 => {
                (arb_expr(env.clone(), depth - 1), arb_expr(env.clone(), depth - 1))
                    .prop_map(|(left, right)| {
                        spanned(ExprKind::BinOp {
                            op: BinOp::Add,
                            left: Rc::new(left),
                            right: Rc::new(right),
                        })
                    })
            },
        ]
        .boxed()
    }
}

/// Generate a small integer that won't overflow on arithmetic
fn small_int() -> BoxedStrategy<i64> {
    (-1000i64..1000).boxed()
}

/// Generate well-typed expressions by parsing valid Gneiss source
/// This is more reliable than AST generation for type preservation tests
fn arb_well_typed_source() -> BoxedStrategy<String> {
    prop_oneof![
        // Simple literals (use small ints to avoid overflow)
        small_int().prop_map(|n| if n < 0 { format!("({})", n) } else { n.to_string() }),
        any::<bool>().prop_map(|b| b.to_string()),
        Just("()".to_string()),

        // Arithmetic (use small ints to avoid overflow)
        (small_int(), small_int()).prop_map(|(a, b)| {
            let a_str = if a < 0 { format!("({})", a) } else { a.to_string() };
            let b_str = if b < 0 { format!("({})", b) } else { b.to_string() };
            format!("{} + {}", a_str, b_str)
        }),
        (small_int(), small_int()).prop_map(|(a, b)| {
            let a_str = if a < 0 { format!("({})", a) } else { a.to_string() };
            let b_str = if b < 0 { format!("({})", b) } else { b.to_string() };
            format!("{} * {}", a_str, b_str)
        }),

        // Comparisons
        (small_int(), small_int()).prop_map(|(a, b)| {
            let a_str = if a < 0 { format!("({})", a) } else { a.to_string() };
            let b_str = if b < 0 { format!("({})", b) } else { b.to_string() };
            format!("{} == {}", a_str, b_str)
        }),
        (small_int(), small_int()).prop_map(|(a, b)| {
            let a_str = if a < 0 { format!("({})", a) } else { a.to_string() };
            let b_str = if b < 0 { format!("({})", b) } else { b.to_string() };
            format!("{} < {}", a_str, b_str)
        }),

        // Boolean operations
        (any::<bool>(), any::<bool>()).prop_map(|(a, b)| format!("{} && {}", a, b)),
        (any::<bool>(), any::<bool>()).prop_map(|(a, b)| format!("{} || {}", a, b)),

        // If expressions with matching branches
        (any::<bool>(), small_int(), small_int())
            .prop_map(|(c, t, e)| {
                let t_str = if t < 0 { format!("({})", t) } else { t.to_string() };
                let e_str = if e < 0 { format!("({})", e) } else { e.to_string() };
                format!("if {} then {} else {}", c, t_str, e_str)
            }),

        // Lambda and application (use non-negative for simplicity)
        (0i64..100).prop_map(|n| format!("(fun x -> x + 1) {}", n)),
        (0i64..100).prop_map(|n| format!("(fun x -> x) {}", n)),

        // Let bindings
        (small_int(), small_int()).prop_map(|(a, b)| {
            let a_str = if a < 0 { format!("({})", a) } else { a.to_string() };
            let b_str = if b < 0 { format!("({})", b) } else { b.to_string() };
            format!("let x = {} in x + {}", a_str, b_str)
        }),

        // Tuples
        (small_int(), any::<bool>()).prop_map(|(a, b)| {
            let a_str = if a < 0 { format!("({})", a) } else { a.to_string() };
            format!("({}, {})", a_str, b)
        }),

        // Lists
        Just("[]".to_string()),
        (0i64..100).prop_map(|n| format!("[{}]", n)),
        (0i64..100, 0i64..100).prop_map(|(a, b)| format!("[{}, {}]", a, b)),

        // Nested let with polymorphism
        Just("let id = fun x -> x in (id 42, id true)".to_string()),

        // Function composition via let
        Just("let f = fun x -> x + 1 in let g = fun x -> x * 2 in f (g 3)".to_string()),
    ]
    .boxed()
}

// ============================================================================
// Property Tests: Unification
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Unification should be symmetric: unify(a, b) succeeds iff unify(b, a) succeeds
    #[test]
    fn unify_symmetric(t1 in arb_ground_type(2), t2 in arb_ground_type(2)) {
        let mut inf1 = Inferencer::new();
        let mut inf2 = Inferencer::new();

        let r1 = inf1.unify_types(&t1, &t2);
        let r2 = inf2.unify_types(&t2, &t1);

        prop_assert_eq!(r1.is_ok(), r2.is_ok(),
            "unify({}, {}) = {:?} but unify({}, {}) = {:?}",
            t1, t2, r1, t2, t1, r2);
    }

    /// Unifying a type with itself should always succeed
    #[test]
    fn unify_reflexive(t in arb_ground_type(3)) {
        let mut inf = Inferencer::new();
        let result = inf.unify_types(&t, &t);
        prop_assert!(result.is_ok(), "unify({}, {}) failed: {:?}", t, t, result);
    }

    /// After successful unification, both types should resolve to the same type
    #[test]
    fn unify_produces_equal_types(t1 in arb_ground_type(2), t2 in arb_ground_type(2)) {
        let mut inf = Inferencer::new();
        if inf.unify_types(&t1, &t2).is_ok() {
            let resolved1 = t1.resolve();
            let resolved2 = t2.resolve();
            // For ground types, they should be structurally equal after unification
            prop_assert_eq!(format!("{}", resolved1), format!("{}", resolved2));
        }
    }
}

// ============================================================================
// Property Tests: Occurs Check
// ============================================================================

proptest! {
    /// The occurs check should detect when a type variable appears in a type
    #[test]
    fn occurs_check_detects_self_reference(var_id in 0u32..100) {
        let var = Type::new_var(var_id, 0);

        // The variable should occur in itself
        prop_assert!(var.occurs(var_id),
            "Type variable t{} should occur in itself", var_id);

        // The variable should occur in a function type containing it
        let ans = Type::new_var(9999, 0);
        let arrow = Type::Arrow {
            arg: Rc::new(var.clone()),
            ret: Rc::new(Type::Int),
            ans_in: Rc::new(ans.clone()),
            ans_out: Rc::new(ans),
        };
        prop_assert!(arrow.occurs(var_id),
            "Type variable t{} should occur in {} -> Int", var_id, var);
    }

    /// A type variable should not occur in unrelated ground types
    #[test]
    fn occurs_check_negative(var_id in 0u32..100, t in arb_ground_type(2)) {
        // Ground types have no type variables, so occurs should be false
        prop_assert!(!t.occurs(var_id),
            "Type variable t{} should not occur in ground type {}", var_id, t);
    }
}

// ============================================================================
// Property Tests: Type Inference
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Type inference should be deterministic
    #[test]
    fn inference_deterministic(source in arb_well_typed_source()) {
        let result1 = infer_source(&source);
        let result2 = infer_source(&source);

        match (&result1, &result2) {
            (Ok(t1), Ok(t2)) => {
                prop_assert_eq!(format!("{}", t1), format!("{}", t2),
                    "Inference of '{}' gave different types: {} vs {}", source, t1, t2);
            }
            (Err(_), Err(_)) => { /* Both failed, that's consistent */ }
            _ => {
                prop_assert!(false,
                    "Inference of '{}' was inconsistent: {:?} vs {:?}",
                    source, result1, result2);
            }
        }
    }

    /// Well-formed expressions should type-check without panicking
    #[test]
    fn inference_does_not_panic(source in arb_well_typed_source()) {
        // This test just checks we don't panic - the result can be Ok or Err
        let _ = infer_source(&source);
    }
}

// ============================================================================
// Property Tests: Type Preservation
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Type preservation: if an expression type-checks and evaluates,
    /// the result should have the expected type
    #[test]
    fn type_preservation(source in arb_well_typed_source()) {
        if let Ok(ty) = infer_source(&source) {
            if let Ok(val) = eval_source(&source) {
                prop_assert!(value_matches_type(&val, &ty),
                    "Expression '{}' has type {} but evaluated to {:?}",
                    source, ty, val);
            }
            // If eval fails, that's a runtime error (division by zero, etc.), not a type error
        }
    }
}

// ============================================================================
// Property Tests: Progress
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Progress: well-typed expressions should either:
    /// 1. Evaluate to a value, OR
    /// 2. Produce a well-defined runtime error (not panic/get stuck)
    #[test]
    fn progress(source in arb_well_typed_source()) {
        if infer_source(&source).is_ok() {
            // Should not panic - may return Ok(value) or Err(runtime_error)
            let result = std::panic::catch_unwind(|| eval_source(&source));
            prop_assert!(result.is_ok(),
                "Well-typed expression '{}' caused a panic", source);
        }
    }
}

// ============================================================================
// Property Tests: Let-Polymorphism
// ============================================================================

proptest! {
    /// The identity function should be polymorphic
    #[test]
    fn identity_is_polymorphic(n in 0i64..1000, b in any::<bool>()) {
        let source = format!(
            "let id = fun x -> x in (id {}, id {})",
            n, b
        );

        let result = infer_source(&source);
        prop_assert!(result.is_ok(),
            "Polymorphic identity should type-check: {:?}", result);

        if let Ok(ty) = result {
            // Should be a tuple type
            let ty_str = format!("{}", ty);
            prop_assert!(ty_str.contains("(") && ty_str.contains(")"),
                "Result should be a tuple, got: {}", ty_str);
        }
    }

    /// Const function should be polymorphic
    #[test]
    fn const_is_polymorphic(n in 0i64..1000, b in any::<bool>()) {
        let source = format!(
            "let const = fun x -> fun y -> x in const {} {}",
            n, b
        );

        let result = infer_source(&source);
        prop_assert!(result.is_ok(),
            "Polymorphic const should type-check: {:?}", result);

        // The result should be Int (the type of the first argument)
        if let Ok(ty) = result {
            prop_assert_eq!(format!("{}", ty), "Int");
        }
    }
}

// ============================================================================
// Specific Regression Tests
// ============================================================================

#[test]
fn test_occurs_check_prevents_infinite_type() {
    // This should fail the occurs check: trying to unify a with a -> b
    let source = "fun x -> x x";
    let result = infer_source(source);
    assert!(result.is_err(), "Self-application should fail occurs check");
}

#[test]
fn test_let_polymorphism_basic() {
    let source = "let id = fun x -> x in (id 1, id true)";
    let result = infer_source(source);
    assert!(result.is_ok(), "Let-polymorphism should allow id to be used at multiple types");
}

#[test]
fn test_value_restriction() {
    // If we had mutable references, this would be the classic unsoundness.
    // With channels, we need to be careful about polymorphism.
    // For now, just test that functions are properly generalized.
    let source = "let f = fun x -> x in f";
    let result = infer_source(source);
    assert!(result.is_ok());
}

#[test]
fn test_nested_let_polymorphism() {
    let source = "let id = fun x -> x in let apply = fun f -> fun x -> f x in apply id 42";
    let result = infer_source(source);
    assert!(result.is_ok(), "Nested let-polymorphism should work: {:?}", result);
}

#[test]
fn test_char_comparison_operators() {
    // Comparison operators should work on Char, not just Int
    let source = "fun c -> c >= '0' && c <= '9'";
    let result = infer_source(source);
    assert!(result.is_ok(), "Char comparison should type-check: {:?}", result);

    // The result should be Char -> Bool
    if let Ok(ty) = result {
        let ty_str = format!("{}", ty);
        assert!(ty_str.contains("Char") && ty_str.contains("Bool"),
            "Expected Char -> Bool, got {}", ty_str);
    }
}

#[test]
fn test_sequence_in_match_arm() {
    // Sequences (semicolons) should be allowed in match arm bodies
    let source = r#"
fun x ->
    match x with
    | 1 -> (); 100
    | 2 -> 200
"#;
    let result = infer_source(source);
    assert!(result.is_ok(), "Sequence in match arm should type-check: {:?}", result);
}

#[test]
fn test_nested_match_with_sequences() {
    // Nested matches with sequences should parse correctly
    let source = r#"
fun x y ->
    match x with
    | 1 ->
        match y with
        | 10 -> (); 1
        | 20 -> (); 2
    | 2 -> 0
"#;
    let result = infer_source(source);
    assert!(result.is_ok(), "Nested match with sequences should type-check: {:?}", result);
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse and type-check a source string
fn infer_source(source: &str) -> Result<Type, String> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lexer error: {}", e))?;

    let mut parser = Parser::new(tokens);
    let expr = parser
        .parse_expr()
        .map_err(|e| format!("Parser error: {}", e))?;

    let mut inferencer = Inferencer::new();
    let env = TypeEnv::new();
    inferencer
        .infer_expr(&env, &expr)
        .map_err(|e| format!("Type error: {}", e))
}

/// Parse and evaluate a source string
fn eval_source(source: &str) -> Result<Value, String> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lexer error: {}", e))?;

    let mut parser = Parser::new(tokens);
    let expr = parser
        .parse_expr()
        .map_err(|e| format!("Parser error: {}", e))?;

    let mut interpreter = Interpreter::new();
    let env = gneiss::eval::EnvInner::new();
    interpreter
        .eval_expr(&env, &expr)
        .map_err(|e| format!("Eval error: {}", e))
}

/// Check if a runtime value matches a static type
fn value_matches_type(val: &Value, ty: &Type) -> bool {
    match (val, ty.resolve()) {
        (Value::Int(_), Type::Int) => true,
        (Value::Float(_), Type::Float) => true,
        (Value::Bool(_), Type::Bool) => true,
        (Value::String(_), Type::String) => true,
        (Value::Char(_), Type::Char) => true,
        (Value::Unit, Type::Unit) => true,
        (Value::Closure { .. }, Type::Arrow { .. }) => true,
        (Value::List(items), Type::Constructor { name, args }) if name == "List" && args.len() == 1 => {
            items.iter().all(|item| value_matches_type(item, &args[0]))
        }
        (Value::Tuple(items), Type::Tuple(types)) => {
            items.len() == types.len()
                && items.iter().zip(types.iter()).all(|(v, t)| value_matches_type(v, t))
        }
        (Value::Constructor { name, fields }, Type::Constructor { name: ty_name, args: _ }) => {
            // For now, just check constructor names match
            // Full check would require looking up constructor info
            name == &ty_name || fields.is_empty() // Simple heuristic
        }
        (Value::Pid(_), Type::Pid) => true,
        (Value::Channel(_), Type::Channel(_)) => true,
        // Type variables can match anything (they're polymorphic)
        (_, Type::Var(_)) => true,
        _ => false,
    }
}
