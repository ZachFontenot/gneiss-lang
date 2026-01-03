//! Type Inference Verification Tests
//!
//! These tests verify that the type inferencer correctly records types for ALL
//! expressions, not just top-level ones. This catches bugs where sub-expressions
//! silently get default types (like Unit) due to missing type recording.
//!
//! Lesson learned: The polymorphic `==` bug hid because we only tested final
//! outputs, not intermediate correctness.

use gneiss::ast::{Expr, ExprKind, Span};
use gneiss::infer::Inferencer;
use gneiss::lexer::Lexer;
use gneiss::parser::Parser;
use gneiss::types::{Type, TypeEnv};

// ============================================================================
// Test Helpers
// ============================================================================

/// Parse source and run type inference, returning both the inferencer state
/// and the parsed expression for inspection.
fn infer_expr_with_state(source: &str) -> Result<(Inferencer, Expr), String> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lex error: {:?}", e))?;

    let expr = Parser::new(tokens)
        .parse_expr()
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let mut inferencer = Inferencer::new();
    let env = TypeEnv::new();
    inferencer
        .infer_expr(&env, &expr)
        .map_err(|e| format!("Type error: {:?}", e))?;

    Ok((inferencer, expr))
}

/// Parse a program and run type inference, returning the inferencer state.
#[allow(dead_code)]
fn infer_program_with_state(source: &str) -> Result<Inferencer, String> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lex error: {:?}", e))?;

    let program = Parser::new(tokens)
        .parse_program()
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let mut inferencer = Inferencer::new();
    inferencer
        .infer_program(&program, TypeEnv::new())
        .map_err(|e| format!("Type error: {:?}", e))?;

    Ok(inferencer)
}

/// Collect all expression spans from an AST expression.
fn collect_expr_spans(expr: &Expr) -> Vec<Span> {
    let mut spans = Vec::new();
    collect_expr_spans_recursive(expr, &mut spans);
    spans
}

fn collect_expr_spans_recursive(expr: &Expr, spans: &mut Vec<Span>) {
    spans.push(expr.span.clone());

    match &expr.node {
        ExprKind::Lit(_) => {}
        ExprKind::Var(_) => {}

        ExprKind::Lambda { params: _, body } => {
            collect_expr_spans_recursive(body, spans);
        }

        ExprKind::App { func, arg } => {
            collect_expr_spans_recursive(func, spans);
            collect_expr_spans_recursive(arg, spans);
        }

        ExprKind::Let {
            pattern: _,
            value,
            body,
        } => {
            collect_expr_spans_recursive(value, spans);
            if let Some(body) = body {
                collect_expr_spans_recursive(body, spans);
            }
        }

        ExprKind::LetRec { bindings, body } => {
            for binding in bindings {
                collect_expr_spans_recursive(&binding.body, spans);
            }
            if let Some(body) = body {
                collect_expr_spans_recursive(body, spans);
            }
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_expr_spans_recursive(cond, spans);
            collect_expr_spans_recursive(then_branch, spans);
            collect_expr_spans_recursive(else_branch, spans);
        }

        ExprKind::BinOp { left, right, .. } => {
            collect_expr_spans_recursive(left, spans);
            collect_expr_spans_recursive(right, spans);
        }

        ExprKind::UnaryOp { operand, .. } => {
            collect_expr_spans_recursive(operand, spans);
        }

        ExprKind::Match { scrutinee, arms } => {
            collect_expr_spans_recursive(scrutinee, spans);
            for arm in arms {
                collect_expr_spans_recursive(&arm.body, spans);
                if let Some(guard) = &arm.guard {
                    collect_expr_spans_recursive(guard, spans);
                }
            }
        }

        ExprKind::Tuple(elements) => {
            for elem in elements {
                collect_expr_spans_recursive(elem, spans);
            }
        }

        ExprKind::List(elements) => {
            for elem in elements {
                collect_expr_spans_recursive(elem, spans);
            }
        }

        ExprKind::Record { fields, .. } => {
            for (_, value) in fields {
                collect_expr_spans_recursive(value, spans);
            }
        }

        ExprKind::RecordUpdate { base, updates } => {
            collect_expr_spans_recursive(base, spans);
            for (_, value) in updates {
                collect_expr_spans_recursive(value, spans);
            }
        }

        ExprKind::FieldAccess { record, .. } => {
            collect_expr_spans_recursive(record, spans);
        }

        ExprKind::Seq { first, second } => {
            collect_expr_spans_recursive(first, spans);
            collect_expr_spans_recursive(second, spans);
        }

        ExprKind::Perform { args, .. } => {
            for arg in args {
                collect_expr_spans_recursive(arg, spans);
            }
        }

        ExprKind::Handle {
            body,
            return_clause,
            handlers,
        } => {
            collect_expr_spans_recursive(body, spans);
            collect_expr_spans_recursive(&return_clause.body, spans);
            for handler in handlers {
                collect_expr_spans_recursive(&handler.body, spans);
            }
        }

        ExprKind::Constructor { args, .. } => {
            for arg in args {
                collect_expr_spans_recursive(arg, spans);
            }
        }

        // Concurrency primitives
        ExprKind::Spawn(expr) => {
            collect_expr_spans_recursive(expr, spans);
        }

        ExprKind::NewChannel => {}

        ExprKind::ChanSend { channel, value } => {
            collect_expr_spans_recursive(channel, spans);
            collect_expr_spans_recursive(value, spans);
        }

        ExprKind::ChanRecv(expr) => {
            collect_expr_spans_recursive(expr, spans);
        }

        ExprKind::Select { arms } => {
            for arm in arms {
                collect_expr_spans_recursive(&arm.channel, spans);
                collect_expr_spans_recursive(&arm.body, spans);
            }
        }

        ExprKind::Hole => {}
    }
}

/// Check if a type is suspiciously Unit when it shouldn't be.
/// Returns true if the expression is something that should NOT be Unit.
#[allow(dead_code)]
fn is_suspicious_unit(expr: &Expr, ty: &Type) -> bool {
    if !matches!(ty, Type::Unit) {
        return false;
    }

    // These expressions legitimately return Unit
    match &expr.node {
        ExprKind::Lit(gneiss::ast::Literal::Unit) => false,
        ExprKind::Perform { .. } => false, // Effects can return Unit
        ExprKind::Seq { .. } => false,     // Seq type is determined by second expr
        // Everything else returning Unit is suspicious unless proven otherwise
        _ => true,
    }
}

// ============================================================================
// Core Invariant Tests
// ============================================================================

#[test]
fn all_subexpressions_have_recorded_types() {
    // Test that the inferencer records types for ALL sub-expressions
    let source = "let f x y = x + y in f 1 2";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");
    let spans = collect_expr_spans(&expr);

    let mut missing = Vec::new();
    for span in &spans {
        if inferencer.get_expr_type(span).is_none() {
            missing.push(span.clone());
        }
    }

    assert!(
        missing.is_empty(),
        "Missing types for {} spans: {:?}",
        missing.len(),
        missing
    );
}

#[test]
fn binop_operands_have_correct_types() {
    // The bug that motivated this: x == y in a polymorphic function had Unit operands
    let source = "fun x y -> x + y";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");

    // Find the BinOp
    fn find_binop(expr: &Expr) -> Option<(&Expr, &Expr)> {
        match &expr.node {
            ExprKind::BinOp { left, right, .. } => Some((left.as_ref(), right.as_ref())),
            ExprKind::Lambda { body, .. } => find_binop(body),
            ExprKind::Let { body, .. } => body.as_ref().and_then(|b| find_binop(b)),
            _ => None,
        }
    }

    let (left, right) = find_binop(&expr).expect("should find binop");

    let left_ty = inferencer
        .get_expr_type(&left.span)
        .expect("left operand should have type");
    let right_ty = inferencer
        .get_expr_type(&right.span)
        .expect("right operand should have type");

    // They should be type variables (polymorphic), not Unit
    assert!(
        !matches!(left_ty, Type::Unit),
        "Left operand type should not be Unit, got: {}",
        left_ty
    );
    assert!(
        !matches!(right_ty, Type::Unit),
        "Right operand type should not be Unit, got: {}",
        right_ty
    );
}

#[test]
fn polymorphic_equality_operands_are_type_vars() {
    // This is the specific case that broke: x == y in assert_eq
    let source = "fun x y -> x == y";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");

    fn find_binop(expr: &Expr) -> Option<(&Expr, &Expr)> {
        match &expr.node {
            ExprKind::BinOp { left, right, .. } => Some((left.as_ref(), right.as_ref())),
            ExprKind::Lambda { body, .. } => find_binop(body),
            _ => None,
        }
    }

    let (left, right) = find_binop(&expr).expect("should find binop");

    let left_ty = inferencer
        .get_expr_type(&left.span)
        .expect("left should have type");
    let right_ty = inferencer
        .get_expr_type(&right.span)
        .expect("right should have type");

    // Both should be type variables since this is a polymorphic function
    fn is_type_var_or_generic(ty: &Type) -> bool {
        matches!(ty, Type::Var(_) | Type::Generic(_))
    }

    assert!(
        is_type_var_or_generic(&left_ty),
        "Left operand should be type var, got: {}",
        left_ty
    );
    assert!(
        is_type_var_or_generic(&right_ty),
        "Right operand should be type var, got: {}",
        right_ty
    );
}

#[test]
fn nested_let_bindings_all_have_types() {
    let source = r#"
        let a = 1 in
        let b = 2 in
        let c = a + b in
        c * 2
    "#;

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");
    let spans = collect_expr_spans(&expr);

    for span in &spans {
        let ty = inferencer
            .get_expr_type(span)
            .expect(&format!("span {:?} should have type", span));

        // None of these should be Unit (all are Int arithmetic)
        assert!(
            !matches!(ty, Type::Unit),
            "Expression at {:?} has suspicious Unit type",
            span
        );
    }
}

#[test]
fn lambda_body_has_correct_type() {
    let source = "fun x -> x + 1";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");

    // Get the lambda body type
    let body = match &expr.node {
        ExprKind::Lambda { body, .. } => body.as_ref(),
        _ => panic!("expected lambda"),
    };

    let body_ty = inferencer
        .get_expr_type(&body.span)
        .expect("body should have type");

    // Body should be Int, not Unit
    assert!(
        matches!(body_ty, Type::Int),
        "Lambda body should be Int, got: {}",
        body_ty
    );
}

#[test]
fn function_application_all_parts_typed() {
    let source = "let f x = x + 1 in f 5";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");
    let spans = collect_expr_spans(&expr);

    let mut suspicious = Vec::new();
    for span in &spans {
        if let Some(ty) = inferencer.get_expr_type(span) {
            if matches!(ty, Type::Unit) {
                suspicious.push((span.clone(), "Unit".to_string()));
            }
        } else {
            suspicious.push((span.clone(), "MISSING".to_string()));
        }
    }

    assert!(
        suspicious.is_empty(),
        "Found suspicious types: {:?}",
        suspicious
    );
}

#[test]
fn if_expression_branches_typed() {
    let source = "if true then 1 else 2";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");

    let (cond, then_br, else_br) = match &expr.node {
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => (cond.as_ref(), then_branch.as_ref(), else_branch.as_ref()),
        _ => panic!("expected if"),
    };

    let cond_ty = inferencer
        .get_expr_type(&cond.span)
        .expect("cond should have type");
    let then_ty = inferencer
        .get_expr_type(&then_br.span)
        .expect("then should have type");
    let else_ty = inferencer
        .get_expr_type(&else_br.span)
        .expect("else should have type");

    assert!(
        matches!(cond_ty, Type::Bool),
        "Condition should be Bool, got: {}",
        cond_ty
    );
    assert!(
        matches!(then_ty, Type::Int),
        "Then branch should be Int, got: {}",
        then_ty
    );
    assert!(
        matches!(else_ty, Type::Int),
        "Else branch should be Int, got: {}",
        else_ty
    );
}

#[test]
fn match_arm_bodies_typed() {
    // Use a simple list pattern since Option requires prelude
    let source = r#"
        match [1, 2, 3] with
        | x :: _ -> x + 1
        | [] -> 0
        end
    "#;

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");
    let spans = collect_expr_spans(&expr);

    // All spans should have recorded types
    for span in &spans {
        assert!(
            inferencer.get_expr_type(span).is_some(),
            "Span {:?} should have recorded type",
            span
        );
    }
}

#[test]
fn higher_order_function_all_typed() {
    let source = "let apply f x = f x in apply (fun y -> y + 1) 5";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");
    let spans = collect_expr_spans(&expr);

    let mut missing = Vec::new();
    let mut unit_count = 0;

    for span in &spans {
        match inferencer.get_expr_type(span) {
            Some(ty) if matches!(ty, Type::Unit) => unit_count += 1,
            None => missing.push(span.clone()),
            _ => {}
        }
    }

    // This expression has no Unit anywhere
    assert_eq!(
        unit_count, 0,
        "Should have no Unit types in this expression"
    );
    assert!(missing.is_empty(), "Missing types for: {:?}", missing);
}

// ============================================================================
// Regression Tests for Specific Bugs
// ============================================================================

#[test]
fn regression_poly_eq_operands_not_unit() {
    // This is exactly the bug we fixed: assert_eq's x == y had Unit operands
    // because sub-expression types weren't recorded
    let source = r#"
        let check x y = x == y in
        check "a" "b"
    "#;

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");

    // Find all BinOps and check their operand types
    fn find_all_binops<'a>(expr: &'a Expr, results: &mut Vec<(&'a Expr, &'a Expr)>) {
        match &expr.node {
            ExprKind::BinOp { left, right, .. } => {
                results.push((left.as_ref(), right.as_ref()));
                find_all_binops(left, results);
                find_all_binops(right, results);
            }
            ExprKind::Lambda { body, .. } => find_all_binops(body, results),
            ExprKind::Let { value, body, .. } => {
                find_all_binops(value, results);
                if let Some(b) = body {
                    find_all_binops(b, results);
                }
            }
            ExprKind::App { func, arg } => {
                find_all_binops(func, results);
                find_all_binops(arg, results);
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                find_all_binops(cond, results);
                find_all_binops(then_branch, results);
                find_all_binops(else_branch, results);
            }
            _ => {}
        }
    }

    let mut binops = Vec::new();
    find_all_binops(&expr, &mut binops);

    for (left, right) in binops {
        let left_ty = inferencer.get_expr_type(&left.span);
        let right_ty = inferencer.get_expr_type(&right.span);

        assert!(
            left_ty.is_some(),
            "Left operand at {:?} missing type",
            left.span
        );
        assert!(
            right_ty.is_some(),
            "Right operand at {:?} missing type",
            right.span
        );

        if let Some(ty) = left_ty {
            assert!(
                !matches!(ty, Type::Unit),
                "Left operand should not be Unit, got: {}",
                ty
            );
        }
        if let Some(ty) = right_ty {
            assert!(
                !matches!(ty, Type::Unit),
                "Right operand should not be Unit, got: {}",
                ty
            );
        }
    }
}

// ============================================================================
// Type Variable Preservation Tests
// ============================================================================

#[test]
fn type_vars_preserved_in_polymorphic_context() {
    // In a polymorphic function, the parameter types should be type variables
    let source = "fun x -> x";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");

    // The body (just `x`) should have a type variable type
    let body = match &expr.node {
        ExprKind::Lambda { body, .. } => body.as_ref(),
        _ => panic!("expected lambda"),
    };

    let body_ty = inferencer
        .get_expr_type(&body.span)
        .expect("body should have type");

    // Identity function body should be a type variable
    fn is_type_var(ty: &Type) -> bool {
        matches!(ty, Type::Var(_) | Type::Generic(_))
    }

    assert!(
        is_type_var(&body_ty),
        "Identity body should be type var, got: {}",
        body_ty
    );
}

#[test]
fn specialized_call_has_concrete_type() {
    // When we call a polymorphic function with concrete args, the result should be concrete
    let source = "let id x = x in id 42";

    let (inferencer, expr) = infer_expr_with_state(source).expect("should type-check");

    // The overall type should be Int
    let overall_ty = inferencer
        .get_expr_type(&expr.span)
        .expect("should have type");

    assert!(
        matches!(overall_ty, Type::Int),
        "id 42 should have type Int, got: {}",
        overall_ty
    );
}
