//! AST Structure Tests - Verify Parser Produces Correct AST
//!
//! These tests verify that the parser produces the expected AST structure.
//! This is Layer 1 of the testing pyramid - the foundation.
//!
//! Categories:
//! 1. Expression structure - Let, Lambda, Application, etc.
//! 2. Pattern structure - Variables, Constructors, Tuples, etc.
//! 3. Declaration structure - Type declarations, Traits, Impls
//! 4. Operator precedence - Verify correct associativity and precedence
//! 5. Complex expressions - Nested structures, pipelines, etc.

use gneiss::ast::{BinOp, ExprKind, PatternKind, UnaryOp};
use gneiss::test_support::parse_expr;

// ============================================================================
// Let Expression Structure
// ============================================================================

mod let_expressions {
    use super::*;

    #[test]
    fn simple_let_structure() {
        let expr = parse_expr("let x = 42 in x").unwrap();

        let ExprKind::Let {
            pattern,
            value,
            body,
            ..
        } = &expr.node
        else {
            panic!("expected Let, got {:?}", expr.node);
        };

        // Pattern should be variable x
        assert!(
            matches!(&pattern.node, PatternKind::Var(name) if name == "x"),
            "pattern should be Var(x), got {:?}",
            pattern.node
        );

        // Value should be literal 42
        assert!(
            matches!(&value.node, ExprKind::Lit(_)),
            "value should be Lit, got {:?}",
            value.node
        );

        // Body should exist and be variable x
        assert!(body.is_some(), "body should exist");
        let body = body.as_ref().unwrap();
        assert!(
            matches!(&body.node, ExprKind::Var(name) if name == "x"),
            "body should be Var(x), got {:?}",
            body.node
        );
    }

    #[test]
    fn let_with_function_params() {
        // `let add x y = x + y in ...` desugars to `let add = fun x y -> x + y in ...`
        let expr = parse_expr("let add x y = x + y in add 1 2").unwrap();

        let ExprKind::Let {
            pattern, value, ..
        } = &expr.node
        else {
            panic!("expected Let, got {:?}", expr.node);
        };

        // Pattern should be function name
        assert!(
            matches!(&pattern.node, PatternKind::Var(name) if name == "add"),
            "pattern should be Var(add)"
        );

        // Value should be a Lambda with 2 parameters
        let ExprKind::Lambda { params, .. } = &value.node else {
            panic!("value should be Lambda, got {:?}", value.node);
        };
        assert_eq!(params.len(), 2, "should have 2 parameters");
    }

    #[test]
    fn nested_let_structure() {
        let expr = parse_expr("let x = 1 in let y = 2 in x + y").unwrap();

        let ExprKind::Let { body, .. } = &expr.node else {
            panic!("expected outer Let");
        };

        let body = body.as_ref().unwrap();
        assert!(
            matches!(&body.node, ExprKind::Let { .. }),
            "inner should be Let, got {:?}",
            body.node
        );
    }

    #[test]
    fn let_rec_structure() {
        let expr = parse_expr("let rec f x = f (x - 1) in f 10").unwrap();

        // Recursive lets use the LetRec variant
        let ExprKind::LetRec { bindings, body } = &expr.node else {
            panic!("expected LetRec, got {:?}", expr.node);
        };

        assert_eq!(bindings.len(), 1, "should have 1 binding");
        assert!(body.is_some(), "should have body");
    }
}

// ============================================================================
// Lambda Expression Structure
// ============================================================================

mod lambda_expressions {
    use super::*;

    #[test]
    fn single_param_lambda() {
        let expr = parse_expr("fun x -> x + 1").unwrap();

        let ExprKind::Lambda { params, body } = &expr.node else {
            panic!("expected Lambda, got {:?}", expr.node);
        };

        assert_eq!(params.len(), 1, "should have 1 parameter");
        assert!(
            matches!(&params[0].node, PatternKind::Var(name) if name == "x"),
            "param should be x"
        );

        // Body should be BinOp
        assert!(
            matches!(&body.node, ExprKind::BinOp { .. }),
            "body should be BinOp"
        );
    }

    #[test]
    fn multi_param_lambda() {
        let expr = parse_expr("fun x y z -> x + y + z").unwrap();

        let ExprKind::Lambda { params, .. } = &expr.node else {
            panic!("expected Lambda");
        };

        assert_eq!(params.len(), 3, "should have 3 parameters");
    }

    #[test]
    fn nested_lambda() {
        let expr = parse_expr("fun x -> fun y -> x + y").unwrap();

        let ExprKind::Lambda { body, .. } = &expr.node else {
            panic!("expected outer Lambda");
        };

        assert!(
            matches!(&body.node, ExprKind::Lambda { .. }),
            "body should be inner Lambda"
        );
    }

    #[test]
    fn lambda_with_pattern() {
        let expr = parse_expr("fun (x, y) -> x + y").unwrap();

        let ExprKind::Lambda { params, .. } = &expr.node else {
            panic!("expected Lambda");
        };

        assert!(
            matches!(&params[0].node, PatternKind::Tuple(_)),
            "param should be tuple pattern"
        );
    }
}

// ============================================================================
// Application Structure
// ============================================================================

mod application {
    use super::*;

    #[test]
    fn simple_application() {
        let expr = parse_expr("f x").unwrap();

        let ExprKind::App { func, arg } = &expr.node else {
            panic!("expected App, got {:?}", expr.node);
        };

        assert!(
            matches!(&func.node, ExprKind::Var(name) if name == "f"),
            "func should be f"
        );
        assert!(
            matches!(&arg.node, ExprKind::Var(name) if name == "x"),
            "arg should be x"
        );
    }

    #[test]
    fn curried_application() {
        // f x y parses as ((f x) y)
        let expr = parse_expr("f x y").unwrap();

        let ExprKind::App { func, arg } = &expr.node else {
            panic!("expected outer App");
        };

        // arg should be y
        assert!(
            matches!(&arg.node, ExprKind::Var(name) if name == "y"),
            "outer arg should be y"
        );

        // func should be (f x)
        let ExprKind::App {
            func: inner_func,
            arg: inner_arg,
        } = &func.node
        else {
            panic!("expected inner App");
        };

        assert!(
            matches!(&inner_func.node, ExprKind::Var(name) if name == "f"),
            "inner func should be f"
        );
        assert!(
            matches!(&inner_arg.node, ExprKind::Var(name) if name == "x"),
            "inner arg should be x"
        );
    }

    #[test]
    fn application_of_lambda() {
        let expr = parse_expr("(fun x -> x + 1) 42").unwrap();

        let ExprKind::App { func, arg } = &expr.node else {
            panic!("expected App");
        };

        assert!(
            matches!(&func.node, ExprKind::Lambda { .. }),
            "func should be Lambda"
        );
        assert!(
            matches!(&arg.node, ExprKind::Lit(_)),
            "arg should be Lit"
        );
    }
}

// ============================================================================
// Binary Operator Structure and Precedence
// ============================================================================

mod operators {
    use super::*;

    #[test]
    fn multiplication_higher_than_addition() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        let expr = parse_expr("1 + 2 * 3").unwrap();

        let ExprKind::BinOp { op, left, right } = &expr.node else {
            panic!("expected BinOp");
        };

        assert!(matches!(op, BinOp::Add), "outer op should be Add");

        // Left should be 1
        assert!(
            matches!(&left.node, ExprKind::Lit(_)),
            "left should be literal"
        );

        // Right should be 2 * 3
        let ExprKind::BinOp {
            op: inner_op,
            left: inner_left,
            right: inner_right,
        } = &right.node
        else {
            panic!("right should be BinOp");
        };

        assert!(matches!(inner_op, BinOp::Mul), "inner op should be Mul");
        assert!(matches!(&inner_left.node, ExprKind::Lit(_)));
        assert!(matches!(&inner_right.node, ExprKind::Lit(_)));
    }

    #[test]
    fn left_associativity_of_addition() {
        // 1 + 2 + 3 should parse as (1 + 2) + 3
        let expr = parse_expr("1 + 2 + 3").unwrap();

        let ExprKind::BinOp { op, left, right } = &expr.node else {
            panic!("expected BinOp");
        };

        assert!(matches!(op, BinOp::Add), "outer op should be Add");

        // Right should be 3 (literal)
        assert!(
            matches!(&right.node, ExprKind::Lit(_)),
            "right should be literal 3"
        );

        // Left should be 1 + 2
        assert!(
            matches!(&left.node, ExprKind::BinOp { op: BinOp::Add, .. }),
            "left should be Add"
        );
    }

    #[test]
    fn comparison_lower_than_arithmetic() {
        // 1 + 2 < 3 + 4 should parse as (1 + 2) < (3 + 4)
        let expr = parse_expr("1 + 2 < 3 + 4").unwrap();

        let ExprKind::BinOp { op, left, right } = &expr.node else {
            panic!("expected BinOp");
        };

        assert!(matches!(op, BinOp::Lt), "outer op should be Lt");
        assert!(
            matches!(&left.node, ExprKind::BinOp { op: BinOp::Add, .. }),
            "left should be Add"
        );
        assert!(
            matches!(&right.node, ExprKind::BinOp { op: BinOp::Add, .. }),
            "right should be Add"
        );
    }

    #[test]
    fn logical_and_lower_than_comparison() {
        // 1 < 2 && 3 < 4 should parse correctly
        let expr = parse_expr("1 < 2 && 3 < 4").unwrap();

        let ExprKind::BinOp { op, left, right } = &expr.node else {
            panic!("expected BinOp");
        };

        assert!(matches!(op, BinOp::And), "outer op should be And");
        assert!(
            matches!(&left.node, ExprKind::BinOp { op: BinOp::Lt, .. }),
            "left should be Lt"
        );
        assert!(
            matches!(&right.node, ExprKind::BinOp { op: BinOp::Lt, .. }),
            "right should be Lt"
        );
    }

    #[test]
    fn unary_minus() {
        let expr = parse_expr("-42").unwrap();

        let ExprKind::UnaryOp { op, operand } = &expr.node else {
            panic!("expected UnaryOp, got {:?}", expr.node);
        };

        assert!(matches!(op, UnaryOp::Neg), "op should be Neg");
        assert!(
            matches!(&operand.node, ExprKind::Lit(_)),
            "operand should be Lit"
        );
    }

    #[test]
    fn cons_right_associative() {
        // 1 :: 2 :: [] should parse as 1 :: (2 :: [])
        let expr = parse_expr("1 :: 2 :: []").unwrap();

        let ExprKind::BinOp { op, left, right } = &expr.node else {
            panic!("expected BinOp");
        };

        assert!(matches!(op, BinOp::Cons), "outer op should be Cons");

        // Left should be 1
        assert!(
            matches!(&left.node, ExprKind::Lit(_)),
            "left should be literal 1"
        );

        // Right should be 2 :: []
        assert!(
            matches!(&right.node, ExprKind::BinOp { op: BinOp::Cons, .. }),
            "right should be Cons"
        );
    }
}

// ============================================================================
// If Expression Structure
// ============================================================================

mod if_expressions {
    use super::*;

    #[test]
    fn simple_if() {
        let expr = parse_expr("if true then 1 else 2").unwrap();

        let ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } = &expr.node
        else {
            panic!("expected If, got {:?}", expr.node);
        };

        assert!(
            matches!(&cond.node, ExprKind::Lit(_)),
            "condition should be Lit"
        );
        assert!(
            matches!(&then_branch.node, ExprKind::Lit(_)),
            "then should be Lit"
        );
        assert!(
            matches!(&else_branch.node, ExprKind::Lit(_)),
            "else should be Lit"
        );
    }

    #[test]
    fn nested_if_in_condition() {
        let expr = parse_expr("if if a then b else c then 1 else 2").unwrap();

        let ExprKind::If { cond, .. } = &expr.node else {
            panic!("expected If");
        };

        assert!(
            matches!(&cond.node, ExprKind::If { .. }),
            "condition should be nested If"
        );
    }
}

// ============================================================================
// Match Expression Structure
// ============================================================================

mod match_expressions {
    use super::*;

    #[test]
    fn simple_match() {
        let expr = parse_expr(
            r#"match x with
            | Some y -> y
            | None -> 0
            end"#,
        )
        .unwrap();

        let ExprKind::Match { scrutinee, arms } = &expr.node else {
            panic!("expected Match, got {:?}", expr.node);
        };

        assert!(
            matches!(&scrutinee.node, ExprKind::Var(name) if name == "x"),
            "scrutinee should be x"
        );
        assert_eq!(arms.len(), 2, "should have 2 arms");
    }

    #[test]
    fn match_arm_patterns() {
        let expr = parse_expr(
            r#"match x with
            | (a, b) -> a + b
            | _ -> 0
            end"#,
        )
        .unwrap();

        let ExprKind::Match { arms, .. } = &expr.node else {
            panic!("expected Match");
        };

        // First arm should be tuple pattern
        assert!(
            matches!(&arms[0].pattern.node, PatternKind::Tuple(_)),
            "first pattern should be Tuple"
        );

        // Second arm should be wildcard
        assert!(
            matches!(&arms[1].pattern.node, PatternKind::Wildcard),
            "second pattern should be Wildcard"
        );
    }

    #[test]
    fn match_with_guard() {
        let expr = parse_expr(
            r#"match x with
            | n if n > 0 -> n
            | _ -> 0
            end"#,
        )
        .unwrap();

        let ExprKind::Match { arms, .. } = &expr.node else {
            panic!("expected Match");
        };

        // First arm should have a guard
        assert!(arms[0].guard.is_some(), "first arm should have guard");
        assert!(arms[1].guard.is_none(), "second arm should not have guard");
    }
}

// ============================================================================
// Pipeline Structure
// ============================================================================

mod pipelines {
    use super::*;

    #[test]
    fn forward_pipe() {
        // x |> f parses as f x
        let expr = parse_expr("x |> f").unwrap();

        let ExprKind::App { func, arg } = &expr.node else {
            panic!("expected App, got {:?}", expr.node);
        };

        assert!(
            matches!(&func.node, ExprKind::Var(name) if name == "f"),
            "func should be f"
        );
        assert!(
            matches!(&arg.node, ExprKind::Var(name) if name == "x"),
            "arg should be x"
        );
    }

    #[test]
    fn chained_pipes() {
        // x |> f |> g should parse as g (f x)
        let expr = parse_expr("x |> f |> g").unwrap();

        let ExprKind::App { func, arg } = &expr.node else {
            panic!("expected outer App");
        };

        // func should be g
        assert!(
            matches!(&func.node, ExprKind::Var(name) if name == "g"),
            "outer func should be g"
        );

        // arg should be (f x)
        let ExprKind::App {
            func: inner_func,
            arg: inner_arg,
        } = &arg.node
        else {
            panic!("inner should be App");
        };

        assert!(
            matches!(&inner_func.node, ExprKind::Var(name) if name == "f"),
            "inner func should be f"
        );
        assert!(
            matches!(&inner_arg.node, ExprKind::Var(name) if name == "x"),
            "inner arg should be x"
        );
    }
}

// ============================================================================
// Tuple and List Structure
// ============================================================================

mod collections {
    use super::*;

    #[test]
    fn tuple_structure() {
        let expr = parse_expr("(1, 2, 3)").unwrap();

        let ExprKind::Tuple(elements) = &expr.node else {
            panic!("expected Tuple, got {:?}", expr.node);
        };

        assert_eq!(elements.len(), 3, "should have 3 elements");
    }

    #[test]
    fn empty_list() {
        let expr = parse_expr("[]").unwrap();

        let ExprKind::List(elements) = &expr.node else {
            panic!("expected List, got {:?}", expr.node);
        };

        assert_eq!(elements.len(), 0, "should be empty");
    }

    #[test]
    fn list_with_elements() {
        let expr = parse_expr("[1, 2, 3]").unwrap();

        let ExprKind::List(elements) = &expr.node else {
            panic!("expected List");
        };

        assert_eq!(elements.len(), 3, "should have 3 elements");
    }
}

// ============================================================================
// Select Expression Structure
// ============================================================================

mod select_expressions {
    use super::*;

    #[test]
    fn simple_select() {
        let expr = parse_expr(
            r#"select
            | x <- ch1 -> x
            | y <- ch2 -> y
            end"#,
        )
        .unwrap();

        let ExprKind::Select { arms } = &expr.node else {
            panic!("expected Select, got {:?}", expr.node);
        };

        assert_eq!(arms.len(), 2, "should have 2 arms");
    }
}

// ============================================================================
// Spawn Structure
// ============================================================================

mod spawn_expressions {
    use super::*;

    #[test]
    fn fiber_spawn() {
        // Fiber.spawn parses as application of a module-qualified name
        let expr = parse_expr("Fiber.spawn (fun () -> 42)").unwrap();

        let ExprKind::App { func, arg } = &expr.node else {
            panic!("expected App, got {:?}", expr.node);
        };

        // arg should be lambda
        assert!(
            matches!(&arg.node, ExprKind::Lambda { .. }),
            "arg should be Lambda, got {:?}",
            arg.node
        );

        // func is the qualified name - just verify it's an App or FieldAccess
        // The exact structure depends on parser implementation
        assert!(
            matches!(&func.node, ExprKind::FieldAccess { .. } | ExprKind::App { .. } | ExprKind::Var(_)),
            "func should be qualified name structure, got {:?}",
            func.node
        );
    }

    #[test]
    fn spawn_as_function_application() {
        // `spawn` is now just a regular function, not a special syntax
        let expr = parse_expr("spawn (fun () -> 42)").unwrap();

        let ExprKind::App { func, arg } = &expr.node else {
            panic!("expected App, got {:?}", expr.node);
        };

        // func should be variable `spawn`
        assert!(
            matches!(&func.node, ExprKind::Var(name) if name == "spawn"),
            "func should be Var(spawn), got {:?}",
            func.node
        );

        // arg should be lambda
        assert!(
            matches!(&arg.node, ExprKind::Lambda { .. }),
            "arg should be Lambda, got {:?}",
            arg.node
        );
    }
}

// ============================================================================
// Channel Operations Structure
// ============================================================================

mod channel_operations {
    use super::*;

    #[test]
    fn channel_new() {
        let expr = parse_expr("Channel.new").unwrap();

        assert!(
            matches!(&expr.node, ExprKind::NewChannel),
            "should be NewChannel, got {:?}",
            expr.node
        );
    }

    #[test]
    fn channel_send() {
        let expr = parse_expr("Channel.send ch 42").unwrap();

        let ExprKind::ChanSend { channel, value } = &expr.node else {
            panic!("expected ChanSend, got {:?}", expr.node);
        };

        assert!(
            matches!(&channel.node, ExprKind::Var(name) if name == "ch"),
            "channel should be ch"
        );
        assert!(
            matches!(&value.node, ExprKind::Lit(_)),
            "value should be Lit"
        );
    }

    #[test]
    fn channel_recv() {
        let expr = parse_expr("Channel.recv ch").unwrap();

        let ExprKind::ChanRecv(channel) = &expr.node else {
            panic!("expected ChanRecv, got {:?}", expr.node);
        };

        assert!(
            matches!(&channel.node, ExprKind::Var(name) if name == "ch"),
            "channel should be ch"
        );
    }
}
