//! Parser unit tests
//!
//! These tests verify the parser's behavior for various language constructs.
//! Tests are organized by category.

use gneiss::ast::{Decl, ExportItem, ExprKind, Item, PatternKind};
use gneiss::lexer::Lexer;
use gneiss::parser::Parser;
use gneiss::Program;

// ============================================================================
// Helpers
// ============================================================================

fn parse(input: &str) -> Program {
    let tokens = Lexer::new(input).tokenize().unwrap();
    Parser::new(tokens).parse_program().unwrap()
}

fn parse_err(input: &str) -> String {
    let tokens = Lexer::new(input).tokenize().unwrap();
    match Parser::new(tokens).parse_program() {
        Ok(_) => panic!("expected parse error"),
        Err(e) => e.to_string(),
    }
}

// ============================================================================
// Let Bindings
// ============================================================================

mod let_bindings {
    use super::*;

    #[test]
    fn simple_let() {
        let prog = parse("let x = 42");
        assert_eq!(prog.items.len(), 1);
        assert!(matches!(&prog.items[0], Item::Decl(Decl::Let { name, .. }) if name == "x"));
    }

    #[test]
    fn let_function() {
        let prog = parse("let add x y = x + y");
        assert_eq!(prog.items.len(), 1);
        if let Item::Decl(Decl::Let { name, params, .. }) = &prog.items[0] {
            assert_eq!(name, "add");
            assert_eq!(params.len(), 2);
        } else {
            panic!("expected let declaration");
        }
    }

    #[test]
    fn let_expression() {
        let prog = parse("let x = 5 in x + 1");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::Let { body, .. } => {
                    assert!(body.is_some(), "let-expression should have a body");
                }
                _ => panic!("expected let expression"),
            },
            _ => panic!("expected Item::Expr"),
        }
    }

    #[test]
    fn let_function_expression() {
        let prog = parse("let f x = x + 1 in f 5");
        assert_eq!(prog.items.len(), 1);
        assert!(matches!(&prog.items[0], Item::Expr(_)));
    }

    #[test]
    fn let_decl_with_double_semi() {
        let prog = parse("let x = 5;; x");
        assert_eq!(prog.items.len(), 2);
        assert!(matches!(&prog.items[0], Item::Decl(Decl::Let { .. })));
        assert!(matches!(&prog.items[1], Item::Expr(_)));
    }

    #[test]
    fn let_rec_mutual() {
        let prog = parse("let rec even n = if n == 0 then true else odd (n - 1) and odd n = if n == 0 then false else even (n - 1)");
        assert_eq!(prog.items.len(), 1);
        if let Item::Decl(Decl::LetRec { bindings, .. }) = &prog.items[0] {
            assert_eq!(bindings.len(), 2);
            assert_eq!(bindings[0].name.node, "even");
            assert_eq!(bindings[1].name.node, "odd");
        } else {
            panic!("expected let rec declaration");
        }
    }

    // Note: qualified names on LHS (let Module.name = ...) were removed.
    // Use proper modules instead: define `let name = ...` in a module file,
    // and callers access it as `Module.name`.
}

// ============================================================================
// Type Declarations
// ============================================================================

mod type_decls {
    use super::*;

    #[test]
    fn variant_type() {
        let prog = parse("type Option a = | Some a | None");
        assert_eq!(prog.items.len(), 1);
        if let Item::Decl(Decl::Type { name, params, constructors, .. }) = &prog.items[0] {
            assert_eq!(name, "Option");
            assert_eq!(params.len(), 1);
            assert_eq!(constructors.len(), 2);
        } else {
            panic!("expected type declaration");
        }
    }

    #[test]
    fn record_type() {
        let prog = parse("type Point = { x : Int, y : Int }");
        assert_eq!(prog.items.len(), 1);
        if let Item::Decl(Decl::Record { name, fields, .. }) = &prog.items[0] {
            assert_eq!(name, "Point");
            assert_eq!(fields.len(), 2);
        } else {
            panic!("expected record declaration");
        }
    }

    #[test]
    fn type_alias() {
        let prog = parse("type IntList = List Int");
        assert_eq!(prog.items.len(), 1);
        assert!(matches!(&prog.items[0], Item::Decl(Decl::TypeAlias { .. })));
    }
}

// ============================================================================
// Typeclasses
// ============================================================================

mod typeclasses {
    use super::*;

    #[test]
    fn trait_decl() {
        let prog = parse("trait Show a = val show : a -> String end");
        assert_eq!(prog.items.len(), 1);
        if let Item::Decl(Decl::Trait { name, type_param, methods, .. }) = &prog.items[0] {
            assert_eq!(name, "Show");
            assert_eq!(type_param, "a");
            assert_eq!(methods.len(), 1);
            assert_eq!(methods[0].name, "show");
        } else {
            panic!("expected trait declaration");
        }
    }

    #[test]
    fn trait_with_supertrait() {
        let prog = parse("trait Ord a : Eq = val compare : a -> a -> Int end");
        if let Item::Decl(Decl::Trait { supertraits, .. }) = &prog.items[0] {
            assert_eq!(supertraits.len(), 1);
            assert_eq!(supertraits[0], "Eq");
        } else {
            panic!("expected trait declaration");
        }
    }

    #[test]
    fn trait_multiple_methods() {
        let prog = parse("trait Eq a = val eq : a -> a -> Bool val neq : a -> a -> Bool end");
        if let Item::Decl(Decl::Trait { methods, .. }) = &prog.items[0] {
            assert_eq!(methods.len(), 2);
            assert_eq!(methods[0].name, "eq");
            assert_eq!(methods[1].name, "neq");
        } else {
            panic!("expected trait declaration");
        }
    }

    #[test]
    fn instance_decl() {
        let prog = parse("impl Show for Int = let show n = int_to_string n end");
        if let Item::Decl(Decl::Instance { trait_name, methods, .. }) = &prog.items[0] {
            assert_eq!(trait_name, "Show");
            assert_eq!(methods.len(), 1);
            assert_eq!(methods[0].name, "show");
        } else {
            panic!("expected instance declaration");
        }
    }

    #[test]
    fn constrained_instance() {
        let prog = parse(r#"impl Show for (List a) where a : Show = let show xs = "list" end"#);
        if let Item::Decl(Decl::Instance { constraints, .. }) = &prog.items[0] {
            assert_eq!(constraints.len(), 1);
            assert_eq!(constraints[0].type_var, "a");
            assert_eq!(constraints[0].trait_name, "Show");
        } else {
            panic!("expected instance declaration");
        }
    }
}

// ============================================================================
// Expressions
// ============================================================================

mod expressions {
    use super::*;

    #[test]
    fn match_requires_end() {
        let prog = parse("match x with | Some y -> y | None -> 0 end");
        assert_eq!(prog.items.len(), 1);
    }

    #[test]
    fn match_without_end_fails() {
        let err = parse_err("match x with | Some y -> y | None -> 0");
        assert!(err.contains("End"));
    }

    #[test]
    fn if_then_else() {
        let prog = parse("if x then 1 else 2");
        if let Item::Expr(expr) = &prog.items[0] {
            assert!(matches!(&expr.node, ExprKind::If { .. }));
        } else {
            panic!("expected if expression");
        }
    }

    #[test]
    fn lambda() {
        let prog = parse("fun x y -> x + y");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Lambda { params, .. } = &expr.node {
                assert_eq!(params.len(), 2);
            } else {
                panic!("expected lambda");
            }
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn sequence() {
        let prog = parse("print 1; print 2; 3");
        if let Item::Expr(expr) = &prog.items[0] {
            assert!(matches!(&expr.node, ExprKind::Seq { .. }));
        } else {
            panic!("expected sequence expression");
        }
    }

    #[test]
    fn binary_operators() {
        let prog = parse("1 + 2 * 3");
        if let Item::Expr(expr) = &prog.items[0] {
            // Should parse as 1 + (2 * 3) due to precedence
            assert!(matches!(&expr.node, ExprKind::BinOp { .. }));
        } else {
            panic!("expected binary operation");
        }
    }

    #[test]
    fn pipe_forward() {
        let prog = parse("x |> f |> g");
        assert_eq!(prog.items.len(), 1);
    }

    #[test]
    fn pipe_backward() {
        let prog = parse("g <| f <| x");
        assert_eq!(prog.items.len(), 1);
    }

    #[test]
    fn record_literal() {
        let prog = parse("Point { x = 1, y = 2 }");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Record { name, fields } = &expr.node {
                assert_eq!(name, "Point");
                assert_eq!(fields.len(), 2);
            } else {
                panic!("expected record literal");
            }
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn record_update() {
        let prog = parse("{ p with x = 3 }");
        if let Item::Expr(expr) = &prog.items[0] {
            assert!(matches!(&expr.node, ExprKind::RecordUpdate { .. }));
        } else {
            panic!("expected record update");
        }
    }

    #[test]
    fn field_access() {
        let prog = parse("point.x");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::FieldAccess { field, .. } = &expr.node {
                assert_eq!(field, "x");
            } else {
                panic!("expected field access");
            }
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn list_literal() {
        let prog = parse("[1, 2, 3]");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::List(items) = &expr.node {
                assert_eq!(items.len(), 3);
            } else {
                panic!("expected list");
            }
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn tuple() {
        let prog = parse("(1, 2, 3)");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Tuple(items) = &expr.node {
                assert_eq!(items.len(), 3);
            } else {
                panic!("expected tuple");
            }
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn unit() {
        let prog = parse("()");
        if let Item::Expr(expr) = &prog.items[0] {
            assert!(matches!(&expr.node, ExprKind::Lit(_)));
        } else {
            panic!("expected unit");
        }
    }
}

// ============================================================================
// Patterns
// ============================================================================

mod patterns {
    use super::*;
    use gneiss::ast::Literal;

    #[test]
    fn wildcard() {
        let prog = parse("match x with | _ -> 0 end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                assert!(matches!(&arms[0].pattern.node, PatternKind::Wildcard));
            }
        }
    }

    #[test]
    fn variable() {
        let prog = parse("match x with | y -> y end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                assert!(matches!(&arms[0].pattern.node, PatternKind::Var(s) if s == "y"));
            }
        }
    }

    #[test]
    fn constructor_nullary() {
        let prog = parse("match x with | None -> 0 end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                if let PatternKind::Constructor { name, args } = &arms[0].pattern.node {
                    assert_eq!(name, "None");
                    assert!(args.is_empty());
                }
            }
        }
    }

    #[test]
    fn constructor_with_args() {
        let prog = parse("match x with | Some y -> y end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                if let PatternKind::Constructor { name, args } = &arms[0].pattern.node {
                    assert_eq!(name, "Some");
                    assert_eq!(args.len(), 1);
                }
            }
        }
    }

    #[test]
    fn cons_pattern() {
        let prog = parse("match xs with | x :: rest -> x end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                assert!(matches!(&arms[0].pattern.node, PatternKind::Cons { .. }));
            }
        }
    }

    #[test]
    fn tuple_pattern() {
        let prog = parse("match p with | (x, y) -> x + y end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                if let PatternKind::Tuple(pats) = &arms[0].pattern.node {
                    assert_eq!(pats.len(), 2);
                }
            }
        }
    }

    #[test]
    fn list_pattern() {
        let prog = parse("match xs with | [a, b] -> a + b end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                if let PatternKind::List(pats) = &arms[0].pattern.node {
                    assert_eq!(pats.len(), 2);
                }
            }
        }
    }

    #[test]
    fn literal_patterns() {
        let prog = parse("match x with | 42 -> true | \"hello\" -> false end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                assert!(matches!(&arms[0].pattern.node, PatternKind::Lit(Literal::Int(42))));
                assert!(matches!(&arms[1].pattern.node, PatternKind::Lit(Literal::String(_))));
            }
        }
    }

    #[test]
    fn record_pattern() {
        let prog = parse("match p with | Point { x, y } -> x + y end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                if let PatternKind::Record { name, fields } = &arms[0].pattern.node {
                    assert_eq!(name, "Point");
                    assert_eq!(fields.len(), 2);
                }
            }
        }
    }

    #[test]
    fn guard() {
        let prog = parse("match x with | n if n > 0 -> n end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Match { arms, .. } = &expr.node {
                assert!(arms[0].guard.is_some());
            }
        }
    }
}

// ============================================================================
// Exports
// ============================================================================

mod exports {
    use super::*;

    #[test]
    fn export_values() {
        let prog = parse("export (foo, bar)");
        assert!(prog.exports.is_some());
        let exports = prog.exports.unwrap();
        assert_eq!(exports.items.len(), 2);
        assert!(matches!(&exports.items[0].node, ExportItem::Value(s) if s == "foo"));
        assert!(matches!(&exports.items[1].node, ExportItem::Value(s) if s == "bar"));
    }

    #[test]
    fn export_type_all() {
        let prog = parse("export (Option(..))");
        let exports = prog.exports.unwrap();
        assert!(matches!(&exports.items[0].node, ExportItem::TypeAll(s) if s == "Option"));
    }

    #[test]
    fn export_type_only() {
        let prog = parse("export (Config)");
        let exports = prog.exports.unwrap();
        assert!(matches!(&exports.items[0].node, ExportItem::TypeOnly(s) if s == "Config"));
    }

    #[test]
    fn export_type_some() {
        let prog = parse("export (Result(Ok, Err))");
        let exports = prog.exports.unwrap();
        if let ExportItem::TypeSome(name, ctors) = &exports.items[0].node {
            assert_eq!(name, "Result");
            assert_eq!(ctors.len(), 2);
        } else {
            panic!("expected TypeSome");
        }
    }

    #[test]
    fn export_mixed() {
        let prog = parse("export (foo, Option(..), Config)");
        let exports = prog.exports.unwrap();
        assert_eq!(exports.items.len(), 3);
    }
}

// ============================================================================
// Imports
// ============================================================================

mod imports {
    use super::*;

    #[test]
    fn simple_import() {
        let prog = parse("import Http");
        assert_eq!(prog.items.len(), 1);
        if let Item::Import(import) = &prog.items[0] {
            assert_eq!(import.node.module_path, "Http");
            assert!(import.node.alias.is_none());
            assert!(import.node.items.is_none());
        } else {
            panic!("expected import");
        }
    }

    #[test]
    fn import_with_alias() {
        let prog = parse("import Http as H");
        if let Item::Import(import) = &prog.items[0] {
            assert_eq!(import.node.alias.as_ref().unwrap(), "H");
        } else {
            panic!("expected import");
        }
    }

    #[test]
    fn selective_import() {
        let prog = parse("import Http (get, post)");
        if let Item::Import(import) = &prog.items[0] {
            let items = import.node.items.as_ref().unwrap();
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].0, "get");
            assert_eq!(items[1].0, "post");
        } else {
            panic!("expected import");
        }
    }

    #[test]
    fn selective_import_with_alias() {
        let prog = parse("import Http (get as fetch)");
        if let Item::Import(import) = &prog.items[0] {
            let items = import.node.items.as_ref().unwrap();
            assert_eq!(items[0].1.as_ref().unwrap(), "fetch");
        } else {
            panic!("expected import");
        }
    }

    #[test]
    fn module_path() {
        let prog = parse("import Collections/HashMap");
        if let Item::Import(import) = &prog.items[0] {
            assert_eq!(import.node.module_path, "Collections/HashMap");
        } else {
            panic!("expected import");
        }
    }
}

// ============================================================================
// Operators
// ============================================================================

mod operators {
    use super::*;

    #[test]
    fn fixity_declaration() {
        let prog = parse("infixl 6 +++");
        assert_eq!(prog.items.len(), 1);
        if let Item::Decl(Decl::Fixity(decl)) = &prog.items[0] {
            assert_eq!(decl.precedence, 6);
            assert_eq!(decl.operators.len(), 1);
            assert_eq!(decl.operators[0], "+++");
        } else {
            panic!("expected fixity declaration");
        }
    }

    #[test]
    fn operator_def_prefix() {
        let prog = parse("let (+++) a b = a + b");
        if let Item::Decl(Decl::OperatorDef { op, params, .. }) = &prog.items[0] {
            assert_eq!(op, "+++");
            assert_eq!(params.len(), 2);
        } else {
            panic!("expected operator definition");
        }
    }

    #[test]
    fn operator_def_infix() {
        let prog = parse("let a +++ b = a + b");
        if let Item::Decl(Decl::OperatorDef { op, params, .. }) = &prog.items[0] {
            assert_eq!(op, "+++");
            assert_eq!(params.len(), 2);
        } else {
            panic!("expected operator definition");
        }
    }
}

// ============================================================================
// Channels
// ============================================================================

mod channels {
    use super::*;

    #[test]
    fn channel_new() {
        let prog = parse("Channel.new");
        if let Item::Expr(expr) = &prog.items[0] {
            assert!(matches!(&expr.node, ExprKind::NewChannel));
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn channel_send() {
        let prog = parse("Channel.send ch 42");
        if let Item::Expr(expr) = &prog.items[0] {
            assert!(matches!(&expr.node, ExprKind::ChanSend { .. }));
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn channel_recv() {
        let prog = parse("Channel.recv ch");
        if let Item::Expr(expr) = &prog.items[0] {
            assert!(matches!(&expr.node, ExprKind::ChanRecv(_)));
        } else {
            panic!("expected expression");
        }
    }

    #[test]
    fn select_expression() {
        let prog = parse("select | x <- ch1 -> x | y <- ch2 -> y end");
        if let Item::Expr(expr) = &prog.items[0] {
            if let ExprKind::Select { arms } = &expr.node {
                assert_eq!(arms.len(), 2);
            } else {
                panic!("expected select");
            }
        } else {
            panic!("expected expression");
        }
    }
}

// ============================================================================
// Val Declarations
// ============================================================================

mod val_decls {
    use super::*;

    #[test]
    fn val_simple() {
        let prog = parse("val x : Int");
        if let Item::Decl(Decl::Val { name, .. }) = &prog.items[0] {
            assert_eq!(name, "x");
        } else {
            panic!("expected val declaration");
        }
    }

    #[test]
    fn val_function() {
        let prog = parse("val map : (a -> b) -> List a -> List b");
        if let Item::Decl(Decl::Val { name, .. }) = &prog.items[0] {
            assert_eq!(name, "map");
        } else {
            panic!("expected val declaration");
        }
    }

    // Note: qualified names (val Module.name : ...) were removed.
    // Use proper modules instead.
}
