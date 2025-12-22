//! Negative parser tests - tests for malformed input that should be rejected
//!
//! These tests verify that the parser correctly rejects invalid syntax
//! and produces appropriate error messages.

use gneiss::lexer::Lexer;
use gneiss::parser::{Parser, ParseError};

/// Parse input and expect failure
fn parse_fails(input: &str) -> ParseError {
    let tokens = Lexer::new(input).tokenize().expect("lexer should succeed");
    let result = Parser::new(tokens).parse_program();
    match result {
        Ok(_) => panic!("expected parse error for: {}", input),
        Err(e) => e,
    }
}

/// Parse input and verify it fails
fn should_fail(input: &str) {
    let tokens = Lexer::new(input).tokenize().expect("lexer should succeed");
    let result = Parser::new(tokens).parse_program();
    assert!(result.is_err(), "expected parse error for: {}", input);
}

// ============================================================================
// Malformed expressions
// ============================================================================

mod expressions {
    use super::*;

    #[test]
    fn missing_operand() {
        should_fail("let x = 1 +");
    }

    #[test]
    fn double_operator() {
        // Two operators in a row is invalid
        should_fail("let x = 1 + + 2");
    }

    #[test]
    fn unclosed_paren() {
        should_fail("let x = (1 + 2");
    }

    #[test]
    fn extra_paren() {
        should_fail("let x = 1 + 2)");
    }

    #[test]
    fn unclosed_bracket() {
        should_fail("let x = [1, 2");
    }

    #[test]
    fn unclosed_brace() {
        should_fail("let x = { a = 1");
    }

    #[test]
    fn empty_sequence() {
        should_fail("let x = ;");
    }

    #[test]
    fn trailing_comma_in_args() {
        // Trailing comma in function application is not allowed
        should_fail("let x = f a,");
    }
}

// ============================================================================
// If expression errors
// ============================================================================

mod if_expr {
    use super::*;

    #[test]
    fn missing_then() {
        should_fail("if true 1 else 2");
    }

    #[test]
    fn missing_else() {
        should_fail("if true then 1");
    }

    #[test]
    fn missing_condition() {
        should_fail("if then 1 else 2");
    }

    #[test]
    fn missing_then_branch() {
        should_fail("if true then else 2");
    }

    #[test]
    fn missing_else_branch() {
        should_fail("if true then 1 else");
    }
}

// ============================================================================
// Match expression errors
// ============================================================================

mod match_expr {
    use super::*;

    #[test]
    fn missing_with() {
        should_fail("match x | Some y -> y end");
    }

    #[test]
    fn missing_end() {
        should_fail("match x with | Some y -> y | None -> 0");
    }

    #[test]
    fn missing_arrow() {
        should_fail("match x with | Some y y end");
    }

    #[test]
    fn missing_pattern() {
        should_fail("match x with | -> 1 end");
    }

    #[test]
    fn missing_body() {
        should_fail("match x with | Some y -> end");
    }

    #[test]
    fn no_arms() {
        should_fail("match x with end");
    }
}

// ============================================================================
// Lambda errors
// ============================================================================

mod lambda {
    use super::*;

    #[test]
    fn missing_arrow() {
        should_fail("fun x x + 1");
    }

    #[test]
    fn missing_body() {
        should_fail("fun x ->");
    }

    #[test]
    fn double_arrow() {
        // Two arrows in a row is invalid
        should_fail("fun x -> -> 1");
    }
}

// ============================================================================
// Let binding errors
// ============================================================================

mod let_bindings {
    use super::*;

    #[test]
    fn missing_equals() {
        should_fail("let x 1");
    }

    #[test]
    fn missing_body() {
        should_fail("let x =");
    }

    #[test]
    fn missing_name() {
        should_fail("let = 1");
    }

    #[test]
    fn let_expr_missing_in() {
        // In expression context, let needs 'in'
        should_fail("fun () -> let x = 1 x");
    }

    #[test]
    fn let_rec_missing_and_between_bindings() {
        // Mutual recursion needs 'and' between bindings
        should_fail("let rec f x = g x g y = f y");
    }
}

// ============================================================================
// Pattern errors
// ============================================================================

mod patterns {
    use super::*;

    #[test]
    fn unclosed_tuple_pattern() {
        should_fail("match x with | (a, b -> 1 end");
    }

    #[test]
    fn unclosed_list_pattern() {
        should_fail("match x with | [a, b -> 1 end");
    }

    #[test]
    fn empty_cons_pattern() {
        should_fail("match x with | :: -> 1 end");
    }

    #[test]
    fn missing_cons_tail() {
        should_fail("match x with | a :: -> 1 end");
    }

    #[test]
    fn invalid_pattern_literal() {
        // Floats are not valid pattern literals
        should_fail("match x with | 1.5 -> 1 end");
    }
}

// ============================================================================
// Type declaration errors
// ============================================================================

mod type_decls {
    use super::*;

    #[test]
    fn missing_equals() {
        should_fail("type Option a Some a | None");
    }

    #[test]
    fn missing_name() {
        should_fail("type = Some a | None");
    }

    #[test]
    fn record_unclosed_brace() {
        should_fail("type Point = { x : Int, y : Int");
    }

    #[test]
    fn record_missing_type() {
        should_fail("type Point = { x : , y : Int }");
    }

    #[test]
    fn record_missing_colon() {
        should_fail("type Point = { x Int, y : Int }");
    }
}

// ============================================================================
// Operator errors
// ============================================================================

mod operators {
    use super::*;

    #[test]
    fn fixity_missing_precedence() {
        should_fail("infixl +");
    }

    #[test]
    fn fixity_invalid_precedence() {
        // Precedence should be a number
        should_fail("infixl high +");
    }

    #[test]
    fn operator_def_missing_body() {
        should_fail("let (<|>) a b =");
    }
}

// ============================================================================
// Import/Export errors
// ============================================================================

mod imports {
    use super::*;

    #[test]
    fn import_missing_module() {
        should_fail("import");
    }

    #[test]
    fn import_unclosed_parens() {
        should_fail("import Foo (bar");
    }

    #[test]
    fn export_unclosed_parens() {
        should_fail("export (foo");
    }
}

// ============================================================================
// Channel expression errors
// ============================================================================

mod channels {
    use super::*;

    #[test]
    fn select_missing_pipe() {
        should_fail("select Channel.recv ch -> x end");
    }

    #[test]
    fn select_missing_arrow() {
        should_fail("select | Channel.recv ch x end");
    }

    #[test]
    fn select_missing_end() {
        should_fail("select | Channel.recv ch -> x");
    }
}

// ============================================================================
// Delimiter errors
// ============================================================================

mod delimiters {
    use super::*;

    #[test]
    fn nested_match_without_end() {
        // Nested match MUST have end
        should_fail("let x = match y with | A -> match z with | B -> 1 | C -> 2 | D -> 3 end");
    }

    #[test]
    fn select_without_end() {
        should_fail("let x = select | Channel.recv ch -> 1");
    }
}

// ============================================================================
// Typeclass errors
// ============================================================================

mod typeclasses {
    use super::*;

    #[test]
    fn trait_missing_name() {
        should_fail("trait where show : a -> String");
    }

    #[test]
    fn trait_missing_where() {
        should_fail("trait Show show : a -> String");
    }

    #[test]
    fn impl_missing_for() {
        should_fail("impl Show Int where show x = int_to_string x");
    }

    #[test]
    fn impl_missing_where() {
        should_fail("impl Show for Int show x = int_to_string x");
    }
}
