//! Property-based tests for typeclass dispatch with constructor types
//!
//! These tests verify that typeclass dictionary dispatch works correctly,
//! especially for constructor types where the constructor name (e.g., "Some")
//! must be mapped to the type name (e.g., "Option") at runtime.
//!
//! Note: Uses Display instead of Show since Show is now in prelude.

use gneiss::ast::Program;
use gneiss::eval::Value;
use gneiss::prelude::parse_prelude;
use gneiss::types::TypeEnv;
use gneiss::{Inferencer, Interpreter, Lexer, Parser};
use proptest::prelude::*;

/// Run a complete program and return the final expression's value
fn run_program(source: &str) -> Result<Value, String> {
    // Parse prelude
    let prelude = parse_prelude().map_err(|e| e.to_string())?;

    // Parse user program
    let tokens = Lexer::new(source).tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let user_program = parser.parse_program().map_err(|e| e.to_string())?;

    // Combine prelude + user program
    let mut combined_items = prelude.items;
    combined_items.extend(user_program.items);
    let program = Program {
        exports: user_program.exports,
        items: combined_items,
    };

    let mut inferencer = Inferencer::new();
    let _env = inferencer
        .infer_program(&program, TypeEnv::new())
        .map_err(|errors| errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n"))?;

    let mut interpreter = Interpreter::new();
    interpreter.set_class_env(inferencer.take_class_env());
    interpreter.set_type_ctx(inferencer.take_type_ctx());

    interpreter.run(&program).map_err(|e| e.to_string())
}

/// Generate a small integer that won't cause issues with parsing
fn small_int() -> impl Strategy<Value = i64> {
    -1000i64..1000
}

/// Format an integer for Gneiss source (wrap negatives in parens)
fn format_int(n: i64) -> String {
    if n < 0 {
        format!("({})", n)
    } else {
        n.to_string()
    }
}

/// Generate a valid uppercase identifier for types/constructors
fn upper_ident_strategy() -> impl Strategy<Value = String> {
    "[A-Z][a-z0-9]{0,5}"
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test that Show instance for Int works with any integer value
    #[test]
    fn show_int_any_value(n in small_int()) {
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

impl Display for Int =
    let display n = int_to_string n
end

display {}
"#, format_int(n));

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = n.to_string();
        prop_assert!(
            matches!(&result, Value::String(s) if s == &expected),
            "Expected String(\"{}\"), got {:?}", expected, result
        );
    }

    /// Test that Show instance for Option works with Some containing any int
    #[test]
    fn show_option_some_int(n in small_int()) {
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some"
        | None -> "None"
        end
end

display (Some {})
"#, format_int(n));

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(&result, Value::String(s) if s == "Some"),
            "Expected String(\"Some\"), got {:?}", result
        );
    }

    /// Test that Show instance for Option None works
    #[test]
    fn show_option_none(_dummy in any::<u8>()) {
        let source = r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some"
        | None -> "None"
        end
end

display None
"#;

        let result = run_program(source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(&result, Value::String(s) if s == "None"),
            "Expected String(\"None\"), got {:?}", result
        );
    }

    /// Test nested Options: Option (Option Int)
    #[test]
    fn show_nested_option(n in small_int()) {
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some"
        | None -> "None"
        end
end

display (Some (Some {}))
"#, format_int(n));

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        // Outer Some matches first, returns "Some"
        prop_assert!(
            matches!(&result, Value::String(s) if s == "Some"),
            "Expected String(\"Some\"), got {:?}", result
        );
    }

    /// Test List type with Show constraint
    #[test]
    fn show_list_int(a in small_int(), b in small_int(), c in small_int()) {
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type List a = | Nil | Cons a (List a)

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (List a) where a : Display =
    let display xs = match xs with
        | Nil -> "[]"
        | Cons h t -> "list"
        end
end

display (Cons {} (Cons {} (Cons {} Nil)))
"#, format_int(a), format_int(b), format_int(c));

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        // Non-empty list matches Cons, returns "list"
        prop_assert!(
            matches!(&result, Value::String(s) if s == "list"),
            "Expected String(\"list\"), got {:?}", result
        );
    }

    /// Test Either type with two type parameters
    #[test]
    fn show_either(n in small_int(), use_left in any::<bool>()) {
        let value = if use_left {
            format!("Left {}", format_int(n))
        } else {
            format!("Right {}", format_int(n))
        };
        let expected = if use_left { "Left" } else { "Right" };

        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type Either a b = | Left a | Right b

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Either a b) where a : Display, b : Display =
    let display e = match e with
        | Left x -> "Left"
        | Right y -> "Right"
        end
end

display ({})
"#, value);

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(&result, Value::String(s) if s == expected),
            "Expected String(\"{}\"), got {:?}", expected, result
        );
    }

    /// Test that multiple traits can coexist
    #[test]
    fn multiple_traits_int(n in small_int()) {
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

trait Double a =
    val double : a -> a
end

impl Display for Int =
    let display n = int_to_string n
end

impl Double for Int =
    let double n = n + n
end

let x = {} in double x
"#, format_int(n));

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        let expected = n * 2;
        prop_assert!(
            matches!(&result, Value::Int(v) if *v == expected),
            "Expected Int({}), got {:?}", expected, result
        );
    }

    /// Test custom type names work correctly
    #[test]
    fn custom_type_name(
        type_name in upper_ident_strategy(),
        ctor_name in upper_ident_strategy(),
        n in small_int()
    ) {
        // Ensure type and constructor names are different to avoid conflicts
        prop_assume!(type_name != ctor_name);

        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type {} a = | {} a

impl Display for Int =
    let display n = int_to_string n
end

impl Display for ({} a) where a : Display =
    let display x = "custom"
end

display ({} {})
"#, type_name, ctor_name, type_name, ctor_name, format_int(n));

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(&result, Value::String(s) if s == "custom"),
            "Expected String(\"custom\"), got {:?}", result
        );
    }

    /// Test that trait methods work inside match arms
    #[test]
    fn show_in_match_arm(n in small_int(), use_some in any::<bool>()) {
        let expected = if use_some { n.to_string() } else { "None".to_string() };
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> display x
        | None -> "None"
        end
end

let input = {} in display input
"#, if use_some { format!("Some {}", format_int(n)) } else { "None".to_string() });

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(&result, Value::String(s) if s == &expected),
            "Expected String(\"{}\"), got {:?}", expected, result
        );
    }

    /// Test that passing constructor values through functions works
    #[test]
    fn show_through_function(n in small_int()) {
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some"
        | None -> "None"
        end
end

let render x = display x in render (Some {})
"#, format_int(n));

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(&result, Value::String(s) if s == "Some"),
            "Expected String(\"Some\"), got {:?}", result
        );
    }

    /// Test deeply nested constructors
    #[test]
    fn show_deep_nesting(depth in 1usize..5) {
        let mut value = "42".to_string();
        for _ in 0..depth {
            value = format!("Some ({})", value);
        }

        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some"
        | None -> "None"
        end
end

display ({})
"#, value);

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        // Outermost Some matches first
        prop_assert!(
            matches!(&result, Value::String(s) if s == "Some"),
            "Expected String(\"Some\"), got {:?}", result
        );
    }

    /// Test that Bool also works (not just Int)
    #[test]
    fn show_option_bool(b in any::<bool>()) {
        let source = format!(r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Bool =
    let display b = if b then "true" else "false"
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some"
        | None -> "None"
        end
end

display (Some {})
"#, b);

        let result = run_program(&source).map_err(|e| TestCaseError::fail(e))?;
        prop_assert!(
            matches!(&result, Value::String(s) if s == "Some"),
            "Expected String(\"Some\"), got {:?}", result
        );
    }
}

// ============================================================================
// Regression tests for specific edge cases
// ============================================================================

#[cfg(test)]
mod regression {
    use super::*;

    #[test]
    fn test_constructor_name_vs_type_name() {
        // This is the core bug - Some vs Option
        let source = r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = "option"
end

display (Some 42)
"#;
        let result = run_program(source).expect("Constructor name vs type name bug");
        assert!(
            matches!(result, Value::String(ref s) if s == "option"),
            "Expected String(\"option\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_multiple_constructors_same_type() {
        // Both Some and None should map to Option
        // Test with None (last expression)
        let source = r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "some"
        | None -> "none"
        end
end

display None
"#;
        let result = run_program(source).expect("Multiple constructors same type");
        assert!(
            matches!(result, Value::String(ref s) if s == "none"),
            "Expected String(\"none\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_recursive_show_call() {
        // show x inside the Show Option instance should dispatch to Show Int
        let source = r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some(" ++ display x ++ ")"
        | None -> "None"
        end
end

display (Some 42)
"#;
        let result = run_program(source).expect("Recursive show call");
        assert!(
            matches!(result, Value::String(ref s) if s == "Some(42)"),
            "Expected String(\"Some(42)\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_three_constructors() {
        // Result type with Ok, Err, and a third constructor
        // Test with Pending (last expression)
        let source = r#"
trait Display a =
    val display : a -> String
end

type Result a b c = | Ok a | Err b | Pending c

impl Display for Int =
    let display n = int_to_string n
end

impl Display for String =
    let display s = s
end

impl Display for Bool =
    let display b = if b then "true" else "false"
end

impl Display for (Result a b c) where a : Display, b : Display, c : Display =
    let display r = match r with
        | Ok x -> "Ok"
        | Err e -> "Err"
        | Pending p -> "Pending"
        end
end

display (Pending true)
"#;
        let result = run_program(source).expect("Three constructors same type");
        assert!(
            matches!(result, Value::String(ref s) if s == "Pending"),
            "Expected String(\"Pending\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_nullary_constructor() {
        // None is a nullary constructor (no arguments)
        let source = r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "some"
        | None -> "none"
        end
end

display None
"#;
        let result = run_program(source).expect("Nullary constructor");
        assert!(
            matches!(result, Value::String(ref s) if s == "none"),
            "Expected String(\"none\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_nested_different_types() {
        // Option (List Int) - nested different parameterized types
        let source = r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None
type List a = | Nil | Cons a (List a)

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (List a) where a : Display =
    let display xs = "list"
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some"
        | None -> "None"
        end
end

display (Some (Cons 1 (Cons 2 Nil)))
"#;
        let result = run_program(source).expect("Nested different parameterized types");
        assert!(
            matches!(result, Value::String(ref s) if s == "Some"),
            "Expected String(\"Some\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_constraint_chain() {
        // Show (Option (Option Int)) requires Show (Option Int) requires Show Int
        let source = r#"
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some(" ++ display x ++ ")"
        | None -> "None"
        end
end

display (Some (Some (Some 42)))
"#;
        let result = run_program(source).expect("Constraint chain");
        assert!(
            matches!(result, Value::String(ref s) if s == "Some(Some(Some(42)))"),
            "Expected String(\"Some(Some(Some(42)))\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_multiple_type_params() {
        // Pair a b with two type parameters
        let source = r#"
trait Display a =
    val display : a -> String
end

type Pair a b = | MkPair a b

impl Display for Int =
    let display n = int_to_string n
end

impl Display for Bool =
    let display b = if b then "true" else "false"
end

impl Display for (Pair a b) where a : Display, b : Display =
    let display p = match p with
        | MkPair x y -> "pair"
        end
end

display (MkPair 42 true)
"#;
        let result = run_program(source).expect("Multiple type params");
        assert!(
            matches!(result, Value::String(ref s) if s == "pair"),
            "Expected String(\"pair\"), got {:?}",
            result
        );
    }
}
