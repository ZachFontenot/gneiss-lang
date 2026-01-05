//! Type Rejection Tests - Soundness Canaries and Rejection Verification
//!
//! These tests verify that the type system REJECTS invalid programs.
//! This is critical: a compiler that accepts everything is useless.
//!
//! Categories:
//! 1. Soundness canaries - Must ALWAYS reject, or we have a soundness bug
//! 2. Type mismatch rejection - Basic type errors
//! 3. Occurs check - Infinite type prevention
//! 4. Scope rejection - Unbound variables
//! 5. Pattern rejection - Invalid pattern usage
//! 6. Channel type safety - No mixed types through channels

use gneiss::test_support::{typecheck_expr, typecheck_program};

// ============================================================================
// Soundness Canaries
// ============================================================================
// If ANY of these tests pass (don't reject), we have a CRITICAL soundness bug.

mod soundness {
    use super::*;

    #[test]
    fn occurs_check_self_application() {
        // fun x -> x x would have infinite type: a = a -> b
        let result = typecheck_expr("fun x -> x x");
        assert!(
            result.is_err(),
            "Self-application must fail occurs check"
        );
    }

    #[test]
    fn occurs_check_recursive_type() {
        // y combinator-like pattern must fail
        let result = typecheck_expr("fun f -> (fun x -> f (x x)) (fun x -> f (x x))");
        assert!(
            result.is_err(),
            "Y combinator must fail occurs check"
        );
    }

    #[test]
    fn polymorphic_channel_mixed_types() {
        // Can't send Int and Bool through same channel
        let result = typecheck_program(
            r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () -> Channel.send ch 42) in
    Channel.send ch true
"#,
        );
        assert!(
            result.is_err(),
            "Mixed types through channel must be rejected"
        );
    }

    #[test]
    fn polymorphic_channel_int_then_string() {
        let result = typecheck_program(
            r#"
let main () =
    let ch = Channel.new in
    Channel.send ch 1;
    Channel.send ch "hello"
"#,
        );
        assert!(
            result.is_err(),
            "Int then String through same channel must be rejected"
        );
    }

    #[test]
    fn escaping_type_variable_not_generalized() {
        // Type variable from outer scope should not be generalized in inner let
        // This tests the value restriction / level tracking
        let result = typecheck_program(
            r#"
let f x =
    let g y = x in
    (g 1, g true)
"#,
        );
        // This should succeed because x is NOT generalized (it escapes)
        // g has type: forall a. a -> typeof(x), so (g 1, g true) is fine
        // The key is that x keeps its type from the outer binding
        assert!(
            result.is_ok(),
            "Escaping variable should work correctly: {:?}",
            result
        );
    }
}

// ============================================================================
// Type Mismatch Rejection
// ============================================================================

mod type_mismatch {
    use super::*;

    #[test]
    fn int_plus_bool() {
        let result = typecheck_expr("1 + true");
        assert!(result.is_err(), "Int + Bool must be rejected");
    }

    #[test]
    fn int_plus_string() {
        let result = typecheck_expr("1 + \"hello\"");
        assert!(result.is_err(), "Int + String must be rejected");
    }

    #[test]
    fn bool_and_int() {
        let result = typecheck_expr("true && 1");
        assert!(result.is_err(), "Bool && Int must be rejected");
    }

    #[test]
    fn if_condition_not_bool() {
        let result = typecheck_expr("if 1 then 2 else 3");
        assert!(result.is_err(), "if with Int condition must be rejected");
    }

    #[test]
    fn if_branches_different_types() {
        let result = typecheck_expr("if true then 1 else \"hello\"");
        assert!(
            result.is_err(),
            "if branches with different types must be rejected"
        );
    }

    #[test]
    fn function_wrong_argument_type() {
        let result = typecheck_expr("(fun x -> x + 1) true");
        assert!(
            result.is_err(),
            "Applying Int function to Bool must be rejected"
        );
    }

    #[test]
    fn apply_non_function() {
        let result = typecheck_expr("42 1");
        assert!(result.is_err(), "Applying non-function must be rejected");
    }

    #[test]
    fn list_mixed_types() {
        let result = typecheck_expr("[1, true, \"hello\"]");
        assert!(result.is_err(), "List with mixed types must be rejected");
    }

    #[test]
    fn cons_wrong_types() {
        let result = typecheck_expr("1 :: [true]");
        assert!(
            result.is_err(),
            "Cons Int to Bool list must be rejected"
        );
    }

    #[test]
    fn comparison_different_types() {
        let result = typecheck_expr("1 == true");
        assert!(
            result.is_err(),
            "Comparing Int and Bool must be rejected"
        );
    }
}

// ============================================================================
// Scope Rejection
// ============================================================================

mod scope {
    use super::*;

    #[test]
    fn unbound_variable() {
        let result = typecheck_expr("x + 1");
        assert!(result.is_err(), "Unbound variable must be rejected");
    }

    #[test]
    fn variable_not_in_scope_after_let() {
        let result = typecheck_expr("let x = 1 in x + y");
        assert!(
            result.is_err(),
            "Reference to undefined y must be rejected"
        );
    }

    #[test]
    fn shadowed_variable_type_change() {
        // This should succeed - shadowing is allowed
        let result = typecheck_expr("let x = 1 in let x = true in x");
        assert!(
            result.is_ok(),
            "Shadowing with different type is allowed: {:?}",
            result
        );
    }

    #[test]
    fn unknown_constructor() {
        let result = typecheck_program(
            r#"
let x = Unknown 42
"#,
        );
        assert!(result.is_err(), "Unknown constructor must be rejected");
    }
}

// ============================================================================
// Pattern Rejection
// ============================================================================

mod patterns {
    use super::*;

    #[test]
    fn match_on_wrong_constructor() {
        let result = typecheck_program(
            r#"
type Option a = | Some a | None

let f x = match x with
    | Left y -> y
    | Right z -> z
end
"#,
        );
        assert!(
            result.is_err(),
            "Matching Option with Left/Right must be rejected"
        );
    }

    #[test]
    fn match_arms_different_types() {
        let result = typecheck_program(
            r#"
type Option a = | Some a | None

let f x = match x with
    | Some y -> y
    | None -> "default"
end
"#,
        );
        // If y is Int, then arms have different types
        let result2 = typecheck_program(
            r#"
type Option a = | Some a | None

let f (x : Option Int) = match x with
    | Some y -> y
    | None -> "default"
end
"#,
        );
        assert!(
            result2.is_err(),
            "Match arms with different types must be rejected"
        );
    }

    #[test]
    fn constructor_wrong_arity() {
        let result = typecheck_program(
            r#"
type Option a = | Some a | None

let x = Some 1 2
"#,
        );
        assert!(
            result.is_err(),
            "Constructor with wrong arity must be rejected"
        );
    }
}

// ============================================================================
// Channel Type Safety
// ============================================================================

mod channels {
    use super::*;

    #[test]
    fn recv_send_type_mismatch() {
        let result = typecheck_program(
            r#"
let main () =
    let ch = Channel.new in
    Channel.send ch 42;
    let x : Bool = Channel.recv ch in
    x
"#,
        );
        assert!(
            result.is_err(),
            "Receiving Bool from Int channel must be rejected"
        );
    }

    #[test]
    fn channel_in_function_type_safety() {
        // Channel created in function should maintain type safety
        let result = typecheck_program(
            r#"
let send_int ch = Channel.send ch 42
let send_bool ch = Channel.send ch true

let main () =
    let ch = Channel.new in
    send_int ch;
    send_bool ch
"#,
        );
        assert!(
            result.is_err(),
            "Using channel with different types must be rejected"
        );
    }
}

// ============================================================================
// Effect Handler Type Safety
// ============================================================================

mod effects {
    use super::*;

    #[test]
    fn handler_return_type_mismatch() {
        let result = typecheck_program(
            r#"
effect Ask =
    | ask : () -> Int
end

let main () =
    handle 42 with
        | return x -> x
        | ask () k -> k "not an int"
    end
"#,
        );
        // k expects Int, giving String should fail
        assert!(
            result.is_err(),
            "Handler resuming with wrong type must be rejected"
        );
    }

    #[test]
    fn handler_arms_different_types() {
        let result = typecheck_program(
            r#"
effect Ask =
    | ask : () -> Int
end

let main () =
    handle (perform Ask.ask ()) with
        | return x -> x
        | ask () k -> "string instead of int"
    end
"#,
        );
        // return returns Int, but ask handler returns String
        assert!(
            result.is_err(),
            "Handler arms with different types must be rejected"
        );
    }
}

// ============================================================================
// Recursive Function Type Safety
// ============================================================================

mod recursion {
    use super::*;

    #[test]
    fn recursive_type_mismatch() {
        let result = typecheck_program(
            r#"
let rec f x = if x then f 1 else 0
"#,
        );
        // f is called with Bool (x) and Int (1) - type mismatch
        assert!(
            result.is_err(),
            "Recursive function with inconsistent argument types must be rejected"
        );
    }

    #[test]
    fn mutual_recursion_type_mismatch() {
        let result = typecheck_program(
            r#"
let rec even n = if n == 0 then true else odd (n - 1)
and odd n = if n == 0 then 0 else even (n - 1)
"#,
        );
        // even returns Bool, odd returns Int, but they call each other
        assert!(
            result.is_err(),
            "Mutually recursive functions with type mismatch must be rejected"
        );
    }
}

// ============================================================================
// Typeclass Rejection
// ============================================================================

mod typeclasses {
    use super::*;

    #[test]
    fn missing_instance() {
        let result = typecheck_program(
            r#"
trait Foo a =
    foo : a -> Int
end

let use_foo x = foo x
let main () = use_foo 42
"#,
        );
        // No Foo instance for Int
        assert!(
            result.is_err(),
            "Using trait without instance must be rejected"
        );
    }

    #[test]
    fn wrong_instance_type() {
        let result = typecheck_program(
            r#"
trait Stringify a =
    stringify : a -> String
end

impl Stringify Int =
    let stringify x = 42
end
"#,
        );
        // stringify should return String, not Int
        assert!(
            result.is_err(),
            "Instance method with wrong return type must be rejected"
        );
    }
}
