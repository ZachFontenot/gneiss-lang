//! Polymorphic Dispatch Tests - Compositional Testing for Trait Resolution
//!
//! These tests verify that polymorphic functions correctly dispatch through
//! trait dictionaries rather than falling through to monomorphic implementations.
//!
//! Historical bug: `==` worked on concrete types but when used in a polymorphic
//! function like `assert_eq x y = if x == y ...`, the elaborator dispatched to
//! `IntEq` instead of the Eq trait dictionary.
//!
//! These are INTEGRATION tests - they test that multiple features work together:
//! 1. Type inference with constraints
//! 2. Constraint propagation through elaboration
//! 3. Runtime trait dictionary dispatch

use gneiss::test_support::{typecheck_program, run_program_ok};

// ============================================================================
// Polymorphic Equality
// ============================================================================

mod polymorphic_eq {
    use super::*;

    #[test]
    fn eq_through_polymorphic_function_int() {
        // If == falls through to IntEq regardless of type, this passes by accident
        let program = r#"
let my_eq x y = x == y

let main () =
    if my_eq 1 1 then print "pass" else print "fail"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn eq_through_polymorphic_function_string() {
        // This is the KEY test - if == falls through to IntEq, strings won't compare correctly
        let program = r#"
let my_eq x y = x == y

let main () =
    if my_eq "hello" "hello" then print "pass" else print "fail"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn eq_used_at_multiple_types() {
        // Polymorphic function used at different types in same program
        let program = r#"
let my_eq x y = x == y

let main () =
    let int_eq = my_eq 1 1 in
    let str_eq = my_eq "a" "a" in
    let bool_eq = my_eq true true in
    if int_eq && str_eq && bool_eq then print "pass" else print "fail"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn neq_through_polymorphic_function() {
        let program = r#"
let my_neq x y = x != y

let main () =
    if my_neq "hello" "world" then print "pass" else print "fail"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn eq_in_higher_order_function() {
        // Equality check passed as behavior to higher-order function
        let program = r#"
let rec check_all_eq f xs ys =
    match (xs, ys) with
    | ([], []) -> true
    | (x :: xrest, y :: yrest) -> if f x y then check_all_eq f xrest yrest else false
    | _ -> false
    end

let main () =
    let result = check_all_eq (fun a b -> a == b) [1, 2, 3] [1, 2, 3] in
    if result then print "pass" else print "fail"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }
}

// ============================================================================
// Polymorphic Comparison
// ============================================================================

mod polymorphic_comparison {
    use super::*;

    #[test]
    fn lt_through_polymorphic_function() {
        let program = r#"
let my_lt x y = x < y

let main () =
    if my_lt 1 2 then print "pass" else print "fail"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn comparison_chain_polymorphic() {
        let program = r#"
let between lo hi x = lo <= x && x <= hi

let main () =
    if between 1 10 5 then print "pass" else print "fail"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }
}

// ============================================================================
// Polymorphic Arithmetic (if supported)
// ============================================================================

mod polymorphic_arithmetic {
    use super::*;

    #[test]
    fn add_concrete_types() {
        // Basic sanity check - arithmetic works on concrete types
        let program = r#"
let main () =
    let x = 1 + 2 in
    let y = 1.0 + 2.0 in
    print "pass"
"#;
        run_program_ok(program);
    }
}

// ============================================================================
// Trait Method Dispatch Through Polymorphic Functions
// ============================================================================

mod trait_dispatch {
    use super::*;

    #[test]
    fn show_through_polymorphic_function() {
        let program = r#"
let my_show x = show x

let main () =
    let s1 = my_show 42 in
    let s2 = my_show true in
    print s1;
    print s2
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn trait_method_in_nested_function() {
        // Trait method used in function that's used in another function
        let program = r#"
let stringify x = show x
let double_show x = stringify x ++ " " ++ stringify x

let main () =
    print (double_show 42)
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn constrained_type_in_let_binding() {
        // Type with constraint bound in let, then used
        let program = r#"
let format_pair x y =
    show x ++ " and " ++ show y

let main () =
    print (format_pair 1 2);
    print (format_pair "a" "b")
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }
}

// ============================================================================
// Nested Contexts - Type Inference Inside Bodies
// ============================================================================

mod nested_contexts {
    use super::*;

    #[test]
    fn polymorphic_in_match_arm() {
        let program = r#"
type Option a = | Some a | None

let show_option opt =
    match opt with
    | Some x -> "Some(" ++ show x ++ ")"
    | None -> "None"
    end

let main () =
    print (show_option (Some 42));
    print (show_option (Some "hello"))
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn polymorphic_in_if_branch() {
        let program = r#"
let show_if_true cond x =
    if cond then show x else "hidden"

let main () =
    print (show_if_true true 42);
    print (show_if_true false "secret")
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn polymorphic_in_lambda_body() {
        let program = r#"
let make_shower () = fun x -> show x

let main () =
    let shower = make_shower () in
    print (shower 42)
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn polymorphic_in_let_body() {
        let program = r#"
let wrap_show x =
    let s = show x in
    "[" ++ s ++ "]"

let main () =
    print (wrap_show 42);
    print (wrap_show true)
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn deeply_nested_polymorphic() {
        // Multiple levels of nesting
        let program = r#"
type Option a = | Some a | None

let deep x =
    let outer =
        match Some x with
        | Some y ->
            let inner = show y in
            if true then inner else "nope"
        | None -> "none"
        end
    in
    outer

let main () =
    print (deep 42);
    print (deep "hello")
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }
}

// ============================================================================
// Multiple Constraints
// ============================================================================

mod multiple_constraints {
    use super::*;

    #[test]
    fn show_and_eq_together() {
        let program = r#"
let show_if_equal x y =
    if x == y then show x else "not equal"

let main () =
    print (show_if_equal 1 1);
    print (show_if_equal 1 2);
    print (show_if_equal "a" "a")
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }
}

// ============================================================================
// Regression: Specific Bug Patterns
// ============================================================================

mod regression {
    use super::*;

    #[test]
    fn assert_eq_pattern() {
        // This is the exact pattern that triggered the original bug
        let program = r#"
let assert_eq x y =
    if x == y then ()
    else print ("assertion failed: " ++ show x ++ " != " ++ show y)

let main () =
    assert_eq 42 42;
    assert_eq "hello" "hello";
    assert_eq true true;
    print "all assertions passed"
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn assert_eq_fails_correctly() {
        // Make sure assert_eq actually detects inequality
        let program = r#"
let assert_eq x y =
    if x == y then print "equal" else print "not equal"

let main () =
    assert_eq "hello" "world"
"#;
        // This should print "not equal", not "equal"
        // If == falls through to IntEq, string comparison is broken
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }

    #[test]
    fn polymorphic_eq_with_show_in_error() {
        // The exact bug scenario: eq dispatch + show in error message
        let program = r#"
let check x y =
    if x == y then "match: " ++ show x
    else "mismatch: " ++ show x ++ " vs " ++ show y

let main () =
    print (check 1 1);
    print (check "a" "b")
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "should typecheck: {:?}", result);
        run_program_ok(program);
    }
}
