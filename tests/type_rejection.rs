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
    fn top_level_value_restriction_channel() {
        // Top-level `let ch = Channel.new` must not generalize to forall a. Channel a.
        // Otherwise unrelated functions can use ch at incompatible types. (gneiss-lang-f4fl)
        let result = typecheck_program(
            r#"
let ch = Channel.new
val recv_str : () -> String
let recv_str () = Channel.recv ch
let main () =
    Fiber.spawn (fun () -> print (recv_str ()));
    Channel.send ch 42
"#,
        );
        assert!(
            result.is_err(),
            "Top-level Channel.new must obey value restriction: {:?}",
            result
        );
    }

    #[test]
    fn top_level_letrec_value_restriction_channel() {
        // Same bug via let rec. (gneiss-lang-f4fl)
        let result = typecheck_program(
            r#"
let rec ch_thing = Channel.new
and dummy () = ()
let main () =
    Fiber.spawn (fun () -> let s : String = Channel.recv ch_thing in print s);
    Channel.send ch_thing 42
"#,
        );
        assert!(
            result.is_err(),
            "Top-level let rec value must obey value restriction: {:?}",
            result
        );
    }

    #[test]
    fn top_level_self_referential_value_let_rejected() {
        // `let x = x + 1` at top level must not typecheck. (gneiss-lang-dbix)
        let result = typecheck_program("let x = x + 1");
        assert!(
            result.is_err(),
            "Self-referential top-level value let must be rejected: {:?}",
            result
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
// Constructor Arity Checking (gneiss-lang-rs5t)
// ============================================================================

mod constructor_arity {
    use super::*;

    #[test]
    fn pattern_with_too_few_args_rejected() {
        // MkPair takes 2 args; pattern provides 1.
        let result = typecheck_program(
            r#"
type Pair a = | MkPair a a
let f p = match p with | MkPair x -> x end
"#,
        );
        assert!(
            result.is_err(),
            "Pattern with wrong constructor arity must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn pattern_with_too_many_args_rejected() {
        let result = typecheck_program(
            r#"
type Pair a = | MkPair a a
let f p = match p with | MkPair x y z -> x end
"#,
        );
        assert!(
            result.is_err(),
            "Pattern with extra arguments must be rejected: {:?}",
            result
        );
    }
}

// ============================================================================
// val without let (gneiss-lang-709d)
// ============================================================================

mod val_without_let {
    use super::*;

    #[test]
    fn val_without_implementation_rejected() {
        let result = typecheck_program(
            r#"
val ch_int : Channel Int
let main () = let v = Channel.recv ch_int in print v
"#,
        );
        assert!(
            result.is_err(),
            "val without matching let must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn val_with_let_works() {
        let result = typecheck_program(
            r#"
val double : Int -> Int
let double x = x + x
let main () = print (double 21)
"#,
        );
        assert!(
            result.is_ok(),
            "val + let pair should typecheck: {:?}",
            result
        );
    }
}

// ============================================================================
// Predicate Discharge (typeclass constraints at top level)
// ============================================================================
// Predicates collected during binding inference must be discharged before the
// next binding starts; otherwise unsatisfiable constraints leak to runtime.
// (gneiss-lang-yz3r)

mod predicate_discharge {
    use super::*;

    #[test]
    fn print_no_show_instance_for_function_rejected() {
        let result = typecheck_program("let main () = print (fun x -> x)");
        assert!(
            result.is_err(),
            "print of a function (no Show instance) must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn show_constraint_propagates_to_call_site() {
        // f has a deferred Show constraint; when called with a function value,
        // the constraint becomes Show (a -> a) which has no instance.
        let result = typecheck_program(
            r#"
let f x = print x
let main () = f (fun x -> x)
"#,
        );
        assert!(
            result.is_err(),
            "Show constraint must be checked at the call site that picks the type: {:?}",
            result
        );
    }

    #[test]
    fn print_int_still_works() {
        // Sanity: print of values that DO have a Show instance must still typecheck.
        let result = typecheck_program("let main () = print 42");
        assert!(
            result.is_ok(),
            "print of Int should typecheck: {:?}",
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
        // Without annotation, y could be String, so this might succeed
        let _untyped = typecheck_program(
            r#"
type Option a = | Some a | None

let f x = match x with
    | Some y -> y
    | None -> "default"
end
"#,
        );
        // With explicit Int annotation, arms have different types - must reject
        let result = typecheck_program(
            r#"
type Option a = | Some a | None

let f (x : Option Int) = match x with
    | Some y -> y
    | None -> "default"
end
"#,
        );
        assert!(
            result.is_err(),
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

// ============================================================================
// Pattern Matching Exhaustiveness  (gneiss-lang-bt75)
// ============================================================================
//
// Non-exhaustive matches are a soundness bug: they typecheck but blow up at
// runtime as `MatchFailed`.  These tests pin down the new
// `NonExhaustivePatterns` error.

mod exhaustiveness {
    use super::*;

    fn err_contains_non_exhaustive(result: &Result<impl std::fmt::Debug, String>) -> bool {
        match result {
            Err(msg) => msg.contains("NonExhaustivePatterns")
                || msg.contains("non-exhaustive"),
            Ok(_) => false,
        }
    }

    // -------- ADT (sum-type) coverage --------

    #[test]
    fn non_exhaustive_adt_missing_constructor() {
        let result = typecheck_program(
            r#"
type Color = | Red | Green | Blue

let f c = match c with
    | Red -> 1
    | Green -> 2
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "Missing Blue must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn exhaustive_adt_all_constructors() {
        let result = typecheck_program(
            r#"
type Color = | Red | Green | Blue

let f c = match c with
    | Red -> 1
    | Green -> 2
    | Blue -> 3
    end
"#,
        );
        assert!(result.is_ok(), "All ctors must typecheck: {:?}", result);
    }

    #[test]
    fn exhaustive_adt_with_wildcard() {
        let result = typecheck_program(
            r#"
type Color = | Red | Green | Blue

let f c = match c with
    | Red -> 1
    | _ -> 0
    end
"#,
        );
        assert!(result.is_ok(), "Wildcard catch-all is exhaustive: {:?}", result);
    }

    #[test]
    fn non_exhaustive_option_missing_none() {
        // Option is in the prelude.
        let result = typecheck_program(
            r#"
let f opt = match opt with
    | Some x -> x
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "Match on Option missing None must be rejected: {:?}",
            result
        );
    }

    // -------- List coverage --------

    #[test]
    fn non_exhaustive_list_only_nil() {
        let result = typecheck_program(
            r#"
let f xs = match xs with
    | [] -> 0
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "List match with only [] must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn non_exhaustive_list_only_cons() {
        let result = typecheck_program(
            r#"
let f xs = match xs with
    | x :: _ -> x
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "List match with only Cons must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn exhaustive_list_nil_and_cons() {
        let result = typecheck_program(
            r#"
let f xs = match xs with
    | [] -> 0
    | x :: _ -> x
    end
"#,
        );
        assert!(result.is_ok(), "[] and _::_ must be exhaustive: {:?}", result);
    }

    #[test]
    fn exhaustive_list_with_literal_singleton_and_default() {
        let result = typecheck_program(
            r#"
let f xs = match xs with
    | [] -> 0
    | [x] -> x
    | _ -> 1
    end
"#,
        );
        assert!(result.is_ok(), "Literal-list arms with default work: {:?}", result);
    }

    // -------- Bool coverage --------

    #[test]
    fn non_exhaustive_bool_only_true() {
        let result = typecheck_program(
            r#"
let f b = match b with
    | true -> 1
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "Bool match with only `true` must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn non_exhaustive_bool_only_false() {
        let result = typecheck_program(
            r#"
let f b = match b with
    | false -> 0
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "Bool match with only `false` must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn exhaustive_bool_both_branches() {
        let result = typecheck_program(
            r#"
let f b = match b with
    | true -> 1
    | false -> 0
    end
"#,
        );
        assert!(result.is_ok(), "Both bool branches are exhaustive: {:?}", result);
    }

    // -------- Nested-pattern combos --------

    #[test]
    fn non_exhaustive_nested_constructor() {
        // (Option, Option) — covering Some/Some and None/None misses
        // Some/None and None/Some.
        let result = typecheck_program(
            r#"
let f p = match p with
    | (Some _, Some _) -> 1
    | (None, None) -> 0
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "Nested (Some, None) and (None, Some) missing — must reject: {:?}",
            result
        );
    }

    #[test]
    fn exhaustive_nested_constructor_with_wildcards() {
        let result = typecheck_program(
            r#"
let f p = match p with
    | (Some _, Some _) -> 1
    | (Some _, None) -> 2
    | (None, Some _) -> 3
    | (None, None) -> 0
    end
"#,
        );
        assert!(result.is_ok(), "All four combinations cover the type: {:?}", result);
    }

    // -------- Guards do not contribute --------

    #[test]
    fn guarded_arm_does_not_count_for_exhaustiveness() {
        // The guard might fail, so even a guarded `| _` is not enough.
        let result = typecheck_program(
            r#"
let f n = match n with
    | x if x > 0 -> 1
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "Sole guarded arm must not count as exhaustive: {:?}",
            result
        );
    }

    #[test]
    fn guarded_then_default_is_exhaustive() {
        let result = typecheck_program(
            r#"
let f n = match n with
    | x if x > 0 -> 1
    | _ -> 0
    end
"#,
        );
        assert!(
            result.is_ok(),
            "Guarded arm followed by default is exhaustive: {:?}",
            result
        );
    }

    // -------- Open types (Int/String/Char) --------

    #[test]
    fn non_exhaustive_int_literals() {
        let result = typecheck_program(
            r#"
let f n = match n with
    | 0 -> 1
    | 1 -> 2
    end
"#,
        );
        assert!(
            err_contains_non_exhaustive(&result),
            "Int literal match without default must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn exhaustive_int_with_wildcard() {
        let result = typecheck_program(
            r#"
let f n = match n with
    | 0 -> 1
    | _ -> 0
    end
"#,
        );
        assert!(result.is_ok(), "Int + wildcard is exhaustive: {:?}", result);
    }
}
