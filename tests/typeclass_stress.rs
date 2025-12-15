//! Stress tests for typeclass features
//!
//! These tests cover edge cases in the typeclass system:
//! - Multiple constraints on functions
//! - Complex constraint propagation
//! - Instance resolution with multiple candidates

use gneiss::eval::Value;
use gneiss::{Inferencer, Interpreter, Lexer, Parser};

/// Run a complete program with typeclass support
fn run_program(source: &str) -> Result<Value, String> {
    let tokens = Lexer::new(source).tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;

    let mut inferencer = Inferencer::new();
    let _env = inferencer
        .infer_program(&program)
        .map_err(|e| e.to_string())?;

    let mut interpreter = Interpreter::new();
    interpreter.set_class_env(inferencer.take_class_env());
    interpreter.set_type_ctx(inferencer.take_type_ctx());

    interpreter.run(&program).map_err(|e| e.to_string())
}

fn run_ok(source: &str) {
    match run_program(source) {
        Ok(_) => {}
        Err(e) => panic!("Program failed: {}\nSource:\n{}", e, source),
    }
}

// ============================================================================
// Multiple Constraints Tests
// ============================================================================

#[test]
fn function_with_two_constraints() {
    // Function requires both Show and Eq constraints
    let program = r#"
trait Show a =
    val show : a -> String
end

trait Eq a =
    val eq : a -> a -> Bool
end

impl Show for Int =
    let show n = int_to_string n
end

impl Eq for Int =
    let eq a b = a == b
end

let show_if_equal x y =
    if eq x y then show x else "different"

let main () =
    show_if_equal 42 42
"#;
    run_ok(program);
}

#[test]
fn constrained_instance_with_two_constraints() {
    // Instance requires two constraints on type parameter
    let program = r#"
trait Show a =
    val show : a -> String
end

trait Eq a =
    val eq : a -> a -> Bool
end

type Pair a = | MkPair a a

impl Show for Int =
    let show n = int_to_string n
end

impl Eq for Int =
    let eq a b = a == b
end

impl Show for (Pair a) where a : Show, a : Eq =
    let show p = match p with
        | MkPair x y -> if eq x y then "equal pair" else "different pair"
end

let main () =
    print (show (MkPair 1 1));
    print (show (MkPair 1 2));
    0
"#;
    run_ok(program);
}

// ============================================================================
// Complex Instance Resolution
// ============================================================================

#[test]
fn nested_constrained_types() {
    // Show for List (Option a) where a : Show
    let program = r#"
trait Show a =
    val show : a -> String
end

type Option a = | Some a | None
type List a = | Nil | Cons a (List a)

impl Show for Int =
    let show n = int_to_string n
end

impl Show for (Option a) where a : Show =
    let show opt = match opt with
        | Some x -> "Some(" ++ show x ++ ")"
        | None -> "None"
end

impl Show for (List a) where a : Show =
    let show xs = match xs with
        | Nil -> "[]"
        | Cons h t -> show h ++ " :: " ++ show t
end

let main () =
    let list = Cons (Some 1) (Cons None (Cons (Some 2) Nil)) in
    print (show list);
    0
"#;
    run_ok(program);
}

#[test]
fn three_type_parameter_instance() {
    // Instance with three type parameters, all constrained
    let program = r#"
trait Show a =
    val show : a -> String
end

type Triple a b c = | MkTriple a b c

impl Show for Int =
    let show n = int_to_string n
end

impl Show for Bool =
    let show b = if b then "true" else "false"
end

impl Show for String =
    let show s = "\"" ++ s ++ "\""
end

impl Show for (Triple a b c) where a : Show, b : Show, c : Show =
    let show t = match t with
        | MkTriple x y z -> "(" ++ show x ++ ", " ++ show y ++ ", " ++ show z ++ ")"
end

let main () =
    let t = MkTriple 42 true "hello" in
    print (show t);
    0
"#;
    run_ok(program);
}

// ============================================================================
// Dictionary Passing Tests
// ============================================================================

#[test]
fn dictionary_passed_through_multiple_calls() {
    // Dictionary must be threaded through nested function calls
    let program = r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

let helper1 x = show x
let helper2 x = helper1 x
let helper3 x = helper2 x

let main () =
    print (helper3 42);
    0
"#;
    run_ok(program);
}

#[test]
fn dictionary_in_higher_order_function() {
    // Pass constrained function as argument
    let program = r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

let apply_show f x = f x

let main () =
    print (apply_show show 42);
    0
"#;
    run_ok(program);
}

// ============================================================================
// Multiple Instances Same Trait
// ============================================================================

#[test]
fn same_trait_different_types() {
    // Show implemented for multiple unrelated types
    // Note: Each show call must be in a separate let binding to avoid
    // type variable monomorphization across the sequence
    let program = r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

impl Show for Bool =
    let show b = if b then "true" else "false"
end

impl Show for String =
    let show s = s
end

let show_int x = show x
let show_bool x = show x
let show_str x = show x

let main () =
    print (show_int 42);
    print (show_bool true);
    print (show_str "hello");
    0
"#;
    run_ok(program);
}

// ============================================================================
// Trait Method in Different Contexts
// ============================================================================

#[test]
fn trait_method_in_let_binding() {
    let program = r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

let main () =
    let result = show 42 in
    print result;
    0
"#;
    run_ok(program);
}

#[test]
fn trait_method_in_if_condition() {
    let program = r#"
trait Eq a =
    val eq : a -> a -> Bool
end

impl Eq for Int =
    let eq a b = a == b
end

let main () =
    let result = if eq 1 1 then 100 else 0 in
    print result;
    result
"#;
    run_ok(program);
}

#[test]
fn trait_method_in_match_arm() {
    let program = r#"
trait Show a =
    val show : a -> String
end

type Option a = | Some a | None

impl Show for Int =
    let show n = int_to_string n
end

let main () =
    let opt = Some 42 in
    let result = match opt with
        | Some x -> show x
        | None -> "none"
    in
    print result;
    0
"#;
    run_ok(program);
}

// ============================================================================
// Instance Selection Edge Cases
// ============================================================================

#[test]
fn specific_instance_over_generic() {
    // When both specific (Show Int) and generic (Show (List a)) could apply,
    // the specific one should be chosen for the right type
    // Note: Separate functions needed to avoid type monomorphization
    let program = r#"
trait Show a =
    val show : a -> String
end

type List a = | Nil | Cons a (List a)

impl Show for Int =
    let show n = int_to_string n
end

impl Show for (List a) where a : Show =
    let show xs = "list"
end

let show_int x = show x
let show_list x = show x

let main () =
    print (show_int 42);
    print (show_list (Cons 1 Nil));
    0
"#;
    run_ok(program);
}

// ============================================================================
// Recursive Trait Method Calls
// ============================================================================

#[test]
fn recursive_show_with_list() {
    // Show List calls show on elements recursively
    let program = r#"
trait Show a =
    val show : a -> String
end

type List a = | Nil | Cons a (List a)

impl Show for Int =
    let show n = int_to_string n
end

impl Show for (List a) where a : Show =
    let show xs = match xs with
        | Nil -> "[]"
        | Cons h Nil -> "[" ++ show h ++ "]"
        | Cons h t -> "[" ++ show h ++ ", ...]"
end

let main () =
    print (show Nil);
    print (show (Cons 1 Nil));
    print (show (Cons 1 (Cons 2 Nil)));
    0
"#;
    run_ok(program);
}

// ============================================================================
// Type Inference with Constraints
// ============================================================================

#[test]
fn inferred_constraint_from_usage() {
    // Type of `f` should be inferred as `a -> String where a : Show`
    let program = r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

let f x = show x

let main () =
    print (f 42);
    0
"#;
    run_ok(program);
}
