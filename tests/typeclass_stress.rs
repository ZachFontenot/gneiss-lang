//! Stress tests for typeclass features
//!
//! These tests cover edge cases in the typeclass system:
//! - Multiple constraints on functions
//! - Complex constraint propagation
//! - Instance resolution with multiple candidates
//!
//! Note: Uses Display instead of Show since Show is now in prelude.

use gneiss::ast::Program;
use gneiss::eval::Value;
use gneiss::prelude::parse_prelude;
use gneiss::types::TypeEnv;
use gneiss::{Inferencer, Interpreter, Lexer, Parser};

/// Run a complete program with typeclass support
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
trait Display a =
    val display : a -> String
end

trait Eq a =
    val eq : a -> a -> Bool
end

impl Display for Int =
    let display n = int_to_string n
end

impl Eq for Int =
    let eq a b = a == b
end

let display_if_equal x y =
    if eq x y then display x else "different"

let main () =
    display_if_equal 42 42
"#;
    run_ok(program);
}

#[test]
fn constrained_instance_with_two_constraints() {
    // Instance requires two constraints on type parameter
    let program = r#"
trait Display a =
    val display : a -> String
end

trait Eq a =
    val eq : a -> a -> Bool
end

type Pair a = | MkPair a a

impl Display for Int =
    let display n = int_to_string n
end

impl Eq for Int =
    let eq a b = a == b
end

impl Display for (Pair a) where a : Display, a : Eq =
    let display p = match p with
        | MkPair x y -> if eq x y then "equal pair" else "different pair"
        end
end

let main () =
    print (display (MkPair 1 1));
    print (display (MkPair 1 2));
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
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None
type List a = | Nil | Cons a (List a)

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (Option a) where a : Display =
    let display opt = match opt with
        | Some x -> "Some(" ++ display x ++ ")"
        | None -> "None"
        end
end

impl Display for (List a) where a : Display =
    let display xs = match xs with
        | Nil -> "[]"
        | Cons h t -> display h ++ " :: " ++ display t
        end
end

let main () =
    let list = Cons (Some 1) (Cons None (Cons (Some 2) Nil)) in
    print (display list);
    0
"#;
    run_ok(program);
}

#[test]
fn three_type_parameter_instance() {
    // Instance with three type parameters, all constrained
    let program = r#"
trait Display a =
    val display : a -> String
end

type Triple a b c = | MkTriple a b c

impl Display for Int =
    let display n = int_to_string n
end

impl Display for Bool =
    let display b = if b then "true" else "false"
end

impl Display for String =
    let display s = "\"" ++ s ++ "\""
end

impl Display for (Triple a b c) where a : Display, b : Display, c : Display =
    let display t = match t with
        | MkTriple x y z -> "(" ++ display x ++ ", " ++ display y ++ ", " ++ display z ++ ")"
        end
end

let main () =
    let t = MkTriple 42 true "hello" in
    print (display t);
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
trait Display a =
    val display : a -> String
end

impl Display for Int =
    let display n = int_to_string n
end

let helper1 x = display x
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
trait Display a =
    val display : a -> String
end

impl Display for Int =
    let display n = int_to_string n
end

let apply_display f x = f x

let main () =
    print (apply_display display 42);
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
trait Display a =
    val display : a -> String
end

impl Display for Int =
    let display n = int_to_string n
end

impl Display for Bool =
    let display b = if b then "true" else "false"
end

impl Display for String =
    let display s = s
end

let display_int x = display x
let display_bool x = display x
let display_str x = display x

let main () =
    print (display_int 42);
    print (display_bool true);
    print (display_str "hello");
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
trait Display a =
    val display : a -> String
end

impl Display for Int =
    let display n = int_to_string n
end

let main () =
    let result = display 42 in
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
trait Display a =
    val display : a -> String
end

type Option a = | Some a | None

impl Display for Int =
    let display n = int_to_string n
end

let main () =
    let opt = Some 42 in
    let result = match opt with
        | Some x -> display x
        | None -> "none"
        end
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
trait Display a =
    val display : a -> String
end

type List a = | Nil | Cons a (List a)

impl Display for Int =
    let display n = int_to_string n
end

impl Display for (List a) where a : Display =
    let display xs = "list"
end

let display_int x = display x
let display_list x = display x

let main () =
    print (display_int 42);
    print (display_list (Cons 1 Nil));
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
        | Cons h Nil -> "[" ++ display h ++ "]"
        | Cons h t -> "[" ++ display h ++ ", ...]"
        end
end

let main () =
    print (display Nil);
    print (display (Cons 1 Nil));
    print (display (Cons 1 (Cons 2 Nil)));
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
trait Display a =
    val display : a -> String
end

impl Display for Int =
    let display n = int_to_string n
end

let f x = display x

let main () =
    print (f 42);
    0
"#;
    run_ok(program);
}
