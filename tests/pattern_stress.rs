//! Stress tests for pattern matching
//!
//! These tests cover edge cases in pattern matching:
//! - Deep pattern nesting
//! - Multiple patterns per arm
//! - Wildcard and variable patterns
//! - Constructor patterns with multiple fields

use gneiss::test_support::run_program_ok;

// ============================================================================
// Deep Pattern Nesting
// ============================================================================

#[test]
fn deeply_nested_option_pattern() {
    let program = r#"
type Option a = | Some a | None

let main () =
    let deep = Some (Some (Some (Some 42))) in
    match deep with
    | Some (Some (Some (Some x))) -> x
    | Some (Some (Some None)) -> 0
    | Some (Some None) -> 0
    | Some None -> 0
    | None -> 0
    end
"#;
    run_program_ok(program);
}

#[test]
fn nested_tuple_patterns() {
    let program = r#"
let main () =
    let nested = ((1, 2), (3, 4)) in
    match nested with
    | ((a, b), (c, d)) -> a + b + c + d
    end
"#;
    run_program_ok(program);
}

#[test]
fn nested_list_pattern() {
    let program = r#"
type List a = | Nil | Cons a (List a)

let main () =
    let xs = Cons 1 (Cons 2 (Cons 3 Nil)) in
    match xs with
    | Cons a (Cons b (Cons c Nil)) -> a + b + c
    | Cons a (Cons b Nil) -> a + b
    | Cons a Nil -> a
    | Nil -> 0
    end
"#;
    run_program_ok(program);
}

// ============================================================================
// Wildcard Patterns
// ============================================================================

#[test]
fn wildcard_ignores_value() {
    let program = r#"
type Option a = | Some a | None

let main () =
    let opt = Some 42 in
    match opt with
    | Some _ -> 1
    | None -> 0
    end
"#;
    run_program_ok(program);
}

#[test]
fn wildcard_catches_all() {
    let program = r#"
type Color = | Red | Green | Blue | Yellow | Orange

let main () =
    let c = Yellow in
    match c with
    | Red -> 1
    | Green -> 2
    | _ -> 0
    end
"#;
    run_program_ok(program);
}

#[test]
fn multiple_wildcards_in_pattern() {
    let program = r#"
let main () =
    let triple = (1, 2, 3) in
    match triple with
    | (_, x, _) -> x
    end
"#;
    run_program_ok(program);
}

// ============================================================================
// Constructor Patterns with Multiple Fields
// ============================================================================

#[test]
fn constructor_with_three_fields() {
    let program = r#"
type Triple a b c = | MkTriple a b c

let main () =
    let t = MkTriple 1 2 3 in
    match t with
    | MkTriple a b c -> a + b + c
    end
"#;
    run_program_ok(program);
}

#[test]
fn either_type_both_branches() {
    let program = r#"
type Either a b = | Left a | Right b

let main () =
    let x = Left 42 in
    let y = Right "hello" in
    let a = match x with
        | Left n -> n
        | Right _ -> 0
        end
    in
    let b = match y with
        | Left _ -> 0
        | Right s -> string_length s
        end
    in
    a + b
"#;
    run_program_ok(program);
}

// ============================================================================
// Overlapping Patterns (First Match Wins)
// ============================================================================

#[test]
fn overlapping_patterns_first_wins() {
    let program = r#"
type Option a = | Some a | None

let main () =
    let opt = Some 42 in
    match opt with
    | Some 42 -> 1
    | Some x -> 2
    | None -> 3
    end
"#;
    run_program_ok(program);
}

#[test]
fn literal_before_variable() {
    let program = r#"
let main () =
    let x = 0 in
    match x with
    | 0 -> "zero"
    | 1 -> "one"
    | n -> "other"
    end
"#;
    run_program_ok(program);
}

// ============================================================================
// Complex ADT Patterns
// ============================================================================

#[test]
fn binary_tree_pattern() {
    let program = r#"
type Tree a = | Leaf a | Node (Tree a) a (Tree a)

let main () =
    let tree = Node (Leaf 1) 2 (Leaf 3) in
    match tree with
    | Leaf x -> x
    | Node (Leaf l) n (Leaf r) -> l + n + r
    | Node _ n _ -> n
    end
"#;
    run_program_ok(program);
}

#[test]
fn expr_ast_pattern() {
    let program = r#"
type Expr =
    | Lit Int
    | Add Expr Expr
    | Mul Expr Expr

let rec eval e =
    match e with
    | Lit n -> n
    | Add l r -> eval l + eval r
    | Mul l r -> eval l * eval r
    end

let main () =
    let expr = Add (Lit 1) (Mul (Lit 2) (Lit 3)) in
    eval expr
"#;
    run_program_ok(program);
}

// ============================================================================
// Pattern in Let Bindings
// ============================================================================

#[test]
fn tuple_destructuring_in_let() {
    let program = r#"
let main () =
    let (a, b) = (1, 2) in
    a + b
"#;
    run_program_ok(program);
}

#[test]
fn nested_destructuring_in_let() {
    let program = r#"
let main () =
    let ((a, b), c) = ((1, 2), 3) in
    a + b + c
"#;
    run_program_ok(program);
}

// ============================================================================
// Pattern in Function Parameters
// ============================================================================

#[test]
fn tuple_pattern_in_function() {
    let program = r#"
let add_pair (a, b) = a + b

let main () =
    add_pair (3, 4)
"#;
    run_program_ok(program);
}

#[test]
fn constructor_pattern_in_function() {
    let program = r#"
type Option a = | Some a | None

let unwrap_or default opt =
    match opt with
    | Some x -> x
    | None -> default
    end

let main () =
    unwrap_or 0 (Some 42)
"#;
    run_program_ok(program);
}

// ============================================================================
// List Pattern Variations
// ============================================================================

#[test]
fn cons_pattern_head_tail() {
    let program = r#"
let main () =
    let xs = [1, 2, 3, 4, 5] in
    match xs with
    | [] -> 0
    | h :: t -> h
    end
"#;
    run_program_ok(program);
}

#[test]
fn cons_pattern_multiple_elements() {
    let program = r#"
let main () =
    let xs = [1, 2, 3, 4, 5] in
    match xs with
    | [] -> 0
    | a :: b :: c :: rest -> a + b + c
    | a :: b :: rest -> a + b
    | a :: rest -> a
    end
"#;
    run_program_ok(program);
}

#[test]
fn empty_list_pattern() {
    let program = r#"
let main () =
    let xs = [] in
    match xs with
    | [] -> "empty"
    | _ :: _ -> "non-empty"
    end
"#;
    run_program_ok(program);
}

// ============================================================================
// Boolean and Literal Patterns
// ============================================================================

#[test]
fn bool_literal_pattern() {
    let program = r#"
let main () =
    let b = true in
    match b with
    | true -> 1
    | false -> 0
    end
"#;
    run_program_ok(program);
}

#[test]
fn int_literal_patterns() {
    let program = r#"
let main () =
    let n = 42 in
    match n with
    | 0 -> "zero"
    | 1 -> "one"
    | 42 -> "answer"
    | _ -> "other"
    end
"#;
    run_program_ok(program);
}

#[test]
fn string_literal_pattern() {
    let program = r#"
let main () =
    let s = "hello" in
    match s with
    | "hello" -> 1
    | "world" -> 2
    | _ -> 0
    end
"#;
    run_program_ok(program);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn single_arm_match() {
    let program = r#"
let main () =
    let x = 42 in
    match x with
    | n -> n * 2
    end
"#;
    run_program_ok(program);
}

#[test]
fn unit_pattern() {
    let program = r#"
let main () =
    let u = () in
    match u with
    | () -> 42
    end
"#;
    run_program_ok(program);
}

#[test]
fn char_literal_pattern() {
    let program = r#"
let main () =
    let c = 'a' in
    match c with
    | 'a' -> 1
    | 'b' -> 2
    | _ -> 0
    end
"#;
    run_program_ok(program);
}
