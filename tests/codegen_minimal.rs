//! Minimal C Code Generation Tests
//!
//! Phase 0: Establish baseline - what currently works in the C backend.
//!
//! These tests compile Gneiss programs to C, then compile and run the C.
//! Each test documents whether it passes or fails to establish our baseline.

use std::fs;
use std::process::Command;

use gneiss::ast::Program;
use gneiss::codegen::{emit_c, lower_tprogram};
use gneiss::elaborate::elaborate;
use gneiss::infer::Inferencer;
use gneiss::lexer::Lexer;
use gneiss::parser::Parser;
use gneiss::prelude::parse_prelude;

/// Compile a Gneiss program to C, compile the C, run it, and return stdout
fn compile_and_run(source: &str) -> Result<String, String> {
    // Parse
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lex error: {:?}", e))?;
    let user_program = Parser::new(tokens)
        .parse_program()
        .map_err(|e| format!("Parse error: {:?}", e))?;

    // Combine with prelude
    let prelude = parse_prelude().map_err(|e| format!("Prelude error: {}", e))?;
    let mut combined_items = prelude.items;
    combined_items.extend(user_program.items);
    let program = Program {
        exports: user_program.exports,
        items: combined_items,
    };

    // Type check
    let mut inferencer = Inferencer::new();
    let type_env = inferencer
        .infer_program(&program)
        .map_err(|e| format!("Type error: {:?}", e))?;

    // Elaborate
    let tprogram =
        elaborate(&program, &inferencer, &type_env).map_err(|e| format!("Elaborate: {:?}", e))?;

    // Lower to Core IR
    let core_program =
        lower_tprogram(&tprogram).map_err(|e| format!("Lower error: {:?}", e))?;

    // Emit C
    let c_code = emit_c(&core_program);

    // Use unique temp files per test (thread ID + timestamp)
    let unique_id = format!("{:?}_{}", std::thread::current().id(), std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
    let c_path = format!("/tmp/gneiss_test_{}.c", unique_id);
    let exe_path = format!("/tmp/gneiss_test_{}", unique_id);
    fs::write(&c_path, &c_code).map_err(|e| format!("Write error: {}", e))?;

    // Find runtime directory
    let runtime_dir = std::env::var("GNEISS_RUNTIME")
        .unwrap_or_else(|_| "runtime".to_string());

    // Compile C
    let compile_result = Command::new("cc")
        .args([
            "-o",
            &exe_path,
            &c_path,
            &format!("{}/gn_runtime.c", runtime_dir),
            &format!("-I{}", runtime_dir),
            "-lm",
        ])
        .output()
        .map_err(|e| format!("CC failed to start: {}", e))?;

    if !compile_result.status.success() {
        let stderr = String::from_utf8_lossy(&compile_result.stderr);
        return Err(format!("C compilation failed:\n{}", stderr));
    }

    // Run executable
    let run_result = Command::new(&exe_path)
        .output()
        .map_err(|e| format!("Run failed: {}", e))?;

    let stdout = String::from_utf8_lossy(&run_result.stdout).to_string();
    let stderr = String::from_utf8_lossy(&run_result.stderr).to_string();

    if !run_result.status.success() {
        return Err(format!(
            "Execution failed (exit {})\nstdout: {}\nstderr: {}",
            run_result.status, stdout, stderr
        ));
    }

    Ok(stdout)
}

/// Compile and run, expecting a specific output
fn assert_output(source: &str, expected: &str) {
    match compile_and_run(source) {
        Ok(output) => {
            let output = output.trim();
            assert_eq!(
                output, expected,
                "Output mismatch.\nExpected: {}\nGot: {}",
                expected, output
            );
        }
        Err(e) => panic!("Compilation/execution failed: {}", e),
    }
}

/// Compile and run, expecting it to fail (documents known failures)
fn expect_failure(source: &str, description: &str) {
    match compile_and_run(source) {
        Ok(output) => {
            panic!(
                "Expected failure but got success!\nDescription: {}\nOutput: {}",
                description, output
            );
        }
        Err(_) => {
            // Expected - this is a known failure
            eprintln!("[KNOWN FAILURE] {}", description);
        }
    }
}

// ============================================================================
// Phase 0 Tests: Minimal Baseline
// ============================================================================

#[test]
fn baseline_literal_int() {
    // Simplest possible program: return an integer literal
    let source = r#"
let main _ = 42
"#;
    assert_output(source, "42");
}

#[test]
fn baseline_arithmetic_add() {
    // Simple arithmetic: 1 + 2
    let source = r#"
let main _ = 1 + 2
"#;
    assert_output(source, "3");
}

#[test]
fn baseline_arithmetic_complex() {
    // More complex arithmetic
    let source = r#"
let main _ = (10 + 5) * 2 - 3
"#;
    assert_output(source, "27");
}

#[test]
fn baseline_if_true() {
    // Conditional: if true branch
    let source = r#"
let main _ = if true then 1 else 2
"#;
    assert_output(source, "1");
}

#[test]
fn baseline_if_false() {
    // Conditional: if false branch
    let source = r#"
let main _ = if false then 1 else 2
"#;
    assert_output(source, "2");
}

#[test]
fn baseline_let_binding() {
    // Simple let binding
    let source = r#"
let main _ =
    let x = 10 in
    let y = 20 in
    x + y
"#;
    assert_output(source, "30");
}

#[test]
fn baseline_function_call() {
    // Named function call
    let source = r#"
let double x = x * 2
let main _ = double 21
"#;
    assert_output(source, "42");
}

#[test]
fn baseline_recursive_factorial() {
    // Recursive function
    let source = r#"
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)

let main _ = factorial 5
"#;
    assert_output(source, "120");
}

// ============================================================================
// Known Failure Tests (document what doesn't work yet)
// ============================================================================

#[test]
fn baseline_lambda_simple() {
    // Simple lambda without captures
    let source = r#"
let main _ = (fun x -> x + 1) 5
"#;
    assert_output(source, "6");
}

#[test]
fn baseline_lambda_identity() {
    // Identity lambda
    let source = r#"
let main _ = (fun x -> x) 42
"#;
    assert_output(source, "42");
}

#[test]
fn baseline_lambda_let_bound() {
    // Lambda bound to a variable
    let source = r#"
let main _ =
    let f = fun x -> x + 1 in
    f 5
"#;
    assert_output(source, "6");
}

#[test]
fn baseline_lambda_capture() {
    // Lambda with captured variable
    let source = r#"
let main _ =
    let y = 10 in
    let f = fun x -> x + y in
    f 5
"#;
    assert_output(source, "15");
}

#[test]
fn baseline_curried_function() {
    // Curried function (returns a closure)
    let source = r#"
let add x = fun y -> x + y
let main _ = add 3 4
"#;
    assert_output(source, "7");
}

#[test]
fn baseline_map() {
    // Higher-order function
    let source = r#"
let main _ = length (map (fun x -> x + 1) [1, 2, 3])
"#;
    assert_output(source, "3");
}

#[test]
fn baseline_filter() {
    // Higher-order filter function
    let source = r#"
let main _ = length (filter (fun x -> x > 2) [1, 2, 3, 4])
"#;
    assert_output(source, "2");
}

#[test]
fn baseline_foldl() {
    // Higher-order fold function
    let source = r#"
let main _ = foldl (fun acc x -> acc + x) 0 [1, 2, 3]
"#;
    assert_output(source, "6");
}

// ============================================================================
// Phase 5 Tests: More complex programs
// ============================================================================

#[test]
fn phase5_compose() {
    // Function composition
    let source = r#"
let compose f g x = f (g x)
let double x = x + x
let add1 x = x + 1
let main _ = compose double add1 5
"#;
    assert_output(source, "12");
}

#[test]
fn phase5_list_sum() {
    // Recursive list sum
    let source = r#"
let rec list_sum xs =
    match xs with
    | [] -> 0
    | x :: rest -> x + list_sum rest
    end
let main _ = list_sum [1, 2, 3, 4, 5]
"#;
    assert_output(source, "15");
}

#[test]
fn phase5_pipe_operator() {
    // Pipe operator with function composition
    let source = r#"
let double x = x * 2
let add1 x = x + 1
let main _ = 5 |> add1 |> double
"#;
    assert_output(source, "12");
}

#[test]
fn phase5_nested_hof() {
    // Nested higher-order functions
    let source = r#"
let main _ = length (map (fun x -> x + 1) (filter (fun x -> x > 1) [1, 2, 3, 4]))
"#;
    assert_output(source, "3");
}

#[test]
#[ignore] // ++ is currently only implemented for strings, not lists - use concat function instead
fn phase5_list_concat() {
    // List concatenation - should use concat or a named function
    let source = r#"
let main _ = length ([1, 2] ++ [3, 4, 5])
"#;
    assert_output(source, "5");
}

#[test]
fn phase5_custom_adt() {
    // Custom ADT: expression evaluator
    let source = r#"
type Expr
  = Num Int
  | Add Expr Expr
  | Mul Expr Expr

let rec eval expr =
  match expr with
  | Num i -> i
  | Add l r -> eval l + eval r
  | Mul l r -> eval l * eval r
  end

let main _ =
  let expr = Add (Mul (Num 2) (Num 3)) (Num 4) in
  eval expr
"#;
    assert_output(source, "10");
}

#[test]
fn phase5_tree_sum() {
    // Binary tree sum
    let source = r#"
type Tree = Leaf Int | Node Tree Tree

let rec tree_sum t =
  match t with
  | Leaf n -> n
  | Node l r -> tree_sum l + tree_sum r
  end

let main _ =
  let t = Node (Node (Leaf 1) (Leaf 2)) (Leaf 3) in
  tree_sum t
"#;
    assert_output(source, "6");
}

#[test]
fn phase5_either_type() {
    // Either type handling
    let source = r#"
type Either a b = Left a | Right b

let get_or_default e d =
  match e with
  | Left _ -> d
  | Right x -> x
  end

let main _ =
  let e1 = Right 42 in
  let e2 = Left "error" in
  get_or_default e1 0 + get_or_default e2 100
"#;
    assert_output(source, "142");
}

#[test]
fn phase5_mutual_recursion() {
    // Mutual recursion (even/odd check)
    let source = r#"
let rec is_even n =
  if n == 0 then true
  else is_odd (n - 1)
and is_odd n =
  if n == 0 then false
  else is_even (n - 1)

let main _ =
  if is_even 10 then 1 else 0
"#;
    assert_output(source, "1");
}

#[test]
fn phase5_reverse_list() {
    // List reversal
    let source = r#"
let rec reverse_acc acc xs =
  match xs with
  | [] -> acc
  | x :: rest -> reverse_acc (x :: acc) rest
  end

let reverse xs = reverse_acc [] xs

let main _ =
  match reverse [1, 2, 3] with
  | x :: _ -> x
  | [] -> 0
  end
"#;
    assert_output(source, "3");
}

#[test]
fn phase5_take_n() {
    // Take first n elements
    let source = r#"
let rec take n xs =
  if n <= 0 then []
  else match xs with
    | [] -> []
    | x :: rest -> x :: take (n - 1) rest
    end

let main _ = length (take 2 [1, 2, 3, 4, 5])
"#;
    assert_output(source, "2");
}

#[test]
fn phase5_zip_with() {
    // Zip two lists with a function
    let source = r#"
let rec zip_with f xs ys =
  match xs with
  | [] -> []
  | x :: rest_x ->
    match ys with
    | [] -> []
    | y :: rest_y -> f x y :: zip_with f rest_x rest_y
    end
  end

let main _ =
  let sums = zip_with (fun a b -> a + b) [1, 2, 3] [10, 20, 30] in
  foldl (fun acc x -> acc + x) 0 sums
"#;
    assert_output(source, "66");
}

#[test]
fn phase5_option_some() {
    // Option Some case
    let source = r#"
let main _ =
    match Some 42 with
    | Some x -> x
    | None -> 0
    end
"#;
    assert_output(source, "42");
}

#[test]
fn phase5_option_none() {
    // Option None case
    let source = r#"
let main _ =
    match None with
    | Some x -> x
    | None -> 99
    end
"#;
    assert_output(source, "99");
}

#[test]
fn phase5_show_int() {
    // Show trait for Int
    let source = r#"
let main _ =
    let s = show 42 in
    let _ = io_print s in
    0
"#;
    // Output is "42" from io_print + "0" from return value
    assert_output(source, "420");
}

#[test]
fn phase5_show_string() {
    // Show trait for String
    let source = r#"
let main _ =
    let s = show "hello" in
    let _ = io_print s in
    0
"#;
    assert_output(source, "hello0");
}

#[test]
fn phase5_show_bool() {
    // Show trait for Bool
    let source = r#"
let main _ =
    let s = show true in
    let _ = io_print s in
    0
"#;
    assert_output(source, "true0");
}
