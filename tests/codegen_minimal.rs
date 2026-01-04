//! Minimal C Code Generation Tests
//!
//! Phase 0: Establish baseline - what currently works in the C backend.
//!
//! These tests compile Gneiss programs to C, then compile and run the C.
//! Each test documents whether it passes or fails to establish our baseline.

use std::fs;
use std::process::Command;

use gneiss::ast::Program;
use gneiss::codegen::{emit_c, lower_mono};
use gneiss::elaborate::elaborate;
use gneiss::infer::Inferencer;
use gneiss::lexer::Lexer;
use gneiss::mono::monomorphize;
use gneiss::parser::Parser;
use gneiss::prelude::parse_prelude;
use gneiss::types::TypeEnv;

/// Compile a Gneiss program to C using the correct pipeline:
/// TAST → Monomorphize → Lower → Emit C
/// This is the ONLY pipeline - per docs/comp_impl_guide.md
fn compile_and_run(source: &str) -> Result<String, String> {
    compile_and_run_impl(source, true)
}

/// Compile and run, expecting a specific output (for tests that explicitly print)
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

/// Assert that the program compiles and runs successfully (exit code 0)
/// Use this with programs that use assert_eq instead of printing
fn expect_success(source: &str) {
    match compile_and_run(source) {
        Ok(_) => {
            // Success - program ran without error
        }
        Err(e) => panic!("Compilation/execution failed: {}", e),
    }
}

/// Compile WITHOUT prelude - for testing core pipeline
fn compile_and_run_no_prelude(source: &str) -> Result<String, String> {
    compile_and_run_impl(source, false)
}

/// Core implementation of compile and run
/// Uses the correct pipeline: TAST → Monomorphize → Lower → Emit C
fn compile_and_run_impl(source: &str, with_prelude: bool) -> Result<String, String> {
    // Parse
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lex error: {:?}", e))?;
    let user_program = Parser::new(tokens)
        .parse_program()
        .map_err(|e| format!("Parse error: {:?}", e))?;

    // Optionally combine with prelude
    let program = if with_prelude {
        let prelude = parse_prelude().map_err(|e| format!("Prelude error: {}", e))?;
        let mut combined_items = prelude.items;
        combined_items.extend(user_program.items);
        Program {
            exports: user_program.exports,
            items: combined_items,
        }
    } else {
        user_program
    };

    // Type check
    let mut inferencer = Inferencer::new();
    let type_env = inferencer
        .infer_program(&program, TypeEnv::new())
        .map_err(|e| format!("Type error: {:?}", e))?;

    // Elaborate to TAST
    let tprogram =
        elaborate(&program, &inferencer, &type_env).map_err(|e| format!("Elaborate: {:?}", e))?;

    // NEW: Monomorphize TAST → MonoProgram
    let mono_program =
        monomorphize(&tprogram).map_err(|e| format!("Monomorphize error: {:?}", e))?;

    // NEW: Lower MonoProgram → CoreIR
    let core_program =
        lower_mono(&mono_program).map_err(|e| format!("Lower error: {:?}", e))?;

    // Emit C
    let c_code = emit_c(&core_program);

    // Use unique temp files per test
    let unique_id = format!(
        "{:?}_{}",
        std::thread::current().id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let c_path = format!("/tmp/gneiss_mono_test_{}.c", unique_id);
    let exe_path = format!("/tmp/gneiss_mono_test_{}", unique_id);
    fs::write(&c_path, &c_code).map_err(|e| format!("Write error: {}", e))?;

    // Find runtime directory
    let runtime_dir = std::env::var("GNEISS_RUNTIME").unwrap_or_else(|_| "runtime".to_string());

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

/// Compile and run WITHOUT prelude, expecting a specific output
fn assert_output_no_prelude(source: &str, expected: &str) {
    match compile_and_run_no_prelude(source) {
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

/// Assert that the program compiles and runs successfully WITHOUT prelude (exit code 0)
fn expect_success_no_prelude(source: &str) {
    match compile_and_run_no_prelude(source) {
        Ok(_) => {
            // Success - program ran without error
        }
        Err(e) => panic!("Compilation/execution failed: {}", e),
    }
}

// ============================================================================
// Phase 0 Tests: Minimal Baseline
// ============================================================================

#[test]
fn baseline_literal_int() {
    // Simplest possible program: return an integer literal
    let source = r#"
let main _ = assert_eq 42 42
"#;
    expect_success(source);
}

#[test]
fn baseline_arithmetic_add() {
    // Simple arithmetic: 1 + 2
    let source = r#"
let main _ = assert_eq (1 + 2) 3
"#;
    expect_success(source);
}

#[test]
fn baseline_arithmetic_complex() {
    // More complex arithmetic
    let source = r#"
let main _ = assert_eq ((10 + 5) * 2 - 3) 27
"#;
    expect_success(source);
}

#[test]
fn baseline_if_true() {
    // Conditional: if true branch
    let source = r#"
let main _ = assert_eq (if true then 1 else 2) 1
"#;
    expect_success(source);
}

#[test]
fn baseline_if_false() {
    // Conditional: if false branch
    let source = r#"
let main _ = assert_eq (if false then 1 else 2) 2
"#;
    expect_success(source);
}

#[test]
fn baseline_let_binding() {
    // Simple let binding
    let source = r#"
let main _ =
    let x = 10 in
    let y = 20 in
    assert_eq (x + y) 30
"#;
    expect_success(source);
}

#[test]
fn baseline_function_call() {
    // Named function call
    let source = r#"
let double x = x * 2
let main _ = assert_eq (double 21) 42
"#;
    expect_success(source);
}

#[test]
fn baseline_recursive_factorial() {
    // Recursive function
    let source = r#"
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)

let main _ = assert_eq (factorial 5) 120
"#;
    expect_success(source);
}

// ============================================================================
// Known Failure Tests (document what doesn't work yet)
// ============================================================================

#[test]
fn baseline_lambda_simple() {
    // Simple lambda without captures
    let source = r#"
let main _ = assert_eq ((fun x -> x + 1) 5) 6
"#;
    expect_success(source);
}

#[test]
fn baseline_lambda_identity() {
    // Identity lambda
    let source = r#"
let main _ = assert_eq ((fun x -> x) 42) 42
"#;
    expect_success(source);
}

#[test]
fn baseline_lambda_let_bound() {
    // Lambda bound to a variable
    let source = r#"
let main _ =
    let f = fun x -> x + 1 in
    assert_eq (f 5) 6
"#;
    expect_success(source);
}

#[test]
fn baseline_lambda_capture() {
    // Lambda with captured variable
    let source = r#"
let main _ =
    let y = 10 in
    let f = fun x -> x + y in
    assert_eq (f 5) 15
"#;
    expect_success(source);
}

#[test]
fn baseline_curried_function() {
    // Curried function (returns a closure)
    let source = r#"
let add x = fun y -> x + y
let main _ = assert_eq (add 3 4) 7
"#;
    expect_success(source);
}

#[test]
fn baseline_map() {
    // Higher-order function
    let source = r#"
let main _ = assert_eq (length (map (fun x -> x + 1) [1, 2, 3])) 3
"#;
    expect_success(source);
}

#[test]
fn baseline_filter() {
    // Higher-order filter function
    let source = r#"
let main _ = assert_eq (length (filter (fun x -> x > 2) [1, 2, 3, 4])) 2
"#;
    expect_success(source);
}

#[test]
fn baseline_foldl() {
    // Higher-order fold function
    let source = r#"
let main _ = assert_eq (foldl (fun acc x -> acc + x) 0 [1, 2, 3]) 6
"#;
    expect_success(source);
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
let main _ = assert_eq (compose double add1 5) 12
"#;
    expect_success(source);
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
let main _ = assert_eq (list_sum [1, 2, 3, 4, 5]) 15
"#;
    expect_success(source);
}

#[test]
fn phase5_pipe_operator() {
    // Pipe operator with function composition
    let source = r#"
let double x = x * 2
let add1 x = x + 1
let main _ = assert_eq (5 |> add1 |> double) 12
"#;
    expect_success(source);
}

#[test]
fn phase5_nested_hof() {
    // Nested higher-order functions
    let source = r#"
let main _ = assert_eq (length (map (fun x -> x + 1) (filter (fun x -> x > 1) [1, 2, 3, 4]))) 3
"#;
    expect_success(source);
}

#[test]
fn phase5_list_concat() {
    // List concatenation using list_append (no ++ operator)
    let source = r#"
let main _ = assert_eq (length (list_append [1, 2] [3, 4, 5])) 5
"#;
    expect_success(source);
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
  assert_eq (eval expr) 10
"#;
    expect_success(source);
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
  assert_eq (tree_sum t) 6
"#;
    expect_success(source);
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
  assert_eq (get_or_default e1 0 + get_or_default e2 100) 142
"#;
    expect_success(source);
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

let main _ = assert_eq (if is_even 10 then 1 else 0) 1
"#;
    expect_success(source);
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
  let result = match reverse [1, 2, 3] with
    | x :: _ -> x
    | [] -> 0
    end
  in assert_eq result 3
"#;
    expect_success(source);
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

let main _ = assert_eq (length (take 2 [1, 2, 3, 4, 5])) 2
"#;
    expect_success(source);
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
  assert_eq (foldl (fun acc x -> acc + x) 0 sums) 66
"#;
    expect_success(source);
}

#[test]
fn phase5_option_some() {
    // Option Some case
    let source = r#"
let main _ =
    let result = match Some 42 with
      | Some x -> x
      | None -> 0
      end
    in assert_eq result 42
"#;
    expect_success(source);
}

#[test]
fn phase5_option_none() {
    // Option None case
    let source = r#"
let main _ =
    let result = match None with
      | Some x -> x
      | None -> 99
      end
    in assert_eq result 99
"#;
    expect_success(source);
}

// These tests verify that polymorphic == correctly dispatches through the
// type-resolved comparison (PolyEq -> StringEq for String types).

#[test]
fn phase5_show_int() {
    // Show trait for Int
    let source = r#"
let main _ = assert_eq (show 42) "42"
"#;
    expect_success(source);
}

#[test]
fn phase5_show_string() {
    // Show trait for String
    let source = r#"
let main _ = assert_eq (show "hello") "hello"
"#;
    expect_success(source);
}

#[test]
fn phase5_show_bool() {
    // Show trait for Bool
    let source = r#"
let main _ = assert_eq (show true) "true"
"#;
    expect_success(source);
}

#[test]
fn trait_method_through_polymorphic_fn() {
    // Test polymorphic function that uses trait method
    // This exercises DictMethodCall resolution
    let source = r#"
let stringify x = show x
let main _ = assert_eq (stringify 42) "42"
"#;
    expect_success(source);
}

#[test]
fn trait_method_nested_poly() {
    // Nested polymorphic calls
    let source = r#"
let stringify x = show x
let double x = stringify x ++ stringify x
let main _ = assert_eq (double 5) "55"
"#;
    expect_success(source);
}

// Note: print tests removed - requires proper monomorphization pass
// Will be re-added after Phase 1 completion (see docs/comp_impl_guide.md)

// ============================================================================
// New Monomorphization Pipeline Tests
// These test the CORRECT architecture: TAST → Mono → CoreIR → C
// ============================================================================

#[test]
fn mono_pipeline_literal_no_prelude() {
    // Test the new pipeline with a simple literal - NO PRELUDE
    // This is the most basic test of the new infrastructure
    let source = r#"
let main _ = assert (42 == 42)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_arithmetic_no_prelude() {
    // Test arithmetic through new pipeline - NO PRELUDE
    let source = r#"
let main _ = assert (1 + 2 == 3)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_function_call_no_prelude() {
    // Test function calls - the key milestone from comp_impl_guide.md:
    // "let add x y = x + y; let main _ = add 1 2"
    // NO PRELUDE - pure user code
    let source = r#"
let add x y = x + y
let main _ = assert (add 1 2 == 3)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_lambda_simple_no_prelude() {
    // Phase 2 test: Simple lambda without captures
    let source = r#"
let main _ = assert ((fun x -> x + 1) 5 == 6)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_lambda_identity_no_prelude() {
    // Phase 2 test: Identity lambda
    let source = r#"
let main _ = assert ((fun x -> x) 42 == 42)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_lambda_let_bound_no_prelude() {
    // Phase 2 test: Lambda bound to variable
    let source = r#"
let main _ =
    let f = fun x -> x + 1 in
    assert (f 5 == 6)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_lambda_capture_no_prelude() {
    // Phase 3 test: Lambda with captured variable
    let source = r#"
let main _ =
    let y = 10 in
    let f = fun x -> x + y in
    assert (f 5 == 15)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_curried_function_no_prelude() {
    // Phase 3 test: Curried function (returns a closure)
    let source = r#"
let add x = fun y -> x + y
let main _ = assert (add 3 4 == 7)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_hof_apply_no_prelude() {
    // Phase 4 test: Higher-order function - apply
    let source = r#"
let apply f x = f x
let double x = x * 2
let main _ = assert (apply double 21 == 42)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_hof_compose_no_prelude() {
    // Phase 4 test: Function composition
    let source = r#"
let compose f g x = f (g x)
let double x = x + x
let add1 x = x + 1
let main _ = assert (compose double add1 5 == 12)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_recursive_list_no_prelude() {
    // Test recursive function on lists - no prelude
    let source = r#"
let rec list_sum xs =
    match xs with
    | [] -> 0
    | x :: rest -> x + list_sum rest
    end
let main _ = assert (list_sum [1, 2, 3, 4, 5] == 15)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_custom_map_no_prelude() {
    // Implement map ourselves to test HOF on lists
    let source = r#"
let rec my_map f xs =
    match xs with
    | [] -> []
    | x :: rest -> f x :: my_map f rest
    end

let rec list_sum xs =
    match xs with
    | [] -> 0
    | x :: rest -> x + list_sum rest
    end

let main _ = assert (list_sum (my_map (fun x -> x + 1) [1, 2, 3]) == 9)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_with_prelude_length() {
    // Test using prelude's length function
    let source = r#"
let main _ = assert_eq (length [1, 2, 3, 4, 5]) 5
"#;
    expect_success(source);
}

#[test]
fn mono_pipeline_tail_recursive_helper() {
    // Test tail-recursive helper pattern: let rec go ... in go
    // This tests that local recursive functions get properly wrapped as closures
    let source = r#"
let rec reverse_acc acc xs =
    match xs with
    | [] -> acc
    | x :: rest -> reverse_acc (x :: acc) rest
    end

let rec list_sum xs =
    match xs with
    | [] -> 0
    | x :: rest -> x + list_sum rest
    end

let main _ = assert (list_sum (reverse_acc [] [1, 2, 3]) == 6)
"#;
    expect_success_no_prelude(source);
}

#[test]
fn mono_pipeline_local_rec_as_value() {
    // Test that local recursive function can be used as a value (passed to HOF)
    let source = r#"
let apply_twice f x = f (f x)

let main _ =
    let rec double n = n + n in
    assert (apply_twice double 5 == 20)
"#;
    expect_success_no_prelude(source);
}

// ============================================================================
// Runtime Boundary Check Tests
// ============================================================================

#[test]
fn boundary_normal_division() {
    // Normal division should work as expected
    let source = r#"
let main _ = assert_eq (10 / 2) 5
"#;
    expect_success(source);
}

#[test]
fn boundary_normal_modulo() {
    // Normal modulo should work as expected
    let source = r#"
let main _ = assert_eq (10 % 3) 1
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_div_success() {
    // safe_div returns Ok for non-zero divisor
    let source = r#"
let main _ =
    match safe_div 10 2 with
    | Ok v -> assert_eq v 5
    | Err _ -> panic "should be Ok"
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_div_zero() {
    // safe_div returns Err for zero divisor
    let source = r#"
let main _ =
    match safe_div 10 0 with
    | Ok _ -> panic "should be Err"
    | Err msg -> assert_eq msg "division by zero"
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_mod_success() {
    // safe_mod returns Ok for non-zero divisor
    let source = r#"
let main _ =
    match safe_mod 10 3 with
    | Ok v -> assert_eq v 1
    | Err _ -> panic "should be Ok"
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_mod_zero() {
    // safe_mod returns Err for zero divisor
    let source = r#"
let main _ =
    match safe_mod 10 0 with
    | Ok _ -> panic "should be Err"
    | Err msg -> assert_eq msg "modulo by zero"
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_head() {
    // head returns Option - Some for non-empty list
    let source = r#"
let main _ =
    match head [1, 2, 3] with
    | Some x -> assert_eq x 1
    | None -> panic "should be Some"
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_head_empty() {
    // head returns Option - None for empty list
    let source = r#"
let main _ =
    match head [] with
    | Some _ -> panic "should be None"
    | None -> ()
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_tail() {
    // tail returns Option - Some for non-empty list
    let source = r#"
let main _ =
    match tail [1, 2, 3] with
    | Some rest -> assert_eq (length rest) 2
    | None -> panic "should be Some"
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_safe_tail_empty() {
    // tail returns Option - None for empty list
    let source = r#"
let main _ =
    match tail [] with
    | Some _ -> panic "should be None"
    | None -> ()
    end
"#;
    expect_success(source);
}

#[test]
fn boundary_head_unsafe_success() {
    // head_unsafe works on non-empty list
    let source = r#"
let main _ = assert_eq (head_unsafe [1, 2, 3]) 1
"#;
    expect_success(source);
}

#[test]
fn boundary_tail_unsafe_success() {
    // tail_unsafe works on non-empty list
    let source = r#"
let main _ = assert_eq (length (tail_unsafe [1, 2, 3])) 2
"#;
    expect_success(source);
}

// =============================================================================
// Dictionary Tests
// =============================================================================

#[test]
fn dict_new_and_insert() {
    let source = r#"
let main _ =
    let d = Dict.new () in
    let d = Dict.insert "key" 42 d in
    let result = Dict.get "key" d in
    match result with
    | Some v -> assert_eq v 42
    | None -> panic "Dict.get failed"
    end
"#;
    expect_success(source);
}

#[test]
fn dict_contains() {
    let source = r#"
let main _ =
    let d = Dict.new () in
    let d = Dict.insert "foo" 1 d in
    let _ = assert (Dict.contains "foo" d) in
    assert (not (Dict.contains "bar" d))
"#;
    expect_success(source);
}

#[test]
fn dict_size() {
    let source = r#"
let main _ =
    let d = Dict.new () in
    let _ = assert_eq (Dict.size d) 0 in
    let d = Dict.insert "a" 1 d in
    let _ = assert_eq (Dict.size d) 1 in
    let d = Dict.insert "b" 2 d in
    assert_eq (Dict.size d) 2
"#;
    expect_success(source);
}

#[test]
fn dict_is_empty() {
    let source = r#"
let main _ =
    let d = Dict.new () in
    let _ = assert (Dict.isEmpty d) in
    let d = Dict.insert "key" 1 d in
    assert (not (Dict.isEmpty d))
"#;
    expect_success(source);
}

#[test]
fn dict_remove() {
    let source = r#"
let main _ =
    let d = Dict.new () in
    let d = Dict.insert "a" 1 d in
    let d = Dict.insert "b" 2 d in
    let _ = assert_eq (Dict.size d) 2 in
    let d = Dict.remove "a" d in
    let _ = assert_eq (Dict.size d) 1 in
    let _ = assert (not (Dict.contains "a" d)) in
    assert (Dict.contains "b" d)
"#;
    expect_success(source);
}

#[test]
fn dict_get_or_default() {
    let source = r#"
let main _ =
    let d = Dict.new () in
    let d = Dict.insert "exists" 42 d in
    let _ = assert_eq (Dict.getOrDefault 0 "exists" d) 42 in
    assert_eq (Dict.getOrDefault 99 "missing" d) 99
"#;
    expect_success(source);
}

#[test]
fn dict_update_existing_key() {
    let source = r#"
let main _ =
    let d = Dict.new () in
    let d = Dict.insert "key" 1 d in
    let d = Dict.insert "key" 2 d in
    let _ = assert_eq (Dict.size d) 1 in
    match Dict.get "key" d with
    | Some v -> assert_eq v 2
    | None -> panic "key should exist"
    end
"#;
    expect_success(source);
}

#[test]
fn dict_merge() {
    let source = r#"
let main _ =
    let d1 = Dict.new () in
    let d1 = Dict.insert "a" 1 d1 in
    let d1 = Dict.insert "b" 2 d1 in
    let d2 = Dict.new () in
    let d2 = Dict.insert "b" 20 d2 in
    let d2 = Dict.insert "c" 3 d2 in
    let merged = Dict.merge d1 d2 in
    let _ = assert_eq (Dict.getOrDefault 0 "a" merged) 1 in
    let _ = assert_eq (Dict.getOrDefault 0 "b" merged) 20 in
    assert_eq (Dict.getOrDefault 0 "c" merged) 3
"#;
    expect_success(source);
}
