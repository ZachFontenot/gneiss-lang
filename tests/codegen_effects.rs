//! Integration tests for algebraic effects in C codegen
//!
//! These tests verify that effect programs compile and run correctly.

use std::env;
use std::fs;
use std::process::Command;

/// Compile and run a Gneiss program, returning stdout
fn compile_and_run(source: &str, test_name: &str) -> Result<String, String> {
    let runtime_path =
        env::var("GNEISS_RUNTIME").unwrap_or_else(|_| "runtime".to_string());

    // Create temp directory for this test
    let temp_dir = env::temp_dir().join(format!("gneiss_effect_test_{}", test_name));
    fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp dir: {}", e))?;

    let source_file = temp_dir.join("test.gn");
    let c_file = temp_dir.join("test.c");
    let exe_file = temp_dir.join("test");

    // Write source
    fs::write(&source_file, source).map_err(|e| format!("Failed to write source: {}", e))?;

    // Compile Gneiss to C (use --emit-c -o to specify output path)
    let compile_output = Command::new(env!("CARGO_BIN_EXE_gneiss"))
        .args([
            "compile",
            "--emit-c",
            "-o",
            c_file.to_str().unwrap(),
            source_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("Failed to run compiler: {}", e))?;

    if !compile_output.status.success() {
        return Err(format!(
            "Gneiss compilation failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&compile_output.stdout),
            String::from_utf8_lossy(&compile_output.stderr)
        ));
    }

    // Read generated C for error reporting
    let c_code = fs::read_to_string(&c_file)
        .map_err(|e| format!("Failed to read generated C at {:?}: {}", c_file, e))?;

    // Compile C to executable
    let cc_output = Command::new("cc")
        .args([
            "-o",
            exe_file.to_str().unwrap(),
            c_file.to_str().unwrap(),
            &format!("{}/gn_runtime.c", runtime_path),
            &format!("-I{}", runtime_path),
            "-O0",
            "-g",
            "-lm",
        ])
        .output()
        .map_err(|e| format!("Failed to run cc: {}", e))?;

    if !cc_output.status.success() {
        return Err(format!(
            "C compilation failed:\nGenerated C:\n{}\n\nstderr: {}",
            c_code,
            String::from_utf8_lossy(&cc_output.stderr)
        ));
    }

    // Run executable
    let run_output = Command::new(&exe_file)
        .output()
        .map_err(|e| format!("Failed to run executable: {}", e))?;

    let stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&run_output.stderr).to_string();

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);

    if !run_output.status.success() {
        return Err(format!(
            "Execution failed (exit {})\nstdout: {}\nstderr: {}",
            run_output.status, stdout, stderr
        ));
    }

    Ok(stdout)
}

/// Assert that the program produces the expected integer output (for printing tests)
fn expect_result(source: &str, expected: i32, test_name: &str) {
    match compile_and_run(source, test_name) {
        Ok(stdout) => {
            let expected_output = format!("{}\n", expected);
            assert_eq!(
                stdout, expected_output,
                "Test '{}' failed: expected output '{}', got '{}'",
                test_name, expected_output.trim(), stdout.trim()
            );
        }
        Err(e) => {
            panic!("Test '{}' failed to compile/run: {}", test_name, e);
        }
    }
}

/// Assert that the program compiles and runs successfully (exit code 0)
/// Use this with programs that use assert_eq instead of printing
fn expect_success(source: &str, test_name: &str) {
    match compile_and_run(source, test_name) {
        Ok(_) => {
            // Success - program ran without error
        }
        Err(e) => {
            panic!("Test '{}' failed: {}", test_name, e);
        }
    }
}

// ============================================================================
// Basic Effect Tests (without actual effects - just testing the infrastructure)
// ============================================================================

#[test]
fn effect_no_effects_pure_program() {
    // A program with no effects should work as before
    let source = r#"
let main _ = println 42
"#;
    expect_result(source, 42, "effect_no_effects_pure_program");
}

#[test]
fn effect_simple_function_call() {
    // Function calls still work
    let source = r#"
let add x y = x + y
let main _ = println (add 20 22)
"#;
    expect_result(source, 42, "effect_simple_function_call");
}

#[test]
fn effect_hof_still_works() {
    // Higher-order functions still work
    let source = r#"
let apply f x = f x
let inc x = x + 1
let main _ = println (apply inc 41)
"#;
    expect_result(source, 42, "effect_hof_still_works");
}

// ============================================================================
// Effect Declaration Tests
// ============================================================================

#[test]
fn effect_simple_state_get() {
    // Simple state effect - just get
    let source = r#"
effect State =
    | get : () -> Int
end

let main _ =
    let result = handle
        perform State.get ()
    with
    | return x -> x
    | get () k -> k 42
    end
    in assert_eq result 42
"#;
    expect_success(source, "effect_simple_state_get");
}

#[test]
fn effect_state_get_put() {
    // State effect with get and put
    let source = r#"
effect State =
    | get : () -> Int
    | put : Int -> ()
end

let increment () =
    let x = perform State.get () in
    perform State.put (x + 1)

let body () =
    let _ = increment () in
    perform State.get ()

let main _ =
    let result = handle
        body ()
    with
    | return x -> x
    | get () k -> k 10
    | put _ k -> k ()
    end
    in assert_eq result 10
"#;
    expect_success(source, "effect_state_get_put");
}

#[test]
fn effect_exception_throw() {
    // Exception effect - throw without catch
    // Handler doesn't call continuation, just returns the thrown value
    let source = r#"
effect Exn =
    | throw : Int -> a
end

let main _ =
    let result = handle
        perform Exn.throw 42
    with
    | return x -> x
    | throw n k -> n
    end
    in assert_eq result 42
"#;
    expect_success(source, "effect_exception_throw");
}

#[test]
fn effect_nested_handlers() {
    // Nested effect handlers
    let source = r#"
effect Reader =
    | ask : () -> Int
end

let inner () =
    perform Reader.ask ()

let main _ =
    let result = handle
        handle
            inner ()
        with
        | return x -> x + 10
        | ask () k -> k 20
        end
    with
    | return x -> x
    | ask () k -> k 100  -- not used, inner handler catches
    end
    in assert_eq result 30
"#;
    expect_success(source, "effect_nested_handlers");
}

#[test]
fn effect_continuation_called_twice() {
    // Multi-shot continuation (choice effect)
    // Handler calls continuation twice - once with true, once with false
    // Result sums both paths: (true path) + (false path) = 10 + 20 = 30
    let source = r#"
effect Choice =
    | choose : () -> Bool
end

let body () =
    if perform Choice.choose () then 10 else 20

let main _ =
    let result = handle
        body ()
    with
    | return x -> x
    | choose () k ->
        let a = k true in
        let b = k false in
        a + b
    end
    in assert_eq result 30
"#;
    expect_success(source, "effect_continuation_called_twice");
}

#[test]
fn effect_deep_handler_semantics() {
    // Deep handler: effects in resumed code still handled
    let source = r#"
effect Counter =
    | inc : () -> ()
    | get : () -> Int
end

let count_twice () =
    let _ = perform Counter.inc () in
    perform Counter.inc ()

let body () =
    let _ = count_twice () in
    perform Counter.get ()

let main _ =
    let result = handle
        body ()
    with
    | return x -> x
    | inc () k -> k ()
    | get () k -> k 2
    end
    in assert_eq result 2
"#;
    expect_success(source, "effect_deep_handler_semantics");
}
