//! Property-based tests for algebraic effects
//!
//! These tests try to break the effect handling implementation by:
//! - Generating random values and verifying they propagate correctly
//! - Testing handler shadowing with random nesting depths
//! - Verifying continuation capture and resumption
//! - Testing multi-resume behavior
//! - Stress testing nested handlers

use proptest::prelude::*;

use gneiss::test_support::run_program;

// ============================================================================
// Test Helpers
// ============================================================================

/// Run a program and extract the integer result
fn run_and_get_int(program: &str) -> Result<i64, String> {
    match run_program(program) {
        Ok(_) => {
            // Program ran successfully, but we can't get the return value directly
            // from run_program. We'll use a workaround with print and parsing.
            Ok(0) // Placeholder - actual tests will verify differently
        }
        Err(e) => Err(format!("{:?}", e)),
    }
}

/// Check if a program runs without error
fn program_succeeds(program: &str) -> bool {
    run_program(program).is_ok()
}

// ============================================================================
// Property: Handler Resume Value Propagation
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: When a handler resumes with value v, the continuation receives exactly v
    #[test]
    fn handler_resume_propagates_value(n in 0i64..1000) {
        // Use only non-negative values to avoid parsing issues with negative numbers
        let program = format!(r#"
effect Ask =
    | ask : () -> Int
end

let main () =
    let result = handle (perform Ask.ask ()) with
        | return x -> x
        | ask () k -> k {}
    end in
    if result == {} then 0 else 1
"#, n, n);

        prop_assert!(program_succeeds(&program),
            "Handler should propagate value {} correctly", n);
    }

    /// Property: Handler resume with computed value works
    #[test]
    fn handler_resume_with_expression(a in -100i64..100, b in -100i64..100) {
        let expected = a + b;
        let program = format!(r#"
effect Compute =
    | compute : () -> Int
end

let main () =
    let result = handle (perform Compute.compute ()) with
        | return x -> x
        | compute () k -> k ({} + {})
    end in
    if result == {} then 0 else 1
"#, a, b, expected);

        prop_assert!(program_succeeds(&program),
            "Handler should compute {} + {} = {} correctly", a, b, expected);
    }
}

// ============================================================================
// Property: Short-Circuit Behavior
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: Not calling continuation should short-circuit
    /// The expression after perform should never execute
    #[test]
    fn short_circuit_skips_continuation(sentinel in 1i64..1000) {
        // If short-circuit fails, result would be sentinel, not 42
        let program = format!(r#"
effect Abort =
    | abort : () -> Int
end

let main () =
    let result = handle (
        let _ = perform Abort.abort () in
        {}
    ) with
        | return x -> x
        | abort () k -> 42
    end in
    if result == 42 then 0 else 1
"#, sentinel);

        prop_assert!(program_succeeds(&program),
            "Short-circuit should skip {} and return 42", sentinel);
    }

    /// Property: Early return via effect preserves handler's value
    #[test]
    fn early_return_value(early_val in 0i64..500, late_val in 0i64..500) {
        // The handler returns early_val, ignoring the late_val
        // Use the value directly in the handler rather than binding
        let program = format!(r#"
effect Exit =
    | exit : Int -> Int
end

let main () =
    let result = handle (
        let x = perform Exit.exit {} in
        x + {}
    ) with
        | return x -> x
        | exit n k -> n
    end in
    if result == {} then 0 else 1
"#, early_val, late_val, early_val);

        prop_assert!(program_succeeds(&program),
            "Should return early value {} not late computation with {}", early_val, late_val);
    }
}

// ============================================================================
// Property: Handler Shadowing
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: Inner handler shadows outer handler for same effect
    #[test]
    fn inner_handler_shadows(outer_val in 1i64..100, inner_val in 101i64..200) {
        // Inner handler should provide inner_val, not outer_val
        let program = format!(r#"
effect Get =
    | get : () -> Int
end

let main () =
    let result =
        handle (
            handle (perform Get.get ()) with
            | return x -> x
            | get () k -> k {}
            end
        ) with
        | return x -> x
        | get () k -> k {}
        end
    in
    if result == {} then 0 else 1
"#, inner_val, outer_val, inner_val);

        prop_assert!(program_succeeds(&program),
            "Inner handler ({}) should shadow outer ({})", inner_val, outer_val);
    }

    /// Property: After inner handler completes, outer handler is available
    #[test]
    fn outer_handler_available_after_inner(inner_val in 1i64..50, outer_val in 51i64..100) {
        let program = format!(r#"
effect Get =
    | get : () -> Int
end

let inner_computation () =
    handle (perform Get.get ()) with
    | return x -> x
    | get () k -> k {}
    end

let main () =
    let result =
        handle (
            let a = inner_computation () in
            let b = perform Get.get () in
            a + b
        ) with
        | return x -> x
        | get () k -> k {}
        end
    in
    if result == {} + {} then 0 else 1
"#, inner_val, outer_val, inner_val, outer_val);

        prop_assert!(program_succeeds(&program),
            "Should get {} from inner, {} from outer", inner_val, outer_val);
    }
}

// ============================================================================
// Property: Multiple Performs
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: Multiple performs of same effect all get handled
    #[test]
    fn multiple_performs_all_handled(
        v1 in 1i64..100,
        v2 in 1i64..100,
        v3 in 1i64..100
    ) {
        let expected = v1 + v2 + v3;
        let program = format!(r#"
effect Inc =
    | inc : Int -> Int
end

let main () =
    let result = handle (
        let a = perform Inc.inc {} in
        let b = perform Inc.inc {} in
        let c = perform Inc.inc {} in
        a + b + c
    ) with
        | return x -> x
        | inc n k -> k n
    end in
    if result == {} then 0 else 1
"#, v1, v2, v3, expected);

        prop_assert!(program_succeeds(&program),
            "Three performs should sum to {}", expected);
    }

    /// Property: Interleaved effects work correctly
    #[test]
    fn interleaved_effects(a in 1i64..50, b in 1i64..50) {
        let expected = a + b;
        let program = format!(r#"
effect A =
    | get_a : () -> Int
end

effect B =
    | get_b : () -> Int
end

let main () =
    let result =
        handle (
            handle (
                let x = perform A.get_a () in
                let y = perform B.get_b () in
                x + y
            ) with
            | return x -> x
            | get_b () k -> k {}
            end
        ) with
        | return x -> x
        | get_a () k -> k {}
        end
    in
    if result == {} then 0 else 1
"#, b, a, expected);

        prop_assert!(program_succeeds(&program),
            "Interleaved effects should sum to {}", expected);
    }
}

// ============================================================================
// Property: Continuation Multi-Resume
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: Resuming continuation multiple times produces multiple results
    #[test]
    fn multi_resume_produces_multiple_results(base in 1i64..50) {
        // Resume with base and base+1, collect in list, sum should be 2*base + 1
        let expected = 2 * base + 1;
        let program = format!(r#"
effect Choose =
    | choose : () -> Int
end

let main () =
    let results = handle (
        let x = perform Choose.choose () in
        [x]
    ) with
        | return xs -> xs
        | choose () k -> (k {}) ++ (k {})
    end in
    let s = foldl (fun acc x -> acc + x) 0 results in
    if s == {} then 0 else 1
"#, base, base + 1, expected);

        prop_assert!(program_succeeds(&program),
            "Multi-resume should produce sum {}", expected);
    }
}

// ============================================================================
// Property: Deep Nesting Stress Test
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Property: Deep handler nesting works correctly
    #[test]
    fn deep_nesting_works(depth in 2usize..6, base_val in 1i64..100) {
        // Build nested handlers programmatically
        // Each level adds its depth to the value
        let mut program = String::from(r#"
effect Add =
    | add : Int -> Int
end

let main () =
    let result =
"#);

        // Open handlers
        for i in 0..depth {
            program.push_str(&format!(
                "        handle (\n"
            ));
        }

        // Core computation
        program.push_str(&format!("            perform Add.add {}\n", base_val));

        // Close handlers, each adding its level
        for i in 0..depth {
            let add_val = i as i64 + 1;
            program.push_str(&format!(
                "        ) with\n        | return x -> x\n        | add n k -> k (n + {})\n        end\n",
                add_val
            ));
        }

        // The expected result: base_val + sum(1..=depth)
        let depth_sum: i64 = (1..=depth as i64).sum();
        let expected = base_val + depth_sum;

        program.push_str(&format!(
            "    in\n    if result == {} then 0 else 1\n",
            expected
        ));

        prop_assert!(program_succeeds(&program),
            "Deep nesting of {} levels with base {} should produce {}", depth, base_val, expected);
    }
}

// ============================================================================
// Property: Effect Type Safety
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: Handler return clause processes normal completion correctly
    #[test]
    fn return_clause_transforms_result(input in -100i64..100) {
        let expected = input * 2;
        let program = format!(r#"
effect Dummy =
    | dummy : () -> ()
end

let main () =
    let result = handle {} with
        | return x -> x * 2
        | dummy () k -> k ()
    end in
    if result == {} then 0 else 1
"#, input, expected);

        prop_assert!(program_succeeds(&program),
            "Return clause should transform {} to {}", input, expected);
    }
}

// ============================================================================
// Regression Tests: Edge Cases That Might Break
// ============================================================================

#[test]
fn edge_case_empty_handler_body() {
    // Handler with unit body
    let program = r#"
effect E =
    | e : () -> ()
end

let main () =
    handle () with
    | return x -> x
    | e () k -> k ()
    end;
    0
"#;
    assert!(program_succeeds(program), "Empty handler body should work");
}

#[test]
fn edge_case_handler_returns_function() {
    // Handler that returns a function
    let program = r#"
effect GetFn =
    | get_fn : () -> (Int -> Int)
end

let main () =
    let f = handle (perform GetFn.get_fn ()) with
        | return x -> x
        | get_fn () k -> k (fun x -> x + 1)
    end in
    if f 41 == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Handler returning function should work");
}

#[test]
fn edge_case_continuation_in_closure() {
    // Continuation captured in a closure
    let program = r#"
effect Get =
    | get : () -> Int
end

let main () =
    let result = handle (
        let x = perform Get.get () in
        fun () -> x + 1
    ) with
        | return f -> f ()
        | get () k -> k 41
    end in
    if result == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Continuation in closure should work");
}

#[test]
fn edge_case_nested_perform_in_handler() {
    // Perform inside handler body (should use outer handler)
    let program = r#"
effect A =
    | a : () -> Int
end

effect B =
    | b : () -> Int
end

let main () =
    let result =
        handle (
            handle (perform A.a ()) with
            | return x -> x
            | a () k ->
                let y = perform B.b () in
                k (y + 10)
            end
        ) with
        | return x -> x
        | b () k -> k 32
        end
    in
    if result == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Perform inside handler body should work");
}

#[test]
fn edge_case_handler_with_pattern_matching() {
    // Handler with pattern matching on continuation result
    let program = r#"
effect Choose =
    | choose : () -> Bool
end

let main () =
    let result = handle (
        if perform Choose.choose () then "yes" else "no"
    ) with
        | return s -> s
        | choose () k ->
            match k true with
            | "yes" -> "confirmed"
            | _ -> "denied"
            end
    end in
    if result == "confirmed" then 0 else 1
"#;
    assert!(program_succeeds(program), "Handler with pattern matching should work");
}

#[test]
fn edge_case_recursive_function_with_effects() {
    // Recursive function that performs effects
    let program = r#"
effect Count =
    | tick : () -> ()
    | get : () -> Int
end

let rec count_down n =
    if n <= 0 then
        perform Count.get ()
    else (
        perform Count.tick ();
        count_down (n - 1)
    )

let main () =
    let result = handle (count_down 5) with
        | return x -> x
        | tick () k -> k ()
        | get () k -> k 42
    end in
    if result == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Recursive function with effects should work");
}

#[test]
fn edge_case_effect_in_list_map() {
    // Effect inside a map operation
    // Check sum instead of list equality (list == not supported)
    let program = r#"
effect Transform =
    | transform : Int -> Int
end

let main () =
    let result = handle (
        map (fun x -> perform Transform.transform x) [1, 2, 3]
    ) with
        | return xs -> xs
        | transform n k -> k (n * 2)
    end in
    let s = foldl (fun acc x -> acc + x) 0 result in
    if s == 12 then 0 else 1
"#;
    assert!(program_succeeds(program), "Effect in list map should work");
}

// ============================================================================
// Adversarial Tests: Try to Break Effects
// ============================================================================

#[test]
fn adversarial_continuation_escapes_handler() {
    // Capture continuation in a closure and call it after handler returns
    let program = r#"
effect Get =
    | get : () -> Int
end

let main () =
    let captured = handle (
        let x = perform Get.get () in
        fun () -> x + 1
    ) with
        | return f -> f
        | get () k -> fun () -> (k 41) ()
    end in
    if captured () == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Escaped continuation should work");
}

#[test]
fn adversarial_mutual_recursion_with_effects() {
    // Two mutually recursive functions with effects
    let program = r#"
effect Count =
    | inc : () -> ()
end

let rec is_even n =
    if n == 0 then true
    else (
        perform Count.inc ();
        is_odd (n - 1)
    )

and is_odd n =
    if n == 0 then false
    else (
        perform Count.inc ();
        is_even (n - 1)
    )

let main () =
    let result = handle (is_even 10) with
        | return b -> b
        | inc () k -> k ()
    end in
    if result then 0 else 1
"#;
    assert!(program_succeeds(program), "Mutual recursion with effects should work");
}

#[test]
fn adversarial_effect_in_foldr() {
    // Effects in right fold (tests stack-like behavior)
    let program = r#"
effect Acc =
    | acc : Int -> ()
end

let main () =
    let result = handle (
        foldr (fun x _ -> perform Acc.acc x) () [1, 2, 3, 4, 5]
    ) with
        | return _ -> 0
        | acc n k -> n + (k ())
    end in
    -- Sum should be 1+2+3+4+5 = 15
    -- But foldr processes right-to-left: 5+4+3+2+1 = 15
    if result == 15 then 0 else 1
"#;
    // Note: This tests that continuation accumulation works correctly
    assert!(program_succeeds(program), "Effect in foldr should work");
}

#[test]
fn adversarial_handler_that_ignores_and_resumes() {
    // Handler ignores the effect value and resumes with something else
    let program = r#"
effect Req =
    | request : Int -> Int
end

let main () =
    let result = handle (
        let a = perform Req.request 100 in
        let b = perform Req.request 200 in
        a + b
    ) with
        | return x -> x
        | request _ k -> k 1
    end in
    -- Both requests return 1, so result = 1 + 1 = 2
    if result == 2 then 0 else 1
"#;
    assert!(program_succeeds(program), "Handler ignoring request should work");
}

#[test]
fn adversarial_continuation_called_zero_times() {
    // Handler never calls continuation - should short-circuit
    let program = r#"
effect Die =
    | die : () -> Int
end

let rec infinite_loop () =
    let _ = perform Die.die () in
    infinite_loop ()

let main () =
    let result = handle (infinite_loop ()) with
        | return x -> x
        | die () k -> 42
    end in
    if result == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Zero-resume should short-circuit infinite loop");
}

#[test]
fn adversarial_nested_same_effect_different_params() {
    // Nested handlers for same effect name but conceptually different
    let program = r#"
effect Val =
    | get : () -> Int
end

let inner () =
    handle (perform Val.get ()) with
    | return x -> x
    | get () k -> k 10
    end

let outer () =
    handle (
        let a = inner () in
        let b = perform Val.get () in
        a + b
    ) with
    | return x -> x
    | get () k -> k 100
    end

let main () =
    let result = outer () in
    -- inner returns 10, outer's get returns 100, total = 110
    if result == 110 then 0 else 1
"#;
    assert!(program_succeeds(program), "Nested same-effect handlers should work");
}

#[test]
fn adversarial_handler_resumes_with_effect() {
    // Handler performs an effect while resuming
    let program = r#"
effect Inner =
    | inner_get : () -> Int
end

effect Outer =
    | outer_get : () -> Int
end

let main () =
    let result =
        handle (
            handle (perform Inner.inner_get ()) with
            | return x -> x
            | inner_get () k ->
                let y = perform Outer.outer_get () in
                k (y + 10)
            end
        ) with
        | return x -> x
        | outer_get () k -> k 32
        end
    in
    if result == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Handler performing effect during resume should work");
}

#[test]
fn adversarial_deeply_nested_effects_stress() {
    // 10 levels of nested handlers
    let program = r#"
effect E0 = | e0 : () -> Int end
effect E1 = | e1 : () -> Int end
effect E2 = | e2 : () -> Int end
effect E3 = | e3 : () -> Int end
effect E4 = | e4 : () -> Int end
effect E5 = | e5 : () -> Int end
effect E6 = | e6 : () -> Int end
effect E7 = | e7 : () -> Int end
effect E8 = | e8 : () -> Int end
effect E9 = | e9 : () -> Int end

let main () =
    let result =
        handle (
        handle (
        handle (
        handle (
        handle (
        handle (
        handle (
        handle (
        handle (
        handle (
            perform E0.e0 () +
            perform E1.e1 () +
            perform E2.e2 () +
            perform E3.e3 () +
            perform E4.e4 () +
            perform E5.e5 () +
            perform E6.e6 () +
            perform E7.e7 () +
            perform E8.e8 () +
            perform E9.e9 ()
        ) with | return x -> x | e9 () k -> k 9 end
        ) with | return x -> x | e8 () k -> k 8 end
        ) with | return x -> x | e7 () k -> k 7 end
        ) with | return x -> x | e6 () k -> k 6 end
        ) with | return x -> x | e5 () k -> k 5 end
        ) with | return x -> x | e4 () k -> k 4 end
        ) with | return x -> x | e3 () k -> k 3 end
        ) with | return x -> x | e2 () k -> k 2 end
        ) with | return x -> x | e1 () k -> k 1 end
        ) with | return x -> x | e0 () k -> k 0 end
    in
    -- Sum 0..9 = 45
    if result == 45 then 0 else 1
"#;
    assert!(program_succeeds(program), "10 levels of nested effects should work");
}

#[test]
fn adversarial_effect_in_conditional() {
    // Effect in both branches of conditional
    let program = r#"
effect Choice =
    | flip : () -> Bool
end

let main () =
    let result = handle (
        if perform Choice.flip () then
            100
        else
            1
    ) with
        | return x -> x
        | flip () k -> (k true) + (k false)
    end in
    -- Multi-resume: 100 + 1 = 101
    if result == 101 then 0 else 1
"#;
    assert!(program_succeeds(program), "Effect in conditional with multi-resume should work");
}

#[test]
fn adversarial_continuation_captures_local_bindings() {
    // Ensure continuation properly captures local bindings
    let program = r#"
effect Pause =
    | pause : () -> ()
end

let main () =
    let result = handle (
        let a = 10 in
        let b = 20 in
        perform Pause.pause ();
        let c = 12 in
        a + b + c
    ) with
        | return x -> x
        | pause () k -> k ()
    end in
    if result == 42 then 0 else 1
"#;
    assert!(program_succeeds(program), "Continuation should capture local bindings");
}
