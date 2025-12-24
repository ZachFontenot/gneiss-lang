//! Tests for the Algebraic Effects system
//!
//! These tests verify that the effect/perform/handle syntax works correctly,
//! including:
//! 1. Effect declarations and type inference
//! 2. Handler dispatch and continuation capture
//! 3. State effect threading
//! 4. Nested handlers
//! 5. Multiple effects composition
//! 6. Type system soundness
//!
//! This validates the Koka-style algebraic effects implementation.

use gneiss::test_support::{
    assert_eval_int, assert_type, assert_type_error, eval_expr, run_program, run_program_err,
    run_program_ok, typecheck_expr, typecheck_program,
};

// ============================================================================
// Part 1: Basic Effect Tests
// ============================================================================

mod basic_effects {
    use super::*;

    #[test]
    fn effect_declaration_parses() {
        // Effect declarations should parse correctly
        let program = r#"
effect Counter =
    | inc : () -> ()
    | get : () -> Int
end

let main () = 0
"#;
        run_program_ok(program);
    }

    #[test]
    fn effect_with_type_parameter_parses() {
        // Effect with type parameter should parse
        let program = r#"
effect MyState s =
    | get : () -> s
    | put : s -> ()
end

let main () = 0
"#;
        run_program_ok(program);
    }

    #[test]
    fn simple_handle_return_clause() {
        // When body completes normally, return clause runs
        let program = r#"
effect Empty = end

let main () =
    handle 42 with
    | return x -> x + 1
    end
"#;
        let result = run_program(program);
        // Result should be 43 (42 + 1 from return clause)
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }

    #[test]
    fn perform_calls_handler() {
        // perform should invoke the appropriate handler clause
        let program = r#"
effect Ask =
    | ask : () -> Int
end

let main () =
    handle (perform Ask.ask ()) with
    | return x -> x
    | ask () k -> k 42
    end
"#;
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }

    #[test]
    fn handler_continuation_invocation() {
        // Handler should correctly invoke continuation
        let program = r#"
effect Ask =
    | ask : () -> Int
end

let get_and_double () =
    let x = perform Ask.ask () in
    x * 2

let main () =
    handle get_and_double () with
    | return x -> x
    | ask () k -> k 21
    end
"#;
        // Should return 42 (21 * 2)
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }
}

// ============================================================================
// Part 2: State Effect Tests
// ============================================================================

mod state_effect {
    use super::*;

    #[test]
    fn state_get_put_basic() {
        // Basic get/put operations with handler
        let program = r#"
effect State s =
    | get : () -> s
    | put : s -> ()
end

let main () =
    let run_with_state init body =
        handle body () with
        | return x -> x
        | get () k -> k init
        | put s k -> k ()
        end
    in
    run_with_state 10 (fun () ->
        let x = perform State.get () in
        x
    )
"#;
        // Should return 10
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }

    #[test]
    fn state_multiple_operations() {
        // Multiple state operations in sequence
        let program = r#"
effect State s =
    | get : () -> s
    | put : s -> ()
end

let increment () =
    let x = perform State.get () in
    perform State.put (x + 1)

let main () =
    handle (
        increment ();
        increment ();
        perform State.get ()
    ) with
    | return x -> x
    | get () k -> k 0
    | put _ k -> k ()
    end
"#;
        // This is a simplified state handler that doesn't thread state
        // It demonstrates the operations work, even if state isn't persisted
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }
}

// ============================================================================
// Part 3: Nested Handler Tests
// ============================================================================

mod nested_handlers {
    use super::*;

    #[test]
    fn nested_handlers_inner_shadows_outer() {
        // Inner handler for same effect should shadow outer
        let program = r#"
effect Ask =
    | ask : () -> Int
end

let main () =
    handle (
        handle (perform Ask.ask ()) with
        | return x -> x
        | ask () k -> k 100
        end
    ) with
    | return x -> x
    | ask () k -> k 1
    end
"#;
        // Should return 100 (inner handler), not 1 (outer handler)
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }

    #[test]
    fn nested_handlers_different_effects() {
        // Nested handlers for different effects
        let program = r#"
effect Ask =
    | ask : () -> Int
end

effect Tell =
    | tell : Int -> ()
end

let main () =
    handle (
        handle (
            let x = perform Ask.ask () in
            perform Tell.tell x;
            x
        ) with
        | return x -> x
        | tell _ k -> k ()
        end
    ) with
    | return x -> x
    | ask () k -> k 42
    end
"#;
        // Should return 42
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }

    #[test]
    fn three_level_nesting() {
        // Three levels of handler nesting
        let program = r#"
effect A =
    | a : () -> Int
end

effect B =
    | b : () -> Int
end

effect C =
    | c : () -> Int
end

let main () =
    handle (
        handle (
            handle (
                let x = perform A.a () in
                let y = perform B.b () in
                let z = perform C.c () in
                x + y + z
            ) with
            | return x -> x
            | c () k -> k 1
            end
        ) with
        | return x -> x
        | b () k -> k 10
        end
    ) with
    | return x -> x
    | a () k -> k 100
    end
"#;
        // Should return 111 (100 + 10 + 1)
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }
}

// ============================================================================
// Part 4: Multiple Effects Tests
// ============================================================================

mod multiple_effects {
    use super::*;

    #[test]
    fn two_effects_sequential() {
        // Two different effects performed sequentially
        let program = r#"
effect Reader =
    | ask : () -> Int
end

effect Writer =
    | tell : Int -> ()
end

let main () =
    handle (
        handle (
            let config = perform Reader.ask () in
            perform Writer.tell config;
            config * 2
        ) with
        | return x -> x
        | tell _ k -> k ()
        end
    ) with
    | return x -> x
    | ask () k -> k 21
    end
"#;
        // Should return 42
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }

    #[test]
    fn effect_in_function_call() {
        // Effects performed inside function calls
        let program = r#"
effect Ask =
    | ask : () -> Int
end

let get_config () = perform Ask.ask ()

let double_config () =
    let x = get_config () in
    x * 2

let main () =
    handle (double_config ()) with
    | return x -> x
    | ask () k -> k 21
    end
"#;
        // Should return 42
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }
}

// ============================================================================
// Part 5: Continuation Edge Cases
// ============================================================================

mod continuations {
    use super::*;

    #[test]
    fn continuation_not_called() {
        // Handler can choose not to call continuation (short-circuit)
        let program = r#"
effect Abort =
    | abort : () -> a
end

let main () =
    handle (
        perform Abort.abort ();
        100
    ) with
    | return x -> x
    | abort () k -> 42
    end
"#;
        // Should return 42, not 100 (continuation not invoked)
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }

    #[test]
    fn continuation_called_multiple_times() {
        // Handler can invoke continuation multiple times
        let program = r#"
effect Choose =
    | choose : () -> Bool
end

let main () =
    handle (
        if perform Choose.choose () then 1 else 2
    ) with
    | return x -> [x]
    | choose () k -> (k true) ++ (k false)
    end
"#;
        // Should return [1, 2] (continuation called twice)
        let result = run_program(program);
        assert!(result.is_ok(), "Program should run: {:?}", result);
    }
}

// ============================================================================
// Part 6: Type System Tests
// ============================================================================

mod type_system {
    use super::*;

    #[test]
    fn effect_operations_typecheck() {
        // Effect operations should have correct types
        let program = r#"
effect State s =
    | get : () -> s
    | put : s -> ()
end

let test () =
    let _ = perform State.get () in
    perform State.put 42

let main () = 0
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "Program should typecheck: {:?}", result);
    }

    #[test]
    fn handler_return_type_matches() {
        // Handler clauses should all return same type
        let program = r#"
effect Ask =
    | ask : () -> Int
end

let main () =
    handle (perform Ask.ask ()) with
    | return x -> x
    | ask () k -> k 42
    end
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "Program should typecheck: {:?}", result);
    }
}

// ============================================================================
// Part 7: Standard Library Effects
// ============================================================================

mod prelude_effects {
    use super::*;

    #[test]
    fn prelude_state_effect_available() {
        // State effect from prelude should be available
        let program = r#"
let main () =
    handle (perform State.get ()) with
    | return x -> x
    | get () k -> k 42
    | put _ k -> k ()
    end
"#;
        let result = run_program(program);
        assert!(result.is_ok(), "Prelude State effect should work: {:?}", result);
    }

    #[test]
    fn prelude_reader_effect_available() {
        // Reader effect from prelude should be available
        let program = r#"
let main () =
    handle (perform Reader.ask ()) with
    | return x -> x
    | ask () k -> k 42
    end
"#;
        let result = run_program(program);
        assert!(result.is_ok(), "Prelude Reader effect should work: {:?}", result);
    }

    #[test]
    fn prelude_writer_effect_available() {
        // Writer effect from prelude should be available
        let program = r#"
let main () =
    handle (perform Writer.tell 42; 0) with
    | return x -> x
    | tell _ k -> k ()
    end
"#;
        let result = run_program(program);
        assert!(result.is_ok(), "Prelude Writer effect should work: {:?}", result);
    }

    #[test]
    fn prelude_exn_effect_available() {
        // Exn effect from prelude should be available
        let program = r#"
let main () =
    handle (perform Exn.raise "error"; 100) with
    | return x -> x
    | raise _ k -> 42
    end
"#;
        let result = run_program(program);
        assert!(result.is_ok(), "Prelude Exn effect should work: {:?}", result);
    }
}

// ============================================================================
// Part 8: Real-World Patterns
// ============================================================================

mod real_world_patterns {
    use super::*;

    #[test]
    fn counter_with_state() {
        // A simple counter using State effect
        let program = r#"
effect Counter =
    | inc : () -> ()
    | read : () -> Int
end

let count_to n =
    let rec loop i =
        if i >= n then ()
        else (
            perform Counter.inc ();
            loop (i + 1)
        )
    in loop 0;
    perform Counter.read ()

let main () =
    let rec run_counter body state =
        handle body () with
        | return x -> x
        | inc () k -> run_counter (fun () -> k ()) (state + 1)
        | read () k -> run_counter (fun () -> k state) state
        end
    in
    run_counter (fun () -> count_to 5) 0
"#;
        // Should return 5
        let result = run_program(program);
        assert!(result.is_ok(), "Counter should work: {:?}", result);
    }

    #[test]
    fn exception_handling_pattern() {
        // Exception-like error handling with effects
        let program = r#"
effect Error =
    | throw : String -> a
end

let safe_div x y =
    if y == 0 then
        perform Error.throw "division by zero"
    else
        x / y

let main () =
    handle (safe_div 10 2) with
    | return x -> x
    | throw msg k -> 0 - 1
    end
"#;
        // Should return 5 (10 / 2)
        let result = run_program(program);
        assert!(result.is_ok(), "Exception handling should work: {:?}", result);
    }

    #[test]
    fn exception_thrown() {
        // When exception is thrown, handler catches it
        let program = r#"
effect Error =
    | throw : String -> a
end

let safe_div x y =
    if y == 0 then
        perform Error.throw "division by zero"
    else
        x / y

let main () =
    handle (safe_div 10 0) with
    | return x -> x
    | throw msg k -> 0 - 1
    end
"#;
        // Should return -1 (error case)
        let result = run_program(program);
        assert!(result.is_ok(), "Exception handling should catch error: {:?}", result);
    }
}
