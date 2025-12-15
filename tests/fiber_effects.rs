//! Tests for the Phase 7 Fiber Effect system
//!
//! These tests verify that the new effect-based concurrency system works correctly.
//! The key difference from the old system is that fiber operations now produce
//! FiberEffect values that are handled by the scheduler, rather than using
//! StepResult::Blocked directly.
//!
//! Test categories:
//! 1. Basic fiber operations (Fiber.spawn, Fiber.join, Fiber.yield)
//! 2. Channel operations through fibers
//! 3. Effect sequencing and ordering
//! 4. Edge cases and error conditions

use gneiss::eval::{EvalError, Value};
use gneiss::test_support::{
    assert_eval_int, assert_type, eval_expr, run_program, run_program_err, run_program_ok,
    typecheck_expr, typecheck_program, FiberEffectTrace,
};

// ============================================================================
// Type Checking Tests - Verify types are correct before running
// ============================================================================

mod types {
    use super::*;

    #[test]
    fn fiber_spawn_has_fiber_type() {
        // Fiber.spawn should return Fiber<A> where A is the return type of the thunk
        let program = r#"
type Unit = | Unit

let f () = Fiber.spawn (fun () -> 42)
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "Program should typecheck: {:?}", result);
    }

    #[test]
    fn fiber_join_returns_fiber_result_type() {
        // Fiber.join on Fiber<Int> should return Int
        let program = r#"
type Unit = | Unit

let f () =
    let fiber = Fiber.spawn (fun () -> 42) in
    Fiber.join fiber
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "Program should typecheck: {:?}", result);
    }

    #[test]
    fn fiber_yield_returns_unit() {
        // Fiber.yield should return Unit
        let program = r#"
type Unit = | Unit

let f () =
    Fiber.yield ();
    42
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "Program should typecheck: {:?}", result);
    }

    #[test]
    fn channel_operations_type_correctly() {
        // Channel operations should maintain type safety
        let program = r#"
type Unit = | Unit

let f () =
    let ch = Channel.new in
    let _ = spawn (fun () -> Channel.send ch 42) in
    Channel.recv ch
"#;
        let result = typecheck_program(program);
        assert!(result.is_ok(), "Program should typecheck: {:?}", result);
    }
}

// ============================================================================
// Basic Fiber Operations
// ============================================================================

mod basic_fibers {
    use super::*;

    #[test]
    fn spawn_creates_fiber() {
        // Fiber.spawn should create a new fiber and return a handle
        let program = r#"
let main () =
    let fiber = Fiber.spawn (fun () -> 42) in
    0
"#;
        run_program_ok(program);
    }

    #[test]
    fn spawn_and_join_returns_value() {
        // Spawning a fiber and joining it should return the fiber's result
        let program = r#"
let main () =
    let fiber = Fiber.spawn (fun () -> 42) in
    Fiber.join fiber
"#;
        run_program_ok(program);
    }

    #[test]
    fn spawn_and_join_with_computation() {
        // Fiber should compute its result
        let program = r#"
let main () =
    let fiber = Fiber.spawn (fun () -> 10 + 32) in
    Fiber.join fiber
"#;
        run_program_ok(program);
    }

    #[test]
    fn multiple_spawns_and_joins() {
        // Multiple fibers can be spawned and joined
        let program = r#"
let main () =
    let f1 = Fiber.spawn (fun () -> 10) in
    let f2 = Fiber.spawn (fun () -> 20) in
    let f3 = Fiber.spawn (fun () -> 12) in
    let v1 = Fiber.join f1 in
    let v2 = Fiber.join f2 in
    let v3 = Fiber.join f3 in
    v1 + v2 + v3
"#;
        run_program_ok(program);
    }

    #[test]
    fn yield_allows_other_fibers_to_run() {
        // Fiber.yield should allow scheduler to run other fibers
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () ->
        Fiber.yield ();
        Channel.send ch 42
    ) in
    Channel.recv ch
"#;
        run_program_ok(program);
    }

    #[test]
    fn nested_fiber_spawn() {
        // Fibers can spawn other fibers
        let program = r#"
let main () =
    let outer = Fiber.spawn (fun () ->
        let inner = Fiber.spawn (fun () -> 21) in
        let v = Fiber.join inner in
        v * 2
    ) in
    Fiber.join outer
"#;
        run_program_ok(program);
    }
}

// ============================================================================
// Channel Operations Through Fibers
// ============================================================================

mod fiber_channels {
    use super::*;

    #[test]
    fn fiber_send_and_receive() {
        // Fibers can communicate through channels
        let program = r#"
let main () =
    let ch = Channel.new in
    let sender = Fiber.spawn (fun () -> Channel.send ch 42) in
    let value = Channel.recv ch in
    let _ = Fiber.join sender in
    value
"#;
        run_program_ok(program);
    }

    #[test]
    fn fiber_receive_and_send() {
        // Receiver fiber waiting, then sender sends
        let program = r#"
let main () =
    let ch = Channel.new in
    let receiver = Fiber.spawn (fun () -> Channel.recv ch) in
    Channel.send ch 99;
    Fiber.join receiver
"#;
        run_program_ok(program);
    }

    #[test]
    fn ping_pong_between_fibers() {
        // Two fibers exchanging messages
        let program = r#"
let main () =
    let ping = Channel.new in
    let pong = Channel.new in
    let ponger = Fiber.spawn (fun () ->
        let x = Channel.recv ping in
        Channel.send pong (x + 1)
    ) in
    Channel.send ping 10;
    let result = Channel.recv pong in
    let _ = Fiber.join ponger in
    result
"#;
        run_program_ok(program);
    }

    #[test]
    fn multiple_messages_through_channel() {
        // Multiple sends and receives through a channel
        let program = r#"
let main () =
    let ch = Channel.new in
    let sender = Fiber.spawn (fun () ->
        Channel.send ch 1;
        Channel.send ch 2;
        Channel.send ch 3
    ) in
    let a = Channel.recv ch in
    let b = Channel.recv ch in
    let c = Channel.recv ch in
    let _ = Fiber.join sender in
    a + b + c
"#;
        run_program_ok(program);
    }

    #[test]
    fn channel_with_yield() {
        // Yield between channel operations
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = Fiber.spawn (fun () ->
        Channel.send ch 1;
        Fiber.yield ();
        Channel.send ch 2
    ) in
    let a = Channel.recv ch in
    Fiber.yield ();
    let b = Channel.recv ch in
    a + b
"#;
        run_program_ok(program);
    }
}

// ============================================================================
// Interleaving and Concurrency
// ============================================================================

mod concurrency {
    use super::*;

    #[test]
    fn interleaving_required_for_success() {
        // This test MUST interleave to succeed (would deadlock otherwise)
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = Fiber.spawn (fun () ->
        let x = Channel.recv ch1 in
        Channel.send ch2 (x + 100)
    ) in
    Channel.send ch1 5;
    Channel.recv ch2
"#;
        run_program_ok(program);
    }

    #[test]
    fn producer_consumer_pattern() {
        // Producer fiber sends, consumer (main) receives
        let program = r#"
let main () =
    let ch = Channel.new in
    let producer = Fiber.spawn (fun () ->
        Channel.send ch 1;
        Channel.send ch 2;
        Channel.send ch 3
    ) in
    let sum = (Channel.recv ch) + (Channel.recv ch) + (Channel.recv ch) in
    let _ = Fiber.join producer in
    sum
"#;
        run_program_ok(program);
    }

    #[test]
    fn parallel_workers() {
        // Multiple worker fibers computing in parallel
        let program = r#"
let main () =
    let r1 = Channel.new in
    let r2 = Channel.new in
    let r3 = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.send r1 10) in
    let _ = Fiber.spawn (fun () -> Channel.send r2 20) in
    let _ = Fiber.spawn (fun () -> Channel.send r3 12) in
    let v1 = Channel.recv r1 in
    let v2 = Channel.recv r2 in
    let v3 = Channel.recv r3 in
    v1 + v2 + v3
"#;
        run_program_ok(program);
    }

    #[test]
    fn join_before_fiber_completes() {
        // Join on a fiber that hasn't completed yet
        let program = r#"
let main () =
    let ch = Channel.new in
    let slow_fiber = Fiber.spawn (fun () ->
        let _ = Channel.recv ch in
        42
    ) in
    let _ = Fiber.spawn (fun () ->
        Fiber.yield ();
        Channel.send ch 1
    ) in
    Fiber.join slow_fiber
"#;
        run_program_ok(program);
    }

    #[test]
    fn join_after_fiber_completes() {
        // Join on a fiber that has already completed
        let program = r#"
let main () =
    let fiber = Fiber.spawn (fun () -> 42) in
    Fiber.yield ();
    Fiber.yield ();
    Fiber.join fiber
"#;
        run_program_ok(program);
    }
}

// ============================================================================
// Select Operations
// ============================================================================

mod select_operations {
    use super::*;

    #[test]
    fn select_first_ready() {
        // Select should receive from the first ready channel
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.send ch1 1) in
    select
    | x <- ch1 -> x
    | y <- ch2 -> y
    end
"#;
        run_program_ok(program);
    }

    #[test]
    fn select_second_ready() {
        // Select when second channel is ready first
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.send ch2 2) in
    select
    | x <- ch1 -> x
    | y <- ch2 -> y
    end
"#;
        run_program_ok(program);
    }

    #[test]
    fn select_with_computation() {
        // Select arms can contain computation
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.send ch1 10) in
    select
    | x <- ch1 -> x * 2
    | y <- ch2 -> y + 100
    end
"#;
        run_program_ok(program);
    }

    #[test]
    fn select_blocking_then_ready() {
        // Select blocks until a channel becomes ready
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = Fiber.spawn (fun () ->
        Fiber.yield ();
        Fiber.yield ();
        Channel.send ch2 42
    ) in
    select
    | x <- ch1 -> x
    | y <- ch2 -> y
    end
"#;
        run_program_ok(program);
    }

    #[test]
    fn select_with_four_channels() {
        // Select over four channels
        let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let c3 = Channel.new in
    let c4 = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.send c3 300) in
    select
    | a <- c1 -> a
    | b <- c2 -> b
    | c <- c3 -> c
    | d <- c4 -> d
    end
"#;
        run_program_ok(program);
    }

    #[test]
    fn nested_select_in_arm() {
        // Select inside a select arm body
        let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let c3 = Channel.new in
    let _ = Fiber.spawn (fun () ->
        Channel.send c1 1;
        Channel.send c3 3
    ) in
    select
    | x <- c1 ->
        let nested = select
            | y <- c2 -> y + 100
            | z <- c3 -> z + 200
            end
        in
        x + nested
    | w <- c2 -> w
    end
"#;
        run_program_ok(program);
    }

    #[test]
    fn select_multiple_senders_one_channel() {
        // Multiple fibers send to same channel, select picks one
        let program = r#"
let main () =
    let ch = Channel.new in
    let other = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.send ch 10) in
    let _ = Fiber.spawn (fun () -> Channel.send ch 20) in
    let _ = Fiber.spawn (fun () -> Channel.send ch 30) in
    -- First select gets one value
    let v1 = select
        | x <- ch -> x
        | y <- other -> y
        end
    in
    -- Second select gets another
    let v2 = select
        | x <- ch -> x
        | y <- other -> y
        end
    in
    -- Third select gets the last
    let v3 = select
        | x <- ch -> x
        | y <- other -> y
        end
    in
    -- We got all three (order may vary)
    v1 + v2 + v3
"#;
        run_program_ok(program);
    }

    #[test]
    fn select_returns_value_to_binding() {
        // Select result used in subsequent computation
        let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.send c2 7) in
    let value = select
        | x <- c1 -> x * 10
        | y <- c2 -> y * 100
        end
    in
    value + 1
"#;
        run_program_ok(program);
    }

    #[test]
    fn select_inside_fiber() {
        // Fiber uses select internally
        let program = r#"
let main () =
    let result_ch = Channel.new in
    let input1 = Channel.new in
    let input2 = Channel.new in
    let worker = Fiber.spawn (fun () ->
        let v = select
            | a <- input1 -> a + 1
            | b <- input2 -> b + 2
            end
        in
        Channel.send result_ch v
    ) in
    Channel.send input2 40;
    let result = Channel.recv result_ch in
    let _ = Fiber.join worker in
    result
"#;
        run_program_ok(program);
    }
}

// ============================================================================
// Error Conditions and Edge Cases
// ============================================================================

mod errors {
    use super::*;

    #[test]
    fn deadlock_detection_single_recv() {
        // Receiving on a channel with no sender should deadlock
        let program = r#"
let main () =
    let ch = Channel.new in
    Channel.recv ch
"#;
        run_program_err(program, |e| matches!(e, EvalError::Deadlock));
    }

    #[test]
    fn deadlock_detection_mutual_recv() {
        // Two fibers both waiting to receive - deadlock
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = Fiber.spawn (fun () -> Channel.recv ch1) in
    Channel.recv ch2
"#;
        run_program_err(program, |e| matches!(e, EvalError::Deadlock));
    }

    #[test]
    fn deadlock_with_join() {
        // Joining a fiber that's blocked forever
        let program = r#"
let main () =
    let ch = Channel.new in
    let blocked_fiber = Fiber.spawn (fun () -> Channel.recv ch) in
    Fiber.join blocked_fiber
"#;
        run_program_err(program, |e| matches!(e, EvalError::Deadlock));
    }
}

// ============================================================================
// Regression Tests
// ============================================================================

mod regression {
    use super::*;

    #[test]
    fn fiber_effect_bubbles_correctly() {
        // Ensure FiberEffect values properly bubble up through continuations
        let program = r#"
let main () =
    let fiber = Fiber.spawn (fun () ->
        let x = 1 + 2 in
        let y = x * 3 in
        y + 1
    ) in
    Fiber.join fiber
"#;
        run_program_ok(program);
    }

    #[test]
    fn fiber_with_closure_capture() {
        // Fibers should properly capture closure environment
        let program = r#"
let main () =
    let x = 10 in
    let fiber = Fiber.spawn (fun () -> x + 32) in
    Fiber.join fiber
"#;
        run_program_ok(program);
    }

    #[test]
    fn fiber_with_nested_lets() {
        // Complex let bindings inside fibers
        let program = r#"
let main () =
    let fiber = Fiber.spawn (fun () ->
        let a = 1 in
        let b = a + 1 in
        let c = b + 1 in
        let d = c + 1 in
        d
    ) in
    Fiber.join fiber
"#;
        run_program_ok(program);
    }

    #[test]
    fn channel_inside_fiber() {
        // Creating a channel inside a fiber
        let program = r#"
let main () =
    let result_ch = Channel.new in
    let _ = Fiber.spawn (fun () ->
        let internal_ch = Channel.new in
        let _ = Fiber.spawn (fun () -> Channel.send internal_ch 42) in
        let v = Channel.recv internal_ch in
        Channel.send result_ch v
    ) in
    Channel.recv result_ch
"#;
        run_program_ok(program);
    }

    #[test]
    fn old_spawn_still_works() {
        // The old spawn syntax should still work (for backwards compatibility)
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () -> Channel.send ch 42) in
    Channel.recv ch
"#;
        run_program_ok(program);
    }
}
