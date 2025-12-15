//! Stress tests for the fiber scheduler
//!
//! These tests verify the scheduler handles edge cases:
//! - High fiber counts
//! - Deep fiber nesting
//! - Rapid channel communication
//! - Many concurrent channels

use gneiss::test_support::{run_program_err, run_program_ok};
use gneiss::eval::EvalError;

// ============================================================================
// High Fiber Count Tests
// ============================================================================

#[test]
fn stress_50_fibers_join_all() {
    // Spawn 50 fibers, each computing a value, join all and sum
    let program = r#"
let main () =
    let ch = Channel.new in
    -- Spawn 50 workers
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    let _ = spawn (fun () -> Channel.send ch 1) in
    -- Collect all results
    let rec collect n acc =
        if n <= 0 then acc
        else collect (n - 1) (acc + (Channel.recv ch))
    in
    collect 50 0
"#;
    run_program_ok(program);
}

#[test]
fn stress_fiber_spawn_in_loop() {
    // Spawn fibers using a recursive pattern
    let program = r#"
let main () =
    let results = Channel.new in
    let rec spawn_workers n =
        if n <= 0 then ()
        else
            let _ = spawn (fun () -> Channel.send results n) in
            spawn_workers (n - 1)
    in
    spawn_workers 20;
    -- Collect 20 results
    let rec collect n acc =
        if n <= 0 then acc
        else collect (n - 1) (acc + (Channel.recv results))
    in
    collect 20 0
"#;
    run_program_ok(program);
}

// ============================================================================
// Deep Fiber Nesting Tests
// ============================================================================

#[test]
fn nested_fiber_depth_10() {
    // 10 levels of nested fibers
    let program = r#"
let main () =
    let f1 = Fiber.spawn (fun () ->
        let f2 = Fiber.spawn (fun () ->
            let f3 = Fiber.spawn (fun () ->
                let f4 = Fiber.spawn (fun () ->
                    let f5 = Fiber.spawn (fun () ->
                        let f6 = Fiber.spawn (fun () ->
                            let f7 = Fiber.spawn (fun () ->
                                let f8 = Fiber.spawn (fun () ->
                                    let f9 = Fiber.spawn (fun () ->
                                        let f10 = Fiber.spawn (fun () -> 42) in
                                        Fiber.join f10
                                    ) in
                                    Fiber.join f9
                                ) in
                                Fiber.join f8
                            ) in
                            Fiber.join f7
                        ) in
                        Fiber.join f6
                    ) in
                    Fiber.join f5
                ) in
                Fiber.join f4
            ) in
            Fiber.join f3
        ) in
        Fiber.join f2
    ) in
    Fiber.join f1
"#;
    run_program_ok(program);
}

#[test]
fn nested_fiber_with_channels() {
    // Nested fibers communicating through channels
    let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let ch3 = Channel.new in
    let outer = Fiber.spawn (fun () ->
        let inner1 = Fiber.spawn (fun () ->
            let inner2 = Fiber.spawn (fun () ->
                Channel.send ch3 100
            ) in
            let v = Channel.recv ch3 in
            Channel.send ch2 (v + 10);
            Fiber.join inner2
        ) in
        let v = Channel.recv ch2 in
        Channel.send ch1 (v + 1);
        Fiber.join inner1
    ) in
    let result = Channel.recv ch1 in
    let _ = Fiber.join outer in
    result
"#;
    run_program_ok(program);
}

// ============================================================================
// Rapid Channel Communication Tests
// ============================================================================

#[test]
fn rapid_pingpong_5_messages() {
    // Ping-pong 5 messages between two fibers
    // Note: Recursive channel loops have type inference issues (gneiss-lang-wzl)
    // Using unrolled pattern instead
    let program = r#"
let main () =
    let ping = Channel.new in
    let pong = Channel.new in
    -- Ponger responds to each ping
    let _ = spawn (fun () ->
        let a = Channel.recv ping in Channel.send pong a;
        let b = Channel.recv ping in Channel.send pong b;
        let c = Channel.recv ping in Channel.send pong c;
        let d = Channel.recv ping in Channel.send pong d;
        let e = Channel.recv ping in Channel.send pong e
    ) in
    -- Pinger sends and receives
    Channel.send ping 1;
    let r1 = Channel.recv pong in
    Channel.send ping 2;
    let r2 = Channel.recv pong in
    Channel.send ping 3;
    let r3 = Channel.recv pong in
    Channel.send ping 4;
    let r4 = Channel.recv pong in
    Channel.send ping 5;
    let r5 = Channel.recv pong in
    r1 + r2 + r3 + r4 + r5
"#;
    run_program_ok(program);
}

#[test]
fn pipeline_chain_5_stages() {
    // 5-stage pipeline, each adding 1
    let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let c3 = Channel.new in
    let c4 = Channel.new in
    let c5 = Channel.new in
    -- Stage 1 -> Stage 2
    let _ = spawn (fun () ->
        let v = Channel.recv c1 in
        Channel.send c2 (v + 1)
    ) in
    -- Stage 2 -> Stage 3
    let _ = spawn (fun () ->
        let v = Channel.recv c2 in
        Channel.send c3 (v + 1)
    ) in
    -- Stage 3 -> Stage 4
    let _ = spawn (fun () ->
        let v = Channel.recv c3 in
        Channel.send c4 (v + 1)
    ) in
    -- Stage 4 -> Stage 5
    let _ = spawn (fun () ->
        let v = Channel.recv c4 in
        Channel.send c5 (v + 1)
    ) in
    -- Send initial value
    Channel.send c1 0;
    -- Receive final value (should be 4)
    Channel.recv c5
"#;
    run_program_ok(program);
}

// ============================================================================
// Many Channels Tests
// ============================================================================

#[test]
fn stress_20_channels() {
    // Create 20 channels and use them all
    let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let c3 = Channel.new in
    let c4 = Channel.new in
    let c5 = Channel.new in
    let c6 = Channel.new in
    let c7 = Channel.new in
    let c8 = Channel.new in
    let c9 = Channel.new in
    let c10 = Channel.new in
    let c11 = Channel.new in
    let c12 = Channel.new in
    let c13 = Channel.new in
    let c14 = Channel.new in
    let c15 = Channel.new in
    let c16 = Channel.new in
    let c17 = Channel.new in
    let c18 = Channel.new in
    let c19 = Channel.new in
    let c20 = Channel.new in
    -- Spawn senders for each
    let _ = spawn (fun () -> Channel.send c1 1) in
    let _ = spawn (fun () -> Channel.send c2 2) in
    let _ = spawn (fun () -> Channel.send c3 3) in
    let _ = spawn (fun () -> Channel.send c4 4) in
    let _ = spawn (fun () -> Channel.send c5 5) in
    let _ = spawn (fun () -> Channel.send c6 6) in
    let _ = spawn (fun () -> Channel.send c7 7) in
    let _ = spawn (fun () -> Channel.send c8 8) in
    let _ = spawn (fun () -> Channel.send c9 9) in
    let _ = spawn (fun () -> Channel.send c10 10) in
    let _ = spawn (fun () -> Channel.send c11 11) in
    let _ = spawn (fun () -> Channel.send c12 12) in
    let _ = spawn (fun () -> Channel.send c13 13) in
    let _ = spawn (fun () -> Channel.send c14 14) in
    let _ = spawn (fun () -> Channel.send c15 15) in
    let _ = spawn (fun () -> Channel.send c16 16) in
    let _ = spawn (fun () -> Channel.send c17 17) in
    let _ = spawn (fun () -> Channel.send c18 18) in
    let _ = spawn (fun () -> Channel.send c19 19) in
    let _ = spawn (fun () -> Channel.send c20 20) in
    -- Receive from all and sum
    let v1 = Channel.recv c1 in
    let v2 = Channel.recv c2 in
    let v3 = Channel.recv c3 in
    let v4 = Channel.recv c4 in
    let v5 = Channel.recv c5 in
    let v6 = Channel.recv c6 in
    let v7 = Channel.recv c7 in
    let v8 = Channel.recv c8 in
    let v9 = Channel.recv c9 in
    let v10 = Channel.recv c10 in
    let v11 = Channel.recv c11 in
    let v12 = Channel.recv c12 in
    let v13 = Channel.recv c13 in
    let v14 = Channel.recv c14 in
    let v15 = Channel.recv c15 in
    let v16 = Channel.recv c16 in
    let v17 = Channel.recv c17 in
    let v18 = Channel.recv c18 in
    let v19 = Channel.recv c19 in
    let v20 = Channel.recv c20 in
    v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 +
    v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19 + v20
"#;
    run_program_ok(program);
}

#[test]
fn channel_created_inside_fiber() {
    // Fiber creates its own channel and uses it internally
    let program = r#"
let main () =
    let result_ch = Channel.new in
    let worker = Fiber.spawn (fun () ->
        let internal = Channel.new in
        let _ = spawn (fun () -> Channel.send internal 42) in
        let v = Channel.recv internal in
        Channel.send result_ch (v * 2)
    ) in
    let result = Channel.recv result_ch in
    let _ = Fiber.join worker in
    result
"#;
    run_program_ok(program);
}

// ============================================================================
// Select Stress Tests
// ============================================================================

#[test]
fn select_with_three_channels() {
    // Select over three channels
    let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let c3 = Channel.new in
    let _ = spawn (fun () -> Channel.send c2 200) in
    select
    | x <- c1 -> x
    | y <- c2 -> y
    | z <- c3 -> z
    end
"#;
    run_program_ok(program);
}

#[test]
fn select_multiple_rounds() {
    // Use select multiple times in sequence
    let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let _ = spawn (fun () ->
        Channel.send c1 1;
        Channel.send c2 2;
        Channel.send c1 3
    ) in
    let a = select
        | x <- c1 -> x
        | y <- c2 -> y
        end
    in
    let b = select
        | x <- c1 -> x
        | y <- c2 -> y
        end
    in
    let c = select
        | x <- c1 -> x
        | y <- c2 -> y
        end
    in
    a + b + c
"#;
    run_program_ok(program);
}

#[test]
fn select_all_blocking_then_one_ready() {
    // All channels block initially, then one becomes ready
    let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let c3 = Channel.new in
    let _ = spawn (fun () ->
        Fiber.yield ();
        Fiber.yield ();
        Channel.send c3 42
    ) in
    select
    | x <- c1 -> x
    | y <- c2 -> y
    | z <- c3 -> z
    end
"#;
    run_program_ok(program);
}

// ============================================================================
// Fiber Join Edge Cases
// ============================================================================

#[test]
fn join_already_completed_fiber() {
    // Join a fiber that completed before join is called
    let program = r#"
let main () =
    let fiber = Fiber.spawn (fun () -> 42) in
    -- Yield multiple times to let fiber complete
    Fiber.yield ();
    Fiber.yield ();
    Fiber.yield ();
    Fiber.join fiber
"#;
    run_program_ok(program);
}

#[test]
fn multiple_joins_same_fiber() {
    // Multiple fibers trying to join the same target
    let program = r#"
let main () =
    let target = Fiber.spawn (fun () -> 100) in
    let results = Channel.new in
    -- Two fibers both want to join the target
    let j1 = Fiber.spawn (fun () ->
        let v = Fiber.join target in
        Channel.send results v
    ) in
    let j2 = Fiber.spawn (fun () ->
        let v = Fiber.join target in
        Channel.send results v
    ) in
    -- Collect both results
    let r1 = Channel.recv results in
    let r2 = Channel.recv results in
    r1 + r2
"#;
    run_program_ok(program);
}

// ============================================================================
// Yield Stress Tests
// ============================================================================

#[test]
fn rapid_yield_alternation() {
    // Two fibers rapidly yielding and communicating
    let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () ->
        Fiber.yield ();
        Fiber.yield ();
        Fiber.yield ();
        Channel.send ch 1;
        Fiber.yield ();
        Fiber.yield ();
        Channel.send ch 2;
        Fiber.yield ();
        Channel.send ch 3
    ) in
    let a = Channel.recv ch in
    Fiber.yield ();
    let b = Channel.recv ch in
    Fiber.yield ();
    let c = Channel.recv ch in
    a + b + c
"#;
    run_program_ok(program);
}

// ============================================================================
// Deadlock Detection Tests
// ============================================================================

#[test]
fn deadlock_cycle_of_two() {
    // Two fibers waiting on each other
    let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let _ = spawn (fun () ->
        let _ = Channel.recv c1 in
        Channel.send c2 1
    ) in
    let _ = Channel.recv c2 in
    Channel.send c1 1
"#;
    run_program_err(program, |e| matches!(e, EvalError::Deadlock));
}

#[test]
fn deadlock_all_waiting() {
    // Three fibers all waiting to receive
    let program = r#"
let main () =
    let c1 = Channel.new in
    let c2 = Channel.new in
    let c3 = Channel.new in
    let _ = spawn (fun () -> Channel.recv c1) in
    let _ = spawn (fun () -> Channel.recv c2) in
    Channel.recv c3
"#;
    run_program_err(program, |e| matches!(e, EvalError::Deadlock));
}

// ============================================================================
// Known Limitations Tests
// ============================================================================

// This test documents a known limitation (gneiss-lang-wzl):
// Recursive functions that use channel operations in a loop pattern
// have type inference issues. The workaround is to use unrolled patterns
// or non-recursive coordination.
#[test]
#[ignore] // Known limitation - type inference issue
fn known_limitation_recursive_channel_loop() {
    let program = r#"
let main () =
    let ping = Channel.new in
    let pong = Channel.new in
    let _ = spawn (fun () ->
        let rec ponger count =
            if count <= 0 then ()
            else
                let n = Channel.recv ping in
                Channel.send pong n;
                ponger (count - 1)
        in
        ponger 5
    ) in
    let rec pinger count acc =
        if count <= 0 then acc
        else
            Channel.send ping count;
            let n = Channel.recv pong in
            pinger (count - 1) (acc + n)
    in
    pinger 5 0
"#;
    run_program_ok(program);
}

// ============================================================================
// Scheduler Fairness Tests
// ============================================================================

#[test]
fn fairness_round_robin() {
    // Fibers should be scheduled fairly (not starve any)
    let program = r#"
let main () =
    let results = Channel.new in
    -- Three fibers each sending their ID
    let _ = spawn (fun () ->
        Fiber.yield ();
        Channel.send results 1
    ) in
    let _ = spawn (fun () ->
        Fiber.yield ();
        Channel.send results 2
    ) in
    let _ = spawn (fun () ->
        Fiber.yield ();
        Channel.send results 3
    ) in
    -- All should eventually complete
    let a = Channel.recv results in
    let b = Channel.recv results in
    let c = Channel.recv results in
    a + b + c
"#;
    run_program_ok(program);
}
