//! Tail Call Optimization tests
//!
//! The CPS interpreter with defunctionalized continuations provides TCO behavior:
//! - Tail calls: continuation stack stays bounded (2-3 frames)
//! - Non-tail calls: continuation stack grows O(n)
//!
//! This means tail-recursive functions can run with O(1) stack space.

use gneiss::test_support::run_program_ok;

// ============================================================================
// Direct Tail Recursion
// ============================================================================

#[test]
fn tco_countdown_100k() {
    // Simple tail-recursive countdown with accumulator
    // This would stack overflow without TCO
    let program = r#"
let rec countdown n acc =
    if n <= 0 then acc
    else countdown (n - 1) (acc + 1)

let main () =
    let result = countdown 100000 0 in
    if result == 100000 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

#[test]
fn tco_sum_list_deep() {
    // Tail-recursive list sum with accumulator
    let program = r#"
let rec make_list n acc =
    if n <= 0 then acc
    else make_list (n - 1) (n :: acc)

let rec sum_tail xs acc =
    match xs with
    | [] -> acc
    | x :: rest -> sum_tail rest (acc + x)
    end

let main () =
    let big_list = make_list 10000 [] in
    let total = sum_tail big_list 0 in
    if total == 50005000 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

#[test]
fn tco_mutual_recursion() {
    // Mutually tail-recursive functions
    let program = r#"
let rec even n = if n == 0 then true else odd (n - 1)
and odd n = if n == 0 then false else even (n - 1)

let main () =
    if even 10000 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

#[test]
fn tco_with_if_branches() {
    // Tail calls in both branches of if
    let program = r#"
let rec search n target =
    if n > target then false
    else if n == target then true
    else search (n + 1) target

let main () =
    if search 0 50000 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

#[test]
fn tco_with_match() {
    // Tail calls in match branches
    let program = r#"
type Direction = | Up | Down

let rec move_until_zero n dir =
    match dir with
    | Up -> if n >= 10000 then n else move_until_zero (n + 1) Up
    | Down -> if n <= 0 then n else move_until_zero (n - 1) Down
    end

let main () =
    let result = move_until_zero 0 Up in
    if result == 10000 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

// ============================================================================
// Continuation-based patterns (validate CPS structure)
// ============================================================================

#[test]
fn tco_continuation_passing() {
    // Classic CPS-style factorial with continuation
    // Tests that closures in tail position work correctly
    let program = r#"
let rec factorial_cps n k =
    if n <= 1 then k 1
    else factorial_cps (n - 1) (fun r -> k (n * r))

let main () =
    let result = factorial_cps 10 id in
    if result == 3628800 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

#[test]
fn tco_trampolined_recursion() {
    // Function that alternates between two recursive patterns
    // Note: these are NOT tail calls due to the + 1
    let program = r#"
let rec ping n =
    if n <= 0 then 0
    else pong (n - 1) + 1
and pong n =
    if n <= 0 then 0
    else ping (n - 1) + 1

let main () =
    let result = ping 1000 in
    if result == 1000 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn tco_with_closure_creation() {
    // Tail recursion that creates closures along the way
    let program = r#"
let rec build_chain n acc =
    if n <= 0 then acc ()
    else
        let next = fun () -> acc () + 1 in
        build_chain (n - 1) next

let main () =
    let result = build_chain 100 (fun () -> 0) in
    if result == 100 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

#[test]
fn tco_with_partial_application() {
    // Tail call involving partial application
    let program = r#"
let add x y = x + y

let rec sum_range start end_ acc =
    if start > end_ then acc
    else sum_range (add start 1) end_ (add acc start)

let main () =
    let result = sum_range 1 1000 0 in
    if result == 500500 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

// ============================================================================
// Large depth stress tests
// ============================================================================

#[test]
fn tco_million_iterations() {
    // One million tail-recursive iterations
    let program = r#"
let rec count n = if n <= 0 then 0 else count (n - 1)

let main () =
    let result = count 1000000 in
    if result == 0 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}

#[test]
#[ignore] // This test takes a few seconds
fn tco_five_million_iterations() {
    // Five million tail-recursive iterations - stress test
    let program = r#"
let rec countdown n acc =
    if n <= 0 then acc
    else countdown (n - 1) (acc + 1)

let main () =
    let result = countdown 5000000 0 in
    if result == 5000000 then print "pass" else print "fail"
"#;
    run_program_ok(program);
}
