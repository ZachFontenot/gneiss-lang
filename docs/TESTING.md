# Gneiss Testing Philosophy

This document defines the testing philosophy for the Gneiss language project.
**All contributors must follow these guidelines.** This philosophy is mandatory,
not optional.

## Why This Matters

During the C backend implementation attempt, we encountered compounding issues:
- AST structure drift (generated code didn't match expected nodes)
- Type inference bugs (types inferred incorrectly or inconsistently)
- Integration gaps (individual parts worked but combined system failed)
- General lack of visibility into semantic stages

These problems stemmed from relying too heavily on output tests ("write program,
run it, see what happens"). Output tests obscure **where** bugs occur in the
compilation pipeline.

## Core Principle: Test the Process, Not Just the Output

The primary purpose of tests is to verify that the *process* of compilation
and evaluation proceeds correctly. Simply checking that a program produces
expected output obscures bugs in intermediate stages.

## The Testing Pyramid

Tests should be organized in layers, with each layer providing different guarantees:

```
                    +-------------------+
                    |   Output Tests    |  <- Least Important (Layer 5)
                    +-------------------+
               +---------------------------+
               |     Runtime Tests         |  <- Layer 4
               +---------------------------+
          +-------------------------------------+
          |       Type Inference Tests         |  <- Layer 3
          +-------------------------------------+
     +---------------------------------------------+
     |         Rejection Tests                     |  <- Layer 2
     +---------------------------------------------+
+----------------------------------------------------------+
|              AST/Parser Tests                            |  <- Most Important (Layer 1)
+----------------------------------------------------------+
```

**Bottom layers are the foundation.** Don't skip to output tests without the
semantic tests underneath.

---

## Layer 1: AST/Parser Tests (Foundation)

**Purpose**: Verify that source code produces the expected AST structure.

**When to use**: For every syntactic construct in the language.

### Good Example
```rust
#[test]
fn let_binding_parses_correctly() {
    let expr = parse_expr("let x = 1 + 2 in x * 3").unwrap();

    // Verify top-level is a Let
    let ExprKind::Let { pattern, value, body } = &expr.node else {
        panic!("expected Let, got {:?}", expr.node);
    };

    // Verify pattern is variable x
    assert!(matches!(&pattern.node, PatternKind::Var(name) if name == "x"));

    // Verify value is a BinOp
    assert!(matches!(&value.node, ExprKind::BinOp { op: BinOp::Add, .. }));

    // Verify body exists
    assert!(body.is_some());
}
```

### Bad Example (Don't do this)
```rust
#[test]
fn let_binding_works() {
    assert_eval_int("let x = 1 + 2 in x * 3", 9);  // No AST verification!
}
```

The bad example tells us nothing about what AST was produced. If parsing broke
but evaluation still happened to work, we wouldn't know.

---

## Layer 2: Rejection Tests (Critical)

**Purpose**: Verify that invalid programs are rejected with appropriate errors.

**When to use**: Every feature needs tests for invalid usage.

A compiler that accepts everything is useless. Rejection tests ensure our type
system actually catches errors.

### Good Example
```rust
#[test]
fn rejects_type_mismatch() {
    let result = typecheck_expr("1 + true");
    assert!(result.is_err(), "should reject adding Int and Bool");

    // Optionally verify the error kind
    let err = result.unwrap_err();
    assert!(err.contains("type mismatch") || err.contains("Type error"));
}

#[test]
fn rejects_unbound_variable() {
    let result = typecheck_expr("x + 1");
    assert!(result.is_err(), "should reject unbound variable");
}

#[test]
fn rejects_infinite_type() {
    let result = typecheck_expr("fun x -> x x");
    assert!(result.is_err(), "should fail occurs check");
}
```

### Soundness Canaries

These tests MUST always reject. If they pass, we have a soundness bug:

```rust
#[test]
fn soundness_polymorphic_channel() {
    // This must be a type error - can't send mixed types through channel
    let result = typecheck_program(r#"
        let main () =
            let ch = Channel.new in
            let _ = Fiber.spawn (fun () -> Channel.send ch 42) in
            Channel.send ch true
    "#);
    assert!(result.is_err(), "Mixed types through channel must be rejected");
}

#[test]
fn soundness_occurs_check() {
    // This must fail occurs check
    assert_type_error("fun x -> x x");
}
```

---

## Layer 3: Type Inference Tests

**Purpose**: Verify that expressions receive correct types.

**When to use**: For every expression form and type-related feature.

### Good Example
```rust
#[test]
fn lambda_type_inference() {
    let (expr, ty) = typecheck_expr("fun x -> x + 1").unwrap();

    // Verify the inferred type
    assert_eq!(format!("{}", ty), "Int -> Int");

    // Optionally verify AST structure too
    assert!(matches!(&expr.node, ExprKind::Lambda { .. }));
}

#[test]
fn polymorphic_identity() {
    let (_, ty) = typecheck_expr("fun x -> x").unwrap();

    // Type should be polymorphic: a -> a
    let ty_str = format!("{}", ty);
    assert!(ty_str.contains("->"), "expected arrow type, got {}", ty_str);
}

#[test]
fn let_polymorphism() {
    // id should be usable at multiple types
    let program = r#"
        let id x = x
        let a = id 42
        let b = id true
    "#;
    let result = typecheck_program(program);
    assert!(result.is_ok(), "let-polymorphism should allow reuse at different types");
}
```

---

## Layer 4: Runtime/Effect Tests

**Purpose**: Verify runtime behavior including effect sequences.

**When to use**: For programs with side effects, concurrency, or IO.

### Good Example
```rust
#[test]
fn fiber_spawn_produces_correct_effects() {
    let program = r#"
        let main () =
            let ch = Channel.new in
            let _ = Fiber.spawn (fun () -> Channel.send ch 42) in
            Channel.recv ch
    "#;

    // First verify it type-checks
    let result = typecheck_program(program);
    assert!(result.is_ok(), "should type-check: {:?}", result);

    // Then trace the effects
    let trace = trace_program(program);

    // Verify the effect sequence
    trace.assert_effects_contain(&[
        RuntimeEffectTrace::ChanNew,
        RuntimeEffectTrace::Fork,
        RuntimeEffectTrace::Send { channel: 0 },
        RuntimeEffectTrace::Recv { channel: 0 },
    ]);
}
```

### Bad Example (Output-only concurrency test)
```rust
#[test]
fn fibers_work() {
    run_program_ok(r#"
        let main () =
            let ch = Channel.new in
            let _ = Fiber.spawn (fun () -> Channel.send ch 42) in
            Channel.recv ch
    "#);
}
```

This tells us nothing about:
- What effects were produced
- In what order they occurred
- Whether the scheduler behaved correctly

---

## Layer 5: Output Tests (Secondary)

**Purpose**: Sanity check that programs produce expected values.

**When to use**: ONLY after Layer 1-4 tests exist for the relevant features.

### Acceptable Example (when combined with semantic tests)
```rust
#[test]
fn fibonacci_computes_correctly() {
    // This is OK because:
    // 1. Parser tests verify let-rec and if-then-else AST
    // 2. Type tests verify recursive function types
    // 3. This is just a sanity check
    let program = r#"
        let rec fib n =
            if n <= 1 then n
            else fib (n - 1) + fib (n - 2)
        in fib 10
    "#;
    assert_eval_int(program, 55);
}
```

---

## Required Test Coverage Matrix

When implementing a new feature, tests MUST be written for each applicable layer:

| Feature | Parser Test | Rejection Test | Type Test | Runtime Test |
|---------|-------------|----------------|-----------|--------------|
| Literals | Required | N/A | Required | Required |
| Let bindings | Required | Required | Required | Required |
| Lambdas | Required | Required | Required | Required |
| Application | Required | Required | Required | Required |
| If expressions | Required | Required | Required | Required |
| Match expressions | Required | Required | Required | Required |
| Type declarations | Required | Required | Required | N/A |
| Typeclasses | Required | Required | Required | Required |
| Channels | Required | Required | Required | Required |
| Effects | Required | Required | Required | Required |
| Records | Required | Required | Required | Required |

**Do NOT skip layers.** A feature without rejection tests is incomplete.

---

## Using test_support.rs

The `src/test_support.rs` module provides helpers for each testing layer.

### Pipeline Inspection

```rust
use gneiss::test_support::*;

// Layer 1: Parse only - inspect AST
let expr = parse_expr("1 + 2").unwrap();
let program = parse_program("let x = 1").unwrap();

// Layer 3: Parse + typecheck - inspect types
let (expr, ty) = typecheck_expr("fun x -> x").unwrap();
let (prog, env) = typecheck_program("let f x = x").unwrap();

// Full inference result (includes effects, constraints)
let (expr, result) = typecheck_expr_full("fun x -> x").unwrap();
```

### AST Matching

```rust
// Check expression kind
assert!(expr_matches("1 + 2", |e| matches!(e, ExprKind::BinOp { .. })));

// Destructure and verify
if let ExprKind::Lambda { params, body } = &expr.node {
    assert_eq!(params.len(), 1);
    // verify body...
}
```

### Type Assertions

```rust
// Exact type match
assert_type("42", "Int");
assert_type("true", "Bool");
assert_type("fun x -> x + 1", "Int -> Int");

// Type error expected
assert_type_error("1 + true");
```

### Effect Tracing

```rust
let trace = trace_program(program);

// Exact sequence
trace.assert_effects(&[
    RuntimeEffectTrace::ChanNew,
    RuntimeEffectTrace::Fork,
    RuntimeEffectTrace::Done,
]);

// Contains in order (allows other effects between)
trace.assert_effects_contain(&[
    RuntimeEffectTrace::Fork,
    RuntimeEffectTrace::Send { channel: 0 },
]);
```

---

## Anti-Patterns to Avoid

### 1. Output-Only Tests

**Bad:**
```rust
#[test]
fn feature_works() {
    run_program_ok(r#"
        let x = feature_under_test
        in x
    "#);
}
```

This tells us nothing about HOW it works.

### 2. ONLY Testing Features in Isolation

**Bad:**
```rust
// Only testing == on literals
#[test]
fn eq_works() {
    assert_eval_bool("1 == 1", true);
    assert_eval_bool("\"a\" == \"a\"", true);
}
// Never testing == through polymorphic functions!
```

**Why this is dangerous:** A real bug occurred where `==` worked on concrete types
but failed when used in a polymorphic function like `assert_eq x y = if x == y ...`.
The elaborator dispatched to `IntEq` instead of the Eq trait dictionary when the
type was a type variable.

**Good - Test both isolation AND composition:**
```rust
#[test]
fn eq_on_literals() {
    assert_eval_bool("1 == 1", true);
}

#[test]
fn eq_through_polymorphic_function() {
    // This catches bugs where trait dispatch fails on type variables
    let program = r#"
        let assert_eq x y = x == y
        let main () = assert_eq "hello" "hello"
    "#;
    // Verify it type-checks with proper constraint
    let (_, env) = typecheck_program(program).unwrap();
    // Then verify runtime behavior
    run_program_ok(program);
}
```

### 3. Untraceable Failures

**Bad:**
```rust
#[test]
fn everything_works() {
    run_program_ok(r#"
        let x = feature1
        let y = feature2
        let z = feature3
        in x + y + z
    "#);
}
```

When this fails, you don't know which feature broke.

**Better - Compositional tests WITH unit test coverage:**
When you have unit tests for feature1, feature2, feature3 individually,
a failing integration test tells you the *interaction* is broken, not
the individual features. This is valuable information.

### 4. No Negative Tests

**Bad:** Having `feature_works()` without `feature_rejects_invalid_usage()`.

### 5. Skipping Semantic Layers

**Bad:** Writing output tests without corresponding parser and type tests.

---

## Compositional Testing

**Unit tests are necessary but not sufficient.** Bugs often hide in how features
compose. The testing pyramid should be supplemented with integration tests that
verify features work together.

### Why Compositional Tests Matter

Real bugs encountered:
1. **Polymorphic dispatch failure**: `==` worked on concrete types but fell
   through to `IntEq` when used in polymorphic functions with type variables
2. **Type inference in nested contexts**: Types weren't being inferred correctly
   inside certain expression bodies (match arms, let bodies, lambda bodies)
3. **Elaboration losing constraints**: Trait constraints present during inference
   were dropped during elaboration to typed IR

These bugs pass isolated tests but fail in realistic programs.

### Integration Test Structure

```rust
#[test]
fn polymorphic_function_preserves_trait_dispatch() {
    // This is an INTEGRATION test - it tests that:
    // 1. Polymorphic functions type-check with constraints
    // 2. Constraints survive through elaboration
    // 3. Runtime dispatch goes through trait dictionaries
    let program = r#"
        trait Stringify a =
            stringify : a -> String
        end

        impl Stringify Int =
            let stringify x = int_to_string x
        end

        let show_twice x = stringify x ++ stringify x

        let main () = show_twice 42
    "#;

    // Verify type inference assigns correct constrained type
    let (_, env) = typecheck_program(program).unwrap();

    // Verify runtime uses trait dispatch, not hardcoded behavior
    run_program_ok(program);
}
```

### When to Write Compositional Tests

1. **After fixing a bug** - The fix should include a test that uses the feature
   in a realistic context, not just the minimal reproduction
2. **For polymorphic code** - Polymorphism + traits + type inference interact
   in subtle ways
3. **For nested constructs** - Match inside let inside lambda inside handler
4. **For the scheduler** - Multiple fibers + channels + effects together

---

## Test-Driven Debugging

When a bug is found:

1. Write a **minimal** test that reproduces the bug
2. Verify the test fails
3. Determine which layer the bug is in (parse? type? eval?)
4. Write the fix
5. Verify the test passes
6. Add tests for similar edge cases

The test should be at the appropriate layer. A type inference bug needs a type
test, not an output test.

---

## Property-Based Testing

Use proptest for generating test cases when properties should hold universally:

```rust
proptest! {
    #[test]
    fn type_inference_deterministic(source in arb_well_typed_source()) {
        let result1 = typecheck_expr(&source);
        let result2 = typecheck_expr(&source);
        prop_assert_eq!(result1.is_ok(), result2.is_ok());
    }

    #[test]
    fn unification_symmetric(t1 in arb_type(2), t2 in arb_type(2)) {
        let r1 = unify(&t1, &t2);
        let r2 = unify(&t2, &t1);
        prop_assert_eq!(r1.is_ok(), r2.is_ok());
    }
}
```

See `tests/properties.rs` for existing property-based tests.

---

## Test File Organization

```
tests/
├── parser_unit.rs       # Layer 1: AST structure tests
├── parser_negative.rs   # Layer 1: Parse rejection tests
├── type_inference.rs    # Layer 3: Type inference tests
├── type_rejection.rs    # Layer 2: Type rejection tests
├── properties.rs        # Property-based type soundness
├── fiber_effects.rs     # Layer 4: Effect sequence tests
├── algebraic_effects.rs # Layer 4: Effect handler tests
├── error_snapshots.rs   # Error message format tests
└── ...
```

---

## Summary

1. **Test each compilation stage** - Don't jump straight to output tests
2. **Rejection tests are mandatory** - A feature without negative tests is incomplete
3. **Use test_support.rs** - It has helpers for every layer
4. **When debugging, identify the layer** - Put the test in the right place
5. **Output tests are secondary** - They verify sanity, not correctness

Following this philosophy prevents the "hidden bug" problem where something
appears to work but is actually broken in subtle ways that compound over time.
