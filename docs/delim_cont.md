# Delimited Continuations Implementation Guide

## Overview

This document specifies the implementation of delimited continuations (`reset`/`shift`) for Gneiss. The agent should read this entire document before starting implementation.

## Conceptual Background

### What is a Continuation?

A continuation represents "the rest of the computation." In the expression:

```
1 + (2 * □) + 4
```

If we're currently computing something for the hole (□), the continuation is "take the result, multiply by 2, add 1, add 4."

In Gneiss's CPS interpreter, the continuation is already explicit—it's the `Cont` stack of `Frame`s.

### What are Delimited Continuations?

Normal continuations capture "the rest of the entire program." Delimited continuations capture only "the rest up to a boundary."

- `reset` installs a boundary (delimiter)
- `shift` captures everything up to that boundary as a callable value

### Core Semantics

```
reset (1 + shift (fun k -> k 10))
```

Evaluation:
1. `reset` pushes a Prompt marker onto the continuation stack
2. Evaluating `1 + shift(...)` pushes arithmetic frames
3. When `shift` executes, it pops frames until Prompt, packaging them as `k`
4. The body `k 10` runs—calling `k` splices the frames back and feeds `10` into them
5. Result: `1 + 10 = 11`

### Key Properties

**Continuation called once:**
```
reset (1 + shift (fun k -> k 10))
-- Result: 11
```

**Continuation called multiple times:**
```
reset (1 + shift (fun k -> k (k 10)))
-- k 10 = 11, then k 11 = 12
-- Result: 12
```

**Continuation discarded (early exit):**
```
reset (1 + shift (fun k -> 42))
-- k is never called, computation abandoned
-- Result: 42
```

**Nested resets (inner shift only captures to inner reset):**
```
reset (1 + reset (2 + shift (fun k -> k 10)))
-- Inner shift captures [2 + □], NOT [1 + [2 + □]]
-- Inner: 2 + 10 = 12
-- Outer: 1 + 12 = 13
-- Result: 13
```

---

## Implementation Specification

### Step 1: Lexer Changes

**File:** `src/lexer.rs`

Add two new tokens to the `Token` enum:

```rust
Reset,   // reset
Shift,   // shift
```

Add keyword recognition in `lex_ident`:

```rust
"reset" => Token::Reset,
"shift" => Token::Shift,
```

### Step 2: AST Changes

**File:** `src/ast.rs`

Add two new variants to `ExprKind`:

```rust
/// Delimited continuation boundary: reset expr
Reset(Rc<Expr>),

/// Capture continuation: shift (fun k -> body)
Shift {
    /// Parameter that binds the captured continuation
    param: Pattern,
    /// Body to execute with continuation bound
    body: Rc<Expr>,
},
```

### Step 3: Parser Changes

**File:** `src/parser.rs`

Add parsing for `reset` and `shift` in the expression parsing logic (likely in `parse_expr_atom` or similar, near where `spawn` is handled).

**Parse reset:**

```rust
Token::Reset => {
    let start = self.current_span();
    self.advance();
    let body = self.parse_expr_atom()?;
    let span = start.merge(&body.span);
    Ok(Spanned::new(ExprKind::Reset(Rc::new(body)), span))
}
```

**Parse shift:**

`shift` must be followed by a lambda. Parse it and extract the parameter:

```rust
Token::Shift => {
    let start = self.current_span();
    self.advance();
    let func = self.parse_expr_atom()?;
    
    match &func.node {
        ExprKind::Lambda { params, body } if params.len() == 1 => {
            let span = start.merge(&func.span);
            Ok(Spanned::new(ExprKind::Shift {
                param: params[0].clone(),
                body: body.clone(),
            }, span))
        }
        _ => Err(ParseError::UnexpectedToken {
            expected: "function (fun k -> ...)".into(),
            found: self.peek().clone(),
        })
    }
}
```

### Step 4: Type Inference Changes

**File:** `src/infer.rs`

Add cases in `infer_expr` for the new expression types.

**Type reset:**

```rust
ExprKind::Reset(body) => {
    // reset e : a  where e : a
    self.infer_expr(env, body)
}
```

**Type shift (simplified typing):**

```rust
ExprKind::Shift { param, body } => {
    // Simplified typing: k : a -> a, body : a, result : a
    // (Full answer-type polymorphism is complex, defer to future)
    let captured_ty = self.fresh_var();
    let cont_ty = Type::arrow(captured_ty.clone(), captured_ty.clone());
    
    let mut body_env = env.clone();
    self.bind_pattern(&mut body_env, param, &cont_ty)?;
    
    let body_ty = self.infer_expr(&body_env, body)?;
    self.unify(&body_ty, &captured_ty)?;
    
    Ok(captured_ty)
}
```

**Note:** This simplified typing requires that the continuation argument type, continuation return type, and shift body type all unify. This handles common cases but rejects some valid programs. Full answer-type polymorphism can be added later.

### Step 5: Eval Changes

**File:** `src/eval.rs`

This is the core implementation.

#### 5a. Add Prompt frame

In `enum Frame`, add:

```rust
/// Delimiter marker for reset
Prompt,
```

#### 5b. Add Continuation value

In `enum Value`, add:

```rust
/// A captured delimited continuation
Continuation {
    frames: Vec<Frame>,
},
```

#### 5c. Handle Reset in step_eval (or equivalent)

When evaluating a `reset` expression:

```rust
ExprKind::Reset(body) => {
    cont.push(Frame::Prompt);
    StepResult::Continue(State::Eval {
        expr: body.clone(),
        env,
        cont,
    })
}
```

#### 5d. Handle Shift in step_eval

When evaluating a `shift` expression:

```rust
ExprKind::Shift { param, body } => {
    // Capture frames up to Prompt
    let mut captured = Vec::new();
    
    loop {
        match cont.pop() {
            None => {
                return StepResult::Error(EvalError::RuntimeError(
                    "shift without enclosing reset".into()
                ));
            }
            Some(Frame::Prompt) => {
                break; // Found delimiter, stop capturing
            }
            Some(frame) => {
                captured.push(frame);
            }
        }
    }
    
    // Frames were popped in reverse order, fix it
    captured.reverse();
    
    // Create continuation value
    let continuation = Value::Continuation { frames: captured };
    
    // Bind parameter to continuation and evaluate body
    let new_env = EnvInner::with_parent(&env);
    if !self.try_bind_pattern(&new_env, &param, &continuation) {
        return StepResult::Error(EvalError::MatchFailed);
    }
    
    StepResult::Continue(State::Eval {
        expr: body.clone(),
        env: new_env,
        cont,
    })
}
```

#### 5e. Handle Prompt in step_apply

When a value reaches a Prompt frame during `step_apply`:

```rust
Frame::Prompt => {
    // reset body completed, value passes through
    StepResult::Continue(State::Apply { value, cont })
}
```

#### 5f. Handle Continuation application

When applying a function and the function is a `Continuation`, splice its frames:

Find where function application is handled (likely in `step_apply` for `Frame::AppArg` or similar). Add a case:

```rust
Value::Continuation { frames } => {
    // Splice captured frames back onto stack
    for frame in frames.into_iter().rev() {
        cont.push(frame);
    }
    // The argument becomes the "return value" to those frames
    StepResult::Continue(State::Apply {
        value: arg_value,
        cont,
    })
}
```

#### 5g. Update type_name for Continuation

In `Value::type_name()`:

```rust
Value::Continuation { .. } => "Continuation",
```

### Step 6: Runtime Changes

**File:** `src/runtime.rs`

No changes required. Delimited continuations are purely a control-flow mechanism within a single process. They don't interact with the scheduler or channels.

---

## Testing

Add these tests to `src/eval.rs` in the `#[cfg(test)]` module:

```rust
// ========================================================================
// Delimited continuation tests
// ========================================================================

#[test]
fn test_reset_no_shift() {
    let val = eval("reset 42").unwrap();
    assert!(matches!(val, Value::Int(42)));
}

#[test]
fn test_reset_with_expr() {
    let val = eval("reset (1 + 2 + 3)").unwrap();
    assert!(matches!(val, Value::Int(6)));
}

#[test]
fn test_shift_discard_continuation() {
    // k not called - early exit
    let val = eval("reset (1 + shift (fun k -> 42))").unwrap();
    assert!(matches!(val, Value::Int(42)));
}

#[test]
fn test_shift_call_once() {
    let val = eval("reset (1 + shift (fun k -> k 10))").unwrap();
    assert!(matches!(val, Value::Int(11)));
}

#[test]
fn test_shift_call_twice() {
    let val = eval("reset (1 + shift (fun k -> k (k 10)))").unwrap();
    // k 10 = 11, k 11 = 12
    assert!(matches!(val, Value::Int(12)));
}

#[test]
fn test_shift_in_let() {
    let val = eval("reset (let x = shift (fun k -> k 5) in x * x)").unwrap();
    assert!(matches!(val, Value::Int(25)));
}

#[test]
fn test_nested_reset_inner_shift() {
    // Inner shift only captures to inner reset
    let val = eval("reset (1 + reset (2 + shift (fun k -> k 10)))").unwrap();
    // Inner: 2 + 10 = 12, Outer: 1 + 12 = 13
    assert!(matches!(val, Value::Int(13)));
}

#[test]
fn test_nested_reset_outer_shift() {
    // Outer shift captures outer context
    let val = eval("reset (1 + shift (fun k -> k (reset (2 + 3))))").unwrap();
    // reset (2+3) = 5, k 5 = 1 + 5 = 6
    assert!(matches!(val, Value::Int(6)));
}

#[test]
fn test_shift_without_reset_errors() {
    let result = eval("shift (fun k -> k 1)");
    assert!(result.is_err());
}

#[test]
fn test_shift_return_continuation() {
    // Return the continuation itself
    let val = eval("reset (1 + shift (fun k -> k))").unwrap();
    assert!(matches!(val, Value::Continuation { .. }));
}

#[test]
fn test_continuation_called_later() {
    let val = eval("
        let k = reset (1 + shift (fun k -> k)) in
        k 10
    ").unwrap();
    assert!(matches!(val, Value::Int(11)));
}

#[test]
fn test_shift_with_match() {
    let val = eval("
        reset (
            let x = shift (fun k -> k 1 + k 2) in
            x * 10
        )
    ").unwrap();
    // k 1 = 10, k 2 = 20, sum = 30
    assert!(matches!(val, Value::Int(30)));
}

#[test]
fn test_reset_in_function() {
    let program = r#"
let with_reset f = reset (f ())

let main () =
    with_reset (fun () -> 1 + shift (fun k -> k 10))
"#;
    run_program(program).unwrap();
}
```

Also add type inference tests in `src/infer.rs`:

```rust
#[test]
fn test_reset_type() {
    let ty = infer("reset 42").unwrap();
    assert!(matches!(ty, Type::Int));
}

#[test]
fn test_shift_type() {
    let ty = infer("reset (1 + shift (fun k -> k 10))").unwrap();
    assert!(matches!(ty, Type::Int));
}

#[test]
fn test_shift_type_mismatch() {
    // Body returns String, but context expects Int
    let result = infer("reset (1 + shift (fun k -> \"hello\"))");
    // With simplified typing, this should be a type error
    assert!(result.is_err());
}
```

---

## Example Programs

After implementation, create these example files:

### `examples/continuations_basic.gn`

```
-- Basic delimited continuation examples

-- Example 1: Simple capture and resume
let example1 () =
    reset (1 + shift (fun k -> k 10))
    -- Result: 11

-- Example 2: Multiple resumptions
let example2 () =
    reset (1 + shift (fun k -> k (k 10)))
    -- k 10 = 11, k 11 = 12
    -- Result: 12

-- Example 3: Discarding the continuation (early exit)
let example3 () =
    reset (
        let x = shift (fun k -> 42) in
        -- This code never runs because k is never called
        x * 100
    )
    -- Result: 42

-- Example 4: Capturing in a let binding
let example4 () =
    reset (
        let x = shift (fun k -> k 5 + k 6) in
        x * x
    )
    -- k 5 = 25, k 6 = 36, sum = 61
    -- Result: 61

let main () =
    print (example1 ());
    print (example2 ());
    print (example3 ());
    print (example4 ())
```

### `examples/continuations_escape.gn`

```
-- Early exit / escape continuation pattern
-- Similar to exceptions but without special syntax

let with_escape body =
    reset (body (fun result -> shift (fun _ -> result)))

-- Using escape for early return
let find_negative xs =
    with_escape (fun escape ->
        let rec check lst =
            match lst with
            | [] -> "no negatives"
            | x :: rest ->
                if x < 0 
                then escape "found negative"
                else check rest
        in check xs
    )

let main () =
    print (find_negative [1, 2, 3, 4, 5]);   -- "no negatives"
    print (find_negative [1, 2, -3, 4, 5])   -- "found negative"
```

### `examples/continuations_collect.gn`

```
-- Collecting multiple values from a computation
-- The continuation is called multiple times, results accumulated

-- Call continuation for each element, collect results
let for_each xs body =
    match xs with
    | [] -> []
    | x :: rest -> body x :: for_each rest body

let example () =
    reset (
        let x = shift (fun k -> for_each [1, 2, 3] k) in
        x * x
    )
    -- k 1 = 1, k 2 = 4, k 3 = 9
    -- Result: [1, 4, 9]

let main () =
    print (example ())
```

### `examples/continuations_state.gn`

```
-- State monad via continuations
-- Shows how delimited continuations can implement effects

let get () = shift (fun k -> fun s -> k s s)
let put new_s = shift (fun k -> fun _ -> k () new_s)

let run_state init computation =
    (reset (
        let result = computation () in
        fun s -> (result, s)
    )) init

-- Example: counter
let counter () =
    let x = get () in
    put (x + 1);
    let y = get () in
    put (y + 1);
    get ()

let main () =
    let (final_value, final_state) = run_state 0 counter in
    print final_value;  -- 2
    print final_state   -- 2
```

---

## External References

These resources may help understand the concepts:

1. **"A Monadic Framework for Delimited Continuations"** (Dybvig, Peyton Jones, Sabry)
   - https://www.cs.indiana.edu/~dyb/pubs/monadicDC.pdf
   - Formal semantics of shift/reset

2. **"Abstracting Control"** (Danvy, Filinski)
   - Classic paper introducing shift/reset
   - https://citeseerx.ist.psu.edu/document?doi=10.1.1.43.8753

3. **Oleg Kiselyov's tutorial**
   - http://okmij.org/ftp/continuations/
   - Many practical examples

4. **Racket documentation on continuations**
   - https://docs.racket-lang.org/reference/cont.html
   - Good practical explanations

5. **"Continuations by example"** (Matt Might)
   - https://matt.might.net/articles/programming-with-continuations--exceptions-backtracking-search-threads-generators-coroutines/
   - Accessible introduction with examples

---

## Verification Checklist

After implementation, verify:

- [ ] `cargo build` succeeds with no errors
- [ ] `cargo test` passes all existing tests
- [ ] All new continuation tests pass
- [ ] `reset 42` evaluates to `42`
- [ ] `reset (1 + shift (fun k -> k 10))` evaluates to `11`
- [ ] `reset (1 + shift (fun k -> k (k 10)))` evaluates to `12`
- [ ] `reset (1 + shift (fun k -> 42))` evaluates to `42`
- [ ] `shift (fun k -> k 1)` without reset produces an error
- [ ] Nested resets work correctly
- [ ] Example programs in `examples/` run without errors

## Implementation Order

Recommended order to minimize debugging:

1. Lexer (add tokens) - test with `cargo test lexer`
2. AST (add variants) - just compile
3. Parser (add parsing) - write a simple parse test
4. Eval `reset` only (push Prompt, handle in apply) - test `reset 42`
5. Eval `shift` capture - test `reset (shift (fun k -> 42))`
6. Eval continuation application - test `reset (shift (fun k -> k 10))`
7. Type inference - test type errors
8. Full test suite
9. Example programs
