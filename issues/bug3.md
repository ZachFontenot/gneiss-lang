# Issue #003: Function Composition Operators Not Implemented at Runtime

## Summary
The function composition operators `>>` and `<<` type-check correctly but throw a runtime error when evaluated. The type checker accepts them, but the evaluator has a stub that returns an error.

## Severity
**MEDIUM** - Feature is advertised but broken.

## Location
- **File:** `src/eval.rs`
- **Function:** `eval_binop` (approximately lines 1109-1113)

## Current Broken Code
```rust
// Compose - create a new closure
(BinOp::Compose, _, _) | (BinOp::ComposeBack, _, _) => {
    Err(EvalError::RuntimeError(
        "function composition not yet implemented".into(),
    ))
}
```

## The Problem
Users can write:
```gneiss
let double = fun x -> x * 2
let add1 = fun x -> x + 1
let pipeline = double >> add1  -- Type checks!
pipeline 5                      -- RUNTIME ERROR!
```

The type checker correctly infers:
- `double : Int -> Int`
- `add1 : Int -> Int`  
- `double >> add1 : Int -> Int`

But at runtime, it crashes with "function composition not yet implemented".

## Required Fix

Replace the error stub with actual composition logic. Function composition should create a new closure that applies the functions in sequence.

### For `f >> g` (compose forward):
`(f >> g) x = g (f x)` — apply f first, then g

### For `f << g` (compose backward):  
`(f << g) x = f (g x)` — apply g first, then f

```rust
(BinOp::Compose, f, g) => {
    // f >> g = fun x -> g (f x)
    // Create a closure that captures f and g
    Ok(Value::Closure {
        params: vec![Spanned::new(
            PatternKind::Var("__compose_arg".into()),
            Span::default(),
        )],
        body: Rc::new(Spanned::new(
            ExprKind::App {
                func: Rc::new(Spanned::new(
                    ExprKind::Var("__g".into()),
                    Span::default(),
                )),
                arg: Rc::new(Spanned::new(
                    ExprKind::App {
                        func: Rc::new(Spanned::new(
                            ExprKind::Var("__f".into()),
                            Span::default(),
                        )),
                        arg: Rc::new(Spanned::new(
                            ExprKind::Var("__compose_arg".into()),
                            Span::default(),
                        )),
                    },
                    Span::default(),
                )),
            },
            Span::default(),
        )),
        env: {
            let compose_env = EnvInner::new();
            compose_env.borrow_mut().define("__f".into(), f.clone());
            compose_env.borrow_mut().define("__g".into(), g.clone());
            compose_env
        },
    })
}
(BinOp::ComposeBack, f, g) => {
    // f << g = fun x -> f (g x)
    // Same as g >> f
    Ok(Value::Closure {
        params: vec![Spanned::new(
            PatternKind::Var("__compose_arg".into()),
            Span::default(),
        )],
        body: Rc::new(Spanned::new(
            ExprKind::App {
                func: Rc::new(Spanned::new(
                    ExprKind::Var("__f".into()),
                    Span::default(),
                )),
                arg: Rc::new(Spanned::new(
                    ExprKind::App {
                        func: Rc::new(Spanned::new(
                            ExprKind::Var("__g".into()),
                            Span::default(),
                        )),
                        arg: Rc::new(Spanned::new(
                            ExprKind::Var("__compose_arg".into()),
                            Span::default(),
                        )),
                    },
                    Span::default(),
                )),
            },
            Span::default(),
        )),
        env: {
            let compose_env = EnvInner::new();
            compose_env.borrow_mut().define("__f".into(), f.clone());
            compose_env.borrow_mut().define("__g".into(), g.clone());
            compose_env
        },
    })
}
```

## Alternative: Simpler Implementation Using a Dedicated Value Type

If creating synthetic AST nodes feels hacky, you can add a new `Value` variant:

### Step 1: Add to `Value` enum in `eval.rs`:
```rust
/// Composed functions
ComposedFn {
    first: Box<Value>,
    second: Box<Value>,
},
```

### Step 2: Handle in `eval_binop`:
```rust
(BinOp::Compose, f, g) => {
    // f >> g: apply f first, then g
    Ok(Value::ComposedFn {
        first: Box::new(f.clone()),
        second: Box::new(g.clone()),
    })
}
(BinOp::ComposeBack, f, g) => {
    // f << g: apply g first, then f
    Ok(Value::ComposedFn {
        first: Box::new(g.clone()),
        second: Box::new(f.clone()),
    })
}
```

### Step 3: Handle application in `do_apply`:
```rust
Value::ComposedFn { first, second } => {
    // Apply first function to arg, then apply second to result
    // This needs to be done in CPS style with the continuation
    
    // Push frame to apply second after first completes
    cont.push(Frame::AppArg { func: *second });
    // Push frame to apply first
    cont.push(Frame::AppArg { func: *first });
    // Return the argument to start the chain
    StepResult::Continue(State::Apply { value: arg, cont })
}
```

### Step 4: Update `type_name`:
```rust
Value::ComposedFn { .. } => "Function",
```

### Step 5: Update `print_value`:
```rust
Value::ComposedFn { .. } => print!("<composed-function>"),
```

## Test Cases

Add these tests to `src/eval.rs`:

```rust
#[test]
fn test_compose_forward() {
    let val = eval("let f = fun x -> x + 1 in let g = fun x -> x * 2 in (f >> g) 5").unwrap();
    // f 5 = 6, g 6 = 12
    assert!(matches!(val, Value::Int(12)));
}

#[test]
fn test_compose_backward() {
    let val = eval("let f = fun x -> x + 1 in let g = fun x -> x * 2 in (f << g) 5").unwrap();
    // g 5 = 10, f 10 = 11
    assert!(matches!(val, Value::Int(11)));
}

#[test]
fn test_compose_chain() {
    let val = eval("let f = fun x -> x + 1 in let g = fun x -> x * 2 in let h = fun x -> x - 3 in (f >> g >> h) 5").unwrap();
    // f 5 = 6, g 6 = 12, h 12 = 9
    assert!(matches!(val, Value::Int(9)));
}

#[test]
fn test_compose_with_different_types() {
    let val = eval("let f = fun x -> x > 0 in let g = fun b -> if b then 1 else 0 in (f >> g) 5").unwrap();
    // f 5 = true, g true = 1
    assert!(matches!(val, Value::Int(1)));
}
```

## Verification
```bash
cargo test test_compose
```
