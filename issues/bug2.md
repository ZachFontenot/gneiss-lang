# Issue #002: Float Arithmetic and Comparisons Fail Type Checking

## Summary
The type checker only accepts `Int` for arithmetic and comparison operators, but the evaluator supports `Float`. This means `3.14 + 2.0` and `3.14 < 2.0` fail to type check even though they would evaluate correctly.

## Severity
**MEDIUM** - Float operations are completely broken at the type level.

## Location
- **File:** `src/infer.rs`  
- **Function:** `infer_expr`, in the `ExprKind::BinOp` match arm (approximately lines 465-490)

## Current Broken Code

### Arithmetic (lines ~465-470):
```rust
BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
    self.unify(&left_ty, &Type::Int)?;
    self.unify(&right_ty, &Type::Int)?;
    Ok(Type::Int)
}
```

### Comparisons (lines ~479-483):
```rust
BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
    self.unify(&left_ty, &Type::Int)?;
    self.unify(&right_ty, &Type::Int)?;
    Ok(Type::Bool)
}
```

## The Problem

The evaluator (`src/eval.rs`) handles float operations:
```rust
// Float arithmetic - WORKS at runtime
(BinOp::Add, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
(BinOp::Sub, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
// etc.
```

But the type checker rejects them because it forces both operands to be `Int`.

### What Fails
```gneiss
3.14 + 2.0      -- Type error: expected Int, found Float
3.14 < 2.0      -- Type error: expected Int, found Float
1.5 * 2.5       -- Type error: expected Int, found Float
```

## Required Fix

Replace the arithmetic and comparison type checking with logic that accepts either Int or Float (but requires both operands to match):

```rust
BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
    // Both operands must have the same type
    self.unify(&left_ty, &right_ty)?;
    
    // Must be either Int or Float
    let resolved = left_ty.resolve();
    match resolved {
        Type::Int => Ok(Type::Int),
        Type::Float => Ok(Type::Float),
        Type::Var(_) => {
            // Type variable - default to Int for now, or leave polymorphic
            // For simplicity, we can try to unify with Int
            // A proper solution would use type classes
            self.unify(&left_ty, &Type::Int)?;
            Ok(Type::Int)
        }
        _ => Err(TypeError::TypeMismatch {
            expected: Type::Int, // or "numeric type"
            found: resolved,
        }),
    }
}
BinOp::Mod => {
    // Modulo only works on Int
    self.unify(&left_ty, &Type::Int)?;
    self.unify(&right_ty, &Type::Int)?;
    Ok(Type::Int)
}
```

```rust
BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
    // Both operands must have the same type
    self.unify(&left_ty, &right_ty)?;
    
    // Must be either Int or Float
    let resolved = left_ty.resolve();
    match resolved {
        Type::Int | Type::Float => Ok(Type::Bool),
        Type::Var(_) => {
            // Default to Int for unresolved type variables
            self.unify(&left_ty, &Type::Int)?;
            Ok(Type::Bool)
        }
        _ => Err(TypeError::TypeMismatch {
            expected: Type::Int,
            found: resolved,
        }),
    }
}
```

## Alternative: Simpler Fix (Less Type Safe)

If you want a quicker fix that just makes it work without full validation:

```rust
BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
    // Just ensure both sides match, allow Int or Float
    self.unify(&left_ty, &right_ty)?;
    Ok(left_ty.resolve())
}

BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
    // Just ensure both sides match
    self.unify(&left_ty, &right_ty)?;
    Ok(Type::Bool)
}
```

**Warning:** This simpler fix allows `"a" + "b"` to type check (though it will fail at runtime for `+`). The evaluator does handle string concatenation for `++` but not `+`.

## Test Cases

Add these tests to `src/infer.rs`:

```rust
#[test]
fn test_float_arithmetic() {
    let ty = infer("3.14 + 2.0").unwrap();
    assert!(matches!(ty.resolve(), Type::Float));
}

#[test]
fn test_float_multiplication() {
    let ty = infer("1.5 * 2.5").unwrap();
    assert!(matches!(ty.resolve(), Type::Float));
}

#[test]
fn test_float_division() {
    let ty = infer("10.0 / 3.0").unwrap();
    assert!(matches!(ty.resolve(), Type::Float));
}

#[test]
fn test_float_comparison() {
    let ty = infer("3.14 < 2.71").unwrap();
    assert!(matches!(ty, Type::Bool));
}

#[test]
fn test_float_comparison_lte() {
    let ty = infer("1.0 <= 2.0").unwrap();
    assert!(matches!(ty, Type::Bool));
}

#[test]
fn test_mixed_int_float_rejected() {
    // Mixing Int and Float should fail
    let result = infer("3.14 + 2");
    assert!(result.is_err(), "Mixing Float and Int should be rejected");
}

#[test]
fn test_int_arithmetic_still_works() {
    let ty = infer("1 + 2").unwrap();
    assert!(matches!(ty.resolve(), Type::Int));
}

#[test]
fn test_int_comparison_still_works() {
    let ty = infer("1 < 2").unwrap();
    assert!(matches!(ty, Type::Bool));
}
```

## Verification
```bash
cargo test test_float
cargo test test_int_arithmetic
```
