# Issue #004: Generalization Can Return Linked Type Variables Instead of Resolved Types

## Summary
In `generalize_inner`, when a type variable should not be generalized (level <= self.level), the function returns `ty.clone()` (the original parameter) instead of the resolved type. This can leak internal linked type variables into type schemes.

## Severity
**HIGH** - Can cause incorrect type behavior in certain edge cases.

## Location
- **File:** `src/infer.rs`
- **Function:** `generalize_inner` (approximately lines 180-210)

## Current Broken Code
```rust
fn generalize_inner(
    &self,
    ty: &Type,
    generics: &mut HashMap<TypeVarId, TypeVarId>,
) -> Type {
    match ty.resolve() {
        Type::Var(var) => match &*var.borrow() {
            TypeVar::Unbound { id, level } if *level > self.level => {
                // ... generalize to generic var ...
            }
            _ => ty.clone(),  // BUG: Returns original `ty`, not resolved type!
        },
        Type::Arrow(a, b) => Type::Arrow(
            Rc::new(self.generalize_inner(&a, generics)),
            Rc::new(self.generalize_inner(&b, generics)),
        ),
        // ... other cases
    }
}
```

## The Problem

Consider this scenario:
1. We have `ty = Var(v1)` where `v1` is linked to `Var(v2)` where `v2` is `Unbound { id: 5, level: 0 }`
2. `self.level` is `0`, so level <= self.level, meaning we should NOT generalize
3. `ty.resolve()` correctly returns `Var(v2)` (the unbound variable)
4. We match `Type::Var(var)` where var is `v2`
5. The guard `*level > self.level` is false (0 > 0 is false)
6. We fall to `_ => ty.clone()`
7. **BUG:** `ty.clone()` returns the ORIGINAL `ty`, which is `Var(v1)` (the LINKED variable)!

The result contains a linked type variable instead of the resolved unbound variable.

## Required Fix

Return the resolved type, not the original parameter:

```rust
fn generalize_inner(
    &self,
    ty: &Type,
    generics: &mut HashMap<TypeVarId, TypeVarId>,
) -> Type {
    let resolved = ty.resolve();  // Resolve once and reuse
    match &resolved {
        Type::Var(var) => match &*var.borrow() {
            TypeVar::Unbound { id, level } if *level > self.level => {
                let gen_id = if let Some(&gen_id) = generics.get(id) {
                    gen_id
                } else {
                    let gen_id = generics.len() as TypeVarId;
                    generics.insert(*id, gen_id);
                    gen_id
                };
                Type::new_generic(gen_id)
            }
            _ => resolved.clone(),  // FIX: Return resolved, not ty
        },
        Type::Arrow(a, b) => Type::Arrow(
            Rc::new(self.generalize_inner(a, generics)),
            Rc::new(self.generalize_inner(b, generics)),
        ),
        Type::Tuple(ts) => Type::Tuple(
            ts.iter()
                .map(|t| self.generalize_inner(t, generics))
                .collect(),
        ),
        Type::List(t) => Type::List(Rc::new(self.generalize_inner(t, generics))),
        Type::Channel(t) => Type::Channel(Rc::new(self.generalize_inner(t, generics))),
        Type::Constructor { name, args } => Type::Constructor {
            name: name.clone(),
            args: args
                .iter()
                .map(|t| self.generalize_inner(t, generics))
                .collect(),
        },
        // Primitives just return the resolved type
        _ => resolved.clone(),
    }
}
```

## Key Changes

1. Store the resolved type in a variable: `let resolved = ty.resolve();`
2. Match on `&resolved` instead of `ty.resolve()` directly
3. In all branches that return the type as-is, return `resolved.clone()` instead of `ty.clone()`

## Test Cases

These tests verify that linked variables are properly resolved:

```rust
#[test]
fn test_generalize_follows_links() {
    // This test ensures that generalization properly resolves linked type variables
    let source = r#"
let f = fun x ->
    let y = x in
    y
"#;
    // If generalization is correct, f should have type `forall a. a -> a`
    // not contain any linked variables
    let result = typecheck_program(source);
    assert!(result.is_ok());
    
    if let Ok(env) = result {
        if let Some(scheme) = env.get("f") {
            // The type string should be clean, not contain internal var references
            let ty_str = format!("{}", scheme);
            assert!(!ty_str.contains("t"), "Type should be generalized: {}", ty_str);
        }
    }
}

#[test]  
fn test_unification_then_generalize() {
    // Create a scenario where unification creates links, then generalize
    let source = r#"
let apply f x = f x
let id = fun x -> x
let result = apply id 42
"#;
    let result = typecheck_program(source);
    assert!(result.is_ok());
}
```

## Verification
```bash
cargo test test_generalize
cargo test test_unification
```
