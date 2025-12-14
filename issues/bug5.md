# Issue #005: Let Bindings with Complex Patterns Lose Polymorphism

## Summary
When a let binding uses a complex pattern (tuple, list, constructor) instead of a simple variable, the bound names become monomorphic even when they should be polymorphic. This is because `bind_pattern_scheme` instantiates the polymorphic scheme before binding pattern variables.

## Severity
**MEDIUM** - Affects usability of tuple/pattern destructuring with polymorphic values.

## Location
- **File:** `src/infer.rs`
- **Function:** `bind_pattern_scheme` (approximately lines 650-670)

## Current Broken Code
```rust
fn bind_pattern_scheme(
    &mut self,
    env: &mut TypeEnv,
    pattern: &Pattern,
    scheme: Scheme,
) -> Result<(), TypeError> {
    match &pattern.node {
        PatternKind::Var(name) => {
            env.insert(name.clone(), scheme);
            Ok(())
        }
        // For complex patterns in let bindings, we just use monomorphic types
        _ => {
            let ty = self.instantiate(&scheme);  // Instantiates, losing polymorphism!
            self.bind_pattern(env, pattern, &ty)
        }
    }
}
```

## The Problem

Consider:
```gneiss
let (f, g) = (fun x -> x, fun y -> y) in (f 1, f true, g "hello")
```

What should happen:
1. The tuple `(fun x -> x, fun y -> y)` has type `(a -> a, b -> b)`
2. After generalization, the scheme is `forall a b. (a -> a, b -> b)`
3. `f` should get scheme `forall a. a -> a`
4. `g` should get scheme `forall b. b -> b`
5. All uses should type-check

What actually happens:
1. The tuple gets scheme `forall a b. (a -> a, b -> b)` ✓
2. `bind_pattern_scheme` sees a tuple pattern, not a variable
3. It calls `self.instantiate(&scheme)`, creating `(t1 -> t1, t2 -> t2)` with fresh vars
4. `f` gets **monomorphic** type `t1 -> t1`
5. `g` gets **monomorphic** type `t2 -> t2`
6. `f 1` fixes `t1 = Int`
7. `f true` **FAILS** because `t1` is already `Int`!

## Required Fix

This is a complex issue to fix properly because you need to "distribute" the polymorphism through the pattern. Here are two approaches:

### Approach 1: Simple Fix - Document the Limitation

Just document that tuple/pattern destructuring doesn't preserve polymorphism:
```rust
// For complex patterns in let bindings, polymorphism is lost.
// Use separate let bindings to preserve polymorphism:
//   let f = fun x -> x in let g = fun y -> y in ...
// Instead of:
//   let (f, g) = (fun x -> x, fun y -> y) in ...
```

### Approach 2: Proper Fix - Generalize Pattern Components

For each component of a pattern that's a variable, we need to extract the corresponding part of the type and generalize it independently.

```rust
fn bind_pattern_scheme(
    &mut self,
    env: &mut TypeEnv,
    pattern: &Pattern,
    scheme: Scheme,
) -> Result<(), TypeError> {
    match &pattern.node {
        PatternKind::Var(name) => {
            env.insert(name.clone(), scheme);
            Ok(())
        }
        PatternKind::Tuple(patterns) => {
            // Instantiate to get the tuple structure, but then re-generalize each component
            let ty = self.instantiate(&scheme);
            if let Type::Tuple(component_types) = ty.resolve() {
                if patterns.len() != component_types.len() {
                    return Err(TypeError::PatternMismatch);
                }
                for (pat, comp_ty) in patterns.iter().zip(component_types.iter()) {
                    // Re-generalize each component
                    let comp_scheme = self.generalize(comp_ty);
                    self.bind_pattern_scheme(env, pat, comp_scheme)?;
                }
                Ok(())
            } else {
                Err(TypeError::PatternMismatch)
            }
        }
        PatternKind::List(patterns) => {
            let ty = self.instantiate(&scheme);
            if let Type::List(elem_ty) = ty.resolve() {
                let elem_scheme = self.generalize(&elem_ty);
                for pat in patterns {
                    self.bind_pattern_scheme(env, pat, elem_scheme.clone())?;
                }
                Ok(())
            } else {
                Err(TypeError::PatternMismatch)
            }
        }
        PatternKind::Constructor { name, args } => {
            // For constructors, we need to extract the field types
            // This is more complex and may require looking up constructor info
            let ty = self.instantiate(&scheme);
            self.bind_pattern(env, pattern, &ty)  // Fall back to monomorphic for now
        }
        // For other patterns (Wildcard, Lit, Cons), fall back to monomorphic
        _ => {
            let ty = self.instantiate(&scheme);
            self.bind_pattern(env, pattern, &ty)
        }
    }
}
```

**Note:** The re-generalization approach has a subtlety: after instantiating a `forall a b. (a -> a, b -> b)` to `(t1 -> t1, t2 -> t2)`, re-generalizing `t1 -> t1` gives `forall c. c -> c`, which is correct. But this only works because we're at the same level. If levels have changed, this might not generalize correctly.

### Approach 3: Best Fix - Don't Instantiate, Decompose the Scheme

The cleanest approach is to decompose the scheme directly without instantiating:

```rust
fn bind_pattern_scheme(
    &mut self,
    env: &mut TypeEnv,
    pattern: &Pattern,
    scheme: Scheme,
) -> Result<(), TypeError> {
    match &pattern.node {
        PatternKind::Var(name) => {
            env.insert(name.clone(), scheme);
            Ok(())
        }
        PatternKind::Tuple(patterns) => {
            // Check if the scheme's type is a tuple
            match &scheme.ty.resolve() {
                Type::Tuple(component_types) if patterns.len() == component_types.len() => {
                    for (pat, comp_ty) in patterns.iter().zip(component_types.iter()) {
                        // Create a scheme for each component with the same generics
                        let comp_scheme = Scheme {
                            num_generics: scheme.num_generics,
                            ty: comp_ty.clone(),
                        };
                        self.bind_pattern_scheme(env, pat, comp_scheme)?;
                    }
                    Ok(())
                }
                _ => {
                    // Fall back to monomorphic binding
                    let ty = self.instantiate(&scheme);
                    self.bind_pattern(env, pattern, &ty)
                }
            }
        }
        PatternKind::Wildcard => Ok(()),
        _ => {
            let ty = self.instantiate(&scheme);
            self.bind_pattern(env, pattern, &ty)
        }
    }
}
```

## Test Cases

```rust
#[test]
fn test_tuple_pattern_polymorphism() {
    // This test may fail with current implementation
    let source = r#"
let (f, g) = (fun x -> x, fun y -> y) in
(f 1, f true, g "hello")
"#;
    let result = typecheck_program(source);
    // With the fix, this should pass
    // Without the fix, this fails because f becomes monomorphic
    assert!(result.is_ok(), "Tuple destructuring should preserve polymorphism: {:?}", result);
}

#[test]
fn test_separate_lets_polymorphic() {
    // This should always work (workaround)
    let source = r#"
let f = fun x -> x in
let g = fun y -> y in
(f 1, f true, g "hello")
"#;
    let result = typecheck_program(source);
    assert!(result.is_ok());
}

#[test]
fn test_tuple_pattern_monomorphic_ok() {
    // Using each bound variable at only one type should work
    let source = r#"
let (f, g) = (fun x -> x, fun y -> y) in
(f 1, g true)
"#;
    let result = typecheck_program(source);
    assert!(result.is_ok());
}
```

## Recommendation

For a v0.1 release, **Approach 1** (document the limitation) is acceptable. Most functional languages have similar restrictions or complications with polymorphism and pattern matching in let bindings.

For a more complete implementation, **Approach 3** is the cleanest but requires careful handling of the generic variable numbering.

## Verification
```bash
cargo test test_tuple_pattern
cargo test test_separate_lets
```
