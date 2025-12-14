# Issue #001: ADT Constructor Field Types Are Ignored (CRITICAL)

## Summary
When registering algebraic data type declarations, the type inference engine creates unrelated fresh type variables for constructor fields instead of using the actual types from the declaration. This breaks the connection between type parameters and their uses in constructor fields.

## Severity
**CRITICAL** - This breaks pattern matching on polymorphic ADTs.

## Location
- **File:** `src/infer.rs`
- **Function:** `register_type_decl` (approximately lines 690-710)

## Current Broken Code
```rust
pub fn register_type_decl(&mut self, decl: &Decl) {
    if let Decl::Type {
        name,
        params,
        constructors,
    } = decl
    {
        for ctor in constructors {
            // BUG: This ignores the actual TypeExpr from the declaration!
            let field_types: Vec<Type> = ctor
                .fields
                .iter()
                .map(|_| self.fresh_var())  // WRONG: creates unrelated fresh vars
                .collect();

            let info = ConstructorInfo {
                type_name: name.clone(),
                type_params: params.len() as u32,
                field_types,  // These have no connection to type params!
            };

            self.type_ctx.add_constructor(ctor.name.clone(), info);
        }
    }
}
```

## The Problem
For a declaration like:
```gneiss
type Option a = Some a | None
```

The parser correctly captures that `Some` has a field of type `a` (stored as `TypeExpr::Var("a")`). But `register_type_decl`:

1. **Ignores** the `TypeExpr` values in `ctor.fields`
2. Creates fresh unbound type variables with `self.fresh_var()`
3. These fresh variables have **no connection** to the type parameter `a`

### Expected Behavior
`Some : forall a. a -> Option a` — the `a` in the field type is the same `a` in the result type.

### Actual Behavior  
`Some : t0 -> Option t1` — where `t0` and `t1` are unrelated fresh variables.

## Impact
```gneiss
type Option a = Some a | None

-- This is broken:
let x = Some 42
match x with
| Some n -> n + 1  -- n does NOT have type Int!
| None -> 0
```

The pattern variable `n` gets an unrelated type variable instead of `Int`.

## Required Fix

### Step 1: Add a helper function to convert `TypeExpr` to `Type`

```rust
/// Convert a surface type expression to an internal type,
/// mapping type parameter names to generic type variable IDs.
fn type_expr_to_type(
    &mut self,
    expr: &TypeExpr,
    param_map: &HashMap<String, TypeVarId>,
) -> Result<Type, TypeError> {
    match &expr.node {
        TypeExprKind::Var(name) => {
            // Type parameter reference -> Generic type variable
            if let Some(&id) = param_map.get(name) {
                Ok(Type::new_generic(id))
            } else {
                // Unknown type variable - could be an error or create fresh var
                Err(TypeError::UnboundVariable(name.clone()))
            }
        }
        TypeExprKind::Named(name) => {
            // Named type (primitive or user-defined with no args)
            match name.as_str() {
                "Int" => Ok(Type::Int),
                "Float" => Ok(Type::Float),
                "Bool" => Ok(Type::Bool),
                "String" => Ok(Type::String),
                "Char" => Ok(Type::Char),
                "Pid" => Ok(Type::Pid),
                _ => Ok(Type::Constructor { name: name.clone(), args: vec![] })
            }
        }
        TypeExprKind::App { constructor, args } => {
            // Type application like `List a` or `Result a b`
            let arg_types: Result<Vec<_>, _> = args
                .iter()
                .map(|a| self.type_expr_to_type(a, param_map))
                .collect();
            let arg_types = arg_types?;
            
            // Get the constructor name
            match &constructor.node {
                TypeExprKind::Named(name) => {
                    Ok(Type::Constructor { name: name.clone(), args: arg_types })
                }
                _ => {
                    // Handle nested type expressions if needed
                    Err(TypeError::PatternMismatch) // Use appropriate error
                }
            }
        }
        TypeExprKind::Arrow { from, to } => {
            let from_ty = self.type_expr_to_type(from, param_map)?;
            let to_ty = self.type_expr_to_type(to, param_map)?;
            Ok(Type::arrow(from_ty, to_ty))
        }
        TypeExprKind::Tuple(types) => {
            let tys: Result<Vec<_>, _> = types
                .iter()
                .map(|t| self.type_expr_to_type(t, param_map))
                .collect();
            Ok(Type::Tuple(tys?))
        }
        TypeExprKind::Channel(inner) => {
            let inner_ty = self.type_expr_to_type(inner, param_map)?;
            Ok(Type::Channel(Rc::new(inner_ty)))
        }
    }
}
```

### Step 2: Fix `register_type_decl`

```rust
pub fn register_type_decl(&mut self, decl: &Decl) {
    if let Decl::Type {
        name,
        params,
        constructors,
    } = decl
    {
        // Map type parameter names to generic variable IDs
        // e.g., for `type Option a = ...`, maps "a" -> 0
        let param_map: HashMap<String, TypeVarId> = params
            .iter()
            .enumerate()
            .map(|(i, param_name)| (param_name.clone(), i as TypeVarId))
            .collect();

        for ctor in constructors {
            // Convert actual TypeExprs to Types with proper generic variables
            let field_types: Vec<Type> = ctor
                .fields
                .iter()
                .filter_map(|type_expr| {
                    self.type_expr_to_type(type_expr, &param_map).ok()
                })
                .collect();

            let info = ConstructorInfo {
                type_name: name.clone(),
                type_params: params.len() as u32,
                field_types,
            };

            self.type_ctx.add_constructor(ctor.name.clone(), info);
        }
    }
}
```

### Step 3: Add necessary import
Make sure `HashMap` is imported at the top of the file:
```rust
use std::collections::HashMap;
```

## Test Cases

Add these tests to verify the fix:

```rust
#[test]
fn test_option_type_pattern_matching() {
    let source = r#"
type Option a = Some a | None

let unwrap_or opt default =
    match opt with
    | Some x -> x
    | None -> default

let result = unwrap_or (Some 42) 0
"#;
    let result = typecheck_program(source);
    assert!(result.is_ok(), "Option pattern matching should typecheck: {:?}", result);
}

#[test]
fn test_option_map() {
    let source = r#"
type Option a = Some a | None

let map f opt =
    match opt with
    | Some x -> Some (f x)
    | None -> None

let result = map (fun n -> n + 1) (Some 10)
"#;
    let result = typecheck_program(source);
    assert!(result.is_ok(), "Option map should typecheck: {:?}", result);
}

#[test]
fn test_either_type() {
    let source = r#"
type Either a b = Left a | Right b

let map_right f either =
    match either with
    | Left x -> Left x
    | Right y -> Right (f y)

let result = map_right (fun n -> n * 2) (Right 21)
"#;
    let result = typecheck_program(source);
    assert!(result.is_ok(), "Either map_right should typecheck: {:?}", result);
}

#[test]
fn test_list_adt() {
    let source = r#"
type List a = Nil | Cons a (List a)

let head lst =
    match lst with
    | Cons x _ -> x
    | Nil -> 0

let result = head (Cons 42 Nil)
"#;
    let result = typecheck_program(source);
    assert!(result.is_ok(), "List head should typecheck: {:?}", result);
}
```

## Verification
After fixing, run:
```bash
cargo test
cargo run -- examples/adt.gn
```

The `examples/adt.gn` file should run without type errors.
