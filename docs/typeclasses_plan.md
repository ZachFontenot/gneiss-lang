# Typeclass Implementation Plan for Gneiss

## Executive Summary

This document outlines a plan to add typeclasses to Gneiss using **dictionary passing** - the classic, well-understood approach from Haskell. We explicitly avoid Higher-Kinded Types (HKT) while still providing powerful abstraction through **constrained instances**.

### The n×m Problem and Our Solution

The "n×m instance problem" you mentioned is solved by **constrained instances**:

```gneiss
-- Instead of writing separate instances:
impl Show for (List Int) = ...
impl Show for (List Bool) = ...
impl Show for (List String) = ...
impl Show for (Option Int) = ...
-- ... ad infinitum

-- Write ONE constrained instance:
impl Show for (List a) where a : Show =
  let show xs = "[" ++ join ", " (map show xs) ++ "]"
end

impl Show for (Option a) where a : Show =
  let show opt = match opt with
    | Some x -> "Some(" ++ show x ++ ")"
    | None -> "None"
end
```

This gives you Show for `List Int`, `List (Option Bool)`, `Option (List String)`, etc. - all from just two instance declarations.

---

## Phase 1: Core Syntax and AST

### 1.1 New AST Nodes

Add to `src/ast.rs`:

```rust
/// Trait declaration
#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: Ident,
    /// Type parameter (single param for now, no HKT)
    pub type_param: Ident,
    /// Supertraits: trait Ord a : Eq
    pub supertraits: Vec<Ident>,
    /// Method signatures
    pub methods: Vec<TraitMethod>,
}

#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: Ident,
    pub type_sig: TypeExpr,
    /// Optional default implementation
    pub default_impl: Option<Expr>,
}

/// Instance declaration
#[derive(Debug, Clone)]
pub struct InstanceDecl {
    /// The trait being implemented
    pub trait_name: Ident,
    /// The type it's implemented for (e.g., "List a", "Int")
    pub target_type: TypeExpr,
    /// Constraints: where a : Show, a : Eq
    pub constraints: Vec<Constraint>,
    /// Method implementations
    pub methods: Vec<(Ident, Expr)>,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub trait_name: Ident,
    pub type_var: Ident,
}

// Add to Decl enum:
pub enum Decl {
    Let { ... },
    Type { ... },
    TypeAlias { ... },
    Trait(TraitDecl),      // NEW
    Instance(InstanceDecl), // NEW
}
```

### 1.2 New Tokens

Add to `src/lexer.rs`:

```rust
pub enum Token {
    // ... existing tokens ...
    Trait,    // trait
    Impl,     // impl
    For,      // for
    Where,    // where
    Val,      // val (for method signatures)
}
```

### 1.3 Parser Extensions

Add to `src/parser.rs`:

```rust
// Parse: trait Show a = val show : a -> String end
fn parse_trait_decl(&mut self) -> Result<TraitDecl, ParseError> { ... }

// Parse: impl Show for Int = let show n = ... end
fn parse_instance_decl(&mut self) -> Result<InstanceDecl, ParseError> { ... }

// Parse: where a : Show, b : Eq
fn parse_constraints(&mut self) -> Result<Vec<Constraint>, ParseError> { ... }
```

---

## Phase 2: Type System Extensions

### 2.1 Qualified Types

The key addition is **qualified types** - types with constraints attached.

Add to `src/types.rs`:

```rust
/// A qualified type: constraints => type
/// e.g., (Show a, Eq a) => a -> String
#[derive(Debug, Clone)]
pub struct QualType {
    pub constraints: Vec<TypeConstraint>,
    pub ty: Type,
}

/// A constraint on a type variable
/// e.g., Show a
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub trait_name: String,
    pub type_arg: Type,
}

/// Updated Scheme to include constraints
#[derive(Debug, Clone)]
pub struct Scheme {
    pub num_generics: u32,
    pub constraints: Vec<TypeConstraint>,  // NEW
    pub ty: Type,
}
```

### 2.2 Trait Environment

Track declared traits and instances:

```rust
/// Information about a trait
#[derive(Debug, Clone)]
pub struct TraitInfo {
    pub name: String,
    pub type_param: String,
    pub supertraits: Vec<String>,
    pub methods: HashMap<String, Type>,  // method name -> type
}

/// Information about an instance
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub trait_name: String,
    /// The head: Int, List a, Option a, etc.
    pub head: Type,
    /// Constraints required: [Show a, Eq a]
    pub constraints: Vec<TypeConstraint>,
    /// Method implementations (for dictionary building)
    pub methods: HashMap<String, Expr>,
}

/// Global trait/instance registry
#[derive(Debug, Clone, Default)]
pub struct TraitEnv {
    pub traits: HashMap<String, TraitInfo>,
    pub instances: Vec<InstanceInfo>,  // Vec because we search through them
}
```

### 2.3 Type Inference Changes

Modify `src/infer.rs`:

```rust
pub struct Inferencer {
    // ... existing fields ...
    trait_env: TraitEnv,
    /// Collected constraints during inference
    deferred_constraints: Vec<TypeConstraint>,
}
```

Key changes to inference:

1. **When inferring a trait method call**: Add a constraint to `deferred_constraints`
2. **At generalization time**: Include unsolved constraints in the Scheme
3. **At instantiation time**: Check constraints can be satisfied

```rust
// When we see `show x` where show : Show a => a -> String
fn infer_method_call(&mut self, method: &str, arg: &Expr, env: &TypeEnv) -> Result<Type, TypeError> {
    // Look up method in trait environment
    let (trait_name, method_type) = self.trait_env.lookup_method(method)?;
    
    // Instantiate the method type with fresh variables
    let (constraints, instantiated_type) = self.instantiate_qualified(method_type);
    
    // Add constraints to be solved later
    self.deferred_constraints.extend(constraints);
    
    // Continue with normal inference
    // ...
}
```

---

## Phase 3: Instance Resolution

This is the heart of the typeclass system - determining which instance to use.

### 3.1 The Resolution Algorithm

```rust
impl TraitEnv {
    /// Find an instance for a constraint like `Show (List Int)`
    /// Returns: the instance + any sub-constraints needed
    pub fn resolve(&self, constraint: &TypeConstraint) -> Result<Resolution, TypeError> {
        for instance in &self.instances {
            if instance.trait_name != constraint.trait_name {
                continue;
            }
            
            // Try to match instance head against constraint type
            // e.g., match `List a` against `List Int`
            if let Some(substitution) = self.match_types(&instance.head, &constraint.type_arg) {
                // Apply substitution to instance constraints
                // e.g., `Show a` becomes `Show Int`
                let sub_constraints: Vec<_> = instance.constraints
                    .iter()
                    .map(|c| c.apply_substitution(&substitution))
                    .collect();
                
                return Ok(Resolution {
                    instance: instance.clone(),
                    substitution,
                    sub_constraints,
                });
            }
        }
        
        Err(TypeError::NoInstance {
            trait_name: constraint.trait_name.clone(),
            ty: constraint.type_arg.clone(),
        })
    }
    
    /// Recursively resolve all constraints
    pub fn resolve_all(&self, constraints: &[TypeConstraint]) -> Result<Vec<Resolution>, TypeError> {
        let mut resolved = Vec::new();
        let mut queue: VecDeque<_> = constraints.iter().cloned().collect();
        
        while let Some(constraint) = queue.pop_front() {
            let resolution = self.resolve(&constraint)?;
            
            // Add sub-constraints to queue
            for sub in &resolution.sub_constraints {
                queue.push_back(sub.clone());
            }
            
            resolved.push(resolution);
        }
        
        Ok(resolved)
    }
}
```

### 3.2 Handling Overlapping Instances

For simplicity, we'll initially disallow overlapping instances:

```gneiss
impl Show for (List a) where a : Show = ...
impl Show for (List Int) = ...  -- ERROR: overlaps with above
```

This is the "Haskell 98" approach. We can add more sophisticated overlap handling later if needed.

---

## Phase 4: Dictionary Passing (Code Generation)

### 4.1 What is a Dictionary?

A dictionary is a record containing the method implementations for a specific type:

```rust
// Conceptually, for `Show Int`:
struct ShowIntDict {
    show: fn(Int) -> String,
}

// For `Show (List a)` with constraint `Show a`:
// The dictionary TAKES a sub-dictionary as input
fn make_show_list_dict(elem_dict: ShowDict<A>) -> ShowDict<List<A>> {
    ShowDict {
        show: |xs| {
            let parts: Vec<_> = xs.iter()
                .map(|x| (elem_dict.show)(x))
                .collect();
            format!("[{}]", parts.join(", "))
        }
    }
}
```

### 4.2 Runtime Representation

Add to `src/eval.rs`:

```rust
pub enum Value {
    // ... existing variants ...
    
    /// A typeclass dictionary
    Dictionary {
        trait_name: String,
        /// Method name -> implementation (closure)
        methods: HashMap<String, Value>,
    },
}
```

### 4.3 Dictionary Construction

During evaluation, when we need a dictionary:

```rust
impl Interpreter {
    /// Build a dictionary for a resolved constraint
    fn build_dictionary(&mut self, resolution: &Resolution, env: &Env) -> Result<Value, EvalError> {
        // First, build dictionaries for sub-constraints
        let sub_dicts: Vec<Value> = resolution.sub_constraints
            .iter()
            .map(|c| {
                let sub_res = self.trait_env.resolve(c)?;
                self.build_dictionary(&sub_res, env)
            })
            .collect::<Result<_, _>>()?;
        
        // Create environment with sub-dictionaries bound
        let dict_env = self.bind_sub_dictionaries(env, &sub_dicts, &resolution.instance);
        
        // Evaluate each method body
        let mut methods = HashMap::new();
        for (method_name, method_body) in &resolution.instance.methods {
            let method_value = self.eval_expr(&dict_env, method_body)?;
            methods.insert(method_name.clone(), method_value);
        }
        
        Ok(Value::Dictionary {
            trait_name: resolution.instance.trait_name.clone(),
            methods,
        })
    }
}
```

### 4.4 Transforming Constrained Functions

A function like:

```gneiss
let show_twice x = show x ++ show x
-- Inferred type: Show a => a -> String
```

Gets transformed internally to:

```gneiss
let show_twice show_dict x = 
    (show_dict.show x) ++ (show_dict.show x)
-- Type: ShowDict a -> a -> String
```

This transformation happens during evaluation when we see a constrained function being applied.

---

## Phase 5: Implementation Order

### Step 1: Lexer & Parser (1-2 days)
- Add tokens: `trait`, `impl`, `for`, `where`, `val`, `end`
- Parse trait declarations
- Parse instance declarations
- Parse type signatures with constraints

### Step 2: AST & Type Representation (1 day)
- Add `TraitDecl`, `InstanceDecl` to AST
- Add `QualType`, `TypeConstraint` to types
- Update `Scheme` to include constraints

### Step 3: Trait/Instance Registration (1 day)
- Build `TraitEnv` from parsed declarations
- Validate: methods match signatures, no duplicate instances

### Step 4: Basic Inference with Constraints (2-3 days)
- Track constraints during inference
- Include constraints in generalized schemes
- Check constraints at instantiation

### Step 5: Instance Resolution (2-3 days)
- Implement matching algorithm
- Handle constrained instances (the key feature!)
- Detect and report overlap errors

### Step 6: Dictionary Passing (2-3 days)
- Dictionary value representation
- Dictionary construction from instances
- Transform constrained calls to pass dictionaries

### Step 7: Testing & Polish (2-3 days)
- Test basic traits: `Show`, `Eq`, `Ord`
- Test constrained instances: `Show (List a)`, `Eq (Option a)`
- Test nested: `Show (List (Option Int))`
- Error messages for missing instances

---

## Phase 6: Example Programs

After implementation, these should all work:

### Basic Traits

```gneiss
trait Show a =
  val show : a -> String
end

trait Eq a =
  val eq : a -> a -> Bool
end

impl Show for Int =
  let show n = int_to_string n
end

impl Eq for Int =
  let eq a b = a == b
end
```

### Constrained Instances (Solving n×m)

```gneiss
impl Show for (Option a) where a : Show =
  let show opt = match opt with
    | Some x -> "Some(" ++ show x ++ ")"
    | None -> "None"
end

impl Show for (List a) where a : Show =
  let show xs = match xs with
    | [] -> "[]"
    | _ -> "[" ++ show_items xs ++ "]"
  
  let show_items xs = match xs with
    | [x] -> show x
    | x :: rest -> show x ++ ", " ++ show_items rest
    | [] -> ""
end

-- Now this works automatically!
let test () =
  show (Some 42);                    -- "Some(42)"
  show [1, 2, 3];                    -- "[1, 2, 3]"
  show (Some [1, 2]);                -- "Some([1, 2])"
  show [Some 1, None, Some 3]        -- "[Some(1), None, Some(3)]"
```

### Supertraits

```gneiss
trait Ord a : Eq =
  val compare : a -> a -> Int  -- -1, 0, 1
end

-- Implementing Ord requires Eq to already be implemented
impl Ord for Int =
  let compare a b = if a < b then -1 else if a > b then 1 else 0
end
```

### Default Implementations

```gneiss
trait Eq a =
  val eq : a -> a -> Bool
  
  -- Default implementation in terms of eq
  let neq a b = not (eq a b)
end

impl Eq for Int =
  let eq a b = a == b
  -- neq is automatically available
end
```

---

## What We're NOT Implementing (and Why)

### Higher-Kinded Types (HKT)

HKT would let us write:

```gneiss
-- NOT SUPPORTED
trait Functor f =
  val map : (a -> b) -> f a -> f b
end
```

This requires `f` to be a type constructor like `List` or `Option`, not a concrete type. This is complex because:
1. Kind inference is needed
2. Unification becomes more complex
3. Error messages are harder

**Alternative**: Write specific functions like `List.map`, `Option.map`. Less abstract but simpler.

### Multi-Parameter Type Classes

```gneiss
-- NOT SUPPORTED (initially)
trait Convert a b =
  val convert : a -> b
end
```

These require more sophisticated instance resolution and can lead to ambiguity. We can add later if needed.

### Associated Types

```gneiss
-- NOT SUPPORTED (initially)
trait Iterator a =
  type Item
  val next : a -> Option Item
end
```

These are a simpler alternative to HKT for some use cases. Could be added in a future phase.

---

## Error Messages to Implement

Good error messages are crucial for usability:

```
Error: No instance of Show for Channel Int
  |
5 |   show ch
  |   ^^^^
  |
  = Note: Show is implemented for: Int, Bool, String, List a (where a : Show), Option a (where a : Show)
  = Hint: Channels cannot be shown as they contain runtime state

Error: Overlapping instances for Show (List Int)
  |
  = Instance 1: impl Show for (List a) where a : Show  (at line 10)
  = Instance 2: impl Show for (List Int)              (at line 20)
  = Hint: Remove one of these instances

Error: Could not deduce (Show a) from context
  |
8 | let print_thing x = show x
  |                     ^^^^
  |
  = The type variable 'a' is not constrained to have Show
  = Hint: Add a constraint: let print_thing x = show x  where x : Show
```

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_basic_instance_resolution() {
    // Show Int resolves to ShowInt instance
}

#[test]
fn test_constrained_instance_resolution() {
    // Show (List Int) resolves via Show (List a) with sub-constraint Show Int
}

#[test]
fn test_nested_constrained_resolution() {
    // Show (List (Option Int)) requires Show (List a), Show (Option b), Show Int
}

#[test]
fn test_no_instance_error() {
    // Show (Channel Int) fails with clear error
}

#[test]
fn test_overlap_detection() {
    // Defining Show (List a) and Show (List Int) is an error
}
```

### Integration Tests

```gneiss
-- test_show_basic.gn
trait Show a = val show : a -> String end
impl Show for Int = let show n = int_to_string n end

let main () = print (show 42)
-- Expected output: 42

-- test_show_list.gn
impl Show for (List a) where a : Show = ...

let main () = print (show [1, 2, 3])
-- Expected output: [1, 2, 3]

-- test_show_nested.gn
let main () = print (show [Some 1, None, Some 2])
-- Expected output: [Some(1), None, Some(2)]
```

---

## Summary

| Feature | Included | Rationale |
|---------|----------|-----------|
| Single-param traits | ✅ | Core feature |
| Constrained instances | ✅ | Solves n×m problem |
| Supertraits | ✅ | Useful, not too complex |
| Default methods | ✅ | Reduces boilerplate |
| Dictionary passing | ✅ | Well-understood, flexible |
| HKT | ❌ | Too complex for now |
| Multi-param classes | ❌ | Defer to later |
| Associated types | ❌ | Defer to later |
| Deriving | ❌ | Nice-to-have, add later |

The key insight is that **constrained instances** give you most of the practical benefit of typeclasses without the complexity of HKT. You can't write `Functor` or `Monad`, but you CAN write `Show`, `Eq`, `Ord`, `Serialize`, `Hash`, etc. - which covers the vast majority of real-world use cases.
