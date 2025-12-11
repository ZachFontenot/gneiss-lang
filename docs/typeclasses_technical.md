# Typeclass Technical Specification

This document provides detailed implementation guidance for adding typeclasses to Gneiss.

## Part 1: Type System Changes

### 1.1 Extended Type Grammar

```
Type ::= Int | Bool | String | ...      -- Primitives
       | a                               -- Type variable
       | Type -> Type                    -- Function
       | (Type, Type, ...)              -- Tuple
       | [Type]                          -- List
       | Constructor Type*               -- ADT application
       | Channel Type                    -- Channel

QualType ::= Constraint* => Type         -- Qualified type

Constraint ::= TraitName Type            -- e.g., Show a, Eq (List a)

Scheme ::= forall a* . QualType          -- Polymorphic qualified type
```

### 1.2 Changes to `src/types.rs`

```rust
/// A constraint: TraitName applied to a type
#[derive(Debug, Clone, PartialEq)]
pub struct Pred {
    pub trait_name: String,
    pub ty: Type,
}

impl Pred {
    pub fn new(trait_name: impl Into<String>, ty: Type) -> Self {
        Self { trait_name: trait_name.into(), ty }
    }
    
    /// Apply a type substitution to this predicate
    pub fn apply(&self, subst: &Substitution) -> Pred {
        Pred {
            trait_name: self.trait_name.clone(),
            ty: self.ty.apply(subst),
        }
    }
    
    /// Get free type variables
    pub fn free_vars(&self) -> HashSet<TypeVarId> {
        self.ty.free_vars()
    }
}

/// A qualified type: predicates => type
#[derive(Debug, Clone)]
pub struct Qual<T> {
    pub preds: Vec<Pred>,
    pub head: T,
}

impl<T> Qual<T> {
    pub fn new(preds: Vec<Pred>, head: T) -> Self {
        Self { preds, head }
    }
    
    pub fn unqualified(head: T) -> Self {
        Self { preds: vec![], head }
    }
}

pub type QualType = Qual<Type>;

/// Updated scheme with predicates
#[derive(Debug, Clone)]
pub struct Scheme {
    pub kinds: Vec<Kind>,  // For now, all * (no HKT)
    pub qual: QualType,
}

impl Scheme {
    pub fn mono(ty: Type) -> Self {
        Self {
            kinds: vec![],
            qual: Qual::unqualified(ty),
        }
    }
    
    pub fn from_qual(num_vars: usize, qual: QualType) -> Self {
        Self {
            kinds: vec![Kind::Star; num_vars],
            qual,
        }
    }
}

/// Kind (for future HKT support, currently just *)
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Star,  // *
    // Arrow(Box<Kind>, Box<Kind>),  // * -> * (for HKT, not implemented)
}
```

### 1.3 Substitution Type

```rust
/// A type substitution: maps type variable IDs to types
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    mapping: HashMap<TypeVarId, Type>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn singleton(var: TypeVarId, ty: Type) -> Self {
        let mut s = Self::new();
        s.mapping.insert(var, ty);
        s
    }
    
    pub fn compose(&self, other: &Substitution) -> Substitution {
        // (s1 @@ s2) means: apply s2 first, then s1
        let mut result = Substitution::new();
        
        // Apply s1 to all of s2's mappings
        for (var, ty) in &other.mapping {
            result.mapping.insert(*var, ty.apply(self));
        }
        
        // Add s1's mappings (s1 takes precedence)
        for (var, ty) in &self.mapping {
            result.mapping.entry(*var).or_insert_with(|| ty.clone());
        }
        
        result
    }
    
    pub fn get(&self, var: TypeVarId) -> Option<&Type> {
        self.mapping.get(&var)
    }
}

impl Type {
    pub fn apply(&self, subst: &Substitution) -> Type {
        match self.resolve() {
            Type::Var(var) => {
                match &*var.borrow() {
                    TypeVar::Unbound { id, .. } | TypeVar::Generic(id) => {
                        subst.get(*id).cloned().unwrap_or_else(|| self.clone())
                    }
                    TypeVar::Link(_) => unreachable!("resolve should follow links"),
                }
            }
            Type::Arrow(a, b) => {
                Type::Arrow(Rc::new(a.apply(subst)), Rc::new(b.apply(subst)))
            }
            Type::Tuple(ts) => {
                Type::Tuple(ts.iter().map(|t| t.apply(subst)).collect())
            }
            Type::List(t) => Type::List(Rc::new(t.apply(subst))),
            Type::Channel(t) => Type::Channel(Rc::new(t.apply(subst))),
            Type::Constructor { name, args } => Type::Constructor {
                name,
                args: args.iter().map(|t| t.apply(subst)).collect(),
            },
            t => t,  // Primitives unchanged
        }
    }
}
```

---

## Part 2: Trait Environment

### 2.1 Core Data Structures

```rust
/// A class/trait declaration
#[derive(Debug, Clone)]
pub struct Class {
    /// Superclasses: Ord requires Eq
    pub supers: Vec<String>,
    /// Methods: name -> type (with 'a' as the class parameter)
    pub methods: HashMap<String, Scheme>,
}

/// An instance declaration
#[derive(Debug, Clone)]
pub struct Instance {
    /// Constraints required: Show a, Eq a
    pub preds: Vec<Pred>,
    /// The instance head predicate: Show (List a)
    pub head: Pred,
    /// Method implementations
    pub impls: HashMap<String, Expr>,
}

/// The class/trait environment
#[derive(Debug, Clone, Default)]
pub struct ClassEnv {
    /// Class name -> class info
    pub classes: HashMap<String, Class>,
    /// All instances (searched linearly for now)
    pub instances: Vec<Instance>,
    /// Default implementations from class definitions
    pub defaults: HashMap<String, HashMap<String, Expr>>,
}
```

### 2.2 Instance Matching

The key algorithm - matching an instance head against a goal:

```rust
impl ClassEnv {
    /// Match a predicate against available instances
    /// Returns: (matching instance, substitution, sub-predicates to solve)
    pub fn match_pred(&self, goal: &Pred) -> Result<(Instance, Substitution, Vec<Pred>), TypeError> {
        for inst in &self.instances {
            if inst.head.trait_name != goal.trait_name {
                continue;
            }
            
            // Try to match inst.head.ty against goal.ty
            // This is ONE-WAY matching: we can substitute in the instance,
            // but NOT in the goal.
            if let Some(subst) = self.match_type(&inst.head.ty, &goal.ty) {
                // Apply substitution to instance predicates
                let sub_preds: Vec<_> = inst.preds
                    .iter()
                    .map(|p| p.apply(&subst))
                    .collect();
                
                return Ok((inst.clone(), subst, sub_preds));
            }
        }
        
        Err(TypeError::NoInstance {
            trait_name: goal.trait_name.clone(),
            ty: goal.ty.clone(),
        })
    }
    
    /// One-way match: can `pattern` match `target` by substituting pattern's vars?
    /// Returns substitution if successful.
    fn match_type(&self, pattern: &Type, target: &Type) -> Option<Substitution> {
        let pattern = pattern.resolve();
        let target = target.resolve();
        
        match (&pattern, &target) {
            // Variable in pattern matches anything
            (Type::Var(var), _) => {
                match &*var.borrow() {
                    TypeVar::Generic(id) | TypeVar::Unbound { id, .. } => {
                        Some(Substitution::singleton(*id, target.clone()))
                    }
                    TypeVar::Link(_) => unreachable!(),
                }
            }
            
            // Exact matches
            (Type::Int, Type::Int) => Some(Substitution::new()),
            (Type::Bool, Type::Bool) => Some(Substitution::new()),
            (Type::String, Type::String) => Some(Substitution::new()),
            // ... other primitives
            
            // Structural matches
            (Type::Arrow(p1, p2), Type::Arrow(t1, t2)) => {
                let s1 = self.match_type(p1, t1)?;
                let s2 = self.match_type(&p2.apply(&s1), &t2.apply(&s1))?;
                Some(s1.compose(&s2))
            }
            
            (Type::List(p), Type::List(t)) => self.match_type(p, t),
            
            (Type::Constructor { name: n1, args: a1 }, 
             Type::Constructor { name: n2, args: a2 }) if n1 == n2 && a1.len() == a2.len() => {
                let mut subst = Substitution::new();
                for (p, t) in a1.iter().zip(a2.iter()) {
                    let s = self.match_type(&p.apply(&subst), &t.apply(&subst))?;
                    subst = subst.compose(&s);
                }
                Some(subst)
            }
            
            _ => None,
        }
    }
}
```

### 2.3 Entailment (Constraint Solving)

Check if a set of assumptions entails a predicate:

```rust
impl ClassEnv {
    /// Check if predicates `ps` entail predicate `p`
    /// i.e., given ps, can we derive p?
    pub fn entail(&self, ps: &[Pred], p: &Pred) -> bool {
        // Direct: p is in ps
        if ps.contains(p) {
            return true;
        }
        
        // By superclass: if we have Ord a, we have Eq a
        for given in ps {
            if given.trait_name != p.trait_name && given.ty == p.ty {
                if let Some(class) = self.classes.get(&given.trait_name) {
                    if class.supers.contains(&p.trait_name) {
                        return true;
                    }
                }
            }
        }
        
        // By instance: find matching instance, check sub-predicates
        if let Ok((_, _, sub_preds)) = self.match_pred(p) {
            return sub_preds.iter().all(|sp| self.entail(ps, sp));
        }
        
        false
    }
    
    /// Reduce predicates to simplest form
    /// Remove predicates that are entailed by others
    pub fn reduce(&self, ps: &[Pred]) -> Vec<Pred> {
        let mut result = Vec::new();
        
        for (i, p) in ps.iter().enumerate() {
            let others: Vec<_> = ps.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| p.clone())
                .chain(result.iter().cloned())
                .collect();
            
            if !self.entail(&others, p) {
                result.push(p.clone());
            }
        }
        
        result
    }
}
```

---

## Part 3: Type Inference Changes

### 3.1 Inference State

```rust
pub struct Inferencer {
    next_var: TypeVarId,
    level: u32,
    
    // NEW: Track constraints
    class_env: ClassEnv,
    
    /// Predicates we've collected but not yet resolved
    /// These are "wanted" predicates
    predicates: Vec<Pred>,
}
```

### 3.2 Inferring Method Calls

When we see a call to a trait method:

```rust
impl Inferencer {
    fn infer_expr(&mut self, env: &TypeEnv, expr: &Expr) -> Result<Type, TypeError> {
        match &expr.node {
            ExprKind::Var(name) => {
                // Check if it's a class method
                if let Some((class_name, method_scheme)) = self.class_env.lookup_method(name) {
                    // Instantiate with fresh variables
                    let (preds, ty) = self.instantiate_scheme(method_scheme);
                    
                    // Record predicates as "wanted"
                    self.predicates.extend(preds);
                    
                    return Ok(ty);
                }
                
                // Regular variable lookup
                // ...
            }
            // ...
        }
    }
}
```

### 3.3 Generalization with Predicates

When generalizing at `let`:

```rust
impl Inferencer {
    fn generalize_with_predicates(
        &mut self,
        env: &TypeEnv,
        ty: &Type,
        preds: Vec<Pred>,
    ) -> Scheme {
        // Find type variables that can be generalized
        let env_vars = self.free_vars_in_env(env);
        let ty_vars = ty.free_vars();
        let generalizable: HashSet<_> = ty_vars.difference(&env_vars).copied().collect();
        
        // Split predicates: 
        // - "deferred" = mention only env vars (caller must provide)
        // - "retained" = mention generalizable vars (become part of scheme)
        let (deferred, retained): (Vec<_>, Vec<_>) = preds.into_iter()
            .partition(|p| {
                p.free_vars().is_disjoint(&generalizable)
            });
        
        // Put deferred back for caller to handle
        self.predicates.extend(deferred);
        
        // Simplify retained predicates
        let retained = self.class_env.reduce(&retained);
        
        // Build scheme
        let num_generics = generalizable.len();
        let qual = QualType::new(retained, ty.clone());
        
        Scheme::from_qual(num_generics, qual)
    }
}
```

### 3.4 Context Reduction

At certain points (top-level bindings, before code gen), we must ensure all predicates can be resolved:

```rust
impl Inferencer {
    /// Resolve all collected predicates, or error
    fn discharge_predicates(&mut self) -> Result<Vec<Pred>, TypeError> {
        let mut remaining = Vec::new();
        
        for pred in std::mem::take(&mut self.predicates) {
            if self.class_env.entail(&[], &pred) {
                // Can be resolved by instances alone - good!
            } else {
                // Must be provided by context
                remaining.push(pred);
            }
        }
        
        Ok(remaining)
    }
}
```

---

## Part 4: Dictionary Passing

### 4.1 Dictionary Representation

```rust
/// A dictionary is just a tuple of method implementations
/// For Show: (show_method,)
/// For Eq: (eq_method, neq_method)
/// For Ord: (eq_dict, compare_method)  -- includes superclass dict

pub enum Value {
    // ... existing ...
    
    /// A typeclass dictionary (represented as a record/tuple)
    Dict {
        trait_name: String,
        methods: Vec<(String, Value)>,
    },
}
```

### 4.2 Dictionary Construction

```rust
impl Interpreter {
    /// Build dictionary for a predicate given context dictionaries
    fn build_dict(
        &mut self,
        pred: &Pred,
        context: &[(Pred, Value)],  // Available dictionaries
    ) -> Result<Value, EvalError> {
        // First, check if we already have this dict in context
        for (ctx_pred, dict) in context {
            if ctx_pred == pred {
                return Ok(dict.clone());
            }
        }
        
        // Find matching instance
        let (inst, subst, sub_preds) = self.class_env.match_pred(pred)
            .map_err(|e| EvalError::RuntimeError(e.to_string()))?;
        
        // Recursively build sub-dictionaries
        let sub_dicts: Vec<_> = sub_preds.iter()
            .map(|sp| {
                let dict = self.build_dict(sp, context)?;
                Ok((sp.clone(), dict))
            })
            .collect::<Result<_, EvalError>>()?;
        
        // Evaluate method implementations with sub-dicts in scope
        let mut methods = Vec::new();
        let method_env = self.make_dict_env(&sub_dicts);
        
        for (method_name, method_body) in &inst.impls {
            let method_val = self.eval_expr(&method_env, method_body)?;
            methods.push((method_name.clone(), method_val));
        }
        
        Ok(Value::Dict {
            trait_name: pred.trait_name.clone(),
            methods,
        })
    }
}
```

### 4.3 Transforming Constrained Functions

A constrained function gets extra dictionary parameters:

```gneiss
-- Source
let show_pair (x, y) = "(" ++ show x ++ ", " ++ show y ++ ")"
-- Inferred: (Show a, Show b) => (a, b) -> String

-- Transformed (conceptually)
let show_pair dict_a dict_b (x, y) = 
    "(" ++ dict_a.show x ++ ", " ++ dict_b.show y ++ ")"
```

At call sites:

```gneiss
-- Source
show_pair (1, true)

-- Transformed  
show_pair showIntDict showBoolDict (1, true)
```

### 4.4 Method Selection from Dictionary

```rust
impl Interpreter {
    fn select_method(&self, dict: &Value, method_name: &str) -> Result<Value, EvalError> {
        match dict {
            Value::Dict { methods, .. } => {
                for (name, val) in methods {
                    if name == method_name {
                        return Ok(val.clone());
                    }
                }
                Err(EvalError::RuntimeError(
                    format!("Method {} not found in dictionary", method_name)
                ))
            }
            _ => Err(EvalError::TypeError("Expected dictionary".into())),
        }
    }
}
```

---

## Part 5: Coherence and Safety

### 5.1 Overlap Check

When adding an instance, check it doesn't overlap with existing ones:

```rust
impl ClassEnv {
    pub fn add_instance(&mut self, inst: Instance) -> Result<(), TypeError> {
        // Check for overlap with existing instances
        for existing in &self.instances {
            if existing.head.trait_name != inst.head.trait_name {
                continue;
            }
            
            // Do the heads unify? If so, they overlap
            if self.heads_overlap(&existing.head.ty, &inst.head.ty) {
                return Err(TypeError::OverlappingInstances {
                    trait_name: inst.head.trait_name,
                    type1: existing.head.ty.clone(),
                    type2: inst.head.ty.clone(),
                });
            }
        }
        
        self.instances.push(inst);
        Ok(())
    }
    
    fn heads_overlap(&self, t1: &Type, t2: &Type) -> bool {
        // Two heads overlap if they can be unified
        // (not just matched - unification is symmetric)
        let mut inf = Inferencer::new();
        inf.unify(t1, t2).is_ok()
    }
}
```

### 5.2 Orphan Check (Optional but Recommended)

Prevent orphan instances for coherence:

```rust
impl ClassEnv {
    pub fn check_orphan(&self, inst: &Instance, current_module: &str) -> Result<(), TypeError> {
        let trait_module = self.trait_module(&inst.head.trait_name);
        let type_module = self.type_module(&inst.head.ty);
        
        // Instance is allowed if:
        // - Trait is defined in current module, OR
        // - Type is defined in current module
        if trait_module != current_module && type_module != current_module {
            return Err(TypeError::OrphanInstance {
                trait_name: inst.head.trait_name.clone(),
                ty: inst.head.ty.clone(),
            });
        }
        
        Ok(())
    }
}
```

---

## Part 6: Complete Example Walkthrough

Let's trace through a complete example:

### Source Code

```gneiss
trait Show a =
  val show : a -> String
end

impl Show for Int =
  let show n = int_to_string n
end

impl Show for (List a) where a : Show =
  let show xs = "[" ++ show_list xs ++ "]"
  
  let show_list xs = match xs with
    | [] -> ""
    | [x] -> show x
    | x :: rest -> show x ++ ", " ++ show_list rest
end

let main () = print (show [1, 2, 3])
```

### Step 1: Parse

```rust
TraitDecl {
    name: "Show",
    type_param: "a",
    methods: [("show", a -> String)],
}

InstanceDecl {
    trait_name: "Show",
    target_type: Int,
    constraints: [],
    methods: [("show", fun n -> int_to_string n)],
}

InstanceDecl {
    trait_name: "Show",
    target_type: List a,
    constraints: [Show a],
    methods: [("show", fun xs -> ...)],
}
```

### Step 2: Type Check

When inferring `show [1, 2, 3]`:

1. `show` is a method of `Show` with type `Show a => a -> String`
2. Instantiate: fresh var `t0`, add predicate `Show t0`, type is `t0 -> String`
3. Argument `[1, 2, 3]` has type `List Int`
4. Unify: `t0 = List Int`
5. Predicate becomes: `Show (List Int)`
6. Result type: `String`

### Step 3: Resolve Predicate

Resolve `Show (List Int)`:

1. Match against instances
2. `Show Int` - head is `Int`, doesn't match `List Int`
3. `Show (List a)` - head is `List a`, matches! Substitution: `a = Int`
4. Sub-predicates: `[Show a]` → `[Show Int]`
5. Resolve `Show Int`: matches `Show Int` instance, no sub-preds
6. Done!

### Step 4: Build Dictionaries

```rust
// showIntDict
Dict {
    trait_name: "Show",
    methods: [("show", <closure: int_to_string>)],
}

// showListIntDict (needs showIntDict)
Dict {
    trait_name: "Show", 
    methods: [("show", <closure: fun xs -> "[" ++ ...>)],
    // The closure captures showIntDict for recursive show calls
}
```

### Step 5: Execute

```rust
// show [1, 2, 3]
// Becomes: (showListIntDict.show) [1, 2, 3]
// Which uses showIntDict.show for each element
// Result: "[1, 2, 3]"
```

---

## Summary: Files to Modify

| File | Changes |
|------|---------|
| `lexer.rs` | Add tokens: `trait`, `impl`, `for`, `where`, `val` |
| `ast.rs` | Add `TraitDecl`, `InstanceDecl`, `Constraint` |
| `parser.rs` | Parse trait/instance declarations |
| `types.rs` | Add `Pred`, `Qual`, `Substitution`, update `Scheme` |
| `infer.rs` | Add `ClassEnv`, predicate collection, entailment |
| `eval.rs` | Add `Value::Dict`, dictionary construction |
| NEW: `class.rs` | Core typeclass logic (could split from infer) |

Estimated total: ~1500-2000 lines of new/modified code.
