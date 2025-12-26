# Union-Find for Type Inference

A guide to implementing efficient type unification using the Union-Find (Disjoint Set Union) data structure.

---

## The Problem

Your current implementation uses linked type variables:

```rust
pub enum TypeVar {
    Unbound { id: TypeVarId, level: u32 },
    Link(Type),  // ← This creates chains
    Generic(TypeVarId),
}
```

When you unify `?a` with `?b`, then `?b` with `?c`, you get:

```
?a → ?b → ?c
```

Following this chain is O(n). Deep unification creates long chains, leading to:
- Stack overflow from recursive `resolve()` calls
- O(n) performance for each variable lookup

---

## The Solution: Union-Find

Union-Find gives you near-O(1) operations with two optimizations:

1. **Path compression** — When you find the root, make all nodes point directly to it
2. **Union by rank** — Attach smaller trees under larger ones

### Complexity Comparison

| Operation | Current (Linked) | Union-Find |
|-----------|-----------------|------------|
| Find root | O(n) worst case | O(α(n)) ≈ O(1) |
| Unify vars | O(1) + O(n) find | O(α(n)) ≈ O(1) |
| Space | O(n) | O(n) |
| Stack usage | O(n) recursive | O(1) iterative |

Where α(n) is the inverse Ackermann function — effectively constant for any realistic input.

---

## Implementation

### Core Union-Find Structure

```rust
use std::cell::Cell;

/// A type variable ID
pub type TypeVarId = u32;

/// Union-Find structure for type inference
pub struct UnionFind {
    /// Parent pointers. If parent[i] == i, it's a root.
    parent: Vec<Cell<TypeVarId>>,
    
    /// Rank for union by rank optimization
    rank: Vec<u32>,
    
    /// The actual type bound to each equivalence class root.
    /// None = still unbound, Some = bound to concrete type
    binding: Vec<Option<Type>>,
    
    /// Level for let-polymorphism (stored per variable)
    level: Vec<u32>,
}

impl UnionFind {
    pub fn new() -> Self {
        UnionFind {
            parent: Vec::new(),
            rank: Vec::new(),
            binding: Vec::new(),
            level: Vec::new(),
        }
    }
    
    /// Create a fresh unbound type variable
    pub fn fresh(&mut self, level: u32) -> TypeVarId {
        let id = self.parent.len() as TypeVarId;
        self.parent.push(Cell::new(id));  // Points to itself (root)
        self.rank.push(0);
        self.binding.push(None);  // Unbound
        self.level.push(level);
        id
    }
    
    /// Find the root of a type variable with path compression
    pub fn find(&self, mut x: TypeVarId) -> TypeVarId {
        // Find root
        let mut root = x;
        while self.parent[root as usize].get() != root {
            root = self.parent[root as usize].get();
        }
        
        // Path compression: make all nodes point directly to root
        while self.parent[x as usize].get() != root {
            let next = self.parent[x as usize].get();
            self.parent[x as usize].set(root);
            x = next;
        }
        
        root
    }
    
    /// Get the binding for a type variable (follows to root)
    pub fn get_binding(&self, x: TypeVarId) -> Option<&Type> {
        let root = self.find(x);
        self.binding[root as usize].as_ref()
    }
    
    /// Get the level for a type variable
    pub fn get_level(&self, x: TypeVarId) -> u32 {
        let root = self.find(x);
        self.level[root as usize]
    }
    
    /// Check if a variable is bound
    pub fn is_bound(&self, x: TypeVarId) -> bool {
        self.get_binding(x).is_some()
    }
    
    /// Unify two type variables (union operation)
    pub fn union(&mut self, x: TypeVarId, y: TypeVarId) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x == root_y {
            return;  // Already in same set
        }
        
        // Union by rank: attach smaller tree under larger
        let rank_x = self.rank[root_x as usize];
        let rank_y = self.rank[root_y as usize];
        
        let (new_root, other) = if rank_x < rank_y {
            (root_y, root_x)
        } else {
            (root_x, root_y)
        };
        
        self.parent[other as usize].set(new_root);
        
        if rank_x == rank_y {
            self.rank[new_root as usize] += 1;
        }
        
        // Merge levels: take the minimum (more polymorphic)
        let min_level = self.level[root_x as usize].min(self.level[root_y as usize]);
        self.level[new_root as usize] = min_level;
        
        // If either was bound, the merged set is bound
        if self.binding[other as usize].is_some() && self.binding[new_root as usize].is_none() {
            self.binding[new_root as usize] = self.binding[other as usize].take();
        }
    }
    
    /// Bind a type variable to a concrete type
    pub fn bind(&mut self, x: TypeVarId, ty: Type) {
        let root = self.find(x);
        debug_assert!(self.binding[root as usize].is_none(), "Already bound!");
        self.binding[root as usize] = Some(ty);
    }
    
    /// Update level of a variable (for let-polymorphism)
    pub fn set_level(&mut self, x: TypeVarId, new_level: u32) {
        let root = self.find(x);
        self.level[root as usize] = self.level[root as usize].min(new_level);
    }
}
```

---

### Simplified Type Representation

With Union-Find, types become cleaner — no more `RefCell` chains:

```rust
#[derive(Debug, Clone)]
pub enum Type {
    /// Type variable — just an ID, Union-Find handles the rest
    Var(TypeVarId),
    
    /// Primitives
    Int,
    Float,
    Bool,
    String,
    Char,
    Unit,
    
    /// Function type
    Arrow {
        arg: Rc<Type>,
        ret: Rc<Type>,
        effects: Row,
    },
    
    /// Type constructor application
    Constructor {
        name: String,
        args: Vec<Type>,
    },
    
    Tuple(Vec<Type>),
    
    Channel(Rc<Type>),
    Fiber(Rc<Type>),
    Dict(Rc<Type>),
    // etc.
}
```

---

### Type Resolution

```rust
impl Type {
    /// Resolve a type, following type variable bindings
    pub fn resolve(&self, uf: &UnionFind) -> Type {
        match self {
            Type::Var(id) => {
                match uf.get_binding(*id) {
                    Some(ty) => ty.resolve(uf),  // Recurse into binding
                    None => Type::Var(uf.find(*id)),  // Return canonical var
                }
            }
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(arg.resolve(uf)),
                ret: Rc::new(ret.resolve(uf)),
                effects: effects.resolve(uf),
            },
            Type::Tuple(ts) => Type::Tuple(
                ts.iter().map(|t| t.resolve(uf)).collect()
            ),
            Type::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args.iter().map(|t| t.resolve(uf)).collect(),
            },
            Type::Channel(inner) => Type::Channel(Rc::new(inner.resolve(uf))),
            Type::Fiber(inner) => Type::Fiber(Rc::new(inner.resolve(uf))),
            Type::Dict(inner) => Type::Dict(Rc::new(inner.resolve(uf))),
            // Primitives are already resolved
            _ => self.clone(),
        }
    }
    
    /// Occurs check: does variable `id` occur in this type?
    pub fn occurs(&self, id: TypeVarId, uf: &UnionFind) -> bool {
        match self.resolve(uf) {
            Type::Var(other_id) => uf.find(other_id) == id,
            Type::Arrow { arg, ret, effects } => {
                arg.occurs(id, uf) || ret.occurs(id, uf) || effects.type_var_occurs(id, uf)
            }
            Type::Tuple(ts) => ts.iter().any(|t| t.occurs(id, uf)),
            Type::Constructor { args, .. } => args.iter().any(|t| t.occurs(id, uf)),
            Type::Channel(inner) | Type::Fiber(inner) | Type::Dict(inner) => {
                inner.occurs(id, uf)
            }
            _ => false,
        }
    }
}
```

---

### Row Union-Find

Effect rows need their own Union-Find:

```rust
/// Union-Find for row variables
pub struct RowUnionFind {
    parent: Vec<Cell<TypeVarId>>,
    rank: Vec<u32>,
    binding: Vec<Option<Row>>,
    level: Vec<u32>,
}

impl RowUnionFind {
    pub fn new() -> Self {
        RowUnionFind {
            parent: Vec::new(),
            rank: Vec::new(),
            binding: Vec::new(),
            level: Vec::new(),
        }
    }
    
    pub fn fresh(&mut self, level: u32) -> TypeVarId {
        let id = self.parent.len() as TypeVarId;
        self.parent.push(Cell::new(id));
        self.rank.push(0);
        self.binding.push(None);
        self.level.push(level);
        id
    }
    
    pub fn find(&self, mut x: TypeVarId) -> TypeVarId {
        let mut root = x;
        while self.parent[root as usize].get() != root {
            root = self.parent[root as usize].get();
        }
        while self.parent[x as usize].get() != root {
            let next = self.parent[x as usize].get();
            self.parent[x as usize].set(root);
            x = next;
        }
        root
    }
    
    pub fn get_binding(&self, x: TypeVarId) -> Option<&Row> {
        let root = self.find(x);
        self.binding[root as usize].as_ref()
    }
    
    pub fn get_level(&self, x: TypeVarId) -> u32 {
        let root = self.find(x);
        self.level[root as usize]
    }
    
    pub fn union(&mut self, x: TypeVarId, y: TypeVarId) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x == root_y {
            return;
        }
        
        let rank_x = self.rank[root_x as usize];
        let rank_y = self.rank[root_y as usize];
        
        let (new_root, other) = if rank_x < rank_y {
            (root_y, root_x)
        } else {
            (root_x, root_y)
        };
        
        self.parent[other as usize].set(new_root);
        
        if rank_x == rank_y {
            self.rank[new_root as usize] += 1;
        }
        
        let min_level = self.level[root_x as usize].min(self.level[root_y as usize]);
        self.level[new_root as usize] = min_level;
        
        if self.binding[other as usize].is_some() && self.binding[new_root as usize].is_none() {
            self.binding[new_root as usize] = self.binding[other as usize].take();
        }
    }
    
    pub fn bind(&mut self, x: TypeVarId, row: Row) {
        let root = self.find(x);
        debug_assert!(self.binding[root as usize].is_none(), "Already bound!");
        self.binding[root as usize] = Some(row);
    }
    
    pub fn set_level(&mut self, x: TypeVarId, new_level: u32) {
        let root = self.find(x);
        self.level[root as usize] = self.level[root as usize].min(new_level);
    }
}
```

---

### Simplified Row Representation

```rust
#[derive(Debug, Clone)]
pub enum Row {
    /// Empty row — no effects (pure)
    Empty,
    
    /// Extend row with an effect: { E params | rest }
    Extend {
        effect: Effect,
        rest: Rc<Row>,
    },
    
    /// Row variable — just an ID
    Var(TypeVarId),
}

impl Row {
    pub fn resolve(&self, row_uf: &RowUnionFind) -> Row {
        match self {
            Row::Var(id) => {
                match row_uf.get_binding(*id) {
                    Some(row) => row.resolve(row_uf),
                    None => Row::Var(row_uf.find(*id)),
                }
            }
            Row::Extend { effect, rest } => Row::Extend {
                effect: effect.clone(),
                rest: Rc::new(rest.resolve(row_uf)),
            },
            Row::Empty => Row::Empty,
        }
    }
    
    pub fn is_empty(&self, row_uf: &RowUnionFind) -> bool {
        matches!(self.resolve(row_uf), Row::Empty)
    }
    
    /// Check if row variable `id` occurs in this row
    pub fn occurs(&self, id: TypeVarId, row_uf: &RowUnionFind) -> bool {
        match self.resolve(row_uf) {
            Row::Empty => false,
            Row::Var(other_id) => row_uf.find(other_id) == id,
            Row::Extend { rest, .. } => rest.occurs(id, row_uf),
        }
    }
    
    /// Check if a type variable occurs in this row (in effect parameters)
    pub fn type_var_occurs(&self, id: TypeVarId, uf: &UnionFind, row_uf: &RowUnionFind) -> bool {
        match self.resolve(row_uf) {
            Row::Empty => false,
            Row::Var(_) => false,
            Row::Extend { effect, rest } => {
                effect.params.iter().any(|t| t.occurs(id, uf)) 
                    || rest.type_var_occurs(id, uf, row_uf)
            }
        }
    }
}
```

---

### Updated Inferencer

```rust
pub struct Inferencer {
    /// Union-Find for type variables
    type_uf: UnionFind,
    
    /// Union-Find for row variables
    row_uf: RowUnionFind,
    
    /// Current let-nesting level for polymorphism
    level: u32,
    
    /// Type environment
    env: TypeEnv,
    
    /// Effect environment
    effect_env: EffectEnv,
    
    /// Type context (constructors, records, etc.)
    type_ctx: TypeContext,
}

impl Inferencer {
    pub fn new() -> Self {
        Inferencer {
            type_uf: UnionFind::new(),
            row_uf: RowUnionFind::new(),
            level: 0,
            env: TypeEnv::new(),
            effect_env: EffectEnv::new(),
            type_ctx: TypeContext::new(),
        }
    }
    
    /// Create a fresh type variable
    pub fn fresh_var(&mut self) -> Type {
        Type::Var(self.type_uf.fresh(self.level))
    }
    
    /// Create a fresh row variable
    pub fn fresh_row_var(&mut self) -> Row {
        Row::Var(self.row_uf.fresh(self.level))
    }
}
```

---

### Iterative Unification

The key benefit: unification becomes iterative, eliminating stack overflow:

```rust
impl Inferencer {
    /// Unify two types — iterative, no recursion!
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let mut stack = vec![(t1.clone(), t2.clone())];
        
        while let Some((t1, t2)) = stack.pop() {
            let t1 = t1.resolve(&self.type_uf);
            let t2 = t2.resolve(&self.type_uf);
            
            match (&t1, &t2) {
                // Two unbound variables — union them
                (Type::Var(id1), Type::Var(id2)) => {
                    let root1 = self.type_uf.find(*id1);
                    let root2 = self.type_uf.find(*id2);
                    if root1 != root2 {
                        self.type_uf.union(root1, root2);
                    }
                }
                
                // Variable and concrete type — bind the variable
                (Type::Var(id), ty) | (ty, Type::Var(id)) => {
                    let root = self.type_uf.find(*id);
                    
                    // Occurs check
                    if ty.occurs(root, &self.type_uf) {
                        return Err(TypeError::OccursCheck { 
                            var_id: root, 
                            ty: ty.clone() 
                        });
                    }
                    
                    // Update levels in ty
                    let var_level = self.type_uf.get_level(root);
                    self.update_levels(ty, var_level);
                    
                    // Bind
                    self.type_uf.bind(root, ty.clone());
                }
                
                // Primitives — must match exactly
                (Type::Int, Type::Int) => {}
                (Type::Bool, Type::Bool) => {}
                (Type::String, Type::String) => {}
                (Type::Unit, Type::Unit) => {}
                (Type::Char, Type::Char) => {}
                (Type::Float, Type::Float) => {}
                
                // Arrow — push components onto stack
                (
                    Type::Arrow { arg: a1, ret: r1, effects: e1 },
                    Type::Arrow { arg: a2, ret: r2, effects: e2 },
                ) => {
                    stack.push(((**a1).clone(), (**a2).clone()));
                    stack.push(((**r1).clone(), (**r2).clone()));
                    self.unify_rows(e1, e2)?;
                }
                
                // Tuple — push all elements
                (Type::Tuple(ts1), Type::Tuple(ts2)) if ts1.len() == ts2.len() => {
                    for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                        stack.push((t1.clone(), t2.clone()));
                    }
                }
                
                // Constructor — push all args
                (
                    Type::Constructor { name: n1, args: a1 },
                    Type::Constructor { name: n2, args: a2 },
                ) if n1 == n2 && a1.len() == a2.len() => {
                    for (t1, t2) in a1.iter().zip(a2.iter()) {
                        stack.push((t1.clone(), t2.clone()));
                    }
                }
                
                // Channel
                (Type::Channel(inner1), Type::Channel(inner2)) => {
                    stack.push(((**inner1).clone(), (**inner2).clone()));
                }
                
                // Fiber
                (Type::Fiber(inner1), Type::Fiber(inner2)) => {
                    stack.push(((**inner1).clone(), (**inner2).clone()));
                }
                
                // Dict
                (Type::Dict(inner1), Type::Dict(inner2)) => {
                    stack.push(((**inner1).clone(), (**inner2).clone()));
                }
                
                // Mismatch
                (t1, t2) => {
                    return Err(TypeError::TypeMismatch {
                        expected: t1.clone(),
                        found: t2.clone(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Unify two effect rows
    pub fn unify_rows(&mut self, r1: &Row, r2: &Row) -> Result<(), TypeError> {
        let mut stack = vec![(r1.clone(), r2.clone())];
        
        while let Some((r1, r2)) = stack.pop() {
            let r1 = r1.resolve(&self.row_uf);
            let r2 = r2.resolve(&self.row_uf);
            
            match (&r1, &r2) {
                // Both empty — done
                (Row::Empty, Row::Empty) => {}
                
                // Two unbound row variables — union them
                (Row::Var(id1), Row::Var(id2)) => {
                    let root1 = self.row_uf.find(*id1);
                    let root2 = self.row_uf.find(*id2);
                    if root1 != root2 {
                        self.row_uf.union(root1, root2);
                    }
                }
                
                // Row variable and empty — bind to empty
                (Row::Var(id), Row::Empty) | (Row::Empty, Row::Var(id)) => {
                    let root = self.row_uf.find(*id);
                    self.row_uf.bind(root, Row::Empty);
                }
                
                // Row variable and extend — bind var to the extended row
                (Row::Var(id), row @ Row::Extend { .. }) 
                | (row @ Row::Extend { .. }, Row::Var(id)) => {
                    let root = self.row_uf.find(*id);
                    
                    // Occurs check for row variable
                    if row.occurs(root, &self.row_uf) {
                        return Err(TypeError::RowOccursCheck { 
                            var_id: root 
                        });
                    }
                    
                    self.row_uf.bind(root, row.clone());
                }
                
                // Both extend — need row rewriting
                (
                    Row::Extend { effect: e1, rest: rest1 },
                    Row::Extend { effect: e2, rest: rest2 },
                ) => {
                    if e1.name == e2.name {
                        // Same effect — unify parameters and tails
                        if e1.params.len() != e2.params.len() {
                            return Err(TypeError::EffectParamMismatch {
                                effect: e1.name.clone(),
                            });
                        }
                        for (p1, p2) in e1.params.iter().zip(e2.params.iter()) {
                            self.unify(p1, p2)?;
                        }
                        stack.push(((**rest1).clone(), (**rest2).clone()));
                    } else {
                        // Different effects — row rewriting
                        // Find e1 in r2 (or vice versa)
                        self.row_rewrite(&e1, rest1, &r2)?;
                    }
                }
                
                // Empty vs Extend — effect not handled
                (Row::Empty, Row::Extend { effect, .. })
                | (Row::Extend { effect, .. }, Row::Empty) => {
                    return Err(TypeError::UnhandledEffect {
                        effect: effect.name.clone(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Row rewriting: find effect e1 in row2 and unify remainders
    fn row_rewrite(
        &mut self,
        e1: &Effect,
        rest1: &Row,
        row2: &Row,
    ) -> Result<(), TypeError> {
        match row2.resolve(&self.row_uf) {
            Row::Empty => {
                Err(TypeError::EffectNotFound {
                    effect: e1.name.clone(),
                })
            }
            Row::Var(id) => {
                // row2 is a variable — instantiate it to { e1 | fresh }
                let root = self.row_uf.find(id);
                let fresh_rest = self.fresh_row_var();
                let new_row = Row::Extend {
                    effect: e1.clone(),
                    rest: Rc::new(fresh_rest.clone()),
                };
                self.row_uf.bind(root, new_row);
                self.unify_rows(rest1, &fresh_rest)
            }
            Row::Extend { effect: e2, rest: rest2 } => {
                if e1.name == e2.name {
                    // Found it — unify effect params and tails
                    if e1.params.len() != e2.params.len() {
                        return Err(TypeError::EffectParamMismatch {
                            effect: e1.name.clone(),
                        });
                    }
                    for (p1, p2) in e1.params.iter().zip(e2.params.iter()) {
                        self.unify(p1, p2)?;
                    }
                    self.unify_rows(rest1, &rest2)
                } else {
                    // Not this one — keep looking, but need to preserve e2
                    // rest1 should unify with { e2 | rest2' } where e1 is in rest2
                    let fresh = self.fresh_row_var();
                    self.row_rewrite(e1, &fresh, &rest2)?;
                    let reconstructed = Row::Extend {
                        effect: e2.clone(),
                        rest: Rc::new(fresh),
                    };
                    self.unify_rows(rest1, &reconstructed)
                }
            }
        }
    }
    
    /// Update levels for all type variables in a type
    fn update_levels(&mut self, ty: &Type, max_level: u32) {
        match ty {
            Type::Var(id) => {
                self.type_uf.set_level(*id, max_level);
            }
            Type::Arrow { arg, ret, effects } => {
                self.update_levels(arg, max_level);
                self.update_levels(ret, max_level);
                self.update_levels_in_row(effects, max_level);
            }
            Type::Tuple(ts) => {
                for t in ts {
                    self.update_levels(t, max_level);
                }
            }
            Type::Constructor { args, .. } => {
                for t in args {
                    self.update_levels(t, max_level);
                }
            }
            Type::Channel(inner) | Type::Fiber(inner) | Type::Dict(inner) => {
                self.update_levels(inner, max_level);
            }
            _ => {}
        }
    }
    
    /// Update levels for row variables and type variables in effect params
    fn update_levels_in_row(&mut self, row: &Row, max_level: u32) {
        match row.resolve(&self.row_uf) {
            Row::Empty => {}
            Row::Var(id) => {
                self.row_uf.set_level(id, max_level);
            }
            Row::Extend { effect, rest } => {
                for param in &effect.params {
                    self.update_levels(param, max_level);
                }
                self.update_levels_in_row(&rest, max_level);
            }
        }
    }
}
```

---

## Generalization

With Union-Find, generalization needs to check variable levels:

```rust
impl Inferencer {
    /// Generalize a type into a polymorphic scheme
    pub fn generalize(&self, ty: &Type) -> Scheme {
        let mut generics = HashMap::new();
        let mut row_generics = HashMap::new();
        let mut next_generic = 0u32;
        
        let generalized = self.generalize_inner(
            ty, 
            &mut generics, 
            &mut row_generics,
            &mut next_generic
        );
        
        Scheme {
            type_vars: generics.into_iter().map(|(_, id)| id).collect(),
            row_vars: row_generics.into_iter().map(|(_, id)| id).collect(),
            ty: generalized,
        }
    }
    
    fn generalize_inner(
        &self,
        ty: &Type,
        generics: &mut HashMap<TypeVarId, TypeVarId>,
        row_generics: &mut HashMap<TypeVarId, TypeVarId>,
        next_generic: &mut u32,
    ) -> Type {
        match ty.resolve(&self.type_uf) {
            Type::Var(id) => {
                let root = self.type_uf.find(id);
                let var_level = self.type_uf.get_level(root);
                
                if var_level > self.level {
                    // This variable should be generalized
                    let generic_id = *generics.entry(root).or_insert_with(|| {
                        let id = *next_generic;
                        *next_generic += 1;
                        id
                    });
                    Type::Generic(generic_id)
                } else {
                    // Keep as a specific variable
                    Type::Var(root)
                }
            }
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(self.generalize_inner(&arg, generics, row_generics, next_generic)),
                ret: Rc::new(self.generalize_inner(&ret, generics, row_generics, next_generic)),
                effects: self.generalize_row(&effects, generics, row_generics, next_generic),
            },
            Type::Tuple(ts) => Type::Tuple(
                ts.iter()
                    .map(|t| self.generalize_inner(t, generics, row_generics, next_generic))
                    .collect()
            ),
            Type::Constructor { name, args } => Type::Constructor {
                name,
                args: args.iter()
                    .map(|t| self.generalize_inner(t, generics, row_generics, next_generic))
                    .collect(),
            },
            // Primitives stay as-is
            other => other,
        }
    }
    
    fn generalize_row(
        &self,
        row: &Row,
        generics: &mut HashMap<TypeVarId, TypeVarId>,
        row_generics: &mut HashMap<TypeVarId, TypeVarId>,
        next_generic: &mut u32,
    ) -> Row {
        match row.resolve(&self.row_uf) {
            Row::Empty => Row::Empty,
            Row::Var(id) => {
                let root = self.row_uf.find(id);
                let var_level = self.row_uf.get_level(root);
                
                if var_level > self.level {
                    let generic_id = *row_generics.entry(root).or_insert_with(|| {
                        let id = *next_generic;
                        *next_generic += 1;
                        id
                    });
                    Row::Generic(generic_id)
                } else {
                    Row::Var(root)
                }
            }
            Row::Extend { effect, rest } => Row::Extend {
                effect: Effect {
                    name: effect.name,
                    params: effect.params.iter()
                        .map(|t| self.generalize_inner(t, generics, row_generics, next_generic))
                        .collect(),
                },
                rest: Rc::new(self.generalize_row(&rest, generics, row_generics, next_generic)),
            },
        }
    }
}
```

---

## Instantiation

```rust
impl Inferencer {
    /// Instantiate a scheme with fresh variables
    pub fn instantiate(&mut self, scheme: &Scheme) -> Type {
        let mut type_subst: HashMap<TypeVarId, Type> = HashMap::new();
        let mut row_subst: HashMap<TypeVarId, Row> = HashMap::new();
        
        // Create fresh vars for each generic
        for generic_id in &scheme.type_vars {
            type_subst.insert(*generic_id, self.fresh_var());
        }
        for generic_id in &scheme.row_vars {
            row_subst.insert(*generic_id, self.fresh_row_var());
        }
        
        self.substitute(&scheme.ty, &type_subst, &row_subst)
    }
    
    fn substitute(
        &self,
        ty: &Type,
        type_subst: &HashMap<TypeVarId, Type>,
        row_subst: &HashMap<TypeVarId, Row>,
    ) -> Type {
        match ty {
            Type::Generic(id) => {
                type_subst.get(id).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Var(id) => Type::Var(*id),  // Already instantiated
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(self.substitute(arg, type_subst, row_subst)),
                ret: Rc::new(self.substitute(ret, type_subst, row_subst)),
                effects: self.substitute_row(effects, type_subst, row_subst),
            },
            Type::Tuple(ts) => Type::Tuple(
                ts.iter().map(|t| self.substitute(t, type_subst, row_subst)).collect()
            ),
            Type::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args.iter().map(|t| self.substitute(t, type_subst, row_subst)).collect(),
            },
            other => other.clone(),
        }
    }
    
    fn substitute_row(
        &self,
        row: &Row,
        type_subst: &HashMap<TypeVarId, Type>,
        row_subst: &HashMap<TypeVarId, Row>,
    ) -> Row {
        match row {
            Row::Empty => Row::Empty,
            Row::Generic(id) => {
                row_subst.get(id).cloned().unwrap_or_else(|| row.clone())
            }
            Row::Var(id) => Row::Var(*id),
            Row::Extend { effect, rest } => Row::Extend {
                effect: Effect {
                    name: effect.name.clone(),
                    params: effect.params.iter()
                        .map(|t| self.substitute(t, type_subst, row_subst))
                        .collect(),
                },
                rest: Rc::new(self.substitute_row(rest, type_subst, row_subst)),
            },
        }
    }
}
```

---

## Migration Path

1. **Add `UnionFind` and `RowUnionFind`** as new structs
2. **Change `Type::Var`** to hold `TypeVarId` instead of `Rc<RefCell<TypeVar>>`
3. **Add `Type::Generic(TypeVarId)`** for polymorphic variables in schemes
4. **Update `Row::Var`** similarly
5. **Add `Row::Generic(TypeVarId)`** for polymorphic row variables
6. **Update `Scheme`** to store lists of generic IDs
7. **Pass `&UnionFind` / `&RowUnionFind`** to resolve/occurs functions
8. **Convert unify to iterative** using the explicit stack
9. **Remove `TypeVar::Link`** — Union-Find handles linking now

---

## Summary

The key insight: your `TypeVar::Link(Type)` is an ad-hoc Union-Find. Making it explicit with:
- **Path compression** — O(1) find instead of O(n)
- **Union by rank** — Balanced trees
- **Iterative unification** — No stack overflow

This is the standard approach used by production type checkers (OCaml, GHC, Rust).
