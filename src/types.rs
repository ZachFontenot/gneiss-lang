//! Internal type representation for type inference
//!
//! Uses Union-Find (Disjoint Set Union) for efficient type variable unification.
//! This provides O(α(n)) ≈ O(1) amortized operations with:
//! - Path compression in find()
//! - Union by rank in union()

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

// Note: RefCell is no longer needed - Union-Find uses Cell for interior mutability

/// A type variable ID
pub type TypeVarId = u32;

// ============================================================================
// Union-Find for Type Variables
// ============================================================================

/// Union-Find structure for type variable unification.
/// Provides near-O(1) operations via path compression and union-by-rank.
#[derive(Debug, Clone)]
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

impl Default for UnionFind {
    fn default() -> Self {
        Self::new()
    }
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

    /// Find the root of a type variable with path compression.
    /// This is O(α(n)) amortized where α is the inverse Ackermann function.
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

    /// Unify two type variables (union operation).
    /// Uses union-by-rank for balanced trees.
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

// ============================================================================
// Union-Find for Row Variables
// ============================================================================

/// Union-Find for row variables (effect rows).
/// Separate from type UnionFind since rows and types are distinct.
#[derive(Debug, Clone)]
pub struct RowUnionFind {
    parent: Vec<Cell<TypeVarId>>,
    rank: Vec<u32>,
    binding: Vec<Option<Row>>,
    level: Vec<u32>,
}

impl Default for RowUnionFind {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn is_bound(&self, x: TypeVarId) -> bool {
        self.get_binding(x).is_some()
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

// ============================================================================
// Type Representation
// ============================================================================

/// The internal representation of types during inference.
/// Type variables are now just IDs - the UnionFind handles bindings/levels.
#[derive(Debug, Clone)]
pub enum Type {
    /// Type variable — just an ID, UnionFind handles the rest
    Var(TypeVarId),

    /// Generic type variable (for polymorphic types in schemes)
    Generic(TypeVarId),

    /// Primitive types
    Int,
    Float,
    Bool,
    String,
    Char,
    Unit,
    Bytes,

    /// I/O handle types (opaque handles for file/socket operations)
    FileHandle,
    TcpSocket,
    TcpListener,

    /// Function type with effect row: σ -> τ { effects }
    /// For pure functions, effects is Row::Empty
    /// For effectful functions, effects contains the required effects
    Arrow {
        arg: Rc<Type>,    // σ: argument type
        ret: Rc<Type>,    // τ: return type
        effects: Row,     // effect row (empty for pure, non-empty for effectful)
    },

    /// Tuple type: (a, b, c)
    Tuple(Vec<Type>),

    /// Named type constructor with arguments: Option a, Result a e, List a
    Constructor {
        name: String,
        args: Vec<Type>,
    },

    /// Channel type: Channel a
    Channel(Rc<Type>),

    /// Process ID type (returned by spawn) - deprecated, use Fiber instead
    Pid,

    /// Fiber type: Fiber a (typed fiber handle, result of spawning)
    /// The type parameter is the result type when joined
    Fiber(Rc<Type>),

    /// Dictionary type: Dict v (String-keyed dictionary with value type v)
    Dict(Rc<Type>),

    /// Set type: Set (String elements)
    Set,
}

// ============================================================================
// Effect Row Types (for Koka-style Algebraic Effects)
// ============================================================================

/// An effect row representing a set of effects with optional extension variable.
/// Syntax: `{ eff1, eff2 s | r }` where r is the row extension variable.
/// Row variables are just IDs - the RowUnionFind handles bindings/levels.
#[derive(Debug, Clone)]
pub enum Row {
    /// Empty row - pure, no effects
    Empty,
    /// Effect present with tail: { eff | rest }
    Extend {
        effect: Effect,
        rest: Rc<Row>,
    },
    /// Row variable — just an ID, RowUnionFind handles the rest
    Var(TypeVarId),
    /// Generic row variable (for polymorphic effect rows in schemes)
    Generic(TypeVarId),
}

/// A named effect with optional type parameters
/// e.g., State s, Reader r, IO
#[derive(Debug, Clone)]
pub struct Effect {
    /// Effect name (e.g., "State", "IO", "Reader")
    pub name: String,
    /// Type parameters for the effect (e.g., [s] for State s)
    pub params: Vec<Type>,
}

impl PartialEq for Effect {
    fn eq(&self, other: &Self) -> bool {
        // For effect comparison, we only compare names
        // Type parameters are handled during unification
        self.name == other.name
    }
}

impl Eq for Effect {}

/// Information about an effect operation
#[derive(Debug, Clone)]
pub struct OperationInfo {
    /// Operation name (e.g., "get", "put", "ask")
    pub name: String,
    /// Parameter types (may contain Generic vars for effect type params)
    pub param_types: Vec<Type>,
    /// Return type
    pub result_type: Type,
    /// Operation-local generic type variable IDs (e.g., for `fail : String -> a` where `a` is not an effect param)
    /// These are quantified at the operation level and get fresh vars at each perform
    pub generics: Vec<TypeVarId>,
}

/// Information about a declared effect
#[derive(Debug, Clone)]
pub struct EffectInfo {
    /// Effect name
    pub name: String,
    /// Type parameter names (e.g., ["s"] for State s)
    pub type_params: Vec<String>,
    /// Operations provided by this effect
    pub operations: Vec<OperationInfo>,
}

/// Environment storing effect declarations
#[derive(Debug, Clone, Default)]
pub struct EffectEnv {
    /// Effect declarations: effect_name -> EffectInfo
    pub effects: HashMap<String, EffectInfo>,
    /// Operation lookup: operation_name -> (effect_name, OperationInfo)
    pub operations: HashMap<String, (String, OperationInfo)>,
}

impl Row {
    /// Create a new generic row variable
    pub fn new_generic(id: TypeVarId) -> Row {
        Row::Generic(id)
    }

    /// Extend a row with an effect
    pub fn extend(effect: Effect, rest: Row) -> Row {
        Row::Extend {
            effect,
            rest: Rc::new(rest),
        }
    }

    /// Check if a type variable syntactically occurs in this row
    pub fn occurs_syntactic_in_row(&self, id: TypeVarId) -> bool {
        match self {
            Row::Empty => false,
            Row::Var(_) | Row::Generic(_) => false, // Row vars don't contain type vars
            Row::Extend { effect, rest } => {
                effect.params.iter().any(|t| t.occurs_syntactic(id))
                    || rest.occurs_syntactic_in_row(id)
            }
        }
    }

    /// Follow all links to get the actual row.
    /// Uses the RowUnionFind for bindings with path compression.
    pub fn resolve(&self, row_uf: &RowUnionFind) -> Row {
        match self {
            Row::Var(id) => {
                match row_uf.get_binding(*id) {
                    Some(row) => row.resolve(row_uf),  // Recurse into binding
                    None => Row::Var(row_uf.find(*id)),  // Return canonical var
                }
            }
            Row::Generic(id) => Row::Generic(*id),
            Row::Extend { effect, rest } => Row::Extend {
                effect: effect.clone(),
                rest: Rc::new(rest.resolve(row_uf)),
            },
            Row::Empty => Row::Empty,
        }
    }

    /// Check if this row contains a given row variable (occurs check for row vars)
    pub fn occurs(&self, id: TypeVarId, row_uf: &RowUnionFind) -> bool {
        match self.resolve(row_uf) {
            Row::Empty => false,
            Row::Var(vid) => row_uf.find(vid) == id,
            Row::Generic(vid) => vid == id,
            Row::Extend { rest, .. } => rest.occurs(id, row_uf),
        }
    }

    /// Check if a type variable occurs in this row (including in effect parameters)
    pub fn type_var_occurs(&self, id: TypeVarId, uf: &UnionFind, row_uf: &RowUnionFind) -> bool {
        match self.resolve(row_uf) {
            Row::Empty => false,
            Row::Var(_) | Row::Generic(_) => false, // Row vars are separate from type vars
            Row::Extend { effect, rest } => {
                effect.params.iter().any(|t| t.occurs(id, uf)) || rest.type_var_occurs(id, uf, row_uf)
            }
        }
    }

    /// Check if this row is empty (pure)
    pub fn is_empty(&self, row_uf: &RowUnionFind) -> bool {
        matches!(self.resolve(row_uf), Row::Empty)
    }

    /// Collect validation issues for this effect row.
    pub fn collect_validation_issues(&self, issues: &mut Vec<String>) {
        match self {
            Row::Empty => {}
            Row::Generic(_) => {
                // Generic is OK - it's a resolved polymorphic row variable
            }
            Row::Var(id) => {
                issues.push(format!("Unresolved row variable: r{}", id));
            }
            Row::Extend { effect, rest } => {
                // Check type params in the effect
                for param in &effect.params {
                    param.collect_validation_issues(issues);
                }
                rest.collect_validation_issues(issues);
            }
        }
    }

    /// Check if this row contains a specific effect
    pub fn contains_effect(&self, effect_name: &str, row_uf: &RowUnionFind) -> bool {
        match self.resolve(row_uf) {
            Row::Empty => false,
            Row::Var(_) | Row::Generic(_) => false, // Unknown - could contain anything
            Row::Extend { effect, rest } => {
                effect.name == effect_name || rest.contains_effect(effect_name, row_uf)
            }
        }
    }

    /// Display the row for user-friendly output (without union-find - for display only)
    /// Note: This shows the structure as-is without resolving bindings.
    /// For resolved display, use display_user_friendly_resolved with the RowUnionFind.
    pub fn display_user_friendly(&self) -> String {
        let mut var_map: HashMap<TypeVarId, char> = HashMap::new();
        let mut next_var = 'r'; // Start row vars at 'r'
        self.display_with_map(&mut var_map, &mut next_var)
    }

    fn display_with_map(
        &self,
        var_map: &mut HashMap<TypeVarId, char>,
        next_var: &mut char,
    ) -> String {
        match self {
            Row::Empty => String::new(),
            Row::Var(id) | Row::Generic(id) => {
                let c = *var_map.entry(*id).or_insert_with(|| {
                    let c = *next_var;
                    *next_var = (*next_var as u8 + 1) as char;
                    c
                });
                c.to_string()
            }
            Row::Extend { effect, rest } => {
                let rest_str = rest.display_with_map(var_map, next_var);
                let effect_str = effect.display_user_friendly();
                if rest_str.is_empty() {
                    effect_str
                } else if rest_str.len() == 1 && rest_str.chars().next().unwrap().is_alphabetic() {
                    // Row variable at end
                    format!("{} | {}", effect_str, rest_str)
                } else {
                    format!("{}, {}", effect_str, rest_str)
                }
            }
        }
    }
}

impl Effect {
    /// Create a simple effect with no type parameters
    pub fn simple(name: impl Into<String>) -> Effect {
        Effect {
            name: name.into(),
            params: vec![],
        }
    }

    /// Create an effect with type parameters
    pub fn with_params(name: impl Into<String>, params: Vec<Type>) -> Effect {
        Effect {
            name: name.into(),
            params,
        }
    }

    /// Display the effect for user-friendly output
    pub fn display_user_friendly(&self) -> String {
        if self.params.is_empty() {
            self.name.clone()
        } else {
            let params_str: Vec<String> = self.params.iter()
                .map(|t| t.display_user_friendly())
                .collect();
            format!("{} {}", self.name, params_str.join(" "))
        }
    }
}

impl EffectEnv {
    /// Create a new empty effect environment
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an effect declaration
    pub fn register_effect(&mut self, info: EffectInfo) {
        let effect_name = info.name.clone();

        // Register each operation
        for op in &info.operations {
            self.operations.insert(
                op.name.clone(),
                (effect_name.clone(), op.clone()),
            );
        }

        self.effects.insert(effect_name, info);
    }

    /// Look up an effect by name
    pub fn get_effect(&self, name: &str) -> Option<&EffectInfo> {
        self.effects.get(name)
    }

    /// Look up an operation by name, returns (effect_name, operation_info)
    pub fn get_operation(&self, name: &str) -> Option<&(String, OperationInfo)> {
        self.operations.get(name)
    }
}

impl Type {
    /// Create a new generic type variable
    pub fn new_generic(id: TypeVarId) -> Type {
        Type::Generic(id)
    }

    /// Create a new unbound type variable (for tests that don't use union-find)
    /// Note: In actual inference, use UnionFind::fresh() instead
    pub fn new_var(id: TypeVarId, _level: u32) -> Type {
        Type::Var(id)
    }

    /// Check if a type variable syntactically occurs in this type
    /// This doesn't follow union-find bindings - use `occurs()` for that
    pub fn occurs_syntactic(&self, id: TypeVarId) -> bool {
        match self {
            Type::Var(v) => *v == id,
            Type::Generic(v) => *v == id,
            Type::Arrow { arg, ret, effects } => {
                arg.occurs_syntactic(id) || ret.occurs_syntactic(id) || effects.occurs_syntactic_in_row(id)
            }
            Type::Tuple(ts) => ts.iter().any(|t| t.occurs_syntactic(id)),
            Type::Constructor { args, .. } => args.iter().any(|t| t.occurs_syntactic(id)),
            Type::Channel(t) | Type::Fiber(t) | Type::Dict(t) => t.occurs_syntactic(id),
            _ => false,
        }
    }

    /// Create a list type: List elem
    pub fn list(elem: Type) -> Type {
        Type::Constructor {
            name: "List".to_string(),
            args: vec![elem],
        }
    }

    /// Create an option type: Option elem
    pub fn option(elem: Type) -> Type {
        Type::Constructor {
            name: "Option".to_string(),
            args: vec![elem],
        }
    }

    /// Follow all links to get the actual type.
    /// Uses the UnionFind for bindings with path compression.
    /// NOTE: This does NOT resolve effect rows in Arrow types. Use resolve_full for that.
    pub fn resolve(&self, uf: &UnionFind) -> Type {
        match self {
            Type::Var(id) => {
                match uf.get_binding(*id) {
                    Some(ty) => ty.resolve(uf),  // Recurse into binding
                    None => Type::Var(uf.find(*id)),  // Return canonical var
                }
            }
            Type::Generic(id) => Type::Generic(*id),
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(arg.resolve(uf)),
                ret: Rc::new(ret.resolve(uf)),
                effects: effects.clone(), // Note: row resolution needs RowUnionFind
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

    /// Fully resolve a type, including effect rows inside Arrow types.
    /// This should be used when storing types for elaboration/codegen.
    pub fn resolve_full(&self, type_uf: &UnionFind, row_uf: &RowUnionFind) -> Type {
        match self {
            Type::Var(id) => {
                match type_uf.get_binding(*id) {
                    Some(ty) => ty.resolve_full(type_uf, row_uf),
                    None => Type::Var(type_uf.find(*id)),
                }
            }
            Type::Generic(id) => Type::Generic(*id),
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(arg.resolve_full(type_uf, row_uf)),
                ret: Rc::new(ret.resolve_full(type_uf, row_uf)),
                effects: effects.resolve(row_uf), // Resolve effect rows!
            },
            Type::Tuple(ts) => Type::Tuple(
                ts.iter().map(|t| t.resolve_full(type_uf, row_uf)).collect()
            ),
            Type::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args.iter().map(|t| t.resolve_full(type_uf, row_uf)).collect(),
            },
            Type::Channel(inner) => Type::Channel(Rc::new(inner.resolve_full(type_uf, row_uf))),
            Type::Fiber(inner) => Type::Fiber(Rc::new(inner.resolve_full(type_uf, row_uf))),
            Type::Dict(inner) => Type::Dict(Rc::new(inner.resolve_full(type_uf, row_uf))),
            // Primitives are already resolved
            _ => self.clone(),
        }
    }

    /// Check if this type is fully resolved (no unbound type variables, resolved effect rows).
    /// Returns a list of issues found, empty if fully resolved.
    pub fn validation_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        self.collect_validation_issues(&mut issues);
        issues
    }

    fn collect_validation_issues(&self, issues: &mut Vec<String>) {
        match self {
            Type::Var(id) => {
                issues.push(format!("Unresolved type variable: t{}", id));
            }
            Type::Generic(_) => {
                // Generic is OK - it's a resolved polymorphic type variable
            }
            Type::Arrow { arg, ret, effects } => {
                arg.collect_validation_issues(issues);
                ret.collect_validation_issues(issues);
                effects.collect_validation_issues(issues);
            }
            Type::Tuple(ts) => {
                for t in ts {
                    t.collect_validation_issues(issues);
                }
            }
            Type::Constructor { args, .. } => {
                for t in args {
                    t.collect_validation_issues(issues);
                }
            }
            Type::Channel(inner) | Type::Fiber(inner) | Type::Dict(inner) => {
                inner.collect_validation_issues(issues);
            }
            // Primitives are always resolved
            _ => {}
        }
    }

    /// Check if this type contains a given type variable (occurs check)
    pub fn occurs(&self, id: TypeVarId, uf: &UnionFind) -> bool {
        match self.resolve(uf) {
            Type::Var(vid) => uf.find(vid) == id,
            Type::Generic(vid) => vid == id,
            Type::Arrow { arg, ret, .. } => {
                // Note: we check arg and ret but not effects here
                // Effects are checked separately with RowUnionFind
                arg.occurs(id, uf) || ret.occurs(id, uf)
            }
            Type::Tuple(types) => types.iter().any(|t| t.occurs(id, uf)),
            Type::Constructor { args, .. } => args.iter().any(|t| t.occurs(id, uf)),
            Type::Channel(t) => t.occurs(id, uf),
            Type::Fiber(t) => t.occurs(id, uf),
            Type::Dict(t) => t.occurs(id, uf),
            Type::Int
            | Type::Float
            | Type::Bool
            | Type::String
            | Type::Char
            | Type::Unit
            | Type::Bytes
            | Type::FileHandle
            | Type::TcpSocket
            | Type::TcpListener
            | Type::Pid
            | Type::Set => false,
        }
    }

    /// Check if this type contains a given type variable, also checking effect rows
    pub fn occurs_with_effects(&self, id: TypeVarId, uf: &UnionFind, row_uf: &RowUnionFind) -> bool {
        match self.resolve(uf) {
            Type::Var(vid) => uf.find(vid) == id,
            Type::Generic(vid) => vid == id,
            Type::Arrow { arg, ret, effects } => {
                arg.occurs_with_effects(id, uf, row_uf)
                    || ret.occurs_with_effects(id, uf, row_uf)
                    || effects.type_var_occurs(id, uf, row_uf)
            }
            Type::Tuple(types) => types.iter().any(|t| t.occurs_with_effects(id, uf, row_uf)),
            Type::Constructor { args, .. } => args.iter().any(|t| t.occurs_with_effects(id, uf, row_uf)),
            Type::Channel(t) => t.occurs_with_effects(id, uf, row_uf),
            Type::Fiber(t) => t.occurs_with_effects(id, uf, row_uf),
            Type::Dict(t) => t.occurs_with_effects(id, uf, row_uf),
            Type::Int
            | Type::Float
            | Type::Bool
            | Type::String
            | Type::Char
            | Type::Unit
            | Type::Bytes
            | Type::FileHandle
            | Type::TcpSocket
            | Type::TcpListener
            | Type::Pid
            | Type::Set => false,
        }
    }

    /// Create a pure function type (empty effect row)
    pub fn arrow(from: Type, to: Type) -> Type {
        Type::Arrow {
            arg: Rc::new(from),
            ret: Rc::new(to),
            effects: Row::Empty,
        }
    }

    /// Create a function type with a specific effect row
    pub fn arrow_with_effects(from: Type, to: Type, effects: Row) -> Type {
        Type::Arrow {
            arg: Rc::new(from),
            ret: Rc::new(to),
            effects,
        }
    }

    /// Create a function type with a polymorphic effect row variable
    /// This is used for functions that can have any effects (polymorphic in their effect row)
    /// The row_var_id should come from RowUnionFind::fresh()
    pub fn arrow_with_effect_var(from: Type, to: Type, row_var_id: TypeVarId) -> Type {
        Type::Arrow {
            arg: Rc::new(from),
            ret: Rc::new(to),
            effects: Row::Var(row_var_id),
        }
    }

    /// DEPRECATED: Kept for backward compatibility during migration
    /// Creates a pure function type - same as arrow()
    pub fn arrow_with_ans(from: Type, to: Type, _ans: Type) -> Type {
        Type::arrow(from, to)
    }

    /// DEPRECATED: Kept for backward compatibility during migration
    /// Creates a pure function type - same as arrow()
    pub fn effectful_arrow(from: Type, to: Type, _ans_in: Type, _ans_out: Type) -> Type {
        Type::arrow(from, to)
    }

    /// Create a multi-argument function type: a -> b -> c -> d
    /// All intermediate arrows are pure (same answer type throughout)
    pub fn arrows(args: Vec<Type>, ret: Type) -> Type {
        args.into_iter()
            .rev()
            .fold(ret, |acc, arg| Type::arrow(arg, acc))
    }

    // ========================================================================
    // User-Friendly Type Display
    // ========================================================================

    /// Display the type with normalized variable names (a, b, c instead of t732).
    /// Note: This shows the structure as-is without resolving bindings.
    /// For resolved display, resolve the type first with a UnionFind.
    pub fn display_user_friendly(&self) -> String {
        let mut var_map: HashMap<TypeVarId, char> = HashMap::new();
        let mut next_var = 'a';
        self.display_with_map(&mut var_map, &mut next_var, true)
    }

    /// Display with normalized variables but showing answer types (for debugging)
    pub fn display_normalized(&self) -> String {
        let mut var_map: HashMap<TypeVarId, char> = HashMap::new();
        let mut next_var = 'a';
        self.display_with_map(&mut var_map, &mut next_var, false)
    }

    /// Internal helper for display with variable normalization
    fn display_with_map(
        &self,
        var_map: &mut HashMap<TypeVarId, char>,
        next_var: &mut char,
        hide_answer_types: bool,
    ) -> String {
        match self {
            Type::Var(id) => {
                let c = *var_map.entry(*id).or_insert_with(|| {
                    let c = *next_var;
                    *next_var = if *next_var == 'z' {
                        'a'
                    } else {
                        (*next_var as u8 + 1) as char
                    };
                    c
                });
                c.to_string()
            }
            Type::Generic(id) => {
                let c = *var_map.entry(*id).or_insert_with(|| {
                    let c = *next_var;
                    *next_var = if *next_var == 'z' {
                        'a'
                    } else {
                        (*next_var as u8 + 1) as char
                    };
                    c
                });
                c.to_string()
            }
            Type::Int => "Int".to_string(),
            Type::Float => "Float".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::String => "String".to_string(),
            Type::Char => "Char".to_string(),
            Type::Unit => "()".to_string(),
            Type::Bytes => "Bytes".to_string(),
            Type::FileHandle => "FileHandle".to_string(),
            Type::TcpSocket => "TcpSocket".to_string(),
            Type::TcpListener => "TcpListener".to_string(),
            Type::Pid => "Pid".to_string(),
            Type::Arrow { arg, ret, effects } => {
                let arg_str = match arg.as_ref() {
                    Type::Arrow { .. } => format!(
                        "({})",
                        arg.display_with_map(var_map, next_var, hide_answer_types)
                    ),
                    _ => arg.display_with_map(var_map, next_var, hide_answer_types),
                };
                let ret_str = ret.display_with_map(var_map, next_var, hide_answer_types);

                // Check if pure (empty effects) or polymorphic
                match effects {
                    Row::Empty => {
                        // Pure function: a -> b
                        format!("{} -> {}", arg_str, ret_str)
                    }
                    Row::Var(_) | Row::Generic(_) if hide_answer_types => {
                        // Polymorphic effects, hide for user-friendly display
                        format!("{} -> {}", arg_str, ret_str)
                    }
                    _ => {
                        // Show effects: a -> b { eff1, eff2 | r }
                        let effects_str = effects.display_user_friendly();
                        if effects_str.is_empty() {
                            format!("{} -> {}", arg_str, ret_str)
                        } else {
                            format!("{} -> {} {{ {} }}", arg_str, ret_str, effects_str)
                        }
                    }
                }
            }
            Type::Tuple(types) => {
                let parts: Vec<String> = types
                    .iter()
                    .map(|t| t.display_with_map(var_map, next_var, hide_answer_types))
                    .collect();
                format!("({})", parts.join(", "))
            }
            Type::Constructor { name, args } => {
                // Special case: display List as [a] instead of List a
                if name == "List" && args.len() == 1 {
                    return format!(
                        "[{}]",
                        args[0].display_with_map(var_map, next_var, hide_answer_types)
                    );
                }
                if args.is_empty() {
                    name.clone()
                } else {
                    let arg_strs: Vec<String> = args
                        .iter()
                        .map(|a| {
                            // Wrap complex types in parens
                            let s = a.display_with_map(var_map, next_var, hide_answer_types);
                            let needs_parens = match a {
                                Type::Arrow { .. } => true,
                                Type::Constructor { args: inner_args, .. } => !inner_args.is_empty(),
                                _ => false,
                            };
                            if needs_parens {
                                format!("({})", s)
                            } else {
                                s
                            }
                        })
                        .collect();
                    format!("{} {}", name, arg_strs.join(" "))
                }
            }
            Type::Channel(t) => {
                format!(
                    "Channel {}",
                    t.display_with_map(var_map, next_var, hide_answer_types)
                )
            }
            Type::Fiber(t) => {
                format!(
                    "Fiber {}",
                    t.display_with_map(var_map, next_var, hide_answer_types)
                )
            }
            Type::Dict(t) => {
                format!(
                    "Dict {}",
                    t.display_with_map(var_map, next_var, hide_answer_types)
                )
            }
            Type::Set => "Set".to_string(),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Note: Display shows the raw type without resolution.
        // For resolved display, use resolve() with a UnionFind first.
        match self {
            Type::Var(id) => write!(f, "t{}", id),
            Type::Generic(id) => write!(f, "{}", (b'a' + (*id % 26) as u8) as char),
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::String => write!(f, "String"),
            Type::Char => write!(f, "Char"),
            Type::Unit => write!(f, "()"),
            Type::Bytes => write!(f, "Bytes"),
            Type::FileHandle => write!(f, "FileHandle"),
            Type::TcpSocket => write!(f, "TcpSocket"),
            Type::TcpListener => write!(f, "TcpListener"),
            Type::Pid => write!(f, "Pid"),
            Type::Arrow { arg, ret, effects } => {
                let arg_str = match arg.as_ref() {
                    Type::Arrow { .. } => format!("({})", arg),
                    _ => format!("{}", arg),
                };
                // Check if pure (empty effects) or has effects
                match effects {
                    Row::Empty => {
                        // Pure function: show as σ → τ
                        write!(f, "{} -> {}", arg_str, ret)
                    }
                    Row::Var(_) | Row::Generic(_) => {
                        // Polymorphic effects - show as pure for simplicity
                        write!(f, "{} -> {}", arg_str, ret)
                    }
                    _ => {
                        // Has effects: show as σ → τ { effects }
                        let effects_str = effects.display_user_friendly();
                        write!(f, "{} -> {} {{ {} }}", arg_str, ret, effects_str)
                    }
                }
            }
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            Type::Constructor { name, args } => {
                // Special case: display List as [a] instead of List a
                if name == "List" && args.len() == 1 {
                    return write!(f, "[{}]", args[0]);
                }
                write!(f, "{}", name)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                Ok(())
            }
            Type::Channel(t) => write!(f, "Channel {}", t),
            Type::Fiber(t) => write!(f, "Fiber {}", t),
            Type::Dict(t) => write!(f, "Dict {}", t),
            Type::Set => write!(f, "Set"),
        }
    }
}

/// A polymorphic type scheme: forall a b. a -> b -> a
/// With optional type class constraints: forall a. Show a => a -> String
#[derive(Debug, Clone)]
pub struct Scheme {
    /// The number of generic type variables
    pub num_generics: u32,
    /// Type class predicates/constraints on the generic variables
    pub predicates: Vec<Pred>,
    /// The underlying type
    pub ty: Type,
}

impl Scheme {
    /// A monomorphic type (no generics, no constraints)
    pub fn mono(ty: Type) -> Scheme {
        Scheme {
            num_generics: 0,
            predicates: vec![],
            ty,
        }
    }

    /// A polymorphic type with constraints
    pub fn with_predicates(num_generics: u32, predicates: Vec<Pred>, ty: Type) -> Scheme {
        Scheme {
            num_generics,
            predicates,
            ty,
        }
    }
}

impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.num_generics > 0 {
            write!(f, "forall ")?;
            for i in 0..self.num_generics {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", (b'a' + (i % 26) as u8) as char)?;
            }
            write!(f, ". ")?;
        }
        // Display type class constraints if present
        if !self.predicates.is_empty() {
            if self.predicates.len() == 1 {
                write!(f, "{} => ", self.predicates[0])?;
            } else {
                write!(f, "(")?;
                for (i, pred) in self.predicates.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", pred)?;
                }
                write!(f, ") => ")?;
            }
        }
        write!(f, "{}", self.ty)
    }
}

// ============================================================================
// Answer-Type Polymorphism for Delimited Continuations
// ============================================================================

/// Result of type inference for an expression.
///
/// The judgment `Γ ⊢ e : τ ! ε` (Koka-style) is represented as:
/// - `ty` = τ (the expression's type)
/// - `effects` = ε (the effects the expression may perform)
///
/// Pure expressions have `effects == Row::Empty`.
#[derive(Debug, Clone)]
pub struct InferResult {
    /// The expression's type (τ in the judgment)
    pub ty: Type,
    /// The effects this expression may perform (ε in the judgment)
    pub effects: Row,
}

impl InferResult {
    /// Create result for a pure expression (no effects).
    pub fn pure(ty: Type) -> Self {
        InferResult {
            ty,
            effects: Row::Empty,
        }
    }

    /// Create result with a specific effect row.
    pub fn with_effects(ty: Type, effects: Row) -> Self {
        InferResult { ty, effects }
    }

    /// Check if this result represents a pure expression (no effects)
    pub fn is_pure(&self, row_uf: &RowUnionFind) -> bool {
        matches!(self.effects.resolve(row_uf), Row::Empty)
    }
}

/// Type environment: maps identifiers to their type schemes
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    bindings: HashMap<String, Scheme>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: String, scheme: Scheme) {
        self.bindings.insert(name, scheme);
    }

    pub fn get(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name)
    }

    pub fn extend(&self, name: String, scheme: Scheme) -> TypeEnv {
        let mut new_env = self.clone();
        new_env.insert(name, scheme);
        new_env
    }

    /// Get an iterator over all bound names
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.bindings.keys()
    }

    /// Get an iterator over all bindings (name, scheme pairs)
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Scheme)> {
        self.bindings.iter()
    }
}

/// Information about a data type constructor
#[derive(Debug, Clone)]
pub struct ConstructorInfo {
    /// The type this constructor belongs to (e.g., "Option")
    pub type_name: String,
    /// Number of type parameters the type takes
    pub type_params: u32,
    /// Types of the fields (may contain generic type vars)
    pub field_types: Vec<Type>,
}

/// Information about a record type
#[derive(Debug, Clone)]
pub struct RecordInfo {
    /// The type name (e.g., "Request")
    pub type_name: String,
    /// Number of type parameters the type takes
    pub type_params: u32,
    /// Field names in declaration order
    pub field_names: Vec<String>,
    /// Field types indexed by name (may contain generic type vars)
    pub field_types: HashMap<String, Type>,
}

/// Information about a type alias
#[derive(Debug, Clone)]
pub struct TypeAliasInfo {
    /// The alias name (e.g., "Thing")
    pub name: String,
    /// Type parameter names (e.g., ["a", "b"] for `type Pair a b = (a, b)`)
    pub params: Vec<String>,
    /// The type this alias expands to (may contain generic type vars)
    pub body: Type,
}

/// Stores information about declared data types
#[derive(Debug, Clone, Default)]
pub struct TypeContext {
    /// Maps constructor names to their info
    pub constructors: HashMap<String, ConstructorInfo>,
    /// Maps record type names to their info
    pub records: HashMap<String, RecordInfo>,
    /// Maps type alias names to their info
    pub type_aliases: HashMap<String, TypeAliasInfo>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_constructor(&mut self, name: String, info: ConstructorInfo) {
        self.constructors.insert(name, info);
    }

    pub fn get_constructor(&self, name: &str) -> Option<&ConstructorInfo> {
        self.constructors.get(name)
    }

    /// Get an iterator over all constructor names
    pub fn constructor_names(&self) -> impl Iterator<Item = &str> {
        self.constructors.keys().map(|s| s.as_str())
    }

    pub fn add_record(&mut self, name: String, info: RecordInfo) {
        self.records.insert(name, info);
    }

    pub fn get_record(&self, name: &str) -> Option<&RecordInfo> {
        self.records.get(name)
    }

    /// Get an iterator over all record type names
    pub fn record_names(&self) -> impl Iterator<Item = &str> {
        self.records.keys().map(|s| s.as_str())
    }

    pub fn add_type_alias(&mut self, name: String, info: TypeAliasInfo) {
        self.type_aliases.insert(name, info);
    }

    pub fn get_type_alias(&self, name: &str) -> Option<&TypeAliasInfo> {
        self.type_aliases.get(name)
    }
}

// ============================================================================
// Typeclass support
// ============================================================================

/// A predicate: TraitName applied to a type (e.g., Show Int, Eq (List a))
#[derive(Debug, Clone)]
pub struct Pred {
    pub trait_name: String,
    pub ty: Type,
}

impl PartialEq for Pred {
    fn eq(&self, other: &Self) -> bool {
        // Predicates should only contain Generic types (from schemes), not Var
        self.trait_name == other.trait_name && types_equal_in_scheme(&self.ty, &other.ty)
    }
}

/// Check if two types are structurally equal in a scheme context.
/// This works without union-find because scheme types use Generic, not Var.
fn types_equal_in_scheme(t1: &Type, t2: &Type) -> bool {
    match (t1, t2) {
        (Type::Int, Type::Int) => true,
        (Type::Float, Type::Float) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::String, Type::String) => true,
        (Type::Char, Type::Char) => true,
        (Type::Unit, Type::Unit) => true,
        (Type::Pid, Type::Pid) => true,
        (Type::Generic(id1), Type::Generic(id2)) => id1 == id2,
        (Type::Var(id1), Type::Var(id2)) => id1 == id2, // Same var id
        (
            Type::Arrow {
                arg: a1,
                ret: r1,
                effects: e1,
            },
            Type::Arrow {
                arg: a2,
                ret: r2,
                effects: e2,
            },
        ) => types_equal_in_scheme(a1, a2) && types_equal_in_scheme(r1, r2) && rows_equal_in_scheme(e1, e2),
        (Type::Tuple(ts1), Type::Tuple(ts2)) => {
            ts1.len() == ts2.len() && ts1.iter().zip(ts2).all(|(x, y)| types_equal_in_scheme(x, y))
        }
        (Type::Constructor { name: n1, args: a1 }, Type::Constructor { name: n2, args: a2 }) => {
            n1 == n2 && a1.len() == a2.len() && a1.iter().zip(a2).all(|(x, y)| types_equal_in_scheme(x, y))
        }
        (Type::Channel(e1), Type::Channel(e2)) => types_equal_in_scheme(e1, e2),
        (Type::Fiber(e1), Type::Fiber(e2)) => types_equal_in_scheme(e1, e2),
        (Type::Dict(e1), Type::Dict(e2)) => types_equal_in_scheme(e1, e2),
        _ => false,
    }
}

fn rows_equal_in_scheme(r1: &Row, r2: &Row) -> bool {
    match (r1, r2) {
        (Row::Empty, Row::Empty) => true,
        (Row::Var(id1), Row::Var(id2)) => id1 == id2,
        (Row::Generic(id1), Row::Generic(id2)) => id1 == id2,
        (
            Row::Extend {
                effect: e1,
                rest: rest1,
            },
            Row::Extend {
                effect: e2,
                rest: rest2,
            },
        ) => {
            e1.name == e2.name
                && e1.params.iter().zip(&e2.params).all(|(t1, t2)| types_equal_in_scheme(t1, t2))
                && rows_equal_in_scheme(rest1, rest2)
        }
        _ => false,
    }
}

/// Check if two types are structurally equal (resolving links)
/// For types with Var, checks if they have the same root in union-find
/// For Generic types (in schemes), compares ids directly
pub fn types_equal(t1: &Type, t2: &Type, uf: &UnionFind, row_uf: &RowUnionFind) -> bool {
    let t1 = t1.resolve(uf);
    let t2 = t2.resolve(uf);

    match (&t1, &t2) {
        (Type::Int, Type::Int) => true,
        (Type::Float, Type::Float) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::String, Type::String) => true,
        (Type::Char, Type::Char) => true,
        (Type::Unit, Type::Unit) => true,
        (Type::Pid, Type::Pid) => true,
        // Two unbound vars are equal if they have the same root
        (Type::Var(id1), Type::Var(id2)) => uf.find(*id1) == uf.find(*id2),
        // Two generic vars are equal if they have the same id
        (Type::Generic(id1), Type::Generic(id2)) => id1 == id2,
        (
            Type::Arrow {
                arg: a1,
                ret: r1,
                effects: e1,
            },
            Type::Arrow {
                arg: a2,
                ret: r2,
                effects: e2,
            },
        ) => types_equal(a1, a2, uf, row_uf) && types_equal(r1, r2, uf, row_uf) && rows_equal(e1, e2, uf, row_uf),
        (Type::Tuple(ts1), Type::Tuple(ts2)) => {
            ts1.len() == ts2.len() && ts1.iter().zip(ts2).all(|(x, y)| types_equal(x, y, uf, row_uf))
        }
        (Type::Constructor { name: n1, args: a1 }, Type::Constructor { name: n2, args: a2 }) => {
            n1 == n2 && a1.len() == a2.len() && a1.iter().zip(a2).all(|(x, y)| types_equal(x, y, uf, row_uf))
        }
        (Type::Channel(e1), Type::Channel(e2)) => types_equal(e1, e2, uf, row_uf),
        _ => false,
    }
}

/// Check if two effect rows are structurally equal (resolving links)
pub fn rows_equal(r1: &Row, r2: &Row, uf: &UnionFind, row_uf: &RowUnionFind) -> bool {
    let r1 = r1.resolve(row_uf);
    let r2 = r2.resolve(row_uf);

    match (&r1, &r2) {
        (Row::Empty, Row::Empty) => true,
        // Two unbound row vars are equal if they have the same root
        (Row::Var(id1), Row::Var(id2)) => row_uf.find(*id1) == row_uf.find(*id2),
        // Two generic row vars are equal if they have the same id
        (Row::Generic(id1), Row::Generic(id2)) => id1 == id2,
        (
            Row::Extend {
                effect: e1,
                rest: rest1,
            },
            Row::Extend {
                effect: e2,
                rest: rest2,
            },
        ) => {
            e1.name == e2.name
                && e1
                    .params
                    .iter()
                    .zip(&e2.params)
                    .all(|(t1, t2)| types_equal(t1, t2, uf, row_uf))
                && rows_equal(rest1, rest2, uf, row_uf)
        }
        _ => false,
    }
}

impl Pred {
    pub fn new(trait_name: impl Into<String>, ty: Type) -> Self {
        Self {
            trait_name: trait_name.into(),
            ty,
        }
    }

    /// Apply a type substitution to this predicate
    pub fn apply(&self, subst: &HashMap<TypeVarId, Type>) -> Pred {
        Pred {
            trait_name: self.trait_name.clone(),
            ty: apply_subst(&self.ty, subst),
        }
    }
}

impl fmt::Display for Pred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.trait_name, self.ty)
    }
}

/// Apply a substitution to a type (replacing generic vars with concrete types)
/// This works on scheme types (with Generic, not Var) so doesn't need union-find
pub fn apply_subst(ty: &Type, subst: &HashMap<TypeVarId, Type>) -> Type {
    match ty {
        Type::Generic(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Type::Var(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Type::Arrow { arg, ret, effects } => Type::Arrow {
            arg: Rc::new(apply_subst(arg, subst)),
            ret: Rc::new(apply_subst(ret, subst)),
            effects: apply_subst_to_row(effects, subst),
        },
        Type::Tuple(types) => Type::Tuple(types.iter().map(|t| apply_subst(t, subst)).collect()),
        Type::Constructor { name, args } => Type::Constructor {
            name: name.clone(),
            args: args.iter().map(|t| apply_subst(t, subst)).collect(),
        },
        Type::Channel(t) => Type::Channel(Rc::new(apply_subst(t, subst))),
        Type::Fiber(t) => Type::Fiber(Rc::new(apply_subst(t, subst))),
        Type::Dict(t) => Type::Dict(Rc::new(apply_subst(t, subst))),
        // Primitives unchanged
        t => t.clone(),
    }
}

/// Apply a substitution to an effect row (for type parameters in effects)
/// This works on scheme types so doesn't need union-find
pub fn apply_subst_to_row(row: &Row, subst: &HashMap<TypeVarId, Type>) -> Row {
    match row {
        Row::Empty => Row::Empty,
        Row::Var(id) => Row::Var(*id), // Row variables are not substituted by type subst
        Row::Generic(id) => Row::Generic(*id), // Generic row vars not substituted by type subst
        Row::Extend { effect, rest } => Row::Extend {
            effect: Effect {
                name: effect.name.clone(),
                params: effect.params.iter().map(|t| apply_subst(t, subst)).collect(),
            },
            rest: Rc::new(apply_subst_to_row(rest, subst)),
        },
    }
}

/// Information about a trait declaration
#[derive(Debug, Clone)]
pub struct TraitInfo {
    pub name: String,
    pub type_param: String,
    pub supertraits: Vec<String>,
    /// Method signatures: name -> type (with type_param as Generic(0))
    pub methods: HashMap<String, Type>,
}

/// Information about an instance declaration
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub trait_name: String,
    /// The instance head type (e.g., Int, List a)
    pub head: Type,
    /// Constraints required for this instance (e.g., [Show a] for Show (List a))
    pub constraints: Vec<Pred>,
    /// Method implementations (stored as AST for later evaluation)
    pub method_impls: Vec<crate::ast::InstanceMethod>,
}

/// Error when adding instances
#[derive(Debug, Clone)]
pub enum ClassError {
    /// The trait doesn't exist
    UnknownTrait(String),
    /// Overlapping instances
    OverlappingInstance {
        trait_name: String,
        existing: Type,
        new: Type,
    },
}

impl fmt::Display for ClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClassError::UnknownTrait(name) => write!(f, "Unknown trait: {}", name),
            ClassError::OverlappingInstance {
                trait_name,
                existing,
                new,
            } => write!(
                f,
                "Overlapping instances for {}: {} and {}",
                trait_name, existing, new
            ),
        }
    }
}

/// The class environment: stores all trait and instance declarations
#[derive(Debug, Clone, Default)]
pub struct ClassEnv {
    pub traits: HashMap<String, TraitInfo>,
    pub instances: Vec<InstanceInfo>,
}

impl ClassEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a trait declaration
    pub fn add_trait(&mut self, info: TraitInfo) {
        self.traits.insert(info.name.clone(), info);
    }

    /// Get a trait by name
    pub fn get_trait(&self, name: &str) -> Option<&TraitInfo> {
        self.traits.get(name)
    }

    /// Register an instance declaration
    /// Returns error if the trait doesn't exist or if there's an overlapping instance
    #[allow(clippy::result_large_err)] // ClassError needs to carry Type info for good error messages
    pub fn add_instance(&mut self, info: InstanceInfo) -> Result<(), ClassError> {
        // Check that the trait exists
        if !self.traits.contains_key(&info.trait_name) {
            return Err(ClassError::UnknownTrait(info.trait_name.clone()));
        }

        // Check for overlapping instances (simplified: just check same trait + unifiable heads)
        for existing in &self.instances {
            if existing.trait_name == info.trait_name
                && types_overlap(&existing.head, &info.head) {
                    return Err(ClassError::OverlappingInstance {
                        trait_name: info.trait_name.clone(),
                        existing: existing.head.clone(),
                        new: info.head.clone(),
                    });
                }
        }

        self.instances.push(info);
        Ok(())
    }

    /// Find instances for a given trait
    pub fn instances_for<'a>(
        &'a self,
        trait_name: &'a str,
    ) -> impl Iterator<Item = &'a InstanceInfo> + 'a {
        self.instances
            .iter()
            .filter(move |i| i.trait_name == trait_name)
    }

    /// Look up which trait provides a method
    pub fn lookup_method(&self, method_name: &str) -> Option<(&str, &Type)> {
        for (trait_name, trait_info) in &self.traits {
            if let Some(method_ty) = trait_info.methods.get(method_name) {
                return Some((trait_name.as_str(), method_ty));
            }
        }
        None
    }
}

/// Check if two types could potentially overlap (simplified unifiability check)
/// This is conservative: returns true if they MIGHT overlap
/// Works on scheme types (with Generic, not Var) so doesn't need union-find
fn types_overlap(t1: &Type, t2: &Type) -> bool {
    match (t1, t2) {
        // Type variables overlap with anything
        (Type::Var(_), _) | (_, Type::Var(_)) => true,
        (Type::Generic(_), _) | (_, Type::Generic(_)) => true,

        // Same primitive types overlap
        (Type::Int, Type::Int) => true,
        (Type::Float, Type::Float) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::String, Type::String) => true,
        (Type::Char, Type::Char) => true,
        (Type::Unit, Type::Unit) => true,
        (Type::Pid, Type::Pid) => true,

        // Different primitives don't overlap
        (Type::Int, _)
        | (Type::Float, _)
        | (Type::Bool, _)
        | (Type::String, _)
        | (Type::Char, _)
        | (Type::Unit, _)
        | (Type::Pid, _) => false,

        // Constructors with same name and arity might overlap
        (Type::Constructor { name: n1, args: a1 }, Type::Constructor { name: n2, args: a2 }) => {
            n1 == n2 && a1.len() == a2.len() && a1.iter().zip(a2).all(|(x, y)| types_overlap(x, y))
        }

        // Channels might overlap if their element types might overlap
        (Type::Channel(e1), Type::Channel(e2)) => types_overlap(e1, e2),

        // Fibers
        (Type::Fiber(e1), Type::Fiber(e2)) => types_overlap(e1, e2),

        // Dicts
        (Type::Dict(e1), Type::Dict(e2)) => types_overlap(e1, e2),

        // Tuples with same arity might overlap
        (Type::Tuple(t1), Type::Tuple(t2)) => {
            t1.len() == t2.len() && t1.iter().zip(t2).all(|(x, y)| types_overlap(x, y))
        }

        // Arrows might overlap (check arg, ret, and effects)
        (
            Type::Arrow {
                arg: a1,
                ret: r1,
                effects: e1,
            },
            Type::Arrow {
                arg: a2,
                ret: r2,
                effects: e2,
            },
        ) => types_overlap(a1, a2) && types_overlap(r1, r2) && rows_overlap(e1, e2),

        // Different type constructors don't overlap
        _ => false,
    }
}

/// Check if two effect rows might overlap (for instance checking)
/// Two rows overlap if they could unify
/// Works on scheme types so doesn't need union-find
pub fn rows_overlap(r1: &Row, r2: &Row) -> bool {
    match (r1, r2) {
        // Empty rows overlap only with empty or variables
        (Row::Empty, Row::Empty) => true,
        (Row::Empty, Row::Var(_)) | (Row::Var(_), Row::Empty) => true,
        (Row::Empty, Row::Generic(_)) | (Row::Generic(_), Row::Empty) => true,
        // Variables overlap with anything
        (Row::Var(_), _) | (_, Row::Var(_)) => true,
        (Row::Generic(_), _) | (_, Row::Generic(_)) => true,
        // Extend rows overlap if effects are compatible
        (
            Row::Extend {
                effect: e1,
                rest: rest1,
            },
            Row::Extend {
                effect: e2,
                rest: rest2,
            },
        ) => {
            // Effects with same name might overlap
            if e1.name == e2.name {
                e1.params
                    .iter()
                    .zip(&e2.params)
                    .all(|(t1, t2)| types_overlap(t1, t2))
                    && rows_overlap(rest1, rest2)
            } else {
                // Different effect names could still unify via row reordering
                true
            }
        }
        _ => false,
    }
}

// ============================================================================
// Instance Resolution
// ============================================================================

/// Type alias for substitution map
pub type Substitution = HashMap<TypeVarId, Type>;

/// Result of resolving a predicate to an instance
#[derive(Debug, Clone)]
pub struct Resolution {
    /// Index into ClassEnv.instances
    pub instance_idx: usize,
    /// Substitution from instance type params to concrete types
    pub subst: Substitution,
    /// Sub-predicates that need to be resolved (from constrained instances)
    pub sub_preds: Vec<Pred>,
}

/// One-way type matching: check if pattern matches target, producing substitution.
/// Unlike unification, this only substitutes variables in the pattern, not the target.
/// This works on scheme types (with Generic, not Var) so doesn't need union-find.
/// Returns None if no match is possible.
pub fn match_type(pattern: &Type, target: &Type) -> Option<Substitution> {
    let mut subst = Substitution::new();
    if match_type_inner(pattern, target, &mut subst) {
        Some(subst)
    } else {
        None
    }
}

fn match_type_inner(pattern: &Type, target: &Type, subst: &mut Substitution) -> bool {
    match (pattern, target) {
        // Generic variable in pattern matches anything
        (Type::Generic(id), _) => {
            if let Some(existing) = subst.get(id) {
                // Must match previously bound type
                types_equal_in_scheme(existing, target)
            } else {
                subst.insert(*id, target.clone());
                true
            }
        }
        // Var variable in pattern matches anything (shouldn't normally happen in schemes)
        (Type::Var(id), _) => {
            if let Some(existing) = subst.get(id) {
                types_equal_in_scheme(existing, target)
            } else {
                subst.insert(*id, target.clone());
                true
            }
        }

        // Same primitives match
        (Type::Int, Type::Int) => true,
        (Type::Float, Type::Float) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::String, Type::String) => true,
        (Type::Char, Type::Char) => true,
        (Type::Unit, Type::Unit) => true,
        (Type::Pid, Type::Pid) => true,

        // Constructors with same name (nominal matching - name is sufficient)
        (Type::Constructor { name: n1, args: a1 }, Type::Constructor { name: n2, args: a2 })
            if n1 == n2 =>
        {
            // Match available args positionally; missing args in target are OK
            // (e.g., pattern `Option a` matches target `Option` from nullary constructor)
            a1.iter()
                .zip(a2)
                .all(|(p, t)| match_type_inner(p, t, subst))
        }

        // Channels
        (Type::Channel(p), Type::Channel(t)) => match_type_inner(p, t, subst),

        // Fibers
        (Type::Fiber(p), Type::Fiber(t)) => match_type_inner(p, t, subst),

        // Dicts
        (Type::Dict(p), Type::Dict(t)) => match_type_inner(p, t, subst),

        // Tuples with same arity
        (Type::Tuple(ps), Type::Tuple(ts)) if ps.len() == ts.len() => ps
            .iter()
            .zip(ts)
            .all(|(p, t)| match_type_inner(p, t, subst)),

        // Arrows (match arg, ret, and effects)
        (
            Type::Arrow {
                arg: pa,
                ret: pr,
                effects: pe,
            },
            Type::Arrow {
                arg: ta,
                ret: tr,
                effects: te,
            },
        ) => {
            match_type_inner(pa, ta, subst)
                && match_type_inner(pr, tr, subst)
                && match_row_inner(pe, te, subst)
        }

        // No match
        _ => false,
    }
}

/// Helper for matching effect rows during instance resolution
fn match_row_inner(pattern: &Row, target: &Row, subst: &mut Substitution) -> bool {
    match (pattern, target) {
        (Row::Empty, Row::Empty) => true,
        // Row variables in pattern are like wildcards
        (Row::Var(_), _) | (Row::Generic(_), _) => true,
        (
            Row::Extend {
                effect: pe,
                rest: pr,
            },
            Row::Extend {
                effect: te,
                rest: tr,
            },
        ) => {
            pe.name == te.name
                && pe
                    .params
                    .iter()
                    .zip(&te.params)
                    .all(|(p, t)| match_type_inner(p, t, subst))
                && match_row_inner(pr, tr, subst)
        }
        _ => false,
    }
}

impl ClassEnv {
    /// Try to resolve a predicate to an instance.
    /// Returns the resolution with substitution and sub-predicates.
    pub fn resolve_pred(&self, pred: &Pred) -> Option<Resolution> {
        for (idx, inst) in self.instances.iter().enumerate() {
            if inst.trait_name != pred.trait_name {
                continue;
            }

            // Try to match the instance head against the predicate's type
            if let Some(subst) = match_type(&inst.head, &pred.ty) {
                // Apply substitution to instance constraints to get sub-predicates
                let sub_preds: Vec<Pred> =
                    inst.constraints.iter().map(|p| p.apply(&subst)).collect();

                return Some(Resolution {
                    instance_idx: idx,
                    subst,
                    sub_preds,
                });
            }
        }
        None
    }

    /// Resolve all predicates recursively.
    /// Returns resolutions for all predicates (including sub-predicates).
    pub fn resolve_all(&self, preds: &[Pred]) -> Result<Vec<Resolution>, String> {
        let mut all_resolutions = Vec::new();
        let mut pending: Vec<Pred> = preds.to_vec();
        let mut visited = std::collections::HashSet::<String>::new();

        while let Some(pred) = pending.pop() {
            // Create a key for cycle detection
            let key = format!("{} {}", pred.trait_name, pred.ty);
            if visited.contains(&key) {
                continue; // Already resolved this predicate
            }
            visited.insert(key);

            match self.resolve_pred(&pred) {
                Some(resolution) => {
                    // Add sub-predicates to pending queue
                    for sub_pred in &resolution.sub_preds {
                        pending.push(sub_pred.clone());
                    }
                    all_resolutions.push(resolution);
                }
                None => {
                    // Check if the type is still a variable - defer judgment
                    // Note: In predicates, use Var or Generic for polymorphic types
                    if matches!(pred.ty, Type::Var(_) | Type::Generic(_)) {
                        // This is a deferred constraint - not an error yet
                        continue;
                    }
                    return Err(format!(
                        "No instance of {} for type {}",
                        pred.trait_name, pred.ty
                    ));
                }
            }
        }

        Ok(all_resolutions)
    }

    /// Check if a set of given predicates entail a goal predicate.
    /// This is used for checking if a function's constraints satisfy its context.
    pub fn entail(&self, given: &[Pred], goal: &Pred) -> bool {
        // Simple case: goal is directly in given
        // Note: predicates use Generic types, so we use types_equal_in_scheme
        for g in given {
            if g.trait_name == goal.trait_name && types_equal_in_scheme(&g.ty, &goal.ty) {
                return true;
            }
        }

        // Try to resolve the goal via instances
        if let Some(resolution) = self.resolve_pred(goal) {
            // All sub-predicates must be entailed by given
            resolution
                .sub_preds
                .iter()
                .all(|sub| self.entail(given, sub))
        } else {
            false
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pred_equality() {
        let p1 = Pred::new("Show", Type::Int);
        let p2 = Pred::new("Show", Type::Int);
        let p3 = Pred::new("Eq", Type::Int);
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_pred_apply_substitution() {
        let mut subst = HashMap::new();
        subst.insert(0, Type::Int);
        let pred = Pred::new("Show", Type::new_generic(0));
        let applied = pred.apply(&subst);
        assert!(matches!(applied.ty, Type::Int));
    }

    #[test]
    fn test_class_env_register_trait() {
        let mut env = ClassEnv::new();
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });
        assert!(env.get_trait("Show").is_some());
        assert!(env.get_trait("Eq").is_none());
    }

    #[test]
    fn test_class_env_register_instance() {
        let mut env = ClassEnv::new();

        // First add the trait
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Then add instance
        let result = env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Int,
            constraints: vec![],
            method_impls: vec![],
        });
        assert!(result.is_ok());
        assert_eq!(env.instances.len(), 1);
    }

    #[test]
    fn test_class_env_unknown_trait_error() {
        let mut env = ClassEnv::new();
        let result = env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Int,
            constraints: vec![],
            method_impls: vec![],
        });
        assert!(matches!(result, Err(ClassError::UnknownTrait(_))));
    }

    #[test]
    fn test_overlap_detection() {
        let mut env = ClassEnv::new();

        // Add Show trait
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Add Show (List a) instance
        env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::new_generic(0)],
            },
            constraints: vec![Pred::new("Show", Type::new_generic(0))],
            method_impls: vec![],
        })
        .unwrap();

        // Try to add Show (List Int) - should fail due to overlap
        let result = env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::Int],
            },
            constraints: vec![],
            method_impls: vec![],
        });
        assert!(matches!(
            result,
            Err(ClassError::OverlappingInstance { .. })
        ));
    }

    #[test]
    fn test_lookup_method() {
        let mut env = ClassEnv::new();
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        let result = env.lookup_method("show");
        assert!(result.is_some());
        let (trait_name, _) = result.unwrap();
        assert_eq!(trait_name, "Show");

        assert!(env.lookup_method("unknown").is_none());
    }

    // ========================================================================
    // Phase 5: Instance Resolution Tests
    // ========================================================================

    #[test]
    fn test_match_type_exact() {
        // Int matches Int
        let subst = match_type(&Type::Int, &Type::Int);
        assert!(subst.is_some());
        assert!(subst.unwrap().is_empty());
    }

    #[test]
    fn test_match_type_different_primitives() {
        // Int doesn't match String
        let subst = match_type(&Type::Int, &Type::String);
        assert!(subst.is_none());
    }

    #[test]
    fn test_match_type_variable() {
        // 'a matches Int with substitution a -> Int
        let pattern = Type::new_generic(0);
        let subst = match_type(&pattern, &Type::Int);
        assert!(subst.is_some());
        let s = subst.unwrap();
        assert_eq!(s.len(), 1);
        assert!(matches!(s.get(&0).unwrap(), Type::Int));
    }

    #[test]
    fn test_match_type_constructor() {
        // List 'a matches List Int
        let pattern = Type::Constructor {
            name: "List".to_string(),
            args: vec![Type::new_generic(0)],
        };
        let target = Type::Constructor {
            name: "List".to_string(),
            args: vec![Type::Int],
        };
        let subst = match_type(&pattern, &target);
        assert!(subst.is_some());
        let s = subst.unwrap();
        assert!(matches!(s.get(&0).unwrap(), Type::Int));
    }

    #[test]
    fn test_match_type_wrong_constructor() {
        // List 'a doesn't match Option Int
        let pattern = Type::Constructor {
            name: "List".to_string(),
            args: vec![Type::new_generic(0)],
        };
        let target = Type::Constructor {
            name: "Option".to_string(),
            args: vec![Type::Int],
        };
        assert!(match_type(&pattern, &target).is_none());
    }

    #[test]
    fn test_resolve_basic_instance() {
        let mut env = ClassEnv::new();

        // Add Show trait
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Add Show Int instance
        env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Int,
            constraints: vec![],
            method_impls: vec![],
        })
        .unwrap();

        // Resolve Show Int
        let pred = Pred::new("Show", Type::Int);
        let resolution = env.resolve_pred(&pred);
        assert!(resolution.is_some());
        let res = resolution.unwrap();
        assert_eq!(res.instance_idx, 0);
        assert!(res.sub_preds.is_empty());
    }

    #[test]
    fn test_resolve_constrained_instance() {
        let mut env = ClassEnv::new();

        // Add Show trait
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Add Show Int instance
        env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Int,
            constraints: vec![],
            method_impls: vec![],
        })
        .unwrap();

        // Add Show (List a) where a : Show
        env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::new_generic(0)],
            },
            constraints: vec![Pred::new("Show", Type::new_generic(0))],
            method_impls: vec![],
        })
        .unwrap();

        // Resolve Show (List Int) -> needs Show Int
        let pred = Pred::new(
            "Show",
            Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::Int],
            },
        );
        let resolution = env.resolve_pred(&pred);
        assert!(resolution.is_some());
        let res = resolution.unwrap();
        assert_eq!(res.instance_idx, 1); // The List instance is index 1
        assert_eq!(res.sub_preds.len(), 1); // Show Int
        assert_eq!(res.sub_preds[0].trait_name, "Show");
        assert!(matches!(res.sub_preds[0].ty, Type::Int));
    }

    #[test]
    fn test_resolve_all_recursive() {
        let mut env = ClassEnv::new();

        // Add Show trait
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Add Show Int instance
        env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Int,
            constraints: vec![],
            method_impls: vec![],
        })
        .unwrap();

        // Add Show (List a) where a : Show
        env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::new_generic(0)],
            },
            constraints: vec![Pred::new("Show", Type::new_generic(0))],
            method_impls: vec![],
        })
        .unwrap();

        // Resolve Show (List Int) - should resolve both List and Int
        let pred = Pred::new(
            "Show",
            Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::Int],
            },
        );
        let result = env.resolve_all(&[pred]);
        assert!(result.is_ok());
        let resolutions = result.unwrap();
        assert_eq!(resolutions.len(), 2); // Show (List Int) and Show Int
    }

    #[test]
    fn test_resolve_no_instance() {
        let mut env = ClassEnv::new();

        // Add Show trait but no instances
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Try to resolve Show Int - should fail
        let pred = Pred::new("Show", Type::Int);
        let result = env.resolve_all(&[pred]);
        assert!(result.is_err());
    }

    #[test]
    fn test_entail_direct() {
        let env = ClassEnv::new();

        // Given [Show Int], entails Show Int
        let given = vec![Pred::new("Show", Type::Int)];
        let goal = Pred::new("Show", Type::Int);
        assert!(env.entail(&given, &goal));
    }

    #[test]
    fn test_entail_via_instance() {
        let mut env = ClassEnv::new();

        // Add Show trait
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            Type::arrow(Type::new_generic(0), Type::String),
        );
        env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Add Show (List a) where a : Show
        env.add_instance(InstanceInfo {
            trait_name: "Show".to_string(),
            head: Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::new_generic(0)],
            },
            constraints: vec![Pred::new("Show", Type::new_generic(0))],
            method_impls: vec![],
        })
        .unwrap();

        // Given [Show Int], entails Show (List Int) via instance
        let given = vec![Pred::new("Show", Type::Int)];
        let goal = Pred::new(
            "Show",
            Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::Int],
            },
        );
        assert!(env.entail(&given, &goal));
    }
}
