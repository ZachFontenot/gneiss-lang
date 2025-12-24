//! Internal type representation for type inference

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// A type variable ID
pub type TypeVarId = u32;

/// The internal representation of types during inference
#[derive(Debug, Clone)]
pub enum Type {
    /// Type variable (possibly unified with another type)
    Var(Rc<RefCell<TypeVar>>),

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

/// A type variable that may or may not be bound
#[derive(Debug, Clone)]
pub enum TypeVar {
    /// Unbound type variable with a unique ID and level (for let-polymorphism)
    Unbound { id: TypeVarId, level: u32 },
    /// Bound to another type
    Link(Type),
    /// Generic type variable (for polymorphic types)
    Generic(TypeVarId),
}

// ============================================================================
// Effect Row Types (for Koka-style Algebraic Effects)
// ============================================================================

/// An effect row representing a set of effects with optional extension variable.
/// Syntax: `{ eff1, eff2 s | r }` where r is the row extension variable.
#[derive(Debug, Clone)]
pub enum Row {
    /// Empty row - pure, no effects
    Empty,
    /// Effect present with tail: { eff | rest }
    Extend {
        effect: Effect,
        rest: Rc<Row>,
    },
    /// Row variable for polymorphism
    Var(Rc<RefCell<RowVar>>),
}

/// A row variable that may or may not be bound
#[derive(Debug, Clone)]
pub enum RowVar {
    /// Unbound row variable with ID and level
    Unbound { id: TypeVarId, level: u32 },
    /// Bound to another row
    Link(Row),
    /// Generic row variable (for polymorphic effect rows)
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
    /// Create a new unbound row variable
    pub fn new_var(id: TypeVarId, level: u32) -> Row {
        Row::Var(Rc::new(RefCell::new(RowVar::Unbound { id, level })))
    }

    /// Create a new generic row variable
    pub fn new_generic(id: TypeVarId) -> Row {
        Row::Var(Rc::new(RefCell::new(RowVar::Generic(id))))
    }

    /// Extend a row with an effect
    pub fn extend(effect: Effect, rest: Row) -> Row {
        Row::Extend {
            effect,
            rest: Rc::new(rest),
        }
    }

    /// Follow all links to get the actual row
    pub fn resolve(&self) -> Row {
        match self {
            Row::Var(var) => match &*var.borrow() {
                RowVar::Link(row) => row.resolve(),
                _ => self.clone(),
            },
            _ => self.clone(),
        }
    }

    /// Check if this row contains a given row variable (occurs check for row vars)
    pub fn occurs(&self, id: TypeVarId) -> bool {
        match self.resolve() {
            Row::Empty => false,
            Row::Var(var) => match &*var.borrow() {
                RowVar::Unbound { id: vid, .. } => *vid == id,
                RowVar::Generic(vid) => *vid == id,
                RowVar::Link(_) => unreachable!("resolve should have followed links"),
            },
            Row::Extend { effect: _, rest } => rest.occurs(id),
        }
    }

    /// Check if a type variable occurs in this row (including in effect parameters)
    pub fn type_var_occurs(&self, id: TypeVarId) -> bool {
        match self.resolve() {
            Row::Empty => false,
            Row::Var(_) => false, // Row vars are separate from type vars
            Row::Extend { effect, rest } => {
                effect.params.iter().any(|t| t.occurs(id)) || rest.type_var_occurs(id)
            }
        }
    }

    /// Check if this row is empty (pure)
    pub fn is_empty(&self) -> bool {
        matches!(self.resolve(), Row::Empty)
    }

    /// Check if this row contains a specific effect
    pub fn contains_effect(&self, effect_name: &str) -> bool {
        match self.resolve() {
            Row::Empty => false,
            Row::Var(_) => false, // Unknown - could contain anything
            Row::Extend { effect, rest } => {
                effect.name == effect_name || rest.contains_effect(effect_name)
            }
        }
    }

    /// Display the row for user-friendly output
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
        match self.resolve() {
            Row::Empty => String::new(),
            Row::Var(var) => {
                let id = match &*var.borrow() {
                    RowVar::Unbound { id, .. } => *id,
                    RowVar::Generic(id) => *id,
                    RowVar::Link(_) => unreachable!(),
                };
                let c = *var_map.entry(id).or_insert_with(|| {
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
    /// Create a new unbound type variable
    pub fn new_var(id: TypeVarId, level: u32) -> Type {
        Type::Var(Rc::new(RefCell::new(TypeVar::Unbound { id, level })))
    }

    /// Create a new generic type variable
    pub fn new_generic(id: TypeVarId) -> Type {
        Type::Var(Rc::new(RefCell::new(TypeVar::Generic(id))))
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

    /// Follow all links to get the actual type
    pub fn resolve(&self) -> Type {
        match self {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Link(ty) => ty.resolve(),
                _ => self.clone(),
            },
            _ => self.clone(),
        }
    }

    /// Check if this type contains a given type variable (occurs check)
    pub fn occurs(&self, id: TypeVarId) -> bool {
        match self.resolve() {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Unbound { id: vid, .. } => *vid == id,
                TypeVar::Generic(vid) => *vid == id,
                TypeVar::Link(_) => unreachable!("resolve should have followed links"),
            },
            Type::Arrow { arg, ret, effects } => {
                arg.occurs(id) || ret.occurs(id) || effects.type_var_occurs(id)
            }
            Type::Tuple(types) => types.iter().any(|t| t.occurs(id)),
            Type::Constructor { args, .. } => args.iter().any(|t| t.occurs(id)),
            Type::Channel(t) => t.occurs(id),
            Type::Fiber(t) => t.occurs(id),
            Type::Dict(t) => t.occurs(id),
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
    pub fn arrow_with_effect_var(from: Type, to: Type, var_id: TypeVarId, level: u32) -> Type {
        Type::Arrow {
            arg: Rc::new(from),
            ret: Rc::new(to),
            effects: Row::new_var(var_id, level),
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
    /// Also hides answer types for user-friendly output.
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
        match self.resolve() {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Unbound { id, .. } => {
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
                TypeVar::Generic(id) => {
                    let c = *var_map.entry(*id).or_insert_with(|| {
                        let c = *next_var;
                        *next_var = if *next_var == 'z' {
                            'a'
                        } else {
                            (*next_var as u8 + 1) as char
                        };
                        c
                    });
                    format!("'{}", c)
                }
                TypeVar::Link(_) => unreachable!("resolve should follow links"),
            },
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
                let arg_str = match arg.resolve() {
                    Type::Arrow { .. } => format!(
                        "({})",
                        arg.display_with_map(var_map, next_var, hide_answer_types)
                    ),
                    _ => arg.display_with_map(var_map, next_var, hide_answer_types),
                };
                let ret_str = ret.display_with_map(var_map, next_var, hide_answer_types);

                // Check if pure (empty effects) or polymorphic
                let effects_resolved = effects.resolve();
                match effects_resolved {
                    Row::Empty => {
                        // Pure function: a -> b
                        format!("{} -> {}", arg_str, ret_str)
                    }
                    Row::Var(_) if hide_answer_types => {
                        // Polymorphic effects, hide for user-friendly display
                        format!("{} -> {}", arg_str, ret_str)
                    }
                    _ => {
                        // Show effects: a -> b { eff1, eff2 | r }
                        let effects_str = effects_resolved.display_user_friendly();
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
                    name
                } else {
                    let arg_strs: Vec<String> = args
                        .iter()
                        .map(|a| {
                            // Wrap complex types in parens
                            let s = a.display_with_map(var_map, next_var, hide_answer_types);
                            let needs_parens = match a.resolve() {
                                Type::Arrow { .. } => true,
                                Type::Constructor {
                                    args: inner_args, ..
                                } => !inner_args.is_empty(),
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
        match self.resolve() {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Unbound { id, .. } => write!(f, "t{}", id),
                TypeVar::Generic(id) => write!(f, "'{}", (b'a' + (*id % 26) as u8) as char),
                TypeVar::Link(_) => unreachable!(),
            },
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
                let arg_str = match arg.resolve() {
                    Type::Arrow { .. } => format!("({})", arg),
                    _ => format!("{}", arg),
                };
                // Check if pure (empty effects) or has effects
                let effects_resolved = effects.resolve();
                match effects_resolved {
                    Row::Empty => {
                        // Pure function: show as σ → τ
                        write!(f, "{} -> {}", arg_str, ret)
                    }
                    Row::Var(_) => {
                        // Polymorphic effects - show as pure for simplicity
                        write!(f, "{} -> {}", arg_str, ret)
                    }
                    _ => {
                        // Has effects: show as σ → τ { effects }
                        let effects_str = effects_resolved.display_user_friendly();
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
    pub fn is_pure(&self) -> bool {
        matches!(self.effects.resolve(), Row::Empty)
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
        self.trait_name == other.trait_name && types_equal(&self.ty, &other.ty)
    }
}

/// Check if two types are structurally equal (resolving links)
pub fn types_equal(t1: &Type, t2: &Type) -> bool {
    let t1 = t1.resolve();
    let t2 = t2.resolve();

    match (&t1, &t2) {
        (Type::Int, Type::Int) => true,
        (Type::Float, Type::Float) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::String, Type::String) => true,
        (Type::Char, Type::Char) => true,
        (Type::Unit, Type::Unit) => true,
        (Type::Pid, Type::Pid) => true,
        (Type::Var(v1), Type::Var(v2)) => match (&*v1.borrow(), &*v2.borrow()) {
            (TypeVar::Unbound { id: id1, .. }, TypeVar::Unbound { id: id2, .. }) => id1 == id2,
            (TypeVar::Generic(id1), TypeVar::Generic(id2)) => id1 == id2,
            _ => false,
        },
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
        ) => types_equal(a1, a2) && types_equal(r1, r2) && rows_equal(e1, e2),
        (Type::Tuple(ts1), Type::Tuple(ts2)) => {
            ts1.len() == ts2.len() && ts1.iter().zip(ts2).all(|(x, y)| types_equal(x, y))
        }
        (Type::Constructor { name: n1, args: a1 }, Type::Constructor { name: n2, args: a2 }) => {
            n1 == n2 && a1.len() == a2.len() && a1.iter().zip(a2).all(|(x, y)| types_equal(x, y))
        }
        (Type::Channel(e1), Type::Channel(e2)) => types_equal(e1, e2),
        _ => false,
    }
}

/// Check if two effect rows are structurally equal (resolving links)
pub fn rows_equal(r1: &Row, r2: &Row) -> bool {
    let r1 = r1.resolve();
    let r2 = r2.resolve();

    match (&r1, &r2) {
        (Row::Empty, Row::Empty) => true,
        (Row::Var(v1), Row::Var(v2)) => match (&*v1.borrow(), &*v2.borrow()) {
            (RowVar::Unbound { id: id1, .. }, RowVar::Unbound { id: id2, .. }) => id1 == id2,
            (RowVar::Generic(id1), RowVar::Generic(id2)) => id1 == id2,
            _ => false,
        },
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
                    .all(|(t1, t2)| types_equal(t1, t2))
                && rows_equal(rest1, rest2)
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
pub fn apply_subst(ty: &Type, subst: &HashMap<TypeVarId, Type>) -> Type {
    match ty.resolve() {
        Type::Var(var) => match &*var.borrow() {
            TypeVar::Generic(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
            TypeVar::Unbound { id, .. } => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
            TypeVar::Link(_) => unreachable!("resolve should follow links"),
        },
        Type::Arrow { arg, ret, effects } => Type::Arrow {
            arg: Rc::new(apply_subst(&arg, subst)),
            ret: Rc::new(apply_subst(&ret, subst)),
            effects: apply_subst_to_row(&effects, subst),
        },
        Type::Tuple(types) => Type::Tuple(types.iter().map(|t| apply_subst(t, subst)).collect()),
        Type::Constructor { name, args } => Type::Constructor {
            name,
            args: args.iter().map(|t| apply_subst(t, subst)).collect(),
        },
        Type::Channel(t) => Type::Channel(Rc::new(apply_subst(&t, subst))),
        t => t, // Primitives unchanged
    }
}

/// Apply a substitution to an effect row (for type parameters in effects)
pub fn apply_subst_to_row(row: &Row, subst: &HashMap<TypeVarId, Type>) -> Row {
    match row.resolve() {
        Row::Empty => Row::Empty,
        Row::Var(var) => Row::Var(var), // Row variables are not substituted by type subst
        Row::Extend { effect, rest } => Row::Extend {
            effect: Effect {
                name: effect.name.clone(),
                params: effect.params.iter().map(|t| apply_subst(t, subst)).collect(),
            },
            rest: Rc::new(apply_subst_to_row(&rest, subst)),
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
fn types_overlap(t1: &Type, t2: &Type) -> bool {
    let t1 = t1.resolve();
    let t2 = t2.resolve();

    match (&t1, &t2) {
        // Type variables overlap with anything
        (Type::Var(_), _) | (_, Type::Var(_)) => true,

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
pub fn rows_overlap(r1: &Row, r2: &Row) -> bool {
    let r1 = r1.resolve();
    let r2 = r2.resolve();

    match (&r1, &r2) {
        // Empty rows overlap only with empty or variables
        (Row::Empty, Row::Empty) => true,
        (Row::Empty, Row::Var(_)) | (Row::Var(_), Row::Empty) => true,
        // Variables overlap with anything
        (Row::Var(_), _) | (_, Row::Var(_)) => true,
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
    let pattern = pattern.resolve();
    let target = target.resolve();

    match (&pattern, &target) {
        // Pattern variable matches anything
        (Type::Var(var), _) => match &*var.borrow() {
            TypeVar::Generic(id) => {
                if let Some(existing) = subst.get(id) {
                    // Must match previously bound type
                    types_equal(existing, &target)
                } else {
                    subst.insert(*id, target.clone());
                    true
                }
            }
            TypeVar::Unbound { id, .. } => {
                if let Some(existing) = subst.get(id) {
                    types_equal(existing, &target)
                } else {
                    subst.insert(*id, target.clone());
                    true
                }
            }
            TypeVar::Link(_) => unreachable!("resolve should follow links"),
        },

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
    let pattern = pattern.resolve();
    let target = target.resolve();

    match (&pattern, &target) {
        (Row::Empty, Row::Empty) => true,
        (Row::Var(v), _) => {
            // Row variables in pattern are like wildcards for now
            // Could be extended to track row substitutions
            match &*v.borrow() {
                RowVar::Generic(_) | RowVar::Unbound { .. } => true,
                RowVar::Link(_) => unreachable!(),
            }
        }
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
                    if matches!(pred.ty.resolve(), Type::Var(_)) {
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
        for g in given {
            if g.trait_name == goal.trait_name && types_equal(&g.ty, &goal.ty) {
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
        assert!(matches!(applied.ty.resolve(), Type::Int));
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
        assert!(matches!(res.sub_preds[0].ty.resolve(), Type::Int));
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
