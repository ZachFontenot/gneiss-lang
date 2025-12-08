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

    /// Function type: a -> b
    Arrow(Rc<Type>, Rc<Type>),

    /// Tuple type: (a, b, c)
    Tuple(Vec<Type>),

    /// List type: [a]
    List(Rc<Type>),

    /// Named type constructor with arguments: Option a, Result a e
    Constructor {
        name: String,
        args: Vec<Type>,
    },

    /// Channel type: Channel a
    Channel(Rc<Type>),

    /// Process ID type (returned by spawn)
    Pid,
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

impl Type {
    /// Create a new unbound type variable
    pub fn new_var(id: TypeVarId, level: u32) -> Type {
        Type::Var(Rc::new(RefCell::new(TypeVar::Unbound { id, level })))
    }

    /// Create a new generic type variable
    pub fn new_generic(id: TypeVarId) -> Type {
        Type::Var(Rc::new(RefCell::new(TypeVar::Generic(id))))
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
            Type::Arrow(a, b) => a.occurs(id) || b.occurs(id),
            Type::Tuple(types) => types.iter().any(|t| t.occurs(id)),
            Type::List(t) => t.occurs(id),
            Type::Constructor { args, .. } => args.iter().any(|t| t.occurs(id)),
            Type::Channel(t) => t.occurs(id),
            Type::Int | Type::Float | Type::Bool | Type::String | Type::Char | Type::Unit | Type::Pid => false,
        }
    }

    /// Create a function type
    pub fn arrow(from: Type, to: Type) -> Type {
        Type::Arrow(Rc::new(from), Rc::new(to))
    }

    /// Create a multi-argument function type: a -> b -> c -> d
    pub fn arrows(args: Vec<Type>, ret: Type) -> Type {
        args.into_iter()
            .rev()
            .fold(ret, |acc, arg| Type::arrow(arg, acc))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.resolve() {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Unbound { id, .. } => write!(f, "t{}", id),
                TypeVar::Generic(id) => write!(f, "'{}", ('a' as u8 + (*id % 26) as u8) as char),
                TypeVar::Link(_) => unreachable!(),
            },
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::String => write!(f, "String"),
            Type::Char => write!(f, "Char"),
            Type::Unit => write!(f, "()"),
            Type::Pid => write!(f, "Pid"),
            Type::Arrow(a, b) => {
                let a_str = match a.resolve() {
                    Type::Arrow(_, _) => format!("({})", a),
                    _ => format!("{}", a),
                };
                write!(f, "{} -> {}", a_str, b)
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
            Type::List(t) => write!(f, "[{}]", t),
            Type::Constructor { name, args } => {
                write!(f, "{}", name)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                Ok(())
            }
            Type::Channel(t) => write!(f, "Channel {}", t),
        }
    }
}

/// A polymorphic type scheme: forall a b. a -> b -> a
#[derive(Debug, Clone)]
pub struct Scheme {
    /// The number of generic type variables
    pub num_generics: u32,
    /// The underlying type
    pub ty: Type,
}

impl Scheme {
    /// A monomorphic type (no generics)
    pub fn mono(ty: Type) -> Scheme {
        Scheme {
            num_generics: 0,
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
                write!(f, "{}", ('a' as u8 + (i % 26) as u8) as char)?;
            }
            write!(f, ". ")?;
        }
        write!(f, "{}", self.ty)
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

/// Stores information about declared data types
#[derive(Debug, Clone, Default)]
pub struct TypeContext {
    /// Maps constructor names to their info
    pub constructors: HashMap<String, ConstructorInfo>,
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
}
