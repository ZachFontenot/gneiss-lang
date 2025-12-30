//! Typed Abstract Syntax Tree (TAST)
//!
//! The TAST is produced by elaboration after type inference. Every node
//! carries its resolved, concrete type. This enables:
//! - Monomorphization of trait methods
//! - Type-directed code generation
//! - Clean separation from untyped parsing

use std::rc::Rc;

use crate::ast::{Ident, Literal, Span};
use crate::types::Type;

// ============================================================================
// Typed Expressions
// ============================================================================

/// A typed expression with resolved type information
#[derive(Debug, Clone)]
pub struct TExpr {
    pub node: TExprKind,
    pub ty: Type,
    pub span: Span,
}

impl TExpr {
    pub fn new(node: TExprKind, ty: Type, span: Span) -> Self {
        TExpr { node, ty, span }
    }
}

/// Typed expression kinds
#[derive(Debug, Clone)]
pub enum TExprKind {
    /// Variable reference
    Var(Ident),

    /// Literal value
    Lit(Literal),

    /// Lambda: fun x -> body
    Lambda {
        params: Vec<TPattern>,
        body: Rc<TExpr>,
    },

    /// Function application
    App {
        func: Rc<TExpr>,
        arg: Rc<TExpr>,
    },

    /// Let binding: let pat = value in body
    Let {
        pattern: TPattern,
        value: Rc<TExpr>,
        body: Option<Rc<TExpr>>,
    },

    /// Recursive let: let rec name params = body in rest
    LetRec {
        bindings: Vec<TRecBinding>,
        body: Option<Rc<TExpr>>,
    },

    /// If expression
    If {
        cond: Rc<TExpr>,
        then_branch: Rc<TExpr>,
        else_branch: Rc<TExpr>,
    },

    /// Pattern match
    Match {
        scrutinee: Rc<TExpr>,
        arms: Vec<TMatchArm>,
    },

    /// Tuple construction
    Tuple(Vec<TExpr>),

    /// List construction
    List(Vec<TExpr>),

    /// Record construction
    Record {
        name: Ident,
        fields: Vec<(Ident, TExpr)>,
    },

    /// Field access: expr.field
    FieldAccess {
        record: Rc<TExpr>,
        field: Ident,
    },

    /// Record update: { expr with field = value }
    RecordUpdate {
        base: Rc<TExpr>,
        updates: Vec<(Ident, TExpr)>,
    },

    /// Constructor application: Some x
    Constructor {
        name: Ident,
        args: Vec<TExpr>,
    },

    /// Binary operation (resolved to concrete types)
    BinOp {
        op: TBinOp,
        left: Rc<TExpr>,
        right: Rc<TExpr>,
    },

    /// Unary operation
    UnaryOp {
        op: TUnaryOp,
        operand: Rc<TExpr>,
    },

    /// Sequence: expr1; expr2
    Seq {
        first: Rc<TExpr>,
        second: Rc<TExpr>,
    },

    /// Monomorphized trait method call (concrete type known)
    /// This is the key for codegen - we know the concrete instance
    MethodCall {
        /// The trait name (e.g., "Show")
        trait_name: Ident,
        /// The method name (e.g., "show")
        method: Ident,
        /// The concrete type for this instance (e.g., Int, List Int)
        instance_ty: Type,
        /// Arguments to the method
        args: Vec<TExpr>,
    },

    /// Dictionary-based method call (type is polymorphic/unknown)
    /// Used when the instance type is a type variable from an enclosing scope
    DictMethodCall {
        /// The trait name (e.g., "Show")
        trait_name: Ident,
        /// The method name (e.g., "show")
        method: Ident,
        /// The type variable ID this dictionary is for
        type_var: u32,
        /// Arguments to the method
        args: Vec<TExpr>,
    },

    /// Concrete dictionary value (for passing to polymorphic functions)
    /// Used when calling a function with class constraints at a known type
    DictValue {
        /// The trait (e.g., "Show")
        trait_name: Ident,
        /// The concrete type (e.g., Int, List Int)
        instance_ty: Type,
    },

    /// Reference to a dictionary parameter (from enclosing polymorphic function)
    /// Used when forwarding dictionaries through polymorphic calls
    DictRef {
        /// The trait name
        trait_name: Ident,
        /// The type variable ID
        type_var: u32,
    },

    /// Effect operation: perform Effect.op args
    Perform {
        effect: Ident,
        op: Ident,
        args: Vec<TExpr>,
    },

    /// Effect handler: handle expr with | op args k -> body end
    Handle {
        body: Rc<TExpr>,
        handler: THandler,
    },

    /// Error placeholder (from elaboration errors)
    Error(String),
}

// ============================================================================
// Typed Patterns
// ============================================================================

/// A typed pattern
#[derive(Debug, Clone)]
pub struct TPattern {
    pub node: TPatternKind,
    pub ty: Type,
    pub span: Span,
}

impl TPattern {
    pub fn new(node: TPatternKind, ty: Type, span: Span) -> Self {
        TPattern { node, ty, span }
    }
}

/// Typed pattern kinds
#[derive(Debug, Clone)]
pub enum TPatternKind {
    /// Wildcard: _
    Wildcard,

    /// Variable binding: x
    Var(Ident),

    /// Literal: 42, "hello"
    Lit(Literal),

    /// Tuple: (a, b, c)
    Tuple(Vec<TPattern>),

    /// List: [a, b, c]
    List(Vec<TPattern>),

    /// Cons: x :: xs
    Cons {
        head: Rc<TPattern>,
        tail: Rc<TPattern>,
    },

    /// Constructor: Some x
    Constructor {
        name: Ident,
        args: Vec<TPattern>,
    },

    /// Record: Point { x, y }
    Record {
        name: Ident,
        fields: Vec<(Ident, Option<TPattern>)>,
    },
}

// ============================================================================
// Supporting Types
// ============================================================================

/// Typed match arm
#[derive(Debug, Clone)]
pub struct TMatchArm {
    pub pattern: TPattern,
    pub guard: Option<Rc<TExpr>>,
    pub body: TExpr,
}

/// Typed recursive binding
#[derive(Debug, Clone)]
pub struct TRecBinding {
    pub name: Ident,
    pub params: Vec<TPattern>,
    pub body: TExpr,
    /// The function's type
    pub ty: Type,
}

/// Typed effect handler
#[derive(Debug, Clone)]
pub struct THandler {
    /// Effect being handled
    pub effect: Option<Ident>,
    /// Return clause: return x -> body
    pub return_clause: Option<THandlerClause>,
    /// Operation handlers
    pub op_clauses: Vec<TOpClause>,
}

/// Handler clause for return
#[derive(Debug, Clone)]
pub struct THandlerClause {
    pub pattern: TPattern,
    pub body: Box<TExpr>,
}

/// Handler clause for an operation
#[derive(Debug, Clone)]
pub struct TOpClause {
    pub op_name: Ident,
    pub params: Vec<TPattern>,
    pub continuation: Ident,
    pub body: Box<TExpr>,
}

/// Typed binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TBinOp {
    // Arithmetic (Int)
    IntAdd,
    IntSub,
    IntMul,
    IntDiv,
    IntMod,

    // Arithmetic (Float)
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatDiv,

    // Comparison (Int)
    IntEq,
    IntNe,
    IntLt,
    IntLe,
    IntGt,
    IntGe,

    // Comparison (Float)
    FloatEq,
    FloatNe,
    FloatLt,
    FloatLe,
    FloatGt,
    FloatGe,

    // String
    StringEq,
    StringNe,
    StringConcat,

    // Boolean
    BoolAnd,
    BoolOr,

    // List
    Cons,

    // Pipe operators (become App during elaboration, but kept for reference)
    Pipe,
    PipeLeft,
    Compose,
    ComposeLeft,
}

/// Typed unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TUnaryOp {
    IntNeg,
    FloatNeg,
    BoolNot,
}

// ============================================================================
// Typed Program
// ============================================================================

/// A typed program
#[derive(Debug, Clone)]
pub struct TProgram {
    /// Type declarations (carried through for reference)
    pub type_decls: Vec<TTypeDecl>,
    /// Effect declarations
    pub effect_decls: Vec<TEffectDecl>,
    /// Trait declarations
    pub trait_decls: Vec<TTraitDecl>,
    /// Instance declarations (monomorphized)
    pub instance_decls: Vec<TInstanceDecl>,
    /// Top-level bindings
    pub bindings: Vec<TBinding>,
    /// Main expression (if any)
    pub main: Option<TExpr>,
}

/// Type declaration
#[derive(Debug, Clone)]
pub struct TTypeDecl {
    pub name: Ident,
    pub params: Vec<Ident>,
    pub constructors: Vec<TConstructor>,
}

/// Constructor in a type declaration
#[derive(Debug, Clone)]
pub struct TConstructor {
    pub name: Ident,
    pub fields: Vec<Type>,
}

/// Effect declaration
#[derive(Debug, Clone)]
pub struct TEffectDecl {
    pub name: Ident,
    pub operations: Vec<TOperation>,
}

/// Effect operation
#[derive(Debug, Clone)]
pub struct TOperation {
    pub name: Ident,
    pub param_tys: Vec<Type>,
    pub return_ty: Type,
}

/// Trait declaration
#[derive(Debug, Clone)]
pub struct TTraitDecl {
    pub name: Ident,
    pub param: Ident,
    pub methods: Vec<TMethodSig>,
}

/// Method signature in a trait
#[derive(Debug, Clone)]
pub struct TMethodSig {
    pub name: Ident,
    pub ty: Type,
}

/// Monomorphized instance declaration
#[derive(Debug, Clone)]
pub struct TInstanceDecl {
    pub trait_name: Ident,
    /// The concrete type this instance is for
    pub instance_ty: Type,
    /// Mangled name for codegen (e.g., "Show_Int", "Show_List_Int")
    pub mangled_name: String,
    /// Method implementations
    pub methods: Vec<TMethodImpl>,
}

/// Method implementation in an instance
#[derive(Debug, Clone)]
pub struct TMethodImpl {
    pub name: Ident,
    /// Mangled name for codegen
    pub mangled_name: String,
    pub params: Vec<TPattern>,
    pub body: TExpr,
}

/// Top-level binding
#[derive(Debug, Clone)]
pub struct TBinding {
    pub name: Ident,
    pub params: Vec<TPattern>,
    pub body: TExpr,
    pub ty: Type,
    /// Dictionary parameters required by this function.
    /// Each entry is (trait_name, type_var_id) - e.g., ("Show", 0) means
    /// this function needs a Show dictionary for its first type parameter.
    /// Empty if the function has no class constraints.
    pub dict_params: Vec<(String, u32)>,
}
