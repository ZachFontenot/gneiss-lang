//! Abstract Syntax Tree for Gneiss

use std::rc::Rc;

pub type Ident = String;

/// Source location for error reporting
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(&self, other: &Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A spanned AST node
#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

// ============================================================================
// Expressions
// ============================================================================

pub type Expr = Spanned<ExprKind>;

#[derive(Debug, Clone)]
pub enum ExprKind {
    // Literals
    Lit(Literal),

    // Variable reference
    Var(Ident),

    // Lambda: fun x y -> body
    Lambda {
        params: Vec<Pattern>,
        body: Rc<Expr>,
    },

    // Application: f x y
    App {
        func: Rc<Expr>,
        arg: Rc<Expr>,
    },

    // Let binding: let x = e1 in e2
    Let {
        pattern: Pattern,
        value: Rc<Expr>,
        body: Option<Rc<Expr>>, // None for top-level lets
    },

    // If expression
    If {
        cond: Rc<Expr>,
        then_branch: Rc<Expr>,
        else_branch: Rc<Expr>,
    },

    // Match expression
    Match {
        scrutinee: Rc<Expr>,
        arms: Vec<MatchArm>,
    },

    // Tuple: (a, b, c)
    Tuple(Vec<Expr>),

    // List: [a, b, c]
    List(Vec<Expr>),

    // Constructor application: Some x, Cons x xs
    Constructor {
        name: Ident,
        args: Vec<Expr>,
    },

    // Binary operator
    BinOp {
        op: BinOp,
        left: Rc<Expr>,
        right: Rc<Expr>,
    },

    // Unary operator
    UnaryOp {
        op: UnaryOp,
        operand: Rc<Expr>,
    },

    // Sequencing: e1; e2
    Seq {
        first: Rc<Expr>,
        second: Rc<Expr>,
    },

    // ========================================================================
    // Concurrency primitives
    // ========================================================================

    // spawn (fun () -> ...)
    Spawn(Rc<Expr>),

    // Channel.new ()
    NewChannel,

    // Channel.send ch value
    ChanSend {
        channel: Rc<Expr>,
        value: Rc<Expr>,
    },

    // Channel.recv ch
    ChanRecv(Rc<Expr>),

    // select [ ... ]
    Select {
        arms: Vec<SelectArm>,
    },

    // ========================================================================
    // Delimited continuations
    // ========================================================================

    /// Delimited continuation boundary: reset expr
    Reset(Rc<Expr>),

    /// Capture continuation: shift (fun k -> body)
    Shift {
        /// Parameter that binds the captured continuation
        param: Pattern,
        /// Body to execute with continuation bound
        body: Rc<Expr>,
    },
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct SelectArm {
    pub channel: Expr,
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
    Bool(bool),
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Comparison
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    // Boolean
    And,
    Or,
    // List
    Cons,
    Concat,
    // Function
    Pipe,     // |>
    PipeBack, // <|
    Compose,  // >>
    ComposeBack, // <<
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

// ============================================================================
// Patterns
// ============================================================================

pub type Pattern = Spanned<PatternKind>;

#[derive(Debug, Clone)]
pub enum PatternKind {
    // Wildcard: _
    Wildcard,

    // Variable binding: x
    Var(Ident),

    // Literal pattern: 42, "hello", true
    Lit(Literal),

    // Tuple pattern: (x, y, z)
    Tuple(Vec<Pattern>),

    // List pattern: [x, y, z]
    List(Vec<Pattern>),

    // Cons pattern: x :: xs
    Cons {
        head: Rc<Pattern>,
        tail: Rc<Pattern>,
    },

    // Constructor pattern: Some x, None
    Constructor {
        name: Ident,
        args: Vec<Pattern>,
    },
}

// ============================================================================
// Types (surface syntax)
// ============================================================================

pub type TypeExpr = Spanned<TypeExprKind>;

#[derive(Debug, Clone)]
pub enum TypeExprKind {
    // Type variable: a
    Var(Ident),

    // Named type: Int, String, List
    Named(Ident),

    // Type application: List a, Result a e
    App {
        constructor: Rc<TypeExpr>,
        args: Vec<TypeExpr>,
    },

    // Function type: a -> b
    Arrow {
        from: Rc<TypeExpr>,
        to: Rc<TypeExpr>,
    },

    // Tuple type: (a, b, c)
    Tuple(Vec<TypeExpr>),

    // Channel type: Channel a
    Channel(Rc<TypeExpr>),
}

// ============================================================================
// Declarations
// ============================================================================

#[derive(Debug, Clone)]
pub enum Decl {
    // let x = e  or  let f a b = e
    Let {
        name: Ident,
        type_ann: Option<TypeExpr>,
        params: Vec<Pattern>,
        body: Expr,
    },

    // type Option a = | Some a | None
    Type {
        name: Ident,
        params: Vec<Ident>,
        constructors: Vec<Constructor>,
    },

    // type alias: type UserId = Int
    TypeAlias {
        name: Ident,
        params: Vec<Ident>,
        body: TypeExpr,
    },

    // trait Show a = val show : a -> String end
    Trait {
        name: Ident,
        type_param: Ident,
        supertraits: Vec<Ident>,
        methods: Vec<TraitMethod>,
    },

    // impl Show for Int = let show n = ... end
    Instance {
        trait_name: Ident,
        target_type: TypeExpr,
        constraints: Vec<Constraint>,
        methods: Vec<InstanceMethod>,
    },
}

#[derive(Debug, Clone)]
pub struct Constructor {
    pub name: Ident,
    pub fields: Vec<TypeExpr>,
}

/// A method signature in a trait declaration: val show : a -> String
#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: Ident,
    pub type_sig: TypeExpr,
}

/// A method implementation in an instance declaration
#[derive(Debug, Clone)]
pub struct InstanceMethod {
    pub name: Ident,
    pub params: Vec<Pattern>,
    pub body: Expr,
}

/// A typeclass constraint: a : Show  (meaning type variable 'a' must implement Show)
#[derive(Debug, Clone)]
pub struct Constraint {
    pub type_var: Ident,
    pub trait_name: Ident,
}

// ============================================================================
// Program
// ============================================================================

/// A top-level item: either a declaration or an expression
#[derive(Debug, Clone)]
pub enum Item {
    Decl(Decl),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}
