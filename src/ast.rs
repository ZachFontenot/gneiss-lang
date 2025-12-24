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

/// Human-readable source position (1-indexed line and column)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// 1-indexed line number
    pub line: usize,
    /// 1-indexed column number (in characters, not bytes)
    pub column: usize,
}

impl Position {
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A span with start and end positions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocatedSpan {
    pub start: Position,
    pub end: Position,
}

impl std::fmt::Display for LocatedSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.start.line == self.end.line {
            write!(
                f,
                "{}:{}-{}",
                self.start.line, self.start.column, self.end.column
            )
        } else {
            write!(f, "{}-{}", self.start, self.end)
        }
    }
}

/// Maps byte offsets to line:column positions.
///
/// This struct pre-computes line boundaries from source text,
/// enabling O(log n) lookup of positions from byte offsets.
#[derive(Debug, Clone)]
pub struct SourceMap {
    /// The original source text
    source: String,
    /// Byte offset of the start of each line (0-indexed)
    /// line_starts[0] = 0 (first line starts at byte 0)
    /// line_starts[1] = byte offset of second line, etc.
    line_starts: Vec<usize>,
}

impl SourceMap {
    /// Create a new SourceMap from source text
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                // Next line starts after the newline
                line_starts.push(i + 1);
            }
        }
        Self {
            source: source.to_string(),
            line_starts,
        }
    }

    /// Get the source text
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the number of lines
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Convert a byte offset to a Position (1-indexed line and column)
    pub fn position(&self, byte_offset: usize) -> Position {
        // Binary search to find the line containing this offset
        let line_idx = match self.line_starts.binary_search(&byte_offset) {
            Ok(idx) => idx,      // Exact match - at start of line
            Err(idx) => idx - 1, // Between line starts - use previous line
        };

        let line_start = self.line_starts[line_idx];

        // Calculate column by counting characters (not bytes) from line start
        // This handles multi-byte UTF-8 characters correctly
        let column = self.source[line_start..byte_offset].chars().count() + 1;

        Position {
            line: line_idx + 1, // 1-indexed
            column,
        }
    }

    /// Convert a Span to a LocatedSpan with line:column positions
    pub fn locate(&self, span: &Span) -> LocatedSpan {
        LocatedSpan {
            start: self.position(span.start),
            end: self.position(span.end),
        }
    }

    /// Get the text content of a line (1-indexed), without the trailing newline
    pub fn line(&self, line_num: usize) -> Option<&str> {
        if line_num == 0 || line_num > self.line_starts.len() {
            return None;
        }

        let line_idx = line_num - 1;
        let start = self.line_starts[line_idx];
        let end = if line_idx + 1 < self.line_starts.len() {
            // Line ends at start of next line, minus the newline character
            self.line_starts[line_idx + 1] - 1
        } else {
            // Last line - goes to end of source
            self.source.len()
        };

        // Handle the case where the line is empty or ends with newline
        let line = &self.source[start..end];
        Some(line.trim_end_matches('\r')) // Handle CRLF
    }

    /// Get the text content of a span
    pub fn span_text(&self, span: &Span) -> &str {
        &self.source[span.start..span.end.min(self.source.len())]
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

    /// Mutually recursive let bindings: let rec f = ... and g = ... in body
    LetRec {
        bindings: Vec<RecBinding>,
        body: Option<Rc<Expr>>, // None for top-level
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

    // ========================================================================
    // Algebraic Effects
    // ========================================================================
    /// Perform an effect operation: perform State.get ()
    Perform {
        /// Effect name (e.g., "State")
        effect: Ident,
        /// Operation name (e.g., "get")
        operation: Ident,
        /// Arguments to the operation
        args: Vec<Expr>,
    },

    /// Handle effects in an expression
    /// handle expr with | return x -> ... | op args k -> ... end
    Handle {
        /// Expression whose effects are being handled
        body: Rc<Expr>,
        /// Return clause: what to do with the final value
        return_clause: HandlerReturn,
        /// Operation handlers
        handlers: Vec<HandlerArm>,
    },

    // ========================================================================
    // Records
    // ========================================================================
    /// Record literal: Request { method = "GET", path = "/" }
    Record {
        /// Type name (e.g., "Request")
        name: Ident,
        /// Field assignments: (field_name, value)
        fields: Vec<(Ident, Expr)>,
    },

    /// Field access: expr.field
    FieldAccess {
        /// The record expression
        record: Rc<Expr>,
        /// The field name to access
        field: Ident,
    },

    /// Record update: { base with field = value, ... }
    RecordUpdate {
        /// The base record expression
        base: Rc<Expr>,
        /// Field updates: (field_name, new_value)
        updates: Vec<(Ident, Expr)>,
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

/// Return clause for an effect handler: | return x -> body
#[derive(Debug, Clone)]
pub struct HandlerReturn {
    /// Pattern for the return value
    pub pattern: Pattern,
    /// Body to execute with the return value (boxed to break recursion)
    pub body: Box<Expr>,
}

/// Handler arm for an effect operation: | op args k -> body
#[derive(Debug, Clone)]
pub struct HandlerArm {
    /// Operation name (e.g., "get" for State.get)
    pub operation: Ident,
    /// Parameters for the operation (not including continuation)
    pub params: Vec<Pattern>,
    /// Continuation parameter name
    pub continuation: Ident,
    /// Handler body (boxed to break recursion)
    pub body: Box<Expr>,
}

/// A single binding in a let rec group
#[derive(Debug, Clone)]
pub struct RecBinding {
    /// Function name (must be a simple identifier for recursion to work)
    pub name: Spanned<String>,
    /// Function parameters
    pub params: Vec<Pattern>,
    /// Function body
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

/// Binary operators (built-in and user-defined)
#[derive(Debug, Clone, PartialEq, Eq)]
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
    Pipe,        // |>
    PipeBack,    // <|
    Compose,     // >>
    ComposeBack, // <<
    // User-defined operator (looked up as a function)
    UserDefined(String),
}

/// An operator: either a built-in or a user-defined symbol
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operator {
    /// Built-in operator with hardcoded semantics
    Builtin(BinOp),
    /// User-defined operator (looked up as a function)
    User(String),
}

impl Operator {
    /// Get the operator symbol as a string
    pub fn symbol(&self) -> String {
        match self {
            Operator::Builtin(op) => match op {
                BinOp::Add => "+".to_string(),
                BinOp::Sub => "-".to_string(),
                BinOp::Mul => "*".to_string(),
                BinOp::Div => "/".to_string(),
                BinOp::Mod => "%".to_string(),
                BinOp::Eq => "==".to_string(),
                BinOp::Neq => "!=".to_string(),
                BinOp::Lt => "<".to_string(),
                BinOp::Gt => ">".to_string(),
                BinOp::Lte => "<=".to_string(),
                BinOp::Gte => ">=".to_string(),
                BinOp::And => "&&".to_string(),
                BinOp::Or => "||".to_string(),
                BinOp::Cons => "::".to_string(),
                BinOp::Concat => "++".to_string(),
                BinOp::Pipe => "|>".to_string(),
                BinOp::PipeBack => "<|".to_string(),
                BinOp::Compose => ">>".to_string(),
                BinOp::ComposeBack => "<<".to_string(),
                BinOp::UserDefined(s) => s.clone(),
            },
            Operator::User(s) => s.clone(),
        }
    }
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol())
    }
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

    // Record pattern: Request { method, path } or Request { method = m, path = p }
    Record {
        /// The record type name
        name: Ident,
        /// Field patterns: (field_name, optional binding pattern)
        /// If pattern is None, the field name is used as the variable binding
        fields: Vec<(Ident, Option<Pattern>)>,
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

    // Function type: a -> b { effects }
    Arrow {
        from: Rc<TypeExpr>,
        to: Rc<TypeExpr>,
        /// Optional effect row: { IO, State s | r }
        effects: Option<EffectRowExpr>,
    },

    // Tuple type: (a, b, c)
    Tuple(Vec<TypeExpr>),

    // Channel type: Channel a
    Channel(Rc<TypeExpr>),

    // List type: [a]
    List(Rc<TypeExpr>),
}

// ============================================================================
// Effect Rows (surface syntax for effect annotations)
// ============================================================================

/// An effect row expression: { IO, State s | r }
#[derive(Debug, Clone)]
pub struct EffectRowExpr {
    /// Concrete effects in the row: IO, State s, etc.
    pub effects: Vec<EffectExpr>,
    /// Optional row variable for polymorphism: the 'r' in { IO | r }
    pub rest: Option<Ident>,
    pub span: Span,
}

/// A single effect: IO, State s, Reader Config, etc.
#[derive(Debug, Clone)]
pub struct EffectExpr {
    /// Effect name (e.g., "IO", "State", "Reader")
    pub name: Ident,
    /// Effect type parameters (e.g., 's' in State s)
    pub params: Vec<TypeExpr>,
    pub span: Span,
}

// ============================================================================
// Operator Fixity Declarations
// ============================================================================

/// Operator associativity for fixity declarations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Associativity {
    /// Left-associative: `a op b op c` = `(a op b) op c`
    Left,
    /// Right-associative: `a op b op c` = `a op (b op c)`
    Right,
    /// Non-associative: `a op b op c` is a parse error
    None,
}

/// Fixity declaration: `infixl 6 +` or `infixr 5 ::`
#[derive(Debug, Clone)]
pub struct FixityDecl {
    pub assoc: Associativity,
    pub precedence: u8,
    pub operators: Vec<String>,
    pub span: Span,
}

// ============================================================================
// Visibility and Imports
// ============================================================================

/// Visibility modifier for declarations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    /// Public: accessible from other modules
    Public,
    /// Private: only accessible within the defining module (default)
    #[default]
    Private,
}

/// An import specification
#[derive(Debug, Clone)]
pub struct ImportSpec {
    /// The module path (e.g., "List" or "Collections/HashMap")
    pub module_path: String,
    /// Optional alias for the module (e.g., "L" in `import List as L`)
    pub alias: Option<String>,
    /// Optional selective imports. None = import whole module, Some = selective
    /// Each item is (name, optional alias), e.g., ("map", Some("listMap"))
    pub items: Option<Vec<(String, Option<String>)>>,
}

// ============================================================================
// Declarations
// ============================================================================

#[derive(Debug, Clone)]
pub enum Decl {
    // let x = e  or  let f a b = e
    Let {
        visibility: Visibility,
        name: Ident,
        type_ann: Option<TypeExpr>,
        params: Vec<Pattern>,
        body: Expr,
    },

    /// Mutually recursive function definitions: let rec f x = ... and g y = ...
    LetRec {
        visibility: Visibility,
        bindings: Vec<RecBinding>,
    },

    // Operator definition: let (<|>) a b = e or let a <|> b = e
    OperatorDef {
        visibility: Visibility,
        op: String,
        params: Vec<Pattern>,
        body: Expr,
    },

    // Fixity declaration: infixl 6 +
    Fixity(FixityDecl),

    // type Option a = | Some a | None
    Type {
        visibility: Visibility,
        name: Ident,
        params: Vec<Ident>,
        constructors: Vec<Constructor>,
    },

    // type alias: type UserId = Int
    TypeAlias {
        visibility: Visibility,
        name: Ident,
        params: Vec<Ident>,
        body: TypeExpr,
    },

    // trait Show a = val show : a -> String end
    Trait {
        visibility: Visibility,
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

    // val xs : [Int]  -- standalone type signature
    // val show_it : a -> String where a : Show  -- with trait constraints
    Val {
        name: Ident,
        type_sig: TypeExpr,
        constraints: Vec<Constraint>,
    },

    // effect State s = | get : () -> s | put : s -> () end
    EffectDecl {
        visibility: Visibility,
        name: Ident,
        params: Vec<Ident>,
        operations: Vec<EffectOperation>,
    },

    // Record type declaration: type Request = { method : String, path : String }
    Record {
        visibility: Visibility,
        name: Ident,
        params: Vec<Ident>,
        fields: Vec<RecordField>,
    },
}

#[derive(Debug, Clone)]
pub struct Constructor {
    pub name: Ident,
    pub fields: Vec<TypeExpr>,
}

/// A field in a record type declaration: field_name : Type
#[derive(Debug, Clone)]
pub struct RecordField {
    pub name: Ident,
    pub ty: TypeExpr,
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

/// An operation in an effect declaration: | get : () -> s
#[derive(Debug, Clone)]
pub struct EffectOperation {
    pub name: Ident,
    pub type_sig: TypeExpr,
}

// ============================================================================
// Exports
// ============================================================================

/// An export item in an export list
#[derive(Debug, Clone, PartialEq)]
pub enum ExportItem {
    /// Export a value/function: `foo`
    Value(String),
    /// Export a type only (constructors private): `MyType`
    TypeOnly(String),
    /// Export a type with all constructors: `MyType(..)`
    TypeAll(String),
    /// Export a type with specific constructors: `MyType(A, B)`
    TypeSome(String, Vec<String>),
}

/// An export declaration at the top of a module
#[derive(Debug, Clone)]
pub struct ExportDecl {
    pub items: Vec<Spanned<ExportItem>>,
    pub span: Span,
}

// ============================================================================
// Program
// ============================================================================

/// A top-level item: declaration, import, or expression
#[derive(Debug, Clone)]
pub enum Item {
    /// An import statement
    Import(Spanned<ImportSpec>),
    /// A declaration (let, type, trait, impl)
    Decl(Decl),
    /// A top-level expression
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Program {
    /// Optional export list. If None, all declarations are public.
    pub exports: Option<ExportDecl>,
    pub items: Vec<Item>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_map_single_line() {
        let source = "let x = 42";
        let map = SourceMap::new(source);

        assert_eq!(map.line_count(), 1);
        assert_eq!(map.position(0), Position::new(1, 1));
        assert_eq!(map.position(4), Position::new(1, 5));
        assert_eq!(map.line(1), Some("let x = 42"));
    }

    #[test]
    fn test_source_map_multiple_lines() {
        let source = "let x = 1\nlet y = 2\nlet z = 3";
        let map = SourceMap::new(source);

        assert_eq!(map.line_count(), 3);

        // First line
        assert_eq!(map.position(0), Position::new(1, 1));
        assert_eq!(map.position(9), Position::new(1, 10));

        // Second line (starts at byte 10)
        assert_eq!(map.position(10), Position::new(2, 1));
        assert_eq!(map.position(14), Position::new(2, 5));

        // Third line (starts at byte 20)
        assert_eq!(map.position(20), Position::new(3, 1));

        assert_eq!(map.line(1), Some("let x = 1"));
        assert_eq!(map.line(2), Some("let y = 2"));
        assert_eq!(map.line(3), Some("let z = 3"));
    }

    #[test]
    fn test_source_map_utf8() {
        // Test with multi-byte UTF-8 characters
        let source = "let π = 3.14";
        let map = SourceMap::new(source);

        // π is 2 bytes in UTF-8, but should be column 5 (1 character)
        assert_eq!(map.position(0), Position::new(1, 1));
        // 'let ' = 4 bytes, 'π' = 2 bytes, so '=' is at byte 7
        // But in columns: 'let ' = 4 chars, 'π' = 1 char, ' ' = 1 char, so '=' is column 7
        assert_eq!(map.position(7), Position::new(1, 7));
    }

    #[test]
    fn test_source_map_span_locate() {
        let source = "let x = 1\nlet y = 2";
        let map = SourceMap::new(source);

        // Span covering "x" on line 1
        let span = Span::new(4, 5);
        let loc = map.locate(&span);
        assert_eq!(loc.start, Position::new(1, 5));
        assert_eq!(loc.end, Position::new(1, 6));

        // Span covering "y" on line 2
        let span = Span::new(14, 15);
        let loc = map.locate(&span);
        assert_eq!(loc.start, Position::new(2, 5));
        assert_eq!(loc.end, Position::new(2, 6));
    }

    #[test]
    fn test_source_map_empty_lines() {
        let source = "let x = 1\n\nlet y = 2";
        let map = SourceMap::new(source);

        assert_eq!(map.line_count(), 3);
        assert_eq!(map.line(1), Some("let x = 1"));
        assert_eq!(map.line(2), Some(""));
        assert_eq!(map.line(3), Some("let y = 2"));
    }

    #[test]
    fn test_source_map_span_text() {
        let source = "let foo = bar";
        let map = SourceMap::new(source);

        let span = Span::new(4, 7);
        assert_eq!(map.span_text(&span), "foo");
    }

    #[test]
    fn test_located_span_display() {
        // Single line span
        let loc = LocatedSpan {
            start: Position::new(5, 10),
            end: Position::new(5, 15),
        };
        assert_eq!(format!("{}", loc), "5:10-15");

        // Multi-line span
        let loc = LocatedSpan {
            start: Position::new(5, 10),
            end: Position::new(7, 3),
        };
        assert_eq!(format!("{}", loc), "5:10-7:3");
    }
}
