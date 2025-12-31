//! Core Intermediate Representation
//!
//! A-Normal Form (ANF) IR with explicit allocation, suitable for Perceus transformation.
//! All intermediate values are named, and control flow is explicit.

use std::collections::HashMap;
use std::fmt;

/// Unique identifier for variables in Core IR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

impl fmt::Display for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Constructor tag (numeric, for fast dispatch)
pub type Tag = u32;

/// Effect identifier
pub type EffectId = u32;

/// Operation identifier within an effect
pub type OpId = u32;

// ============================================================================
// Core IR Expressions
// ============================================================================

/// Core IR expression in A-Normal Form
#[derive(Debug, Clone)]
pub enum CoreExpr {
    /// Variable reference
    Var(VarId),

    /// Literal value (unboxed where possible)
    Lit(CoreLit),

    /// Let binding: let x = e1 in e2
    Let {
        name: VarId,
        /// Name hint for debugging/readability
        name_hint: Option<String>,
        value: Box<Atom>,
        body: Box<CoreExpr>,
    },

    /// Let binding with complex RHS
    LetExpr {
        name: VarId,
        name_hint: Option<String>,
        value: Box<CoreExpr>,
        body: Box<CoreExpr>,
    },

    /// Function application (non-tail position)
    App {
        func: VarId,
        args: Vec<VarId>,
    },

    /// Tail call (in tail position only)
    TailApp {
        func: VarId,
        args: Vec<VarId>,
    },

    /// Allocate a constructor/data value
    Alloc {
        tag: Tag,
        /// Type name for debugging
        type_name: Option<String>,
        /// Constructor name for debugging
        ctor_name: Option<String>,
        fields: Vec<VarId>,
    },

    /// Pattern match / case analysis
    Case {
        scrutinee: VarId,
        alts: Vec<Alt>,
        default: Option<Box<CoreExpr>>,
    },

    /// If expression (special case of Case for booleans)
    If {
        cond: VarId,
        then_branch: Box<CoreExpr>,
        else_branch: Box<CoreExpr>,
    },

    /// Lambda expression (will be lambda-lifted)
    Lam {
        params: Vec<VarId>,
        param_hints: Vec<Option<String>>,
        body: Box<CoreExpr>,
    },

    /// Recursive function definition
    LetRec {
        name: VarId,
        name_hint: Option<String>,
        params: Vec<VarId>,
        param_hints: Vec<Option<String>>,
        func_body: Box<CoreExpr>,
        body: Box<CoreExpr>,
    },

    /// Mutually recursive functions
    LetRecMutual {
        bindings: Vec<RecBinding>,
        body: Box<CoreExpr>,
    },

    /// Perform an effect operation
    Perform {
        effect: EffectId,
        effect_name: Option<String>,
        op: OpId,
        op_name: Option<String>,
        args: Vec<VarId>,
    },

    /// Handle effects
    Handle {
        body: Box<CoreExpr>,
        handler: Handler,
    },

    // =========================================================================
    // CPS-transformed expressions (produced by CPS transformation pass)
    // =========================================================================

    /// CPS-style function application
    /// The continuation receives the result of the function call
    AppCont {
        func: VarId,
        args: Vec<VarId>,
        /// Continuation to receive the result
        cont: VarId,
    },

    /// Resume a continuation with a value
    /// Used when returning from effectful code back to the continuation
    Resume {
        cont: VarId,
        value: VarId,
    },

    /// Capture continuation and invoke handler (CPS version of Perform)
    /// Captures the current continuation up to the nearest matching handler
    /// and invokes the handler's operation clause
    CaptureK {
        effect: EffectId,
        effect_name: Option<String>,
        op: OpId,
        op_name: Option<String>,
        args: Vec<VarId>,
        /// Current continuation to capture
        cont: VarId,
    },

    /// Install handler and run body (CPS version of Handle)
    /// Sets up a handler context, runs the body with that handler active,
    /// and handles effects/return
    WithHandler {
        effect: EffectId,
        effect_name: Option<String>,
        /// CPS handler with closures for return and operations
        handler: CPSHandler,
        /// The body to run with the handler active
        body: Box<CoreExpr>,
        /// Outer continuation to resume when handler completes
        outer_cont: VarId,
    },

    /// Sequence expressions (for effects/IO)
    Seq {
        first: Box<CoreExpr>,
        second: Box<CoreExpr>,
    },

    /// Return a value (explicit in CPS-like contexts)
    Return(VarId),

    /// Primitive operation (built-in, maps directly to C)
    PrimOp {
        op: PrimOp,
        args: Vec<VarId>,
    },

    /// External call (FFI)
    ExternCall {
        name: String,
        args: Vec<VarId>,
    },

    /// Dictionary method call (for polymorphic trait methods)
    /// Looks up method in the dictionary and calls it
    DictCall {
        dict: VarId,
        method: String,
        args: Vec<VarId>,
    },

    /// Concrete dictionary value for a specific trait instance
    /// e.g., Show_Int dictionary
    DictValue {
        trait_name: String,
        instance_ty: String, // mangled type name
    },

    /// Reference to a dictionary parameter
    DictRef(VarId),

    /// Project a field from a tuple/struct
    Proj {
        tuple: VarId,
        index: usize,
    },

    /// Error/unreachable
    Error(String),
}

/// Atomic expression (no further computation needed)
#[derive(Debug, Clone)]
pub enum Atom {
    /// Variable
    Var(VarId),
    /// Literal
    Lit(CoreLit),
    /// Allocate constructor
    Alloc {
        tag: Tag,
        type_name: Option<String>,
        ctor_name: Option<String>,
        fields: Vec<VarId>,
    },
    /// Primitive operation
    PrimOp { op: PrimOp, args: Vec<VarId> },
    /// Lambda
    Lam {
        params: Vec<VarId>,
        param_hints: Vec<Option<String>>,
        body: Box<CoreExpr>,
    },
    /// Function application
    App { func: VarId, args: Vec<VarId> },
}

/// Case alternative
#[derive(Debug, Clone)]
pub struct Alt {
    /// Constructor tag to match
    pub tag: Tag,
    /// Tag name for debugging
    pub tag_name: Option<String>,
    /// Binders for constructor fields
    pub binders: Vec<VarId>,
    pub binder_hints: Vec<Option<String>>,
    /// Optional guard condition (for patterns like [x] that need sub-pattern checks)
    pub guard: Option<Box<CoreExpr>>,
    /// Body expression
    pub body: CoreExpr,
}

/// Recursive binding
#[derive(Debug, Clone)]
pub struct RecBinding {
    pub name: VarId,
    pub name_hint: Option<String>,
    pub params: Vec<VarId>,
    pub param_hints: Vec<Option<String>>,
    pub body: CoreExpr,
}

/// Effect handler
#[derive(Debug, Clone)]
pub struct Handler {
    /// Effect being handled
    pub effect: EffectId,
    pub effect_name: Option<String>,
    /// Return clause
    pub return_var: VarId,
    pub return_body: Box<CoreExpr>,
    /// Operation handlers
    pub ops: Vec<OpHandler>,
}

/// Handler for a single effect operation
#[derive(Debug, Clone)]
pub struct OpHandler {
    pub op: OpId,
    pub op_name: Option<String>,
    /// Parameters for the operation
    pub params: Vec<VarId>,
    /// Continuation parameter
    pub cont: VarId,
    /// Handler body
    pub body: CoreExpr,
}

/// CPS-transformed effect handler
/// Contains closures for return clause and each operation
#[derive(Debug, Clone)]
pub struct CPSHandler {
    /// Return handler closure: (value, outer_k) -> result
    /// Called when the handled body returns normally
    pub return_handler: VarId,
    /// Operation handlers: each is (args..., k, outer_k) -> result
    pub op_handlers: Vec<CPSOpHandler>,
}

/// CPS-transformed operation handler
#[derive(Debug, Clone)]
pub struct CPSOpHandler {
    pub op: OpId,
    pub op_name: Option<String>,
    /// Handler closure VarId
    /// This closure takes: (op_args..., captured_k, outer_k) -> result
    pub handler_fn: VarId,
}

// ============================================================================
// Literals
// ============================================================================

#[derive(Debug, Clone)]
pub enum CoreLit {
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
    Bool(bool),
    Unit,
}

// ============================================================================
// Primitive Operations
// ============================================================================

/// Built-in primitive operations that map directly to C
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimOp {
    // Arithmetic (Int)
    IntAdd,
    IntSub,
    IntMul,
    IntDiv,
    IntMod,
    IntNeg,

    // Arithmetic (Float)
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatDiv,
    FloatNeg,

    // Comparison
    IntEq,
    IntNe,
    IntLt,
    IntLe,
    IntGt,
    IntGe,
    FloatEq,
    FloatNe,
    FloatLt,
    FloatLe,
    FloatGt,
    FloatGe,

    // Boolean
    BoolAnd,
    BoolOr,
    BoolNot,

    // String
    StringConcat,
    StringLength,
    StringEq,

    // Conversion
    IntToFloat,
    FloatToInt,
    IntToString,
    FloatToString,
    CharToInt,
    IntToChar,

    // Tuple/record access
    TupleGet(u32),  // Get nth element

    // List primitives
    ListCons,
    ListHead,
    ListTail,
    ListIsEmpty,

    // Tag checking (for pattern guards)
    TagEq(Tag), // Check if value's tag equals given tag
}

impl fmt::Display for PrimOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimOp::IntAdd => write!(f, "int_add"),
            PrimOp::IntSub => write!(f, "int_sub"),
            PrimOp::IntMul => write!(f, "int_mul"),
            PrimOp::IntDiv => write!(f, "int_div"),
            PrimOp::IntMod => write!(f, "int_mod"),
            PrimOp::IntNeg => write!(f, "int_neg"),
            PrimOp::FloatAdd => write!(f, "float_add"),
            PrimOp::FloatSub => write!(f, "float_sub"),
            PrimOp::FloatMul => write!(f, "float_mul"),
            PrimOp::FloatDiv => write!(f, "float_div"),
            PrimOp::FloatNeg => write!(f, "float_neg"),
            PrimOp::IntEq => write!(f, "int_eq"),
            PrimOp::IntNe => write!(f, "int_ne"),
            PrimOp::IntLt => write!(f, "int_lt"),
            PrimOp::IntLe => write!(f, "int_le"),
            PrimOp::IntGt => write!(f, "int_gt"),
            PrimOp::IntGe => write!(f, "int_ge"),
            PrimOp::FloatEq => write!(f, "float_eq"),
            PrimOp::FloatNe => write!(f, "float_ne"),
            PrimOp::FloatLt => write!(f, "float_lt"),
            PrimOp::FloatLe => write!(f, "float_le"),
            PrimOp::FloatGt => write!(f, "float_gt"),
            PrimOp::FloatGe => write!(f, "float_ge"),
            PrimOp::BoolAnd => write!(f, "bool_and"),
            PrimOp::BoolOr => write!(f, "bool_or"),
            PrimOp::BoolNot => write!(f, "bool_not"),
            PrimOp::StringConcat => write!(f, "string_concat"),
            PrimOp::StringLength => write!(f, "string_length"),
            PrimOp::StringEq => write!(f, "string_eq"),
            PrimOp::IntToFloat => write!(f, "int_to_float"),
            PrimOp::FloatToInt => write!(f, "float_to_int"),
            PrimOp::IntToString => write!(f, "int_to_string"),
            PrimOp::FloatToString => write!(f, "float_to_string"),
            PrimOp::CharToInt => write!(f, "char_to_int"),
            PrimOp::IntToChar => write!(f, "int_to_char"),
            PrimOp::TupleGet(n) => write!(f, "tuple_get_{}", n),
            PrimOp::ListCons => write!(f, "list_cons"),
            PrimOp::ListHead => write!(f, "list_head"),
            PrimOp::ListTail => write!(f, "list_tail"),
            PrimOp::ListIsEmpty => write!(f, "list_is_empty"),
            PrimOp::TagEq(tag) => write!(f, "tag_eq_{}", tag),
        }
    }
}

// ============================================================================
// Types in Core IR (for layout/size information)
// ============================================================================

/// Core type representation (simplified, for codegen)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreType {
    /// Unboxed integer (fits in a register)
    Int,
    /// Unboxed float
    Float,
    /// Unboxed boolean
    Bool,
    /// Unboxed character
    Char,
    /// Unit type (zero-size)
    Unit,
    /// Boxed value (pointer to heap object)
    Box(Box<CoreType>),
    /// Function type
    Fun {
        params: Vec<CoreType>,
        ret: Box<CoreType>,
    },
    /// Sum type (tagged union)
    Sum {
        name: String,
        variants: Vec<(Tag, String, Vec<CoreType>)>,
    },
    /// Product type (tuple)
    Tuple(Vec<CoreType>),
    /// Record type
    Record {
        name: String,
        fields: Vec<(String, CoreType)>,
    },
    /// String (heap-allocated)
    String,
    /// Type variable (should be resolved before codegen)
    Var(u32),
    /// Dictionary for type class constraints (runtime dispatch)
    Dict,
}

// ============================================================================
// Top-level definitions
// ============================================================================

/// A complete Core IR program
#[derive(Debug, Clone)]
pub struct CoreProgram {
    /// Type definitions
    pub types: Vec<TypeDef>,
    /// Top-level function definitions
    pub functions: Vec<FunDef>,
    /// Builtin function mappings (VarId -> builtin name)
    pub builtins: Vec<(VarId, String)>,
    /// Main expression
    pub main: Option<CoreExpr>,
}

/// Type definition
#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: String,
    pub params: Vec<String>,
    pub variants: Vec<VariantDef>,
}

/// Variant of a sum type
#[derive(Debug, Clone)]
pub struct VariantDef {
    pub tag: Tag,
    pub name: String,
    pub fields: Vec<CoreType>,
}

/// Top-level function definition
#[derive(Debug, Clone)]
pub struct FunDef {
    pub name: String,
    pub var_id: VarId,
    pub params: Vec<VarId>,
    pub param_hints: Vec<String>,
    pub param_types: Vec<CoreType>,
    pub return_type: CoreType,
    pub body: CoreExpr,
    /// Is this function tail-recursive?
    pub is_tail_recursive: bool,
}

// ============================================================================
// Variable generation
// ============================================================================

/// Generator for fresh variable IDs
#[derive(Debug, Clone)]
pub struct VarGen {
    next: u32,
}

impl VarGen {
    pub fn new() -> Self {
        VarGen { next: 0 }
    }

    /// Create a VarGen starting after a given ID
    pub fn starting_after(max_id: u32) -> Self {
        VarGen { next: max_id + 1 }
    }

    pub fn fresh(&mut self) -> VarId {
        let id = VarId(self.next);
        self.next += 1;
        id
    }

    /// Reset for a new scope (e.g., new function)
    pub fn reset(&mut self) {
        self.next = 0;
    }
}

impl Default for VarGen {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tag generation for constructors
// ============================================================================

/// Maps constructor names to tags
#[derive(Debug, Clone, Default)]
pub struct TagTable {
    /// (type_name, ctor_name) -> tag
    tags: HashMap<(String, String), Tag>,
    next_tag: Tag,
}

impl TagTable {
    pub fn new() -> Self {
        TagTable {
            tags: HashMap::new(),
            next_tag: 0,
        }
    }

    /// Get or create a tag for a constructor
    pub fn get_or_create(&mut self, type_name: &str, ctor_name: &str) -> Tag {
        let key = (type_name.to_string(), ctor_name.to_string());
        if let Some(&tag) = self.tags.get(&key) {
            tag
        } else {
            let tag = self.next_tag;
            self.next_tag += 1;
            self.tags.insert(key, tag);
            tag
        }
    }

    /// Get tag for a constructor (returns None if not registered)
    pub fn get(&self, type_name: &str, ctor_name: &str) -> Option<Tag> {
        self.tags.get(&(type_name.to_string(), ctor_name.to_string())).copied()
    }

    /// Register built-in tags
    pub fn register_builtins(&mut self) {
        // Bool
        self.get_or_create("Bool", "False");  // 0
        self.get_or_create("Bool", "True");   // 1

        // Option
        self.get_or_create("Option", "None"); // 2
        self.get_or_create("Option", "Some"); // 3

        // Result
        self.get_or_create("Result", "Ok");   // 4
        self.get_or_create("Result", "Err");  // 5

        // List
        self.get_or_create("List", "Nil");    // 6
        self.get_or_create("List", "Cons");   // 7
    }
}

// ============================================================================
// Pretty printing
// ============================================================================

impl CoreExpr {
    /// Pretty print the expression
    pub fn pretty(&self, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match self {
            CoreExpr::Var(v) => format!("{}", v),
            CoreExpr::Lit(lit) => format!("{:?}", lit),
            CoreExpr::Let { name, name_hint, value, body } => {
                let hint = name_hint.as_ref().map(|h| format!(" /* {} */", h)).unwrap_or_default();
                format!(
                    "let {}{} = {} in\n{}{}",
                    name, hint,
                    value.pretty(),
                    pad,
                    body.pretty(indent)
                )
            }
            CoreExpr::LetExpr { name, name_hint, value, body } => {
                let hint = name_hint.as_ref().map(|h| format!(" /* {} */", h)).unwrap_or_default();
                format!(
                    "let {}{} =\n{}  {} in\n{}{}",
                    name, hint,
                    pad,
                    value.pretty(indent + 1),
                    pad,
                    body.pretty(indent)
                )
            }
            CoreExpr::App { func, args } => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                format!("{}({})", func, args_str.join(", "))
            }
            CoreExpr::TailApp { func, args } => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                format!("TAIL {}({})", func, args_str.join(", "))
            }
            CoreExpr::Alloc { tag, ctor_name, fields, .. } => {
                let name = ctor_name.as_ref().map(|s| s.as_str()).unwrap_or("?");
                let fields_str: Vec<_> = fields.iter().map(|f| format!("{}", f)).collect();
                format!("ALLOC {}#{} [{}]", name, tag, fields_str.join(", "))
            }
            CoreExpr::Case { scrutinee, alts, default } => {
                let mut s = format!("case {} of\n", scrutinee);
                for alt in alts {
                    let binders: Vec<_> = alt.binders.iter().map(|b| format!("{}", b)).collect();
                    let tag_name = alt.tag_name.as_deref().unwrap_or("?");
                    s.push_str(&format!(
                        "{}  | {}#{} [{}] ->\n{}    {}\n",
                        pad, tag_name, alt.tag, binders.join(", "),
                        pad, alt.body.pretty(indent + 2)
                    ));
                }
                if let Some(def) = default {
                    s.push_str(&format!("{}  | _ -> {}\n", pad, def.pretty(indent + 2)));
                }
                s
            }
            CoreExpr::If { cond, then_branch, else_branch } => {
                format!(
                    "if {} then\n{}  {}\n{}else\n{}  {}",
                    cond,
                    pad, then_branch.pretty(indent + 1),
                    pad,
                    pad, else_branch.pretty(indent + 1)
                )
            }
            CoreExpr::Lam { params, body, .. } => {
                let params_str: Vec<_> = params.iter().map(|p| format!("{}", p)).collect();
                format!("fun ({}) ->\n{}  {}", params_str.join(", "), pad, body.pretty(indent + 1))
            }
            CoreExpr::LetRec { name, params, func_body, body, .. } => {
                let params_str: Vec<_> = params.iter().map(|p| format!("{}", p)).collect();
                format!(
                    "let rec {} ({}) =\n{}  {} in\n{}{}",
                    name, params_str.join(", "),
                    pad, func_body.pretty(indent + 1),
                    pad, body.pretty(indent)
                )
            }
            CoreExpr::PrimOp { op, args } => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                format!("PRIM {}({})", op, args_str.join(", "))
            }
            CoreExpr::Return(v) => format!("return {}", v),
            CoreExpr::Seq { first, second } => {
                format!("{};\n{}{}", first.pretty(indent), pad, second.pretty(indent))
            }
            CoreExpr::Error(msg) => format!("ERROR: {}", msg),

            // CPS-transformed expressions
            CoreExpr::AppCont { func, args, cont } => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                format!("APPCONT {}({}) -> {}", func, args_str.join(", "), cont)
            }
            CoreExpr::Resume { cont, value } => {
                format!("RESUME {} with {}", cont, value)
            }
            CoreExpr::CaptureK {
                effect_name,
                op_name,
                args,
                ..
            } => {
                let effect = effect_name.as_deref().unwrap_or("?");
                let op = op_name.as_deref().unwrap_or("?");
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                format!("CAPTURE_K {}.{}({})", effect, op, args_str.join(", "))
            }
            CoreExpr::WithHandler {
                effect_name,
                body,
                outer_cont,
                ..
            } => {
                let effect = effect_name.as_deref().unwrap_or("?");
                format!(
                    "WITH_HANDLER {} do\n{}  {}\n{}end -> {}",
                    effect,
                    pad,
                    body.pretty(indent + 1),
                    pad,
                    outer_cont
                )
            }

            _ => format!("{:?}", self),
        }
    }
}

impl Atom {
    pub fn pretty(&self) -> String {
        match self {
            Atom::Var(v) => format!("{}", v),
            Atom::Lit(lit) => format!("{:?}", lit),
            Atom::Alloc { tag, ctor_name, fields, .. } => {
                let name = ctor_name.as_ref().map(|s| s.as_str()).unwrap_or("?");
                let fields_str: Vec<_> = fields.iter().map(|f| format!("{}", f)).collect();
                format!("ALLOC {}#{} [{}]", name, tag, fields_str.join(", "))
            }
            Atom::PrimOp { op, args } => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                format!("PRIM {}({})", op, args_str.join(", "))
            }
            Atom::Lam { params, body, .. } => {
                let params_str: Vec<_> = params.iter().map(|p| format!("{}", p)).collect();
                format!("fun ({}) -> {}", params_str.join(", "), body.pretty(0))
            }
            Atom::App { func, args } => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                format!("{}({})", func, args_str.join(", "))
            }
        }
    }
}
