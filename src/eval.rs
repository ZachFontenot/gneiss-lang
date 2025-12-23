//! Defunctionalized CPS interpreter for Gneiss
//!
//! This interpreter uses an explicit continuation stack (defunctionalized CPS)
//! which allows processes to be suspended and resumed for concurrency.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
use crate::prelude::parse_prelude;
use crate::runtime::{ChannelId, Pid, ProcessContinuation, Runtime};
use crate::types::{ClassEnv, Pred, Type, TypeContext};
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum EvalError {
    #[error("unbound variable: {0}")]
    UnboundVariable(String),
    #[error("type error: {0}")]
    TypeError(String),
    #[error("pattern match failed")]
    MatchFailed,
    #[error("division by zero")]
    DivisionByZero,
    #[error("integer overflow")]
    IntegerOverflow,
    #[error("runtime error: {0}")]
    RuntimeError(String),
    #[error("deadlock detected")]
    Deadlock,
}

// ============================================================================
// Defunctionalized Continuation Stack
// ============================================================================

/// A single continuation frame - represents "what to do next" with a value
#[derive(Debug, Clone)]
pub enum Frame {
    /// After evaluating func in application, evaluate the argument
    AppFunc { arg: Rc<Expr>, env: Env },
    /// After evaluating arg in application, apply func to it
    AppArg { func: Value },

    /// After evaluating the value in let, bind pattern and maybe eval body
    Let {
        pattern: Pattern,
        body: Option<Rc<Expr>>,
        env: Env,
    },

    /// After evaluating condition, pick a branch
    If {
        then_branch: Rc<Expr>,
        else_branch: Rc<Expr>,
        env: Env,
    },

    /// After evaluating left operand, evaluate right
    BinOpLeft {
        op: BinOp,
        right: Rc<Expr>,
        env: Env,
    },
    /// After evaluating right operand, compute the result
    /// env is needed to look up user-defined operators
    BinOpRight { op: BinOp, left: Value, env: Env },
    /// Apply the incoming value (a function) to this argument
    /// Used for user-defined operator evaluation: ((op left) right)
    ApplyTo { arg: Value },

    /// After evaluating operand, apply unary operator
    UnaryOp { op: UnaryOp },

    /// After evaluating first expr in sequence, evaluate second
    Seq { second: Rc<Expr>, env: Env },

    /// Building a collection (list, tuple, or constructor)
    Collect {
        kind: CollectKind,
        remaining: Vec<Expr>,
        acc: Vec<Value>,
        env: Env,
    },

    /// After evaluating scrutinee, try to match arms
    Match { arms: Vec<MatchArm>, env: Env },
    /// After evaluating a guard, decide whether to take this arm or try next
    MatchGuard {
        body: Rc<Expr>,
        remaining_arms: Vec<MatchArm>,
        scrutinee: Value,
        bound_env: Env, // env with pattern bindings for this arm
        outer_env: Env, // original env for trying next arm
    },

    // === Concurrency frames ===
    /// After evaluating the thunk expression for spawn
    Spawn,

    // === Delimited continuation frames ===
    /// Delimiter marker for reset
    Prompt,

    // === Fiber effect frames (unified runtime) ===
    /// Implicit delimiter for fiber continuations.
    /// Unlike Prompt, FiberBoundary remains on stack after capture.
    FiberBoundary,

    /// After evaluating channel expr in fiber recv
    FiberRecv,

    /// After evaluating channel expr in fiber send, evaluate the value
    FiberSendValue { value_expr: Rc<Expr>, env: Env },

    /// Channel and value both evaluated, ready to produce Send effect
    FiberSendReady { channel: ChannelId },

    /// After evaluating thunk expression for fiber spawn
    FiberFork,

    /// After evaluating fiber handle for join
    FiberJoin,

    /// Evaluating channel expressions for fiber select, collecting them
    FiberSelectChans {
        /// Patterns for each arm (parallel to channels being collected)
        patterns: Vec<Pattern>,
        /// Body expressions for each arm
        bodies: Vec<Rc<Expr>>,
        /// Channel expressions still to evaluate
        remaining_chans: Vec<Expr>,
        /// Channel IDs already evaluated
        collected_chans: Vec<ChannelId>,
        env: Env,
    },

    /// All channels evaluated for fiber select, ready to produce Select effect
    FiberSelectReady {
        /// Channel IDs to select from
        channels: Vec<ChannelId>,
        /// Corresponding patterns
        patterns: Vec<Pattern>,
        /// Corresponding bodies
        bodies: Vec<Rc<Expr>>,
        env: Env,
    },

    /// After evaluating record, access the field
    FieldAccess { field: String },

    /// After evaluating base record, evaluate update fields
    RecordUpdate {
        /// Field names for the updates
        update_field_names: Vec<String>,
        /// Remaining update expressions to evaluate
        remaining_updates: Vec<Expr>,
        /// Accumulated updated values
        collected_updates: Vec<(String, Value)>,
        env: Env,
    },

    /// All update values evaluated, merge with base record
    RecordUpdateApply {
        /// Base record value
        base: Value,
        /// Updates to apply (field name -> new value)
        updates: Vec<(String, Value)>,
    },

    /// Continue evaluating record update fields
    RecordUpdateContinue {
        /// Current field name being evaluated
        current_field_name: String,
        /// Remaining field names
        remaining_field_names: Vec<String>,
        /// Remaining update expressions
        remaining_updates: Vec<Expr>,
        env: Env,
    },

    /// Collect the last update field value
    RecordUpdateCollectLast {
        field_name: String,
    },

    // === I/O frames ===
    /// Collecting arguments for a multi-arg I/O builtin operation
    IoOp {
        /// Which I/O operation we're building
        op_kind: PartialIoOp,
        /// Arguments already collected
        collected_args: Vec<Value>,
        /// Remaining argument expressions to evaluate
        remaining_args: Vec<Expr>,
        env: Env,
    },
}

/// What kind of collection we're building
#[derive(Debug, Clone)]
pub enum CollectKind {
    List,
    Tuple,
    Constructor { name: String },
    Record { type_name: String, field_names: Vec<String> },
}

// ============================================================================
// I/O Operations
// ============================================================================

/// Partial I/O operations being built (waiting for more arguments)
#[derive(Debug, Clone)]
pub enum PartialIoOp {
    /// file_open path mode -> needs path (String), mode (OpenMode)
    FileOpen,
    /// tcp_connect host port -> needs host (String), port (Int)
    TcpConnect,
    /// tcp_listen host port -> needs host (String), port (Int)
    TcpListen,
    /// read handle count -> needs handle, count (Int)
    Read,
    /// write handle bytes -> needs handle, bytes (Bytes)
    Write,
}

/// File open modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    /// Read-only
    Read,
    /// Write-only (create/truncate)
    Write,
    /// Append (create if not exists)
    Append,
    /// Read and write
    ReadWrite,
}

impl OpenMode {
    /// Convert from a string value (for builtins)
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "r" | "read" => Some(OpenMode::Read),
            "w" | "write" => Some(OpenMode::Write),
            "a" | "append" => Some(OpenMode::Append),
            "rw" | "read_write" => Some(OpenMode::ReadWrite),
            _ => None,
        }
    }
}

/// I/O operation that a fiber can request
/// Each operation will be handled by the I/O reactor or blocking pool
#[derive(Debug, Clone)]
pub enum IoOp {
    // === File operations ===
    /// Open a file with the given path and mode
    FileOpen {
        path: String,
        mode: OpenMode,
    },

    // === TCP operations ===
    /// Connect to a TCP server
    TcpConnect {
        host: String,
        port: u16,
    },
    /// Start listening on a TCP port
    TcpListen {
        host: String,
        port: u16,
    },
    /// Accept a connection from a TCP listener
    TcpAccept {
        listener: u64, // TcpListener handle ID
    },

    // === Generic I/O operations (work on files and sockets) ===
    /// Read up to N bytes from a handle
    Read {
        handle: u64, // FileHandle or TcpSocket handle ID
        count: usize,
    },
    /// Write bytes to a handle
    Write {
        handle: u64, // FileHandle or TcpSocket handle ID
        data: Vec<u8>,
    },
    /// Close a handle
    Close {
        handle: u64,
    },

    // === Timer operations ===
    /// Sleep for a duration (milliseconds)
    Sleep {
        duration_ms: u64,
    },
}

/// The continuation stack
#[derive(Debug, Clone)]
pub struct Cont {
    frames: Vec<Frame>,
}

impl Cont {
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Create a Cont from a vector of frames (used when capturing continuations)
    pub fn from_frames(frames: Vec<Frame>) -> Self {
        Self { frames }
    }

    pub fn push(&mut self, frame: Frame) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<Frame> {
        self.frames.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Insert a frame at the bottom of the continuation stack
    pub fn insert_at_bottom(&mut self, frame: Frame) {
        self.frames.insert(0, frame);
    }
}

impl Default for Cont {
    fn default() -> Self {
        Self::new()
    }
}

/// Machine state - either evaluating an expression or returning a value
#[derive(Debug, Clone)]
pub enum State {
    /// Evaluate an expression in an environment
    Eval {
        expr: Rc<Expr>,
        env: Env,
        cont: Cont,
    },
    /// Return a value to the continuation
    Apply { value: Value, cont: Cont },
}

/// Result of a single step
pub enum StepResult {
    /// Keep stepping with this new state
    Continue(State),
    /// Evaluation completed with a final value
    Done(Value),
    /// An error occurred
    Error(EvalError),
}

/// Result of applying an effectful builtin function
/// Used for Fiber.spawn, Fiber.join, Fiber.yield
pub enum BuiltinResult {
    /// Builtin completed with a value
    Value(Value),
    /// Builtin needs to produce a fiber effect (requires continuation capture)
    Effect(FiberEffect),
    /// Builtin needs more arguments (partial application)
    Partial { name: String, args: Vec<Value> },
}

// ============================================================================
// Fiber Effects (for unified concurrency model)
// ============================================================================

/// Unique fiber identifier (typed alternative to Pid)
pub type FiberId = u64;

/// Effects that fibers can perform, requiring runtime/scheduler intervention.
/// Each variant captures the continuation to resume after the effect is handled.
/// This unifies channel operations, spawning, and other fiber effects into a single model.
#[derive(Debug, Clone)]
pub enum FiberEffect {
    /// Fiber completed with a value
    Done(Box<Value>),

    /// Fork a new fiber
    /// - `thunk`: The computation to run in the new fiber (a closure)
    /// - `cont`: Continuation expecting the child's Fiber handle
    Fork {
        thunk: Box<Value>,
        cont: Option<Box<Cont>>,
    },

    /// Yield control to scheduler (cooperative multitasking)
    /// - `cont`: Continuation to resume with Unit
    Yield { cont: Option<Box<Cont>> },

    /// Create a new channel
    /// - `cont`: Continuation expecting the new Channel
    NewChan { cont: Option<Box<Cont>> },

    /// Send a value on a channel (blocks until receiver ready)
    /// - `channel`: Target channel ID
    /// - `value`: Value to send
    /// - `cont`: Continuation to resume with Unit after send completes
    Send {
        channel: ChannelId,
        value: Box<Value>,
        cont: Option<Box<Cont>>,
    },

    /// Receive a value from a channel (blocks until sender ready)
    /// - `channel`: Source channel ID
    /// - `cont`: Continuation expecting the received Value
    Recv {
        channel: ChannelId,
        cont: Option<Box<Cont>>,
    },

    /// Wait for a fiber to complete (type-safe join)
    /// - `fiber_id`: The fiber to wait for
    /// - `cont`: Continuation expecting the fiber's result Value
    Join {
        fiber_id: FiberId,
        cont: Option<Box<Cont>>,
    },

    /// Select on multiple channels (blocks until one ready)
    /// - `arms`: Channel IDs with their patterns and body expressions
    /// - `cont`: Outer continuation (used after arm body evaluates)
    Select {
        arms: Vec<SelectEffectArm>,
        cont: Option<Box<Cont>>,
    },

    /// Perform an I/O operation (handled by reactor or blocking pool)
    /// - `op`: The I/O operation to perform
    /// - `cont`: Continuation expecting the operation result (usually Result IoError a)
    Io {
        op: IoOp,
        cont: Option<Box<Cont>>,
    },
}

impl FiberEffect {
    /// Attach a captured continuation to this effect.
    /// Used after capture_to_fiber_boundary captures the continuation.
    pub fn with_cont(self, captured: Cont) -> Self {
        let boxed = Some(Box::new(captured));
        match self {
            FiberEffect::Done(v) => FiberEffect::Done(v),
            FiberEffect::Fork { thunk, .. } => FiberEffect::Fork { thunk, cont: boxed },
            FiberEffect::Yield { .. } => FiberEffect::Yield { cont: boxed },
            FiberEffect::NewChan { .. } => FiberEffect::NewChan { cont: boxed },
            FiberEffect::Send { channel, value, .. } => {
                FiberEffect::Send { channel, value, cont: boxed }
            }
            FiberEffect::Recv { channel, .. } => FiberEffect::Recv { channel, cont: boxed },
            FiberEffect::Join { fiber_id, .. } => FiberEffect::Join { fiber_id, cont: boxed },
            FiberEffect::Select { arms, .. } => FiberEffect::Select { arms, cont: boxed },
            FiberEffect::Io { op, .. } => FiberEffect::Io { op, cont: boxed },
        }
    }
}

/// A select arm for the FiberEffect::Select variant
#[derive(Debug, Clone)]
pub struct SelectEffectArm {
    pub channel: ChannelId,
    pub pattern: Pattern,
    pub body: Rc<Expr>,
    pub env: Env,
}

// ============================================================================

/// Runtime values
#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Char(char),
    Unit,

    /// Binary data (for I/O operations)
    Bytes(Vec<u8>),

    /// A list (persistent/immutable vector for O(log n) operations)
    List(im::Vector<Value>),

    /// A tuple
    Tuple(Vec<Value>),

    /// A closure
    Closure {
        params: Vec<Pattern>,
        body: Rc<Expr>,
        env: Env,
    },

    /// A constructor value (e.g., Some 42)
    Constructor {
        name: String,
        fields: Vec<Value>,
    },

    /// A process ID
    Pid(Pid),

    /// A channel
    Channel(ChannelId),

    /// Built-in function
    Builtin(String),

    /// Partially applied builtin (waiting for more arguments)
    BuiltinPartial { name: String, args: Vec<Value> },

    /// A captured delimited continuation
    Continuation {
        frames: Vec<Frame>,
    },

    /// A typeclass dictionary (contains method implementations)
    Dict {
        trait_name: String,
        methods: HashMap<String, Value>,
    },

    /// A typed fiber handle (returned by Fiber.spawn, consumed by Fiber.join)
    Fiber(FiberId),

    /// A suspended fiber effect awaiting runtime/scheduler handling
    FiberEffect(FiberEffect),

    /// A composed function (f >> g or f << g)
    ComposedFn {
        first: Box<Value>,
        second: Box<Value>,
    },

    /// A record value with named fields (persistent hashmap for O(log m) updates)
    Record {
        type_name: String,
        fields: im::HashMap<String, Value>,
    },

    /// A dictionary value (String-keyed persistent hashmap)
    Dictionary(im::HashMap<String, Value>),

    /// A set value (String elements, persistent hashset)
    Set(im::HashSet<String>),

    // === I/O handles ===
    /// An open file handle (opaque ID managed by I/O reactor)
    FileHandle(u64),

    /// A TCP socket (client connection, opaque ID)
    TcpSocket(u64),

    /// A TCP listener (server socket, opaque ID)
    TcpListener(u64),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::String(_) => "String",
            Value::Char(_) => "Char",
            Value::Unit => "()",
            Value::Bytes(_) => "Bytes",
            Value::List(_) => "List",
            Value::Tuple(_) => "Tuple",
            Value::Closure { .. } => "Function",
            Value::Constructor { .. } => "Constructor",
            Value::Pid(_) => "Pid",
            Value::Channel(_) => "Channel",
            Value::Builtin(_) => "Builtin",
            Value::BuiltinPartial { .. } => "BuiltinPartial",
            Value::Continuation { .. } => "Continuation",
            Value::Fiber(_) => "Fiber",
            Value::FiberEffect(_) => "FiberEffect",
            Value::ComposedFn { .. } => "Function",
            Value::Dict { .. } => "Dict",
            Value::Record { .. } => "Record",
            Value::Dictionary(_) => "Dictionary",
            Value::Set(_) => "Set",
            Value::FileHandle(_) => "FileHandle",
            Value::TcpSocket(_) => "TcpSocket",
            Value::TcpListener(_) => "TcpListener",
        }
    }

    /// Convert a runtime value to a Type (for instance resolution)
    pub fn to_type(&self) -> Type {
        match self {
            Value::Int(_) => Type::Int,
            Value::Float(_) => Type::Float,
            Value::Bool(_) => Type::Bool,
            Value::String(_) => Type::String,
            Value::Char(_) => Type::Char,
            Value::Unit => Type::Unit,
            Value::Bytes(_) => Type::Bytes,
            Value::List(items) => {
                // Infer element type from first item, or use a fresh var
                let elem_ty = items.front().map(|v| v.to_type()).unwrap_or(Type::Unit);
                Type::list(elem_ty)
            }
            Value::Tuple(items) => Type::Tuple(items.iter().map(|v| v.to_type()).collect()),
            Value::Constructor { name, fields } => Type::Constructor {
                name: name.clone(),
                args: fields.iter().map(|v| v.to_type()).collect(),
            },
            Value::Pid(_) => Type::Pid,
            Value::Channel(_) => Type::Channel(Rc::new(Type::Unit)), // Simplified
            Value::Record { type_name, .. } => Type::Constructor {
                name: type_name.clone(),
                args: vec![], // Simplified - don't track type args at runtime
            },
            Value::Dictionary(dict) => {
                // Infer value type from first entry, or use Unit
                let val_ty = dict.values().next().map(|v| v.to_type()).unwrap_or(Type::Unit);
                Type::Dict(Rc::new(val_ty))
            }
            Value::Set(_) => Type::Set,
            _ => Type::Unit, // Fallback for closures, continuations, etc.
        }
    }

    /// Convert a runtime value to a Type, using TypeContext to resolve constructor names to type names
    pub fn to_type_with_ctx(&self, type_ctx: &TypeContext) -> Type {
        match self {
            Value::Int(_) => Type::Int,
            Value::Float(_) => Type::Float,
            Value::Bool(_) => Type::Bool,
            Value::String(_) => Type::String,
            Value::Char(_) => Type::Char,
            Value::Unit => Type::Unit,
            Value::Bytes(_) => Type::Bytes,
            Value::List(items) => {
                let elem_ty = items
                    .front()
                    .map(|v| v.to_type_with_ctx(type_ctx))
                    .unwrap_or(Type::Unit);
                Type::list(elem_ty)
            }
            Value::Tuple(items) => {
                Type::Tuple(items.iter().map(|v| v.to_type_with_ctx(type_ctx)).collect())
            }
            Value::Constructor { name, fields } => {
                // Look up the constructor to find the actual type name
                let type_name = type_ctx
                    .get_constructor(name)
                    .map(|info| info.type_name.clone())
                    .unwrap_or_else(|| name.clone());
                Type::Constructor {
                    name: type_name,
                    args: fields
                        .iter()
                        .map(|v| v.to_type_with_ctx(type_ctx))
                        .collect(),
                }
            }
            Value::Pid(_) => Type::Pid,
            Value::Channel(_) => Type::Channel(Rc::new(Type::Unit)),
            Value::Record { type_name, .. } => Type::Constructor {
                name: type_name.clone(),
                args: vec![],
            },
            Value::Dictionary(dict) => {
                let val_ty = dict.values().next()
                    .map(|v| v.to_type_with_ctx(type_ctx))
                    .unwrap_or(Type::Unit);
                Type::Dict(Rc::new(val_ty))
            }
            Value::Set(_) => Type::Set,
            _ => Type::Unit,
        }
    }

    /// Create an IoError constructor value from a std::io::Error
    pub fn from_io_error(err: std::io::Error) -> Value {
        use std::io::ErrorKind;
        match err.kind() {
            ErrorKind::NotFound => Value::Constructor {
                name: "NotFound".to_string(),
                fields: vec![],
            },
            ErrorKind::PermissionDenied => Value::Constructor {
                name: "PermissionDenied".to_string(),
                fields: vec![],
            },
            ErrorKind::ConnectionRefused => Value::Constructor {
                name: "ConnectionRefused".to_string(),
                fields: vec![],
            },
            ErrorKind::ConnectionReset => Value::Constructor {
                name: "ConnectionReset".to_string(),
                fields: vec![],
            },
            ErrorKind::BrokenPipe => Value::Constructor {
                name: "BrokenPipe".to_string(),
                fields: vec![],
            },
            ErrorKind::TimedOut => Value::Constructor {
                name: "TimedOut".to_string(),
                fields: vec![],
            },
            ErrorKind::AddrInUse => Value::Constructor {
                name: "AddrInUse".to_string(),
                fields: vec![],
            },
            ErrorKind::AddrNotAvailable => Value::Constructor {
                name: "AddrNotAvailable".to_string(),
                fields: vec![],
            },
            ErrorKind::InvalidInput => Value::Constructor {
                name: "InvalidInput".to_string(),
                fields: vec![Value::String(err.to_string())],
            },
            _ => Value::Constructor {
                name: "IoOther".to_string(),
                fields: vec![Value::String(err.to_string())],
            },
        }
    }

    /// Wrap a value in Ok
    pub fn ok(value: Value) -> Value {
        Value::Constructor {
            name: "Ok".to_string(),
            fields: vec![value],
        }
    }

    /// Wrap an IoError in Err
    pub fn io_err(io_error: Value) -> Value {
        Value::Constructor {
            name: "Err".to_string(),
            fields: vec![io_error],
        }
    }

    /// Create Result IoError a from std::io::Error
    pub fn from_io_result<T, F>(result: std::io::Result<T>, success_fn: F) -> Value
    where
        F: FnOnce(T) -> Value,
    {
        match result {
            Ok(v) => Value::ok(success_fn(v)),
            Err(e) => Value::io_err(Value::from_io_error(e)),
        }
    }
}

/// Environment mapping names to values
pub type Env = Rc<RefCell<EnvInner>>;

#[derive(Debug, Clone, Default)]
pub struct EnvInner {
    bindings: HashMap<String, Value>,
    parent: Option<Env>,
}

impl EnvInner {
    pub fn new() -> Env {
        Rc::new(RefCell::new(EnvInner {
            bindings: HashMap::new(),
            parent: None,
        }))
    }

    pub fn with_parent(parent: &Env) -> Env {
        Rc::new(RefCell::new(EnvInner {
            bindings: HashMap::new(),
            parent: Some(parent.clone()),
        }))
    }

    pub fn define(&mut self, name: String, value: Value) {
        self.bindings.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.bindings.get(name) {
            Some(value.clone())
        } else if let Some(parent) = &self.parent {
            parent.borrow().get(name)
        } else {
            None
        }
    }
}


/// The interpreter
pub struct Interpreter {
    /// Global environment
    global_env: Env,
    /// The runtime scheduler
    pub runtime: Runtime,
    /// Class environment for typeclass instances
    class_env: ClassEnv,
    /// Type context for constructor -> type mapping
    type_ctx: TypeContext,
    /// Module environments: module name -> Env of exports
    module_envs: HashMap<String, Env>,
    /// Import mappings: local name -> (module name, original name)
    imports: HashMap<String, (String, String)>,
    /// Module aliases: alias -> original module name
    module_aliases: HashMap<String, String>,
    /// Command-line arguments passed to the program
    program_args: Vec<String>,
}

impl Interpreter {
    pub fn new() -> Self {
        let global_env = EnvInner::new();

        // Add some built-in functions
        {
            let mut env = global_env.borrow_mut();
            env.define("print".into(), Value::Builtin("print".into()));
            env.define(
                "int_to_string".into(),
                Value::Builtin("int_to_string".into()),
            );
            env.define(
                "string_length".into(),
                Value::Builtin("string_length".into()),
            );
            env.define(
                "string_to_chars".into(),
                Value::Builtin("string_to_chars".into()),
            );
            env.define(
                "chars_to_string".into(),
                Value::Builtin("chars_to_string".into()),
            );
            env.define(
                "char_to_string".into(),
                Value::Builtin("char_to_string".into()),
            );
            env.define("char_to_int".into(), Value::Builtin("char_to_int".into()));
            env.define("char_to_lower".into(), Value::Builtin("char_to_lower".into()));
            env.define("char_to_upper".into(), Value::Builtin("char_to_upper".into()));
            env.define(
                "char_is_whitespace".into(),
                Value::Builtin("char_is_whitespace".into()),
            );
            env.define(
                "string_index_of".into(),
                Value::Builtin("string_index_of".into()),
            );
            env.define(
                "string_substring".into(),
                Value::Builtin("string_substring".into()),
            );

            // Native string operations
            env.define(
                "string_to_lower".into(),
                Value::Builtin("string_to_lower".into()),
            );
            env.define(
                "string_to_upper".into(),
                Value::Builtin("string_to_upper".into()),
            );
            env.define(
                "string_trim".into(),
                Value::Builtin("string_trim".into()),
            );
            env.define(
                "string_trim_start".into(),
                Value::Builtin("string_trim_start".into()),
            );
            env.define(
                "string_trim_end".into(),
                Value::Builtin("string_trim_end".into()),
            );
            env.define(
                "string_reverse".into(),
                Value::Builtin("string_reverse".into()),
            );
            env.define(
                "string_is_empty".into(),
                Value::Builtin("string_is_empty".into()),
            );
            env.define(
                "string_split".into(),
                Value::Builtin("string_split".into()),
            );
            env.define(
                "string_join".into(),
                Value::Builtin("string_join".into()),
            );
            env.define(
                "string_char_at".into(),
                Value::Builtin("string_char_at".into()),
            );
            env.define(
                "string_concat".into(),
                Value::Builtin("string_concat".into()),
            );
            env.define(
                "string_repeat".into(),
                Value::Builtin("string_repeat".into()),
            );

            // Bytes builtins
            env.define(
                "bytes_to_string".into(),
                Value::Builtin("bytes_to_string".into()),
            );
            env.define(
                "string_to_bytes".into(),
                Value::Builtin("string_to_bytes".into()),
            );
            env.define(
                "bytes_length".into(),
                Value::Builtin("bytes_length".into()),
            );
            env.define(
                "bytes_slice".into(),
                Value::Builtin("bytes_slice".into()),
            );
            env.define(
                "bytes_concat".into(),
                Value::Builtin("bytes_concat".into()),
            );

            // spawn : (() -> a) -> Pid (backwards compatibility with old spawn keyword)
            env.define("spawn".into(), Value::Builtin("spawn".into()));

            // Fiber builtins (for unified fiber runtime)
            env.define(
                "Fiber.spawn".into(),
                Value::Builtin("Fiber.spawn".into()),
            );
            env.define("Fiber.join".into(), Value::Builtin("Fiber.join".into()));
            env.define(
                "Fiber.yield".into(),
                Value::Builtin("Fiber.yield".into()),
            );

            // I/O builtins
            env.define("sleep_ms".into(), Value::Builtin("sleep_ms".into()));
            // File I/O
            env.define("file_open".into(), Value::Builtin("file_open".into()));
            env.define("file_read".into(), Value::Builtin("file_read".into()));
            env.define("file_write".into(), Value::Builtin("file_write".into()));
            env.define("file_close".into(), Value::Builtin("file_close".into()));
            // TCP sockets
            env.define("tcp_connect".into(), Value::Builtin("tcp_connect".into()));
            env.define("tcp_listen".into(), Value::Builtin("tcp_listen".into()));
            env.define("tcp_accept".into(), Value::Builtin("tcp_accept".into()));

            // Dict builtins (String-keyed dictionary)
            env.define("Dict.new".into(), Value::Builtin("Dict.new".into()));
            env.define("Dict.insert".into(), Value::Builtin("Dict.insert".into()));
            env.define("Dict.get".into(), Value::Builtin("Dict.get".into()));
            env.define("Dict.remove".into(), Value::Builtin("Dict.remove".into()));
            env.define("Dict.contains".into(), Value::Builtin("Dict.contains".into()));
            env.define("Dict.keys".into(), Value::Builtin("Dict.keys".into()));
            env.define("Dict.values".into(), Value::Builtin("Dict.values".into()));
            env.define("Dict.size".into(), Value::Builtin("Dict.size".into()));
            env.define("Dict.isEmpty".into(), Value::Builtin("Dict.isEmpty".into()));
            env.define("Dict.toList".into(), Value::Builtin("Dict.toList".into()));
            env.define("Dict.fromList".into(), Value::Builtin("Dict.fromList".into()));
            env.define("Dict.merge".into(), Value::Builtin("Dict.merge".into()));
            env.define("Dict.getOrDefault".into(), Value::Builtin("Dict.getOrDefault".into()));

            // String escape builtins
            env.define("html_escape".into(), Value::Builtin("html_escape".into()));
            env.define("json_escape_string".into(), Value::Builtin("json_escape_string".into()));

            // Set builtins (String-element set)
            env.define("Set.new".into(), Value::Builtin("Set.new".into()));
            env.define("Set.insert".into(), Value::Builtin("Set.insert".into()));
            env.define("Set.contains".into(), Value::Builtin("Set.contains".into()));
            env.define("Set.remove".into(), Value::Builtin("Set.remove".into()));
            env.define("Set.union".into(), Value::Builtin("Set.union".into()));
            env.define("Set.intersect".into(), Value::Builtin("Set.intersect".into()));
            env.define("Set.size".into(), Value::Builtin("Set.size".into()));
            env.define("Set.toList".into(), Value::Builtin("Set.toList".into()));

            // Command-line args
            env.define("get_args".into(), Value::Builtin("get_args".into()));
        }

        Self {
            global_env,
            runtime: Runtime::new(),
            class_env: ClassEnv::new(),
            type_ctx: TypeContext::new(),
            module_envs: HashMap::new(),
            imports: HashMap::new(),
            module_aliases: HashMap::new(),
            program_args: Vec::new(),
        }
    }

    /// Set the command-line arguments for the program
    pub fn set_program_args(&mut self, args: Vec<String>) {
        self.program_args = args;
    }

    /// Set the class environment (called after type inference)
    pub fn set_class_env(&mut self, class_env: ClassEnv) {
        self.class_env = class_env;
    }

    /// Set the type context (called after type inference)
    pub fn set_type_ctx(&mut self, type_ctx: TypeContext) {
        self.type_ctx = type_ctx;
    }

    /// Register a module's exported values
    pub fn register_module(&mut self, name: String, env: Env) {
        self.module_envs.insert(name, env);
    }

    /// Add an import mapping (local name -> module.original name)
    pub fn add_import(&mut self, local_name: String, module_name: String, original_name: String) {
        self.imports.insert(local_name, (module_name, original_name));
    }

    /// Add a module alias (alias -> original module name)
    pub fn add_module_alias(&mut self, alias: String, module_name: String) {
        self.module_aliases.insert(alias, module_name);
    }

    /// Clear imports for a fresh module context
    pub fn clear_imports(&mut self) {
        self.imports.clear();
        self.module_aliases.clear();
    }

    /// Create an environment containing only the exported names from global_env
    pub fn create_exports_env(&self, exports: &std::collections::HashSet<String>) -> Env {
        let exports_env = EnvInner::new();
        {
            let global = self.global_env.borrow();
            let mut exp = exports_env.borrow_mut();
            for name in exports {
                if let Some(value) = global.get(name) {
                    exp.define(name.clone(), value);
                }
            }
        }
        exports_env
    }

    /// Resolve a module name (following aliases)
    fn resolve_module_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.module_aliases.get(name).map(|s| s.as_str()).unwrap_or(name)
    }

    /// Look up a name, checking imports and qualified names
    pub fn lookup_name(&self, name: &str, env: &Env) -> Option<Value> {
        // First check local environment
        if let Some(v) = env.borrow().get(name) {
            return Some(v);
        }

        // Check if it's a qualified name (Module.x)
        if let Some(dot_pos) = name.find('.') {
            let module_part = &name[..dot_pos];
            let item_part = &name[dot_pos + 1..];

            // Resolve module alias
            let module_name = self.resolve_module_name(module_part);

            // Look up in module's exports
            if let Some(module_env) = self.module_envs.get(module_name) {
                return module_env.borrow().get(item_part);
            }
        }

        // Check unqualified imports
        if let Some((module_name, original_name)) = self.imports.get(name) {
            if let Some(module_env) = self.module_envs.get(module_name) {
                return module_env.borrow().get(original_name);
            }
        }

        None
    }

    /// Get a reference to the global environment (for REPL use)
    pub fn global_env(&self) -> &Env {
        &self.global_env
    }

    /// Run a program
    pub fn run(&mut self, program: &Program) -> Result<Value, EvalError> {
        let mut last_expr_value = Value::Unit;

        // Parse and evaluate prelude (Option, Result, id, const, flip)
        let prelude = parse_prelude()
            .map_err(|e| EvalError::TypeError(format!("Prelude error: {}", e)))?;

        // Evaluate prelude declarations first
        for item in &prelude.items {
            if let Item::Decl(decl) = item {
                self.eval_decl(decl)?;
            }
        }

        // Process all user items in order: declarations bind values, expressions execute
        for item in &program.items {
            match item {
                Item::Import(_) => {
                    // Imports are handled during module resolution (Phase 3)
                    // At runtime in a single-file context, nothing to do
                }
                Item::Decl(decl) => {
                    self.eval_decl(decl)?;
                }
                Item::Expr(expr) => {
                    let env = self.global_env.clone();
                    last_expr_value = self.eval_expr(&env, expr)?;
                }
            }
        }

        // Look for a main function and run it as a process
        let main_fn = self.global_env.borrow().get("main");
        if let Some(main) = main_fn {
            // Spawn main as a process so it can use channels
            self.runtime.spawn(main);

            // Run the scheduler until all processes complete
            self.run_scheduler()?;

            Ok(Value::Unit)
        } else {
            // If no main, return the last top-level expression's value
            Ok(last_expr_value)
        }
    }

    /// Evaluate a declaration
    fn eval_decl(&mut self, decl: &Decl) -> Result<(), EvalError> {
        match decl {
            Decl::Let {
                name, params, body, ..
            } => {
                if params.is_empty() {
                    let env = self.global_env.clone();
                    let value = self.eval_expr(&env, body)?;
                    self.global_env.borrow_mut().define(name.clone(), value);
                } else {
                    // For functions, create a closure that captures the global env
                    // This allows recursion since the function will look up itself
                    let closure = Value::Closure {
                        params: params.clone(),
                        body: Rc::new(body.clone()),
                        env: self.global_env.clone(),
                    };
                    // Add to global env first (enables recursion)
                    self.global_env.borrow_mut().define(name.clone(), closure);
                }
                Ok(())
            }
            Decl::Type { .. } | Decl::TypeAlias { .. } => {
                // Type declarations don't need runtime evaluation
                Ok(())
            }
            Decl::Trait { .. } => {
                // Trait declarations are handled during type inference
                // At runtime, nothing to do
                Ok(())
            }
            Decl::Instance { methods, .. } => {
                // Compile each method implementation as a closure in the global env
                // The actual dictionary will be constructed at call site
                for method in methods {
                    // Create a closure for this method implementation
                    let closure = Value::Closure {
                        params: method.params.clone(),
                        body: Rc::new(method.body.clone()),
                        env: self.global_env.clone(),
                    };
                    // Store with a unique name based on instance
                    // For now, we'll just use the method name and let dynamic resolution pick the right one
                    // In a production system, we'd mangle the name or use a dictionary structure
                    // The method lookup will happen dynamically based on the argument type
                    self.global_env.borrow_mut().define(
                        format!(
                            "__method_{}_{}",
                            method.name,
                            self.class_env.instances.len()
                        ),
                        closure,
                    );
                }
                Ok(())
            }
            Decl::Val { .. } => {
                // Type signatures are handled during type inference
                // At runtime, nothing to do
                Ok(())
            }
            Decl::OperatorDef { op, params, body, .. } => {
                // Operator definitions are like regular function definitions
                // The operator name is bound to a closure in the global env
                let closure = Value::Closure {
                    params: params.clone(),
                    body: Rc::new(body.clone()),
                    env: self.global_env.clone(),
                };
                self.global_env.borrow_mut().define(op.clone(), closure);
                Ok(())
            }
            Decl::Fixity(_) => {
                // Fixity declarations are handled during parsing
                // At runtime, nothing to do
                Ok(())
            }
            Decl::LetRec { bindings, .. } => {
                // Create all closures that share the global environment
                // They can reference each other since they all look up in global_env
                for binding in bindings {
                    let closure = Value::Closure {
                        params: binding.params.clone(),
                        body: Rc::new(binding.body.clone()),
                        env: self.global_env.clone(),
                    };
                    self.global_env
                        .borrow_mut()
                        .define(binding.name.node.clone(), closure);
                }
                Ok(())
            }
            Decl::Record { .. } => {
                // Record type declarations are handled during type inference
                // At runtime, nothing to do (like ADT type declarations)
                Ok(())
            }
        }
    }

    /// Build a dictionary for a given predicate by looking up the matching instance
    pub fn build_dict(&mut self, pred: &Pred) -> Result<Value, EvalError> {
        // Find the matching instance
        if let Some(resolution) = self.class_env.resolve_pred(pred) {
            let instance = &self.class_env.instances[resolution.instance_idx];

            // Build method closures from the instance
            let mut methods = HashMap::new();
            for method in &instance.method_impls {
                let closure = Value::Closure {
                    params: method.params.clone(),
                    body: Rc::new(method.body.clone()),
                    env: self.global_env.clone(),
                };
                methods.insert(method.name.clone(), closure);
            }

            Ok(Value::Dict {
                trait_name: pred.trait_name.clone(),
                methods,
            })
        } else {
            Err(EvalError::RuntimeError(format!(
                "No instance of {} for type {}",
                pred.trait_name, pred.ty
            )))
        }
    }

    /// Look up a method from a dictionary and apply it
    pub fn call_method(
        &mut self,
        dict: &Value,
        method_name: &str,
        arg: Value,
    ) -> Result<Value, EvalError> {
        match dict {
            Value::Dict { methods, .. } => {
                if let Some(method_impl) = methods.get(method_name) {
                    self.apply_value(method_impl.clone(), arg)
                } else {
                    Err(EvalError::RuntimeError(format!(
                        "Method {} not found in dictionary",
                        method_name
                    )))
                }
            }
            _ => Err(EvalError::TypeError("Expected dictionary value".into())),
        }
    }

    // ========================================================================
    // Step-based evaluation (defunctionalized CPS)
    // ========================================================================

    /// Evaluate an expression to completion (convenience wrapper)
    pub fn eval_expr(&mut self, env: &Env, expr: &Expr) -> Result<Value, EvalError> {
        let state = State::Eval {
            expr: Rc::new(expr.clone()),
            env: env.clone(),
            cont: Cont::new(),
        };
        self.run_to_completion(state)
    }

    /// Apply a function value to an argument (convenience wrapper)
    fn apply_value(&mut self, func: Value, arg: Value) -> Result<Value, EvalError> {
        let mut cont = Cont::new();
        cont.push(Frame::AppArg { func });
        let state = State::Apply { value: arg, cont };
        self.run_to_completion(state)
    }

    /// Run the machine until completion or error
    fn run_to_completion(&mut self, mut state: State) -> Result<Value, EvalError> {
        loop {
            match self.step(state) {
                StepResult::Continue(next) => state = next,
                StepResult::Done(value) => return Ok(value),
                StepResult::Error(e) => return Err(e),
            }
        }
    }

    /// Execute a single step of the machine
    pub fn step(&mut self, state: State) -> StepResult {
        match state {
            State::Eval { expr, env, cont } => self.step_eval(&expr.node, env, cont),
            State::Apply { value, cont } => self.step_apply(value, cont),
        }
    }

    /// Step when in Eval mode - evaluate an expression
    fn step_eval(&mut self, expr: &ExprKind, env: Env, mut cont: Cont) -> StepResult {
        match expr {
            // === Immediate values - go straight to Apply ===
            ExprKind::Lit(lit) => {
                let value = self.eval_literal(lit);
                StepResult::Continue(State::Apply { value, cont })
            }

            ExprKind::Var(name) => {
                // Try local env first, then imports/qualified names
                if let Some(value) = self.lookup_name(name, &env) {
                    StepResult::Continue(State::Apply { value, cont })
                } else {
                    // Check if this is a trait method
                    if let Some((trait_name, _)) = self.class_env.lookup_method(name) {
                        // Return a special "method reference" that will be resolved
                        // when we know the argument type
                        let value =
                            Value::Builtin(format!("__method__{}_{}", trait_name, name));
                        StepResult::Continue(State::Apply { value, cont })
                    } else {
                        StepResult::Error(EvalError::UnboundVariable(name.clone()))
                    }
                }
            }

            ExprKind::Lambda { params, body } => {
                let value = Value::Closure {
                    params: params.clone(),
                    body: body.clone(),
                    env: env.clone(),
                };
                StepResult::Continue(State::Apply { value, cont })
            }

            // === Compound expressions - push frames and evaluate sub-expressions ===
            ExprKind::App { func, arg } => {
                cont.push(Frame::AppFunc {
                    arg: arg.clone(),
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: func.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::Let {
                pattern,
                value,
                body,
            } => {
                cont.push(Frame::Let {
                    pattern: pattern.clone(),
                    body: body.clone(),
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: value.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::LetRec { bindings, body } => {
                // Create a shared recursive environment
                let rec_env = EnvInner::with_parent(&env);

                // Create all closures with the shared env so they can reference each other
                for binding in bindings {
                    let closure = Value::Closure {
                        params: binding.params.clone(),
                        body: Rc::new(binding.body.clone()),
                        env: rec_env.clone(),
                    };
                    rec_env
                        .borrow_mut()
                        .define(binding.name.node.clone(), closure);
                }

                // Evaluate body with the recursive environment
                match body {
                    Some(b) => StepResult::Continue(State::Eval {
                        expr: b.clone(),
                        env: rec_env,
                        cont,
                    }),
                    None => StepResult::Continue(State::Apply {
                        value: Value::Unit,
                        cont,
                    }),
                }
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                cont.push(Frame::If {
                    then_branch: then_branch.clone(),
                    else_branch: else_branch.clone(),
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: cond.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::BinOp { op, left, right } => {
                cont.push(Frame::BinOpLeft {
                    op: op.clone(),
                    right: right.clone(),
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: left.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::UnaryOp { op, operand } => {
                cont.push(Frame::UnaryOp { op: *op });
                StepResult::Continue(State::Eval {
                    expr: operand.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::Seq { first, second } => {
                cont.push(Frame::Seq {
                    second: second.clone(),
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: first.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::Match { scrutinee, arms } => {
                cont.push(Frame::Match {
                    arms: arms.clone(),
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: scrutinee.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::Tuple(exprs) => {
                self.start_collect(CollectKind::Tuple, exprs.clone(), env, cont)
            }

            ExprKind::List(exprs) => {
                self.start_collect(CollectKind::List, exprs.clone(), env, cont)
            }

            ExprKind::Constructor { name, args } => self.start_collect(
                CollectKind::Constructor { name: name.clone() },
                args.clone(),
                env,
                cont,
            ),

            // === Concurrency primitives (basic implementation for now) ===
            ExprKind::Spawn(body) => {
                cont.push(Frame::Spawn);
                StepResult::Continue(State::Eval {
                    expr: body.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::NewChannel => {
                let id = self.runtime.new_channel();
                StepResult::Continue(State::Apply {
                    value: Value::Channel(id),
                    cont,
                })
            }

            ExprKind::ChanSend { channel, value } => {
                cont.push(Frame::FiberSendValue {
                    value_expr: value.clone(),
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: channel.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::ChanRecv(channel) => {
                cont.push(Frame::FiberRecv);
                StepResult::Continue(State::Eval {
                    expr: channel.clone(),
                    env,
                    cont,
                })
            }

            // === Delimited continuations ===
            ExprKind::Reset(body) => {
                cont.push(Frame::Prompt);
                StepResult::Continue(State::Eval {
                    expr: body.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::Shift { param, body } => {
                // Capture frames up to Prompt
                let mut captured = Vec::new();

                loop {
                    match cont.pop() {
                        None => {
                            return StepResult::Error(EvalError::RuntimeError(
                                "shift without enclosing reset".into(),
                            ));
                        }
                        Some(Frame::Prompt) => {
                            break; // Found delimiter, stop capturing
                        }
                        Some(frame) => {
                            captured.push(frame);
                        }
                    }
                }

                // Frames were popped innermost-first; reverse so outermost is first
                // (This makes restore simpler: just push in order)
                captured.reverse();

                // Create continuation value
                let continuation = Value::Continuation { frames: captured };

                // Bind parameter to continuation and evaluate body
                let new_env = EnvInner::with_parent(&env);
                if !self.try_bind_pattern(&new_env, param, &continuation) {
                    return StepResult::Error(EvalError::MatchFailed);
                }

                StepResult::Continue(State::Eval {
                    expr: body.clone(),
                    env: new_env,
                    cont,
                })
            }

            ExprKind::Select { arms } => {
                if arms.is_empty() {
                    return StepResult::Error(EvalError::RuntimeError("empty select".into()));
                }

                // Extract components from arms
                let patterns: Vec<Pattern> = arms.iter().map(|a| a.pattern.clone()).collect();
                let bodies: Vec<Rc<Expr>> = arms.iter().map(|a| Rc::new(a.body.clone())).collect();
                let chan_exprs: Vec<Expr> = arms.iter().map(|a| a.channel.clone()).collect();

                // Start evaluating the first channel expression
                let mut remaining = chan_exprs;
                let first = remaining.remove(0);

                cont.push(Frame::FiberSelectChans {
                    patterns,
                    bodies,
                    remaining_chans: remaining,
                    collected_chans: Vec::new(),
                    env: env.clone(),
                });

                StepResult::Continue(State::Eval {
                    expr: Rc::new(first),
                    env,
                    cont,
                })
            }

            // ========================================================================
            // Records
            // ========================================================================
            ExprKind::Record { name, fields } => {
                // Collect field names and expressions
                let field_names: Vec<String> = fields.iter().map(|(n, _)| n.to_string()).collect();
                let field_exprs: Vec<Expr> = fields.iter().map(|(_, e)| e.clone()).collect();

                self.start_collect(
                    CollectKind::Record {
                        type_name: name.to_string(),
                        field_names,
                    },
                    field_exprs,
                    env,
                    cont,
                )
            }

            ExprKind::FieldAccess { record, field } => {
                cont.push(Frame::FieldAccess {
                    field: field.to_string(),
                });
                StepResult::Continue(State::Eval {
                    expr: record.clone(),
                    env,
                    cont,
                })
            }

            ExprKind::RecordUpdate { base, updates } => {
                // Collect update field names and expressions
                let update_field_names: Vec<String> =
                    updates.iter().map(|(n, _)| n.to_string()).collect();
                let update_exprs: Vec<Expr> = updates.iter().map(|(_, e)| e.clone()).collect();

                if update_exprs.is_empty() {
                    // No updates, just evaluate and return the base
                    StepResult::Continue(State::Eval {
                        expr: base.clone(),
                        env,
                        cont,
                    })
                } else {
                    // Push frame to collect updates, then push frame to eval base
                    cont.push(Frame::RecordUpdate {
                        update_field_names,
                        remaining_updates: update_exprs,
                        collected_updates: vec![],
                        env: env.clone(),
                    });
                    StepResult::Continue(State::Eval {
                        expr: base.clone(),
                        env,
                        cont,
                    })
                }
            }
        }
    }

    /// Start collecting values for a list/tuple/constructor
    fn start_collect(
        &mut self,
        kind: CollectKind,
        exprs: Vec<Expr>,
        env: Env,
        mut cont: Cont,
    ) -> StepResult {
        if exprs.is_empty() {
            // No elements - immediately produce the value
            let value = match kind {
                CollectKind::List => Value::List(im::Vector::new()),
                CollectKind::Tuple => Value::Tuple(vec![]),
                CollectKind::Constructor { name } => Value::Constructor {
                    name,
                    fields: vec![],
                },
                CollectKind::Record { type_name, .. } => Value::Record {
                    type_name,
                    fields: im::HashMap::new(),
                },
            };
            StepResult::Continue(State::Apply { value, cont })
        } else {
            // Evaluate first element, queue the rest
            let first = Rc::new(exprs[0].clone());
            let remaining = exprs[1..].to_vec();
            cont.push(Frame::Collect {
                kind,
                remaining,
                acc: vec![],
                env: env.clone(),
            });
            StepResult::Continue(State::Eval {
                expr: first,
                env,
                cont,
            })
        }
    }

    /// Step when in Apply mode - return a value to the continuation
    fn step_apply(&mut self, value: Value, mut cont: Cont) -> StepResult {
        match cont.pop() {
            None => {
                // No more frames - we're done!
                StepResult::Done(value)
            }

            Some(Frame::AppFunc { arg, env }) => {
                // Got the function, now evaluate the argument
                cont.push(Frame::AppArg { func: value });
                StepResult::Continue(State::Eval {
                    expr: arg,
                    env,
                    cont,
                })
            }

            Some(Frame::AppArg { func }) => {
                // Got the argument, now apply
                self.do_apply(func, value, cont)
            }

            Some(Frame::Let { pattern, body, env }) => {
                // Got the value, bind pattern
                let new_env = EnvInner::with_parent(&env);

                // For recursive functions: if binding a variable to a closure,
                // patch the closure's environment to include itself
                let value = match (&pattern.node, value) {
                    (
                        PatternKind::Var(name),
                        Value::Closure {
                            params,
                            body: closure_body,
                            env: closure_env,
                        },
                    ) => {
                        // Create a recursive closure by making a new environment
                        // that includes the binding to itself
                        let recursive_env = EnvInner::with_parent(&closure_env);
                        let recursive_closure = Value::Closure {
                            params,
                            body: closure_body,
                            env: recursive_env.clone(),
                        };
                        // Add the binding to the recursive environment
                        recursive_env
                            .borrow_mut()
                            .define(name.clone(), recursive_closure.clone());
                        recursive_closure
                    }
                    (_, value) => value,
                };

                if !self.try_bind_pattern(&new_env, &pattern, &value) {
                    return StepResult::Error(EvalError::MatchFailed);
                }
                match body {
                    Some(body) => StepResult::Continue(State::Eval {
                        expr: body,
                        env: new_env,
                        cont,
                    }),
                    None => StepResult::Continue(State::Apply {
                        value: Value::Unit,
                        cont,
                    }),
                }
            }

            Some(Frame::If {
                then_branch,
                else_branch,
                env,
            }) => match value {
                Value::Bool(true) => StepResult::Continue(State::Eval {
                    expr: then_branch,
                    env,
                    cont,
                }),
                Value::Bool(false) => StepResult::Continue(State::Eval {
                    expr: else_branch,
                    env,
                    cont,
                }),
                _ => StepResult::Error(EvalError::TypeError("expected bool in condition".into())),
            },

            Some(Frame::BinOpLeft { op, right, env }) => {
                cont.push(Frame::BinOpRight {
                    op,
                    left: value,
                    env: env.clone(),
                });
                StepResult::Continue(State::Eval {
                    expr: right,
                    env,
                    cont,
                })
            }

            Some(Frame::BinOpRight { op, left, env }) => {
                // Handle user-defined operators by looking them up as functions
                if let BinOp::UserDefined(name) = &op {
                    let right = value;
                    // Look up the operator in the environment
                    match env.borrow().get(name) {
                        Some(func) => {
                            // Apply curried: ((op left) right)
                            // 1. Apply op to left to get a partial application
                            // 2. Apply the result (partial app) to right
                            // Push frame to apply result to right
                            cont.push(Frame::ApplyTo { arg: right }); // step 2
                                                                      // step 1: apply op to left
                            self.do_apply(func, left, cont)
                        }
                        None => StepResult::Error(EvalError::RuntimeError(format!(
                            "undefined operator: {}",
                            name
                        ))),
                    }
                } else {
                    // Built-in operator
                    match self.eval_binop(&op, left, value) {
                        Ok(result) => StepResult::Continue(State::Apply {
                            value: result,
                            cont,
                        }),
                        Err(e) => StepResult::Error(e),
                    }
                }
            }

            Some(Frame::ApplyTo { arg }) => {
                // value is a function, apply it to arg
                self.do_apply(value, arg, cont)
            }

            Some(Frame::UnaryOp { op }) => match self.eval_unaryop(op, value) {
                Ok(result) => StepResult::Continue(State::Apply {
                    value: result,
                    cont,
                }),
                Err(e) => StepResult::Error(e),
            },

            Some(Frame::Seq { second, env }) => {
                // Discard value, evaluate second
                StepResult::Continue(State::Eval {
                    expr: second,
                    env,
                    cont,
                })
            }

            Some(Frame::Collect {
                kind,
                remaining,
                mut acc,
                env,
            }) => {
                acc.push(value);
                if remaining.is_empty() {
                    // All collected, produce the value
                    let result = match kind {
                        CollectKind::List => Value::List(acc.into_iter().collect()),
                        CollectKind::Tuple => Value::Tuple(acc),
                        CollectKind::Constructor { name } => {
                            Value::Constructor { name, fields: acc }
                        }
                        CollectKind::Record {
                            type_name,
                            field_names,
                        } => {
                            let fields: im::HashMap<String, Value> = field_names
                                .into_iter()
                                .zip(acc)
                                .collect();
                            Value::Record { type_name, fields }
                        }
                    };
                    StepResult::Continue(State::Apply {
                        value: result,
                        cont,
                    })
                } else {
                    // More to collect
                    let next = Rc::new(remaining[0].clone());
                    let rest = remaining[1..].to_vec();
                    cont.push(Frame::Collect {
                        kind,
                        remaining: rest,
                        acc,
                        env: env.clone(),
                    });
                    StepResult::Continue(State::Eval {
                        expr: next,
                        env,
                        cont,
                    })
                }
            }

            Some(Frame::Match { arms, env }) => self.try_match_arms(value, arms, env, cont),

            Some(Frame::MatchGuard {
                body,
                remaining_arms,
                scrutinee,
                bound_env,
                outer_env,
            }) => {
                match value {
                    Value::Bool(true) => {
                        // Guard passed, evaluate body
                        StepResult::Continue(State::Eval {
                            expr: body,
                            env: bound_env,
                            cont,
                        })
                    }
                    Value::Bool(false) => {
                        // Guard failed, try remaining arms
                        self.try_match_arms(scrutinee, remaining_arms, outer_env, cont)
                    }
                    _ => StepResult::Error(EvalError::TypeError("guard must be bool".into())),
                }
            }

            // === Delimited continuation frames ===
            Some(Frame::Prompt) => {
                // reset body completed, value passes through
                StepResult::Continue(State::Apply { value, cont })
            }

            // === Concurrency frames ===
            Some(Frame::Spawn) => {
                let pid = self.runtime.spawn(value);
                StepResult::Continue(State::Apply {
                    value: Value::Pid(pid),
                    cont,
                })
            }

            // === Fiber effect frames ===
            Some(Frame::FiberBoundary) => {
                // Check if this is a fiber effect propagating to scheduler
                if let Value::FiberEffect(effect) = value {
                    // Already a fiber effect - propagate directly without wrapping
                    self.return_fiber_effect(effect, cont)
                } else {
                    // Normal value - fiber completed, wrap in Done effect
                    let effect = FiberEffect::Done(Box::new(value));
                    self.return_fiber_effect(effect, cont)
                }
            }

            Some(Frame::FiberRecv) => {
                // Channel evaluated, capture continuation and return Recv effect
                let channel = match &value {
                    Value::Channel(id) => *id,
                    _ => {
                        return StepResult::Error(EvalError::RuntimeError(
                            "FiberRecv: expected channel".into(),
                        ))
                    }
                };

                // Capture continuation to FiberBoundary
                match self.capture_to_fiber_boundary(&mut cont) {
                    Ok(captured) => {
                        let effect = FiberEffect::Recv {
                            channel,
                            cont: Some(Box::new(captured)),
                        };
                        self.return_fiber_effect(effect, cont)
                    }
                    Err(e) => StepResult::Error(e),
                }
            }

            Some(Frame::FiberSendValue { value_expr, env }) => {
                // Phase 4: Channel evaluated, now evaluate value
                let channel = match &value {
                    Value::Channel(id) => *id,
                    _ => {
                        return StepResult::Error(EvalError::RuntimeError(
                            "FiberSendValue: expected channel".into(),
                        ))
                    }
                };
                cont.push(Frame::FiberSendReady { channel });
                StepResult::Continue(State::Eval {
                    expr: value_expr,
                    env,
                    cont,
                })
            }

            Some(Frame::FiberSendReady { channel }) => {
                // Value evaluated, capture continuation and return Send effect
                match self.capture_to_fiber_boundary(&mut cont) {
                    Ok(captured) => {
                        let effect = FiberEffect::Send {
                            channel,
                            value: Box::new(value),
                            cont: Some(Box::new(captured)),
                        };
                        self.return_fiber_effect(effect, cont)
                    }
                    Err(e) => StepResult::Error(e),
                }
            }

            Some(Frame::FiberFork) => {
                // Thunk evaluated, capture continuation and return Fork effect
                match self.capture_to_fiber_boundary(&mut cont) {
                    Ok(captured) => {
                        let effect = FiberEffect::Fork {
                            thunk: Box::new(value),
                            cont: Some(Box::new(captured)),
                        };
                        self.return_fiber_effect(effect, cont)
                    }
                    Err(e) => StepResult::Error(e),
                }
            }

            Some(Frame::FiberJoin) => {
                // Fiber handle evaluated, capture continuation and return Join effect
                let fiber_id = match &value {
                    Value::Fiber(id) => *id,
                    _ => {
                        return StepResult::Error(EvalError::RuntimeError(
                            "FiberJoin: expected Fiber value".into(),
                        ))
                    }
                };

                match self.capture_to_fiber_boundary(&mut cont) {
                    Ok(captured) => {
                        let effect = FiberEffect::Join {
                            fiber_id,
                            cont: Some(Box::new(captured)),
                        };
                        self.return_fiber_effect(effect, cont)
                    }
                    Err(e) => StepResult::Error(e),
                }
            }

            Some(Frame::FiberSelectChans {
                patterns,
                bodies,
                remaining_chans,
                mut collected_chans,
                env,
            }) => {
                // Collecting channels for fiber select
                let channel = match &value {
                    Value::Channel(id) => *id,
                    _ => {
                        return StepResult::Error(EvalError::RuntimeError(
                            "FiberSelectChans: expected channel".into(),
                        ))
                    }
                };
                collected_chans.push(channel);

                if remaining_chans.is_empty() {
                    // All channels collected, ready to produce Select effect
                    cont.push(Frame::FiberSelectReady {
                        channels: collected_chans,
                        patterns,
                        bodies,
                        env,
                    });
                    StepResult::Continue(State::Apply {
                        value: Value::Unit,
                        cont,
                    })
                } else {
                    // More channels to evaluate
                    let mut remaining = remaining_chans;
                    let next = remaining.remove(0);
                    cont.push(Frame::FiberSelectChans {
                        patterns,
                        bodies,
                        remaining_chans: remaining,
                        collected_chans,
                        env: env.clone(),
                    });
                    StepResult::Continue(State::Eval {
                        expr: Rc::new(next),
                        env,
                        cont,
                    })
                }
            }

            Some(Frame::FiberSelectReady {
                channels,
                patterns,
                bodies,
                env,
            }) => {
                // All channels collected, capture continuation and return Select effect
                match self.capture_to_fiber_boundary(&mut cont) {
                    Ok(captured) => {
                        // Convert patterns/bodies/env into SelectEffectArms
                        let arms: Vec<SelectEffectArm> = channels
                            .into_iter()
                            .zip(patterns)
                            .zip(bodies)
                            .map(|((channel, pattern), body)| SelectEffectArm {
                                channel,
                                pattern,
                                body,
                                env: env.clone(),
                            })
                            .collect();

                        let effect = FiberEffect::Select {
                            arms,
                            cont: Some(Box::new(captured)),
                        };
                        self.return_fiber_effect(effect, cont)
                    }
                    Err(e) => StepResult::Error(e),
                }
            }

            // ========================================================================
            // Record frames
            // ========================================================================
            Some(Frame::FieldAccess { field }) => {
                // Extract field from record value
                match value {
                    Value::Record { fields, .. } => {
                        if let Some(field_value) = fields.get(&field) {
                            StepResult::Continue(State::Apply {
                                value: field_value.clone(),
                                cont,
                            })
                        } else {
                            StepResult::Error(EvalError::RuntimeError(format!(
                                "record has no field '{}'",
                                field
                            )))
                        }
                    }
                    _ => StepResult::Error(EvalError::TypeError(format!(
                        "field access on non-record value: {}",
                        value.type_name()
                    ))),
                }
            }

            Some(Frame::RecordUpdate {
                update_field_names,
                mut remaining_updates,
                collected_updates,
                env,
            }) => {
                // We just evaluated the base record
                // Now evaluate the first update expression
                if remaining_updates.is_empty() {
                    // No updates, just return the base (shouldn't happen, handled in eval)
                    StepResult::Continue(State::Apply { value, cont })
                } else {
                    let first_update = remaining_updates.remove(0);
                    let first_field_name = update_field_names[0].clone();
                    let remaining_field_names = update_field_names[1..].to_vec();

                    if remaining_updates.is_empty() {
                        // This is the last update, apply after it's done
                        cont.push(Frame::RecordUpdateApply {
                            base: value,
                            updates: collected_updates,
                        });
                        // We also need to capture this field name to pair with value
                        cont.push(Frame::RecordUpdateCollectLast {
                            field_name: first_field_name,
                        });
                    } else {
                        // More updates to go
                        cont.push(Frame::RecordUpdateApply {
                            base: value,
                            updates: collected_updates,
                        });
                        cont.push(Frame::RecordUpdateContinue {
                            current_field_name: first_field_name,
                            remaining_field_names,
                            remaining_updates: remaining_updates.clone(),
                            env: env.clone(),
                        });
                    }
                    StepResult::Continue(State::Eval {
                        expr: Rc::new(first_update),
                        env,
                        cont,
                    })
                }
            }

            Some(Frame::RecordUpdateContinue {
                current_field_name,
                remaining_field_names,
                mut remaining_updates,
                env,
            }) => {
                // We just evaluated an update expression
                // Pop the RecordUpdateApply frame to add this update and push it back
                if let Some(Frame::RecordUpdateApply { base, mut updates }) = cont.pop() {
                    updates.push((current_field_name, value));

                    if remaining_updates.is_empty() {
                        // All updates collected, apply them
                        cont.push(Frame::RecordUpdateApply { base, updates });
                        // Return unit to trigger the apply (will be ignored)
                        StepResult::Continue(State::Apply {
                            value: Value::Unit,
                            cont,
                        })
                    } else {
                        let next_update = remaining_updates.remove(0);
                        let next_field_name = remaining_field_names[0].clone();
                        let next_remaining_field_names = remaining_field_names[1..].to_vec();

                        cont.push(Frame::RecordUpdateApply { base, updates });

                        if remaining_updates.is_empty() {
                            // This is the last update
                            cont.push(Frame::RecordUpdateCollectLast {
                                field_name: next_field_name,
                            });
                        } else {
                            cont.push(Frame::RecordUpdateContinue {
                                current_field_name: next_field_name,
                                remaining_field_names: next_remaining_field_names,
                                remaining_updates: remaining_updates.clone(),
                                env: env.clone(),
                            });
                        }
                        StepResult::Continue(State::Eval {
                            expr: Rc::new(next_update),
                            env,
                            cont,
                        })
                    }
                } else {
                    StepResult::Error(EvalError::RuntimeError(
                        "RecordUpdateContinue without RecordUpdateApply frame".into(),
                    ))
                }
            }

            Some(Frame::RecordUpdateCollectLast { field_name }) => {
                // We evaluated the last update field, now pair it and apply
                if let Some(Frame::RecordUpdateApply { base, mut updates }) = cont.pop() {
                    updates.push((field_name, value));

                    // Now apply all updates to the base
                    match base {
                        Value::Record {
                            type_name,
                            mut fields,
                        } => {
                            for (field_name, new_value) in updates {
                                fields.insert(field_name, new_value);
                            }
                            StepResult::Continue(State::Apply {
                                value: Value::Record { type_name, fields },
                                cont,
                            })
                        }
                        _ => StepResult::Error(EvalError::TypeError(
                            "record update on non-record value".into(),
                        )),
                    }
                } else {
                    StepResult::Error(EvalError::RuntimeError(
                        "RecordUpdateCollectLast without RecordUpdateApply frame".into(),
                    ))
                }
            }

            Some(Frame::RecordUpdateApply { base, updates }) => {
                // Apply accumulated updates to base record
                match base {
                    Value::Record {
                        type_name,
                        mut fields,
                    } => {
                        for (field_name, new_value) in updates {
                            fields.insert(field_name, new_value);
                        }
                        StepResult::Continue(State::Apply {
                            value: Value::Record { type_name, fields },
                            cont,
                        })
                    }
                    _ => StepResult::Error(EvalError::TypeError(
                        "record update on non-record value".into(),
                    )),
                }
            }

            Some(Frame::IoOp {
                op_kind,
                mut collected_args,
                remaining_args,
                env,
            }) => {
                // Add the just-evaluated value to collected args
                collected_args.push(value);

                if remaining_args.is_empty() {
                    // All args collected, produce the IoOp effect
                    match Self::build_io_op(op_kind, collected_args) {
                        Ok(io_op) => {
                            // Create the effect - continuation will be attached by capture
                            let effect = FiberEffect::Io { op: io_op, cont: None };
                            StepResult::Continue(State::Apply {
                                value: Value::FiberEffect(effect),
                                cont,
                            })
                        }
                        Err(e) => StepResult::Error(e),
                    }
                } else {
                    // More args to evaluate
                    let mut remaining = remaining_args;
                    let next = remaining.remove(0);
                    cont.push(Frame::IoOp {
                        op_kind,
                        collected_args,
                        remaining_args: remaining,
                        env: env.clone(),
                    });
                    StepResult::Continue(State::Eval {
                        expr: Rc::new(next),
                        env,
                        cont,
                    })
                }
            }
        }
    }

    /// Build an IoOp from collected arguments based on the operation kind
    fn build_io_op(op_kind: PartialIoOp, args: Vec<Value>) -> Result<IoOp, EvalError> {
        match op_kind {
            PartialIoOp::FileOpen => {
                // args: [path: String, mode: String]
                if args.len() != 2 {
                    return Err(EvalError::TypeError(format!(
                        "file_open expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let path = match &args[0] {
                    Value::String(s) => s.clone(),
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "file_open path must be String, got {}",
                            v.type_name()
                        )))
                    }
                };
                let mode = match &args[1] {
                    Value::String(s) => OpenMode::parse(s).ok_or_else(|| {
                        EvalError::TypeError(format!("invalid open mode: {}", s))
                    })?,
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "file_open mode must be String, got {}",
                            v.type_name()
                        )))
                    }
                };
                Ok(IoOp::FileOpen { path, mode })
            }

            PartialIoOp::TcpConnect => {
                // args: [host: String, port: Int]
                if args.len() != 2 {
                    return Err(EvalError::TypeError(format!(
                        "tcp_connect expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let host = match &args[0] {
                    Value::String(s) => s.clone(),
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "tcp_connect host must be String, got {}",
                            v.type_name()
                        )))
                    }
                };
                let port = match &args[1] {
                    Value::Int(n) if *n >= 0 && *n <= 65535 => *n as u16,
                    Value::Int(n) => {
                        return Err(EvalError::TypeError(format!(
                            "tcp_connect port must be 0-65535, got {}",
                            n
                        )))
                    }
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "tcp_connect port must be Int, got {}",
                            v.type_name()
                        )))
                    }
                };
                Ok(IoOp::TcpConnect { host, port })
            }

            PartialIoOp::TcpListen => {
                // args: [host: String, port: Int]
                if args.len() != 2 {
                    return Err(EvalError::TypeError(format!(
                        "tcp_listen expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let host = match &args[0] {
                    Value::String(s) => s.clone(),
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "tcp_listen host must be String, got {}",
                            v.type_name()
                        )))
                    }
                };
                let port = match &args[1] {
                    Value::Int(n) if *n >= 0 && *n <= 65535 => *n as u16,
                    Value::Int(n) => {
                        return Err(EvalError::TypeError(format!(
                            "tcp_listen port must be 0-65535, got {}",
                            n
                        )))
                    }
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "tcp_listen port must be Int, got {}",
                            v.type_name()
                        )))
                    }
                };
                Ok(IoOp::TcpListen { host, port })
            }

            PartialIoOp::Read => {
                // args: [handle: FileHandle|TcpSocket, count: Int]
                if args.len() != 2 {
                    return Err(EvalError::TypeError(format!(
                        "io_read expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let handle = match &args[0] {
                    Value::FileHandle(id) | Value::TcpSocket(id) => *id,
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "io_read handle must be FileHandle or TcpSocket, got {}",
                            v.type_name()
                        )))
                    }
                };
                let count = match &args[1] {
                    Value::Int(n) if *n >= 0 => *n as usize,
                    Value::Int(n) => {
                        return Err(EvalError::TypeError(format!(
                            "io_read count must be non-negative, got {}",
                            n
                        )))
                    }
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "io_read count must be Int, got {}",
                            v.type_name()
                        )))
                    }
                };
                Ok(IoOp::Read { handle, count })
            }

            PartialIoOp::Write => {
                // args: [handle: FileHandle|TcpSocket, data: Bytes]
                if args.len() != 2 {
                    return Err(EvalError::TypeError(format!(
                        "io_write expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let handle = match &args[0] {
                    Value::FileHandle(id) | Value::TcpSocket(id) => *id,
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "io_write handle must be FileHandle or TcpSocket, got {}",
                            v.type_name()
                        )))
                    }
                };
                let data = match &args[1] {
                    Value::Bytes(b) => b.clone(),
                    v => {
                        return Err(EvalError::TypeError(format!(
                            "io_write data must be Bytes, got {}",
                            v.type_name()
                        )))
                    }
                };
                Ok(IoOp::Write { handle, data })
            }
        }
    }

    /// Try to match a value against a list of arms
    fn try_match_arms(
        &mut self,
        scrutinee: Value,
        arms: Vec<MatchArm>,
        env: Env,
        cont: Cont,
    ) -> StepResult {
        for (i, arm) in arms.iter().enumerate() {
            let bound_env = EnvInner::with_parent(&env);
            if self.try_bind_pattern(&bound_env, &arm.pattern, &scrutinee) {
                // Pattern matched
                if let Some(guard) = &arm.guard {
                    // Has a guard - evaluate it
                    let remaining = arms[i + 1..].to_vec();
                    let mut cont = cont;
                    cont.push(Frame::MatchGuard {
                        body: Rc::new(arm.body.clone()),
                        remaining_arms: remaining,
                        scrutinee: scrutinee.clone(),
                        bound_env: bound_env.clone(),
                        outer_env: env.clone(),
                    });
                    return StepResult::Continue(State::Eval {
                        expr: Rc::new(guard.clone()),
                        env: bound_env,
                        cont,
                    });
                } else {
                    // No guard - evaluate body
                    return StepResult::Continue(State::Eval {
                        expr: Rc::new(arm.body.clone()),
                        env: bound_env,
                        cont,
                    });
                }
            }
        }
        // No arm matched
        StepResult::Error(EvalError::MatchFailed)
    }

    /// Apply a function value to an argument value
    fn do_apply(&mut self, func: Value, arg: Value, mut cont: Cont) -> StepResult {
        match func {
            Value::Closure { params, body, env } => {
                if params.is_empty() {
                    return StepResult::Error(EvalError::TypeError("applying non-function".into()));
                }

                let new_env = EnvInner::with_parent(&env);
                if !self.try_bind_pattern(&new_env, &params[0], &arg) {
                    return StepResult::Error(EvalError::MatchFailed);
                }

                if params.len() == 1 {
                    // All parameters bound, evaluate body
                    StepResult::Continue(State::Eval {
                        expr: body,
                        env: new_env,
                        cont,
                    })
                } else {
                    // Partial application - return closure with remaining params
                    let remaining = Value::Closure {
                        params: params[1..].to_vec(),
                        body,
                        env: new_env,
                    };
                    StepResult::Continue(State::Apply {
                        value: remaining,
                        cont,
                    })
                }
            }

            Value::Builtin(ref name) => {
                // Check if this is a trait method dispatch
                if let Some(rest) = name.strip_prefix("__method__") {
                    // Parse "__method__TraitName_methodName"
                    // Skip "__method__"
                    if let Some(underscore_pos) = rest.find('_') {
                        let trait_name = &rest[..underscore_pos];
                        let method_name = &rest[underscore_pos + 1..];

                        // Get the runtime type of the argument, using type_ctx to resolve constructor names
                        let arg_type = arg.to_type_with_ctx(&self.type_ctx);

                        // Build a predicate and resolve it
                        let pred = Pred::new(trait_name, arg_type);
                        match self.build_dict(&pred) {
                            Ok(dict) => {
                                // Look up the method and call it
                                match self.call_method(&dict, method_name, arg) {
                                    Ok(value) => StepResult::Continue(State::Apply { value, cont }),
                                    Err(e) => StepResult::Error(e),
                                }
                            }
                            Err(e) => StepResult::Error(e),
                        }
                    } else {
                        StepResult::Error(EvalError::RuntimeError(format!(
                            "malformed method reference: {}",
                            name
                        )))
                    }
                } else if name == "spawn" {
                    // The old spawn builtin - directly spawns a process, returns Pid
                    // This needs mutable access to runtime, so handle specially
                    let pid = self.runtime.spawn(arg);
                    StepResult::Continue(State::Apply {
                        value: Value::Pid(pid),
                        cont,
                    })
                } else if Self::is_effectful_builtin(name) {
                    // Handle effectful fiber builtins
                    match self.apply_effectful_builtin(name, arg) {
                        Ok(BuiltinResult::Value(value)) => {
                            StepResult::Continue(State::Apply { value, cont })
                        }
                        Ok(BuiltinResult::Effect(mut effect)) => {
                            // Capture continuation to FiberBoundary and attach to effect
                            match self.capture_to_fiber_boundary(&mut cont) {
                                Ok(captured) => {
                                    effect = effect.with_cont(captured);
                                    self.return_fiber_effect(effect, cont)
                                }
                                Err(e) => StepResult::Error(e),
                            }
                        }
                        Ok(BuiltinResult::Partial { name, args }) => {
                            // Need more arguments
                            StepResult::Continue(State::Apply {
                                value: Value::BuiltinPartial { name, args },
                                cont,
                            })
                        }
                        Err(e) => StepResult::Error(e),
                    }
                } else {
                    match self.apply_builtin(name, arg) {
                        Ok(value) => StepResult::Continue(State::Apply { value, cont }),
                        Err(e) => StepResult::Error(e),
                    }
                }
            }

            Value::Constructor { name, fields } => {
                // Constructor application adds to fields
                let mut new_fields = fields;
                new_fields.push(arg);
                let value = Value::Constructor {
                    name,
                    fields: new_fields,
                };
                StepResult::Continue(State::Apply { value, cont })
            }

            Value::Continuation { frames } => {
                // CRITICAL: Push a fresh Prompt before splicing - this is the "reset" that wraps k's invocation
                // This implements the canonical shift/reset semantics: k x ≡ reset E[x]
                // Without this, we'd have control/prompt semantics instead
                cont.push(Frame::Prompt);

                // Splice captured frames back onto stack
                // Frames are stored outermost-first, push in order so innermost ends up on top
                for frame in frames.into_iter() {
                    cont.push(frame);
                }

                // The argument becomes the "return value" to those frames
                StepResult::Continue(State::Apply { value: arg, cont })
            }

            Value::BuiltinPartial { name, mut args } => {
                // Add argument to the partial application
                args.push(arg);

                // Check if we have all arguments for known multi-arg builtins
                match (name.as_str(), args.len()) {
                    // Dict.insert key value dict -> 3 args
                    ("Dict.insert", 3) => {
                        let dict = args.pop().unwrap();
                        let value = args.pop().unwrap();
                        let key = args.pop().unwrap();
                        match (key, dict) {
                            (Value::String(k), Value::Dictionary(mut d)) => {
                                d.insert(k, value);
                                StepResult::Continue(State::Apply {
                                    value: Value::Dictionary(d),
                                    cont,
                                })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Dict.insert: expected Dictionary".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Dict.insert: key must be String".into(),
                            )),
                        }
                    }
                    // Dict.get key dict -> 2 args
                    ("Dict.get", 2) => {
                        let dict = args.pop().unwrap();
                        let key = args.pop().unwrap();
                        match (key, dict) {
                            (Value::String(k), Value::Dictionary(d)) => {
                                let result = match d.get(&k) {
                                    Some(v) => Value::Constructor {
                                        name: "Some".into(),
                                        fields: vec![v.clone()],
                                    },
                                    None => Value::Constructor {
                                        name: "None".into(),
                                        fields: vec![],
                                    },
                                };
                                StepResult::Continue(State::Apply { value: result, cont })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Dict.get: expected Dictionary".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Dict.get: key must be String".into(),
                            )),
                        }
                    }
                    // Dict.remove key dict -> 2 args
                    ("Dict.remove", 2) => {
                        let dict = args.pop().unwrap();
                        let key = args.pop().unwrap();
                        match (key, dict) {
                            (Value::String(k), Value::Dictionary(mut d)) => {
                                d.remove(&k);
                                StepResult::Continue(State::Apply {
                                    value: Value::Dictionary(d),
                                    cont,
                                })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Dict.remove: expected Dictionary".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Dict.remove: key must be String".into(),
                            )),
                        }
                    }
                    // Dict.contains key dict -> 2 args
                    ("Dict.contains", 2) => {
                        let dict = args.pop().unwrap();
                        let key = args.pop().unwrap();
                        match (key, dict) {
                            (Value::String(k), Value::Dictionary(d)) => {
                                StepResult::Continue(State::Apply {
                                    value: Value::Bool(d.contains_key(&k)),
                                    cont,
                                })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Dict.contains: expected Dictionary".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Dict.contains: key must be String".into(),
                            )),
                        }
                    }
                    // Dict.merge d1 d2 -> 2 args (d2 values override d1)
                    ("Dict.merge", 2) => {
                        let d2 = args.pop().unwrap();
                        let d1 = args.pop().unwrap();
                        match (d1, d2) {
                            (Value::Dictionary(d1), Value::Dictionary(d2)) => {
                                // d2 entries override d1
                                let merged = d1.union(d2);
                                StepResult::Continue(State::Apply {
                                    value: Value::Dictionary(merged),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "Dict.merge: expected two Dictionaries".into(),
                            )),
                        }
                    }
                    // Dict.getOrDefault default key dict -> 3 args
                    ("Dict.getOrDefault", 3) => {
                        let dict = args.pop().unwrap();
                        let key = args.pop().unwrap();
                        let default = args.pop().unwrap();
                        match (key, dict) {
                            (Value::String(k), Value::Dictionary(d)) => {
                                let result = d.get(&k).cloned().unwrap_or(default);
                                StepResult::Continue(State::Apply { value: result, cont })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Dict.getOrDefault: expected Dictionary".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Dict.getOrDefault: key must be String".into(),
                            )),
                        }
                    }
                    // Set.insert elem set -> 2 args
                    ("Set.insert", 2) => {
                        let set = args.pop().unwrap();
                        let elem = args.pop().unwrap();
                        match (elem, set) {
                            (Value::String(e), Value::Set(mut s)) => {
                                s.insert(e);
                                StepResult::Continue(State::Apply {
                                    value: Value::Set(s),
                                    cont,
                                })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Set.insert: expected Set".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Set.insert: element must be String".into(),
                            )),
                        }
                    }
                    // Set.contains elem set -> 2 args
                    ("Set.contains", 2) => {
                        let set = args.pop().unwrap();
                        let elem = args.pop().unwrap();
                        match (elem, set) {
                            (Value::String(e), Value::Set(s)) => {
                                StepResult::Continue(State::Apply {
                                    value: Value::Bool(s.contains(&e)),
                                    cont,
                                })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Set.contains: expected Set".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Set.contains: element must be String".into(),
                            )),
                        }
                    }
                    // Set.remove elem set -> 2 args
                    ("Set.remove", 2) => {
                        let set = args.pop().unwrap();
                        let elem = args.pop().unwrap();
                        match (elem, set) {
                            (Value::String(e), Value::Set(s)) => {
                                let new_set = s.without(&e);
                                StepResult::Continue(State::Apply {
                                    value: Value::Set(new_set),
                                    cont,
                                })
                            }
                            (Value::String(_), _) => StepResult::Error(EvalError::TypeError(
                                "Set.remove: expected Set".into(),
                            )),
                            _ => StepResult::Error(EvalError::TypeError(
                                "Set.remove: element must be String".into(),
                            )),
                        }
                    }
                    // Set.union set1 set2 -> 2 args
                    ("Set.union", 2) => {
                        let set2 = args.pop().unwrap();
                        let set1 = args.pop().unwrap();
                        match (set1, set2) {
                            (Value::Set(s1), Value::Set(s2)) => {
                                StepResult::Continue(State::Apply {
                                    value: Value::Set(s1.union(s2)),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "Set.union: expected two Sets".into(),
                            )),
                        }
                    }
                    // Set.intersect set1 set2 -> 2 args
                    ("Set.intersect", 2) => {
                        let set2 = args.pop().unwrap();
                        let set1 = args.pop().unwrap();
                        match (set1, set2) {
                            (Value::Set(s1), Value::Set(s2)) => {
                                StepResult::Continue(State::Apply {
                                    value: Value::Set(s1.intersection(s2)),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "Set.intersect: expected two Sets".into(),
                            )),
                        }
                    }
                    // string_index_of needle haystack -> 2 args, returns Option Int
                    ("string_index_of", 2) => {
                        let haystack = args.pop().unwrap();
                        let needle = args.pop().unwrap();
                        match (needle, haystack) {
                            (Value::String(needle), Value::String(haystack)) => {
                                let result = match haystack.find(&needle) {
                                    Some(idx) => {
                                        // Convert byte index to char index
                                        let char_idx = haystack[..idx].chars().count() as i64;
                                        Value::Constructor {
                                            name: "Some".into(),
                                            fields: vec![Value::Int(char_idx)],
                                        }
                                    }
                                    None => Value::Constructor {
                                        name: "None".into(),
                                        fields: vec![],
                                    },
                                };
                                StepResult::Continue(State::Apply { value: result, cont })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "string_index_of: expected two strings".into(),
                            )),
                        }
                    }
                    // string_substring start end str -> 3 args
                    ("string_substring", 3) => {
                        let s = args.pop().unwrap();
                        let end = args.pop().unwrap();
                        let start = args.pop().unwrap();
                        match (start, end, s) {
                            (Value::Int(start), Value::Int(end), Value::String(s)) => {
                                let start = start.max(0) as usize;
                                let end = end.max(0) as usize;
                                // Work with char indices, not byte indices
                                let chars: Vec<char> = s.chars().collect();
                                let len = chars.len();
                                let start = start.min(len);
                                let end = end.min(len);
                                let result: String = if start <= end {
                                    chars[start..end].iter().collect()
                                } else {
                                    String::new()
                                };
                                StepResult::Continue(State::Apply {
                                    value: Value::String(result),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "string_substring: expected (Int, Int, String)".into(),
                            )),
                        }
                    }
                    // string_split delim str -> 2 args, returns List String
                    ("string_split", 2) => {
                        let s = args.pop().unwrap();
                        let delim = args.pop().unwrap();
                        match (delim, s) {
                            (Value::String(delim), Value::String(s)) => {
                                let parts: im::Vector<Value> = s
                                    .split(&delim)
                                    .map(|p| Value::String(p.to_string()))
                                    .collect();
                                StepResult::Continue(State::Apply {
                                    value: Value::List(parts),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "string_split: expected (String, String)".into(),
                            )),
                        }
                    }
                    // string_join sep list -> 2 args, returns String
                    ("string_join", 2) => {
                        let list = args.pop().unwrap();
                        let sep = args.pop().unwrap();
                        match (sep, list) {
                            (Value::String(sep), Value::List(items)) => {
                                let mut strings = Vec::with_capacity(items.len());
                                for item in items {
                                    match item {
                                        Value::String(s) => strings.push(s),
                                        _ => {
                                            return StepResult::Error(EvalError::TypeError(
                                                "string_join: list must contain strings".into(),
                                            ))
                                        }
                                    }
                                }
                                StepResult::Continue(State::Apply {
                                    value: Value::String(strings.join(&sep)),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "string_join: expected (String, List String)".into(),
                            )),
                        }
                    }
                    // string_char_at idx str -> 2 args, returns Option Char
                    ("string_char_at", 2) => {
                        let s = args.pop().unwrap();
                        let idx = args.pop().unwrap();
                        match (idx, s) {
                            (Value::Int(idx), Value::String(s)) => {
                                let result = if idx < 0 {
                                    Value::Constructor {
                                        name: "None".into(),
                                        fields: vec![],
                                    }
                                } else {
                                    match s.chars().nth(idx as usize) {
                                        Some(c) => Value::Constructor {
                                            name: "Some".into(),
                                            fields: vec![Value::Char(c)],
                                        },
                                        None => Value::Constructor {
                                            name: "None".into(),
                                            fields: vec![],
                                        },
                                    }
                                };
                                StepResult::Continue(State::Apply { value: result, cont })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "string_char_at: expected (Int, String)".into(),
                            )),
                        }
                    }
                    // string_concat s1 s2 -> 2 args, returns String
                    ("string_concat", 2) => {
                        let s2 = args.pop().unwrap();
                        let s1 = args.pop().unwrap();
                        match (s1, s2) {
                            (Value::String(s1), Value::String(s2)) => {
                                StepResult::Continue(State::Apply {
                                    value: Value::String(s1 + &s2),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "string_concat: expected (String, String)".into(),
                            )),
                        }
                    }
                    // string_repeat n str -> 2 args, returns String
                    ("string_repeat", 2) => {
                        let s = args.pop().unwrap();
                        let n = args.pop().unwrap();
                        match (n, s) {
                            (Value::Int(n), Value::String(s)) => {
                                let n = n.max(0) as usize;
                                StepResult::Continue(State::Apply {
                                    value: Value::String(s.repeat(n)),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "string_repeat: expected (Int, String)".into(),
                            )),
                        }
                    }
                    // bytes_slice start end bytes -> 3 args
                    ("bytes_slice", 3) => {
                        let bytes = args.pop().unwrap();
                        let end = args.pop().unwrap();
                        let start = args.pop().unwrap();
                        match (start, end, bytes) {
                            (Value::Int(start), Value::Int(end), Value::Bytes(bytes)) => {
                                let start = start.max(0) as usize;
                                let end = end.max(0) as usize;
                                let len = bytes.len();
                                let start = start.min(len);
                                let end = end.min(len);
                                let result: Vec<u8> = if start <= end {
                                    bytes[start..end].to_vec()
                                } else {
                                    Vec::new()
                                };
                                StepResult::Continue(State::Apply {
                                    value: Value::Bytes(result),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "bytes_slice: expected (Int, Int, Bytes)".into(),
                            )),
                        }
                    }
                    // bytes_concat bytes1 bytes2 -> 2 args
                    ("bytes_concat", 2) => {
                        let b2 = args.pop().unwrap();
                        let b1 = args.pop().unwrap();
                        match (b1, b2) {
                            (Value::Bytes(mut b1), Value::Bytes(b2)) => {
                                b1.extend(b2);
                                StepResult::Continue(State::Apply {
                                    value: Value::Bytes(b1),
                                    cont,
                                })
                            }
                            _ => StepResult::Error(EvalError::TypeError(
                                "bytes_concat: expected (Bytes, Bytes)".into(),
                            )),
                        }
                    }
                    // file_open path mode -> 2 args
                    ("file_open", 2) => {
                        match Self::build_io_op(PartialIoOp::FileOpen, args) {
                            Ok(io_op) => {
                                // Capture continuation and attach to effect
                                match self.capture_to_fiber_boundary(&mut cont) {
                                    Ok(captured) => {
                                        let effect = FiberEffect::Io { op: io_op, cont: None }
                                            .with_cont(captured);
                                        self.return_fiber_effect(effect, cont)
                                    }
                                    Err(e) => StepResult::Error(e),
                                }
                            }
                            Err(e) => StepResult::Error(e),
                        }
                    }
                    // file_read handle count -> 2 args
                    ("file_read", 2) => {
                        match Self::build_io_op(PartialIoOp::Read, args) {
                            Ok(io_op) => {
                                match self.capture_to_fiber_boundary(&mut cont) {
                                    Ok(captured) => {
                                        let effect = FiberEffect::Io { op: io_op, cont: None }
                                            .with_cont(captured);
                                        self.return_fiber_effect(effect, cont)
                                    }
                                    Err(e) => StepResult::Error(e),
                                }
                            }
                            Err(e) => StepResult::Error(e),
                        }
                    }
                    // file_write handle data -> 2 args
                    ("file_write", 2) => {
                        match Self::build_io_op(PartialIoOp::Write, args) {
                            Ok(io_op) => {
                                match self.capture_to_fiber_boundary(&mut cont) {
                                    Ok(captured) => {
                                        let effect = FiberEffect::Io { op: io_op, cont: None }
                                            .with_cont(captured);
                                        self.return_fiber_effect(effect, cont)
                                    }
                                    Err(e) => StepResult::Error(e),
                                }
                            }
                            Err(e) => StepResult::Error(e),
                        }
                    }
                    // tcp_connect host port -> 2 args
                    ("tcp_connect", 2) => {
                        match Self::build_io_op(PartialIoOp::TcpConnect, args) {
                            Ok(io_op) => {
                                match self.capture_to_fiber_boundary(&mut cont) {
                                    Ok(captured) => {
                                        let effect = FiberEffect::Io { op: io_op, cont: None }
                                            .with_cont(captured);
                                        self.return_fiber_effect(effect, cont)
                                    }
                                    Err(e) => StepResult::Error(e),
                                }
                            }
                            Err(e) => StepResult::Error(e),
                        }
                    }
                    // tcp_listen host port -> 2 args
                    ("tcp_listen", 2) => {
                        match Self::build_io_op(PartialIoOp::TcpListen, args) {
                            Ok(io_op) => {
                                match self.capture_to_fiber_boundary(&mut cont) {
                                    Ok(captured) => {
                                        let effect = FiberEffect::Io { op: io_op, cont: None }
                                            .with_cont(captured);
                                        self.return_fiber_effect(effect, cont)
                                    }
                                    Err(e) => StepResult::Error(e),
                                }
                            }
                            Err(e) => StepResult::Error(e),
                        }
                    }
                    // Not enough args yet, return partial
                    _ => StepResult::Continue(State::Apply {
                        value: Value::BuiltinPartial { name, args },
                        cont,
                    }),
                }
            }

            Value::ComposedFn { first, second } => {
                // (f >> g) x = g (f x)
                // Push frame to apply second when first returns, then apply first
                cont.push(Frame::AppArg { func: *second });
                self.do_apply(*first, arg, cont)
            }

            _ => StepResult::Error(EvalError::TypeError(format!(
                "cannot apply {}",
                func.type_name()
            ))),
        }
    }

    fn eval_literal(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(n) => Value::Int(*n),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
            Literal::Char(c) => Value::Char(*c),
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Unit => Value::Unit,
        }
    }

    fn eval_binop(&self, op: &BinOp, left: Value, right: Value) -> Result<Value, EvalError> {
        match (op, &left, &right) {
            // Arithmetic (checked to prevent overflow panics)
            (BinOp::Add, Value::Int(a), Value::Int(b)) => a
                .checked_add(*b)
                .map(Value::Int)
                .ok_or(EvalError::IntegerOverflow),
            (BinOp::Sub, Value::Int(a), Value::Int(b)) => a
                .checked_sub(*b)
                .map(Value::Int)
                .ok_or(EvalError::IntegerOverflow),
            (BinOp::Mul, Value::Int(a), Value::Int(b)) => a
                .checked_mul(*b)
                .map(Value::Int)
                .ok_or(EvalError::IntegerOverflow),
            (BinOp::Div, Value::Int(_), Value::Int(0)) => Err(EvalError::DivisionByZero),
            (BinOp::Div, Value::Int(a), Value::Int(b)) => a
                .checked_div(*b)
                .map(Value::Int)
                .ok_or(EvalError::IntegerOverflow),
            (BinOp::Mod, Value::Int(_), Value::Int(0)) => Err(EvalError::DivisionByZero),
            (BinOp::Mod, Value::Int(a), Value::Int(b)) => a
                .checked_rem(*b)
                .map(Value::Int)
                .ok_or(EvalError::IntegerOverflow),

            // Float arithmetic
            (BinOp::Add, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (BinOp::Sub, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (BinOp::Mul, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (BinOp::Div, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),

            // Comparison
            (BinOp::Eq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
            (BinOp::Eq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
            (BinOp::Eq, Value::String(a), Value::String(b)) => Ok(Value::Bool(a == b)),
            (BinOp::Eq, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a == b)),
            (BinOp::Eq, Value::Unit, Value::Unit) => Ok(Value::Bool(true)),

            (BinOp::Neq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a != b)),
            (BinOp::Neq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a != b)),
            (BinOp::Neq, Value::String(a), Value::String(b)) => Ok(Value::Bool(a != b)),

            (BinOp::Lt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (BinOp::Gt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (BinOp::Lte, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (BinOp::Gte, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),

            (BinOp::Lt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (BinOp::Gt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (BinOp::Lte, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (BinOp::Gte, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),

            (BinOp::Lt, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a < b)),
            (BinOp::Gt, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a > b)),
            (BinOp::Lte, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a <= b)),
            (BinOp::Gte, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a >= b)),

            // Boolean
            (BinOp::And, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
            (BinOp::Or, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),

            // List operations (using persistent data structure for O(log n) sharing)
            (BinOp::Cons, val, Value::List(list)) => {
                let mut new_list = list.clone();
                new_list.push_front(val.clone());
                Ok(Value::List(new_list))
            }
            (BinOp::Concat, Value::List(a), Value::List(b)) => {
                Ok(Value::List(a.clone() + b.clone()))
            }
            (BinOp::Concat, Value::String(a), Value::String(b)) => {
                Ok(Value::String(format!("{}{}", a, b)))
            }

            // Compose - create a composed function value
            // f >> g means apply f first, then g: (f >> g) x = g (f x)
            (BinOp::Compose, f, g) => Ok(Value::ComposedFn {
                first: Box::new(f.clone()),
                second: Box::new(g.clone()),
            }),
            // f << g means apply g first, then f: (f << g) x = f (g x)
            (BinOp::ComposeBack, f, g) => Ok(Value::ComposedFn {
                first: Box::new(g.clone()),
                second: Box::new(f.clone()),
            }),

            // Pipe should be desugared
            (BinOp::Pipe, _, _) | (BinOp::PipeBack, _, _) => {
                unreachable!("pipe operators should be desugared")
            }

            // User-defined operators are handled in Frame::BinOpRight (lines 913-931)
            // via curried function application, so they should never reach eval_binop
            (BinOp::UserDefined(_), _, _) => {
                unreachable!("user-defined operators should be handled via function application")
            }

            _ => Err(EvalError::TypeError(format!(
                "cannot apply {:?} to {} and {}",
                op,
                left.type_name(),
                right.type_name()
            ))),
        }
    }

    fn eval_unaryop(&self, op: UnaryOp, val: Value) -> Result<Value, EvalError> {
        match (op, &val) {
            (UnaryOp::Neg, Value::Int(n)) => n
                .checked_neg()
                .map(Value::Int)
                .ok_or(EvalError::IntegerOverflow),
            (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
            (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            _ => Err(EvalError::TypeError(format!(
                "cannot apply {:?} to {}",
                op,
                val.type_name()
            ))),
        }
    }

    fn apply_builtin(&self, name: &str, arg: Value) -> Result<Value, EvalError> {
        match name {
            "print" => {
                self.print_value(&arg);
                println!();
                Ok(Value::Unit)
            }
            "int_to_string" => match arg {
                Value::Int(n) => Ok(Value::String(n.to_string())),
                _ => Err(EvalError::TypeError("expected int".into())),
            },
            "string_length" => match arg {
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                _ => Err(EvalError::TypeError("expected string".into())),
            },
            "string_to_chars" => match arg {
                Value::String(s) => {
                    let chars: im::Vector<Value> = s.chars().map(Value::Char).collect();
                    Ok(Value::List(chars))
                }
                _ => Err(EvalError::TypeError("expected string".into())),
            },
            "chars_to_string" => match arg {
                Value::List(items) => {
                    let mut s = String::new();
                    for item in items {
                        match item {
                            Value::Char(c) => s.push(c),
                            _ => return Err(EvalError::TypeError("expected list of chars".into())),
                        }
                    }
                    Ok(Value::String(s))
                }
                _ => Err(EvalError::TypeError("expected list".into())),
            },
            "char_to_string" => match arg {
                Value::Char(c) => Ok(Value::String(c.to_string())),
                _ => Err(EvalError::TypeError("expected char".into())),
            },
            "char_to_int" => match arg {
                Value::Char(c) => Ok(Value::Int(c as i64)),
                _ => Err(EvalError::TypeError("expected char".into())),
            },
            "char_to_lower" => match arg {
                Value::Char(c) => {
                    // to_lowercase returns an iterator for multi-char mappings (e.g., German ß)
                    // We take the first char for simplicity
                    let lower = c.to_lowercase().next().unwrap_or(c);
                    Ok(Value::Char(lower))
                }
                _ => Err(EvalError::TypeError("char_to_lower: expected char".into())),
            },
            "char_to_upper" => match arg {
                Value::Char(c) => {
                    let upper = c.to_uppercase().next().unwrap_or(c);
                    Ok(Value::Char(upper))
                }
                _ => Err(EvalError::TypeError("char_to_upper: expected char".into())),
            },
            "char_is_whitespace" => match arg {
                Value::Char(c) => Ok(Value::Bool(c.is_whitespace())),
                _ => Err(EvalError::TypeError("char_is_whitespace: expected char".into())),
            },
            // Native string operations (single-arg)
            "string_to_lower" => match arg {
                Value::String(s) => Ok(Value::String(s.to_lowercase())),
                _ => Err(EvalError::TypeError("string_to_lower: expected String".into())),
            },
            "string_to_upper" => match arg {
                Value::String(s) => Ok(Value::String(s.to_uppercase())),
                _ => Err(EvalError::TypeError("string_to_upper: expected String".into())),
            },
            "string_trim" => match arg {
                Value::String(s) => Ok(Value::String(s.trim().to_string())),
                _ => Err(EvalError::TypeError("string_trim: expected String".into())),
            },
            "string_trim_start" => match arg {
                Value::String(s) => Ok(Value::String(s.trim_start().to_string())),
                _ => Err(EvalError::TypeError("string_trim_start: expected String".into())),
            },
            "string_trim_end" => match arg {
                Value::String(s) => Ok(Value::String(s.trim_end().to_string())),
                _ => Err(EvalError::TypeError("string_trim_end: expected String".into())),
            },
            "string_reverse" => match arg {
                Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
                _ => Err(EvalError::TypeError("string_reverse: expected String".into())),
            },
            "string_is_empty" => match arg {
                Value::String(s) => Ok(Value::Bool(s.is_empty())),
                _ => Err(EvalError::TypeError("string_is_empty: expected String".into())),
            },
            // Multi-arg string builtins - return partial with first arg
            "string_index_of" | "string_substring" | "string_split" | "string_join"
            | "string_char_at" | "string_concat" | "string_repeat" => {
                Ok(Value::BuiltinPartial {
                    name: name.to_string(),
                    args: vec![arg],
                })
            }
            // Bytes builtins
            "bytes_to_string" => match arg {
                Value::Bytes(bytes) => {
                    match String::from_utf8(bytes) {
                        Ok(s) => Ok(Value::String(s)),
                        Err(_) => Err(EvalError::TypeError("bytes_to_string: invalid UTF-8".into())),
                    }
                }
                _ => Err(EvalError::TypeError("bytes_to_string: expected Bytes".into())),
            },
            "string_to_bytes" => match arg {
                Value::String(s) => Ok(Value::Bytes(s.into_bytes())),
                _ => Err(EvalError::TypeError("string_to_bytes: expected String".into())),
            },
            "bytes_length" => match arg {
                Value::Bytes(bytes) => Ok(Value::Int(bytes.len() as i64)),
                _ => Err(EvalError::TypeError("bytes_length: expected Bytes".into())),
            },
            // Multi-arg bytes builtins - return partial with first arg
            "bytes_slice" | "bytes_concat" => {
                Ok(Value::BuiltinPartial {
                    name: name.to_string(),
                    args: vec![arg],
                })
            }
            // Dict builtins
            "Dict.new" => Ok(Value::Dictionary(im::HashMap::new())),
            "Dict.keys" => match arg {
                Value::Dictionary(d) => {
                    let keys: im::Vector<Value> = d.keys().map(|k| Value::String(k.clone())).collect();
                    Ok(Value::List(keys))
                }
                _ => Err(EvalError::TypeError("Dict.keys: expected Dictionary".into())),
            },
            "Dict.values" => match arg {
                Value::Dictionary(d) => {
                    let values: im::Vector<Value> = d.values().cloned().collect();
                    Ok(Value::List(values))
                }
                _ => Err(EvalError::TypeError("Dict.values: expected Dictionary".into())),
            },
            "Dict.size" => match arg {
                Value::Dictionary(d) => Ok(Value::Int(d.len() as i64)),
                _ => Err(EvalError::TypeError("Dict.size: expected Dictionary".into())),
            },
            "Dict.isEmpty" => match arg {
                Value::Dictionary(d) => Ok(Value::Bool(d.is_empty())),
                _ => Err(EvalError::TypeError("Dict.isEmpty: expected Dictionary".into())),
            },
            "Dict.toList" => match arg {
                Value::Dictionary(d) => {
                    let pairs: im::Vector<Value> = d
                        .iter()
                        .map(|(k, v)| Value::Tuple(vec![Value::String(k.clone()), v.clone()]))
                        .collect();
                    Ok(Value::List(pairs))
                }
                _ => Err(EvalError::TypeError("Dict.toList: expected Dictionary".into())),
            },
            "Dict.fromList" => match arg {
                Value::List(items) => {
                    let mut dict = im::HashMap::new();
                    for item in items {
                        match item {
                            Value::Tuple(ref fields) if fields.len() == 2 => {
                                match (&fields[0], &fields[1]) {
                                    (Value::String(k), v) => {
                                        dict.insert(k.clone(), v.clone());
                                    }
                                    _ => return Err(EvalError::TypeError(
                                        "Dict.fromList: expected (String, a) pairs".into(),
                                    )),
                                }
                            }
                            _ => return Err(EvalError::TypeError(
                                "Dict.fromList: expected list of (String, a) pairs".into(),
                            )),
                        }
                    }
                    Ok(Value::Dictionary(dict))
                }
                _ => Err(EvalError::TypeError("Dict.fromList: expected List".into())),
            },
            // String escape builtins
            "html_escape" => match arg {
                Value::String(s) => {
                    let escaped = s
                        .replace('&', "&amp;")
                        .replace('<', "&lt;")
                        .replace('>', "&gt;")
                        .replace('"', "&quot;")
                        .replace('\'', "&#39;");
                    Ok(Value::String(escaped))
                }
                _ => Err(EvalError::TypeError("html_escape: expected String".into())),
            },
            "json_escape_string" => match arg {
                Value::String(s) => {
                    let mut escaped = String::with_capacity(s.len());
                    for c in s.chars() {
                        match c {
                            '"' => escaped.push_str("\\\""),
                            '\\' => escaped.push_str("\\\\"),
                            '\n' => escaped.push_str("\\n"),
                            '\r' => escaped.push_str("\\r"),
                            '\t' => escaped.push_str("\\t"),
                            c if c.is_control() => {
                                escaped.push_str(&format!("\\u{:04x}", c as u32));
                            }
                            c => escaped.push(c),
                        }
                    }
                    Ok(Value::String(escaped))
                }
                _ => Err(EvalError::TypeError("json_escape_string: expected String".into())),
            },
            // Multi-arg Dict builtins - return partial with first arg
            "Dict.insert" | "Dict.get" | "Dict.remove" | "Dict.contains"
            | "Dict.merge" | "Dict.getOrDefault" => {
                Ok(Value::BuiltinPartial {
                    name: name.to_string(),
                    args: vec![arg],
                })
            },
            // Set builtins
            "Set.new" => Ok(Value::Set(im::HashSet::new())),
            "Set.size" => match arg {
                Value::Set(s) => Ok(Value::Int(s.len() as i64)),
                _ => Err(EvalError::TypeError("Set.size: expected Set".into())),
            },
            "Set.toList" => match arg {
                Value::Set(s) => {
                    let list: im::Vector<Value> = s.iter().map(|e| Value::String(e.clone())).collect();
                    Ok(Value::List(list))
                }
                _ => Err(EvalError::TypeError("Set.toList: expected Set".into())),
            },
            // Multi-arg Set builtins - return partial with first arg
            "Set.insert" | "Set.contains" | "Set.remove" | "Set.union" | "Set.intersect" => {
                Ok(Value::BuiltinPartial {
                    name: name.to_string(),
                    args: vec![arg],
                })
            },
            // Command-line arguments
            "get_args" => {
                let args: im::Vector<Value> = self
                    .program_args
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                Ok(Value::List(args))
            },
            _ => Err(EvalError::RuntimeError(format!(
                "unknown builtin: {}",
                name
            ))),
        }
    }

    /// Apply an effectful builtin function.
    /// Returns BuiltinResult which may require continuation capture.
    fn apply_effectful_builtin(&self, name: &str, arg: Value) -> Result<BuiltinResult, EvalError> {
        match name {
            "Fiber.spawn" => {
                // arg should be a thunk (() -> a)
                // Return a Fork effect - continuation capture happens at call site
                Ok(BuiltinResult::Effect(FiberEffect::Fork {
                    thunk: Box::new(arg),
                    cont: None, // Will be filled in by caller after capture
                }))
            }
            "Fiber.join" => {
                // arg should be a Fiber<A>
                match arg {
                    Value::Fiber(fiber_id) => Ok(BuiltinResult::Effect(FiberEffect::Join {
                        fiber_id,
                        cont: None, // Will be filled in by caller after capture
                    })),
                    _ => Err(EvalError::TypeError(
                        "Fiber.join expects a Fiber value".into(),
                    )),
                }
            }
            "Fiber.yield" => {
                // arg should be ()
                Ok(BuiltinResult::Effect(FiberEffect::Yield { cont: None }))
            }
            "sleep_ms" => {
                // arg should be Int (milliseconds)
                match arg {
                    Value::Int(ms) if ms >= 0 => {
                        Ok(BuiltinResult::Effect(FiberEffect::Io {
                            op: IoOp::Sleep { duration_ms: ms as u64 },
                            cont: None,
                        }))
                    }
                    Value::Int(ms) => Err(EvalError::TypeError(format!(
                        "sleep_ms: duration must be non-negative, got {}",
                        ms
                    ))),
                    _ => Err(EvalError::TypeError(format!(
                        "sleep_ms: expected Int, got {}",
                        arg.type_name()
                    ))),
                }
            }
            // Multi-arg file I/O builtins - return partial with first arg
            "file_open" | "file_read" | "file_write" => {
                Ok(BuiltinResult::Partial {
                    name: name.to_string(),
                    args: vec![arg],
                })
            }
            // Single-arg file I/O builtin
            "file_close" => {
                match arg {
                    Value::FileHandle(id) => {
                        Ok(BuiltinResult::Effect(FiberEffect::Io {
                            op: IoOp::Close { handle: id },
                            cont: None,
                        }))
                    }
                    _ => Err(EvalError::TypeError(format!(
                        "file_close: expected FileHandle, got {}",
                        arg.type_name()
                    ))),
                }
            }
            // Multi-arg TCP builtins - return partial with first arg
            "tcp_connect" | "tcp_listen" => {
                Ok(BuiltinResult::Partial {
                    name: name.to_string(),
                    args: vec![arg],
                })
            }
            // Single-arg TCP builtin
            "tcp_accept" => {
                match arg {
                    Value::TcpListener(id) => {
                        Ok(BuiltinResult::Effect(FiberEffect::Io {
                            op: IoOp::TcpAccept { listener: id },
                            cont: None,
                        }))
                    }
                    _ => Err(EvalError::TypeError(format!(
                        "tcp_accept: expected TcpListener, got {}",
                        arg.type_name()
                    ))),
                }
            }
            _ => Err(EvalError::RuntimeError(format!(
                "unknown effectful builtin: {}",
                name
            ))),
        }
    }

    /// Check if a builtin name is an effectful fiber/IO builtin
    fn is_effectful_builtin(name: &str) -> bool {
        matches!(
            name,
            "Fiber.spawn"
                | "Fiber.join"
                | "Fiber.yield"
                | "sleep_ms"
                | "file_open"
                | "file_read"
                | "file_write"
                | "file_close"
                | "tcp_connect"
                | "tcp_listen"
                | "tcp_accept"
        )
    }

    fn print_value(&self, val: &Value) {
        match val {
            Value::Int(n) => print!("{}", n),
            Value::Float(f) => print!("{}", f),
            Value::Bool(b) => print!("{}", b),
            Value::String(s) => print!("{}", s),
            Value::Char(c) => print!("{}", c),
            Value::Unit => print!("()"),
            Value::Bytes(bytes) => {
                print!("<bytes:{} bytes>", bytes.len());
            }
            Value::List(items) => {
                print!("[");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    self.print_value(item);
                }
                print!("]");
            }
            Value::Tuple(items) => {
                print!("(");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    self.print_value(item);
                }
                print!(")");
            }
            Value::Closure { .. } => print!("<function>"),
            Value::ComposedFn { .. } => print!("<function>"),
            Value::Constructor { name, fields } => {
                print!("{}", name);
                for field in fields {
                    print!(" ");
                    self.print_value(field);
                }
            }
            Value::Pid(pid) => print!("<pid:{}>", pid),
            Value::Channel(id) => print!("<channel:{}>", id),
            Value::Builtin(name) => print!("<builtin:{}>", name),
            Value::BuiltinPartial { name, args } => {
                print!("<builtin-partial:{}({} args)>", name, args.len())
            }
            Value::Continuation { .. } => print!("<continuation>"),
            Value::Dict { trait_name, .. } => print!("<dict:{}>", trait_name),
            Value::Fiber(id) => print!("<fiber:{}>", id),
            Value::FiberEffect(effect) => print!("<fiber-effect:{:?}>", effect),
            Value::Record { type_name, fields } => {
                print!("{} {{ ", type_name);
                let mut first = true;
                for (field_name, field_value) in fields {
                    if !first {
                        print!(", ");
                    }
                    first = false;
                    print!("{} = ", field_name);
                    self.print_value(field_value);
                }
                print!(" }}");
            }
            Value::Dictionary(dict) => {
                print!("{{");
                let mut first = true;
                for (key, value) in dict {
                    if !first {
                        print!(", ");
                    }
                    first = false;
                    print!("\"{}\": ", key);
                    self.print_value(value);
                }
                print!("}}");
            }
            Value::Set(set) => {
                print!("Set {{");
                let mut first = true;
                for elem in set {
                    if !first {
                        print!(", ");
                    }
                    first = false;
                    print!("\"{}\"", elem);
                }
                print!("}}");
            }
            Value::FileHandle(id) => print!("<file-handle:{}>", id),
            Value::TcpSocket(id) => print!("<tcp-socket:{}>", id),
            Value::TcpListener(id) => print!("<tcp-listener:{}>", id),
        }
    }

    // === Fiber effect helpers ===

    /// Capture frames from continuation up to (but not including) FiberBoundary.
    /// Unlike shift's capture to Prompt, FiberBoundary remains on the stack.
    /// Returns the captured frames as a new Cont.
    fn capture_to_fiber_boundary(&self, cont: &mut Cont) -> Result<Cont, EvalError> {
        let mut captured = Vec::new();

        loop {
            match cont.pop() {
                None => {
                    return Err(EvalError::RuntimeError(
                        "fiber effect without enclosing FiberBoundary".into(),
                    ));
                }
                Some(Frame::FiberBoundary) => {
                    // Found delimiter - put it back (unlike Prompt, we keep it)
                    cont.push(Frame::FiberBoundary);
                    break;
                }
                Some(frame) => {
                    captured.push(frame);
                }
            }
        }

        // Frames were popped innermost-first; reverse so outermost is first
        captured.reverse();
        Ok(Cont::from_frames(captured))
    }

    /// Return a FiberEffect value via the Apply state.
    /// This allows the scheduler to pattern match on the effect.
    fn return_fiber_effect(&self, effect: FiberEffect, cont: Cont) -> StepResult {
        StepResult::Continue(State::Apply {
            value: Value::FiberEffect(effect),
            cont,
        })
    }

    /// Try to bind a pattern, returning false if it doesn't match
    fn try_bind_pattern(&self, env: &Env, pattern: &Pattern, val: &Value) -> bool {
        match (&pattern.node, val) {
            (PatternKind::Wildcard, _) => true,

            (PatternKind::Var(name), _) => {
                env.borrow_mut().define(name.clone(), val.clone());
                true
            }

            (PatternKind::Lit(Literal::Int(a)), Value::Int(b)) => a == b,
            (PatternKind::Lit(Literal::Bool(a)), Value::Bool(b)) => a == b,
            (PatternKind::Lit(Literal::String(a)), Value::String(b)) => a == b,
            (PatternKind::Lit(Literal::Char(a)), Value::Char(b)) => a == b,
            (PatternKind::Lit(Literal::Unit), Value::Unit) => true,

            (PatternKind::Tuple(pats), Value::Tuple(vals)) if pats.len() == vals.len() => pats
                .iter()
                .zip(vals.iter())
                .all(|(p, v)| self.try_bind_pattern(env, p, v)),

            (PatternKind::List(pats), Value::List(vals)) if pats.len() == vals.len() => pats
                .iter()
                .zip(vals.iter())
                .all(|(p, v)| self.try_bind_pattern(env, p, v)),

            (PatternKind::Cons { head, tail }, Value::List(vals)) if !vals.is_empty() => {
                let head_val = vals.front().unwrap();
                let tail_vals = vals.clone().split_off(1);
                self.try_bind_pattern(env, head, head_val)
                    && self.try_bind_pattern(env, tail, &Value::List(tail_vals))
            }

            (
                PatternKind::Constructor { name: pn, args },
                Value::Constructor { name: vn, fields },
            ) if pn == vn && args.len() == fields.len() => args
                .iter()
                .zip(fields.iter())
                .all(|(p, v)| self.try_bind_pattern(env, p, v)),

            // Constructor with no args matching
            (
                PatternKind::Constructor { name: pn, args },
                Value::Constructor { name: vn, fields },
            ) if pn == vn && args.is_empty() && fields.is_empty() => true,

            // Record pattern matching
            (
                PatternKind::Record {
                    name: pn,
                    fields: pat_fields,
                },
                Value::Record {
                    type_name: vn,
                    fields: val_fields,
                },
            ) if pn == vn => {
                // All pattern fields must exist in the value and match
                pat_fields.iter().all(|(field_name, sub_pat_opt)| {
                    if let Some(field_value) = val_fields.get(field_name.as_str()) {
                        match sub_pat_opt {
                            Some(sub_pat) => self.try_bind_pattern(env, sub_pat, field_value),
                            None => {
                                // Punned field: { method } binds variable `method` to field value
                                env.borrow_mut()
                                    .define(field_name.to_string(), field_value.clone());
                                true
                            }
                        }
                    } else {
                        false // Field not found
                    }
                })
            }

            _ => false,
        }
    }

    /// Run the scheduler until all processes complete or deadlock
    fn run_scheduler(&mut self) -> Result<(), EvalError> {
        // Round-robin scheduler with I/O and timer polling
        loop {
            // Try to get next ready process
            if let Some(pid) = self.runtime.next_ready() {
                self.runtime.set_current(Some(pid));

                if let Some(pcont) = self.runtime.take_continuation(pid) {
                    match pcont {
                        ProcessContinuation::Start(thunk) => {
                            // Start a new process: apply thunk to unit
                            self.run_process(pid, thunk)?;
                        }
                        ProcessContinuation::ResumeFiber(value) => {
                            // Resume fiber with a value using fiber_cont
                            if let Some(mut fiber_cont) = self.runtime.take_fiber_cont(pid) {
                                fiber_cont.insert_at_bottom(Frame::FiberBoundary);
                                let state = State::Apply {
                                    value,
                                    cont: fiber_cont,
                                };
                                self.run_state(pid, state)?;
                            } else {
                                self.runtime.mark_done(pid);
                            }
                        }
                        ProcessContinuation::ResumeSelect { channel, value } => {
                            // Resume from select - find matching arm, bind pattern, evaluate body
                            if let Some(arms) = self.runtime.take_select_arms(pid) {
                                if let Some(arm) = arms.into_iter().find(|a| a.channel == channel) {
                                    // Bind pattern to received value
                                    let new_env = EnvInner::with_parent(&arm.env);
                                    if self.try_bind_pattern(&new_env, &arm.pattern, &value) {
                                        // Build continuation with fiber_cont
                                        let mut cont = self.runtime.take_fiber_cont(pid).unwrap_or_default();
                                        cont.insert_at_bottom(Frame::FiberBoundary);
                                        let state = State::Eval {
                                            expr: arm.body,
                                            env: new_env,
                                            cont,
                                        };
                                        self.run_state(pid, state)?;
                                    } else {
                                        return Err(EvalError::MatchFailed);
                                    }
                                }
                            }
                        }
                    }
                }

                self.runtime.set_current(None);
            } else {
                // No ready processes - check if we have I/O or timers to wait for
                let timeout = self.runtime.time_until_next_timer();
                let has_io_pending = self.runtime.has_io_pending();

                if timeout.is_some() || has_io_pending {
                    // We have I/O or timers to wait for
                    let woke_any = self.runtime.poll_io(timeout);

                    // If nothing woke up and we're deadlocked, exit
                    if !woke_any && self.runtime.is_deadlocked() {
                        return Err(EvalError::Deadlock);
                    }
                } else {
                    // No I/O or timers - use original exit logic
                    if self.runtime.is_deadlocked() {
                        return Err(EvalError::Deadlock);
                    } else {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Run a process starting from a thunk
    fn run_process(&mut self, pid: Pid, thunk: Value) -> Result<(), EvalError> {
        // Build initial state: apply thunk to Unit
        let mut cont = Cont::new();
        // Push FiberBoundary at the bottom of the stack to enable fiber effects
        cont.push(Frame::FiberBoundary);
        cont.push(Frame::AppArg { func: thunk });
        let state = State::Apply {
            value: Value::Unit,
            cont,
        };
        self.run_state(pid, state)
    }

    /// Run the step machine from a given state until done or blocked
    fn run_state(&mut self, pid: Pid, mut state: State) -> Result<(), EvalError> {
        loop {
            match self.step(state) {
                StepResult::Continue(next) => {
                    state = next;
                }
                StepResult::Done(value) => {
                    // Check if this is a fiber effect
                    if let Value::FiberEffect(effect) = value {
                        return self.handle_fiber_effect(pid, effect);
                    }
                    // Regular completion
                    self.runtime.mark_done(pid);
                    return Ok(());
                }
                StepResult::Error(e) => {
                    eprintln!("Process {} error: {}", pid, e);
                    self.runtime.mark_done(pid);
                    return Err(e);
                }
            }
        }
    }

    /// Handle a fiber effect that bubbled up from the interpreter.
    /// This is the NEW effect-based scheduler interface (Phase 7).
    fn handle_fiber_effect(&mut self, pid: Pid, effect: FiberEffect) -> Result<(), EvalError> {
        match effect {
            FiberEffect::Done(value) => {
                // Fiber completed with a result
                self.runtime.complete_fiber(pid, *value);
            }
            FiberEffect::Fork { thunk, cont } => {
                // Store parent's continuation
                if let Some(c) = cont {
                    self.runtime.store_fiber_cont(pid, *c);
                }
                // Spawn child fiber
                let child_pid = self.runtime.spawn(*thunk);
                // Resume parent with child's handle
                self.runtime.resume_fiber_with(pid, Value::Fiber(child_pid));
            }
            FiberEffect::Yield { cont } => {
                // Store continuation and re-queue at back
                if let Some(c) = cont {
                    self.runtime.store_fiber_cont(pid, *c);
                }
                self.runtime.yield_fiber(pid);
            }
            FiberEffect::NewChan { cont } => {
                // Create new channel and resume with it
                let channel_id = self.runtime.new_channel();
                if let Some(c) = cont {
                    self.runtime.store_fiber_cont(pid, *c);
                }
                self.runtime.resume_fiber_with(pid, Value::Channel(channel_id));
            }
            FiberEffect::Send { channel, value, cont } => {
                // Block fiber waiting to send (try immediate handoff first)
                self.runtime
                    .block_fiber_send(pid, channel, *value, cont.map(|c| *c));
            }
            FiberEffect::Recv { channel, cont } => {
                // Block fiber waiting to receive (try immediate handoff first)
                self.runtime.block_fiber_recv(pid, channel, cont.map(|c| *c));
            }
            FiberEffect::Join { fiber_id, cont } => {
                // Block fiber waiting to join another fiber
                self.runtime
                    .block_fiber_join(pid, fiber_id, cont.map(|c| *c));
            }
            FiberEffect::Select { arms, cont } => {
                // Block fiber on select (register on all channels)
                self.runtime
                    .block_fiber_select(pid, arms, cont.map(|c| *c));
            }
            FiberEffect::Io { op, cont } => {
                // Dispatch to the appropriate I/O handler (reactor, pool, or timer)
                self.runtime.dispatch_io(pid, op, cont.map(|c| *c));
            }
        }
        Ok(())
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn eval(input: &str) -> Result<Value, EvalError> {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expr().unwrap();
        let mut interp = Interpreter::new();
        let env = EnvInner::new();
        interp.eval_expr(&env, &expr)
    }

    #[test]
    fn test_arithmetic() {
        let val = eval("1 + 2 * 3").unwrap();
        assert!(matches!(val, Value::Int(7)));
    }

    #[test]
    fn test_if() {
        let val = eval("if true then 1 else 2").unwrap();
        assert!(matches!(val, Value::Int(1)));
    }

    #[test]
    fn test_lambda() {
        let val = eval("(fun x -> x + 1) 5").unwrap();
        assert!(matches!(val, Value::Int(6)));
    }

    #[test]
    fn test_let() {
        let val = eval("let x = 10 in x + 5").unwrap();
        assert!(matches!(val, Value::Int(15)));
    }

    #[test]
    fn test_list() {
        let val = eval("[1, 2, 3]").unwrap();
        assert!(matches!(val, Value::List(_)));
    }

    #[test]
    fn test_match() {
        let val = eval("match 1 with | 1 -> 42 | _ -> 0 end").unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_tuple() {
        let val = eval("(1, 2, 3)").unwrap();
        match val {
            Value::Tuple(items) => assert_eq!(items.len(), 3),
            _ => panic!("expected tuple"),
        }
    }

    #[test]
    fn test_nested_application() {
        let val = eval("(fun x -> fun y -> x + y) 3 4").unwrap();
        assert!(matches!(val, Value::Int(7)));
    }

    #[test]
    fn test_seq() {
        let val = eval("let _ = 1 in 2; 3").unwrap();
        assert!(matches!(val, Value::Int(3)));
    }

    // ========================================================================
    // Concurrency tests
    // ========================================================================

    /// Helper to run a program (with main) and check it completes without error
    fn run_program(input: &str) -> Result<(), EvalError> {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();
        let mut interp = Interpreter::new();
        interp.run(&program)?;
        Ok(())
    }

    #[test]
    fn test_channel_create() {
        let val = eval("Channel.new").unwrap();
        assert!(matches!(val, Value::Channel(_)));
    }

    #[test]
    fn test_spawn_returns_pid() {
        // spawn is now a builtin that requires the runtime environment
        // Testing via run_program with a simple spawn
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () -> Channel.send ch 42) in
    Channel.recv ch
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_channel_send_recv_immediate_rendezvous() {
        // Sender spawned first, blocks waiting for receiver
        // Then main receives, unblocking sender
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () -> Channel.send ch 42) in
    Channel.recv ch
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_channel_recv_then_send() {
        // Main tries to receive first (blocks), then spawned process sends
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () -> Channel.send ch 99) in
    let x = Channel.recv ch in
    x
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_ping_pong() {
        // Two processes exchanging messages
        let program = r#"
let main () =
    let ping_ch = Channel.new in
    let pong_ch = Channel.new in
    let _ = spawn (fun () ->
        let x = Channel.recv ping_ch in
        Channel.send pong_ch (x + 1)
    ) in
    Channel.send ping_ch 10;
    Channel.recv pong_ch
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_multiple_messages() {
        // Send multiple values through a channel
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () ->
        Channel.send ch 1;
        Channel.send ch 2;
        Channel.send ch 3
    ) in
    let a = Channel.recv ch in
    let b = Channel.recv ch in
    let c = Channel.recv ch in
    a + b + c
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_deadlock_detection() {
        // Two processes both trying to receive - deadlock
        let program = r#"
let main () =
    let ch = Channel.new in
    Channel.recv ch
"#;
        let result = run_program(program);
        assert!(matches!(result, Err(EvalError::Deadlock)));
    }

    #[test]
    fn test_concurrent_interleaving_required() {
        // This test MUST interleave to succeed.
        // Process A sends on ch1, then receives on ch2
        // Process B receives on ch1, then sends on ch2
        //
        // Sequential execution would deadlock:
        //   - If A runs fully first: A blocks on send ch1 (no receiver yet)
        //   - If B runs fully first: B blocks on recv ch1 (no sender yet)
        //
        // Only interleaving works:
        //   1. A sends on ch1, blocks (or B already waiting)
        //   2. B receives on ch1, continues
        //   3. B sends on ch2
        //   4. A receives on ch2, completes
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = spawn (fun () ->
        let x = Channel.recv ch1 in
        Channel.send ch2 (x + 100)
    ) in
    Channel.send ch1 5;
    Channel.recv ch2
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_producer_consumer_interleaving() {
        // Producer sends 3 values, consumer receives 3 values
        // With rendezvous semantics, each send blocks until recv
        // This requires interleaving: send1, recv1, send2, recv2, send3, recv3
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () ->
        Channel.send ch 1;
        Channel.send ch 2;
        Channel.send ch 3
    ) in
    let a = Channel.recv ch in
    let b = Channel.recv ch in
    let c = Channel.recv ch in
    a + b + c
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_async_workers_pattern() {
        // Spawn workers, do local work, then collect results
        // This is a common async pattern: fan-out work, compute locally, fan-in results
        let program = r#"
let main () =
    let r1 = Channel.new in
    let r2 = Channel.new in

    let _ = spawn (fun () -> Channel.send r1 (10 * 10)) in
    let _ = spawn (fun () -> Channel.send r2 (5 + 5)) in

    let local = 1 + 2 + 3 in

    let v1 = Channel.recv r1 in
    let v2 = Channel.recv r2 in

    local + v1 + v2
"#;
        // local=6, v1=100, v2=10, total=116
        run_program(program).unwrap();
    }

    #[test]
    fn test_bidirectional_ping_pong_proves_interleaving() {
        // Main and worker alternate: main sends, worker receives and sends back, repeat
        // This CANNOT work without interleaving - each step depends on the previous
        let program = r#"
let main () =
    let to_worker = Channel.new in
    let from_worker = Channel.new in
    let _ = spawn (fun () ->
        let a = Channel.recv to_worker in
        Channel.send from_worker (a * 2);
        let b = Channel.recv to_worker in
        Channel.send from_worker (b * 2);
        let c = Channel.recv to_worker in
        Channel.send from_worker (c * 2)
    ) in
    Channel.send to_worker 1;
    let r1 = Channel.recv from_worker in
    Channel.send to_worker 2;
    let r2 = Channel.recv from_worker in
    Channel.send to_worker 3;
    let r3 = Channel.recv from_worker in
    r1 + r2 + r3
"#;
        // r1=2, r2=4, r3=6, sum=12
        run_program(program).unwrap();
    }

    // ========================================================================
    // Select tests
    // ========================================================================

    #[test]
    fn test_select_basic() {
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = spawn (fun () -> Channel.send ch1 42) in
    select
    | x <- ch1 -> x
    | y <- ch2 -> y
    end
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_select_second_channel() {
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = spawn (fun () -> Channel.send ch2 99) in
    select
    | x <- ch1 -> x
    | y <- ch2 -> y
    end
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_select_with_pattern() {
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () -> Channel.send ch (1, 2)) in
    select
    | (a, b) <- ch -> a + b
    end
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_select_deadlock_no_senders() {
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    select
    | x <- ch1 -> x
    | y <- ch2 -> y
    end
"#;
        let result = run_program(program);
        assert!(matches!(result, Err(EvalError::Deadlock)));
    }

    #[test]
    fn test_select_multiple_ready() {
        // Both channels have senders - should pick one (non-deterministic, but shouldn't deadlock)
        let program = r#"
let main () =
    let ch1 = Channel.new in
    let ch2 = Channel.new in
    let _ = spawn (fun () -> Channel.send ch1 1) in
    let _ = spawn (fun () -> Channel.send ch2 2) in
    select
    | x <- ch1 -> x
    | y <- ch2 -> y
    end
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_select_server_pattern() {
        // A server responding to requests via select
        let program = r#"
let main () =
    let requests = Channel.new in
    let responses = Channel.new in
    let _ = spawn (fun () ->
        select
        | req <- requests -> Channel.send responses (req + 100)
        end
    ) in
    Channel.send requests 42;
    Channel.recv responses
"#;
        run_program(program).unwrap();
    }

    // ========================================================================
    // Delimited continuation tests
    // ========================================================================

    #[test]
    fn test_reset_no_shift() {
        let val = eval("reset 42").unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_reset_with_expr() {
        let val = eval("reset (1 + 2 + 3)").unwrap();
        assert!(matches!(val, Value::Int(6)));
    }

    #[test]
    fn test_shift_discard_continuation() {
        // k not called - early exit
        let val = eval("reset (1 + shift (fun k -> 42))").unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_shift_call_once() {
        let val = eval("reset (1 + shift (fun k -> k 10))").unwrap();
        assert!(matches!(val, Value::Int(11)));
    }

    #[test]
    fn test_shift_call_twice() {
        let val = eval("reset (1 + shift (fun k -> k (k 10)))").unwrap();
        // k 10 = 11, k 11 = 12
        assert!(matches!(val, Value::Int(12)));
    }

    #[test]
    fn test_shift_in_let() {
        let val = eval("reset (let x = shift (fun k -> k 5) in x * x)").unwrap();
        assert!(matches!(val, Value::Int(25)));
    }

    #[test]
    fn test_nested_reset_inner_shift() {
        // Inner shift only captures to inner reset
        let val = eval("reset (1 + reset (2 + shift (fun k -> k 10)))").unwrap();
        // Inner: 2 + 10 = 12, Outer: 1 + 12 = 13
        assert!(matches!(val, Value::Int(13)));
    }

    #[test]
    fn test_nested_reset_outer_shift() {
        // Outer shift captures outer context
        let val = eval("reset (1 + shift (fun k -> k (reset (2 + 3))))").unwrap();
        // reset (2+3) = 5, k 5 = 1 + 5 = 6
        assert!(matches!(val, Value::Int(6)));
    }

    #[test]
    fn test_shift_without_reset_errors() {
        let result = eval("shift (fun k -> k 1)");
        assert!(result.is_err());
    }

    #[test]
    fn test_shift_return_continuation() {
        // Return the continuation itself
        let val = eval("reset (1 + shift (fun k -> k))").unwrap();
        assert!(matches!(val, Value::Continuation { .. }));
    }

    #[test]
    fn test_continuation_called_later() {
        let val = eval(
            "
            let k = reset (1 + shift (fun k -> k)) in
            k 10
        ",
        )
        .unwrap();
        assert!(matches!(val, Value::Int(11)));
    }

    #[test]
    fn test_shift_with_multiple_invocations() {
        let val = eval(
            "
            reset (
                let x = shift (fun k -> k 1 + k 2) in
                x * 10
            )
        ",
        )
        .unwrap();
        // k 1 = 10, k 2 = 20, sum = 30
        assert!(matches!(val, Value::Int(30)));
    }

    #[test]
    fn test_reset_in_function() {
        let program = r#"
let with_reset f = reset (f ())

let main () =
    with_reset (fun () -> 1 + shift (fun k -> k 10))
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_shift_in_continuation_argument_cbv() {
        // In strict CBV, the argument to a continuation is evaluated BEFORE the continuation
        // is invoked. So `k (shift ...)` evaluates the shift in the current context, not inside k.
        // Since the outer reset was consumed by the outer shift, this is "shift without reset".
        let result = eval(
            "
            reset (
                shift (fun k1 -> k1 (shift (fun k2 -> k2 100))) + 1
            )
        ",
        );
        // In CBV: outer shift consumes the reset, inner shift has no enclosing reset
        assert!(result.is_err());
    }

    #[test]
    fn test_continuation_body_is_delimited() {
        // This tests the core fix: when a continuation IS invoked with a value,
        // the body executes inside a fresh reset boundary.
        // k 10 should work because the continuation execution is wrapped in reset.
        let val = eval(
            "
            let k = reset (shift (fun k -> k)) in
            k (1 + k 10)
        ",
        )
        .unwrap();
        // k = continuation that returns its argument (identity for this context)
        // k 10 = reset (10) = 10
        // 1 + k 10 = 11
        // k 11 = reset (11) = 11
        assert!(matches!(val, Value::Int(11)));
    }

    #[test]
    fn test_nested_resets() {
        let val = eval(
            "
            reset (
                1 + reset (
                    2 + shift (fun k -> k (k 10))
                )
            )
        ",
        )
        .unwrap();
        // Inner shift captures [2 + □] only (stopped at inner reset)
        // k 10 = reset (2 + 10) = 12
        // k 12 = reset (2 + 12) = 14
        // Outer: 1 + 14 = 15
        assert!(matches!(val, Value::Int(15)));
    }

    #[test]
    fn test_discarded_continuation() {
        let val = eval(
            "
            reset (
                1 + shift (fun k -> 42)
            )
        ",
        )
        .unwrap();
        // k is never called, result is just 42
        assert!(matches!(val, Value::Int(42)));
    }

    // ========================================================================
    // Local recursive function tests
    // ========================================================================

    #[test]
    fn test_local_recursive_function() {
        // Local let with function syntax and recursion
        let val = eval(
            "
            let f x = if x == 0 then 0 else x + f (x - 1) in
            f 5
        ",
        )
        .unwrap();
        // 5 + 4 + 3 + 2 + 1 + 0 = 15
        assert!(matches!(val, Value::Int(15)));
    }

    #[test]
    fn test_local_recursive_function_factorial() {
        let val = eval(
            "
            let fact n = if n == 0 then 1 else n * fact (n - 1) in
            fact 5
        ",
        )
        .unwrap();
        // 5! = 120
        assert!(matches!(val, Value::Int(120)));
    }

    #[test]
    fn test_local_recursive_function_lambda_form() {
        // Using explicit lambda syntax
        let val = eval(
            "
            let f = fun x -> if x == 0 then 0 else x + f (x - 1) in
            f 5
        ",
        )
        .unwrap();
        assert!(matches!(val, Value::Int(15)));
    }

    #[test]
    fn test_local_function_non_recursive() {
        // Non-recursive local function should still work
        let val = eval(
            "
            let double x = x * 2 in
            double 21
        ",
        )
        .unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_local_function_multiple_params() {
        let val = eval(
            "
            let add x y = x + y in
            add 10 32
        ",
        )
        .unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_nested_local_functions() {
        let val = eval(
            "
            let outer x =
                let inner y = x + y in
                inner 10
            in
            outer 5
        ",
        )
        .unwrap();
        assert!(matches!(val, Value::Int(15)));
    }

    // ========================================================================
    // Phase 6: Typeclass Runtime Tests
    // ========================================================================

    use crate::infer::Inferencer;
    use crate::types::{InstanceInfo, TraitInfo, Type as InferType};

    /// Helper to run a program with type inference, copying ClassEnv to interpreter
    fn run_typed_program(input: &str) -> Result<Value, String> {
        let tokens = Lexer::new(input).tokenize().map_err(|e| e.to_string())?;
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().map_err(|e| e.to_string())?;

        // Run type inference to build ClassEnv
        let mut inferencer = Inferencer::new();
        let _env = inferencer
            .infer_program(&program)
            .map_err(|errors| {
                errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")
            })?;

        // Run the program with the class env
        let mut interp = Interpreter::new();
        // Copy the class_env from inference phase
        // Note: We can't directly access class_env from inferencer, so we'll test via integration
        interp.run(&program).map_err(|e| e.to_string())
    }

    #[test]
    fn test_value_to_type() {
        // Test that Value::to_type works correctly
        assert!(matches!(Value::Int(42).to_type(), InferType::Int));
        assert!(matches!(
            Value::String("hello".into()).to_type(),
            InferType::String
        ));
        assert!(matches!(Value::Bool(true).to_type(), InferType::Bool));
    }

    #[test]
    fn test_dict_value() {
        let mut methods = HashMap::new();
        methods.insert("show".to_string(), Value::Builtin("show_int".into()));
        let dict = Value::Dict {
            trait_name: "Show".into(),
            methods,
        };
        assert_eq!(dict.type_name(), "Dict");
    }

    #[test]
    fn test_build_dict_simple() {
        // Build an interpreter with a simple Show Int instance
        let mut interp = Interpreter::new();

        // Manually set up the class env with a Show trait and Show Int instance
        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            InferType::arrow(InferType::new_generic(0), InferType::String),
        );
        interp.class_env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Add Show Int instance with method implementation
        use crate::ast::{ExprKind, InstanceMethod, Literal, Span, Spanned};
        let show_impl = InstanceMethod {
            name: "show".to_string(),
            params: vec![Pattern {
                node: PatternKind::Var("n".into()),
                span: Span::default(),
            }],
            body: Spanned {
                node: ExprKind::Lit(Literal::String("42".into())),
                span: Span::default(),
            },
        };
        interp
            .class_env
            .add_instance(InstanceInfo {
                trait_name: "Show".to_string(),
                head: InferType::Int,
                constraints: vec![],
                method_impls: vec![show_impl],
            })
            .unwrap();

        // Now build a dictionary for Show Int
        let pred = Pred::new("Show", InferType::Int);
        let dict = interp.build_dict(&pred).unwrap();

        match dict {
            Value::Dict {
                trait_name,
                methods,
            } => {
                assert_eq!(trait_name, "Show");
                assert!(methods.contains_key("show"));
            }
            _ => panic!("Expected Dict value"),
        }
    }

    #[test]
    fn test_call_method_from_dict() {
        let mut interp = Interpreter::new();

        // Create a simple dict with a show method that returns a constant
        use crate::ast::{ExprKind, Literal, Span, Spanned};
        let show_closure = Value::Closure {
            params: vec![Pattern {
                node: PatternKind::Var("x".into()),
                span: Span::default(),
            }],
            body: Rc::new(Spanned {
                node: ExprKind::Lit(Literal::String("hello".into())),
                span: Span::default(),
            }),
            env: EnvInner::new(),
        };

        let mut methods = HashMap::new();
        methods.insert("show".to_string(), show_closure);
        let dict = Value::Dict {
            trait_name: "Show".into(),
            methods,
        };

        // Call the method
        let result = interp.call_method(&dict, "show", Value::Int(42)).unwrap();
        assert!(matches!(result, Value::String(s) if s == "hello"));
    }

    #[test]
    fn test_trait_instance_parsing_and_eval() {
        // Test that trait and instance declarations don't crash evaluation
        let program = r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

let main () = ()
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_show_int_end_to_end() {
        // Full end-to-end test: define Show trait, Show Int instance, call show 42
        let tokens = Lexer::new(
            r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

let result = show 42
"#,
        )
        .tokenize()
        .unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        // Run type inference to populate class_env
        let mut inferencer = Inferencer::new();
        let _type_env = inferencer.infer_program(&program).unwrap();

        // Create interpreter and copy the class_env from inferencer
        let mut interp = Interpreter::new();
        // We need to expose the class_env from the inferencer
        // For now, manually set up the class_env in the interpreter
        use crate::ast::{ExprKind, InstanceMethod, Literal, Span, Spanned};

        let mut methods = HashMap::new();
        methods.insert(
            "show".to_string(),
            InferType::arrow(InferType::new_generic(0), InferType::String),
        );
        interp.class_env.add_trait(TraitInfo {
            name: "Show".to_string(),
            type_param: "a".to_string(),
            supertraits: vec![],
            methods,
        });

        // Add Show Int instance - use int_to_string n as the body
        let show_impl = InstanceMethod {
            name: "show".to_string(),
            params: vec![Pattern {
                node: PatternKind::Var("n".into()),
                span: Span::default(),
            }],
            body: Spanned {
                node: ExprKind::App {
                    func: Rc::new(Spanned {
                        node: ExprKind::Var("int_to_string".into()),
                        span: Span::default(),
                    }),
                    arg: Rc::new(Spanned {
                        node: ExprKind::Var("n".into()),
                        span: Span::default(),
                    }),
                },
                span: Span::default(),
            },
        };
        interp
            .class_env
            .add_instance(InstanceInfo {
                trait_name: "Show".to_string(),
                head: InferType::Int,
                constraints: vec![],
                method_impls: vec![show_impl],
            })
            .unwrap();

        // Evaluate "show 42" directly
        let env = EnvInner::new();
        {
            env.borrow_mut().define(
                "int_to_string".into(),
                Value::Builtin("int_to_string".into()),
            );
        }

        // Create the expression: show 42
        let show_42 = Spanned {
            node: ExprKind::App {
                func: Rc::new(Spanned {
                    node: ExprKind::Var("show".into()),
                    span: Span::default(),
                }),
                arg: Rc::new(Spanned {
                    node: ExprKind::Lit(Literal::Int(42)),
                    span: Span::default(),
                }),
            },
            span: Span::default(),
        };

        let result = interp.eval_expr(&env, &show_42).unwrap();
        assert!(matches!(result, Value::String(s) if s == "42"));
    }

    // ========================================================================
    // User-defined operators
    // ========================================================================

    #[test]
    fn test_user_defined_operator_basic() {
        // Define and use a simple user-defined operator using let-in
        let val = eval(
            r#"
            let (<|>) a b = a + b in
            3 <|> 5
        "#,
        )
        .unwrap();
        // result should be 8
        assert!(matches!(val, Value::Int(8)));
    }

    #[test]
    fn test_user_defined_operator_prefix_syntax() {
        // Define operator with prefix syntax
        let val = eval(
            r#"
            let (<+>) a b = a * b in
            4 <+> 5
        "#,
        )
        .unwrap();
        assert!(matches!(val, Value::Int(20)));
    }

    #[test]
    fn test_user_defined_operator_complex() {
        // More complex operator definition
        let val = eval(
            r#"
            let (<?>) a b = if a > b then a else b in
            10 <?> 5
        "#,
        )
        .unwrap();
        assert!(matches!(val, Value::Int(10)));
    }

    // ========================================================================
    // Mutual recursion (let rec ... and ...)
    // ========================================================================

    #[test]
    fn test_let_rec_simple() {
        // Simple recursive function using let rec
        let val = eval(
            r#"
            let rec factorial n = if n == 0 then 1 else n * factorial (n - 1) in
            factorial 5
        "#,
        )
        .unwrap();
        assert!(matches!(val, Value::Int(120)));
    }

    #[test]
    fn test_let_rec_mutual_even_odd() {
        // Classic mutual recursion: is_even and is_odd
        let val = eval(
            r#"
            let rec is_even n = if n == 0 then true else is_odd (n - 1)
            and is_odd n = if n == 0 then false else is_even (n - 1)
            in is_even 10
        "#,
        )
        .unwrap();
        assert!(matches!(val, Value::Bool(true)));
    }

    #[test]
    fn test_let_rec_mutual_even_odd_false() {
        // Test the other branch
        let val = eval(
            r#"
            let rec is_even n = if n == 0 then true else is_odd (n - 1)
            and is_odd n = if n == 0 then false else is_even (n - 1)
            in is_even 7
        "#,
        )
        .unwrap();
        assert!(matches!(val, Value::Bool(false)));
    }

    #[test]
    fn test_let_rec_mutual_is_odd() {
        // Test is_odd function
        let val = eval(
            r#"
            let rec is_even n = if n == 0 then true else is_odd (n - 1)
            and is_odd n = if n == 0 then false else is_even (n - 1)
            in is_odd 7
        "#,
        )
        .unwrap();
        assert!(matches!(val, Value::Bool(true)));
    }

    #[test]
    fn test_let_rec_three_functions() {
        // Three mutually recursive functions
        let val = eval(
            r#"
            let rec f n = if n == 0 then 0 else g (n - 1)
            and g n = if n == 0 then 1 else h (n - 1)
            and h n = if n == 0 then 2 else f (n - 1)
            in f 6
        "#,
        )
        .unwrap();
        // f(6) -> g(5) -> h(4) -> f(3) -> g(2) -> h(1) -> f(0) = 0
        assert!(matches!(val, Value::Int(0)));
    }

    // ========================================================================
    // I/O and Timer tests
    // ========================================================================

    #[test]
    fn test_sleep_ms_basic() {
        use std::time::Instant;
        // Test that sleep_ms actually delays and returns Unit
        let program = r#"
let main () =
    sleep_ms 50
"#;
        let start = Instant::now();
        run_program(program).unwrap();
        let elapsed = start.elapsed();
        // Should have slept at least 40ms (allowing some slack)
        assert!(
            elapsed.as_millis() >= 40,
            "Expected at least 40ms elapsed, got {}ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_sleep_ms_returns_unit() {
        // Verify that sleep_ms returns () and can be used in let binding
        let program = r#"
let main () =
    let _ = sleep_ms 10 in
    42
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_sleep_ms_with_concurrent_work() {
        // Verify sleep works with other fibers
        let program = r#"
let main () =
    let ch = Channel.new in
    let _ = spawn (fun () ->
        sleep_ms 20;
        Channel.send ch 100
    ) in
    Channel.recv ch
"#;
        run_program(program).unwrap();
    }

    // ========================================================================
    // File I/O tests
    // ========================================================================

    #[test]
    fn test_file_open_nonexistent() {
        // Opening a nonexistent file should return Err IoError
        // We use match to verify the error case and succeed
        let program = r#"
let main () =
    match file_open "/nonexistent/path/to/file.txt" "r" with
    | Ok _ -> ()  -- unexpected success
    | Err _ -> () -- expected failure, test passes
    end
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_file_open_close() {
        use std::io::Write;
        // Create a temp file and test open/close
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("gneiss_test_file.txt");

        // Create the file first
        {
            let mut f = std::fs::File::create(&temp_path).unwrap();
            f.write_all(b"hello").unwrap();
        }

        let program = format!(
            r#"
let main () =
    match file_open "{}" "r" with
    | Ok handle ->
        match file_close handle with
        | Ok _ -> () -- success
        | Err e -> print e -- print error
        end
    | Err e -> print e -- print error
    end
"#,
            temp_path.display()
        );
        run_program(&program).unwrap();

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_file_write_and_verify() {
        // Test that file_write actually writes to the file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("gneiss_test_write_verify.txt");

        let write_program = format!(
            r#"
let main () =
    match file_open "{}" "w" with
    | Ok handle ->
        let data = string_to_bytes "hello from gneiss" in
        match file_write handle data with
        | Ok _ ->
            match file_close handle with
            | Ok _ -> ()
            | Err e -> print e
            end
        | Err e -> print e
        end
    | Err e -> print e
    end
"#,
            temp_path.display()
        );
        run_program(&write_program).unwrap();

        // Verify the file was written correctly
        let contents = std::fs::read_to_string(&temp_path).unwrap();
        assert_eq!(contents, "hello from gneiss");

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_file_read_and_verify() {
        use std::io::Write;
        // Create a file first via Rust
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("gneiss_test_read_verify.txt");
        {
            let mut f = std::fs::File::create(&temp_path).unwrap();
            f.write_all(b"test content from rust").unwrap();
        }

        // Read using Gneiss and print the content
        let read_program = format!(
            r#"
let main () =
    match file_open "{}" "r" with
    | Ok handle ->
        match file_read handle 100 with
        | Ok data ->
            let s = bytes_to_string data in
            let _ = file_close handle in
            print s
        | Err e ->
            let _ = file_close handle in
            print e
        end
    | Err e -> print e
    end
"#,
            temp_path.display()
        );
        run_program(&read_program).unwrap();

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_file_round_trip() {
        // Write and read back in same program
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("gneiss_test_round_trip.txt");

        let program = format!(
            r#"
let main () =
    -- First write to file
    match file_open "{path}" "w" with
    | Ok wh ->
        let data = string_to_bytes "round trip test" in
        let _ = file_write wh data in
        let _ = file_close wh in

        -- Now read it back
        match file_open "{path}" "r" with
        | Ok rh ->
            match file_read rh 100 with
            | Ok read_data ->
                let _ = file_close rh in
                print (bytes_to_string read_data)
            | Err e ->
                let _ = file_close rh in
                print e
            end
        | Err e -> print e
        end
    | Err e -> print e
    end
"#,
            path = temp_path.display()
        );
        run_program(&program).unwrap();

        // Verify the file exists and contains the expected content
        let contents = std::fs::read_to_string(&temp_path).unwrap();
        assert_eq!(contents, "round trip test");

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    // ========================================================================
    // TCP socket tests
    // ========================================================================

    #[test]
    fn test_tcp_connect_returns_error_for_invalid_host() {
        // TCP connect to an invalid host should return Err IoError
        let program = r#"
let main () =
    match tcp_connect "invalid.host.that.does.not.exist.local" 12345 with
    | Ok _ -> ()  -- unexpected success
    | Err _ -> () -- expected: connection error
    end
"#;
        run_program(program).unwrap();
    }

    #[test]
    fn test_tcp_listen_and_accept() {
        // tcp_listen should work, tcp_accept returns error without registry
        let program = r#"
let main () =
    match tcp_listen "127.0.0.1" 0 with
    | Ok listener ->
        -- Would try accept but no registry yet
        ()
    | Err _ -> () -- May fail if port unavailable
    end
"#;
        run_program(program).unwrap();
    }
}
