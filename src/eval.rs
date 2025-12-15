//! Defunctionalized CPS interpreter for Gneiss
//!
//! This interpreter uses an explicit continuation stack (defunctionalized CPS)
//! which allows processes to be suspended and resumed for concurrency.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
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
    /// After evaluating channel expr in send, evaluate the value
    SendChan { value_expr: Rc<Expr>, env: Env },
    /// After evaluating value in send, perform the send
    SendVal { channel: ChannelId },
    /// After evaluating channel expr in recv, perform the recv
    Recv,

    // === Delimited continuation frames ===
    /// Delimiter marker for reset
    Prompt,

    // === Select frames ===
    /// Evaluating channel expressions for select, collecting them
    SelectChans {
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

    /// All channels evaluated, now perform the select
    SelectReady {
        /// Channel IDs to select from
        channels: Vec<ChannelId>,
        /// Corresponding patterns
        patterns: Vec<Pattern>,
        /// Corresponding bodies
        bodies: Vec<Rc<Expr>>,
        env: Env,
    },
}

/// What kind of collection we're building
#[derive(Debug, Clone)]
pub enum CollectKind {
    List,
    Tuple,
    Constructor { name: String },
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

    pub fn push(&mut self, frame: Frame) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<Frame> {
        self.frames.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
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
    /// Process needs to block (for concurrency)
    Blocked { reason: BlockReason, state: State },
    /// An error occurred
    Error(EvalError),
}

/// Why a process is blocked (for concurrency)
#[derive(Debug, Clone)]
pub enum BlockReason {
    Send { channel: ChannelId, value: Value },
    Recv { channel: ChannelId },
    Select { channels: Vec<ChannelId> },
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

    /// A list
    List(Vec<Value>),

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
            Value::List(_) => "List",
            Value::Tuple(_) => "Tuple",
            Value::Closure { .. } => "Function",
            Value::Constructor { .. } => "Constructor",
            Value::Pid(_) => "Pid",
            Value::Channel(_) => "Channel",
            Value::Builtin(_) => "Builtin",
            Value::Continuation { .. } => "Continuation",
            Value::Fiber(_) => "Fiber",
            Value::FiberEffect(_) => "FiberEffect",
            Value::Dict { .. } => "Dict",
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
            Value::List(items) => {
                // Infer element type from first item, or use a fresh var
                let elem_ty = items.first().map(|v| v.to_type()).unwrap_or(Type::Unit);
                Type::list(elem_ty)
            }
            Value::Tuple(items) => Type::Tuple(items.iter().map(|v| v.to_type()).collect()),
            Value::Constructor { name, fields } => Type::Constructor {
                name: name.clone(),
                args: fields.iter().map(|v| v.to_type()).collect(),
            },
            Value::Pid(_) => Type::Pid,
            Value::Channel(_) => Type::Channel(Rc::new(Type::Unit)), // Simplified
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
            Value::List(items) => {
                let elem_ty = items
                    .first()
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
            _ => Type::Unit,
        }
    }
}

/// Environment mapping names to values
pub type Env = Rc<RefCell<EnvInner>>;

#[derive(Debug, Clone)]
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

impl Default for EnvInner {
    fn default() -> Self {
        Self {
            bindings: HashMap::new(),
            parent: None,
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
        }

        Self {
            global_env,
            runtime: Runtime::new(),
            class_env: ClassEnv::new(),
            type_ctx: TypeContext::new(),
        }
    }

    /// Set the class environment (called after type inference)
    pub fn set_class_env(&mut self, class_env: ClassEnv) {
        self.class_env = class_env;
    }

    /// Set the type context (called after type inference)
    pub fn set_type_ctx(&mut self, type_ctx: TypeContext) {
        self.type_ctx = type_ctx;
    }

    /// Run a program
    pub fn run(&mut self, program: &Program) -> Result<Value, EvalError> {
        let mut last_expr_value = Value::Unit;

        // Process all items in order: declarations bind values, expressions execute
        for item in &program.items {
            match item {
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
            Decl::OperatorDef { op, params, body } => {
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
            Decl::LetRec { bindings } => {
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

    /// Run the machine until completion, error, or block
    fn run_to_completion(&mut self, mut state: State) -> Result<Value, EvalError> {
        loop {
            match self.step(state) {
                StepResult::Continue(next) => state = next,
                StepResult::Done(value) => return Ok(value),
                StepResult::Error(e) => return Err(e),
                StepResult::Blocked { .. } => {
                    return Err(EvalError::RuntimeError("main thread cannot block".into()));
                }
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
                match env.borrow().get(name) {
                    Some(value) => StepResult::Continue(State::Apply { value, cont }),
                    None => {
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
                cont.push(Frame::SendChan {
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
                cont.push(Frame::Recv);
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
                    // Empty select blocks forever (or error)
                    return StepResult::Error(EvalError::RuntimeError("empty select".into()));
                }

                // Extract components from arms
                let patterns: Vec<Pattern> = arms.iter().map(|a| a.pattern.clone()).collect();
                let bodies: Vec<Rc<Expr>> = arms.iter().map(|a| Rc::new(a.body.clone())).collect();
                let chan_exprs: Vec<Expr> = arms.iter().map(|a| a.channel.clone()).collect();

                // Start evaluating the first channel expression
                let mut remaining = chan_exprs;
                let first = remaining.remove(0);

                cont.push(Frame::SelectChans {
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
                CollectKind::List => Value::List(vec![]),
                CollectKind::Tuple => Value::Tuple(vec![]),
                CollectKind::Constructor { name } => Value::Constructor {
                    name,
                    fields: vec![],
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
                        CollectKind::List => Value::List(acc),
                        CollectKind::Tuple => Value::Tuple(acc),
                        CollectKind::Constructor { name } => {
                            Value::Constructor { name, fields: acc }
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

            Some(Frame::SendChan { value_expr, env }) => {
                let channel = match value {
                    Value::Channel(id) => id,
                    _ => return StepResult::Error(EvalError::TypeError("expected channel".into())),
                };
                cont.push(Frame::SendVal { channel });
                StepResult::Continue(State::Eval {
                    expr: value_expr,
                    env,
                    cont,
                })
            }

            Some(Frame::SendVal { channel }) => {
                // We have the channel and the value to send
                // Check we're in a process context (requires main function)
                if self.runtime.current_pid().is_none() {
                    return StepResult::Error(EvalError::RuntimeError(
                        "Channel.send requires a process context (define a main function)".into(),
                    ));
                }
                // Check if there's a waiting receiver (rendezvous)
                if self.runtime.send(channel, value) {
                    // Immediate rendezvous - sender continues with Unit
                    StepResult::Continue(State::Apply {
                        value: Value::Unit,
                        cont,
                    })
                } else {
                    // No receiver ready - block this process
                    // The continuation expects Unit when we resume
                    StepResult::Blocked {
                        reason: BlockReason::Send {
                            channel,
                            value: Value::Unit,
                        },
                        state: State::Apply {
                            value: Value::Unit,
                            cont,
                        },
                    }
                }
            }

            Some(Frame::Recv) => {
                let channel = match value {
                    Value::Channel(id) => id,
                    _ => return StepResult::Error(EvalError::TypeError("expected channel".into())),
                };
                // Check we're in a process context (requires main function)
                if self.runtime.current_pid().is_none() {
                    return StepResult::Error(EvalError::RuntimeError(
                        "Channel.recv requires a process context (define a main function)".into(),
                    ));
                }
                // Check if there's a waiting sender (rendezvous)
                if let Some(received) = self.runtime.recv(channel) {
                    // Immediate rendezvous - got value from sender
                    StepResult::Continue(State::Apply {
                        value: received,
                        cont,
                    })
                } else {
                    // No sender ready - block this process
                    // When we resume, we'll have a received value
                    StepResult::Blocked {
                        reason: BlockReason::Recv { channel },
                        state: State::Apply {
                            value: Value::Unit,
                            cont,
                        }, // placeholder, will be replaced
                    }
                }
            }

            // === Select frames ===
            Some(Frame::SelectChans {
                patterns,
                bodies,
                mut remaining_chans,
                mut collected_chans,
                env,
            }) => {
                // Value should be a channel
                let channel_id = match value {
                    Value::Channel(id) => id,
                    _ => {
                        return StepResult::Error(EvalError::TypeError(
                            "select arm must be a channel".into(),
                        ))
                    }
                };

                collected_chans.push(channel_id);

                if remaining_chans.is_empty() {
                    // All channels evaluated, now do the select
                    cont.push(Frame::SelectReady {
                        channels: collected_chans,
                        patterns,
                        bodies,
                        env: env.clone(),
                    });

                    // Return unit to trigger the SelectReady frame
                    StepResult::Continue(State::Apply {
                        value: Value::Unit,
                        cont,
                    })
                } else {
                    // More channels to evaluate
                    let next = remaining_chans.remove(0);

                    cont.push(Frame::SelectChans {
                        patterns,
                        bodies,
                        remaining_chans,
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

            Some(Frame::SelectReady {
                channels,
                patterns,
                bodies,
                env,
            }) => {
                // Check if we're resuming after being woken (a channel fired)
                if let Some(pid) = self.runtime.current_pid() {
                    if let Some(fired_channel) = self.runtime.take_select_fired_channel(pid) {
                        // Find which arm this channel corresponds to
                        if let Some(i) = channels.iter().position(|&ch| ch == fired_channel) {
                            // Get the received value
                            let recv_value =
                                self.runtime.take_received_value(pid).unwrap_or(Value::Unit);

                            // Bind pattern and evaluate body
                            let new_env = EnvInner::with_parent(&env);
                            if self.try_bind_pattern(&new_env, &patterns[i], &recv_value) {
                                return StepResult::Continue(State::Eval {
                                    expr: bodies[i].clone(),
                                    env: new_env,
                                    cont,
                                });
                            } else {
                                return StepResult::Error(EvalError::MatchFailed);
                            }
                        }
                    }
                }

                // Not resuming from a wakeup - check if any channel has a waiting sender
                for (i, &channel_id) in channels.iter().enumerate() {
                    if let Some(recv_value) = self.runtime.try_recv(channel_id) {
                        // Got a value! Bind pattern and evaluate body
                        let new_env = EnvInner::with_parent(&env);
                        if self.try_bind_pattern(&new_env, &patterns[i], &recv_value) {
                            return StepResult::Continue(State::Eval {
                                expr: bodies[i].clone(),
                                env: new_env,
                                cont,
                            });
                        } else {
                            return StepResult::Error(EvalError::MatchFailed);
                        }
                    }
                }

                // No channel ready - block on all of them
                // Check we're in a process context (requires main function)
                if self.runtime.current_pid().is_none() {
                    return StepResult::Error(EvalError::RuntimeError(
                        "select requires a process context (define a main function)".into(),
                    ));
                }
                self.runtime.block_on_select(&channels);

                StepResult::Blocked {
                    reason: BlockReason::Select {
                        channels: channels.clone(),
                    },
                    state: State::Apply {
                        value: Value::Unit,
                        cont: {
                            let mut new_cont = cont;
                            new_cont.push(Frame::SelectReady {
                                channels,
                                patterns,
                                bodies,
                                env,
                            });
                            new_cont
                        },
                    },
                }
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
                if name.starts_with("__method__") {
                    // Parse "__method__TraitName_methodName"
                    let rest = &name[10..]; // Skip "__method__"
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
            // Arithmetic
            (BinOp::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (BinOp::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (BinOp::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (BinOp::Div, Value::Int(_), Value::Int(0)) => Err(EvalError::DivisionByZero),
            (BinOp::Div, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
            (BinOp::Mod, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),

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

            (BinOp::Lt, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a < b)),
            (BinOp::Gt, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a > b)),
            (BinOp::Lte, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a <= b)),
            (BinOp::Gte, Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a >= b)),

            // Boolean
            (BinOp::And, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
            (BinOp::Or, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),

            // List operations
            (BinOp::Cons, val, Value::List(list)) => {
                let mut new_list = vec![val.clone()];
                new_list.extend(list.clone());
                Ok(Value::List(new_list))
            }
            (BinOp::Concat, Value::List(a), Value::List(b)) => {
                let mut new_list = a.clone();
                new_list.extend(b.clone());
                Ok(Value::List(new_list))
            }
            (BinOp::Concat, Value::String(a), Value::String(b)) => {
                Ok(Value::String(format!("{}{}", a, b)))
            }

            // Compose - create a new closure
            (BinOp::Compose, _, _) | (BinOp::ComposeBack, _, _) => Err(EvalError::RuntimeError(
                "function composition not yet implemented".into(),
            )),

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
            (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
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
                    let chars: Vec<Value> = s.chars().map(Value::Char).collect();
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
            _ => Err(EvalError::RuntimeError(format!(
                "unknown builtin: {}",
                name
            ))),
        }
    }

    fn print_value(&self, val: &Value) {
        match val {
            Value::Int(n) => print!("{}", n),
            Value::Float(f) => print!("{}", f),
            Value::Bool(b) => print!("{}", b),
            Value::String(s) => print!("{}", s),
            Value::Char(c) => print!("{}", c),
            Value::Unit => print!("()"),
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
            Value::Continuation { .. } => print!("<continuation>"),
            Value::Dict { trait_name, .. } => print!("<dict:{}>", trait_name),
            Value::Fiber(id) => print!("<fiber:{}>", id),
            Value::FiberEffect(effect) => print!("<fiber-effect:{:?}>", effect),
        }
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
                self.try_bind_pattern(env, head, &vals[0])
                    && self.try_bind_pattern(env, tail, &Value::List(vals[1..].to_vec()))
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

            _ => false,
        }
    }

    /// Run the scheduler until all processes complete or deadlock
    fn run_scheduler(&mut self) -> Result<(), EvalError> {
        // Simple round-robin scheduler
        while let Some(pid) = self.runtime.next_ready() {
            self.runtime.set_current(Some(pid));

            if let Some(pcont) = self.runtime.take_continuation(pid) {
                match pcont {
                    ProcessContinuation::Start(thunk) => {
                        // Start a new process: apply thunk to unit
                        self.run_process(pid, thunk)?;
                    }
                    ProcessContinuation::ResumeAfterRecv => {
                        // Process was waiting to receive - resume with received value
                        if let Some(saved_cont) = self.runtime.take_saved_cont(pid) {
                            let value =
                                self.runtime.take_received_value(pid).unwrap_or(Value::Unit);
                            // Resume by applying the received value to the saved continuation
                            let state = State::Apply {
                                value,
                                cont: saved_cont,
                            };
                            self.run_state(pid, state)?;
                        } else {
                            // No saved continuation - shouldn't happen
                            self.runtime.mark_done(pid);
                        }
                    }
                    ProcessContinuation::ResumeAfterSelect => {
                        // Process was blocked on select - SelectReady frame handles the value
                        if let Some(saved_cont) = self.runtime.take_saved_cont(pid) {
                            // Don't take the received value here - SelectReady frame will get it
                            let state = State::Apply {
                                value: Value::Unit,
                                cont: saved_cont,
                            };
                            self.run_state(pid, state)?;
                        } else {
                            self.runtime.mark_done(pid);
                        }
                    }
                    ProcessContinuation::ResumeAfterSend => {
                        // Process was waiting to send - resume with Unit
                        if let Some(saved_cont) = self.runtime.take_saved_cont(pid) {
                            let state = State::Apply {
                                value: Value::Unit,
                                cont: saved_cont,
                            };
                            self.run_state(pid, state)?;
                        } else {
                            self.runtime.mark_done(pid);
                        }
                    }
                }
            }

            self.runtime.set_current(None);
        }

        if self.runtime.is_deadlocked() {
            Err(EvalError::Deadlock)
        } else {
            Ok(())
        }
    }

    /// Run a process starting from a thunk
    fn run_process(&mut self, pid: Pid, thunk: Value) -> Result<(), EvalError> {
        // Build initial state: apply thunk to Unit
        let mut cont = Cont::new();
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
                StepResult::Done(_value) => {
                    self.runtime.mark_done(pid);
                    return Ok(());
                }
                StepResult::Blocked {
                    state: blocked_state,
                    ..
                } => {
                    // Save the continuation for later resumption
                    // The blocked_state contains the continuation to resume with
                    if let State::Apply { cont, .. } = blocked_state {
                        self.runtime.save_cont(pid, cont);
                    }
                    // Process is now blocked - scheduler will pick up another process
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
        let val = eval("match 1 with | 1 -> 42 | _ -> 0").unwrap();
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
        let val = eval("spawn (fun () -> 42)").unwrap();
        assert!(matches!(val, Value::Pid(_)));
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
            .map_err(|e| e.to_string())?;

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
}
