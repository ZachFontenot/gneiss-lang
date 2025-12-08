//! Tree-walking interpreter for Gneiss

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
use crate::runtime::{ChannelId, Pid, ProcessContinuation, Runtime};
use thiserror::Error;

#[derive(Error, Debug)]
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
    Constructor { name: String, fields: Vec<Value> },

    /// A process ID
    Pid(Pid),

    /// A channel
    Channel(ChannelId),

    /// Built-in function
    Builtin(String),
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
}

impl Interpreter {
    pub fn new() -> Self {
        let global_env = EnvInner::new();

        // Add some built-in functions
        {
            let mut env = global_env.borrow_mut();
            env.define("print".into(), Value::Builtin("print".into()));
            env.define("int_to_string".into(), Value::Builtin("int_to_string".into()));
            env.define("string_length".into(), Value::Builtin("string_length".into()));
        }

        Self {
            global_env,
            runtime: Runtime::new(),
        }
    }

    /// Run a program
    pub fn run(&mut self, program: &Program) -> Result<Value, EvalError> {
        // First, evaluate all declarations in the global environment
        for decl in &program.declarations {
            self.eval_decl(decl)?;
        }

        // Look for a main function and run it
        let main_fn = self.global_env.borrow().get("main");
        if let Some(main) = main_fn {
            let result = self.apply(main, Value::Unit)?;

            // Run the scheduler if there are spawned processes
            self.run_scheduler()?;

            Ok(result)
        } else {
            Ok(Value::Unit)
        }
    }

    /// Evaluate a declaration
    fn eval_decl(&mut self, decl: &Decl) -> Result<(), EvalError> {
        match decl {
            Decl::Let {
                name,
                params,
                body,
                ..
            } => {
                if params.is_empty() {
                    let value = self.eval(&self.global_env.clone(), body)?;
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
        }
    }

    /// Evaluate an expression
    pub fn eval(&mut self, env: &Env, expr: &Expr) -> Result<Value, EvalError> {
        match &expr.node {
            ExprKind::Lit(lit) => Ok(self.eval_literal(lit)),

            ExprKind::Var(name) => env
                .borrow()
                .get(name)
                .ok_or_else(|| EvalError::UnboundVariable(name.clone())),

            ExprKind::Lambda { params, body } => Ok(Value::Closure {
                params: params.clone(),
                body: body.clone(),
                env: env.clone(),
            }),

            ExprKind::App { func, arg } => {
                let func_val = self.eval(env, func)?;
                let arg_val = self.eval(env, arg)?;
                self.apply(func_val, arg_val)
            }

            ExprKind::Let {
                pattern,
                value,
                body,
            } => {
                let val = self.eval(env, value)?;
                let new_env = EnvInner::with_parent(env);
                self.bind_pattern(&new_env, pattern, &val)?;

                if let Some(body) = body {
                    self.eval(&new_env, body)
                } else {
                    Ok(Value::Unit)
                }
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_val = self.eval(env, cond)?;
                match cond_val {
                    Value::Bool(true) => self.eval(env, then_branch),
                    Value::Bool(false) => self.eval(env, else_branch),
                    _ => Err(EvalError::TypeError("expected bool in condition".into())),
                }
            }

            ExprKind::Match { scrutinee, arms } => {
                let val = self.eval(env, scrutinee)?;

                for arm in arms {
                    let new_env = EnvInner::with_parent(env);
                    if self.try_bind_pattern(&new_env, &arm.pattern, &val) {
                        // Check guard if present
                        if let Some(guard) = &arm.guard {
                            let guard_val = self.eval(&new_env, guard)?;
                            if let Value::Bool(false) = guard_val {
                                continue;
                            }
                        }
                        return self.eval(&new_env, &arm.body);
                    }
                }

                Err(EvalError::MatchFailed)
            }

            ExprKind::Tuple(exprs) => {
                let values: Result<Vec<_>, _> =
                    exprs.iter().map(|e| self.eval(env, e)).collect();
                Ok(Value::Tuple(values?))
            }

            ExprKind::List(exprs) => {
                let values: Result<Vec<_>, _> =
                    exprs.iter().map(|e| self.eval(env, e)).collect();
                Ok(Value::List(values?))
            }

            ExprKind::Constructor { name, args } => {
                let values: Result<Vec<_>, _> =
                    args.iter().map(|e| self.eval(env, e)).collect();
                Ok(Value::Constructor {
                    name: name.clone(),
                    fields: values?,
                })
            }

            ExprKind::BinOp { op, left, right } => {
                let l = self.eval(env, left)?;
                let r = self.eval(env, right)?;
                self.eval_binop(*op, l, r)
            }

            ExprKind::UnaryOp { op, operand } => {
                let val = self.eval(env, operand)?;
                self.eval_unaryop(*op, val)
            }

            ExprKind::Seq { first, second } => {
                self.eval(env, first)?;
                self.eval(env, second)
            }

            // Concurrency primitives
            ExprKind::Spawn(body) => {
                let thunk = self.eval(env, body)?;
                let pid = self.runtime.spawn(thunk);
                Ok(Value::Pid(pid))
            }

            ExprKind::NewChannel => {
                let id = self.runtime.new_channel();
                Ok(Value::Channel(id))
            }

            ExprKind::ChanSend { channel, value } => {
                let chan_val = self.eval(env, channel)?;
                let val = self.eval(env, value)?;

                let channel_id = match chan_val {
                    Value::Channel(id) => id,
                    _ => return Err(EvalError::TypeError("expected channel".into())),
                };

                // In a real implementation, this would suspend the process
                // For now, we'll handle it in the scheduler
                if self.runtime.current_pid().is_some() {
                    self.runtime.send(channel_id, val);
                }
                Ok(Value::Unit)
            }

            ExprKind::ChanRecv(channel) => {
                let chan_val = self.eval(env, channel)?;

                let channel_id = match chan_val {
                    Value::Channel(id) => id,
                    _ => return Err(EvalError::TypeError("expected channel".into())),
                };

                // In a real implementation, this would suspend the process
                if let Some(_pid) = self.runtime.current_pid() {
                    if let Some(value) = self.runtime.recv(channel_id) {
                        return Ok(value);
                    }
                }
                // Blocked - return unit for now (scheduler handles this)
                Ok(Value::Unit)
            }

            ExprKind::Select { arms: _ } => {
                // Select is complex - for now, unimplemented
                Err(EvalError::RuntimeError("select not yet implemented".into()))
            }
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

    fn eval_binop(&self, op: BinOp, left: Value, right: Value) -> Result<Value, EvalError> {
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
            (BinOp::Compose, _, _) | (BinOp::ComposeBack, _, _) => {
                Err(EvalError::RuntimeError(
                    "function composition not yet implemented".into(),
                ))
            }

            // Pipe should be desugared
            (BinOp::Pipe, _, _) | (BinOp::PipeBack, _, _) => {
                unreachable!("pipe operators should be desugared")
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

    /// Apply a function to an argument
    fn apply(&mut self, func: Value, arg: Value) -> Result<Value, EvalError> {
        match func {
            Value::Closure { params, body, env } => {
                if params.is_empty() {
                    return Err(EvalError::TypeError("applying non-function".into()));
                }

                let new_env = EnvInner::with_parent(&env);

                // Bind the first parameter
                self.bind_pattern(&new_env, &params[0], &arg)?;

                if params.len() == 1 {
                    // All parameters bound, evaluate body
                    self.eval(&new_env, &body)
                } else {
                    // Return a closure with remaining parameters
                    Ok(Value::Closure {
                        params: params[1..].to_vec(),
                        body,
                        env: new_env,
                    })
                }
            }
            Value::Builtin(name) => self.apply_builtin(&name, arg),
            Value::Constructor { name, fields } => {
                // Constructor application adds to fields
                let mut new_fields = fields;
                new_fields.push(arg);
                Ok(Value::Constructor {
                    name,
                    fields: new_fields,
                })
            }
            _ => Err(EvalError::TypeError(format!(
                "cannot apply {}",
                func.type_name()
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

            (PatternKind::Tuple(pats), Value::Tuple(vals)) if pats.len() == vals.len() => {
                pats.iter()
                    .zip(vals.iter())
                    .all(|(p, v)| self.try_bind_pattern(env, p, v))
            }

            (PatternKind::List(pats), Value::List(vals)) if pats.len() == vals.len() => {
                pats.iter()
                    .zip(vals.iter())
                    .all(|(p, v)| self.try_bind_pattern(env, p, v))
            }

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
            (PatternKind::Constructor { name: pn, args }, Value::Constructor { name: vn, fields })
                if pn == vn && args.is_empty() && fields.is_empty() =>
            {
                true
            }

            _ => false,
        }
    }

    /// Bind a pattern or return an error
    fn bind_pattern(&self, env: &Env, pattern: &Pattern, val: &Value) -> Result<(), EvalError> {
        if self.try_bind_pattern(env, pattern, val) {
            Ok(())
        } else {
            Err(EvalError::MatchFailed)
        }
    }

    /// Run the scheduler until all processes complete or deadlock
    fn run_scheduler(&mut self) -> Result<(), EvalError> {
        // Simple round-robin scheduler
        while let Some(pid) = self.runtime.next_ready() {
            self.runtime.set_current(Some(pid));

            if let Some(cont) = self.runtime.take_continuation(pid) {
                match cont {
                    ProcessContinuation::Start(thunk) => {
                        // Apply the thunk to unit
                        match self.apply(thunk, Value::Unit) {
                            Ok(_) => {
                                self.runtime.mark_done(pid);
                            }
                            Err(e) => {
                                eprintln!("Process {} error: {}", pid, e);
                                self.runtime.mark_done(pid);
                            }
                        }
                    }
                    ProcessContinuation::AfterRecv => {
                        // Process was waiting for a value
                        if let Some(_value) = self.runtime.take_received_value(pid) {
                            // In a full implementation, we'd resume the process
                            // with this value as the result of the recv
                            self.runtime.mark_done(pid);
                        }
                    }
                    ProcessContinuation::AfterSend => {
                        // Send completed, process can continue
                        self.runtime.mark_done(pid);
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
        interp.eval(&env, &expr)
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
}
