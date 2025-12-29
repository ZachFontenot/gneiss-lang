//! Test support infrastructure for comprehensive testing of the Gneiss compiler pipeline.
//!
//! This module provides tools for:
//! - Inspecting intermediate pipeline stages (AST, typed AST, etc.)
//! - Tracing evaluation step-by-step
//! - Recording and asserting on RuntimeEffects
//! - Structured value comparison
//!
//! # Philosophy
//! Tests should verify not just that programs produce correct output, but that
//! the *process* of compilation and evaluation proceeds as expected. This means
//! we need visibility into:
//! - The structure of the parsed AST
//! - Types inferred for expressions
//! - The sequence of interpreter states during evaluation
//! - RuntimeEffects produced and how the scheduler handles them

use crate::ast::{Expr, ExprKind, Program};
use crate::eval::{EnvInner, EvalError, RuntimeEffect, Interpreter, Value};
use crate::infer::Inferencer;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::prelude::parse_prelude;
use crate::types::{InferResult, Type, TypeEnv};

// ============================================================================
// Pipeline Inspection
// ============================================================================

/// Parse an expression and return the AST
pub fn parse_expr(input: &str) -> Result<Expr, String> {
    let tokens = Lexer::new(input)
        .tokenize()
        .map_err(|e| format!("Lexer error: {:?}", e))?;
    let mut parser = Parser::new(tokens);
    parser.parse_expr().map_err(|e| format!("Parse error: {:?}", e))
}

/// Parse a program and return the AST
pub fn parse_program(input: &str) -> Result<Program, String> {
    let tokens = Lexer::new(input)
        .tokenize()
        .map_err(|e| format!("Lexer error: {:?}", e))?;
    let mut parser = Parser::new(tokens);
    parser
        .parse_program()
        .map_err(|e| format!("Parse error: {:?}", e))
}

/// Parse and type-check an expression, returning both AST and inferred type
pub fn typecheck_expr(input: &str) -> Result<(Expr, Type), String> {
    let expr = parse_expr(input)?;
    let mut inferencer = Inferencer::new();
    let env = TypeEnv::new();
    let ty = inferencer
        .infer_expr(&env, &expr)
        .map_err(|e| format!("Type error: {:?}", e))?;
    Ok((expr, ty))
}

/// Parse and type-check an expression, returning full inference result
pub fn typecheck_expr_full(input: &str) -> Result<(Expr, InferResult), String> {
    let expr = parse_expr(input)?;
    let mut inferencer = Inferencer::new();
    let env = TypeEnv::new();
    let result = inferencer
        .infer_expr_full(&env, &expr)
        .map_err(|e| format!("Type error: {:?}", e))?;
    Ok((expr, result))
}

/// Parse and type-check a program, returning AST and type environment
pub fn typecheck_program(input: &str) -> Result<(Program, TypeEnv), String> {
    // Parse prelude
    let prelude = parse_prelude().map_err(|e| format!("Prelude parse error: {:?}", e))?;

    // Parse user program
    let user_program = parse_program(input)?;

    // Combine prelude + user program
    let mut combined_items = prelude.items;
    combined_items.extend(user_program.items);
    let program = Program {
        exports: user_program.exports,
        items: combined_items,
    };

    let mut inferencer = Inferencer::new();
    let env = inferencer
        .infer_program(&program)
        .map_err(|e| format!("Type error: {:?}", e))?;
    Ok((program, env))
}

// ============================================================================
// Evaluation Tracing
// ============================================================================

/// A single step in evaluation
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// Interpreter took a step, transitioning between states
    Step {
        /// Description of the state before the step
        from_state: String,
        /// What happened (Continue, Done, Blocked, Error)
        result: String,
    },
    /// A RuntimeEffect was produced
    Effect(RuntimeEffectTrace),
    /// Scheduler made a decision
    SchedulerEvent(String),
    /// A process state changed
    ProcessEvent {
        pid: u64,
        event: String,
    },
}

/// Traced representation of a RuntimeEffect (without the actual continuations)
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeEffectTrace {
    Done,
    Fork,
    Yield,
    ChanNew,
    Send { channel: u64 },
    Recv { channel: u64 },
    Join { fiber_id: u64 },
    Select { channel_count: usize },
    Io { op_name: String },
}

// Keep the old name as an alias for backward compatibility
pub type FiberEffectTrace = RuntimeEffectTrace;

impl From<&RuntimeEffect> for RuntimeEffectTrace {
    fn from(effect: &RuntimeEffect) -> Self {
        match effect {
            RuntimeEffect::Done(_) => RuntimeEffectTrace::Done,
            RuntimeEffect::Fork { .. } => RuntimeEffectTrace::Fork,
            RuntimeEffect::Yield { .. } => RuntimeEffectTrace::Yield,
            RuntimeEffect::ChanNew { .. } => RuntimeEffectTrace::ChanNew,
            RuntimeEffect::Send { channel, .. } => RuntimeEffectTrace::Send { channel: *channel },
            RuntimeEffect::Recv { channel, .. } => RuntimeEffectTrace::Recv { channel: *channel },
            RuntimeEffect::Join { fiber_id, .. } => RuntimeEffectTrace::Join { fiber_id: *fiber_id },
            RuntimeEffect::Select { arms, .. } => RuntimeEffectTrace::Select {
                channel_count: arms.len(),
            },
            RuntimeEffect::Io { op, .. } => RuntimeEffectTrace::Io {
                op_name: format!("{:?}", op).split('{').next().unwrap_or("Io").trim().to_string(),
            },
        }
    }
}

/// Evaluation trace recorder
#[derive(Debug, Default)]
pub struct EvalTrace {
    pub events: Vec<TraceEvent>,
    pub effects: Vec<FiberEffectTrace>,
    pub final_value: Option<Value>,
    pub error: Option<EvalError>,
}

impl EvalTrace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_effect(&mut self, effect: &RuntimeEffect) {
        let trace = RuntimeEffectTrace::from(effect);
        self.effects.push(trace.clone());
        self.events.push(TraceEvent::Effect(trace));
    }

    pub fn record_step(&mut self, from_state: &str, result: &str) {
        self.events.push(TraceEvent::Step {
            from_state: from_state.to_string(),
            result: result.to_string(),
        });
    }

    pub fn record_scheduler(&mut self, event: &str) {
        self.events
            .push(TraceEvent::SchedulerEvent(event.to_string()));
    }

    pub fn record_process(&mut self, pid: u64, event: &str) {
        self.events.push(TraceEvent::ProcessEvent {
            pid,
            event: event.to_string(),
        });
    }

    /// Check that specific effects were produced in order
    pub fn assert_effects(&self, expected: &[FiberEffectTrace]) {
        assert_eq!(
            self.effects, expected,
            "Effect sequence mismatch.\nExpected: {:?}\nActual: {:?}",
            expected, self.effects
        );
    }

    /// Check that at least these effects were produced (in order, possibly with others between)
    pub fn assert_effects_contain(&self, expected: &[FiberEffectTrace]) {
        let mut expected_iter = expected.iter();
        let mut current_expected = expected_iter.next();

        for actual in &self.effects {
            if let Some(exp) = current_expected {
                if actual == exp {
                    current_expected = expected_iter.next();
                }
            }
        }

        assert!(
            current_expected.is_none(),
            "Missing expected effects. Expected: {:?}\nActual: {:?}",
            expected,
            self.effects
        );
    }

    /// Print a human-readable trace for debugging
    pub fn print_trace(&self) {
        println!("=== Evaluation Trace ===");
        for (i, event) in self.events.iter().enumerate() {
            match event {
                TraceEvent::Step { from_state, result } => {
                    println!("{:4}: STEP {} -> {}", i, from_state, result);
                }
                TraceEvent::Effect(eff) => {
                    println!("{:4}: EFFECT {:?}", i, eff);
                }
                TraceEvent::SchedulerEvent(msg) => {
                    println!("{:4}: SCHEDULER {}", i, msg);
                }
                TraceEvent::ProcessEvent { pid, event } => {
                    println!("{:4}: PROCESS {} {}", i, pid, event);
                }
            }
        }
        println!("=== End Trace ===");
    }
}

// ============================================================================
// Value Comparison and Assertions
// ============================================================================

/// Deep equality for Values, ignoring internal details like closure environments
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        (Value::Channel(x), Value::Channel(y)) => x == y,
        (Value::Pid(x), Value::Pid(y)) => x == y,
        (Value::Fiber(x), Value::Fiber(y)) => x == y,
        (Value::List(xs), Value::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| values_equal(x, y))
        }
        (Value::Tuple(xs), Value::Tuple(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| values_equal(x, y))
        }
        (
            Value::Constructor {
                name: name_a,
                fields: fields_a,
            },
            Value::Constructor {
                name: name_b,
                fields: fields_b,
            },
        ) => {
            name_a == name_b
                && fields_a.len() == fields_b.len()
                && fields_a
                    .iter()
                    .zip(fields_b.iter())
                    .all(|(a, b)| values_equal(a, b))
        }
        // Closures are compared by identity (can't really compare them structurally)
        (Value::Closure { .. }, Value::Closure { .. }) => false,
        _ => false,
    }
}

/// Assert that evaluation produces a specific value
pub fn assert_eval_value(input: &str, expected: Value) {
    let result = eval_expr(input);
    match result {
        Ok(actual) => {
            assert!(
                values_equal(&actual, &expected),
                "Value mismatch for: {}\nExpected: {:?}\nActual: {:?}",
                input,
                expected,
                actual
            );
        }
        Err(e) => {
            panic!(
                "Evaluation failed for: {}\nExpected: {:?}\nError: {:?}",
                input, expected, e
            );
        }
    }
}

/// Assert that evaluation produces an integer
pub fn assert_eval_int(input: &str, expected: i64) {
    assert_eval_value(input, Value::Int(expected));
}

/// Assert that evaluation produces a boolean
pub fn assert_eval_bool(input: &str, expected: bool) {
    assert_eval_value(input, Value::Bool(expected));
}

/// Assert that evaluation produces a string
pub fn assert_eval_string(input: &str, expected: &str) {
    assert_eval_value(input, Value::String(expected.to_string()));
}

/// Assert that evaluation fails with a specific error type
pub fn assert_eval_error<F>(input: &str, check: F)
where
    F: FnOnce(&EvalError) -> bool,
{
    let result = eval_expr(input);
    match result {
        Ok(v) => {
            panic!(
                "Expected error for: {}\nBut got value: {:?}",
                input, v
            );
        }
        Err(e) => {
            assert!(
                check(&e),
                "Error type mismatch for: {}\nActual error: {:?}",
                input,
                e
            );
        }
    }
}

/// Assert that type inference produces a specific type
pub fn assert_type(input: &str, expected_type: &str) {
    match typecheck_expr(input) {
        Ok((_, ty)) => {
            let ty_str = format!("{}", ty);
            assert_eq!(
                ty_str, expected_type,
                "Type mismatch for: {}\nExpected: {}\nActual: {}",
                input, expected_type, ty_str
            );
        }
        Err(e) => {
            panic!("Type inference failed for: {}\nError: {}", input, e);
        }
    }
}

/// Assert that type inference fails
pub fn assert_type_error(input: &str) {
    match typecheck_expr(input) {
        Ok((_, ty)) => {
            panic!(
                "Expected type error for: {}\nBut got type: {}",
                input, ty
            );
        }
        Err(_) => {
            // Expected
        }
    }
}

// ============================================================================
// Simple Evaluation Helpers
// ============================================================================

/// Evaluate an expression and return the value
pub fn eval_expr(input: &str) -> Result<Value, EvalError> {
    let tokens = Lexer::new(input).tokenize().map_err(|e| {
        EvalError::RuntimeError(format!("Lexer error: {:?}", e))
    })?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr().map_err(|e| {
        EvalError::RuntimeError(format!("Parse error: {:?}", e))
    })?;
    let mut interp = Interpreter::new();
    let env = EnvInner::new();
    interp.eval_expr(&env, &expr)
}

/// Run a program and return the final result (if any)
/// This performs full type checking before execution.
pub fn run_program(input: &str) -> Result<Option<Value>, EvalError> {
    // Parse prelude
    let prelude = parse_prelude().map_err(|e| {
        EvalError::RuntimeError(format!("Prelude parse error: {:?}", e))
    })?;

    // Parse user program
    let tokens = Lexer::new(input).tokenize().map_err(|e| {
        EvalError::RuntimeError(format!("Lexer error: {:?}", e))
    })?;
    let mut parser = Parser::new(tokens);
    let user_program = parser.parse_program().map_err(|e| {
        EvalError::RuntimeError(format!("Parse error: {:?}", e))
    })?;

    // Combine prelude + user program
    let mut combined_items = prelude.items;
    combined_items.extend(user_program.items);
    let program = Program {
        exports: user_program.exports,
        items: combined_items,
    };

    // Type check before running - this is critical for catching type errors!
    let mut inferencer = Inferencer::new();
    inferencer.infer_program(&program).map_err(|e| {
        EvalError::RuntimeError(format!("Type error: {:?}", e))
    })?;

    // Pass class environment and type context to interpreter for trait resolution
    let mut interp = Interpreter::new();
    interp.set_class_env(inferencer.take_class_env());
    interp.set_type_ctx(inferencer.take_type_ctx());
    interp.run(&program)?;
    Ok(None) // TODO: capture main's return value
}

/// Run a program expecting success
pub fn run_program_ok(input: &str) {
    match run_program(input) {
        Ok(_) => {}
        Err(e) => panic!("Program failed: {:?}\nProgram:\n{}", e, input),
    }
}

/// Run a program expecting a specific error
pub fn run_program_err<F>(input: &str, check: F)
where
    F: FnOnce(&EvalError) -> bool,
{
    match run_program(input) {
        Ok(_) => panic!("Expected error but program succeeded.\nProgram:\n{}", input),
        Err(e) => {
            assert!(
                check(&e),
                "Wrong error type.\nActual: {:?}\nProgram:\n{}",
                e,
                input
            );
        }
    }
}

// ============================================================================
// AST Inspection Helpers
// ============================================================================

/// Check if an expression matches a specific pattern
pub fn expr_matches<F>(input: &str, predicate: F) -> bool
where
    F: FnOnce(&ExprKind) -> bool,
{
    match parse_expr(input) {
        Ok(expr) => predicate(&expr.node),
        Err(_) => false,
    }
}

/// Get the top-level expression kind
pub fn expr_kind(input: &str) -> Option<String> {
    parse_expr(input).ok().map(|e| format!("{:?}", std::mem::discriminant(&e.node)))
}

// ============================================================================
// Test Macros
// ============================================================================

/// Macro for asserting evaluation produces an integer
#[macro_export]
macro_rules! assert_evals_to_int {
    ($input:expr, $expected:expr) => {
        $crate::test_support::assert_eval_int($input, $expected)
    };
}

/// Macro for asserting evaluation produces a boolean
#[macro_export]
macro_rules! assert_evals_to_bool {
    ($input:expr, $expected:expr) => {
        $crate::test_support::assert_eval_bool($input, $expected)
    };
}

/// Macro for asserting a type
#[macro_export]
macro_rules! assert_has_type {
    ($input:expr, $expected:expr) => {
        $crate::test_support::assert_type($input, $expected)
    };
}

/// Macro for asserting program runs successfully
#[macro_export]
macro_rules! assert_program_ok {
    ($input:expr) => {
        $crate::test_support::run_program_ok($input)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expr() {
        let expr = parse_expr("1 + 2").unwrap();
        assert!(matches!(&expr.node, ExprKind::BinOp { .. }));
    }

    #[test]
    fn test_typecheck_expr() {
        let (_, ty) = typecheck_expr("1 + 2").unwrap();
        assert_eq!(format!("{}", ty), "Int");
    }

    #[test]
    fn test_assert_eval_int() {
        assert_eval_int("1 + 2", 3);
        assert_eval_int("10 * 5", 50);
    }

    #[test]
    fn test_assert_eval_bool() {
        assert_eval_bool("true && false", false);
        assert_eval_bool("1 < 2", true);
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&Value::Int(42), &Value::Int(42)));
        assert!(!values_equal(&Value::Int(42), &Value::Int(43)));
        assert!(values_equal(
            &Value::List(im::vector![Value::Int(1), Value::Int(2)]),
            &Value::List(im::vector![Value::Int(1), Value::Int(2)])
        ));
    }

    #[test]
    fn test_effect_trace() {
        use crate::eval::Cont;
        let mut trace = EvalTrace::new();
        trace.record_effect(&RuntimeEffect::ChanNew { cont: Cont::new() });
        trace.record_effect(&RuntimeEffect::Fork {
            thunk: Box::new(Value::Unit),
            cont: Cont::new(),
        });

        trace.assert_effects(&[RuntimeEffectTrace::ChanNew, RuntimeEffectTrace::Fork]);
    }

    #[test]
    fn test_assert_type() {
        assert_type("42", "Int");
        assert_type("true", "Bool");
        assert_type("\"hello\"", "String");
    }

    #[test]
    fn test_assert_type_error() {
        assert_type_error("1 + true");
    }

    #[test]
    fn test_expr_matches() {
        assert!(expr_matches("1 + 2", |e| matches!(e, ExprKind::BinOp { .. })));
        assert!(expr_matches("fun x -> x", |e| matches!(e, ExprKind::Lambda { .. })));
        assert!(!expr_matches("1 + 2", |e| matches!(e, ExprKind::Lambda { .. })));
    }
}
