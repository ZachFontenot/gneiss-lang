//! Monomorphization Pass
//!
//! Transforms the polymorphic TAST into a monomorphized program where
//! every function is specialized to concrete types. This eliminates
//! all type variables before lowering to Core IR.
//!
//! Key insight: After monomorphization, there is no polymorphism.
//! Every function has concrete types, making codegen straightforward.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ast::{Ident, Literal, Span};
use crate::tast::{
    TBinOp, TBinding, TExpr, TExprKind, TMatchArm, TPattern, TPatternKind, TProgram,
    TRecBinding, TUnaryOp,
};
use crate::types::Type;

// ============================================================================
// Monomorphized Types (no type variables)
// ============================================================================

/// A monomorphized type - no type variables remain
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MonoType {
    Int,
    Float,
    Bool,
    String,
    Char,
    Unit,
    Bytes,
    /// Tuple of concrete types
    Tuple(Vec<MonoType>),
    /// Named type constructor with concrete type arguments
    /// e.g., List Int, Option String
    Constructor { name: String, args: Vec<MonoType> },
    /// Function type with concrete parameter and return types
    Function { params: Vec<MonoType>, ret: Box<MonoType> },
}

impl MonoType {
    /// Convert from Type, substituting any type variables
    pub fn from_type(ty: &Type, subst: &HashMap<u32, MonoType>) -> Self {
        match ty {
            Type::Int => MonoType::Int,
            Type::Float => MonoType::Float,
            Type::Bool => MonoType::Bool,
            Type::String => MonoType::String,
            Type::Char => MonoType::Char,
            Type::Unit => MonoType::Unit,
            Type::Bytes => MonoType::Bytes,
            Type::Var(id) | Type::Generic(id) => {
                subst.get(id).cloned().unwrap_or(MonoType::Unit)
            }
            Type::Tuple(elems) => {
                MonoType::Tuple(elems.iter().map(|t| MonoType::from_type(t, subst)).collect())
            }
            Type::Constructor { name, args } => MonoType::Constructor {
                name: name.clone(),
                args: args.iter().map(|t| MonoType::from_type(t, subst)).collect(),
            },
            Type::Arrow { arg, ret, .. } => MonoType::Function {
                params: vec![MonoType::from_type(arg, subst)],
                ret: Box::new(MonoType::from_type(ret, subst)),
            },
            Type::Channel(inner) => MonoType::Constructor {
                name: "Channel".to_string(),
                args: vec![MonoType::from_type(inner, subst)],
            },
            Type::Pid => MonoType::Constructor {
                name: "Pid".to_string(),
                args: vec![],
            },
            Type::Fiber(inner) => MonoType::Constructor {
                name: "Fiber".to_string(),
                args: vec![MonoType::from_type(inner, subst)],
            },
            Type::Dict(inner) => MonoType::Constructor {
                name: "Dict".to_string(),
                args: vec![MonoType::from_type(inner, subst)],
            },
            Type::Set => MonoType::Constructor {
                name: "Set".to_string(),
                args: vec![],
            },
            // IO types
            Type::FileHandle => MonoType::Constructor {
                name: "FileHandle".to_string(),
                args: vec![],
            },
            Type::TcpSocket => MonoType::Constructor {
                name: "TcpSocket".to_string(),
                args: vec![],
            },
            Type::TcpListener => MonoType::Constructor {
                name: "TcpListener".to_string(),
                args: vec![],
            },
        }
    }

    /// Generate a mangled name for this type (for function name suffixes)
    pub fn mangle(&self) -> String {
        match self {
            MonoType::Int => "Int".to_string(),
            MonoType::Float => "Float".to_string(),
            MonoType::Bool => "Bool".to_string(),
            MonoType::String => "String".to_string(),
            MonoType::Char => "Char".to_string(),
            MonoType::Unit => "Unit".to_string(),
            MonoType::Bytes => "Bytes".to_string(),
            MonoType::Tuple(elems) => {
                let parts: Vec<_> = elems.iter().map(|t| t.mangle()).collect();
                format!("Tuple_{}", parts.join("_"))
            }
            MonoType::Constructor { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let parts: Vec<_> = args.iter().map(|t| t.mangle()).collect();
                    format!("{}_{}", name, parts.join("_"))
                }
            }
            MonoType::Function { params, ret } => {
                let param_parts: Vec<_> = params.iter().map(|t| t.mangle()).collect();
                format!("Fn_{}_{}", param_parts.join("_"), ret.mangle())
            }
        }
    }
}

// ============================================================================
// Monomorphized Function Identity
// ============================================================================

/// A function identity that includes its type instantiation
/// e.g., "map" instantiated with Int becomes MonoFnId { base_name: "map", type_args: [Int] }
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MonoFnId {
    pub base_name: String,
    pub type_args: Vec<MonoType>,
}

impl MonoFnId {
    pub fn new(name: &str) -> Self {
        MonoFnId {
            base_name: name.to_string(),
            type_args: vec![],
        }
    }

    pub fn with_args(name: &str, args: Vec<MonoType>) -> Self {
        MonoFnId {
            base_name: name.to_string(),
            type_args: args,
        }
    }

    /// Generate the mangled function name for codegen
    pub fn mangled_name(&self) -> String {
        if self.type_args.is_empty() {
            self.base_name.clone()
        } else {
            let type_suffix: Vec<_> = self.type_args.iter().map(|t| t.mangle()).collect();
            format!("{}_{}", self.base_name, type_suffix.join("_"))
        }
    }
}

// ============================================================================
// Monomorphized Expressions
// ============================================================================

/// A monomorphized expression - all types are concrete
#[derive(Debug, Clone)]
pub struct MonoExpr {
    pub node: MonoExprKind,
    pub ty: MonoType,
    pub span: Span,
}

impl MonoExpr {
    pub fn new(node: MonoExprKind, ty: MonoType, span: Span) -> Self {
        MonoExpr { node, ty, span }
    }
}

/// Monomorphized expression kinds
#[derive(Debug, Clone)]
pub enum MonoExprKind {
    /// Variable reference
    Var(Ident),

    /// Literal value
    Lit(Literal),

    /// Lambda (will be lifted during closure conversion)
    Lambda {
        params: Vec<MonoPattern>,
        body: Box<MonoExpr>,
    },

    /// Function application
    App {
        func: Box<MonoExpr>,
        arg: Box<MonoExpr>,
    },

    /// Direct call to a known monomorphized function
    Call {
        func: MonoFnId,
        args: Vec<MonoExpr>,
    },

    /// Let binding
    Let {
        pattern: MonoPattern,
        value: Box<MonoExpr>,
        body: Option<Box<MonoExpr>>,
    },

    /// Recursive let
    LetRec {
        bindings: Vec<MonoRecBinding>,
        body: Option<Box<MonoExpr>>,
    },

    /// If expression
    If {
        cond: Box<MonoExpr>,
        then_branch: Box<MonoExpr>,
        else_branch: Box<MonoExpr>,
    },

    /// Pattern match
    Match {
        scrutinee: Box<MonoExpr>,
        arms: Vec<MonoMatchArm>,
    },

    /// Tuple construction
    Tuple(Vec<MonoExpr>),

    /// List construction
    List(Vec<MonoExpr>),

    /// Record construction
    Record {
        name: Ident,
        fields: Vec<(Ident, MonoExpr)>,
    },

    /// Field access
    FieldAccess {
        record: Box<MonoExpr>,
        field: Ident,
    },

    /// Constructor application
    Constructor {
        name: Ident,
        args: Vec<MonoExpr>,
    },

    /// Binary operation (already type-specific from TAST)
    BinOp {
        op: TBinOp,
        left: Box<MonoExpr>,
        right: Box<MonoExpr>,
    },

    /// Unary operation
    UnaryOp {
        op: TUnaryOp,
        operand: Box<MonoExpr>,
    },

    /// Sequence
    Seq {
        first: Box<MonoExpr>,
        second: Box<MonoExpr>,
    },

    /// Error placeholder
    Error(String),
}

// ============================================================================
// Monomorphized Patterns
// ============================================================================

#[derive(Debug, Clone)]
pub struct MonoPattern {
    pub node: MonoPatternKind,
    pub ty: MonoType,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum MonoPatternKind {
    Wildcard,
    Var(Ident),
    Lit(Literal),
    Tuple(Vec<MonoPattern>),
    List(Vec<MonoPattern>),
    Cons {
        head: Box<MonoPattern>,
        tail: Box<MonoPattern>,
    },
    Constructor {
        name: Ident,
        args: Vec<MonoPattern>,
    },
    Record {
        name: Ident,
        fields: Vec<(Ident, Option<MonoPattern>)>,
    },
}

// ============================================================================
// Supporting Structures
// ============================================================================

#[derive(Debug, Clone)]
pub struct MonoMatchArm {
    pub pattern: MonoPattern,
    pub guard: Option<Box<MonoExpr>>,
    pub body: MonoExpr,
}

#[derive(Debug, Clone)]
pub struct MonoRecBinding {
    pub name: Ident,
    pub params: Vec<MonoPattern>,
    pub body: MonoExpr,
    pub ty: MonoType,
}

// ============================================================================
// Monomorphized Function
// ============================================================================

#[derive(Debug, Clone)]
pub struct MonoFn {
    pub id: MonoFnId,
    pub params: Vec<MonoPattern>,
    pub body: MonoExpr,
    pub return_type: MonoType,
}

// ============================================================================
// Monomorphized Program
// ============================================================================

/// A fully monomorphized program - no type variables remain
#[derive(Debug, Clone)]
pub struct MonoProgram {
    /// All monomorphized functions (including specializations)
    pub functions: HashMap<MonoFnId, MonoFn>,
    /// Type definitions (unchanged from TAST)
    pub type_defs: Vec<MonoTypeDef>,
    /// Main expression (if present)
    pub main: Option<MonoExpr>,
}

#[derive(Debug, Clone)]
pub struct MonoTypeDef {
    pub name: String,
    pub variants: Vec<MonoVariant>,
}

#[derive(Debug, Clone)]
pub struct MonoVariant {
    pub name: String,
    pub fields: Vec<MonoType>,
}

// ============================================================================
// Monomorphization Context
// ============================================================================

/// Context for monomorphization pass
pub struct MonoCtx<'a> {
    /// The source TAST program
    tprogram: &'a TProgram,
    /// Work queue of functions to monomorphize
    work_queue: VecDeque<(String, Vec<MonoType>)>,
    /// Already processed functions
    done: HashMap<MonoFnId, MonoFn>,
    /// Current type substitution (type var id -> concrete type)
    subst: HashMap<u32, MonoType>,
    /// Set of function IDs currently being processed (to detect recursion)
    in_progress: HashSet<MonoFnId>,
}

impl<'a> MonoCtx<'a> {
    pub fn new(tprogram: &'a TProgram) -> Self {
        MonoCtx {
            tprogram,
            work_queue: VecDeque::new(),
            done: HashMap::new(),
            subst: HashMap::new(),
            in_progress: HashSet::new(),
        }
    }

    /// Queue a function for monomorphization
    pub fn queue_function(&mut self, name: &str, type_args: Vec<MonoType>) {
        let fn_id = MonoFnId::with_args(name, type_args.clone());
        if !self.done.contains_key(&fn_id) && !self.in_progress.contains(&fn_id) {
            self.work_queue.push_back((name.to_string(), type_args));
        }
    }

    /// Find a binding by name in the TAST
    fn find_binding(&self, name: &str) -> Option<&TBinding> {
        self.tprogram.bindings.iter().find(|b| b.name == name)
    }

    /// Convert a Type to MonoType using current substitution
    fn mono_type(&self, ty: &Type) -> MonoType {
        MonoType::from_type(ty, &self.subst)
    }

    /// Monomorphize an expression
    fn mono_expr(&mut self, expr: &TExpr) -> MonoExpr {
        let ty = self.mono_type(&expr.ty);
        let span = expr.span.clone();

        let node = match &expr.node {
            TExprKind::Var(name) => MonoExprKind::Var(name.clone()),

            TExprKind::Lit(lit) => MonoExprKind::Lit(lit.clone()),

            TExprKind::Lambda { params, body } => {
                let mono_params: Vec<_> = params.iter().map(|p| self.mono_pattern(p)).collect();
                let mono_body = self.mono_expr(body);
                MonoExprKind::Lambda {
                    params: mono_params,
                    body: Box::new(mono_body),
                }
            }

            TExprKind::App { func, arg } => {
                let mono_func = self.mono_expr(func);
                let mono_arg = self.mono_expr(arg);
                MonoExprKind::App {
                    func: Box::new(mono_func),
                    arg: Box::new(mono_arg),
                }
            }

            TExprKind::Let { pattern, value, body } => {
                let mono_pat = self.mono_pattern(pattern);
                let mono_val = self.mono_expr(value);
                let mono_body = body.as_ref().map(|b| Box::new(self.mono_expr(b)));
                MonoExprKind::Let {
                    pattern: mono_pat,
                    value: Box::new(mono_val),
                    body: mono_body,
                }
            }

            TExprKind::LetRec { bindings, body } => {
                let mono_bindings: Vec<_> = bindings
                    .iter()
                    .map(|b| self.mono_rec_binding(b))
                    .collect();
                let mono_body = body.as_ref().map(|b| Box::new(self.mono_expr(b)));
                MonoExprKind::LetRec {
                    bindings: mono_bindings,
                    body: mono_body,
                }
            }

            TExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let mono_cond = self.mono_expr(cond);
                let mono_then = self.mono_expr(then_branch);
                let mono_else = self.mono_expr(else_branch);
                MonoExprKind::If {
                    cond: Box::new(mono_cond),
                    then_branch: Box::new(mono_then),
                    else_branch: Box::new(mono_else),
                }
            }

            TExprKind::Match { scrutinee, arms } => {
                let mono_scrutinee = self.mono_expr(scrutinee);
                let mono_arms: Vec<_> = arms.iter().map(|a| self.mono_match_arm(a)).collect();
                MonoExprKind::Match {
                    scrutinee: Box::new(mono_scrutinee),
                    arms: mono_arms,
                }
            }

            TExprKind::Tuple(elems) => {
                let mono_elems: Vec<_> = elems.iter().map(|e| self.mono_expr(e)).collect();
                MonoExprKind::Tuple(mono_elems)
            }

            TExprKind::List(elems) => {
                let mono_elems: Vec<_> = elems.iter().map(|e| self.mono_expr(e)).collect();
                MonoExprKind::List(mono_elems)
            }

            TExprKind::Record { name, fields } => {
                let mono_fields: Vec<_> = fields
                    .iter()
                    .map(|(n, e)| (n.clone(), self.mono_expr(e)))
                    .collect();
                MonoExprKind::Record {
                    name: name.clone(),
                    fields: mono_fields,
                }
            }

            TExprKind::FieldAccess { record, field } => {
                let mono_record = self.mono_expr(record);
                MonoExprKind::FieldAccess {
                    record: Box::new(mono_record),
                    field: field.clone(),
                }
            }

            TExprKind::RecordUpdate { base, updates } => {
                // Desugar record update to record construction
                // For now, treat as field access + record construction
                let mono_base = self.mono_expr(base);
                let mono_updates: Vec<_> = updates
                    .iter()
                    .map(|(n, e)| (n.clone(), self.mono_expr(e)))
                    .collect();
                // Simplified: just return the updates as a record
                // Full implementation would merge with base
                MonoExprKind::Record {
                    name: "Updated".to_string(),
                    fields: mono_updates,
                }
            }

            TExprKind::Constructor { name, args } => {
                let mono_args: Vec<_> = args.iter().map(|a| self.mono_expr(a)).collect();
                MonoExprKind::Constructor {
                    name: name.clone(),
                    args: mono_args,
                }
            }

            TExprKind::BinOp { op, left, right } => {
                let mono_left = self.mono_expr(left);
                let mono_right = self.mono_expr(right);
                MonoExprKind::BinOp {
                    op: *op,
                    left: Box::new(mono_left),
                    right: Box::new(mono_right),
                }
            }

            TExprKind::UnaryOp { op, operand } => {
                let mono_operand = self.mono_expr(operand);
                MonoExprKind::UnaryOp {
                    op: *op,
                    operand: Box::new(mono_operand),
                }
            }

            TExprKind::Seq { first, second } => {
                let mono_first = self.mono_expr(first);
                let mono_second = self.mono_expr(second);
                MonoExprKind::Seq {
                    first: Box::new(mono_first),
                    second: Box::new(mono_second),
                }
            }

            TExprKind::MethodCall {
                trait_name,
                method,
                instance_ty,
                args,
            } => {
                // Monomorphize trait method call to direct call
                let concrete_ty = self.mono_type(instance_ty);
                let mangled = format!("{}_{}_{}", trait_name, concrete_ty.mangle(), method);
                let fn_id = MonoFnId::new(&mangled);

                let mono_args: Vec<_> = args.iter().map(|a| self.mono_expr(a)).collect();

                MonoExprKind::Call {
                    func: fn_id,
                    args: mono_args,
                }
            }

            TExprKind::DictMethodCall { .. } => {
                // Dictionary method calls should not exist after proper elaboration
                // For now, return an error
                MonoExprKind::Error("DictMethodCall should be resolved".to_string())
            }

            TExprKind::DictValue { .. } | TExprKind::DictRef { .. } => {
                // These are artifacts of the old approach - should not exist
                MonoExprKind::Error("Dict nodes should not exist".to_string())
            }

            TExprKind::Perform { effect, op, args } => {
                // Effects will be handled in a later phase
                // For now, preserve the structure
                let mono_args: Vec<_> = args.iter().map(|a| self.mono_expr(a)).collect();
                // Convert to a placeholder - effects handled in effect lowering phase
                MonoExprKind::Call {
                    func: MonoFnId::new(&format!("perform_{}_{}", effect, op)),
                    args: mono_args,
                }
            }

            TExprKind::Handle { body, handler } => {
                // Effects handled in later phase
                let mono_body = self.mono_expr(body);
                // For now, just return the body - proper effect handling comes later
                mono_body.node
            }

            TExprKind::Error(msg) => MonoExprKind::Error(msg.clone()),
        };

        MonoExpr::new(node, ty, span)
    }

    /// Monomorphize a pattern
    fn mono_pattern(&self, pattern: &TPattern) -> MonoPattern {
        let ty = self.mono_type(&pattern.ty);
        let span = pattern.span.clone();

        let node = match &pattern.node {
            TPatternKind::Wildcard => MonoPatternKind::Wildcard,
            TPatternKind::Var(name) => MonoPatternKind::Var(name.clone()),
            TPatternKind::Lit(lit) => MonoPatternKind::Lit(lit.clone()),
            TPatternKind::Tuple(elems) => {
                let mono_elems: Vec<_> = elems.iter().map(|p| self.mono_pattern(p)).collect();
                MonoPatternKind::Tuple(mono_elems)
            }
            TPatternKind::List(elems) => {
                let mono_elems: Vec<_> = elems.iter().map(|p| self.mono_pattern(p)).collect();
                MonoPatternKind::List(mono_elems)
            }
            TPatternKind::Cons { head, tail } => {
                let mono_head = self.mono_pattern(head);
                let mono_tail = self.mono_pattern(tail);
                MonoPatternKind::Cons {
                    head: Box::new(mono_head),
                    tail: Box::new(mono_tail),
                }
            }
            TPatternKind::Constructor { name, args } => {
                let mono_args: Vec<_> = args.iter().map(|p| self.mono_pattern(p)).collect();
                MonoPatternKind::Constructor {
                    name: name.clone(),
                    args: mono_args,
                }
            }
            TPatternKind::Record { name, fields } => {
                let mono_fields: Vec<_> = fields
                    .iter()
                    .map(|(n, p)| (n.clone(), p.as_ref().map(|pat| self.mono_pattern(pat))))
                    .collect();
                MonoPatternKind::Record {
                    name: name.clone(),
                    fields: mono_fields,
                }
            }
        };

        MonoPattern { node, ty, span }
    }

    /// Monomorphize a match arm
    fn mono_match_arm(&mut self, arm: &TMatchArm) -> MonoMatchArm {
        let mono_pattern = self.mono_pattern(&arm.pattern);
        let mono_guard = arm.guard.as_ref().map(|g| Box::new(self.mono_expr(g)));
        let mono_body = self.mono_expr(&arm.body);

        MonoMatchArm {
            pattern: mono_pattern,
            guard: mono_guard,
            body: mono_body,
        }
    }

    /// Monomorphize a recursive binding
    fn mono_rec_binding(&mut self, binding: &TRecBinding) -> MonoRecBinding {
        let mono_params: Vec<_> = binding
            .params
            .iter()
            .map(|p| self.mono_pattern(p))
            .collect();
        let mono_body = self.mono_expr(&binding.body);
        let mono_ty = self.mono_type(&binding.ty);

        MonoRecBinding {
            name: binding.name.clone(),
            params: mono_params,
            body: mono_body,
            ty: mono_ty,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a type contains any Generic type variables
fn type_has_generics(ty: &Type) -> bool {
    match ty {
        Type::Generic(_) => true,
        Type::Constructor { args, .. } => args.iter().any(|t| type_has_generics(t)),
        Type::Arrow { arg, ret, .. } => type_has_generics(arg) || type_has_generics(ret),
        Type::Tuple(elems) => elems.iter().any(|t| type_has_generics(t)),
        Type::Channel(inner) | Type::Fiber(inner) | Type::Dict(inner) => type_has_generics(inner),
        Type::Int | Type::Float | Type::String | Type::Bool | Type::Char
        | Type::Unit | Type::Bytes | Type::FileHandle | Type::TcpSocket
        | Type::TcpListener | Type::Pid | Type::Set => false,
        Type::Var(_) => false, // Type vars are resolved, not generic
    }
}

// ============================================================================
// Main Monomorphization Function
// ============================================================================

/// Monomorphize a typed program
pub fn monomorphize(tprogram: &TProgram) -> Result<MonoProgram, String> {
    let mut ctx = MonoCtx::new(tprogram);

    // Monomorphize main expression if present
    let main = if let Some(main_expr) = &tprogram.main {
        Some(ctx.mono_expr(main_expr))
    } else {
        None
    };

    // Process all top-level bindings
    // For Phase 1, we're not doing full demand-driven monomorphization yet
    // Just process all bindings with their concrete types
    for binding in &tprogram.bindings {
        let fn_id = MonoFnId::new(&binding.name);

        if ctx.done.contains_key(&fn_id) {
            continue;
        }

        ctx.in_progress.insert(fn_id.clone());

        let mono_params: Vec<_> = binding
            .params
            .iter()
            .map(|p| ctx.mono_pattern(p))
            .collect();
        let mono_body = ctx.mono_expr(&binding.body);
        let return_type = ctx.mono_type(&binding.ty);

        let mono_fn = MonoFn {
            id: fn_id.clone(),
            params: mono_params,
            body: mono_body,
            return_type,
        };

        ctx.in_progress.remove(&fn_id);
        ctx.done.insert(fn_id, mono_fn);
    }

    // Process trait instance declarations
    // These create monomorphized functions like Show_Int_show, Show_Unit_show, etc.
    // Skip generic instances (those with Generic type variables) as they require
    // dictionary passing which isn't supported yet.
    for instance in &tprogram.instance_decls {
        // Skip generic instances - they can't be monomorphized without specialization
        if type_has_generics(&instance.instance_ty) {
            continue;
        }

        for method in &instance.methods {
            let fn_id = MonoFnId::new(&method.mangled_name);

            if ctx.done.contains_key(&fn_id) {
                continue;
            }

            ctx.in_progress.insert(fn_id.clone());

            let mono_params: Vec<_> = method
                .params
                .iter()
                .map(|p| ctx.mono_pattern(p))
                .collect();
            let mono_body = ctx.mono_expr(&method.body);

            // Infer return type from body
            let return_type = ctx.mono_type(&method.body.ty);

            let mono_fn = MonoFn {
                id: fn_id.clone(),
                params: mono_params,
                body: mono_body,
                return_type,
            };

            ctx.in_progress.remove(&fn_id);
            ctx.done.insert(fn_id, mono_fn);
        }
    }

    // Convert type definitions
    let type_defs: Vec<_> = tprogram
        .type_decls
        .iter()
        .map(|td| MonoTypeDef {
            name: td.name.clone(),
            variants: td
                .constructors
                .iter()
                .map(|c| MonoVariant {
                    name: c.name.clone(),
                    fields: c
                        .fields
                        .iter()
                        .map(|f| MonoType::from_type(f, &HashMap::new()))
                        .collect(),
                })
                .collect(),
        })
        .collect();

    Ok(MonoProgram {
        functions: ctx.done,
        type_defs,
        main,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_type_mangle() {
        assert_eq!(MonoType::Int.mangle(), "Int");
        assert_eq!(MonoType::String.mangle(), "String");

        let list_int = MonoType::Constructor {
            name: "List".to_string(),
            args: vec![MonoType::Int],
        };
        assert_eq!(list_int.mangle(), "List_Int");

        let tuple = MonoType::Tuple(vec![MonoType::Int, MonoType::Bool]);
        assert_eq!(tuple.mangle(), "Tuple_Int_Bool");
    }

    #[test]
    fn test_mono_fn_id() {
        let id = MonoFnId::new("map");
        assert_eq!(id.mangled_name(), "map");

        let id_with_args = MonoFnId::with_args("map", vec![MonoType::Int]);
        assert_eq!(id_with_args.mangled_name(), "map_Int");
    }
}
