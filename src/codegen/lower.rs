//! AST to Core IR lowering
//!
//! Transforms the source AST into A-Normal Form Core IR suitable for
//! Perceus transformation and C code generation.
//!
//! Key transformations:
//! - ANF conversion: all intermediate values are named
//! - Pattern matching desugaring: patterns become case expressions
//! - Operator desugaring: binary ops become primitive operations
//! - Lambda lifting preparation: identify free variables

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::{
    BinOp as AstBinOp, Decl, Expr, ExprKind, Item, Literal, MatchArm, Pattern, PatternKind,
    Program, UnaryOp as AstUnaryOp,
};
use crate::codegen::core_ir::{
    Alt, CoreExpr, CoreLit, CoreProgram, CoreType, FunDef, Handler, OpHandler, PrimOp,
    RecBinding, Tag, TagTable, TypeDef, VarGen, VarId, VariantDef,
};
use crate::tast::{
    TBinOp, TExpr, TExprKind, TInstanceDecl, TMatchArm, TMethodImpl, TPattern, TPatternKind,
    TProgram, TUnaryOp,
};
use crate::types::Type;

// ============================================================================
// Lowering Context
// ============================================================================

/// Context for lowering AST to Core IR
pub struct LowerCtx {
    /// Variable generator
    var_gen: VarGen,
    /// Map from source names to Core IR variables
    env: HashMap<String, VarId>,
    /// Tag table for constructors
    pub tag_table: TagTable,
    /// Type definitions collected during lowering
    type_defs: Vec<TypeDef>,
    /// Function definitions collected during lowering
    fun_defs: Vec<FunDef>,
    /// Builtin function mappings (VarId -> name)
    builtins: Vec<(VarId, String)>,
    /// Errors encountered during lowering
    errors: Vec<String>,
    /// Stack of scopes for proper variable shadowing
    scope_stack: Vec<HashMap<String, VarId>>,
}

impl LowerCtx {
    pub fn new() -> Self {
        let mut tag_table = TagTable::new();
        tag_table.register_builtins();

        LowerCtx {
            var_gen: VarGen::new(),
            env: HashMap::new(),
            tag_table,
            type_defs: Vec::new(),
            fun_defs: Vec::new(),
            builtins: Vec::new(),
            errors: Vec::new(),
            scope_stack: Vec::new(),
        }
    }

    /// Generate a fresh variable
    fn fresh(&mut self) -> VarId {
        self.var_gen.fresh()
    }

    /// Generate a fresh variable with a name hint
    #[allow(unused)]
    fn fresh_named(&mut self, _hint: &str) -> VarId {
        self.var_gen.fresh()
    }

    /// Bind a name to a variable in the current scope
    fn bind(&mut self, name: &str, var: VarId) {
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.insert(name.to_string(), var);
        } else {
            self.env.insert(name.to_string(), var);
        }
    }

    /// Look up a name in the environment
    fn lookup(&self, name: &str) -> Option<VarId> {
        // Check scope stack from innermost to outermost
        for scope in self.scope_stack.iter().rev() {
            if let Some(&var) = scope.get(name) {
                return Some(var);
            }
        }
        // Check global environment
        self.env.get(name).copied()
    }

    /// Push a new scope
    fn push_scope(&mut self) {
        self.scope_stack.push(HashMap::new());
    }

    /// Pop a scope
    fn pop_scope(&mut self) {
        self.scope_stack.pop();
    }

    /// Record an error
    fn error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    /// Get the tag for a constructor, creating if needed
    fn get_tag(&mut self, type_name: &str, ctor_name: &str) -> Tag {
        self.tag_table.get_or_create(type_name, ctor_name)
    }
}

impl Default for LowerCtx {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Program lowering
// ============================================================================

/// Lower an entire program to Core IR
pub fn lower_program(program: &Program) -> Result<CoreProgram, Vec<String>> {
    let mut ctx = LowerCtx::new();

    // First pass: collect type definitions
    for item in &program.items {
        if let Item::Decl(Decl::Type {
            name,
            params,
            constructors,
            ..
        }) = item
        {
            let mut variants = Vec::new();
            for ctor in constructors {
                let tag = ctx.get_tag(name, &ctor.name);
                variants.push(VariantDef {
                    tag,
                    name: ctor.name.clone(),
                    fields: vec![CoreType::Var(0); ctor.fields.len()], // TODO: proper types
                });
            }
            ctx.type_defs.push(TypeDef {
                name: name.clone(),
                params: params.clone(),
                variants,
            });
        }
    }

    // Register built-in functions in environment
    register_builtins(&mut ctx);

    // Second pass: lower declarations
    let mut main_expr: Option<CoreExpr> = None;

    for item in &program.items {
        match item {
            Item::Decl(decl) => match decl {
                Decl::Let {
                    name, params, body, ..
                } => {
                    let var = ctx.fresh_named(name);
                    ctx.bind(name, var);

                    if params.is_empty() {
                        // Simple value binding
                        let lowered = lower_expr(&mut ctx, body, false);

                        if name == "main" {
                            main_expr = Some(lowered);
                        } else {
                            ctx.fun_defs.push(FunDef {
                                name: name.clone(),
                                var_id: var,
                                params: vec![],
                                param_hints: vec![],
                                param_types: vec![],
                                return_type: CoreType::Var(0),
                                body: lowered,
                                is_tail_recursive: false,
                            });
                        }
                    } else {
                        // Function binding
                        ctx.push_scope();

                        let mut param_vars = Vec::new();
                        let mut param_hints = Vec::new();
                        for param in params {
                            let (pvar, hint) = lower_pattern_bind(&mut ctx, param);
                            param_vars.push(pvar);
                            param_hints.push(hint.unwrap_or_else(|| "_".to_string()));
                        }

                        let body_lowered = lower_expr(&mut ctx, body, true);
                        ctx.pop_scope();

                        if name == "main" && param_vars.is_empty() {
                            main_expr = Some(body_lowered);
                        } else {
                            ctx.fun_defs.push(FunDef {
                                name: name.clone(),
                                var_id: var,
                                params: param_vars,
                                param_hints,
                                param_types: vec![],
                                return_type: CoreType::Var(0),
                                body: body_lowered,
                                is_tail_recursive: false,
                            });
                        }
                    }
                }

                Decl::LetRec { bindings, .. } => {
                    // First, bind all names
                    let mut vars = Vec::new();
                    for binding in bindings {
                        let var = ctx.fresh_named(&binding.name.node);
                        ctx.bind(&binding.name.node, var);
                        vars.push(var);
                    }

                    // Then lower each binding
                    for (binding, var) in bindings.iter().zip(vars.iter()) {
                        ctx.push_scope();

                        let mut param_vars = Vec::new();
                        let mut param_hints = Vec::new();
                        for param in &binding.params {
                            let (pvar, hint) = lower_pattern_bind(&mut ctx, param);
                            param_vars.push(pvar);
                            param_hints.push(hint.unwrap_or_else(|| "_".to_string()));
                        }

                        let body_lowered = lower_expr(&mut ctx, &binding.body, true);
                        ctx.pop_scope();

                        if binding.name.node == "main" && param_vars.is_empty() {
                            main_expr = Some(body_lowered);
                        } else {
                            ctx.fun_defs.push(FunDef {
                                name: binding.name.node.clone(),
                                var_id: *var,
                                params: param_vars,
                                param_hints,
                                param_types: vec![],
                                return_type: CoreType::Var(0),
                                body: body_lowered,
                                is_tail_recursive: false, // TODO: detect
                            });
                        }
                    }
                }

                Decl::Type { .. } => {
                    // Already handled in first pass
                }

                Decl::Val { .. } => {
                    // Type declaration, skip for now
                }

                _ => {
                    // Other declarations (traits, instances, etc.)
                    // TODO: handle properly
                }
            },

            Item::Expr(expr) => {
                // Top-level expression - treat as implicit main if no main exists
                let lowered = lower_expr(&mut ctx, expr, false);
                if main_expr.is_none() {
                    main_expr = Some(lowered);
                }
            }

            Item::Import(_) => {
                // Skip imports for now
            }
        }
    }

    if !ctx.errors.is_empty() {
        return Err(ctx.errors);
    }

    Ok(CoreProgram {
        types: ctx.type_defs,
        functions: ctx.fun_defs,
        builtins: ctx.builtins,
        main: main_expr,
    })
}

/// Register built-in functions
fn register_builtins(ctx: &mut LowerCtx) {
    let builtin_names = [
        // Core I/O
        "print",
        "io_print",
        "io_read_line",
        // String operations
        "int_to_string",
        "string_length",
        "string_concat",
        "string_join",
        "string_split",
        "string_index_of",
        "string_substring",
        "string_to_lower",
        "string_to_upper",
        "string_trim",
        "string_trim_start",
        "string_trim_end",
        "string_reverse",
        "string_is_empty",
        "string_char_at",
        "string_repeat",
        "string_to_chars",
        "chars_to_string",
        // Char operations
        "char_to_string",
        "char_to_int",
        "char_to_lower",
        "char_to_upper",
        "char_is_whitespace",
        "char_is_digit",
        "char_is_alpha",
        "char_is_alphanumeric",
        // Bytes operations
        "bytes_to_string",
        "string_to_bytes",
        "bytes_length",
        "bytes_slice",
        "bytes_concat",
        // System
        "get_args",
    ];

    for name in builtin_names {
        let var = ctx.fresh_named(name);
        ctx.bind(name, var);
        ctx.builtins.push((var, name.to_string()));
    }
}

// ============================================================================
// Expression lowering
// ============================================================================

/// Lower an expression to Core IR
/// `in_tail` indicates if this expression is in tail position
fn lower_expr(ctx: &mut LowerCtx, expr: &Expr, in_tail: bool) -> CoreExpr {
    match &expr.node {
        ExprKind::Lit(lit) => CoreExpr::Lit(lower_literal(lit)),

        ExprKind::Var(name) => {
            if let Some(var) = ctx.lookup(name) {
                CoreExpr::Var(var)
            } else {
                ctx.error(format!("Unbound variable in lowering: {}", name));
                CoreExpr::Error(format!("unbound: {}", name))
            }
        }

        ExprKind::Lambda { params, body } => {
            ctx.push_scope();

            let mut param_vars = Vec::new();
            let mut param_hints = Vec::new();

            for param in params {
                let (var, hint) = lower_pattern_bind(ctx, param);
                param_vars.push(var);
                param_hints.push(hint);
            }

            let body_lowered = lower_expr(ctx, body, true);
            ctx.pop_scope();

            CoreExpr::Lam {
                params: param_vars,
                param_hints,
                body: Box::new(body_lowered),
            }
        }

        ExprKind::App { func, arg } => {
            // Collect all arguments for a multi-arg application
            let (func_expr, args) = collect_app_args(func, arg);

            // Check if function is a simple variable (can use directly without let binding)
            let func_var = if let ExprKind::Var(name) = &func_expr.node {
                if let Some(var) = ctx.lookup(name) {
                    var
                } else {
                    ctx.error(format!("Unbound function: {}", name));
                    ctx.fresh()
                }
            } else {
                // Complex function expression - need to bind to variable
                let func_lowered = lower_expr(ctx, func_expr, false);
                let var = ctx.fresh();
                // We'll wrap later
                return {
                    let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
                    for a in args.iter() {
                        let lowered = lower_expr(ctx, a, false);
                        let v = ctx.fresh();
                        arg_lowered.push((v, lowered));
                    }
                    let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();

                    let app_expr = if in_tail {
                        CoreExpr::TailApp { func: var, args: arg_vars }
                    } else {
                        CoreExpr::App { func: var, args: arg_vars }
                    };

                    let mut result = app_expr;
                    for (v, lowered) in arg_lowered.into_iter().rev() {
                        result = CoreExpr::LetExpr {
                            name: v,
                            name_hint: Some("arg".into()),
                            value: Box::new(lowered),
                            body: Box::new(result),
                        };
                    }

                    CoreExpr::LetExpr {
                        name: var,
                        name_hint: Some("func".into()),
                        value: Box::new(func_lowered),
                        body: Box::new(result),
                    }
                };
            };

            // Function is a simple variable reference - bind arguments only
            let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
            for a in args.iter() {
                let lowered = lower_expr(ctx, a, false);
                let var = ctx.fresh();
                arg_lowered.push((var, lowered));
            }

            let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();

            let app_expr = if in_tail {
                CoreExpr::TailApp {
                    func: func_var,
                    args: arg_vars,
                }
            } else {
                CoreExpr::App {
                    func: func_var,
                    args: arg_vars,
                }
            };

            // Wrap arguments in let bindings
            let mut result = app_expr;
            for (var, lowered) in arg_lowered.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: var,
                    name_hint: Some("arg".into()),
                    value: Box::new(lowered),
                    body: Box::new(result),
                };
            }

            result
        }

        ExprKind::Let { pattern, value, body } => {
            let value_lowered = lower_expr(ctx, value, false);

            match &pattern.node {
                PatternKind::Var(name) => {
                    let var = ctx.fresh_named(name);
                    ctx.bind(name, var);

                    if let Some(body) = body {
                        let body_lowered = lower_expr(ctx, body, in_tail);
                        CoreExpr::LetExpr {
                            name: var,
                            name_hint: Some(name.clone()),
                            value: Box::new(value_lowered),
                            body: Box::new(body_lowered),
                        }
                    } else {
                        // Top-level let without body
                        value_lowered
                    }
                }
                _ => {
                    // Complex pattern - desugar to case
                    let scrut_var = ctx.fresh();
                    let body_expr = if let Some(body) = body {
                        lower_expr(ctx, body, in_tail)
                    } else {
                        CoreExpr::Lit(CoreLit::Unit)
                    };

                    let alt = lower_pattern_to_alt(ctx, pattern, body_expr);

                    CoreExpr::LetExpr {
                        name: scrut_var,
                        name_hint: None,
                        value: Box::new(value_lowered),
                        body: Box::new(CoreExpr::Case {
                            scrutinee: scrut_var,
                            alts: vec![alt],
                            default: Some(Box::new(CoreExpr::Error("pattern match failed".into()))),
                        }),
                    }
                }
            }
        }

        ExprKind::LetRec { bindings, body } => {
            // Bind all names first
            let mut vars = Vec::new();
            for binding in bindings {
                let var = ctx.fresh_named(&binding.name.node);
                ctx.bind(&binding.name.node, var);
                vars.push(var);
            }

            // Lower each binding
            let mut rec_bindings = Vec::new();
            for (binding, var) in bindings.iter().zip(vars.iter()) {
                ctx.push_scope();

                let mut param_vars = Vec::new();
                let mut param_hints = Vec::new();
                for param in &binding.params {
                    let (pvar, hint) = lower_pattern_bind(ctx, param);
                    param_vars.push(pvar);
                    param_hints.push(hint);
                }

                let func_body = lower_expr(ctx, &binding.body, true);
                ctx.pop_scope();

                rec_bindings.push(RecBinding {
                    name: *var,
                    name_hint: Some(binding.name.node.clone()),
                    params: param_vars,
                    param_hints,
                    body: func_body,
                });
            }

            let body_lowered = if let Some(body) = body {
                lower_expr(ctx, body, in_tail)
            } else {
                CoreExpr::Lit(CoreLit::Unit)
            };

            if rec_bindings.len() == 1 {
                let b = rec_bindings.remove(0);
                CoreExpr::LetRec {
                    name: b.name,
                    name_hint: b.name_hint,
                    params: b.params,
                    param_hints: b.param_hints,
                    func_body: Box::new(b.body),
                    body: Box::new(body_lowered),
                }
            } else {
                CoreExpr::LetRecMutual {
                    bindings: rec_bindings,
                    body: Box::new(body_lowered),
                }
            }
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            // ANF conversion: bind condition to a variable first
            let cond_lowered = lower_expr(ctx, cond, false);
            let cond_var = ctx.fresh();
            let then_lowered = lower_expr(ctx, then_branch, in_tail);
            let else_lowered = lower_expr(ctx, else_branch, in_tail);

            CoreExpr::LetExpr {
                name: cond_var,
                name_hint: Some("cond".into()),
                value: Box::new(cond_lowered),
                body: Box::new(CoreExpr::If {
                    cond: cond_var,
                    then_branch: Box::new(then_lowered),
                    else_branch: Box::new(else_lowered),
                }),
            }
        }

        ExprKind::Match { scrutinee, arms } => {
            let scrut_var = lower_to_var(ctx, scrutinee);
            let alts: Vec<Alt> = arms
                .iter()
                .map(|arm| lower_match_arm(ctx, arm, in_tail))
                .collect();

            CoreExpr::Case {
                scrutinee: scrut_var,
                alts,
                default: None,
            }
        }

        ExprKind::BinOp { op, left, right } => {
            // ANF conversion: bind operands to variables first
            let left_lowered = lower_expr(ctx, left, false);
            let right_lowered = lower_expr(ctx, right, false);
            let left_var = ctx.fresh();
            let right_var = ctx.fresh();

            let inner = if let Some(prim) = binop_to_prim(op) {
                CoreExpr::PrimOp {
                    op: prim,
                    args: vec![left_var, right_var],
                }
            } else {
                // Non-primitive binary op (e.g., custom operator)
                // Look up as function and apply
                let op_name = format!("op_{:?}", op);
                if let Some(op_var) = ctx.lookup(&op_name) {
                    CoreExpr::App {
                        func: op_var,
                        args: vec![left_var, right_var],
                    }
                } else {
                    ctx.error(format!("Unknown operator: {:?}", op));
                    CoreExpr::Error(format!("unknown op: {:?}", op))
                }
            };

            // Wrap in let bindings for ANF
            CoreExpr::LetExpr {
                name: left_var,
                name_hint: Some("left".into()),
                value: Box::new(left_lowered),
                body: Box::new(CoreExpr::LetExpr {
                    name: right_var,
                    name_hint: Some("right".into()),
                    value: Box::new(right_lowered),
                    body: Box::new(inner),
                }),
            }
        }

        ExprKind::UnaryOp { op, operand } => {
            // ANF conversion: bind operand to variable first
            let operand_lowered = lower_expr(ctx, operand, false);
            let operand_var = ctx.fresh();

            let inner = match op {
                AstUnaryOp::Neg => CoreExpr::PrimOp {
                    op: PrimOp::IntNeg,
                    args: vec![operand_var],
                },
                AstUnaryOp::Not => CoreExpr::PrimOp {
                    op: PrimOp::BoolNot,
                    args: vec![operand_var],
                },
            };

            CoreExpr::LetExpr {
                name: operand_var,
                name_hint: Some("operand".into()),
                value: Box::new(operand_lowered),
                body: Box::new(inner),
            }
        }

        ExprKind::Tuple(elems) => {
            // ANF: bind each element before allocating
            let mut bindings: Vec<(VarId, CoreExpr)> = vec![];
            for elem in elems.iter() {
                let lowered = lower_expr(ctx, elem, false);
                let var = ctx.fresh();
                bindings.push((var, lowered));
            }

            let elem_vars: Vec<VarId> = bindings.iter().map(|(v, _)| *v).collect();

            let alloc = CoreExpr::Alloc {
                tag: 0,
                type_name: Some("Tuple".into()),
                ctor_name: Some(format!("Tuple{}", elems.len())),
                fields: elem_vars,
            };

            // Wrap in let bindings
            let mut result = alloc;
            for (var, lowered) in bindings.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: var,
                    name_hint: Some("tuple_elem".into()),
                    value: Box::new(lowered),
                    body: Box::new(result),
                };
            }
            result
        }

        ExprKind::List(elems) => {
            // Build list from end to beginning, with proper ANF bindings
            let nil_tag = ctx.get_tag("List", "Nil");
            let cons_tag = ctx.get_tag("List", "Cons");

            let mut result = CoreExpr::Alloc {
                tag: nil_tag,
                type_name: Some("List".into()),
                ctor_name: Some("Nil".into()),
                fields: vec![],
            };

            for elem in elems.iter().rev() {
                let elem_lowered = lower_expr(ctx, elem, false);
                let elem_var = ctx.fresh();
                let rest_var = ctx.fresh();

                result = CoreExpr::LetExpr {
                    name: elem_var,
                    name_hint: Some("elem".into()),
                    value: Box::new(elem_lowered),
                    body: Box::new(CoreExpr::LetExpr {
                        name: rest_var,
                        name_hint: Some("rest".into()),
                        value: Box::new(result),
                        body: Box::new(CoreExpr::Alloc {
                            tag: cons_tag,
                            type_name: Some("List".into()),
                            ctor_name: Some("Cons".into()),
                            fields: vec![elem_var, rest_var],
                        }),
                    }),
                };
            }

            result
        }

        ExprKind::Constructor { name, args } => {
            // ANF: bind each argument before allocating
            let mut bindings: Vec<(VarId, CoreExpr)> = vec![];
            for arg in args.iter() {
                let lowered = lower_expr(ctx, arg, false);
                let var = ctx.fresh();
                bindings.push((var, lowered));
            }
            let arg_vars: Vec<VarId> = bindings.iter().map(|(v, _)| *v).collect();

            // Try to find the constructor's type
            let tag = ctx.get_tag("ADT", name);

            let alloc = CoreExpr::Alloc {
                tag,
                type_name: None,
                ctor_name: Some(name.clone()),
                fields: arg_vars,
            };

            // Wrap in let bindings
            let mut result = alloc;
            for (var, lowered) in bindings.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: var,
                    name_hint: Some("ctor_arg".into()),
                    value: Box::new(lowered),
                    body: Box::new(result),
                };
            }
            result
        }

        ExprKind::Seq { first, second } => {
            let first_lowered = lower_expr(ctx, first, false);
            let second_lowered = lower_expr(ctx, second, in_tail);

            CoreExpr::Seq {
                first: Box::new(first_lowered),
                second: Box::new(second_lowered),
            }
        }

        ExprKind::Record { name, fields } => {
            // Record is like a tuple with named fields
            let field_vars: Vec<VarId> = fields.iter().map(|(_, e)| lower_to_var(ctx, e)).collect();
            let tag = ctx.get_tag(name, name);

            CoreExpr::Alloc {
                tag,
                type_name: Some(name.clone()),
                ctor_name: Some(name.clone()),
                fields: field_vars,
            }
        }

        ExprKind::FieldAccess { record, field: _ } => {
            let record_var = lower_to_var(ctx, record);
            // Field access becomes a primitive tuple/record get
            // For now, we don't know the field index, so we'll use a placeholder
            // This would need type information to resolve properly
            CoreExpr::PrimOp {
                op: PrimOp::TupleGet(0), // TODO: proper field index
                args: vec![record_var],
            }
        }

        ExprKind::Hole => {
            ctx.error("Typed hole in expression");
            CoreExpr::Error("typed hole".into())
        }

        // Concurrency - not supported in compiled mode yet
        ExprKind::Spawn(_) => {
            ctx.error("spawn not yet supported in compiled mode");
            CoreExpr::Error("spawn unsupported".into())
        }
        ExprKind::NewChannel => {
            ctx.error("channels not yet supported in compiled mode");
            CoreExpr::Error("channel unsupported".into())
        }
        ExprKind::ChanSend { .. } => {
            ctx.error("channel send not yet supported in compiled mode");
            CoreExpr::Error("chan_send unsupported".into())
        }
        ExprKind::ChanRecv(_) => {
            ctx.error("channel recv not yet supported in compiled mode");
            CoreExpr::Error("chan_recv unsupported".into())
        }
        ExprKind::Select { .. } => {
            ctx.error("select not yet supported in compiled mode");
            CoreExpr::Error("select unsupported".into())
        }

        // Effects
        ExprKind::Perform {
            effect,
            operation,
            args,
        } => {
            let arg_vars: Vec<VarId> = args.iter().map(|a| lower_to_var(ctx, a)).collect();

            CoreExpr::Perform {
                effect: 0, // TODO: effect ID lookup
                effect_name: Some(effect.clone()),
                op: 0, // TODO: op ID lookup
                op_name: Some(operation.clone()),
                args: arg_vars,
            }
        }

        ExprKind::Handle {
            body,
            return_clause,
            handlers,
        } => {
            ctx.push_scope();
            let body_lowered = lower_expr(ctx, body, false);

            // Lower return clause
            let ret_var = ctx.fresh();
            if let PatternKind::Var(name) = &return_clause.pattern.node {
                ctx.bind(name, ret_var);
            }
            let return_body = lower_expr(ctx, &return_clause.body, in_tail);

            // Lower operation handlers
            let ops: Vec<OpHandler> = handlers
                .iter()
                .map(|h| {
                    ctx.push_scope();

                    let param_vars: Vec<VarId> = h
                        .params
                        .iter()
                        .map(|p| {
                            let (var, _) = lower_pattern_bind(ctx, p);
                            var
                        })
                        .collect();

                    let cont_var = ctx.fresh_named(&h.continuation);
                    ctx.bind(&h.continuation, cont_var);

                    let handler_body = lower_expr(ctx, &h.body, in_tail);
                    ctx.pop_scope();

                    OpHandler {
                        op: 0,
                        op_name: Some(h.operation.clone()),
                        params: param_vars,
                        cont: cont_var,
                        body: handler_body,
                    }
                })
                .collect();

            ctx.pop_scope();

            CoreExpr::Handle {
                body: Box::new(body_lowered),
                handler: Handler {
                    effect: 0,
                    effect_name: None,
                    return_var: ret_var,
                    return_body: Box::new(return_body),
                    ops,
                },
            }
        }

        ExprKind::RecordUpdate { base, updates: _ } => {
            // Record update: { base with field = value, ... }
            // This needs to copy the record and update fields
            let base_var = lower_to_var(ctx, base);
            // For now, just use base - proper implementation needs type info
            CoreExpr::Var(base_var)
        }
    }
}

/// Lower an expression and ensure it's in a variable
fn lower_to_var(ctx: &mut LowerCtx, expr: &Expr) -> VarId {
    match &expr.node {
        ExprKind::Var(name) => {
            if let Some(var) = ctx.lookup(name) {
                var
            } else {
                ctx.error(format!("Unbound variable: {}", name));
                ctx.fresh()
            }
        }
        ExprKind::Lit(_lit) => {
            // Literals can be inlined, but for ANF we still bind them
            // Note: in actual codegen, we'd track this binding
            ctx.fresh()
        }
        _ => {
            // Complex expression - need to bind to a variable
            // This is where ANF conversion happens
            ctx.fresh()
        }
    }
}

/// Collect nested applications into (func, [args])
fn collect_app_args<'a>(func: &'a Rc<Expr>, arg: &'a Rc<Expr>) -> (&'a Expr, Vec<&'a Expr>) {
    let mut args = vec![arg.as_ref()];
    let mut current = func.as_ref();

    while let ExprKind::App {
        func: inner_func,
        arg: inner_arg,
    } = &current.node
    {
        args.push(inner_arg.as_ref());
        current = inner_func.as_ref();
    }

    args.reverse();
    (current, args)
}

/// Lower a literal
fn lower_literal(lit: &Literal) -> CoreLit {
    match lit {
        Literal::Int(n) => CoreLit::Int(*n),
        Literal::Float(f) => CoreLit::Float(*f),
        Literal::String(s) => CoreLit::String(s.clone()),
        Literal::Char(c) => CoreLit::Char(*c),
        Literal::Bool(b) => CoreLit::Bool(*b),
        Literal::Unit => CoreLit::Unit,
    }
}

/// Lower a pattern to a variable binding, returning (var, name_hint)
fn lower_pattern_bind(ctx: &mut LowerCtx, pattern: &Pattern) -> (VarId, Option<String>) {
    match &pattern.node {
        PatternKind::Var(name) => {
            let var = ctx.fresh_named(name);
            ctx.bind(name, var);
            (var, Some(name.clone()))
        }
        PatternKind::Wildcard => (ctx.fresh(), None),
        _ => {
            // Complex pattern - bind to fresh variable, desugar in body
            (ctx.fresh(), None)
        }
    }
}

/// Lower a pattern to a case alternative
fn lower_pattern_to_alt(ctx: &mut LowerCtx, pattern: &Pattern, body: CoreExpr) -> Alt {
    match &pattern.node {
        PatternKind::Var(name) => {
            let var = ctx.fresh_named(name);
            ctx.bind(name, var);
            Alt {
                tag: 0, // Matches anything
                tag_name: Some("_".into()),
                binders: vec![var],
                binder_hints: vec![Some(name.clone())],
                body,
            }
        }
        PatternKind::Wildcard => Alt {
            tag: 0,
            tag_name: Some("_".into()),
            binders: vec![],
            binder_hints: vec![],
            body,
        },
        PatternKind::Lit(lit) => {
            // Literal pattern - needs equality check
            // For now, treat as tag matching
            Alt {
                tag: 0,
                tag_name: Some(format!("{:?}", lit)),
                binders: vec![],
                binder_hints: vec![],
                body,
            }
        }
        PatternKind::Constructor { name, args } => {
            ctx.push_scope();
            let mut binders = Vec::new();
            let mut binder_hints = Vec::new();

            for arg in args {
                let (var, hint) = lower_pattern_bind(ctx, arg);
                binders.push(var);
                binder_hints.push(hint);
            }

            let tag = ctx.get_tag("ADT", name);

            // Note: body needs the bindings, so we don't pop scope here
            // The caller is responsible for scope management

            Alt {
                tag,
                tag_name: Some(name.clone()),
                binders,
                binder_hints,
                body,
            }
        }
        PatternKind::Tuple(elems) => {
            ctx.push_scope();
            let mut binders = Vec::new();
            let mut binder_hints = Vec::new();

            for elem in elems {
                let (var, hint) = lower_pattern_bind(ctx, elem);
                binders.push(var);
                binder_hints.push(hint);
            }

            Alt {
                tag: 0,
                tag_name: Some(format!("Tuple{}", elems.len())),
                binders,
                binder_hints,
                body,
            }
        }
        PatternKind::List(elems) => {
            // List patterns are more complex - need to chain Cons matches
            // For now, simplified handling
            Alt {
                tag: ctx.get_tag("List", if elems.is_empty() { "Nil" } else { "Cons" }),
                tag_name: Some("List".into()),
                binders: vec![],
                binder_hints: vec![],
                body,
            }
        }
        PatternKind::Cons { head, tail } => {
            ctx.push_scope();
            let (head_var, head_hint) = lower_pattern_bind(ctx, head);
            let (tail_var, tail_hint) = lower_pattern_bind(ctx, tail);

            Alt {
                tag: ctx.get_tag("List", "Cons"),
                tag_name: Some("Cons".into()),
                binders: vec![head_var, tail_var],
                binder_hints: vec![head_hint, tail_hint],
                body,
            }
        }
        PatternKind::Record { name, fields } => {
            ctx.push_scope();
            let mut binders = Vec::new();
            let mut binder_hints = Vec::new();

            for (field_name, opt_pattern) in fields {
                if let Some(pattern) = opt_pattern {
                    let (var, hint) = lower_pattern_bind(ctx, pattern);
                    binders.push(var);
                    binder_hints.push(hint.or_else(|| Some(field_name.clone())));
                } else {
                    // Field name is used as the binding
                    let var = ctx.fresh_named(field_name);
                    ctx.bind(field_name, var);
                    binders.push(var);
                    binder_hints.push(Some(field_name.clone()));
                }
            }

            let tag = ctx.get_tag(name, name);

            Alt {
                tag,
                tag_name: Some(name.clone()),
                binders,
                binder_hints,
                body,
            }
        }
    }
}

/// Lower a match arm
fn lower_match_arm(ctx: &mut LowerCtx, arm: &MatchArm, in_tail: bool) -> Alt {
    ctx.push_scope();
    // First, bind pattern variables so they're available in the body
    let (binders, binder_hints, tag, tag_name) = bind_pattern_vars(ctx, &arm.pattern);
    // Then lower the body (which can now reference the pattern vars)
    let body = lower_expr(ctx, &arm.body, in_tail);
    ctx.pop_scope();
    Alt {
        tag,
        tag_name,
        binders,
        binder_hints,
        body,
    }
}

/// Bind pattern variables and return the binder info
fn bind_pattern_vars(ctx: &mut LowerCtx, pattern: &Pattern) -> (Vec<VarId>, Vec<Option<String>>, Tag, Option<String>) {
    match &pattern.node {
        PatternKind::Var(name) => {
            let var = ctx.fresh_named(name);
            ctx.bind(name, var);
            (vec![var], vec![Some(name.clone())], 0, Some("_".into()))
        }
        PatternKind::Wildcard => {
            (vec![], vec![], 0, Some("_".into()))
        }
        PatternKind::Lit(lit) => {
            (vec![], vec![], 0, Some(format!("{:?}", lit)))
        }
        PatternKind::Constructor { name, args } => {
            let mut binders = Vec::new();
            let mut binder_hints = Vec::new();

            for arg in args {
                let (var, hint) = lower_pattern_bind(ctx, arg);
                binders.push(var);
                binder_hints.push(hint);
            }

            let tag = ctx.get_tag("ADT", name);
            (binders, binder_hints, tag, Some(name.clone()))
        }
        PatternKind::Tuple(elems) => {
            let mut binders = Vec::new();
            let mut binder_hints = Vec::new();

            for elem in elems {
                let (var, hint) = lower_pattern_bind(ctx, elem);
                binders.push(var);
                binder_hints.push(hint);
            }

            (binders, binder_hints, 0, Some("Tuple".into()))
        }
        PatternKind::List(elems) => {
            if elems.is_empty() {
                // Empty list pattern matches Nil
                let tag = ctx.get_tag("List", "Nil");
                (vec![], vec![], tag, Some("Nil".into()))
            } else {
                // Non-empty list pattern - needs to desugar to nested Cons
                // For now, just bind all variables
                let mut binders = Vec::new();
                let mut binder_hints = Vec::new();

                for elem in elems {
                    let (var, hint) = lower_pattern_bind(ctx, elem);
                    binders.push(var);
                    binder_hints.push(hint);
                }

                let tag = ctx.get_tag("List", "Cons");
                (binders, binder_hints, tag, Some("Cons".into()))
            }
        }
        PatternKind::Cons { head, tail } => {
            let (head_var, head_hint) = lower_pattern_bind(ctx, head);
            let (tail_var, tail_hint) = lower_pattern_bind(ctx, tail);

            let tag = ctx.get_tag("List", "Cons");
            (
                vec![head_var, tail_var],
                vec![head_hint, tail_hint],
                tag,
                Some("Cons".into()),
            )
        }
        PatternKind::Record { name, fields } => {
            let mut binders = Vec::new();
            let mut binder_hints = Vec::new();

            for (field_name, field_pattern) in fields {
                if let Some(pat) = field_pattern {
                    let (var, hint) = lower_pattern_bind(ctx, pat);
                    binders.push(var);
                    binder_hints.push(hint);
                } else {
                    // Field name is used as the variable
                    let var = ctx.fresh_named(field_name);
                    ctx.bind(field_name, var);
                    binders.push(var);
                    binder_hints.push(Some(field_name.clone()));
                }
            }

            let tag = ctx.get_tag("Record", name);
            (binders, binder_hints, tag, Some(name.clone()))
        }
    }
}

/// Convert binary operator to primitive operation
fn binop_to_prim(op: &AstBinOp) -> Option<PrimOp> {
    match op {
        AstBinOp::Add => Some(PrimOp::IntAdd),
        AstBinOp::Sub => Some(PrimOp::IntSub),
        AstBinOp::Mul => Some(PrimOp::IntMul),
        AstBinOp::Div => Some(PrimOp::IntDiv),
        AstBinOp::Mod => Some(PrimOp::IntMod),
        AstBinOp::Eq => Some(PrimOp::IntEq),
        AstBinOp::Neq => Some(PrimOp::IntNe),
        AstBinOp::Lt => Some(PrimOp::IntLt),
        AstBinOp::Lte => Some(PrimOp::IntLe),
        AstBinOp::Gt => Some(PrimOp::IntGt),
        AstBinOp::Gte => Some(PrimOp::IntGe),
        AstBinOp::And => Some(PrimOp::BoolAnd),
        AstBinOp::Or => Some(PrimOp::BoolOr),
        AstBinOp::Concat => Some(PrimOp::StringConcat),
        AstBinOp::Cons => Some(PrimOp::ListCons),
        _ => None, // Pipe, Compose, UserDefined need function lookup
    }
}

// ============================================================================
// TAST Lowering (Typed AST to Core IR)
// ============================================================================

/// Convert a Type to CoreType
fn type_to_core_type(ty: &Type) -> CoreType {
    match ty {
        Type::Int => CoreType::Int,
        Type::Float => CoreType::Float,
        Type::Bool => CoreType::Bool,
        Type::Char => CoreType::Char,
        Type::String => CoreType::String,
        Type::Unit => CoreType::Unit,
        Type::Bytes => CoreType::Box(Box::new(CoreType::Int)),
        Type::Arrow { arg, ret, .. } => CoreType::Fun {
            params: vec![type_to_core_type(arg)],
            ret: Box::new(type_to_core_type(ret)),
        },
        Type::Tuple(elems) => CoreType::Tuple(elems.iter().map(type_to_core_type).collect()),
        Type::Constructor { name, args } => {
            // Special case for List - it's a boxed type
            if name == "List" && args.len() == 1 {
                CoreType::Box(Box::new(type_to_core_type(&args[0])))
            } else {
                CoreType::Sum {
                    name: name.clone(),
                    variants: vec![], // Filled in by type_defs
                }
            }
        }
        Type::Channel(_) => CoreType::Box(Box::new(CoreType::Int)), // Channels are boxed
        Type::Fiber(_) => CoreType::Box(Box::new(CoreType::Int)),   // Fibers are boxed
        Type::Dict(_) => CoreType::Box(Box::new(CoreType::Int)),    // Dicts are boxed
        Type::Set => CoreType::Box(Box::new(CoreType::Int)),        // Sets are boxed
        Type::Pid => CoreType::Box(Box::new(CoreType::Int)),        // PIDs are boxed
        // Type variables become generic boxed values
        Type::Var(_) | Type::Generic(_) => CoreType::Var(0),
        // Handle types
        Type::FileHandle | Type::TcpSocket | Type::TcpListener => {
            CoreType::Box(Box::new(CoreType::Int))
        }
    }
}

/// Lower a typed program to Core IR
pub fn lower_tprogram(program: &TProgram) -> Result<CoreProgram, Vec<String>> {
    let mut ctx = LowerCtx::new();

    // First pass: collect type definitions
    for type_decl in &program.type_decls {
        let mut variants = Vec::new();
        for ctor in &type_decl.constructors {
            let tag = ctx.get_tag(&type_decl.name, &ctor.name);
            variants.push(VariantDef {
                tag,
                name: ctor.name.clone(),
                fields: ctor.fields.iter().map(type_to_core_type).collect(),
            });
        }
        ctx.type_defs.push(TypeDef {
            name: type_decl.name.clone(),
            params: type_decl.params.clone(),
            variants,
        });
    }

    // Register built-in functions in environment
    register_builtins(&mut ctx);

    // Second pass: forward-declare all binding names
    // This allows functions to reference each other and allows instance methods
    // to reference prelude functions like `map`
    let mut binding_vars: Vec<(String, VarId)> = Vec::new();
    for binding in &program.bindings {
        let var = ctx.fresh_named(&binding.name);
        ctx.bind(&binding.name, var);
        binding_vars.push((binding.name.clone(), var));
    }

    // Third pass: lower trait instance methods
    // Now they can reference all prelude functions
    for instance in &program.instance_decls {
        lower_instance(&mut ctx, instance);
    }

    // Fourth pass: lower binding bodies
    let mut main_expr: Option<CoreExpr> = None;

    for (binding, (_name, var)) in program.bindings.iter().zip(binding_vars.iter()) {
        let var = *var;

        if binding.params.is_empty() {
            // Simple value binding
            let lowered = lower_texpr(&mut ctx, &binding.body, false);

            if binding.name == "main" {
                main_expr = Some(lowered);
            } else {
                ctx.fun_defs.push(FunDef {
                    name: binding.name.clone(),
                    var_id: var,
                    params: vec![],
                    param_hints: vec![],
                    param_types: vec![],
                    return_type: type_to_core_type(&binding.ty),
                    body: lowered,
                    is_tail_recursive: false,
                });
            }
        } else {
            // Function binding
            ctx.push_scope();

            let mut param_vars = Vec::new();
            let mut param_hints = Vec::new();
            let mut param_types = Vec::new();

            for param in &binding.params {
                let (pvar, hint) = lower_tpattern_bind(&mut ctx, param);
                param_vars.push(pvar);
                param_hints.push(hint.unwrap_or_else(|| "_".to_string()));
                param_types.push(type_to_core_type(&param.ty));
            }

            let body_lowered = lower_texpr(&mut ctx, &binding.body, true);
            ctx.pop_scope();

            if binding.name == "main" && param_vars.is_empty() {
                main_expr = Some(body_lowered);
            } else {
                ctx.fun_defs.push(FunDef {
                    name: binding.name.clone(),
                    var_id: var,
                    params: param_vars,
                    param_hints,
                    param_types,
                    return_type: type_to_core_type(&binding.ty),
                    body: body_lowered,
                    is_tail_recursive: false,
                });
            }
        }
    }

    // Lower main expression if present
    if let Some(main) = &program.main {
        main_expr = Some(lower_texpr(&mut ctx, main, true));
    }

    if !ctx.errors.is_empty() {
        return Err(ctx.errors);
    }

    Ok(CoreProgram {
        types: ctx.type_defs,
        functions: ctx.fun_defs,
        builtins: ctx.builtins,
        main: main_expr,
    })
}

/// Lower a typed expression to Core IR
fn lower_texpr(ctx: &mut LowerCtx, expr: &TExpr, in_tail: bool) -> CoreExpr {
    match &expr.node {
        TExprKind::Lit(lit) => CoreExpr::Lit(lower_literal(lit)),

        TExprKind::Var(name) => {
            if let Some(var) = ctx.lookup(name) {
                CoreExpr::Var(var)
            } else {
                ctx.error(format!("Unbound variable in lowering: {}", name));
                CoreExpr::Error(format!("unbound: {}", name))
            }
        }

        TExprKind::Lambda { params, body } => {
            ctx.push_scope();

            let mut param_vars = Vec::new();
            let mut param_hints = Vec::new();

            for param in params {
                let (var, hint) = lower_tpattern_bind(ctx, param);
                param_vars.push(var);
                param_hints.push(hint);
            }

            let body_lowered = lower_texpr(ctx, body, true);
            ctx.pop_scope();

            CoreExpr::Lam {
                params: param_vars,
                param_hints,
                body: Box::new(body_lowered),
            }
        }

        TExprKind::App { func, arg } => {
            // Collect all arguments for multi-arg application
            let (func_expr, args) = collect_tapp_args(func, arg);

            // Check if function is a simple variable
            let func_var = if let TExprKind::Var(name) = &func_expr.node {
                if let Some(var) = ctx.lookup(name) {
                    var
                } else {
                    ctx.error(format!("Unbound function: {}", name));
                    ctx.fresh()
                }
            } else {
                // Complex function expression - need to bind to variable
                let func_lowered = lower_texpr(ctx, func_expr, false);
                let var = ctx.fresh();
                return {
                    let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
                    for a in args.iter() {
                        let lowered = lower_texpr(ctx, a, false);
                        let v = ctx.fresh();
                        arg_lowered.push((v, lowered));
                    }
                    let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();

                    let app_expr = if in_tail {
                        CoreExpr::TailApp { func: var, args: arg_vars }
                    } else {
                        CoreExpr::App { func: var, args: arg_vars }
                    };

                    let mut result = app_expr;
                    for (v, lowered) in arg_lowered.into_iter().rev() {
                        result = CoreExpr::LetExpr {
                            name: v,
                            name_hint: Some("arg".into()),
                            value: Box::new(lowered),
                            body: Box::new(result),
                        };
                    }

                    CoreExpr::LetExpr {
                        name: var,
                        name_hint: Some("func".into()),
                        value: Box::new(func_lowered),
                        body: Box::new(result),
                    }
                };
            };

            // Function is a simple variable reference
            let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
            for a in args.iter() {
                let lowered = lower_texpr(ctx, a, false);
                let var = ctx.fresh();
                arg_lowered.push((var, lowered));
            }

            let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();

            let app_expr = if in_tail {
                CoreExpr::TailApp { func: func_var, args: arg_vars }
            } else {
                CoreExpr::App { func: func_var, args: arg_vars }
            };

            let mut result = app_expr;
            for (var, lowered) in arg_lowered.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: var,
                    name_hint: Some("arg".into()),
                    value: Box::new(lowered),
                    body: Box::new(result),
                };
            }

            result
        }

        TExprKind::Let { pattern, value, body } => {
            let value_lowered = lower_texpr(ctx, value, false);

            match &pattern.node {
                TPatternKind::Var(name) => {
                    let var = ctx.fresh_named(name);
                    ctx.bind(name, var);

                    let body_lowered = if let Some(body) = body {
                        lower_texpr(ctx, body, in_tail)
                    } else {
                        CoreExpr::Lit(CoreLit::Unit)
                    };

                    CoreExpr::LetExpr {
                        name: var,
                        name_hint: Some(name.clone()),
                        value: Box::new(value_lowered),
                        body: Box::new(body_lowered),
                    }
                }
                _ => {
                    // Complex pattern - use case expression with nested pattern support
                    let scrutinee_var = ctx.fresh();
                    let (binders, hints, tag, tag_name, nested_patterns) =
                        lower_tpattern_case_with_nested(ctx, pattern);

                    // Build case wrappers first - this binds all pattern variables
                    let case_wrappers = build_nested_case_wrappers(ctx, &nested_patterns);

                    // Now lower the body - all pattern variables are bound
                    let body_lowered = if let Some(body) = body {
                        lower_texpr(ctx, body, in_tail)
                    } else {
                        CoreExpr::Lit(CoreLit::Unit)
                    };

                    // Apply case wrappers to the body
                    let wrapped_body = apply_case_wrappers(case_wrappers, body_lowered);

                    CoreExpr::LetExpr {
                        name: scrutinee_var,
                        name_hint: Some("pat".into()),
                        value: Box::new(value_lowered),
                        body: Box::new(CoreExpr::Case {
                            scrutinee: scrutinee_var,
                            alts: vec![Alt {
                                tag,
                                tag_name,
                                binders,
                                binder_hints: hints,
                                body: wrapped_body,
                            }],
                            default: Some(Box::new(CoreExpr::Error(
                                "pattern match failed".to_string(),
                            ))),
                        }),
                    }
                }
            }
        }

        TExprKind::LetRec { bindings, body } => {
            // First, bind all names
            let mut vars = Vec::new();
            for binding in bindings {
                let var = ctx.fresh_named(&binding.name);
                ctx.bind(&binding.name, var);
                vars.push((var, binding.name.clone()));
            }

            // Lower each binding
            let mut rec_bindings = Vec::new();
            for (binding, (var, name)) in bindings.iter().zip(vars.iter()) {
                ctx.push_scope();
                let mut param_vars = Vec::new();
                let mut param_hints = Vec::new();
                for param in &binding.params {
                    let (pvar, hint) = lower_tpattern_bind(ctx, param);
                    param_vars.push(pvar);
                    param_hints.push(hint);
                }
                let body_lowered = lower_texpr(ctx, &binding.body, true);
                ctx.pop_scope();

                rec_bindings.push(RecBinding {
                    name: *var,
                    name_hint: Some(name.clone()),
                    params: param_vars,
                    param_hints,
                    body: body_lowered,
                });
            }

            let body_lowered = if let Some(body) = body {
                lower_texpr(ctx, body, in_tail)
            } else {
                CoreExpr::Lit(CoreLit::Unit)
            };

            CoreExpr::LetRecMutual {
                bindings: rec_bindings,
                body: Box::new(body_lowered),
            }
        }

        TExprKind::If { cond, then_branch, else_branch } => {
            let cond_lowered = lower_texpr(ctx, cond, false);
            let cond_var = ctx.fresh();

            let then_lowered = lower_texpr(ctx, then_branch, in_tail);
            let else_lowered = lower_texpr(ctx, else_branch, in_tail);

            CoreExpr::LetExpr {
                name: cond_var,
                name_hint: Some("cond".into()),
                value: Box::new(cond_lowered),
                body: Box::new(CoreExpr::If {
                    cond: cond_var,
                    then_branch: Box::new(then_lowered),
                    else_branch: Box::new(else_lowered),
                }),
            }
        }

        TExprKind::Match { scrutinee, arms } => {
            let scrutinee_lowered = lower_texpr(ctx, scrutinee, false);
            let scrutinee_var = ctx.fresh();

            let alts = lower_tmatch_arms(ctx, arms, in_tail);

            CoreExpr::LetExpr {
                name: scrutinee_var,
                name_hint: Some("match".into()),
                value: Box::new(scrutinee_lowered),
                body: Box::new(CoreExpr::Case {
                    scrutinee: scrutinee_var,
                    alts,
                    default: Some(Box::new(CoreExpr::Error("match failed".to_string()))),
                }),
            }
        }

        TExprKind::BinOp { op, left, right } => {
            if let Some(prim) = tbinop_to_prim(op) {
                let left_lowered = lower_texpr(ctx, left, false);
                let right_lowered = lower_texpr(ctx, right, false);
                let left_var = ctx.fresh();
                let right_var = ctx.fresh();

                CoreExpr::LetExpr {
                    name: left_var,
                    name_hint: Some("left".into()),
                    value: Box::new(left_lowered),
                    body: Box::new(CoreExpr::LetExpr {
                        name: right_var,
                        name_hint: Some("right".into()),
                        value: Box::new(right_lowered),
                        body: Box::new(CoreExpr::PrimOp {
                            op: prim,
                            args: vec![left_var, right_var],
                        }),
                    }),
                }
            } else {
                // Pipe/compose operators become function application
                match op {
                    TBinOp::Pipe => {
                        // x |> f  =>  f x
                        lower_texpr(
                            ctx,
                            &TExpr::new(
                                TExprKind::App {
                                    func: right.clone(),
                                    arg: left.clone(),
                                },
                                expr.ty.clone(),
                                expr.span.clone(),
                            ),
                            in_tail,
                        )
                    }
                    TBinOp::PipeLeft => {
                        // f <| x  =>  f x
                        lower_texpr(
                            ctx,
                            &TExpr::new(
                                TExprKind::App {
                                    func: left.clone(),
                                    arg: right.clone(),
                                },
                                expr.ty.clone(),
                                expr.span.clone(),
                            ),
                            in_tail,
                        )
                    }
                    TBinOp::Cons => {
                        let left_lowered = lower_texpr(ctx, left, false);
                        let right_lowered = lower_texpr(ctx, right, false);
                        let left_var = ctx.fresh();
                        let right_var = ctx.fresh();

                        CoreExpr::LetExpr {
                            name: left_var,
                            name_hint: Some("head".into()),
                            value: Box::new(left_lowered),
                            body: Box::new(CoreExpr::LetExpr {
                                name: right_var,
                                name_hint: Some("tail".into()),
                                value: Box::new(right_lowered),
                                body: Box::new(CoreExpr::PrimOp {
                                    op: PrimOp::ListCons,
                                    args: vec![left_var, right_var],
                                }),
                            }),
                        }
                    }
                    _ => {
                        ctx.error(format!("Unsupported binary operator: {:?}", op));
                        CoreExpr::Error("unsupported binop".to_string())
                    }
                }
            }
        }

        TExprKind::UnaryOp { op, operand } => {
            let operand_lowered = lower_texpr(ctx, operand, false);
            let operand_var = ctx.fresh();

            let prim = match op {
                TUnaryOp::IntNeg => PrimOp::IntNeg,
                TUnaryOp::FloatNeg => PrimOp::FloatNeg,
                TUnaryOp::BoolNot => PrimOp::BoolNot,
            };

            CoreExpr::LetExpr {
                name: operand_var,
                name_hint: Some("operand".into()),
                value: Box::new(operand_lowered),
                body: Box::new(CoreExpr::PrimOp {
                    op: prim,
                    args: vec![operand_var],
                }),
            }
        }

        TExprKind::Tuple(elems) => {
            let mut lowered: Vec<(VarId, CoreExpr)> = vec![];
            for elem in elems {
                let var = ctx.fresh();
                lowered.push((var, lower_texpr(ctx, elem, false)));
            }

            let vars: Vec<VarId> = lowered.iter().map(|(v, _)| *v).collect();
            let tuple_expr = CoreExpr::Alloc {
                tag: 0,
                type_name: Some("Tuple".into()),
                ctor_name: Some(format!("Tuple{}", elems.len())),
                fields: vars,
            };

            let mut result = tuple_expr;
            for (var, expr) in lowered.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: var,
                    name_hint: Some("elem".into()),
                    value: Box::new(expr),
                    body: Box::new(result),
                };
            }

            result
        }

        TExprKind::List(elems) => {
            let nil_tag = ctx.get_tag("List", "Nil");
            let cons_tag = ctx.get_tag("List", "Cons");

            if elems.is_empty() {
                CoreExpr::Alloc {
                    tag: nil_tag,
                    type_name: Some("List".into()),
                    ctor_name: Some("Nil".into()),
                    fields: vec![],
                }
            } else {
                let mut lowered: Vec<(VarId, CoreExpr)> = vec![];
                for elem in elems {
                    let var = ctx.fresh();
                    lowered.push((var, lower_texpr(ctx, elem, false)));
                }

                // Build list from right to left using cons
                let mut result = CoreExpr::Alloc {
                    tag: nil_tag,
                    type_name: Some("List".into()),
                    ctor_name: Some("Nil".into()),
                    fields: vec![],
                };

                for (elem_var, _) in lowered.iter().rev() {
                    let rest_var = ctx.fresh();
                    result = CoreExpr::LetExpr {
                        name: rest_var,
                        name_hint: Some("rest".into()),
                        value: Box::new(result),
                        body: Box::new(CoreExpr::Alloc {
                            tag: cons_tag,
                            type_name: Some("List".into()),
                            ctor_name: Some("Cons".into()),
                            fields: vec![*elem_var, rest_var],
                        }),
                    };
                }

                // Wrap with element bindings
                for (var, expr) in lowered.into_iter().rev() {
                    result = CoreExpr::LetExpr {
                        name: var,
                        name_hint: Some("elem".into()),
                        value: Box::new(expr),
                        body: Box::new(result),
                    };
                }

                result
            }
        }

        TExprKind::Constructor { name, args } => {
            // Get or create a tag for this constructor
            let tag = ctx.get_tag("ADT", name);

            if args.is_empty() {
                CoreExpr::Alloc {
                    tag,
                    type_name: None,
                    ctor_name: Some(name.clone()),
                    fields: vec![],
                }
            } else {
                let mut lowered: Vec<(VarId, CoreExpr)> = vec![];
                for arg in args {
                    let var = ctx.fresh();
                    lowered.push((var, lower_texpr(ctx, arg, false)));
                }

                let vars: Vec<VarId> = lowered.iter().map(|(v, _)| *v).collect();
                let alloc_expr = CoreExpr::Alloc {
                    tag,
                    type_name: None,
                    ctor_name: Some(name.clone()),
                    fields: vars,
                };

                let mut result = alloc_expr;
                for (var, expr) in lowered.into_iter().rev() {
                    result = CoreExpr::LetExpr {
                        name: var,
                        name_hint: Some("field".into()),
                        value: Box::new(expr),
                        body: Box::new(result),
                    };
                }

                result
            }
        }

        TExprKind::Seq { first, second } => {
            let first_lowered = lower_texpr(ctx, first, false);
            let second_lowered = lower_texpr(ctx, second, in_tail);
            let discard_var = ctx.fresh();

            CoreExpr::LetExpr {
                name: discard_var,
                name_hint: Some("_".into()),
                value: Box::new(first_lowered),
                body: Box::new(second_lowered),
            }
        }

        TExprKind::Record { name, fields } => {
            let mut lowered: Vec<(VarId, CoreExpr)> = vec![];
            for (_, field_expr) in fields {
                let var = ctx.fresh();
                lowered.push((var, lower_texpr(ctx, field_expr, false)));
            }

            let tag = ctx.get_tag("Record", name);
            let vars: Vec<VarId> = lowered.iter().map(|(v, _)| *v).collect();
            let record_expr = CoreExpr::Alloc {
                tag,
                type_name: Some(name.clone()),
                ctor_name: Some(name.clone()),
                fields: vars,
            };

            let mut result = record_expr;
            for (var, expr) in lowered.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: var,
                    name_hint: Some("field".into()),
                    value: Box::new(expr),
                    body: Box::new(result),
                };
            }

            result
        }

        TExprKind::FieldAccess { record, field: _ } => {
            // TODO: implement field access properly
            let record_lowered = lower_texpr(ctx, record, false);
            let record_var = ctx.fresh();

            CoreExpr::LetExpr {
                name: record_var,
                name_hint: Some("record".into()),
                value: Box::new(record_lowered),
                body: Box::new(CoreExpr::Error("field access not implemented".to_string())),
            }
        }

        TExprKind::RecordUpdate { base, updates: _ } => {
            // TODO: implement record update properly
            let base_lowered = lower_texpr(ctx, base, false);
            let base_var = ctx.fresh();

            CoreExpr::LetExpr {
                name: base_var,
                name_hint: Some("base".into()),
                value: Box::new(base_lowered),
                body: Box::new(CoreExpr::Error("record update not implemented".to_string())),
            }
        }

        TExprKind::MethodCall {
            trait_name,
            method,
            instance_ty,
            args,
        } => {
            // Generate monomorphized function name
            // Format: Trait_Type_method (e.g., Show_Int_show)
            let mangled_name = format!("{}_{}_{}", trait_name, mangle_type(instance_ty), method);

            // Look up the mangled function
            if let Some(func_var) = ctx.lookup(&mangled_name) {
                // Lower arguments
                let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
                for arg in args {
                    let var = ctx.fresh();
                    arg_lowered.push((var, lower_texpr(ctx, arg, false)));
                }

                let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();

                let app_expr = if in_tail {
                    CoreExpr::TailApp { func: func_var, args: arg_vars }
                } else {
                    CoreExpr::App { func: func_var, args: arg_vars }
                };

                let mut result = app_expr;
                for (var, expr) in arg_lowered.into_iter().rev() {
                    result = CoreExpr::LetExpr {
                        name: var,
                        name_hint: Some("arg".into()),
                        value: Box::new(expr),
                        body: Box::new(result),
                    };
                }

                result
            } else {
                // Fallback: look up the base method name
                if let Some(func_var) = ctx.lookup(method) {
                    let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
                    for arg in args {
                        let var = ctx.fresh();
                        arg_lowered.push((var, lower_texpr(ctx, arg, false)));
                    }

                    let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();
                    let app_expr = CoreExpr::App { func: func_var, args: arg_vars };

                    let mut result = app_expr;
                    for (var, expr) in arg_lowered.into_iter().rev() {
                        result = CoreExpr::LetExpr {
                            name: var,
                            name_hint: Some("arg".into()),
                            value: Box::new(expr),
                            body: Box::new(result),
                        };
                    }

                    result
                } else {
                    ctx.error(format!(
                        "Method {} not found for type {:?}",
                        method, instance_ty
                    ));
                    CoreExpr::Error(format!("method not found: {}", method))
                }
            }
        }

        TExprKind::Perform { effect, op, args } => {
            let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
            for arg in args {
                let var = ctx.fresh();
                arg_lowered.push((var, lower_texpr(ctx, arg, false)));
            }

            let perform = CoreExpr::Perform {
                effect: 0, // Effect ID (placeholder)
                effect_name: Some(effect.clone()),
                op: 0, // Op ID (placeholder)
                op_name: Some(op.clone()),
                args: arg_lowered.iter().map(|(v, _)| *v).collect(),
            };

            let mut result = perform;
            for (var, expr) in arg_lowered.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: var,
                    name_hint: Some("arg".into()),
                    value: Box::new(expr),
                    body: Box::new(result),
                };
            }

            result
        }

        TExprKind::Handle { body, handler } => {
            let body_lowered = lower_texpr(ctx, body, false);

            let ops: Vec<OpHandler> = handler
                .op_clauses
                .iter()
                .map(|clause| {
                    ctx.push_scope();

                    let mut param_vars = Vec::new();
                    for param in &clause.params {
                        let (var, _) = lower_tpattern_bind(ctx, param);
                        param_vars.push(var);
                    }

                    let cont_var = ctx.fresh_named(&clause.continuation);
                    ctx.bind(&clause.continuation, cont_var);

                    let handler_body = lower_texpr(ctx, &clause.body, true);
                    ctx.pop_scope();

                    OpHandler {
                        op: 0, // Op ID placeholder
                        op_name: Some(clause.op_name.clone()),
                        params: param_vars,
                        cont: cont_var,
                        body: handler_body,
                    }
                })
                .collect();

            // Handle return clause
            let (return_var, return_body) = if let Some(clause) = &handler.return_clause {
                ctx.push_scope();
                let (var, _) = lower_tpattern_bind(ctx, &clause.pattern);
                let body = lower_texpr(ctx, &clause.body, true);
                ctx.pop_scope();
                (var, Box::new(body))
            } else {
                // Default return: just return the value
                let var = ctx.fresh();
                (var, Box::new(CoreExpr::Var(var)))
            };

            CoreExpr::Handle {
                body: Box::new(body_lowered),
                handler: Handler {
                    effect: 0, // Effect ID placeholder
                    effect_name: handler.effect.clone(),
                    return_var,
                    return_body,
                    ops,
                },
            }
        }

        TExprKind::Error(msg) => CoreExpr::Error(msg.clone()),
    }
}

/// Collect arguments from nested applications (for TAST)
fn collect_tapp_args<'a>(func: &'a TExpr, arg: &'a TExpr) -> (&'a TExpr, Vec<&'a TExpr>) {
    let mut args = vec![arg];
    let mut current = func;

    while let TExprKind::App {
        func: inner_func,
        arg: inner_arg,
    } = &current.node
    {
        args.push(inner_arg.as_ref());
        current = inner_func.as_ref();
    }

    args.reverse();
    (current, args)
}

/// Bind a typed pattern and return the variable
/// This recursively binds all variables in complex patterns
fn lower_tpattern_bind(ctx: &mut LowerCtx, pattern: &TPattern) -> (VarId, Option<String>) {
    match &pattern.node {
        TPatternKind::Var(name) => {
            let var = ctx.fresh_named(name);
            ctx.bind(name, var);
            (var, Some(name.clone()))
        }
        TPatternKind::Wildcard => {
            let var = ctx.fresh();
            (var, None)
        }
        TPatternKind::Cons { head, tail } => {
            // Recursively bind variables in head and tail
            let _ = lower_tpattern_bind(ctx, head);
            let _ = lower_tpattern_bind(ctx, tail);
            let var = ctx.fresh();
            (var, None)
        }
        TPatternKind::Tuple(elems) => {
            // Recursively bind variables in all tuple elements
            for elem in elems {
                let _ = lower_tpattern_bind(ctx, elem);
            }
            let var = ctx.fresh();
            (var, None)
        }
        TPatternKind::List(elems) => {
            // Recursively bind variables in all list elements
            for elem in elems {
                let _ = lower_tpattern_bind(ctx, elem);
            }
            let var = ctx.fresh();
            (var, None)
        }
        TPatternKind::Constructor { args, .. } => {
            // Recursively bind variables in constructor arguments
            for arg in args {
                let _ = lower_tpattern_bind(ctx, arg);
            }
            let var = ctx.fresh();
            (var, None)
        }
        TPatternKind::Record { fields, .. } => {
            // Recursively bind variables in record fields
            for (_field_name, pat) in fields {
                if let Some(inner_pat) = pat {
                    let _ = lower_tpattern_bind(ctx, inner_pat);
                }
            }
            let var = ctx.fresh();
            (var, None)
        }
        TPatternKind::Lit(_) => {
            let var = ctx.fresh();
            (var, None)
        }
    }
}

/// Lower typed match arms to Core IR alternatives
fn lower_tmatch_arms(ctx: &mut LowerCtx, arms: &[TMatchArm], in_tail: bool) -> Vec<Alt> {
    arms.iter()
        .map(|arm| {
            ctx.push_scope();
            let (binders, hints, tag, tag_name, nested_patterns) =
                lower_tpattern_case_with_nested(ctx, &arm.pattern);

            // Build the nested case structure first - this binds all pattern variables
            // Returns a function that will wrap the body in nested cases
            let case_wrappers = build_nested_case_wrappers(ctx, &nested_patterns);

            // Now lower the body - all pattern variables are bound
            let body = lower_texpr(ctx, &arm.body, in_tail);

            // Apply the case wrappers to the body (innermost first)
            let wrapped_body = apply_case_wrappers(case_wrappers, body);

            ctx.pop_scope();

            Alt {
                tag,
                tag_name,
                binders,
                binder_hints: hints,
                body: wrapped_body,
            }
        })
        .collect()
}

/// A case wrapper that will wrap a body expression in a case expression
struct CaseWrapper {
    scrutinee: VarId,
    tag: Tag,
    tag_name: Option<String>,
    binders: Vec<VarId>,
}

/// Build case wrappers for nested patterns and bind all pattern variables
fn build_nested_case_wrappers(
    ctx: &mut LowerCtx,
    nested_patterns: &[(VarId, TPattern)],
) -> Vec<CaseWrapper> {
    let mut wrappers = Vec::new();

    for (scrutinee, pattern) in nested_patterns {
        build_case_wrapper_for_pattern(ctx, *scrutinee, pattern, &mut wrappers);
    }

    wrappers
}

/// Build case wrappers for a single pattern, recursively handling nested patterns
fn build_case_wrapper_for_pattern(
    ctx: &mut LowerCtx,
    scrutinee: VarId,
    pattern: &TPattern,
    wrappers: &mut Vec<CaseWrapper>,
) {
    match &pattern.node {
        TPatternKind::Var(name) => {
            // Just bind the variable to the scrutinee (no case wrapper needed)
            ctx.bind(name, scrutinee);
        }

        TPatternKind::Wildcard => {
            // Nothing to bind or wrap
        }

        TPatternKind::Lit(_) => {
            // TODO: Handle literal patterns with equality check
        }

        TPatternKind::Cons { head, tail } => {
            let head_var = ctx.fresh();
            let tail_var = ctx.fresh();
            let tag = ctx.tag_table.get_or_create("List", "Cons");

            wrappers.push(CaseWrapper {
                scrutinee,
                tag,
                tag_name: Some("Cons".to_string()),
                binders: vec![head_var, tail_var],
            });

            // Recursively handle sub-patterns
            match &head.node {
                TPatternKind::Var(name) => ctx.bind(name, head_var),
                TPatternKind::Wildcard => {}
                _ => build_case_wrapper_for_pattern(ctx, head_var, head, wrappers),
            }

            match &tail.node {
                TPatternKind::Var(name) => ctx.bind(name, tail_var),
                TPatternKind::Wildcard => {}
                _ => build_case_wrapper_for_pattern(ctx, tail_var, tail, wrappers),
            }
        }

        TPatternKind::Tuple(elems) => {
            let mut binder_vars = Vec::new();
            for _ in elems {
                binder_vars.push(ctx.fresh());
            }

            wrappers.push(CaseWrapper {
                scrutinee,
                tag: 0,
                tag_name: Some(format!("Tuple{}", elems.len())),
                binders: binder_vars.clone(),
            });

            for (var, elem) in binder_vars.into_iter().zip(elems.iter()) {
                match &elem.node {
                    TPatternKind::Var(name) => ctx.bind(name, var),
                    TPatternKind::Wildcard => {}
                    _ => build_case_wrapper_for_pattern(ctx, var, elem, wrappers),
                }
            }
        }

        TPatternKind::List(elems) if elems.is_empty() => {
            let tag = ctx.tag_table.get_or_create("List", "Nil");
            wrappers.push(CaseWrapper {
                scrutinee,
                tag,
                tag_name: Some("Nil".to_string()),
                binders: vec![],
            });
        }

        TPatternKind::List(elems) => {
            // Non-empty list pattern - compile as nested Cons
            // [a, b, c] becomes Cons(a, Cons(b, Cons(c, Nil)))
            let mut current_scrutinee = scrutinee;
            let cons_tag = ctx.tag_table.get_or_create("List", "Cons");

            for (i, elem) in elems.iter().enumerate() {
                let head_var = ctx.fresh();
                let tail_var = ctx.fresh();

                wrappers.push(CaseWrapper {
                    scrutinee: current_scrutinee,
                    tag: cons_tag,
                    tag_name: Some("Cons".to_string()),
                    binders: vec![head_var, tail_var],
                });

                // Bind or recurse for head
                match &elem.node {
                    TPatternKind::Var(name) => ctx.bind(name, head_var),
                    TPatternKind::Wildcard => {}
                    _ => build_case_wrapper_for_pattern(ctx, head_var, elem, wrappers),
                }

                // Continue with tail for remaining elements
                if i < elems.len() - 1 {
                    current_scrutinee = tail_var;
                } else {
                    // Last element - tail should be Nil
                    let nil_tag = ctx.tag_table.get_or_create("List", "Nil");
                    wrappers.push(CaseWrapper {
                        scrutinee: tail_var,
                        tag: nil_tag,
                        tag_name: Some("Nil".to_string()),
                        binders: vec![],
                    });
                }
            }
        }

        TPatternKind::Constructor { name, args } => {
            let tag = ctx.get_tag("ADT", name);
            let mut binder_vars = Vec::new();
            for _ in args {
                binder_vars.push(ctx.fresh());
            }

            wrappers.push(CaseWrapper {
                scrutinee,
                tag,
                tag_name: Some(name.clone()),
                binders: binder_vars.clone(),
            });

            for (var, arg) in binder_vars.into_iter().zip(args.iter()) {
                match &arg.node {
                    TPatternKind::Var(n) => ctx.bind(n, var),
                    TPatternKind::Wildcard => {}
                    _ => build_case_wrapper_for_pattern(ctx, var, arg, wrappers),
                }
            }
        }

        TPatternKind::Record { name, fields } => {
            let tag = ctx.get_tag("Record", name);
            let mut binder_vars = Vec::new();
            for _ in fields {
                binder_vars.push(ctx.fresh());
            }

            wrappers.push(CaseWrapper {
                scrutinee,
                tag,
                tag_name: Some(name.clone()),
                binders: binder_vars.clone(),
            });

            for (var, (field_name, pat)) in binder_vars.into_iter().zip(fields.iter()) {
                if let Some(inner_pat) = pat {
                    match &inner_pat.node {
                        TPatternKind::Var(n) => ctx.bind(n, var),
                        TPatternKind::Wildcard => {}
                        _ => build_case_wrapper_for_pattern(ctx, var, inner_pat, wrappers),
                    }
                } else {
                    ctx.bind(field_name, var);
                }
            }
        }
    }
}

/// Apply case wrappers to a body expression, building nested case expressions
fn apply_case_wrappers(wrappers: Vec<CaseWrapper>, body: CoreExpr) -> CoreExpr {
    // Apply wrappers in reverse order (innermost first)
    let mut result = body;
    for wrapper in wrappers.into_iter().rev() {
        result = CoreExpr::Case {
            scrutinee: wrapper.scrutinee,
            alts: vec![Alt {
                tag: wrapper.tag,
                tag_name: wrapper.tag_name,
                binders: wrapper.binders,
                binder_hints: vec![],
                body: result,
            }],
            default: Some(Box::new(CoreExpr::Error("pattern match failed".to_string()))),
        };
    }
    result
}

/// Check if a pattern is trivial (just binds a variable or is wildcard)
fn is_trivial_pattern(pattern: &TPattern) -> bool {
    matches!(
        &pattern.node,
        TPatternKind::Var(_) | TPatternKind::Wildcard
    )
}

/// Lower a pattern for case expression, returning nested patterns that need further matching
fn lower_tpattern_case_with_nested(
    ctx: &mut LowerCtx,
    pattern: &TPattern,
) -> (Vec<VarId>, Vec<Option<String>>, Tag, Option<String>, Vec<(VarId, TPattern)>) {
    match &pattern.node {
        TPatternKind::Wildcard => (vec![], vec![], 0, None, vec![]),

        TPatternKind::Var(name) => {
            let var = ctx.fresh_named(name);
            ctx.bind(name, var);
            (vec![var], vec![Some(name.clone())], 0, None, vec![])
        }

        TPatternKind::Lit(_) => {
            // Literals need special handling
            (vec![], vec![], 0, None, vec![])
        }

        TPatternKind::Tuple(elems) => {
            let mut binders = Vec::new();
            let mut hints = Vec::new();
            let mut nested = Vec::new();

            for elem in elems {
                let var = ctx.fresh();
                binders.push(var);
                hints.push(None);

                if !is_trivial_pattern(elem) {
                    // This element needs further pattern matching
                    nested.push((var, elem.clone()));
                } else {
                    // Trivial pattern - just bind the variable
                    if let TPatternKind::Var(name) = &elem.node {
                        ctx.bind(name, var);
                    }
                }
            }
            (binders, hints, 0, Some(format!("Tuple{}", elems.len())), nested)
        }

        TPatternKind::List(elems) => {
            let mut binders = Vec::new();
            let mut hints = Vec::new();
            let mut nested = Vec::new();

            for elem in elems {
                let var = ctx.fresh();
                binders.push(var);
                hints.push(None);

                if !is_trivial_pattern(elem) {
                    nested.push((var, elem.clone()));
                } else if let TPatternKind::Var(name) = &elem.node {
                    ctx.bind(name, var);
                }
            }

            let (tag, tag_name) = if elems.is_empty() {
                (ctx.tag_table.get_or_create("List", "Nil"), Some("Nil".to_string()))
            } else {
                (ctx.tag_table.get_or_create("List", "Cons"), Some("Cons".to_string()))
            };
            (binders, hints, tag, tag_name, nested)
        }

        TPatternKind::Cons { head, tail } => {
            let head_var = ctx.fresh();
            let tail_var = ctx.fresh();
            let tag = ctx.tag_table.get_or_create("List", "Cons");

            let mut nested = Vec::new();

            // Handle head
            if !is_trivial_pattern(head) {
                nested.push((head_var, (**head).clone()));
            } else if let TPatternKind::Var(name) = &head.node {
                ctx.bind(name, head_var);
            }

            // Handle tail
            if !is_trivial_pattern(tail) {
                nested.push((tail_var, (**tail).clone()));
            } else if let TPatternKind::Var(name) = &tail.node {
                ctx.bind(name, tail_var);
            }

            (
                vec![head_var, tail_var],
                vec![None, None],
                tag,
                Some("Cons".to_string()),
                nested,
            )
        }

        TPatternKind::Constructor { name, args } => {
            let mut binders = Vec::new();
            let mut hints = Vec::new();
            let mut nested = Vec::new();

            for arg in args {
                let var = ctx.fresh();
                binders.push(var);
                hints.push(None);

                if !is_trivial_pattern(arg) {
                    nested.push((var, arg.clone()));
                } else if let TPatternKind::Var(n) = &arg.node {
                    ctx.bind(n, var);
                }
            }

            let tag = ctx.get_tag("ADT", name);
            (binders, hints, tag, Some(name.clone()), nested)
        }

        TPatternKind::Record { name, fields } => {
            let mut binders = Vec::new();
            let mut hints = Vec::new();
            let mut nested = Vec::new();

            for (field_name, pat) in fields {
                let var = ctx.fresh();
                binders.push(var);
                hints.push(Some(field_name.clone()));

                if let Some(inner_pat) = pat {
                    if !is_trivial_pattern(inner_pat) {
                        nested.push((var, inner_pat.clone()));
                    } else if let TPatternKind::Var(n) = &inner_pat.node {
                        ctx.bind(n, var);
                    }
                } else {
                    // No pattern means bind the field name
                    ctx.bind(field_name, var);
                }
            }

            let tag = ctx.get_tag("Record", name);
            (binders, hints, tag, Some(name.clone()), nested)
        }
    }
}

/// Convert typed binary operator to primitive operation
fn tbinop_to_prim(op: &TBinOp) -> Option<PrimOp> {
    match op {
        TBinOp::IntAdd => Some(PrimOp::IntAdd),
        TBinOp::IntSub => Some(PrimOp::IntSub),
        TBinOp::IntMul => Some(PrimOp::IntMul),
        TBinOp::IntDiv => Some(PrimOp::IntDiv),
        TBinOp::IntMod => Some(PrimOp::IntMod),
        TBinOp::FloatAdd => Some(PrimOp::FloatAdd),
        TBinOp::FloatSub => Some(PrimOp::FloatSub),
        TBinOp::FloatMul => Some(PrimOp::FloatMul),
        TBinOp::FloatDiv => Some(PrimOp::FloatDiv),
        TBinOp::IntEq => Some(PrimOp::IntEq),
        TBinOp::IntNe => Some(PrimOp::IntNe),
        TBinOp::IntLt => Some(PrimOp::IntLt),
        TBinOp::IntLe => Some(PrimOp::IntLe),
        TBinOp::IntGt => Some(PrimOp::IntGt),
        TBinOp::IntGe => Some(PrimOp::IntGe),
        TBinOp::FloatEq => Some(PrimOp::FloatEq),
        TBinOp::FloatNe => Some(PrimOp::FloatNe),
        TBinOp::FloatLt => Some(PrimOp::FloatLt),
        TBinOp::FloatLe => Some(PrimOp::FloatLe),
        TBinOp::FloatGt => Some(PrimOp::FloatGt),
        TBinOp::FloatGe => Some(PrimOp::FloatGe),
        TBinOp::StringEq => Some(PrimOp::StringEq),
        TBinOp::StringNe => None, // No direct PrimOp, needs negation of StringEq
        TBinOp::StringConcat => Some(PrimOp::StringConcat),
        TBinOp::BoolAnd => Some(PrimOp::BoolAnd),
        TBinOp::BoolOr => Some(PrimOp::BoolOr),
        TBinOp::Cons => Some(PrimOp::ListCons),
        _ => None, // Pipe, Compose handled separately
    }
}

/// Lower a trait instance to function definitions
fn lower_instance(ctx: &mut LowerCtx, instance: &TInstanceDecl) {
    for method in &instance.methods {
        lower_method(ctx, method, &instance.instance_ty);
    }
}

/// Lower a trait method implementation to a function definition
fn lower_method(ctx: &mut LowerCtx, method: &TMethodImpl, _instance_ty: &Type) {
    // Create a fresh variable for this method
    let var = ctx.fresh_named(&method.mangled_name);
    ctx.bind(&method.mangled_name, var);

    // Also bind the short name for lookups
    ctx.bind(&method.name, var);

    ctx.push_scope();

    // Lower parameters
    let mut param_vars = Vec::new();
    let mut param_hints = Vec::new();
    let mut param_types = Vec::new();

    for param in &method.params {
        let (pvar, hint) = lower_tpattern_bind(ctx, param);
        param_vars.push(pvar);
        param_hints.push(hint.unwrap_or_else(|| "_".to_string()));
        param_types.push(type_to_core_type(&param.ty));
    }

    // Lower body
    let body_lowered = lower_texpr(ctx, &method.body, true);
    ctx.pop_scope();

    // Get return type from the method body
    let return_type = type_to_core_type(&method.body.ty);

    ctx.fun_defs.push(FunDef {
        name: method.mangled_name.clone(),
        var_id: var,
        params: param_vars,
        param_hints,
        param_types,
        return_type,
        body: body_lowered,
        is_tail_recursive: false,
    });
}

/// Mangle a type into a string for function naming
/// Must match elaborate.rs mangle_type for consistency
fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::Int => "Int".to_string(),
        Type::Float => "Float".to_string(),
        Type::Bool => "Bool".to_string(),
        Type::Char => "Char".to_string(),
        Type::String => "String".to_string(),
        Type::Unit => "Unit".to_string(),
        Type::Bytes => "Bytes".to_string(),
        Type::Tuple(elems) => {
            let parts: Vec<String> = elems.iter().map(mangle_type).collect();
            format!("Tuple_{}", parts.join("_"))
        }
        Type::Constructor { name, args } => {
            if args.is_empty() {
                name.clone()
            } else {
                let parts: Vec<String> = args.iter().map(mangle_type).collect();
                format!("{}_{}", name, parts.join("_"))
            }
        }
        Type::Arrow { .. } => "Fn".to_string(),
        Type::Var(id) => format!("Var{}", id),
        Type::Generic(id) => format!("Gen{}", id),
        Type::Channel(_) => "Channel".to_string(),
        Type::Dict(_) => "Dict".to_string(),
        Type::Set => "Set".to_string(),
        Type::Pid => "Pid".to_string(),
        Type::Fiber(_) => "Fiber".to_string(),
        Type::FileHandle => "FileHandle".to_string(),
        Type::TcpSocket => "TcpSocket".to_string(),
        Type::TcpListener => "TcpListener".to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser};

    fn parse_and_lower(source: &str) -> Result<CoreProgram, Vec<String>> {
        let tokens = Lexer::new(source).tokenize().unwrap();
        let program = Parser::new(tokens).parse_program().unwrap();
        lower_program(&program)
    }

    #[test]
    fn test_lower_literal() {
        let result = parse_and_lower("let x = 42");
        assert!(result.is_ok());
        let prog = result.unwrap();
        assert!(!prog.functions.is_empty() || prog.main.is_some());
    }

    #[test]
    fn test_lower_binop() {
        let result = parse_and_lower("let x = 1 + 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_lower_function() {
        let result = parse_and_lower("let rec f x = x + 1");
        assert!(result.is_ok());
        let prog = result.unwrap();
        assert!(!prog.functions.is_empty());
    }

    #[test]
    fn test_lower_if() {
        let result = parse_and_lower("let x = if true then 1 else 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_lower_lambda() {
        let result = parse_and_lower("let f = fun x -> x + 1");
        assert!(result.is_ok());
    }
}
