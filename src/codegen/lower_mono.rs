//! MonoProgram to Core IR lowering
//!
//! Transforms the monomorphized IR into A-Normal Form Core IR suitable for
//! C code generation. At this point, all types are concrete.
//!
//! Key transformations:
//! - ANF conversion: all intermediate values are named
//! - Pattern matching desugaring: patterns become case expressions
//! - All functions are concrete - no polymorphism handling needed

use std::collections::{BTreeMap, HashMap};

use crate::ast::Literal;
use crate::codegen::core_ir::{
    Alt, CoreExpr, CoreLit, CoreProgram, CoreType, EffectId, FunDef, Handler, OpHandler, OpId,
    PrimOp, RecBinding, Tag, TagTable, TypeDef, VarGen, VarId, VariantDef,
};
use crate::mono::{
    MonoExpr, MonoExprKind, MonoFn, MonoFnId, MonoHandler, MonoMatchArm, MonoPattern,
    MonoPatternKind, MonoProgram, MonoRecBinding, MonoType,
};
use crate::tast::{TBinOp, TUnaryOp};

// ============================================================================
// Lowering Context
// ============================================================================

/// Context for lowering MonoProgram to Core IR
pub struct LowerMonoCtx {
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
    /// Effect name to ID mapping (BTreeMap for deterministic ordering)
    effect_ids: BTreeMap<String, EffectId>,
    /// Next effect ID to assign
    next_effect_id: EffectId,
    /// Operation name to ID mapping per effect (effect_name -> (op_name -> OpId))
    /// Op IDs are assigned per-effect starting from 0, so each effect's ops are 0, 1, 2...
    op_ids: BTreeMap<String, BTreeMap<String, OpId>>,
}

impl LowerMonoCtx {
    pub fn new() -> Self {
        let mut tag_table = TagTable::new();
        tag_table.register_builtins();

        LowerMonoCtx {
            var_gen: VarGen::new(),
            env: HashMap::new(),
            tag_table,
            type_defs: Vec::new(),
            fun_defs: Vec::new(),
            builtins: Vec::new(),
            errors: Vec::new(),
            scope_stack: Vec::new(),
            effect_ids: BTreeMap::new(),
            next_effect_id: 0,
            op_ids: BTreeMap::new(),
        }
    }

    /// Get or create an effect ID for the given effect name
    fn get_effect_id(&mut self, effect: &str) -> EffectId {
        if let Some(&id) = self.effect_ids.get(effect) {
            id
        } else {
            let id = self.next_effect_id;
            self.next_effect_id += 1;
            self.effect_ids.insert(effect.to_string(), id);
            id
        }
    }

    /// Get or create an op ID for the given effect and operation name.
    /// Op IDs are assigned per-effect starting from 0, ensuring each effect's
    /// operations are densely indexed for the runtime op_fns array.
    fn get_op_id(&mut self, effect: &str, op: &str) -> OpId {
        let effect_ops = self.op_ids.entry(effect.to_string()).or_insert_with(BTreeMap::new);
        if let Some(&id) = effect_ops.get(op) {
            id
        } else {
            let id = effect_ops.len() as OpId;  // Next ID is the current count
            effect_ops.insert(op.to_string(), id);
            id
        }
    }

    /// Generate a fresh variable
    fn fresh(&mut self) -> VarId {
        self.var_gen.fresh()
    }

    /// Generate a fresh variable with a name hint
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
    fn error(&mut self, msg: String) {
        self.errors.push(msg);
    }

    /// Get a tag for a constructor
    fn get_tag(&mut self, type_name: &str, ctor_name: &str) -> Tag {
        self.tag_table.get_or_create(type_name, ctor_name)
    }
}

// ============================================================================
// Type Conversion
// ============================================================================

/// Convert MonoType to CoreType
fn mono_type_to_core(ty: &MonoType) -> CoreType {
    match ty {
        MonoType::Int => CoreType::Int,
        MonoType::Float => CoreType::Float,
        MonoType::Bool => CoreType::Bool,
        MonoType::Char => CoreType::Char,
        MonoType::Unit => CoreType::Unit,
        MonoType::String => CoreType::String,
        MonoType::Bytes => CoreType::Box(Box::new(CoreType::Int)), // Approximate
        MonoType::Tuple(elems) => {
            CoreType::Tuple(elems.iter().map(mono_type_to_core).collect())
        }
        MonoType::Constructor { name, args } => {
            if args.is_empty() {
                // Simple sum type
                CoreType::Sum {
                    name: name.clone(),
                    variants: vec![],
                }
            } else {
                // Parameterized type - treat as boxed
                CoreType::Box(Box::new(CoreType::Sum {
                    name: name.clone(),
                    variants: vec![],
                }))
            }
        }
        MonoType::Function { params, ret } => CoreType::Fun {
            params: params.iter().map(mono_type_to_core).collect(),
            ret: Box::new(mono_type_to_core(ret)),
        },
    }
}

// ============================================================================
// Main Lowering Function
// ============================================================================

/// Lower a MonoProgram to Core IR
pub fn lower_mono(program: &MonoProgram) -> Result<CoreProgram, Vec<String>> {
    let mut ctx = LowerMonoCtx::new();

    // Register built-in functions
    register_builtins(&mut ctx);

    // Convert type definitions
    for type_def in &program.type_defs {
        let mut variants = Vec::new();
        for variant in &type_def.variants {
            let tag = ctx.get_tag(&type_def.name, &variant.name);
            variants.push(VariantDef {
                tag,
                name: variant.name.clone(),
                fields: variant.fields.iter().map(mono_type_to_core).collect(),
            });
        }
        ctx.type_defs.push(TypeDef {
            name: type_def.name.clone(),
            params: vec![], // No type params after monomorphization
            variants,
        });
    }

    // Forward-declare all functions
    let mut fn_vars: HashMap<MonoFnId, VarId> = HashMap::new();
    for (fn_id, _) in &program.functions {
        let var = ctx.fresh_named(&fn_id.mangled_name());
        ctx.bind(&fn_id.mangled_name(), var);
        fn_vars.insert(fn_id.clone(), var);
    }

    // Lower all functions in deterministic order
    // (HashMap iteration order is non-deterministic, which can cause effect IDs to vary)
    let mut sorted_fn_ids: Vec<_> = program.functions.keys().collect();
    sorted_fn_ids.sort_by_key(|id| id.mangled_name());

    for fn_id in sorted_fn_ids {
        let mono_fn = &program.functions[fn_id];
        let var = fn_vars[fn_id];
        lower_function(&mut ctx, var, mono_fn);
    }

    // Lower main expression
    let main = program.main.as_ref().map(|expr| lower_expr(&mut ctx, expr, true));

    if !ctx.errors.is_empty() {
        return Err(ctx.errors);
    }

    Ok(CoreProgram {
        types: ctx.type_defs,
        functions: ctx.fun_defs,
        builtins: ctx.builtins,
        main,
    })
}

/// Register built-in functions
fn register_builtins(ctx: &mut LowerMonoCtx) {
    let builtins = [
        // I/O
        "io_print",
        "io_read_line",
        // Integers
        "int_to_string",
        // Strings
        "string_length",
        "string_concat",
        "string_chars",
        "string_to_chars",
        "chars_to_string",
        "string_index_of",
        "string_substring",
        "string_split",
        "string_join",
        "string_char_at",
        "string_repeat",
        "string_to_lower",
        "string_to_upper",
        "string_trim",
        "string_trim_start",
        "string_trim_end",
        "string_reverse",
        "string_is_empty",
        "string_to_bytes",
        // Bytes
        "bytes_to_string",
        "bytes_length",
        "bytes_slice",
        "bytes_concat",
        // Characters
        "char_to_string",
        "char_to_int",
        "char_to_lower",
        "char_to_upper",
        "char_is_whitespace",
        "char_is_digit",
        "char_is_alpha",
        "char_is_alphanumeric",
        "char_code",
        "char_from_code",
        // Files
        "file_open",
        "file_read",
        "file_write",
        "file_close",
        // Networking
        "tcp_connect",
        "tcp_listen",
        "tcp_accept",
        // Concurrency
        "spawn",
        "Fiber.spawn",
        "Fiber.join",
        "Fiber.yield",
        "sleep_ms",
        // Dict operations
        "Dict.new",
        "Dict.insert",
        "Dict.get",
        "Dict.remove",
        "Dict.contains",
        "Dict.keys",
        "Dict.values",
        "Dict.size",
        "Dict.isEmpty",
        "Dict.toList",
        "Dict.fromList",
        "Dict.merge",
        "Dict.getOrDefault",
        // Set operations
        "Set.new",
        "Set.insert",
        "Set.contains",
        "Set.remove",
        "Set.union",
        "Set.intersect",
        "Set.size",
        "Set.toList",
        // Misc
        "error",
        "print",
        "debug",
        "get_args",
        "html_escape",
        "json_escape_string",
    ];

    for name in builtins {
        let var = ctx.fresh_named(name);
        ctx.bind(name, var);
        ctx.builtins.push((var, name.to_string()));
    }
}

/// Lower a function definition
fn lower_function(ctx: &mut LowerMonoCtx, var: VarId, mono_fn: &MonoFn) {
    ctx.push_scope();

    // Bind parameters
    let mut param_vars = Vec::new();
    let mut param_hints = Vec::new();
    let mut param_types = Vec::new();

    for param in &mono_fn.params {
        let (pvar, hint) = lower_pattern_bind(ctx, param);
        param_vars.push(pvar);
        param_hints.push(hint.unwrap_or_else(|| "_".to_string()));
        param_types.push(mono_type_to_core(&param.ty));
    }

    // Lower body
    let body = lower_expr(ctx, &mono_fn.body, true);
    let return_type = mono_type_to_core(&mono_fn.return_type);

    ctx.pop_scope();

    ctx.fun_defs.push(FunDef {
        name: mono_fn.id.mangled_name(),
        var_id: var,
        params: param_vars,
        param_hints,
        param_types,
        return_type,
        body,
        is_tail_recursive: false, // TODO: detect TCO
    });
}

// ============================================================================
// Expression Lowering
// ============================================================================

/// Lower a MonoExpr to CoreExpr
fn lower_expr(ctx: &mut LowerMonoCtx, expr: &MonoExpr, in_tail: bool) -> CoreExpr {
    match &expr.node {
        MonoExprKind::Lit(lit) => CoreExpr::Lit(lower_literal(lit)),

        MonoExprKind::Var(name) => {
            if let Some(var) = ctx.lookup(name) {
                CoreExpr::Var(var)
            } else {
                ctx.error(format!("Unbound variable in lowering: {}", name));
                CoreExpr::Error(format!("unbound: {}", name))
            }
        }

        MonoExprKind::Lambda { params, body } => {
            ctx.push_scope();

            let mut param_vars = Vec::new();
            let mut param_hints: Vec<Option<String>> = Vec::new();

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

        MonoExprKind::App { func, arg } => {
            // Collect all arguments for multi-arg application
            let (func_expr, args) = collect_app_args(func, arg);

            // Lower function - if it's just a variable, use it directly
            // This is important for CPS transform to recognize continuation calls
            let func_lowered = lower_expr(ctx, func_expr, false);
            let (func_var, func_binding) = match &func_lowered {
                CoreExpr::Var(v) => (*v, None),
                _ => {
                    let v = ctx.fresh();
                    (v, Some(func_lowered))
                }
            };

            // Lower arguments
            let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
            for a in &args {
                let lowered = lower_expr(ctx, a, false);
                let v = ctx.fresh();
                arg_lowered.push((v, lowered));
            }

            let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();

            // Build the application
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

            // Wrap in lets
            let mut result = app_expr;
            for (v, lowered) in arg_lowered.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: v,
                    name_hint: Some("arg".into()),
                    value: Box::new(lowered),
                    body: Box::new(result),
                };
            }

            // Only add func binding if the function wasn't already a variable
            if let Some(func_expr) = func_binding {
                CoreExpr::LetExpr {
                    name: func_var,
                    name_hint: Some("func".into()),
                    value: Box::new(func_expr),
                    body: Box::new(result),
                }
            } else {
                result
            }
        }

        MonoExprKind::Call { func, args } => {
            // Direct call to known monomorphized function
            if let Some(var) = ctx.lookup(&func.mangled_name()) {
                let mut arg_lowered: Vec<(VarId, CoreExpr)> = vec![];
                for a in args {
                    let lowered = lower_expr(ctx, a, false);
                    let v = ctx.fresh();
                    arg_lowered.push((v, lowered));
                }

                let arg_vars: Vec<VarId> = arg_lowered.iter().map(|(v, _)| *v).collect();

                let app_expr = if in_tail {
                    CoreExpr::TailApp {
                        func: var,
                        args: arg_vars,
                    }
                } else {
                    CoreExpr::App {
                        func: var,
                        args: arg_vars,
                    }
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
                result
            } else {
                ctx.error(format!("Unknown function: {}", func.mangled_name()));
                CoreExpr::Error(format!("unknown fn: {}", func.mangled_name()))
            }
        }

        MonoExprKind::Let { pattern, value, body } => {
            let value_lowered = lower_expr(ctx, value, false);
            let (var, _hint) = lower_pattern_bind(ctx, pattern);

            let body_lowered = if let Some(b) = body {
                lower_expr(ctx, b, in_tail)
            } else {
                CoreExpr::Var(var)
            };

            CoreExpr::LetExpr {
                name: var,
                name_hint: Some(pattern_hint(pattern)),
                value: Box::new(value_lowered),
                body: Box::new(body_lowered),
            }
        }

        MonoExprKind::LetRec { bindings, body } => {
            // Pre-bind all binding names to fresh vars so they're available
            // for both recursive calls within binding bodies AND the continuation body
            let mut prebound_vars = Vec::new();
            for b in bindings {
                let name_var = ctx.fresh_named(&b.name);
                ctx.bind(&b.name, name_var);
                prebound_vars.push(name_var);
            }

            // Lower recursive bindings (they'll use the pre-bound names)
            let rec_bindings: Vec<RecBinding> = bindings
                .iter()
                .zip(prebound_vars.iter())
                .map(|(b, &prebound)| lower_rec_binding_with_var(ctx, b, prebound))
                .collect();

            // Lower the continuation body (has access to all binding names)
            let body_lowered = if let Some(b) = body {
                lower_expr(ctx, b, in_tail)
            } else {
                CoreExpr::Lit(CoreLit::Unit)
            };

            // For now, emit as nested LetRec for single bindings
            // Multiple mutual recursion needs LetRecMutual
            if rec_bindings.len() == 1 {
                let b = &rec_bindings[0];
                CoreExpr::LetRec {
                    name: b.name,
                    name_hint: b.name_hint.clone(),
                    params: b.params.clone(),
                    param_hints: b.param_hints.clone(),
                    func_body: Box::new(b.body.clone()),
                    body: Box::new(body_lowered),
                }
            } else {
                CoreExpr::LetRecMutual {
                    bindings: rec_bindings,
                    body: Box::new(body_lowered),
                }
            }
        }

        MonoExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
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

        MonoExprKind::Match { scrutinee, arms } => {
            let scrutinee_lowered = lower_expr(ctx, scrutinee, false);
            let scrutinee_var = ctx.fresh();

            // Process arms, separating alts from default case
            // Only treat Wildcard and Var patterns as default cases
            let mut alts = Vec::new();
            let mut default_expr = None;

            for arm in arms {
                match &arm.pattern.node {
                    MonoPatternKind::Wildcard => {
                        // Wildcard pattern - use as default
                        ctx.push_scope();
                        let body = lower_expr(ctx, &arm.body, in_tail);
                        ctx.pop_scope();
                        default_expr = Some(Box::new(body));
                        break; // Only use first default case
                    }
                    MonoPatternKind::Var(name) => {
                        // Variable pattern - bind to scrutinee and use as default
                        ctx.push_scope();
                        ctx.bind(name, scrutinee_var);
                        let body = lower_expr(ctx, &arm.body, in_tail);
                        ctx.pop_scope();
                        default_expr = Some(Box::new(body));
                        break;
                    }
                    _ => {
                        // Other patterns - process normally
                        if let Some(alt) = lower_match_arm(ctx, arm, in_tail) {
                            alts.push(alt);
                        }
                        // Note: if lower_match_arm returns None for other patterns,
                        // they are silently dropped (existing behavior)
                    }
                }
            }

            CoreExpr::LetExpr {
                name: scrutinee_var,
                name_hint: Some("match".into()),
                value: Box::new(scrutinee_lowered),
                body: Box::new(CoreExpr::Case {
                    scrutinee: scrutinee_var,
                    alts,
                    default: default_expr,
                }),
            }
        }

        MonoExprKind::Tuple(elems) => {
            let mut elem_vars = Vec::new();
            let mut bindings = Vec::new();

            for elem in elems {
                let lowered = lower_expr(ctx, elem, false);
                let var = ctx.fresh();
                bindings.push((var, lowered));
                elem_vars.push(var);
            }

            let alloc = CoreExpr::Alloc {
                tag: 0, // Tuple tag
                type_name: Some("Tuple".to_string()),
                ctor_name: Some("Tuple".to_string()),
                fields: elem_vars,
            };

            wrap_in_lets(bindings, alloc)
        }

        MonoExprKind::List(elems) => {
            if elems.is_empty() {
                // Empty list
                CoreExpr::Alloc {
                    tag: ctx.get_tag("List", "Nil"),
                    type_name: Some("List".to_string()),
                    ctor_name: Some("Nil".to_string()),
                    fields: vec![],
                }
            } else {
                // Build list from end: [] then x :: xs
                // We need to build up nested lets that bind each cons cell
                let nil_var = ctx.fresh();
                let nil = CoreExpr::Alloc {
                    tag: ctx.get_tag("List", "Nil"),
                    type_name: Some("List".to_string()),
                    ctor_name: Some("Nil".to_string()),
                    fields: vec![],
                };

                // Track the current tail variable
                let mut tail_var = nil_var;

                // Build bindings in reverse order
                let mut bindings: Vec<(VarId, Option<String>, CoreExpr)> = vec![];
                bindings.push((nil_var, Some("nil".to_string()), nil));

                for elem in elems.iter().rev() {
                    let elem_lowered = lower_expr(ctx, elem, false);
                    let elem_var = ctx.fresh();
                    let cons_var = ctx.fresh();

                    let cons = CoreExpr::Alloc {
                        tag: ctx.get_tag("List", "Cons"),
                        type_name: Some("List".to_string()),
                        ctor_name: Some("Cons".to_string()),
                        fields: vec![elem_var, tail_var],
                    };

                    bindings.push((elem_var, Some("elem".to_string()), elem_lowered));
                    bindings.push((cons_var, Some("cons".to_string()), cons));

                    tail_var = cons_var;
                }

                // The final tail_var is our result
                let result_var = tail_var;

                // Build nested lets from end to start
                let mut result = CoreExpr::Var(result_var);
                for (var, hint, value) in bindings.into_iter().rev() {
                    result = CoreExpr::LetExpr {
                        name: var,
                        name_hint: hint,
                        value: Box::new(value),
                        body: Box::new(result),
                    };
                }

                result
            }
        }

        MonoExprKind::Constructor { name, args } => {
            let tag = ctx.get_tag("ADT", name); // Generic ADT

            let mut arg_vars = Vec::new();
            let mut bindings = Vec::new();

            for arg in args {
                let lowered = lower_expr(ctx, arg, false);
                let var = ctx.fresh();
                bindings.push((var, lowered));
                arg_vars.push(var);
            }

            let alloc = CoreExpr::Alloc {
                tag,
                type_name: Some("ADT".to_string()),
                ctor_name: Some(name.clone()),
                fields: arg_vars,
            };

            wrap_in_lets(bindings, alloc)
        }

        MonoExprKind::BinOp { op, left, right } => {
            let left_lowered = lower_expr(ctx, left, false);
            let right_lowered = lower_expr(ctx, right, false);
            let left_var = ctx.fresh();
            let right_var = ctx.fresh();

            let prim_op = match op {
                TBinOp::IntAdd => PrimOp::IntAdd,
                TBinOp::IntSub => PrimOp::IntSub,
                TBinOp::IntMul => PrimOp::IntMul,
                TBinOp::IntDiv => PrimOp::IntDiv,
                TBinOp::IntMod => PrimOp::IntMod,
                TBinOp::FloatAdd => PrimOp::FloatAdd,
                TBinOp::FloatSub => PrimOp::FloatSub,
                TBinOp::FloatMul => PrimOp::FloatMul,
                TBinOp::FloatDiv => PrimOp::FloatDiv,
                TBinOp::IntEq => PrimOp::IntEq,
                TBinOp::IntNe => PrimOp::IntNe,
                TBinOp::IntLt => PrimOp::IntLt,
                TBinOp::IntLe => PrimOp::IntLe,
                TBinOp::IntGt => PrimOp::IntGt,
                TBinOp::IntGe => PrimOp::IntGe,
                TBinOp::FloatEq => PrimOp::FloatEq,
                TBinOp::FloatNe => PrimOp::FloatNe,
                TBinOp::FloatLt => PrimOp::FloatLt,
                TBinOp::FloatLe => PrimOp::FloatLe,
                TBinOp::FloatGt => PrimOp::FloatGt,
                TBinOp::FloatGe => PrimOp::FloatGe,
                TBinOp::StringEq => PrimOp::StringEq,
                TBinOp::StringNe => {
                    // StringNe needs to be lowered as NOT(StringEq)
                    // For now, treat as StringEq (will need post-processing)
                    ctx.error("StringNe not yet supported - using StringEq".to_string());
                    PrimOp::StringEq
                }
                TBinOp::StringConcat => PrimOp::StringConcat,
                TBinOp::BoolAnd => PrimOp::BoolAnd,
                TBinOp::BoolOr => PrimOp::BoolOr,
                TBinOp::Cons => PrimOp::ListCons,
                _ => {
                    ctx.error(format!("Unsupported binary op: {:?}", op));
                    PrimOp::IntAdd // Fallback
                }
            };

            CoreExpr::LetExpr {
                name: left_var,
                name_hint: Some("left".into()),
                value: Box::new(left_lowered),
                body: Box::new(CoreExpr::LetExpr {
                    name: right_var,
                    name_hint: Some("right".into()),
                    value: Box::new(right_lowered),
                    body: Box::new(CoreExpr::PrimOp {
                        op: prim_op,
                        args: vec![left_var, right_var],
                    }),
                }),
            }
        }

        MonoExprKind::UnaryOp { op, operand } => {
            let operand_lowered = lower_expr(ctx, operand, false);
            let operand_var = ctx.fresh();

            let prim_op = match op {
                TUnaryOp::IntNeg => PrimOp::IntNeg,
                TUnaryOp::FloatNeg => PrimOp::FloatNeg,
                TUnaryOp::BoolNot => PrimOp::BoolNot,
            };

            CoreExpr::LetExpr {
                name: operand_var,
                name_hint: Some("operand".into()),
                value: Box::new(operand_lowered),
                body: Box::new(CoreExpr::PrimOp {
                    op: prim_op,
                    args: vec![operand_var],
                }),
            }
        }

        MonoExprKind::Seq { first, second } => {
            let first_lowered = lower_expr(ctx, first, false);
            let second_lowered = lower_expr(ctx, second, in_tail);

            CoreExpr::Seq {
                first: Box::new(first_lowered),
                second: Box::new(second_lowered),
            }
        }

        MonoExprKind::Record { name, fields } => {
            // Lower record as a tuple-like allocation
            let mut field_vars = Vec::new();
            let mut bindings = Vec::new();

            for (_, expr) in fields {
                let lowered = lower_expr(ctx, expr, false);
                let var = ctx.fresh();
                bindings.push((var, lowered));
                field_vars.push(var);
            }

            let alloc = CoreExpr::Alloc {
                tag: 0, // Record tag
                type_name: Some("Record".to_string()),
                ctor_name: Some(name.clone()),
                fields: field_vars,
            };

            wrap_in_lets(bindings, alloc)
        }

        MonoExprKind::FieldAccess { record, field: _ } => {
            let record_lowered = lower_expr(ctx, record, false);
            let record_var = ctx.fresh();

            // Field access becomes projection - for now just return the record
            // Full implementation would compute field index
            CoreExpr::LetExpr {
                name: record_var,
                name_hint: Some("record".into()),
                value: Box::new(record_lowered),
                body: Box::new(CoreExpr::Var(record_var)),
            }
        }

        MonoExprKind::Perform { effect, op, args } => {
            // Lower arguments and bind to fresh variables
            let mut arg_vars = Vec::new();
            let mut bindings = Vec::new();

            for arg in args {
                let lowered = lower_expr(ctx, arg, false);
                let v = ctx.fresh();
                bindings.push((v, lowered));
                arg_vars.push(v);
            }

            let effect_id = ctx.get_effect_id(effect);
            let op_id = ctx.get_op_id(effect, op);

            let perform = CoreExpr::Perform {
                effect: effect_id,
                effect_name: Some(effect.clone()),
                op: op_id,
                op_name: Some(op.clone()),
                args: arg_vars,
            };

            // Wrap in let bindings for arguments
            let mut result = perform;
            for (v, lowered) in bindings.into_iter().rev() {
                result = CoreExpr::LetExpr {
                    name: v,
                    name_hint: Some("arg".into()),
                    value: Box::new(lowered),
                    body: Box::new(result),
                };
            }
            result
        }

        MonoExprKind::Handle { body, handler } => {
            let lowered_body = lower_expr(ctx, body, false);
            let lowered_handler = lower_handler(ctx, handler);

            CoreExpr::Handle {
                body: Box::new(lowered_body),
                handler: lowered_handler,
            }
        }

        MonoExprKind::Error(msg) => CoreExpr::Error(msg.clone()),

        // Dictionary-based trait dispatch - these need to be handled specially
        MonoExprKind::DictCall {
            trait_name,
            method,
            dict,
            args,
        } => {
            // Check if the dictionary is a simple DictBuild (no sub-dicts)
            // In that case, we can generate a direct function call
            match &dict.node {
                MonoExprKind::DictBuild { for_type, sub_dicts, .. } if sub_dicts.is_empty() => {
                    // Simple instance - generate direct call
                    let fn_name = format!("{}_{}_{}", trait_name, for_type.mangle(), method);

                    // Look up or create the function variable
                    let fn_var = if let Some(v) = ctx.lookup(&fn_name) {
                        v
                    } else {
                        // Create a fresh variable for this function
                        let v = ctx.fresh_named(&fn_name);
                        ctx.bind(&fn_name, v);
                        v
                    };

                    // Lower the arguments
                    let mut arg_bindings: Vec<(VarId, CoreExpr)> = vec![];
                    let mut arg_vars: Vec<VarId> = vec![];

                    for arg in args {
                        let lowered = lower_expr(ctx, arg, false);
                        match lowered {
                            CoreExpr::Var(v) => arg_vars.push(v),
                            _ => {
                                let v = ctx.fresh();
                                arg_bindings.push((v, lowered));
                                arg_vars.push(v);
                            }
                        }
                    }

                    // Build the call
                    let mut result = CoreExpr::App {
                        func: fn_var,
                        args: arg_vars,
                    };

                    // Wrap with let bindings if needed
                    for (v, expr) in arg_bindings.into_iter().rev() {
                        result = CoreExpr::LetExpr {
                            name: v,
                            name_hint: None,
                            value: Box::new(expr),
                            body: Box::new(result),
                        };
                    }

                    result
                }
                MonoExprKind::DictBuild { for_type, .. } => {
                    // Parametric instance with sub-dicts (e.g., Show (List Int))
                    // Generate a direct call to the specialized function
                    // The specialized function is generated by mono.rs
                    let fn_name = format!("{}_{}_{}", trait_name, for_type.mangle(), method);

                    let fn_var = if let Some(v) = ctx.lookup(&fn_name) {
                        v
                    } else {
                        let v = ctx.fresh_named(&fn_name);
                        ctx.bind(&fn_name, v);
                        v
                    };

                    // Lower the arguments
                    let mut arg_bindings: Vec<(VarId, CoreExpr)> = vec![];
                    let mut arg_vars: Vec<VarId> = vec![];

                    for arg in args {
                        let lowered = lower_expr(ctx, arg, false);
                        match lowered {
                            CoreExpr::Var(v) => arg_vars.push(v),
                            _ => {
                                let v = ctx.fresh();
                                arg_bindings.push((v, lowered));
                                arg_vars.push(v);
                            }
                        }
                    }

                    let mut result = CoreExpr::App {
                        func: fn_var,
                        args: arg_vars,
                    };

                    for (v, expr) in arg_bindings.into_iter().rev() {
                        result = CoreExpr::LetExpr {
                            name: v,
                            name_hint: None,
                            value: Box::new(expr),
                            body: Box::new(result),
                        };
                    }

                    result
                }
                MonoExprKind::DictParam { .. } => {
                    // Dictionary is a parameter - need actual dictionary dispatch
                    // For now, return error
                    CoreExpr::Error(format!(
                        "DictCall through dict param not yet implemented: {}.{}",
                        trait_name, method
                    ))
                }
                _ => {
                    CoreExpr::Error(format!(
                        "DictCall with unknown dict type: {}.{}",
                        trait_name, method
                    ))
                }
            }
        }

        MonoExprKind::DictParam { param_idx, .. } => {
            // Reference to a dictionary parameter
            // Generate a variable reference using a mangled name
            let dict_var_name = format!("__dict_{}", param_idx);
            // Look up or create a fresh var
            if let Some(var_id) = ctx.lookup(&dict_var_name) {
                CoreExpr::Var(var_id)
            } else {
                let var_id = ctx.fresh_named(&dict_var_name);
                ctx.bind(&dict_var_name, var_id);
                CoreExpr::Var(var_id)
            }
        }

        MonoExprKind::DictBuild {
            trait_name,
            for_type,
            sub_dicts,
        } => {
            // Build a dictionary for a type
            // For simple instances (no sub_dicts), reference a static dictionary
            // For parametric instances, call a factory function
            if sub_dicts.is_empty() {
                // Simple instance - reference static dict
                let dict_name = format!("{}_{}_dict", trait_name, for_type.mangle());
                if let Some(var_id) = ctx.lookup(&dict_name) {
                    CoreExpr::Var(var_id)
                } else {
                    let var_id = ctx.fresh_named(&dict_name);
                    ctx.bind(&dict_name, var_id);
                    CoreExpr::Var(var_id)
                }
            } else {
                // Parametric instance - call factory
                CoreExpr::Error(format!(
                    "DictBuild with sub-dicts not yet implemented: {} for {:?}",
                    trait_name, for_type
                ))
            }
        }
    }
}

/// Lower a MonoHandler to a Handler
fn lower_handler(ctx: &mut LowerMonoCtx, handler: &MonoHandler) -> Handler {
    // Get effect ID - infer from op_clauses if not explicitly specified
    let effect_name = handler.effect.clone().unwrap_or_else(|| {
        // Try to infer effect name from op_clauses by looking up in op_ids
        // The op_ids map is structured as effect_name -> (op_name -> OpId)
        for clause in &handler.op_clauses {
            for (effect, ops) in ctx.op_ids.iter() {
                if ops.contains_key(&clause.op_name) {
                    return effect.clone();
                }
            }
        }
        // Fallback if we can't infer
        "unknown".to_string()
    });
    let effect_id = ctx.get_effect_id(&effect_name);

    // Lower return clause
    let (return_var, return_body) = if let Some(clause) = &handler.return_clause {
        ctx.push_scope();
        let (var, _) = lower_pattern_bind(ctx, &clause.pattern);
        let body = lower_expr(ctx, &clause.body, false);
        ctx.pop_scope();
        (var, Box::new(body))
    } else {
        // Default return clause: return x -> x
        let var = ctx.fresh();
        (var, Box::new(CoreExpr::Var(var)))
    };

    // Lower operation handlers
    let ops = handler.op_clauses.iter().map(|clause| {
        ctx.push_scope();

        // Bind parameters
        let params: Vec<VarId> = clause.params.iter().map(|p| {
            let (var, _) = lower_pattern_bind(ctx, p);
            var
        }).collect();

        // Bind continuation
        let cont = ctx.fresh_named(&clause.continuation);
        ctx.bind(&clause.continuation, cont);

        let body = lower_expr(ctx, &clause.body, false);
        ctx.pop_scope();

        let op_id = ctx.get_op_id(&effect_name, &clause.op_name);

        OpHandler {
            op: op_id,
            op_name: Some(clause.op_name.clone()),
            params,
            cont,
            body,
        }
    }).collect();

    Handler {
        effect: effect_id,
        effect_name: Some(effect_name),
        return_var,
        return_body,
        ops,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Lower a literal
fn lower_literal(lit: &Literal) -> CoreLit {
    match lit {
        Literal::Int(n) => CoreLit::Int(*n),
        Literal::Float(f) => CoreLit::Float(*f),
        Literal::Bool(b) => CoreLit::Bool(*b),
        Literal::String(s) => CoreLit::String(s.clone()),
        Literal::Char(c) => CoreLit::Char(*c),
        Literal::Unit => CoreLit::Unit,
    }
}

/// Projection info for nested pattern binding
struct PatternProjection {
    /// The variable to bind
    var: VarId,
    /// The name hint
    hint: Option<String>,
    /// The source variable to project from
    source_var: VarId,
    /// The field index to project
    field_idx: usize,
}

/// Bind a pattern and return the variable.
/// Also returns any projections needed to bind nested patterns.
fn lower_pattern_bind_with_projections(
    ctx: &mut LowerMonoCtx,
    pattern: &MonoPattern,
) -> (VarId, Option<String>, Vec<PatternProjection>) {
    match &pattern.node {
        MonoPatternKind::Var(name) => {
            let var = ctx.fresh_named(name);
            ctx.bind(name, var);
            (var, Some(name.clone()), vec![])
        }
        MonoPatternKind::Wildcard => {
            let var = ctx.fresh();
            (var, None, vec![])
        }
        MonoPatternKind::Tuple(elems) => {
            // Create a variable for the whole tuple
            let tuple_var = ctx.fresh();
            let mut projections = Vec::new();

            // Create projections for each tuple element
            for (i, elem) in elems.iter().enumerate() {
                match &elem.node {
                    MonoPatternKind::Var(name) => {
                        let elem_var = ctx.fresh_named(name);
                        ctx.bind(name, elem_var);
                        projections.push(PatternProjection {
                            var: elem_var,
                            hint: Some(name.clone()),
                            source_var: tuple_var,
                            field_idx: i,
                        });
                    }
                    MonoPatternKind::Wildcard => {
                        // No binding needed
                    }
                    _ => {
                        // Nested complex patterns - for now, just create a var
                        let elem_var = ctx.fresh();
                        projections.push(PatternProjection {
                            var: elem_var,
                            hint: None,
                            source_var: tuple_var,
                            field_idx: i,
                        });
                    }
                }
            }
            (tuple_var, None, projections)
        }
        _ => {
            // Other complex patterns - bind a variable for the whole thing
            let var = ctx.fresh();
            (var, None, vec![])
        }
    }
}

/// Bind a pattern and return the variable (simple version without projections)
fn lower_pattern_bind(ctx: &mut LowerMonoCtx, pattern: &MonoPattern) -> (VarId, Option<String>) {
    let (var, hint, _) = lower_pattern_bind_with_projections(ctx, pattern);
    (var, hint)
}

/// Get a name hint from a pattern
fn pattern_hint(pattern: &MonoPattern) -> String {
    match &pattern.node {
        MonoPatternKind::Var(name) => name.clone(),
        _ => "_".to_string(),
    }
}

/// Lower a recursive binding with a pre-bound variable.
/// The name has already been bound in the enclosing scope, so recursive calls
/// and references from the continuation body will find it.
fn lower_rec_binding_with_var(
    ctx: &mut LowerMonoCtx,
    binding: &MonoRecBinding,
    prebound_var: VarId,
) -> RecBinding {
    // Push scope for parameters (but the function name is already bound in outer scope)
    ctx.push_scope();

    let mut param_vars = Vec::new();
    let mut param_hints: Vec<Option<String>> = Vec::new();
    for param in &binding.params {
        let (var, hint) = lower_pattern_bind(ctx, param);
        param_vars.push(var);
        param_hints.push(hint);
    }

    let body = lower_expr(ctx, &binding.body, true);
    ctx.pop_scope();

    RecBinding {
        name: prebound_var,
        name_hint: Some(binding.name.clone()),
        params: param_vars,
        param_hints,
        body,
    }
}

/// Lower a match arm
fn lower_match_arm(ctx: &mut LowerMonoCtx, arm: &MonoMatchArm, in_tail: bool) -> Option<Alt> {
    ctx.push_scope();

    // Get tag from pattern, along with any projections needed
    // The tuple includes an optional guard for patterns like [x] that need sub-pattern checks
    let mut all_projections = Vec::new();
    let (tag, tag_name, binders, binder_hints, guard) = match &arm.pattern.node {
        MonoPatternKind::Constructor { name, args } => {
            let tag = ctx.get_tag("ADT", name);
            let mut binders = Vec::new();
            let mut binder_hints = Vec::new();
            for p in args {
                let (var, hint, projs) = lower_pattern_bind_with_projections(ctx, p);
                binders.push(var);
                binder_hints.push(hint);
                all_projections.extend(projs);
            }
            (tag, Some(name.clone()), binders, binder_hints, None)
        }
        MonoPatternKind::List(elems) if elems.is_empty() => {
            // Empty list pattern []
            let tag = ctx.get_tag("List", "Nil");
            (tag, Some("Nil".to_string()), vec![], vec![], None)
        }
        MonoPatternKind::List(elems) if elems.len() == 1 => {
            // Single-element list pattern [x]
            // Matches Cons(head, tail) where tail is Nil
            let cons_tag = ctx.get_tag("List", "Cons");
            let nil_tag = ctx.get_tag("List", "Nil");

            // Bind head to the pattern element, tail to a fresh var for guard
            let (head_var, head_hint, head_projs) =
                lower_pattern_bind_with_projections(ctx, &elems[0]);
            let tail_var = ctx.fresh_named("tail");
            all_projections.extend(head_projs);

            // Create guard: tail must have tag Nil
            let guard = Some(Box::new(CoreExpr::PrimOp {
                op: PrimOp::TagEq(nil_tag),
                args: vec![tail_var],
            }));

            (cons_tag, Some("Cons".to_string()), vec![head_var, tail_var], vec![head_hint, Some("tail".into())], guard)
        }
        MonoPatternKind::Cons { head, tail } => {
            // Cons pattern x :: xs (head might be a tuple pattern)
            let tag = ctx.get_tag("List", "Cons");
            let (head_var, head_hint, head_projs) =
                lower_pattern_bind_with_projections(ctx, head);
            let (tail_var, tail_hint, tail_projs) =
                lower_pattern_bind_with_projections(ctx, tail);
            all_projections.extend(head_projs);
            all_projections.extend(tail_projs);
            (
                tag,
                Some("Cons".to_string()),
                vec![head_var, tail_var],
                vec![head_hint, tail_hint],
                None,
            )
        }
        MonoPatternKind::Lit(Literal::Int(n)) => {
            // Integer literal pattern
            (*n as u32, Some(format!("{}", n)), vec![], vec![], None)
        }
        MonoPatternKind::Lit(Literal::Bool(b)) => {
            let (tag, name) = if *b { (1u32, "True") } else { (0u32, "False") };
            (tag, Some(name.to_string()), vec![], vec![], None)
        }
        MonoPatternKind::Wildcard => {
            // Wildcard pattern - we need a default case
            // Return None to indicate this should be handled as default
            ctx.pop_scope();
            return None;
        }
        MonoPatternKind::Var(name) => {
            // Variable pattern - binds to the scrutinee
            // This should be a default case
            let var = ctx.fresh_named(name);
            ctx.bind(name, var);
            // Return None and handle as default
            ctx.pop_scope();
            return None;
        }
        _ => {
            // Other patterns - treat as default
            ctx.pop_scope();
            return None;
        }
    };

    // Lower the arm body
    let mut body = lower_expr(ctx, &arm.body, in_tail);

    // Wrap body with projection let-bindings (in reverse order so they nest correctly)
    for proj in all_projections.into_iter().rev() {
        body = CoreExpr::LetExpr {
            name: proj.var,
            name_hint: proj.hint,
            value: Box::new(CoreExpr::Proj {
                tuple: proj.source_var,
                index: proj.field_idx,
            }),
            body: Box::new(body),
        };
    }

    ctx.pop_scope();

    Some(Alt {
        tag,
        tag_name,
        binders,
        binder_hints,
        guard,
        body,
    })
}

/// Collect arguments from nested App expressions
fn collect_app_args<'a>(func: &'a MonoExpr, arg: &'a MonoExpr) -> (&'a MonoExpr, Vec<&'a MonoExpr>) {
    let mut args = vec![arg];
    let mut current = func;

    while let MonoExprKind::App {
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

/// Wrap an expression in let bindings
fn wrap_in_lets(bindings: Vec<(VarId, CoreExpr)>, body: CoreExpr) -> CoreExpr {
    let mut result = body;
    for (var, value) in bindings.into_iter().rev() {
        result = CoreExpr::LetExpr {
            name: var,
            name_hint: None,
            value: Box::new(value),
            body: Box::new(result),
        };
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_type_to_core() {
        assert!(matches!(mono_type_to_core(&MonoType::Int), CoreType::Int));
        assert!(matches!(
            mono_type_to_core(&MonoType::Bool),
            CoreType::Bool
        ));
    }
}
