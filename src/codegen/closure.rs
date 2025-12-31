//! Closure Conversion Pass
//!
//! Transforms CoreIR (with Lam expressions) into FlatIR (no lambdas, explicit closures).
//!
//! ## Architecture
//!
//! ```text
//! CoreIR (with Lam) → closure_convert() → FlatIR (MakeClosure/CallClosure) → C
//! ```
//!
//! ## Key Transformations
//!
//! 1. Find free variables in each lambda
//! 2. Lift lambda to top-level function with env parameter
//! 3. Create env struct for captures
//! 4. Replace lambda with MakeClosure
//! 5. Replace App with CallClosure or CallDirect

use std::collections::{HashMap, HashSet};

use super::core_ir::{
    Alt, Atom, CoreExpr, CoreLit, CoreProgram, CoreType, FunDef, PrimOp, TypeDef, VarId, VarGen,
    CPSHandler, CPSOpHandler, Handler, OpHandler,
};

// ============================================================================
// Flat IR Types (Post-Closure Conversion)
// ============================================================================

/// A program after closure conversion - no lambda expressions remain
#[derive(Debug, Clone)]
pub struct FlatProgram {
    /// Type definitions (passed through unchanged)
    pub types: Vec<TypeDef>,
    /// All functions (original top-level + lifted lambdas)
    pub functions: Vec<FlatFn>,
    /// Environment structs for closures with captures
    pub env_structs: Vec<EnvStruct>,
    /// Builtin function mappings
    pub builtins: Vec<(VarId, String)>,
    /// Main expression (transformed)
    pub main: Option<FlatExpr>,
}

/// A top-level function after closure conversion
#[derive(Debug, Clone)]
pub struct FlatFn {
    pub name: String,
    pub var_id: VarId,
    /// Environment parameter (None if no captures)
    pub env_param: Option<EnvStructId>,
    /// Regular parameters
    pub params: Vec<VarId>,
    pub param_hints: Vec<String>,
    pub param_types: Vec<CoreType>,
    pub return_type: CoreType,
    pub body: FlatExpr,
    pub is_tail_recursive: bool,
}

/// Environment struct for a closure's captured variables
#[derive(Debug, Clone)]
pub struct EnvStruct {
    pub id: EnvStructId,
    /// Original function this env is for
    pub for_func: String,
    /// Captured variable names and their types (for debugging)
    pub fields: Vec<(VarId, String, CoreType)>,
}

/// ID for environment structs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvStructId(pub u32);

// ============================================================================
// Flat IR Expressions
// ============================================================================

/// Expressions after closure conversion - NO Lam!
#[derive(Debug, Clone)]
pub enum FlatExpr {
    /// Variable reference
    Var(VarId),

    /// Literal value
    Lit(CoreLit),

    /// Let binding with atomic RHS
    Let {
        name: VarId,
        name_hint: Option<String>,
        value: Box<FlatAtom>,
        body: Box<FlatExpr>,
    },

    /// Let binding with complex RHS
    LetExpr {
        name: VarId,
        name_hint: Option<String>,
        value: Box<FlatExpr>,
        body: Box<FlatExpr>,
    },

    /// Create a closure (replaces Lam)
    MakeClosure {
        /// Name of the lifted top-level function
        func: String,
        /// Function's VarId (for lookup)
        func_var: VarId,
        /// Arity of the function
        arity: usize,
        /// Values to capture in the environment
        captures: Vec<VarId>,
    },

    /// Call a closure (unknown function)
    CallClosure {
        closure: VarId,
        args: Vec<VarId>,
    },

    /// Call a known function directly (optimization)
    CallDirect {
        func: String,
        func_var: VarId,
        args: Vec<VarId>,
    },

    /// Tail call to closure
    TailCallClosure {
        closure: VarId,
        args: Vec<VarId>,
    },

    /// Tail call to known function
    TailCallDirect {
        func: String,
        func_var: VarId,
        args: Vec<VarId>,
    },

    /// Allocate a constructor/data value
    Alloc {
        tag: u32,
        type_name: Option<String>,
        ctor_name: Option<String>,
        fields: Vec<VarId>,
    },

    /// Pattern match / case analysis
    Case {
        scrutinee: VarId,
        alts: Vec<FlatAlt>,
        default: Option<Box<FlatExpr>>,
    },

    /// If expression
    If {
        cond: VarId,
        then_branch: Box<FlatExpr>,
        else_branch: Box<FlatExpr>,
    },

    /// Recursive function definition (lifted to top-level, but syntax remains for local binding)
    LetRec {
        name: VarId,
        name_hint: Option<String>,
        /// Reference to the lifted function
        lifted_func: String,
        body: Box<FlatExpr>,
    },

    /// Mutually recursive functions
    LetRecMutual {
        /// (var_id, lifted_func_name)
        bindings: Vec<(VarId, String)>,
        body: Box<FlatExpr>,
    },

    /// Sequence
    Seq {
        first: Box<FlatExpr>,
        second: Box<FlatExpr>,
    },

    /// Primitive operation
    PrimOp {
        op: PrimOp,
        args: Vec<VarId>,
    },

    /// Return value
    Return(VarId),

    /// External call
    ExternCall {
        name: String,
        args: Vec<VarId>,
    },

    /// Dictionary method call
    DictCall {
        dict: VarId,
        method: String,
        args: Vec<VarId>,
    },

    /// Dictionary value
    DictValue {
        trait_name: String,
        instance_ty: String,
    },

    /// Dictionary reference
    DictRef(VarId),

    /// Tuple projection
    Proj {
        tuple: VarId,
        index: usize,
    },

    /// Error
    Error(String),

    // Effect-related (CPS-transformed)
    /// Perform effect operation
    Perform {
        effect: u32,
        effect_name: Option<String>,
        op: u32,
        op_name: Option<String>,
        args: Vec<VarId>,
    },

    /// Handle effects
    Handle {
        body: Box<FlatExpr>,
        handler: FlatHandler,
    },

    /// CPS: AppCont
    AppCont {
        func: VarId,
        args: Vec<VarId>,
        cont: VarId,
    },

    /// CPS: Resume
    Resume {
        cont: VarId,
        value: VarId,
    },

    /// CPS: CaptureK
    CaptureK {
        effect: u32,
        effect_name: Option<String>,
        op: u32,
        op_name: Option<String>,
        args: Vec<VarId>,
        cont: VarId,
    },

    /// CPS: WithHandler
    WithHandler {
        effect: u32,
        effect_name: Option<String>,
        handler: FlatCPSHandler,
        body: Box<FlatExpr>,
        outer_cont: VarId,
    },
}

/// Atomic expressions (no computation)
#[derive(Debug, Clone)]
pub enum FlatAtom {
    Var(VarId),
    Lit(CoreLit),
    Alloc {
        tag: u32,
        type_name: Option<String>,
        ctor_name: Option<String>,
        fields: Vec<VarId>,
    },
    PrimOp {
        op: PrimOp,
        args: Vec<VarId>,
    },
    /// Make a closure (replaces Atom::Lam)
    MakeClosure {
        func: String,
        func_var: VarId,
        arity: usize,
        captures: Vec<VarId>,
    },
    /// Call (for atoms that are calls)
    Call {
        func: VarId,
        args: Vec<VarId>,
    },
}

/// Case alternative
#[derive(Debug, Clone)]
pub struct FlatAlt {
    pub tag: u32,
    pub tag_name: Option<String>,
    pub binders: Vec<VarId>,
    pub binder_hints: Vec<Option<String>>,
    pub body: FlatExpr,
}

/// Effect handler (post-closure conversion)
#[derive(Debug, Clone)]
pub struct FlatHandler {
    pub effect: u32,
    pub effect_name: Option<String>,
    pub return_var: VarId,
    pub return_body: Box<FlatExpr>,
    pub ops: Vec<FlatOpHandler>,
}

/// Operation handler
#[derive(Debug, Clone)]
pub struct FlatOpHandler {
    pub op: u32,
    pub op_name: Option<String>,
    pub params: Vec<VarId>,
    pub cont: VarId,
    pub body: FlatExpr,
}

/// CPS handler (post-closure conversion)
#[derive(Debug, Clone)]
pub struct FlatCPSHandler {
    pub return_handler: VarId,
    pub op_handlers: Vec<FlatCPSOpHandler>,
}

/// CPS operation handler
#[derive(Debug, Clone)]
pub struct FlatCPSOpHandler {
    pub op: u32,
    pub op_name: Option<String>,
    pub handler_fn: VarId,
}

// ============================================================================
// Closure Conversion Context
// ============================================================================

struct ClosureCtx {
    /// Generated variable counter
    var_gen: VarGen,
    /// Environment struct counter
    env_counter: u32,
    /// Lifted functions to add to the program
    lifted_functions: Vec<FlatFn>,
    /// Environment structs for closures
    env_structs: Vec<EnvStruct>,
    /// Known top-level function names (for CallDirect optimization)
    known_functions: HashMap<VarId, String>,
    /// Known function arities
    func_arities: HashMap<VarId, usize>,
    /// Counter for unique function names
    func_counter: u32,
}

impl ClosureCtx {
    fn new(program: &CoreProgram) -> Self {
        let mut known_functions = HashMap::new();
        let mut func_arities = HashMap::new();

        // Register all top-level functions
        for fun in &program.functions {
            known_functions.insert(fun.var_id, fun.name.clone());
            func_arities.insert(fun.var_id, fun.params.len());
        }

        // Find max VarId across the ENTIRE program (including all expressions)
        // This is critical to avoid VarId collisions with lifted lambdas
        let mut max_var = 0u32;
        for fun in &program.functions {
            max_var = max_var.max(fun.var_id.0);
            for p in &fun.params {
                max_var = max_var.max(p.0);
            }
            max_var = max_var.max(max_var_in_expr(&fun.body));
        }
        if let Some(main) = &program.main {
            max_var = max_var.max(max_var_in_expr(main));
        }
        for (var_id, _) in &program.builtins {
            max_var = max_var.max(var_id.0);
        }

        ClosureCtx {
            var_gen: VarGen::starting_after(max_var),
            env_counter: 0,
            lifted_functions: Vec::new(),
            env_structs: Vec::new(),
            known_functions,
            func_arities,
            func_counter: 0,
        }
    }

    fn fresh_var(&mut self) -> VarId {
        self.var_gen.fresh()
    }

    fn fresh_env_id(&mut self) -> EnvStructId {
        let id = EnvStructId(self.env_counter);
        self.env_counter += 1;
        id
    }

    fn fresh_func_name(&mut self, hint: &str) -> String {
        let name = format!("lambda_{}_{}", hint, self.func_counter);
        self.func_counter += 1;
        name
    }

    fn is_known_function(&self, var: VarId) -> bool {
        self.known_functions.contains_key(&var)
    }

    fn get_function_name(&self, var: VarId) -> Option<&String> {
        self.known_functions.get(&var)
    }

    fn get_function_arity(&self, var: VarId) -> Option<usize> {
        self.func_arities.get(&var).copied()
    }
}

// ============================================================================
// VarId Analysis
// ============================================================================

/// Find the maximum VarId used in an expression (for avoiding collisions)
fn max_var_in_expr(expr: &CoreExpr) -> u32 {
    match expr {
        CoreExpr::Var(v) => v.0,
        CoreExpr::Lit(_) => 0,
        CoreExpr::Let { name, value, body, .. } => {
            name.0.max(max_var_in_atom(value)).max(max_var_in_expr(body))
        }
        CoreExpr::LetExpr { name, value, body, .. } => {
            name.0.max(max_var_in_expr(value)).max(max_var_in_expr(body))
        }
        CoreExpr::App { func, args } | CoreExpr::TailApp { func, args } => {
            args.iter().map(|v| v.0).fold(func.0, |a, b| a.max(b))
        }
        CoreExpr::Alloc { fields, .. } => {
            fields.iter().map(|v| v.0).max().unwrap_or(0)
        }
        CoreExpr::Case { scrutinee, alts, default } => {
            let mut m = scrutinee.0;
            for alt in alts {
                for b in &alt.binders {
                    m = m.max(b.0);
                }
                m = m.max(max_var_in_expr(&alt.body));
            }
            if let Some(d) = default {
                m = m.max(max_var_in_expr(d));
            }
            m
        }
        CoreExpr::If { cond, then_branch, else_branch } => {
            cond.0.max(max_var_in_expr(then_branch)).max(max_var_in_expr(else_branch))
        }
        CoreExpr::Lam { params, body, .. } => {
            let m = params.iter().map(|v| v.0).max().unwrap_or(0);
            m.max(max_var_in_expr(body))
        }
        CoreExpr::LetRec { name, params, func_body, body, .. } => {
            let mut m = name.0;
            for p in params {
                m = m.max(p.0);
            }
            m.max(max_var_in_expr(func_body)).max(max_var_in_expr(body))
        }
        CoreExpr::LetRecMutual { bindings, body } => {
            let mut m = 0;
            for b in bindings {
                m = m.max(b.name.0);
                for p in &b.params {
                    m = m.max(p.0);
                }
                m = m.max(max_var_in_expr(&b.body));
            }
            m.max(max_var_in_expr(body))
        }
        CoreExpr::Seq { first, second } => {
            max_var_in_expr(first).max(max_var_in_expr(second))
        }
        CoreExpr::PrimOp { args, .. } => {
            args.iter().map(|v| v.0).max().unwrap_or(0)
        }
        CoreExpr::Return(v) => v.0,
        CoreExpr::ExternCall { args, .. } => {
            args.iter().map(|v| v.0).max().unwrap_or(0)
        }
        CoreExpr::DictCall { dict, args, .. } => {
            args.iter().map(|v| v.0).fold(dict.0, |a, b| a.max(b))
        }
        CoreExpr::DictValue { .. } => 0,
        CoreExpr::DictRef(v) => v.0,
        CoreExpr::Proj { tuple, .. } => tuple.0,
        CoreExpr::Error(_) => 0,
        CoreExpr::Perform { args, .. } => {
            args.iter().map(|v| v.0).max().unwrap_or(0)
        }
        CoreExpr::Handle { body, handler } => {
            let mut m = max_var_in_expr(body);
            m = m.max(handler.return_var.0);
            m = m.max(max_var_in_expr(&handler.return_body));
            for op in &handler.ops {
                for p in &op.params {
                    m = m.max(p.0);
                }
                m = m.max(op.cont.0);
                m = m.max(max_var_in_expr(&op.body));
            }
            m
        }
        CoreExpr::AppCont { func, args, cont } => {
            args.iter().map(|v| v.0).fold(func.0.max(cont.0), |a, b| a.max(b))
        }
        CoreExpr::Resume { cont, value } => cont.0.max(value.0),
        CoreExpr::CaptureK { args, cont, .. } => {
            args.iter().map(|v| v.0).fold(cont.0, |a, b| a.max(b))
        }
        CoreExpr::WithHandler { handler, body, outer_cont, .. } => {
            let mut m = outer_cont.0;
            m = m.max(handler.return_handler.0);
            for op in &handler.op_handlers {
                m = m.max(op.handler_fn.0);
            }
            m.max(max_var_in_expr(body))
        }
    }
}

fn max_var_in_atom(atom: &Atom) -> u32 {
    match atom {
        Atom::Var(v) => v.0,
        Atom::Lit(_) => 0,
        Atom::Alloc { fields, .. } => fields.iter().map(|v| v.0).max().unwrap_or(0),
        Atom::PrimOp { args, .. } => args.iter().map(|v| v.0).max().unwrap_or(0),
        Atom::Lam { params, body, .. } => {
            let m = params.iter().map(|v| v.0).max().unwrap_or(0);
            m.max(max_var_in_expr(body))
        }
        Atom::App { func, args } => {
            args.iter().map(|v| v.0).fold(func.0, |a, b| a.max(b))
        }
    }
}

// ============================================================================
// Free Variable Analysis
// ============================================================================

/// Find free variables in an expression
fn free_vars(expr: &CoreExpr) -> HashSet<VarId> {
    let mut vars = HashSet::new();
    collect_free_vars(expr, &HashSet::new(), &mut vars);
    vars
}

fn collect_free_vars(expr: &CoreExpr, bound: &HashSet<VarId>, free: &mut HashSet<VarId>) {
    match expr {
        CoreExpr::Var(v) => {
            if !bound.contains(v) {
                free.insert(*v);
            }
        }
        CoreExpr::Lit(_) => {}
        CoreExpr::Let { name, value, body, .. } => {
            collect_free_vars_atom(value, bound, free);
            let mut inner = bound.clone();
            inner.insert(*name);
            collect_free_vars(body, &inner, free);
        }
        CoreExpr::LetExpr { name, value, body, .. } => {
            collect_free_vars(value, bound, free);
            let mut inner = bound.clone();
            inner.insert(*name);
            collect_free_vars(body, &inner, free);
        }
        CoreExpr::App { func, args } | CoreExpr::TailApp { func, args } => {
            if !bound.contains(func) {
                free.insert(*func);
            }
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        CoreExpr::Alloc { fields, .. } => {
            for f in fields {
                if !bound.contains(f) {
                    free.insert(*f);
                }
            }
        }
        CoreExpr::Case { scrutinee, alts, default } => {
            if !bound.contains(scrutinee) {
                free.insert(*scrutinee);
            }
            for alt in alts {
                let mut alt_bound = bound.clone();
                for b in &alt.binders {
                    alt_bound.insert(*b);
                }
                collect_free_vars(&alt.body, &alt_bound, free);
            }
            if let Some(d) = default {
                collect_free_vars(d, bound, free);
            }
        }
        CoreExpr::If { cond, then_branch, else_branch } => {
            if !bound.contains(cond) {
                free.insert(*cond);
            }
            collect_free_vars(then_branch, bound, free);
            collect_free_vars(else_branch, bound, free);
        }
        CoreExpr::Lam { params, body, .. } => {
            let mut inner = bound.clone();
            for p in params {
                inner.insert(*p);
            }
            collect_free_vars(body, &inner, free);
        }
        CoreExpr::LetRec { name, params, func_body, body, .. } => {
            let mut inner = bound.clone();
            inner.insert(*name);
            for p in params {
                inner.insert(*p);
            }
            collect_free_vars(func_body, &inner, free);
            collect_free_vars(body, &inner, free);
        }
        CoreExpr::LetRecMutual { bindings, body } => {
            let mut inner = bound.clone();
            for b in bindings {
                inner.insert(b.name);
            }
            for b in bindings {
                let mut func_bound = inner.clone();
                for p in &b.params {
                    func_bound.insert(*p);
                }
                collect_free_vars(&b.body, &func_bound, free);
            }
            collect_free_vars(body, &inner, free);
        }
        CoreExpr::Seq { first, second } => {
            collect_free_vars(first, bound, free);
            collect_free_vars(second, bound, free);
        }
        CoreExpr::PrimOp { args, .. } => {
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        CoreExpr::Return(v) => {
            if !bound.contains(v) {
                free.insert(*v);
            }
        }
        CoreExpr::ExternCall { args, .. } => {
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        CoreExpr::DictCall { dict, args, .. } => {
            if !bound.contains(dict) {
                free.insert(*dict);
            }
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        CoreExpr::DictValue { .. } => {}
        CoreExpr::DictRef(v) => {
            if !bound.contains(v) {
                free.insert(*v);
            }
        }
        CoreExpr::Proj { tuple, .. } => {
            if !bound.contains(tuple) {
                free.insert(*tuple);
            }
        }
        CoreExpr::Error(_) => {}
        CoreExpr::Perform { args, .. } => {
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        CoreExpr::Handle { body, handler } => {
            collect_free_vars(body, bound, free);
            let mut ret_bound = bound.clone();
            ret_bound.insert(handler.return_var);
            collect_free_vars(&handler.return_body, &ret_bound, free);
            for op in &handler.ops {
                let mut op_bound = bound.clone();
                for p in &op.params {
                    op_bound.insert(*p);
                }
                op_bound.insert(op.cont);
                collect_free_vars(&op.body, &op_bound, free);
            }
        }
        // CPS expressions
        CoreExpr::AppCont { func, args, cont } => {
            if !bound.contains(func) {
                free.insert(*func);
            }
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
            if !bound.contains(cont) {
                free.insert(*cont);
            }
        }
        CoreExpr::Resume { cont, value } => {
            if !bound.contains(cont) {
                free.insert(*cont);
            }
            if !bound.contains(value) {
                free.insert(*value);
            }
        }
        CoreExpr::CaptureK { args, cont, .. } => {
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
            if !bound.contains(cont) {
                free.insert(*cont);
            }
        }
        CoreExpr::WithHandler { handler, body, outer_cont, .. } => {
            if !bound.contains(&handler.return_handler) {
                free.insert(handler.return_handler);
            }
            for op in &handler.op_handlers {
                if !bound.contains(&op.handler_fn) {
                    free.insert(op.handler_fn);
                }
            }
            collect_free_vars(body, bound, free);
            if !bound.contains(outer_cont) {
                free.insert(*outer_cont);
            }
        }
    }
}

fn collect_free_vars_atom(atom: &Atom, bound: &HashSet<VarId>, free: &mut HashSet<VarId>) {
    match atom {
        Atom::Var(v) => {
            if !bound.contains(v) {
                free.insert(*v);
            }
        }
        Atom::Lit(_) => {}
        Atom::Alloc { fields, .. } => {
            for f in fields {
                if !bound.contains(f) {
                    free.insert(*f);
                }
            }
        }
        Atom::PrimOp { args, .. } => {
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        Atom::Lam { params, body, .. } => {
            let mut inner = bound.clone();
            for p in params {
                inner.insert(*p);
            }
            collect_free_vars(body, &inner, free);
        }
        Atom::App { func, args } => {
            if !bound.contains(func) {
                free.insert(*func);
            }
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
    }
}

// ============================================================================
// Closure Conversion
// ============================================================================

/// Convert a CoreProgram to FlatProgram (closure conversion)
pub fn closure_convert(program: &CoreProgram) -> FlatProgram {
    let mut ctx = ClosureCtx::new(program);

    // Convert top-level functions
    let mut functions = Vec::new();
    for fun in &program.functions {
        let flat_fn = convert_function(&mut ctx, fun);
        functions.push(flat_fn);
    }

    // Convert main expression
    let main = program.main.as_ref().map(|e| {
        convert_expr(&mut ctx, e, &HashSet::new())
    });

    // Add lifted functions
    functions.extend(ctx.lifted_functions);

    FlatProgram {
        types: program.types.clone(),
        functions,
        env_structs: ctx.env_structs,
        builtins: program.builtins.clone(),
        main,
    }
}

fn convert_function(ctx: &mut ClosureCtx, fun: &FunDef) -> FlatFn {
    // Build the set of in-scope variables for this function
    let mut in_scope: HashSet<VarId> = HashSet::new();
    for p in &fun.params {
        in_scope.insert(*p);
    }
    // Add all known top-level functions to scope
    for var in ctx.known_functions.keys() {
        in_scope.insert(*var);
    }

    let body = convert_expr(ctx, &fun.body, &in_scope);

    FlatFn {
        name: fun.name.clone(),
        var_id: fun.var_id,
        env_param: None, // Top-level functions don't have env
        params: fun.params.clone(),
        param_hints: fun.param_hints.clone(),
        param_types: fun.param_types.clone(),
        return_type: fun.return_type.clone(),
        body,
        is_tail_recursive: fun.is_tail_recursive,
    }
}

fn convert_expr(ctx: &mut ClosureCtx, expr: &CoreExpr, in_scope: &HashSet<VarId>) -> FlatExpr {
    match expr {
        CoreExpr::Var(v) => FlatExpr::Var(*v),
        CoreExpr::Lit(lit) => FlatExpr::Lit(lit.clone()),

        CoreExpr::Let { name, name_hint, value, body } => {
            let flat_value = convert_atom(ctx, value, in_scope);
            let mut new_scope = in_scope.clone();
            new_scope.insert(*name);
            let flat_body = convert_expr(ctx, body, &new_scope);
            FlatExpr::Let {
                name: *name,
                name_hint: name_hint.clone(),
                value: Box::new(flat_value),
                body: Box::new(flat_body),
            }
        }

        CoreExpr::LetExpr { name, name_hint, value, body } => {
            let flat_value = convert_expr(ctx, value, in_scope);
            let mut new_scope = in_scope.clone();
            new_scope.insert(*name);
            let flat_body = convert_expr(ctx, body, &new_scope);
            FlatExpr::LetExpr {
                name: *name,
                name_hint: name_hint.clone(),
                value: Box::new(flat_value),
                body: Box::new(flat_body),
            }
        }

        CoreExpr::Lam { params, param_hints, body } => {
            // This is the heart of closure conversion!
            // 1. Find free variables
            let body_free = free_vars(body);
            let param_set: HashSet<VarId> = params.iter().copied().collect();
            let captures: Vec<VarId> = body_free
                .difference(&param_set)
                .filter(|v| in_scope.contains(v) && !ctx.is_known_function(**v))
                .copied()
                .collect();

            // 2. Create lifted function
            let func_name = ctx.fresh_func_name("anon");
            let func_var = ctx.fresh_var();

            // Create env struct if there are captures
            let env_param = if !captures.is_empty() {
                let env_id = ctx.fresh_env_id();
                ctx.env_structs.push(EnvStruct {
                    id: env_id,
                    for_func: func_name.clone(),
                    fields: captures.iter().map(|v| (*v, format!("cap_{}", v.0), CoreType::Box(Box::new(CoreType::Unit)))).collect(),
                });
                Some(env_id)
            } else {
                None
            };

            // Build scope for the lifted function body
            let mut func_scope: HashSet<VarId> = HashSet::new();
            for p in params {
                func_scope.insert(*p);
            }
            for c in &captures {
                func_scope.insert(*c);
            }
            // Add known functions
            for var in ctx.known_functions.keys() {
                func_scope.insert(*var);
            }

            // Convert body
            let flat_body = convert_expr(ctx, body, &func_scope);

            // Register the lifted function
            ctx.known_functions.insert(func_var, func_name.clone());
            ctx.func_arities.insert(func_var, params.len());

            let lifted = FlatFn {
                name: func_name.clone(),
                var_id: func_var,
                env_param,
                params: params.clone(),
                param_hints: param_hints.iter().map(|h| h.clone().unwrap_or_default()).collect(),
                param_types: vec![CoreType::Box(Box::new(CoreType::Unit)); params.len()],
                return_type: CoreType::Box(Box::new(CoreType::Unit)),
                body: flat_body,
                is_tail_recursive: false,
            };
            ctx.lifted_functions.push(lifted);

            // 3. Return MakeClosure expression
            FlatExpr::MakeClosure {
                func: func_name,
                func_var,
                arity: params.len(),
                captures,
            }
        }

        CoreExpr::App { func, args } => {
            // Determine if this is a known function call
            if ctx.is_known_function(*func) {
                let func_name = ctx.get_function_name(*func).unwrap().clone();
                let arity = ctx.get_function_arity(*func).unwrap_or(args.len());

                if args.len() <= arity {
                    // Fully applied or under-applied: direct call
                    FlatExpr::CallDirect {
                        func: func_name,
                        func_var: *func,
                        args: args.clone(),
                    }
                } else {
                    // Over-applied: call with arity args, then apply rest
                    // For now, fall back to CallClosure which handles currying
                    FlatExpr::CallClosure {
                        closure: *func,
                        args: args.clone(),
                    }
                }
            } else {
                FlatExpr::CallClosure {
                    closure: *func,
                    args: args.clone(),
                }
            }
        }

        CoreExpr::TailApp { func, args } => {
            if ctx.is_known_function(*func) {
                let func_name = ctx.get_function_name(*func).unwrap().clone();
                let arity = ctx.get_function_arity(*func).unwrap_or(args.len());

                if args.len() <= arity {
                    // Fully applied or under-applied: direct tail call
                    FlatExpr::TailCallDirect {
                        func: func_name,
                        func_var: *func,
                        args: args.clone(),
                    }
                } else {
                    // Over-applied: fall back to closure call which handles currying
                    FlatExpr::TailCallClosure {
                        closure: *func,
                        args: args.clone(),
                    }
                }
            } else {
                FlatExpr::TailCallClosure {
                    closure: *func,
                    args: args.clone(),
                }
            }
        }

        CoreExpr::Alloc { tag, type_name, ctor_name, fields } => {
            FlatExpr::Alloc {
                tag: *tag,
                type_name: type_name.clone(),
                ctor_name: ctor_name.clone(),
                fields: fields.clone(),
            }
        }

        CoreExpr::Case { scrutinee, alts, default } => {
            let flat_alts = alts.iter().map(|alt| {
                let mut alt_scope = in_scope.clone();
                for b in &alt.binders {
                    alt_scope.insert(*b);
                }
                FlatAlt {
                    tag: alt.tag,
                    tag_name: alt.tag_name.clone(),
                    binders: alt.binders.clone(),
                    binder_hints: alt.binder_hints.clone(),
                    body: convert_expr(ctx, &alt.body, &alt_scope),
                }
            }).collect();

            let flat_default = default.as_ref().map(|d| Box::new(convert_expr(ctx, d, in_scope)));

            FlatExpr::Case {
                scrutinee: *scrutinee,
                alts: flat_alts,
                default: flat_default,
            }
        }

        CoreExpr::If { cond, then_branch, else_branch } => {
            FlatExpr::If {
                cond: *cond,
                then_branch: Box::new(convert_expr(ctx, then_branch, in_scope)),
                else_branch: Box::new(convert_expr(ctx, else_branch, in_scope)),
            }
        }

        CoreExpr::LetRec { name, name_hint, params, param_hints, func_body, body } => {
            // Lift the recursive function to top-level
            let func_name = ctx.fresh_func_name(name_hint.as_deref().unwrap_or("rec"));
            let func_var = *name;

            // Register before converting body (for self-reference)
            ctx.known_functions.insert(func_var, func_name.clone());
            ctx.func_arities.insert(func_var, params.len());

            // Find free variables in func_body (like Lam does)
            let body_free = free_vars(func_body);
            let mut bound_set: HashSet<VarId> = params.iter().copied().collect();
            bound_set.insert(func_var); // The function itself is bound (for recursion)

            let captures: Vec<VarId> = body_free
                .difference(&bound_set)
                .filter(|v| in_scope.contains(v) && !ctx.is_known_function(**v))
                .copied()
                .collect();

            // Create env struct if there are captures
            let env_param = if !captures.is_empty() {
                let env_id = ctx.fresh_env_id();
                ctx.env_structs.push(EnvStruct {
                    id: env_id,
                    for_func: func_name.clone(),
                    fields: captures.iter().map(|v| (*v, format!("cap_{}", v.0), CoreType::Box(Box::new(CoreType::Unit)))).collect(),
                });
                Some(env_id)
            } else {
                None
            };

            // Build scope for function body: params + captures + known functions + self
            let mut func_scope: HashSet<VarId> = HashSet::new();
            func_scope.insert(func_var);
            for p in params {
                func_scope.insert(*p);
            }
            for c in &captures {
                func_scope.insert(*c);
            }
            for var in ctx.known_functions.keys() {
                func_scope.insert(*var);
            }

            let flat_func_body = convert_expr(ctx, func_body, &func_scope);

            let lifted = FlatFn {
                name: func_name.clone(),
                var_id: func_var,
                env_param,
                params: params.clone(),
                param_hints: param_hints.iter().map(|h| h.clone().unwrap_or_default()).collect(),
                param_types: vec![CoreType::Box(Box::new(CoreType::Unit)); params.len()],
                return_type: CoreType::Box(Box::new(CoreType::Unit)),
                body: flat_func_body,
                is_tail_recursive: true,
            };
            ctx.lifted_functions.push(lifted);

            // Convert body with the function in scope
            let mut body_scope = in_scope.clone();
            body_scope.insert(func_var);
            let flat_body = convert_expr(ctx, body, &body_scope);

            // If captures exist, wrap function ref in MakeClosure
            let arity = params.len();
            if !captures.is_empty() {
                FlatExpr::Let {
                    name: func_var,
                    name_hint: name_hint.clone(),
                    value: Box::new(FlatAtom::MakeClosure {
                        func: func_name,
                        func_var,
                        arity,
                        captures,
                    }),
                    body: Box::new(flat_body),
                }
            } else {
                FlatExpr::LetRec {
                    name: func_var,
                    name_hint: name_hint.clone(),
                    lifted_func: func_name,
                    body: Box::new(flat_body),
                }
            }
        }

        CoreExpr::LetRecMutual { bindings, body } => {
            // First, register all function names
            let mut func_names = Vec::new();
            let mut inner_scope = in_scope.clone();
            let mut all_binding_names: HashSet<VarId> = HashSet::new();
            for b in bindings {
                let func_name = ctx.fresh_func_name(b.name_hint.as_deref().unwrap_or("mut_rec"));
                ctx.known_functions.insert(b.name, func_name.clone());
                ctx.func_arities.insert(b.name, b.params.len());
                func_names.push((b.name, func_name, b.name_hint.clone()));
                inner_scope.insert(b.name);
                all_binding_names.insert(b.name);
            }

            // Track which bindings have captures (var_id, captures, arity)
            let mut bindings_with_captures: Vec<(VarId, Vec<VarId>, usize)> = Vec::new();

            // Convert and lift each function
            for (binding, (_, func_name, _)) in bindings.iter().zip(func_names.iter()) {
                // Find free variables in this binding's body
                let body_free = free_vars(&binding.body);
                let mut bound_set: HashSet<VarId> = binding.params.iter().copied().collect();
                // All mutual bindings are bound (for mutual recursion)
                bound_set.extend(&all_binding_names);

                let captures: Vec<VarId> = body_free
                    .difference(&bound_set)
                    .filter(|v| in_scope.contains(v) && !ctx.is_known_function(**v))
                    .copied()
                    .collect();

                // Create env struct if there are captures
                let arity = binding.params.len();
                let env_param = if !captures.is_empty() {
                    let env_id = ctx.fresh_env_id();
                    ctx.env_structs.push(EnvStruct {
                        id: env_id,
                        for_func: func_name.clone(),
                        fields: captures.iter().map(|v| (*v, format!("cap_{}", v.0), CoreType::Box(Box::new(CoreType::Unit)))).collect(),
                    });
                    bindings_with_captures.push((binding.name, captures.clone(), arity));
                    Some(env_id)
                } else {
                    None
                };

                // Build scope for function body: params + captures + known functions + all bindings
                let mut func_scope: HashSet<VarId> = HashSet::new();
                for p in &binding.params {
                    func_scope.insert(*p);
                }
                for c in &captures {
                    func_scope.insert(*c);
                }
                for var in ctx.known_functions.keys() {
                    func_scope.insert(*var);
                }
                func_scope.extend(&all_binding_names);

                let flat_body = convert_expr(ctx, &binding.body, &func_scope);

                let lifted = FlatFn {
                    name: func_name.clone(),
                    var_id: binding.name,
                    env_param,
                    params: binding.params.clone(),
                    param_hints: binding.param_hints.iter().map(|h| h.clone().unwrap_or_default()).collect(),
                    param_types: vec![CoreType::Box(Box::new(CoreType::Unit)); binding.params.len()],
                    return_type: CoreType::Box(Box::new(CoreType::Unit)),
                    body: flat_body,
                    is_tail_recursive: true,
                };
                ctx.lifted_functions.push(lifted);
            }

            // Convert body
            let flat_body = convert_expr(ctx, body, &inner_scope);

            // If any bindings have captures, wrap them in MakeClosure
            if !bindings_with_captures.is_empty() {
                // Wrap each captured binding in a Let with MakeClosure
                let mut result = flat_body;
                for (var_id, captures, arity) in bindings_with_captures.into_iter().rev() {
                    let func_name = func_names.iter()
                        .find(|(v, _, _)| *v == var_id)
                        .map(|(_, n, _)| n.clone())
                        .unwrap();
                    let name_hint = func_names.iter()
                        .find(|(v, _, _)| *v == var_id)
                        .and_then(|(_, _, h)| h.clone());
                    result = FlatExpr::Let {
                        name: var_id,
                        name_hint,
                        value: Box::new(FlatAtom::MakeClosure {
                            func: func_name,
                            func_var: var_id,
                            arity,
                            captures,
                        }),
                        body: Box::new(result),
                    };
                }
                result
            } else {
                FlatExpr::LetRecMutual {
                    bindings: func_names.into_iter().map(|(v, n, _)| (v, n)).collect(),
                    body: Box::new(flat_body),
                }
            }
        }

        CoreExpr::Seq { first, second } => {
            FlatExpr::Seq {
                first: Box::new(convert_expr(ctx, first, in_scope)),
                second: Box::new(convert_expr(ctx, second, in_scope)),
            }
        }

        CoreExpr::PrimOp { op, args } => {
            FlatExpr::PrimOp {
                op: *op,
                args: args.clone(),
            }
        }

        CoreExpr::Return(v) => FlatExpr::Return(*v),

        CoreExpr::ExternCall { name, args } => {
            FlatExpr::ExternCall {
                name: name.clone(),
                args: args.clone(),
            }
        }

        CoreExpr::DictCall { dict, method, args } => {
            FlatExpr::DictCall {
                dict: *dict,
                method: method.clone(),
                args: args.clone(),
            }
        }

        CoreExpr::DictValue { trait_name, instance_ty } => {
            FlatExpr::DictValue {
                trait_name: trait_name.clone(),
                instance_ty: instance_ty.clone(),
            }
        }

        CoreExpr::DictRef(v) => FlatExpr::DictRef(*v),

        CoreExpr::Proj { tuple, index } => {
            FlatExpr::Proj {
                tuple: *tuple,
                index: *index,
            }
        }

        CoreExpr::Error(msg) => FlatExpr::Error(msg.clone()),

        CoreExpr::Perform { effect, effect_name, op, op_name, args } => {
            FlatExpr::Perform {
                effect: *effect,
                effect_name: effect_name.clone(),
                op: *op,
                op_name: op_name.clone(),
                args: args.clone(),
            }
        }

        CoreExpr::Handle { body, handler } => {
            let flat_body = convert_expr(ctx, body, in_scope);

            let mut ret_scope = in_scope.clone();
            ret_scope.insert(handler.return_var);
            let flat_return_body = convert_expr(ctx, &handler.return_body, &ret_scope);

            let flat_ops = handler.ops.iter().map(|op| {
                let mut op_scope = in_scope.clone();
                for p in &op.params {
                    op_scope.insert(*p);
                }
                op_scope.insert(op.cont);
                FlatOpHandler {
                    op: op.op,
                    op_name: op.op_name.clone(),
                    params: op.params.clone(),
                    cont: op.cont,
                    body: convert_expr(ctx, &op.body, &op_scope),
                }
            }).collect();

            FlatExpr::Handle {
                body: Box::new(flat_body),
                handler: FlatHandler {
                    effect: handler.effect,
                    effect_name: handler.effect_name.clone(),
                    return_var: handler.return_var,
                    return_body: Box::new(flat_return_body),
                    ops: flat_ops,
                },
            }
        }

        // CPS expressions
        CoreExpr::AppCont { func, args, cont } => {
            FlatExpr::AppCont {
                func: *func,
                args: args.clone(),
                cont: *cont,
            }
        }

        CoreExpr::Resume { cont, value } => {
            FlatExpr::Resume {
                cont: *cont,
                value: *value,
            }
        }

        CoreExpr::CaptureK { effect, effect_name, op, op_name, args, cont } => {
            FlatExpr::CaptureK {
                effect: *effect,
                effect_name: effect_name.clone(),
                op: *op,
                op_name: op_name.clone(),
                args: args.clone(),
                cont: *cont,
            }
        }

        CoreExpr::WithHandler { effect, effect_name, handler, body, outer_cont } => {
            let flat_body = convert_expr(ctx, body, in_scope);
            FlatExpr::WithHandler {
                effect: *effect,
                effect_name: effect_name.clone(),
                handler: FlatCPSHandler {
                    return_handler: handler.return_handler,
                    op_handlers: handler.op_handlers.iter().map(|op| {
                        FlatCPSOpHandler {
                            op: op.op,
                            op_name: op.op_name.clone(),
                            handler_fn: op.handler_fn,
                        }
                    }).collect(),
                },
                body: Box::new(flat_body),
                outer_cont: *outer_cont,
            }
        }
    }
}

fn convert_atom(ctx: &mut ClosureCtx, atom: &Atom, in_scope: &HashSet<VarId>) -> FlatAtom {
    match atom {
        Atom::Var(v) => FlatAtom::Var(*v),
        Atom::Lit(lit) => FlatAtom::Lit(lit.clone()),
        Atom::Alloc { tag, type_name, ctor_name, fields } => {
            FlatAtom::Alloc {
                tag: *tag,
                type_name: type_name.clone(),
                ctor_name: ctor_name.clone(),
                fields: fields.clone(),
            }
        }
        Atom::PrimOp { op, args } => {
            FlatAtom::PrimOp {
                op: *op,
                args: args.clone(),
            }
        }
        Atom::Lam { params, param_hints, body } => {
            // Same closure conversion as CoreExpr::Lam
            let body_free = free_vars(body);
            let param_set: HashSet<VarId> = params.iter().copied().collect();
            let captures: Vec<VarId> = body_free
                .difference(&param_set)
                .filter(|v| in_scope.contains(v) && !ctx.is_known_function(**v))
                .copied()
                .collect();

            let func_name = ctx.fresh_func_name("atom_lam");
            let func_var = ctx.fresh_var();

            let env_param = if !captures.is_empty() {
                let env_id = ctx.fresh_env_id();
                ctx.env_structs.push(EnvStruct {
                    id: env_id,
                    for_func: func_name.clone(),
                    fields: captures.iter().map(|v| (*v, format!("cap_{}", v.0), CoreType::Box(Box::new(CoreType::Unit)))).collect(),
                });
                Some(env_id)
            } else {
                None
            };

            let mut func_scope: HashSet<VarId> = HashSet::new();
            for p in params {
                func_scope.insert(*p);
            }
            for c in &captures {
                func_scope.insert(*c);
            }
            for var in ctx.known_functions.keys() {
                func_scope.insert(*var);
            }

            let flat_body = convert_expr(ctx, body, &func_scope);

            ctx.known_functions.insert(func_var, func_name.clone());
            ctx.func_arities.insert(func_var, params.len());

            let lifted = FlatFn {
                name: func_name.clone(),
                var_id: func_var,
                env_param,
                params: params.clone(),
                param_hints: param_hints.iter().map(|h| h.clone().unwrap_or_default()).collect(),
                param_types: vec![CoreType::Box(Box::new(CoreType::Unit)); params.len()],
                return_type: CoreType::Box(Box::new(CoreType::Unit)),
                body: flat_body,
                is_tail_recursive: false,
            };
            ctx.lifted_functions.push(lifted);

            FlatAtom::MakeClosure {
                func: func_name,
                func_var,
                arity: params.len(),
                captures,
            }
        }
        Atom::App { func, args } => {
            FlatAtom::Call {
                func: *func,
                args: args.clone(),
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_vars_simple() {
        // Test that free_vars correctly identifies free variables
        let expr = CoreExpr::Var(VarId(0));
        let fv = free_vars(&expr);
        assert!(fv.contains(&VarId(0)));
    }

    #[test]
    fn test_free_vars_lam() {
        // fun x -> x + y  -- y is free
        let expr = CoreExpr::Lam {
            params: vec![VarId(0)],
            param_hints: vec![None],
            body: Box::new(CoreExpr::PrimOp {
                op: PrimOp::IntAdd,
                args: vec![VarId(0), VarId(1)],
            }),
        };
        let fv = free_vars(&expr);
        assert!(!fv.contains(&VarId(0))); // x is bound
        assert!(fv.contains(&VarId(1)));  // y is free
    }
}
