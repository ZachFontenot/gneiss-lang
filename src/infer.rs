//! Hindley-Milner type inference

// TODO: Box TypeError to reduce Result size (tracked in roadmap)
#![allow(clippy::result_large_err)]

use crate::ast::*;
use crate::errors::find_similar;
use crate::prelude::parse_prelude;
use crate::types::*;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use thiserror::Error;

/// Substitute Generic type variables with concrete types.
/// Generic(0) is replaced with args[0], Generic(1) with args[1], etc.
/// Note: This operates on unresolved types since Generic vars don't have bindings.
fn substitute_generics(ty: &Type, args: &[Type]) -> Type {
    match ty {
        Type::Generic(id) => {
            args.get(*id as usize).cloned().unwrap_or_else(|| ty.clone())
        }
        Type::Var(id) => Type::Var(*id), // Type variables pass through unchanged
        Type::Arrow { arg, ret, effects } => Type::Arrow {
            arg: Rc::new(substitute_generics(arg, args)),
            ret: Rc::new(substitute_generics(ret, args)),
            effects: substitute_generics_in_row(effects, args),
        },
        Type::Tuple(ts) => {
            Type::Tuple(ts.iter().map(|t| substitute_generics(t, args)).collect())
        }
        Type::Channel(t) => Type::Channel(Rc::new(substitute_generics(t, args))),
        Type::Fiber(t) => Type::Fiber(Rc::new(substitute_generics(t, args))),
        Type::Dict(t) => Type::Dict(Rc::new(substitute_generics(t, args))),
        Type::Constructor { name, args: cargs } => Type::Constructor {
            name: name.clone(),
            args: cargs.iter().map(|t| substitute_generics(t, args)).collect(),
        },
        other => other.clone(),
    }
}

/// Substitute generic type variables in an effect row
fn substitute_generics_in_row(row: &Row, args: &[Type]) -> Row {
    match row {
        Row::Empty => Row::Empty,
        Row::Var(id) => Row::Var(*id), // Row variables don't get substituted by type args
        Row::Generic(id) => Row::Generic(*id), // Generic row variables pass through
        Row::Extend { effect, rest } => Row::Extend {
            effect: Effect {
                name: effect.name.clone(),
                params: effect
                    .params
                    .iter()
                    .map(|t| substitute_generics(t, args))
                    .collect(),
            },
            rest: Rc::new(substitute_generics_in_row(rest, args)),
        },
    }
}

/// Context for where a unification error occurred
#[derive(Debug, Clone)]
pub enum UnifyContext {
    /// Unifying function argument with parameter type
    FunctionArgument {
        func_name: Option<String>,
        param_num: usize,
    },
    /// Unifying function body with declared return type
    FunctionReturn { func_name: Option<String> },
    /// Unifying let binding value with pattern type
    LetBinding { name: String },
    /// Unifying if condition with Bool
    IfCondition,
    /// Unifying if branches (then and else must match)
    IfBranches,
    /// Unifying match arms (all arms must return same type)
    MatchArms,
    /// Unifying match scrutinee with pattern type
    MatchScrutinee,
    /// Unifying operands of a binary operator
    BinOp { op: String, side: &'static str },
    /// Unifying list elements (all must have same type)
    ListElement { index: usize },
    /// Unifying tuple elements
    TupleElement { index: usize },
    /// Unifying constructor argument
    ConstructorArg { ctor_name: String, param_num: usize },
    /// Recursive binding type
    RecursiveBinding { name: String },
}

impl std::fmt::Display for UnifyContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnifyContext::FunctionArgument {
                func_name: Some(name),
                param_num,
            } => {
                write!(f, "in argument {} of `{}`", param_num, name)
            }
            UnifyContext::FunctionArgument {
                func_name: None,
                param_num,
            } => {
                write!(f, "in argument {}", param_num)
            }
            UnifyContext::FunctionReturn {
                func_name: Some(name),
            } => {
                write!(f, "in return type of `{}`", name)
            }
            UnifyContext::FunctionReturn { func_name: None } => {
                write!(f, "in return type")
            }
            UnifyContext::LetBinding { name } => {
                write!(f, "in binding for `{}`", name)
            }
            UnifyContext::IfCondition => {
                write!(f, "in if condition (expected Bool)")
            }
            UnifyContext::IfBranches => {
                write!(f, "in if branches (then and else must have same type)")
            }
            UnifyContext::MatchArms => {
                write!(f, "in match arms (all arms must have same type)")
            }
            UnifyContext::MatchScrutinee => {
                write!(f, "in match scrutinee")
            }
            UnifyContext::BinOp { op, side } => {
                write!(f, "in {} operand of `{}`", side, op)
            }
            UnifyContext::ListElement { index } => {
                write!(f, "in list element {}", index)
            }
            UnifyContext::TupleElement { index } => {
                write!(f, "in tuple element {}", index)
            }
            UnifyContext::ConstructorArg {
                ctor_name,
                param_num,
            } => {
                write!(
                    f,
                    "in argument {} of constructor `{}`",
                    param_num, ctor_name
                )
            }
            UnifyContext::RecursiveBinding { name } => {
                write!(f, "in recursive binding `{}`", name)
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("unbound variable: {name}")]
    UnboundVariable {
        name: String,
        span: Span,
        suggestions: Vec<String>,
    },
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        expected: Type,
        found: Type,
        span: Option<Span>,
        context: Option<UnifyContext>,
    },
    #[error("occurs check failed: {var_id} occurs in {ty}")]
    OccursCheck {
        var_id: TypeVarId,
        ty: Type,
        span: Option<Span>,
    },
    #[error("unknown constructor: {name}")]
    UnknownConstructor {
        name: String,
        span: Span,
        suggestions: Vec<String>,
    },
    #[error("pattern type mismatch")]
    PatternMismatch { span: Span },
    #[error("non-exhaustive patterns")]
    NonExhaustivePatterns { span: Span },
    #[error("unknown trait: {name}")]
    UnknownTrait { name: String, span: Option<Span> },
    #[error("overlapping instances for trait {trait_name}: {existing} and {new}")]
    OverlappingInstance {
        trait_name: String,
        existing: Type,
        new: Type,
        span: Option<Span>,
    },
    #[error("no instance of {trait_name} for type {ty}")]
    NoInstance {
        trait_name: String,
        ty: Type,
        span: Option<Span>,
    },
    #[error("unknown record type: {name}")]
    UnknownRecordType {
        name: String,
        span: Span,
    },
    #[error("missing record field: {field}")]
    MissingRecordField {
        record_type: String,
        field: String,
        span: Span,
    },
    #[error("unknown record field: {field}")]
    UnknownRecordField {
        record_type: String,
        field: String,
        span: Span,
    },
    #[error("not a record type: {ty}")]
    NotARecordType {
        ty: Type,
        span: Span,
    },
    #[error("cannot infer record type from update expression")]
    CannotInferRecordType {
        span: Span,
    },
    #[error("{0}")]
    Other(String),
}

impl TypeError {
    /// Add context to a type mismatch error
    pub fn with_context(self, ctx: UnifyContext) -> TypeError {
        match self {
            TypeError::TypeMismatch {
                expected,
                found,
                span,
                context: _,
            } => TypeError::TypeMismatch {
                expected,
                found,
                span,
                context: Some(ctx),
            },
            other => other,
        }
    }
}

pub struct Inferencer {
    /// Union-Find for type variables
    type_uf: UnionFind,
    /// Union-Find for row variables
    row_uf: RowUnionFind,
    /// Current let-nesting level (for polymorphism)
    level: u32,
    /// Type context for constructors
    type_ctx: TypeContext,
    /// Class environment for typeclasses
    class_env: ClassEnv,
    /// Effect environment for algebraic effects
    effect_env: EffectEnv,
    /// Wanted predicates (constraints collected during inference)
    wanted_preds: Vec<Pred>,
    /// Module environments: module name -> TypeEnv of exports
    module_envs: HashMap<String, TypeEnv>,
    /// Import mappings for current module: local name -> (module name, original name)
    imports: HashMap<String, (String, String)>,
    /// Module aliases: alias -> original module name
    module_aliases: HashMap<String, String>,
    /// Track which imports have been used (for unused import warnings)
    used_imports: HashSet<String>,
    /// Track which module aliases have been used
    used_module_aliases: HashSet<String>,
    /// Accumulated type errors (for multi-error reporting)
    errors: Vec<TypeError>,
}

impl Inferencer {
    pub fn new() -> Self {
        Self {
            type_uf: UnionFind::new(),
            row_uf: RowUnionFind::new(),
            level: 0,
            type_ctx: TypeContext::new(),
            class_env: ClassEnv::new(),
            effect_env: EffectEnv::default(),
            wanted_preds: Vec::new(),
            module_envs: HashMap::new(),
            imports: HashMap::new(),
            module_aliases: HashMap::new(),
            used_imports: HashSet::new(),
            used_module_aliases: HashSet::new(),
            errors: Vec::new(),
        }
    }

    /// Register a module's type environment for qualified access
    pub fn register_module(&mut self, name: String, env: TypeEnv) {
        self.module_envs.insert(name, env);
    }

    /// Add an import mapping: `import Module (item)` or `import Module (item as alias)`
    pub fn add_import(&mut self, local_name: String, module_name: String, original_name: String) {
        self.imports.insert(local_name, (module_name, original_name));
    }

    /// Add a module alias: `import Module as Alias`
    pub fn add_module_alias(&mut self, alias: String, module_name: String) {
        self.module_aliases.insert(alias, module_name);
    }

    /// Register an effect declaration
    /// Converts the AST effect declaration into EffectInfo and OperationInfo
    pub fn register_effect(
        &mut self,
        name: &str,
        params: &[String],
        operations: &[crate::ast::EffectOperation],
    ) -> Result<(), TypeError> {
        // Build a param_map: type param name -> generic ID
        let mut param_map: HashMap<String, TypeVarId> = HashMap::new();
        for (i, param) in params.iter().enumerate() {
            param_map.insert(param.clone(), i as TypeVarId);
        }

        // Convert each operation's type signature
        let mut op_infos = Vec::new();
        for op in operations {
            // The operation's type_sig is a function type like () -> s or s -> ()
            // We need to extract param types, result type, and operation-local generics
            let (param_types, result_type, generics) =
                self.extract_operation_signature(&op.type_sig, &param_map)?;

            let op_info = OperationInfo {
                name: op.name.clone(),
                param_types,
                result_type,
                generics,
            };
            op_infos.push(op_info.clone());

            // Also register in operations lookup (for qualified access like State.get)
            self.effect_env.operations.insert(
                op.name.clone(),
                (name.to_string(), op_info),
            );
        }

        let effect_info = EffectInfo {
            name: name.to_string(),
            type_params: params.to_vec(),
            operations: op_infos,
        };

        self.effect_env.effects.insert(name.to_string(), effect_info);
        Ok(())
    }

    /// Extract parameter types, result type, and operation-local generics from a signature
    /// e.g., `() -> s` gives ([], s, []), `String -> a` gives ([String], a, [0]) if `a` is local
    ///
    /// Supports operation-local polymorphic type variables: if `a` is used in the
    /// signature but not declared as an effect type parameter, it becomes a
    /// Generic type variable local to this operation. This allows patterns like:
    ///   effect Error = | throw : String -> a end
    /// where `a` is polymorphic (the operation never returns normally).
    ///
    /// Returns: (param_types, result_type, operation_generic_ids)
    fn extract_operation_signature(
        &mut self,
        type_sig: &TypeExpr,
        effect_param_map: &HashMap<String, TypeVarId>,
    ) -> Result<(Vec<Type>, Type, Vec<TypeVarId>), TypeError> {
        match &type_sig.node {
            TypeExprKind::Arrow { from, to, .. } => {
                // Collect all type variables used in this operation's signature
                let mut all_vars = Vec::new();
                Self::collect_type_vars(type_sig, &mut all_vars);

                // Find operation-local type vars (not in effect's param_map)
                let mut extended_param_map = effect_param_map.clone();
                let mut op_generic_ids = Vec::new();
                let base_id = effect_param_map.len() as TypeVarId;

                for var_name in &all_vars {
                    if !extended_param_map.contains_key(var_name) {
                        // This is an operation-local polymorphic type variable
                        let id = base_id + op_generic_ids.len() as TypeVarId;
                        extended_param_map.insert(var_name.clone(), id);
                        op_generic_ids.push(id);
                    }
                }

                // Convert from and to to Types using the extended map
                let arg_ty = self.type_expr_to_type(from, &extended_param_map)?;
                let ret_ty = self.type_expr_to_type(to, &extended_param_map)?;

                // If arg is a tuple, extract its components as params
                // Otherwise, single param (Unit counts as a single unit argument)
                let param_types = match arg_ty {
                    Type::Tuple(ts) => ts,
                    other => vec![other],  // Unit is still a param that must be passed
                };

                Ok((param_types, ret_ty, op_generic_ids))
            }
            _ => {
                // Not a function type - this is an error
                Err(TypeError::Other(format!(
                    "Operation signature must be a function type, got {:?}",
                    type_sig.node
                )))
            }
        }
    }

    /// Clear imports for a new module context
    pub fn clear_imports(&mut self) {
        self.imports.clear();
        self.module_aliases.clear();
        self.used_imports.clear();
        self.used_module_aliases.clear();
    }

    /// Get list of unused selective imports
    pub fn get_unused_imports(&self) -> Vec<String> {
        self.imports
            .keys()
            .filter(|name| !self.used_imports.contains(*name))
            .cloned()
            .collect()
    }

    /// Get list of unused module aliases
    pub fn get_unused_module_aliases(&self) -> Vec<String> {
        self.module_aliases
            .keys()
            .filter(|alias| !self.used_module_aliases.contains(*alias))
            .cloned()
            .collect()
    }

    /// Record a type error for later reporting (error accumulation mode)
    fn record_error(&mut self, error: TypeError) {
        self.errors.push(error);
    }

    /// Check if any errors have been accumulated
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Take the accumulated errors, leaving an empty vector
    pub fn take_errors(&mut self) -> Vec<TypeError> {
        std::mem::take(&mut self.errors)
    }

    /// Clear accumulated errors (for starting a new module)
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Get a reference to accumulated errors
    pub fn errors(&self) -> &[TypeError] {
        &self.errors
    }

    /// Look up a name, checking imports and module-qualified names.
    /// Tracks usage for unused import warnings.
    fn lookup_name(&mut self, env: &TypeEnv, name: &str) -> Option<Scheme> {
        // 1. Check local environment first
        if let Some(scheme) = env.get(name) {
            return Some(scheme.clone());
        }

        // 2. Check unqualified imports
        if let Some((module_name, original_name)) = self.imports.get(name).cloned() {
            if let Some(module_env) = self.module_envs.get(&module_name) {
                if let Some(scheme) = module_env.get(&original_name) {
                    // Mark this import as used
                    self.used_imports.insert(name.to_string());
                    return Some(scheme.clone());
                }
            }
        }

        // 3. Check for qualified name (Module.name)
        if let Some(dot_pos) = name.find('.') {
            let (module_part, item_name) = name.split_at(dot_pos);
            let item_name = &item_name[1..]; // skip the dot

            // Resolve module alias if present
            let actual_module = self.module_aliases
                .get(module_part)
                .cloned();

            let module_to_check = actual_module.as_deref().unwrap_or(module_part);

            if let Some(module_env) = self.module_envs.get(module_to_check) {
                if let Some(scheme) = module_env.get(item_name) {
                    // Mark module alias as used if one was used
                    if actual_module.is_some() {
                        self.used_module_aliases.insert(module_part.to_string());
                    }
                    return Some(scheme.clone());
                }
            }
        }

        // 4. Check for qualified name in imported modules (e.g., Response.notFound from Http module)
        // This handles cases where a function has a dot in its name like "Response.notFound"
        if name.contains('.') {
            for module_env in self.module_envs.values() {
                if let Some(scheme) = module_env.get(name) {
                    return Some(scheme.clone());
                }
            }
        }

        None
    }

    /// Generate a fresh type variable
    fn fresh_var(&mut self) -> Type {
        Type::Var(self.type_uf.fresh(self.level))
    }

    /// Generate a fresh row variable for effect polymorphism
    fn fresh_row_var(&mut self) -> Row {
        Row::Var(self.row_uf.fresh(self.level))
    }

    /// Unify two types with source location for error reporting
    fn unify_at(&mut self, t1: &Type, t2: &Type, span: &Span) -> Result<(), TypeError> {
        self.unify_inner(t1, t2, Some(span))
    }

    /// Unify two types (no span - use unify_at when possible)
    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        self.unify_inner(t1, t2, None)
    }

    /// Unify with context for better error messages
    fn unify_with_context(
        &mut self,
        t1: &Type,
        t2: &Type,
        span: &Span,
        context: UnifyContext,
    ) -> Result<(), TypeError> {
        self.unify_inner(t1, t2, Some(span))
            .map_err(|e| e.with_context(context))
    }

    /// Core unification logic with optional span for error reporting
    fn unify_inner(&mut self, t1: &Type, t2: &Type, span: Option<&Span>) -> Result<(), TypeError> {
        let t1 = t1.resolve(&self.type_uf);
        let t2 = t2.resolve(&self.type_uf);

        // Debug output when GNEISS_DEBUG_TYPES is set
        if std::env::var("GNEISS_DEBUG_TYPES").is_ok() {
            eprintln!("[unify] t1={:?}", t1);
            eprintln!("[unify] t2={:?}", t2);
        }

        match (&t1, &t2) {
            // Same primitive types
            (Type::Int, Type::Int) => Ok(()),
            (Type::Float, Type::Float) => Ok(()),
            (Type::Bool, Type::Bool) => Ok(()),
            (Type::String, Type::String) => Ok(()),
            (Type::Char, Type::Char) => Ok(()),
            (Type::Unit, Type::Unit) => Ok(()),
            (Type::Bytes, Type::Bytes) => Ok(()),
            (Type::Pid, Type::Pid) => Ok(()),
            (Type::FileHandle, Type::FileHandle) => Ok(()),
            (Type::TcpSocket, Type::TcpSocket) => Ok(()),
            (Type::TcpListener, Type::TcpListener) => Ok(()),

            // Two type variables - union them
            (Type::Var(id1), Type::Var(id2)) => {
                let root1 = self.type_uf.find(*id1);
                let root2 = self.type_uf.find(*id2);
                if root1 != root2 {
                    self.type_uf.union(root1, root2);
                }
                Ok(())
            }

            // Type variable and concrete type - bind the variable
            (Type::Var(id), other) | (other, Type::Var(id)) => {
                let root = self.type_uf.find(*id);

                // Occurs check
                if other.occurs(root, &self.type_uf) {
                    return Err(TypeError::OccursCheck {
                        var_id: root,
                        ty: other.clone(),
                        span: span.cloned(),
                    });
                }

                // Update levels for let-polymorphism
                let var_level = self.type_uf.get_level(root);
                self.update_levels(other, var_level);

                // Bind the variable
                self.type_uf.bind(root, other.clone());
                Ok(())
            }

            // Generic type variables shouldn't appear during unification
            (Type::Generic(_), _) | (_, Type::Generic(_)) => Err(TypeError::TypeMismatch {
                expected: t1.clone(),
                found: t2.clone(),
                span: span.cloned(),
                context: None,
            }),

            // Function types (arg, ret, and effects must unify)
            (
                Type::Arrow {
                    arg: a1,
                    ret: r1,
                    effects: e1,
                },
                Type::Arrow {
                    arg: a2,
                    ret: r2,
                    effects: e2,
                },
            ) => {
                self.unify_inner(a1, a2, span)?;
                self.unify_inner(r1, r2, span)?;
                self.unify_rows(e1, e2, span)?;
                Ok(())
            }

            // Tuples
            (Type::Tuple(ts1), Type::Tuple(ts2)) if ts1.len() == ts2.len() => {
                for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                    self.unify_inner(t1, t2, span)?;
                }
                Ok(())
            }

            // Channels
            (Type::Channel(t1), Type::Channel(t2)) => self.unify_inner(t1, t2, span),

            // Fibers (typed fiber handles)
            (Type::Fiber(t1), Type::Fiber(t2)) => self.unify_inner(t1, t2, span),

            // Dictionaries (String-keyed maps)
            (Type::Dict(t1), Type::Dict(t2)) => self.unify_inner(t1, t2, span),

            // Sets (String elements)
            (Type::Set, Type::Set) => Ok(()),

            // Named constructors
            (
                Type::Constructor { name: n1, args: a1 },
                Type::Constructor { name: n2, args: a2 },
            ) if n1 == n2 && a1.len() == a2.len() => {
                for (t1, t2) in a1.iter().zip(a2.iter()) {
                    self.unify_inner(t1, t2, span)?;
                }
                Ok(())
            }

            _ => Err(TypeError::TypeMismatch {
                expected: t1,
                found: t2,
                span: span.cloned(),
                context: None,
            }),
        }
    }

    /// Unify two effect rows
    fn unify_rows(&mut self, r1: &Row, r2: &Row, span: Option<&Span>) -> Result<(), TypeError> {
        let r1 = r1.resolve(&self.row_uf);
        let r2 = r2.resolve(&self.row_uf);

        match (&r1, &r2) {
            // Both empty - succeed
            (Row::Empty, Row::Empty) => Ok(()),

            // Two row variables - union them
            (Row::Var(id1), Row::Var(id2)) => {
                let root1 = self.row_uf.find(*id1);
                let root2 = self.row_uf.find(*id2);
                if root1 != root2 {
                    self.row_uf.union(root1, root2);
                }
                Ok(())
            }

            // Row variable and empty - bind to empty
            (Row::Var(id), Row::Empty) | (Row::Empty, Row::Var(id)) => {
                let root = self.row_uf.find(*id);
                self.row_uf.bind(root, Row::Empty);
                Ok(())
            }

            // Row variable and extend - bind var to the extended row
            (Row::Var(id), other @ Row::Extend { .. })
            | (other @ Row::Extend { .. }, Row::Var(id)) => {
                let root = self.row_uf.find(*id);

                // Occurs check for row variable
                if other.occurs(root, &self.row_uf) {
                    return Err(TypeError::Other(
                        "Row occurs check failed: row variable would be infinite".to_string(),
                    ));
                }

                self.row_uf.bind(root, other.clone());
                Ok(())
            }

            // Generic row vars shouldn't appear during unification
            (Row::Generic(_), _) | (_, Row::Generic(_)) => Err(TypeError::Other(
                "Cannot unify generic row variable".to_string(),
            )),

            // Both extend - unify effect params and tails
            (
                Row::Extend {
                    effect: e1,
                    rest: rest1,
                },
                Row::Extend {
                    effect: e2,
                    rest: rest2,
                },
            ) => {
                if e1.name == e2.name {
                    // Same effect - unify parameters
                    if e1.params.len() != e2.params.len() {
                        return Err(TypeError::Other(format!(
                            "Effect {} has mismatched parameter count",
                            e1.name
                        )));
                    }
                    for (p1, p2) in e1.params.iter().zip(&e2.params) {
                        self.unify_inner(p1, p2, span)?;
                    }
                    self.unify_rows(rest1, rest2, span)
                } else {
                    // Different effects - need row rewriting
                    // Try to find e1 in r2's tail
                    self.unify_rows_rewrite(e1, rest1, &r2, span)
                }
            }

            // Empty vs Extend - fail (unless we allow effect subtyping)
            (Row::Empty, Row::Extend { effect, .. })
            | (Row::Extend { effect, .. }, Row::Empty) => Err(TypeError::Other(format!(
                "Effect {} is not handled",
                effect.name
            ))),
        }
    }

    /// Row rewriting: find effect e1 somewhere in r2 and unify the remainders
    fn unify_rows_rewrite(
        &mut self,
        e1: &Effect,
        rest1: &Row,
        r2: &Row,
        span: Option<&Span>,
    ) -> Result<(), TypeError> {
        match r2.resolve(&self.row_uf) {
            Row::Empty => Err(TypeError::Other(format!(
                "Effect {} not found in row",
                e1.name
            ))),
            Row::Var(id) => {
                // Extend r2 with e1 and unify rest1 with new tail
                let root = self.row_uf.find(id);
                let new_tail = self.fresh_row_var();
                let extended = Row::Extend {
                    effect: e1.clone(),
                    rest: Rc::new(new_tail.clone()),
                };
                self.row_uf.bind(root, extended);
                self.unify_rows(rest1, &new_tail, span)
            }
            Row::Generic(_) => Err(TypeError::Other(
                "Cannot unify generic row variable".to_string(),
            )),
            Row::Extend { effect: e2, rest: rest2 } => {
                if e1.name == e2.name {
                    // Found it - unify parameters and tails
                    if e1.params.len() != e2.params.len() {
                        return Err(TypeError::Other(format!(
                            "Effect {} has mismatched parameter count",
                            e1.name
                        )));
                    }
                    for (p1, p2) in e1.params.iter().zip(&e2.params) {
                        self.unify_inner(p1, p2, span)?;
                    }
                    self.unify_rows(rest1, &rest2, span)
                } else {
                    // Keep looking - e2 stays, we look in rest2
                    self.unify_rows_rewrite(e1, rest1, &rest2, span)
                }
            }
        }
    }

    /// Union two effect rows, combining their effects.
    ///
    /// This is used when combining effects from multiple expressions,
    /// such as in function application where we need to combine:
    /// - Effects from evaluating the function
    /// - Effects from evaluating the argument
    /// - Latent effects in the function's arrow type
    ///
    /// The resulting row contains all effects from both inputs.
    /// Row variables are preserved with fresh tails for polymorphism.
    pub fn union_rows(&mut self, r1: &Row, r2: &Row) -> Row {
        let r1 = r1.resolve(&self.row_uf);
        let r2 = r2.resolve(&self.row_uf);

        match (&r1, &r2) {
            // Empty is identity for union
            (Row::Empty, _) => r2.clone(),
            (_, Row::Empty) => r1.clone(),

            // Extend: prepend effect and recurse
            (Row::Extend { effect, rest }, other) => Row::Extend {
                effect: effect.clone(),
                rest: Rc::new(self.union_rows(rest, other)),
            },

            // Two row variables: create a fresh var that represents their union
            // The actual constraints will be resolved during later unification
            (Row::Var(id1), Row::Var(id2)) => {
                let root1 = self.row_uf.find(*id1);
                let root2 = self.row_uf.find(*id2);
                if root1 == root2 {
                    // Same variable, just return it
                    r1.clone()
                } else {
                    // Different variables - we need to be careful here
                    // For now, create a fresh var and leave the constraint implicit
                    // This is sound but may not be most general
                    self.fresh_row_var()
                }
            }

            // Generic row variables - just return the other or a fresh var
            (Row::Generic(_), other) | (other, Row::Generic(_)) => other.clone(),

            // Known row vs variable: variable's effects plus the known effects
            // We extend the known row with a fresh tail to preserve polymorphism
            (Row::Var(_), Row::Extend { .. }) => {
                // r2 has known effects, r1 is variable
                // Result should include r2's effects plus whatever r1 adds
                // For simplicity, we use r2 and let unification handle constraints
                r2.clone()
            }
        }
    }

    /// Subtract a set of handled effects from a row.
    ///
    /// Used by `handle` to remove the effects it handles, leaving
    /// the remaining (unhandled) effects in the result.
    ///
    /// Returns the row with handled effects removed.
    pub fn subtract_effects(&mut self, row: &Row, handled: &HashSet<String>) -> Row {
        let row = row.resolve(&self.row_uf);

        match row {
            Row::Empty => Row::Empty,
            Row::Var(id) => {
                // Row variable - we can't remove effects statically
                // Create a fresh variable for the result
                // The constraint that it doesn't contain handled effects
                // is implicit and should be checked elsewhere
                Row::Var(id)
            }
            Row::Generic(id) => Row::Generic(id),
            Row::Extend { effect, rest } => {
                if handled.contains(&effect.name) {
                    // This effect is handled - remove it
                    self.subtract_effects(&rest, handled)
                } else {
                    // Keep this effect
                    Row::Extend {
                        effect: effect.clone(),
                        rest: Rc::new(self.subtract_effects(&rest, handled)),
                    }
                }
            }
        }
    }

    /// Update type variable levels for let-polymorphism
    fn update_levels(&mut self, ty: &Type, level: u32) {
        match ty {
            Type::Var(id) => {
                self.type_uf.set_level(*id, level);
            }
            Type::Generic(_) => {}
            Type::Arrow { arg, ret, effects } => {
                self.update_levels(arg, level);
                self.update_levels(ret, level);
                self.update_levels_in_row(effects, level);
            }
            Type::Tuple(ts) => {
                for t in ts {
                    self.update_levels(t, level);
                }
            }
            Type::Channel(t) => self.update_levels(t, level),
            Type::Fiber(t) => self.update_levels(t, level),
            Type::Dict(t) => self.update_levels(t, level),
            Type::Constructor { args, .. } => {
                for t in args {
                    self.update_levels(t, level);
                }
            }
            _ => {}
        }
    }

    /// Update row variable levels for let-polymorphism
    fn update_levels_in_row(&mut self, row: &Row, level: u32) {
        match row {
            Row::Var(id) => {
                self.row_uf.set_level(*id, level);
            }
            Row::Generic(_) => {}
            Row::Extend { effect, rest } => {
                for param in &effect.params {
                    self.update_levels(param, level);
                }
                self.update_levels_in_row(rest, level);
            }
            Row::Empty => {}
        }
    }

    /// Instantiate a polymorphic type scheme, adding predicates to wanted_preds
    fn instantiate(&mut self, scheme: &Scheme) -> Type {
        if scheme.num_generics == 0 && scheme.predicates.is_empty() {
            return scheme.ty.clone();
        }

        let mut substitution: HashMap<TypeVarId, Type> = HashMap::new();
        for i in 0..scheme.num_generics {
            substitution.insert(i, self.fresh_var());
        }

        // Instantiate predicates and add to wanted_preds
        for pred in &scheme.predicates {
            let instantiated_pred = Pred {
                trait_name: pred.trait_name.clone(),
                ty: self.substitute(&pred.ty, &substitution),
            };
            self.wanted_preds.push(instantiated_pred);
        }

        self.substitute(&scheme.ty, &substitution)
    }

    /// Substitute generic type variables
    fn substitute(&self, ty: &Type, subst: &HashMap<TypeVarId, Type>) -> Type {
        match ty {
            Type::Generic(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
            Type::Var(id) => Type::Var(*id), // Regular type variables pass through
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(self.substitute(arg, subst)),
                ret: Rc::new(self.substitute(ret, subst)),
                effects: self.substitute_in_row(effects, subst),
            },
            Type::Tuple(ts) => Type::Tuple(ts.iter().map(|t| self.substitute(t, subst)).collect()),
            Type::Channel(t) => Type::Channel(Rc::new(self.substitute(t, subst))),
            Type::Fiber(t) => Type::Fiber(Rc::new(self.substitute(t, subst))),
            Type::Dict(t) => Type::Dict(Rc::new(self.substitute(t, subst))),
            Type::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args.iter().map(|t| self.substitute(t, subst)).collect(),
            },
            other => other.clone(),
        }
    }

    /// Substitute generic type variables in an effect row
    fn substitute_in_row(&self, row: &Row, subst: &HashMap<TypeVarId, Type>) -> Row {
        match row {
            Row::Empty => Row::Empty,
            Row::Var(id) => Row::Var(*id), // Row variables not affected by type substitution
            Row::Generic(id) => Row::Generic(*id), // Generic row vars pass through
            Row::Extend { effect, rest } => Row::Extend {
                effect: Effect {
                    name: effect.name.clone(),
                    params: effect
                        .params
                        .iter()
                        .map(|t| self.substitute(t, subst))
                        .collect(),
                },
                rest: Rc::new(self.substitute_in_row(rest, subst)),
            },
        }
    }

    /// Generalize a type to a polymorphic scheme, capturing relevant predicates
    fn generalize(&mut self, ty: &Type) -> Scheme {
        let mut generics: HashMap<TypeVarId, TypeVarId> = HashMap::new();
        let generalized = self.generalize_inner(ty, &mut generics);

        // Take all predicates, then partition them
        let all_preds: Vec<_> = std::mem::take(&mut self.wanted_preds);

        let (captured, remaining): (Vec<_>, Vec<_>) = all_preds
            .into_iter()
            .partition(|pred| Self::pred_mentions_generalized_vars_static(pred, &generics, self.level, &self.type_uf, &self.row_uf));

        self.wanted_preds = remaining;

        // Convert captured predicates to use Generic type vars
        let preds = captured
            .into_iter()
            .map(|pred| Self::apply_generic_subst_to_pred_static(&pred, &generics, self.level, &self.type_uf, &self.row_uf))
            .collect();

        Scheme {
            num_generics: generics.len() as u32,
            predicates: preds,
            ty: generalized,
        }
    }

    /// Collect level-0 "placeholder" type vars from a type and add fresh var substitutions
    /// These are created by Type::arrow() and need to be replaced to avoid sharing across method calls
    fn collect_level_zero_vars(ty: &Type, subst: &mut HashMap<TypeVarId, Type>, inferencer: &mut Inferencer) {
        match ty.resolve(&inferencer.type_uf) {
            Type::Var(id) => {
                // Check if this is an unbound level-0 variable
                if !inferencer.type_uf.is_bound(id) && inferencer.type_uf.get_level(id) == 0 {
                    // Level-0 var that's not already in subst - create fresh var for it
                    subst.entry(id).or_insert_with(|| inferencer.fresh_var());
                }
            }
            Type::Arrow { arg, ret, effects } => {
                Self::collect_level_zero_vars(&arg, subst, inferencer);
                Self::collect_level_zero_vars(&ret, subst, inferencer);
                Self::collect_level_zero_vars_in_row(&effects, subst, inferencer);
            }
            Type::Tuple(ts) => {
                for t in ts.iter() {
                    Self::collect_level_zero_vars(t, subst, inferencer);
                }
            }
            Type::Constructor { args, .. } => {
                for t in args.iter() {
                    Self::collect_level_zero_vars(t, subst, inferencer);
                }
            }
            Type::Channel(t) | Type::Fiber(t) => {
                Self::collect_level_zero_vars(&t, subst, inferencer);
            }
            _ => {}
        }
    }

    /// Collect level-0 type variables in an effect row
    fn collect_level_zero_vars_in_row(
        row: &Row,
        subst: &mut HashMap<TypeVarId, Type>,
        inferencer: &mut Inferencer,
    ) {
        match row.resolve(&inferencer.row_uf) {
            Row::Empty => {}
            Row::Var(_) => {} // Row vars are separate from type vars
            Row::Extend { effect, rest } => {
                for param in &effect.params {
                    Self::collect_level_zero_vars(param, subst, inferencer);
                }
                Self::collect_level_zero_vars_in_row(&rest, subst, inferencer);
            }
            Row::Generic(_) => {} // Generic row vars are also separate
        }
    }

    /// Check if a predicate mentions any of the generalized type variables (static version)
    fn pred_mentions_generalized_vars_static(
        pred: &Pred,
        generics: &HashMap<TypeVarId, TypeVarId>,
        current_level: u32,
        type_uf: &UnionFind,
        row_uf: &RowUnionFind,
    ) -> bool {
        Self::type_mentions_generalized_vars_static(&pred.ty, generics, current_level, type_uf, row_uf)
    }

    /// Check if a type contains any of the generalized type variables (static version)
    fn type_mentions_generalized_vars_static(
        ty: &Type,
        generics: &HashMap<TypeVarId, TypeVarId>,
        current_level: u32,
        type_uf: &UnionFind,
        row_uf: &RowUnionFind,
    ) -> bool {
        let resolved = ty.resolve(type_uf);
        match &resolved {
            Type::Var(id) => {
                // Check if this is an unbound variable with level > current_level
                if !type_uf.is_bound(*id) && type_uf.get_level(*id) > current_level {
                    generics.contains_key(id)
                } else {
                    false
                }
            }
            Type::Arrow { arg, ret, effects } => {
                Self::type_mentions_generalized_vars_static(arg, generics, current_level, type_uf, row_uf)
                    || Self::type_mentions_generalized_vars_static(ret, generics, current_level, type_uf, row_uf)
                    || Self::row_mentions_generalized_vars_static(effects, generics, current_level, type_uf, row_uf)
            }
            Type::Tuple(ts) => ts
                .iter()
                .any(|t| Self::type_mentions_generalized_vars_static(t, generics, current_level, type_uf, row_uf)),
            Type::Channel(t) | Type::Fiber(t) => {
                Self::type_mentions_generalized_vars_static(t, generics, current_level, type_uf, row_uf)
            }
            Type::Constructor { args, .. } => args
                .iter()
                .any(|t| Self::type_mentions_generalized_vars_static(t, generics, current_level, type_uf, row_uf)),
            _ => false,
        }
    }

    /// Check if a row mentions any generalized type variables
    fn row_mentions_generalized_vars_static(
        row: &Row,
        generics: &HashMap<TypeVarId, TypeVarId>,
        current_level: u32,
        type_uf: &UnionFind,
        row_uf: &RowUnionFind,
    ) -> bool {
        match row.resolve(row_uf) {
            Row::Empty => false,
            Row::Var(_) => false, // Row vars are separate
            Row::Extend { effect, rest } => {
                effect
                    .params
                    .iter()
                    .any(|t| Self::type_mentions_generalized_vars_static(t, generics, current_level, type_uf, row_uf))
                    || Self::row_mentions_generalized_vars_static(&rest, generics, current_level, type_uf, row_uf)
            }
            Row::Generic(_) => false, // Generic row vars are separate
        }
    }

    /// Apply generic substitution to a predicate (static version)
    fn apply_generic_subst_to_pred_static(
        pred: &Pred,
        generics: &HashMap<TypeVarId, TypeVarId>,
        current_level: u32,
        type_uf: &UnionFind,
        row_uf: &RowUnionFind,
    ) -> Pred {
        Pred {
            trait_name: pred.trait_name.clone(),
            ty: Self::apply_generic_subst_to_type_static(&pred.ty, generics, current_level, type_uf, row_uf),
        }
    }

    /// Apply generic substitution to a type (static version)
    fn apply_generic_subst_to_type_static(
        ty: &Type,
        generics: &HashMap<TypeVarId, TypeVarId>,
        current_level: u32,
        type_uf: &UnionFind,
        row_uf: &RowUnionFind,
    ) -> Type {
        let resolved = ty.resolve(type_uf);
        match &resolved {
            Type::Var(id) => {
                // Check if this is an unbound variable with level > current_level
                if !type_uf.is_bound(*id) && type_uf.get_level(*id) > current_level {
                    if let Some(&gen_id) = generics.get(id) {
                        Type::Generic(gen_id)
                    } else {
                        resolved.clone()
                    }
                } else {
                    resolved.clone()
                }
            }
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(Self::apply_generic_subst_to_type_static(
                    arg,
                    generics,
                    current_level,
                    type_uf,
                    row_uf,
                )),
                ret: Rc::new(Self::apply_generic_subst_to_type_static(
                    ret,
                    generics,
                    current_level,
                    type_uf,
                    row_uf,
                )),
                effects: Self::apply_generic_subst_to_row_static(effects, generics, current_level, type_uf, row_uf),
            },
            Type::Tuple(ts) => Type::Tuple(
                ts.iter()
                    .map(|t| Self::apply_generic_subst_to_type_static(t, generics, current_level, type_uf, row_uf))
                    .collect(),
            ),
            Type::Channel(t) => Type::Channel(Rc::new(Self::apply_generic_subst_to_type_static(
                t,
                generics,
                current_level,
                type_uf,
                row_uf,
            ))),
            Type::Fiber(t) => Type::Fiber(Rc::new(Self::apply_generic_subst_to_type_static(
                t,
                generics,
                current_level,
                type_uf,
                row_uf,
            ))),
            Type::Dict(t) => Type::Dict(Rc::new(Self::apply_generic_subst_to_type_static(
                t,
                generics,
                current_level,
                type_uf,
                row_uf,
            ))),
            Type::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|t| Self::apply_generic_subst_to_type_static(t, generics, current_level, type_uf, row_uf))
                    .collect(),
            },
            _ => resolved.clone(),
        }
    }

    /// Apply generic substitution to an effect row (static version)
    fn apply_generic_subst_to_row_static(
        row: &Row,
        generics: &HashMap<TypeVarId, TypeVarId>,
        current_level: u32,
        type_uf: &UnionFind,
        row_uf: &RowUnionFind,
    ) -> Row {
        match row.resolve(row_uf) {
            Row::Empty => Row::Empty,
            Row::Var(id) => Row::Var(id), // Row vars not affected by type generics
            Row::Extend { effect, rest } => Row::Extend {
                effect: Effect {
                    name: effect.name.clone(),
                    params: effect
                        .params
                        .iter()
                        .map(|t| Self::apply_generic_subst_to_type_static(t, generics, current_level, type_uf, row_uf))
                        .collect(),
                },
                rest: Rc::new(Self::apply_generic_subst_to_row_static(
                    &rest,
                    generics,
                    current_level,
                    type_uf,
                    row_uf,
                )),
            },
            Row::Generic(id) => Row::Generic(id), // Generic row vars not affected by type generics
        }
    }

    fn generalize_inner(&self, ty: &Type, generics: &mut HashMap<TypeVarId, TypeVarId>) -> Type {
        let resolved = ty.resolve(&self.type_uf);
        match &resolved {
            Type::Var(id) => {
                // Check if this is an unbound variable with level > self.level
                if !self.type_uf.is_bound(*id) && self.type_uf.get_level(*id) > self.level {
                    let gen_id = if let Some(&gen_id) = generics.get(id) {
                        gen_id
                    } else {
                        let gen_id = generics.len() as TypeVarId;
                        generics.insert(*id, gen_id);
                        gen_id
                    };
                    Type::Generic(gen_id)
                } else {
                    resolved.clone()
                }
            }
            Type::Arrow { arg, ret, effects } => Type::Arrow {
                arg: Rc::new(self.generalize_inner(arg, generics)),
                ret: Rc::new(self.generalize_inner(ret, generics)),
                effects: self.generalize_inner_row(effects, generics),
            },
            Type::Tuple(ts) => Type::Tuple(
                ts.iter()
                    .map(|t| self.generalize_inner(t, generics))
                    .collect(),
            ),
            Type::Channel(t) => Type::Channel(Rc::new(self.generalize_inner(t, generics))),
            Type::Fiber(t) => Type::Fiber(Rc::new(self.generalize_inner(t, generics))),
            Type::Dict(t) => Type::Dict(Rc::new(self.generalize_inner(t, generics))),
            Type::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|t| self.generalize_inner(t, generics))
                    .collect(),
            },
            _ => resolved.clone(),
        }
    }

    /// Generalize type variables in an effect row
    fn generalize_inner_row(&self, row: &Row, generics: &mut HashMap<TypeVarId, TypeVarId>) -> Row {
        match row.resolve(&self.row_uf) {
            Row::Empty => Row::Empty,
            Row::Var(id) => Row::Var(id), // Row vars generalize separately (not implemented yet)
            Row::Extend { effect, rest } => Row::Extend {
                effect: Effect {
                    name: effect.name.clone(),
                    params: effect
                        .params
                        .iter()
                        .map(|t| self.generalize_inner(t, generics))
                        .collect(),
                },
                rest: Rc::new(self.generalize_inner_row(&rest, generics)),
            },
            Row::Generic(id) => Row::Generic(id), // Already generalized
        }
    }

    /// Check if an expression is a syntactic value (for the value restriction).
    /// Only syntactic values can be generalized in let-bindings.
    /// This prevents unsound polymorphism for effectful expressions like Channel.new.
    fn is_syntactic_value(expr: &Expr) -> bool {
        match &expr.node {
            // Literals are values
            ExprKind::Lit(_) => true,
            // Variables are values (they refer to already-computed values)
            ExprKind::Var(_) => true,
            // Lambdas are values (they don't execute until applied)
            ExprKind::Lambda { .. } => true,
            // Constructors with value arguments are values
            ExprKind::Constructor { args, .. } => args.iter().all(Self::is_syntactic_value),
            // Tuples of values are values
            ExprKind::Tuple(exprs) => exprs.iter().all(Self::is_syntactic_value),
            // Lists of values are values
            ExprKind::List(exprs) => exprs.iter().all(Self::is_syntactic_value),
            // Everything else (applications, channel ops, etc.) is not a value
            _ => false,
        }
    }

    /// Extract function name from an application for error context.
    /// For curried applications like `f a b`, returns ("f", 2) for the innermost arg.
    /// Returns (Some(name), arg_position) where position counts from 1.
    fn extract_func_info(expr: &Expr) -> (Option<String>, usize) {
        fn go(e: &Expr, depth: usize) -> (Option<String>, usize) {
            match &e.node {
                ExprKind::Var(name) => (Some(name.clone()), depth),
                ExprKind::App { func, .. } => go(func, depth + 1),
                _ => (None, depth),
            }
        }
        go(expr, 1)
    }

    /// Infer the type of an expression (compatibility wrapper).
    /// This is the public API that returns just the type.
    /// Internally uses effect tracking for algebraic effects.
    pub fn infer_expr(&mut self, env: &TypeEnv, expr: &Expr) -> Result<Type, TypeError> {
        let result = self.infer_expr_full(env, expr)?;
        // Resolve the type to follow union-find bindings
        Ok(result.ty.resolve(&self.type_uf))
    }

    /// Infer a single expression with full effect tracking.
    /// Returns InferResult with ty and effects row.
    /// Pure expressions have an empty effect row (Row::Empty).
    /// Effectful expressions accumulate effects in the row.
    pub fn infer_expr_full(
        &mut self,
        env: &TypeEnv,
        expr: &Expr,
    ) -> Result<InferResult, TypeError> {
        match &expr.node {
            // Literals are pure
            ExprKind::Lit(lit) => {
                let ty = self.infer_literal(lit);
                Ok(InferResult::pure(ty))
            }

            // Variables are pure
            ExprKind::Var(name) => {
                // Try module-aware lookup first (includes local env, imports, and qualified names)
                let ty = if let Some(scheme) = self.lookup_name(env, name) {
                    self.instantiate(&scheme)
                } else if let Some((trait_name, method_ty)) = self.class_env.lookup_method(name) {
                    // Clone to avoid borrow conflicts
                    let trait_name = trait_name.to_string();
                    let method_ty = method_ty.clone();

                    // This is a trait method - instantiate with fresh vars and add predicate
                    let fresh_ty = self.fresh_var();

                    // The method type has Generic(0) for the trait's type param
                    // It may also have level-0 "placeholder" vars for answer types from Type::arrow
                    // We need to substitute both:
                    // 1. Generic(0) -> fresh type var for the trait param
                    // 2. Level-0 unbound vars -> fresh answer type vars
                    let mut subst = HashMap::new();
                    subst.insert(0, fresh_ty.clone());

                    // Find and replace level-0 placeholder vars (created by Type::arrow)
                    // These have level 0 and need fresh vars to avoid sharing across different method calls
                    Self::collect_level_zero_vars(&method_ty, &mut subst, self);

                    let instantiated = apply_subst(&method_ty, &subst);

                    // Add a wanted predicate: TraitName fresh_ty
                    self.wanted_preds.push(Pred::new(trait_name, fresh_ty));

                    instantiated
                } else {
                    // Collect suggestions for "did you mean?"
                    let mut candidates: Vec<&str> = env.keys().map(|s| s.as_str()).collect();
                    // Also suggest imported names
                    for name in self.imports.keys() {
                        candidates.push(name.as_str());
                    }
                    let suggestions = find_similar(name, candidates, 2);
                    return Err(TypeError::UnboundVariable {
                        name: name.clone(),
                        span: expr.span.clone(),
                        suggestions,
                    });
                };
                Ok(InferResult::pure(ty))
            }

            // Lambda: building a closure is PURE (evaluating lambda doesn't run body)
            // But the function type captures the body's latent answer-type effects
            //
            // Rule:
            //   Γ, x : σ; α ⊢ e : τ; β
            //   ─────────────────────────
            //   Γ ⊢ₚ λx.e : (σ/α → τ/β)   -- lambda itself is pure
            ExprKind::Lambda { params, body } => {
                let mut current_env = env.clone();
                let mut param_types = Vec::new();

                for param in params {
                    let param_ty = self.fresh_var();
                    self.bind_pattern(&mut current_env, param, &param_ty)?;
                    param_types.push(param_ty);
                }

                // Infer body with full answer-type tracking
                let body_result = self.infer_expr_full(&current_env, body)?;

                // Build the function type
                // For multi-param lambdas, fold from right: a -> b -> c becomes a -> (b -> c)
                // TODO: Track effects properly once effect inference is implemented
                let mut func_ty = Type::Arrow {
                    arg: Rc::new(param_types.pop().unwrap()),
                    ret: Rc::new(body_result.ty),
                    effects: Row::Empty, // Pure for now - effect inference will fill this in
                };

                // Add remaining params as pure intermediate arrows
                for param_ty in param_types.into_iter().rev() {
                    func_ty = Type::Arrow {
                        arg: Rc::new(param_ty),
                        ret: Rc::new(func_ty),
                        effects: Row::Empty, // Pure - returning a function doesn't execute effects
                    };
                }

                // Lambda ITSELF is pure (creating closure doesn't run body)
                Ok(InferResult::pure(func_ty))
            }

            // Application: effect tracking
            //
            // Rule (Koka-style):
            //   Γ ⊢ e₁ : (σ → τ ! ε₁) ! ε₂    -- function with latent effects ε₁
            //   Γ ⊢ e₂ : σ ! ε₃                -- argument
            //   ────────────────────────────────
            //   Γ ⊢ e₁ e₂ : τ ! ε₁ ∪ ε₂ ∪ ε₃  -- combined effects
            //
            ExprKind::App { func, arg } => {
                // Extract function name for better error messages
                let (func_name, param_num) = Self::extract_func_info(func);

                // Infer function and argument types
                let fun_result = self.infer_expr_full(env, func)?;
                let arg_result = self.infer_expr_full(env, arg)?;

                // Fresh variables for function type components
                let param_ty = self.fresh_var(); // σ
                let ret_ty = self.fresh_var(); // τ
                let latent_effects = self.fresh_row_var();

                // Function must have type σ → τ { latent_effects }
                let expected_fun = Type::Arrow {
                    arg: Rc::new(param_ty.clone()),
                    ret: Rc::new(ret_ty.clone()),
                    effects: latent_effects.clone(),
                };
                self.unify_at(&fun_result.ty, &expected_fun, &func.span)?;

                // Argument must have type σ (with context for better errors)
                self.unify_with_context(
                    &param_ty,
                    &arg_result.ty,
                    &arg.span,
                    UnifyContext::FunctionArgument {
                        func_name,
                        param_num,
                    },
                )?;

                // Combined effects = fun.effects ∪ arg.effects ∪ latent_effects
                let combined = self.union_rows(&fun_result.effects, &arg_result.effects);
                let combined = self.union_rows(&combined, &latent_effects);

                Ok(InferResult::with_effects(ret_ty, combined))
            }

            // Let binding with effect tracking
            //
            // Purity restriction: Only pure values can be generalized
            ExprKind::Let {
                pattern,
                value,
                body,
            } => {
                // Check if this is a recursive function binding:
                // let name = fun ... -> ... in body
                // If so, we need to put 'name' in scope before inferring the lambda
                let is_recursive_fn = matches!(
                    (&pattern.node, &value.node),
                    (PatternKind::Var(_), ExprKind::Lambda { .. })
                );

                // Enter a new level for let-polymorphism
                self.level += 1;

                let value_result = if is_recursive_fn {
                    // For recursive functions, add the name to env with a fresh type
                    // before inferring the lambda body
                    if let PatternKind::Var(name) = &pattern.node {
                        let mut recursive_env = env.clone();
                        let preliminary_ty = self.fresh_var();
                        recursive_env.insert(name.clone(), Scheme::mono(preliminary_ty.clone()));

                        let inferred_result = self.infer_expr_full(&recursive_env, value)?;
                        self.unify_at(&preliminary_ty, &inferred_result.ty, &value.span)?;
                        inferred_result
                    } else {
                        unreachable!()
                    }
                } else {
                    self.infer_expr_full(env, value)?
                };

                self.level -= 1;

                // Check if binding is pure (no effects)
                // Only pure expressions can be generalized
                let is_pure = value_result.is_pure(&self.row_uf);

                // Value restriction: only generalize syntactic values that are also pure
                let scheme = if Self::is_syntactic_value(value) && is_pure {
                    self.generalize(&value_result.ty)
                } else {
                    // Don't generalize - keep the monomorphic type
                    Scheme {
                        num_generics: 0,
                        predicates: vec![],
                        ty: value_result.ty.clone(),
                    }
                };

                // Bind the pattern
                let mut new_env = env.clone();
                self.bind_pattern_scheme(&mut new_env, pattern, scheme)?;

                if let Some(body) = body {
                    let body_result = self.infer_expr_full(&new_env, body)?;

                    // Combined effects = value.effects ∪ body.effects
                    let combined = self.union_rows(&value_result.effects, &body_result.effects);

                    Ok(InferResult::with_effects(body_result.ty, combined))
                } else {
                    // No body - just a declaration
                    Ok(InferResult::with_effects(Type::Unit, value_result.effects))
                }
            }

            // Mutually recursive let bindings: let rec f = ... and g = ... in body
            ExprKind::LetRec { bindings, body } => {
                // Enter a new level for let-polymorphism
                self.level += 1;

                // Step 1: Create preliminary types for all bindings
                // This allows mutual references
                // If there's a pre-existing val declaration, use that type to help break
                // infinite type cycles in answer types
                let mut rec_env = env.clone();
                let mut preliminary_types: Vec<Type> = Vec::new();

                for binding in bindings {
                    // Check if there's an existing type signature from a val declaration
                    let preliminary_ty = if let Some(scheme) = env.get(&binding.name.node) {
                        // Use the declared type as the preliminary type
                        // This helps break infinite type cycles for recursive functions
                        self.instantiate(scheme)
                    } else {
                        // No declaration, use fresh variable
                        self.fresh_var()
                    };
                    preliminary_types.push(preliminary_ty.clone());
                    rec_env.insert(binding.name.node.clone(), Scheme::mono(preliminary_ty));
                }

                // Step 2: Infer each binding's body using the recursive environment
                let mut value_results: Vec<InferResult> = Vec::new();
                for binding in bindings {
                    // Build a lambda type for function with params
                    let mut body_env = rec_env.clone();
                    let mut param_types = Vec::new();

                    for param in &binding.params {
                        let param_ty = self.fresh_var();
                        self.bind_pattern(&mut body_env, param, &param_ty)?;
                        param_types.push(param_ty);
                    }

                    let body_result = self.infer_expr_full(&body_env, &binding.body)?;

                    // Build the function type
                    let func_ty = if param_types.is_empty() {
                        body_result.ty.clone()
                    } else {
                        // Build arrow type from right to left
                        // TODO: Track effects properly once effect inference is implemented
                        let mut func_ty = Type::Arrow {
                            arg: Rc::new(param_types.pop().unwrap()),
                            ret: Rc::new(body_result.ty.clone()),
                            effects: Row::Empty, // Pure for now
                        };

                        while let Some(param_ty) = param_types.pop() {
                            func_ty = Type::Arrow {
                                arg: Rc::new(param_ty),
                                ret: Rc::new(func_ty),
                                effects: Row::Empty, // Pure
                            };
                        }
                        func_ty
                    };

                    value_results.push(InferResult::with_effects(func_ty, body_result.effects));
                }

                // Step 3: Unify preliminary types with inferred types
                for (i, (binding, result)) in bindings.iter().zip(value_results.iter()).enumerate()
                {
                    self.unify_with_context(
                        &preliminary_types[i],
                        &result.ty,
                        &binding.name.span,
                        UnifyContext::RecursiveBinding {
                            name: binding.name.node.clone(),
                        },
                    )?;
                }

                self.level -= 1;

                // Step 4: Generalize and bind in the outer environment
                let mut new_env = env.clone();
                for (i, binding) in bindings.iter().enumerate() {
                    let scheme = self.generalize(&preliminary_types[i]);
                    new_env.insert(binding.name.node.clone(), scheme);
                }

                // Step 5: Infer body if present
                if let Some(body) = body {
                    let body_result = self.infer_expr_full(&new_env, body)?;
                    Ok(body_result)
                } else {
                    // No body - just declarations, pure
                    Ok(InferResult::pure(Type::Unit))
                }
            }

            // If: combine effects from condition and branches
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_result = self.infer_expr_full(env, cond)?;
                self.unify_with_context(
                    &cond_result.ty,
                    &Type::Bool,
                    &cond.span,
                    UnifyContext::IfCondition,
                )?;

                let then_result = self.infer_expr_full(env, then_branch)?;
                let else_result = self.infer_expr_full(env, else_branch)?;

                // Branches must have same type
                self.unify_with_context(
                    &then_result.ty,
                    &else_result.ty,
                    &else_branch.span,
                    UnifyContext::IfBranches,
                )?;

                // Combined effects: cond ∪ then ∪ else
                // (Only one branch executes, but we're conservative)
                let branch_effects = self.union_rows(&then_result.effects, &else_result.effects);
                let combined = self.union_rows(&cond_result.effects, &branch_effects);

                Ok(InferResult::with_effects(then_result.ty, combined))
            }

            // Match: combine effects from scrutinee and all arms
            ExprKind::Match { scrutinee, arms } => {
                let scrutinee_result = self.infer_expr_full(env, scrutinee)?;
                let result_ty = self.fresh_var();
                let mut combined_effects = scrutinee_result.effects;

                for arm in arms {
                    let mut arm_env = env.clone();
                    self.bind_pattern(&mut arm_env, &arm.pattern, &scrutinee_result.ty)?;

                    // Handle guard if present
                    if let Some(guard) = &arm.guard {
                        let guard_result = self.infer_expr_full(&arm_env, guard)?;
                        self.unify_at(&guard_result.ty, &Type::Bool, &guard.span)?;
                        combined_effects = self.union_rows(&combined_effects, &guard_result.effects);
                    }

                    let body_result = self.infer_expr_full(&arm_env, &arm.body)?;
                    self.unify_with_context(
                        &result_ty,
                        &body_result.ty,
                        &arm.body.span,
                        UnifyContext::MatchArms,
                    )?;

                    combined_effects = self.union_rows(&combined_effects, &body_result.effects);
                }

                Ok(InferResult::with_effects(result_ty, combined_effects))
            }

            // Tuple: pure (empty tuple normalizes to Unit)
            ExprKind::Tuple(exprs) => {
                let types: Result<Vec<_>, _> =
                    exprs.iter().map(|e| self.infer_expr(env, e)).collect();
                let types = types?;
                let ty = if types.is_empty() {
                    Type::Unit
                } else {
                    Type::Tuple(types)
                };
                Ok(InferResult::pure(ty))
            }

            // List: pure
            ExprKind::List(exprs) => {
                let elem_ty = self.fresh_var();
                for e in exprs {
                    let ty = self.infer_expr(env, e)?;
                    self.unify_at(&elem_ty, &ty, &e.span)?;
                }
                Ok(InferResult::pure(Type::list(elem_ty)))
            }

            // Constructor: pure
            ExprKind::Constructor { name, args } => {
                if let Some(info) = self.type_ctx.get_constructor(name).cloned() {
                    // Create fresh type variables for the type parameters
                    let mut type_args = Vec::new();
                    for _ in 0..info.type_params {
                        type_args.push(self.fresh_var());
                    }

                    // Substitute type parameters in field types
                    let mut subst: HashMap<TypeVarId, Type> = HashMap::new();
                    for (i, ty) in type_args.iter().enumerate() {
                        subst.insert(i as TypeVarId, ty.clone());
                    }

                    // The result type of the constructor
                    let result_ty = Type::Constructor {
                        name: info.type_name.clone(),
                        args: type_args.clone(),
                    };

                    // If no args provided but constructor has fields, return a curried function type
                    // e.g., Some : a -> Option a
                    if args.is_empty() && !info.field_types.is_empty() {
                        // Build curried function type: field1 -> field2 -> ... -> ResultType
                        // Each intermediate arrow is pure (same answer type)
                        let mut ty = result_ty;
                        for field_ty in info.field_types.iter().rev() {
                            let param_ty = self.substitute(field_ty, &subst);
                            ty = Type::Arrow {
                                arg: Rc::new(param_ty),
                                ret: Rc::new(ty),
                                effects: Row::Empty, // Pure
                            };
                        }
                        return Ok(InferResult::pure(ty));
                    }

                    // Check field types against provided arguments
                    if args.len() != info.field_types.len() {
                        return Err(TypeError::TypeMismatch {
                            expected: result_ty,
                            found: Type::Unit,
                            span: Some(expr.span.clone()),
                            context: None,
                        });
                    }

                    for (arg, field_ty) in args.iter().zip(&info.field_types) {
                        let arg_ty = self.infer_expr(env, arg)?;
                        let expected_ty = self.substitute(field_ty, &subst);
                        self.unify_at(&arg_ty, &expected_ty, &arg.span)?;
                    }

                    Ok(InferResult::pure(result_ty))
                } else {
                    // Unknown constructor - gather suggestions from known constructors
                    let candidates: Vec<&str> = self.type_ctx.constructor_names().collect();
                    let suggestions = find_similar(name, candidates, 2);
                    Err(TypeError::UnknownConstructor {
                        name: name.clone(),
                        span: expr.span.clone(),
                        suggestions,
                    })
                }
            }

            // BinOp: combine effects from both operands
            ExprKind::BinOp { op, left, right } => {
                let left_result = self.infer_expr_full(env, left)?;
                let right_result = self.infer_expr_full(env, right)?;

                let result_ty = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        // Arithmetic works on Int or Float (ad-hoc polymorphism)
                        self.unify_at(&left_result.ty, &right_result.ty, &right.span)?;
                        let resolved = left_result.ty.resolve(&self.type_uf);
                        match &resolved {
                            Type::Int | Type::Float => resolved,
                            Type::Var(_) => Type::Int, // Default unresolved to Int
                            _ => {
                                return Err(TypeError::TypeMismatch {
                                    expected: Type::Int,
                                    found: resolved,
                                    span: Some(left.span.clone()),
                                    context: None,
                                });
                            }
                        }
                    }
                    BinOp::Eq | BinOp::Neq => {
                        self.unify_at(&left_result.ty, &right_result.ty, &right.span)?;
                        Type::Bool
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        // Comparison operators work on any type that unifies (Int, Char, etc.)
                        self.unify_at(&left_result.ty, &right_result.ty, &right.span)?;
                        Type::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        self.unify_at(&left_result.ty, &Type::Bool, &left.span)?;
                        self.unify_at(&right_result.ty, &Type::Bool, &right.span)?;
                        Type::Bool
                    }
                    BinOp::Cons => {
                        let elem_ty = left_result.ty.clone();
                        let list_ty = Type::list(elem_ty);
                        self.unify_at(&right_result.ty, &list_ty, &right.span)?;
                        list_ty
                    }
                    BinOp::Concat => {
                        self.unify_at(&left_result.ty, &right_result.ty, &right.span)?;
                        // Works for lists and strings
                        left_result.ty.clone()
                    }
                    BinOp::Pipe | BinOp::PipeBack => {
                        // These are desugared in the parser
                        unreachable!("pipe operators should be desugared")
                    }
                    BinOp::Compose | BinOp::ComposeBack => {
                        // f >> g : (a -> b) -> (b -> c) -> (a -> c)
                        let a = self.fresh_var();
                        let b = self.fresh_var();
                        let c = self.fresh_var();

                        match op {
                            BinOp::Compose => {
                                self.unify_at(
                                    &left_result.ty,
                                    &Type::arrow(a.clone(), b.clone()),
                                    &left.span,
                                )?;
                                self.unify_at(
                                    &right_result.ty,
                                    &Type::arrow(b, c.clone()),
                                    &right.span,
                                )?;
                            }
                            BinOp::ComposeBack => {
                                self.unify_at(
                                    &left_result.ty,
                                    &Type::arrow(b.clone(), c.clone()),
                                    &left.span,
                                )?;
                                self.unify_at(
                                    &right_result.ty,
                                    &Type::arrow(a.clone(), b),
                                    &right.span,
                                )?;
                            }
                            _ => unreachable!(),
                        }
                        Type::arrow(a, c)
                    }
                    BinOp::UserDefined(op_name) => {
                        // User-defined operator: look up as function and apply to both operands
                        // op_name : a -> b -> c, where left : a, right : b, result : c
                        let result_ty = self.fresh_var();
                        let func_ty = Type::arrow(
                            left_result.ty.clone(),
                            Type::arrow(right_result.ty.clone(), result_ty.clone()),
                        );
                        // Look up the operator in the environment
                        match env.get(op_name) {
                            Some(scheme) => {
                                let instantiated = self.instantiate(scheme);
                                self.unify_at(&instantiated, &func_ty, &expr.span)?;
                            }
                            None => {
                                return Err(TypeError::UnboundVariable {
                                    name: op_name.clone(),
                                    span: expr.span.clone(),
                                    suggestions: vec![], // No suggestions for operators
                                });
                            }
                        }
                        result_ty
                    }
                };

                // Combined effects = left ∪ right
                let combined = self.union_rows(&left_result.effects, &right_result.effects);
                Ok(InferResult::with_effects(result_ty, combined))
            }

            // UnaryOp: pure
            ExprKind::UnaryOp { op, operand } => {
                let operand_ty = self.infer_expr(env, operand)?;
                let result_ty = match op {
                    UnaryOp::Neg => {
                        self.unify_at(&operand_ty, &Type::Int, &operand.span)?;
                        Type::Int
                    }
                    UnaryOp::Not => {
                        self.unify_at(&operand_ty, &Type::Bool, &operand.span)?;
                        Type::Bool
                    }
                };
                Ok(InferResult::pure(result_ty))
            }

            // Seq: combine effects from both expressions
            ExprKind::Seq { first, second } => {
                let first_result = self.infer_expr_full(env, first)?;
                let second_result = self.infer_expr_full(env, second)?;

                // Combined effects = first ∪ second
                let combined = self.union_rows(&first_result.effects, &second_result.effects);
                Ok(InferResult::with_effects(second_result.ty, combined))
            }

            // Concurrency primitives: pure (effects are at runtime, not type-level)
            ExprKind::Spawn(body) => {
                let body_ty = self.infer_expr(env, body)?;
                // Body should be a thunk: () -> a
                let ret_ty = self.fresh_var();
                self.unify_at(&body_ty, &Type::arrow(Type::Unit, ret_ty), &body.span)?;
                Ok(InferResult::pure(Type::Pid))
            }

            ExprKind::NewChannel => {
                let elem_ty = self.fresh_var();
                Ok(InferResult::pure(Type::Channel(Rc::new(elem_ty))))
            }

            ExprKind::ChanSend { channel, value } => {
                let chan_ty = self.infer_expr(env, channel)?;
                let val_ty = self.infer_expr(env, value)?;
                self.unify_at(&chan_ty, &Type::Channel(Rc::new(val_ty)), &channel.span)?;
                Ok(InferResult::pure(Type::Unit))
            }

            ExprKind::ChanRecv(channel) => {
                let chan_ty = self.infer_expr(env, channel)?;
                let elem_ty = self.fresh_var();
                self.unify_at(
                    &chan_ty,
                    &Type::Channel(Rc::new(elem_ty.clone())),
                    &channel.span,
                )?;
                Ok(InferResult::pure(elem_ty))
            }

            ExprKind::Select { arms } => {
                // All arms must have compatible channel element types
                // All arm bodies must have the same result type
                let result_ty = self.fresh_var();

                for arm in arms {
                    // Infer channel type
                    let chan_ty = self.infer_expr(env, &arm.channel)?;
                    let elem_ty = self.fresh_var();
                    self.unify_at(
                        &chan_ty,
                        &Type::Channel(Rc::new(elem_ty.clone())),
                        &arm.channel.span,
                    )?;

                    // Bind pattern in arm's environment
                    let mut arm_env = env.clone();
                    self.bind_pattern(&mut arm_env, &arm.pattern, &elem_ty)?;

                    // Infer body type and unify with result
                    let body_ty = self.infer_expr(&arm_env, &arm.body)?;
                    self.unify_at(&result_ty, &body_ty, &arm.body.span)?;
                }

                Ok(InferResult::pure(result_ty))
            }

            // ========================================================================
            // Algebraic Effects
            // ========================================================================
            ExprKind::Perform { effect, operation, args } => {
                // Look up the operation in the effect environment
                let op_name = operation;
                let (effect_name, op_info) = if let Some(info) = self.effect_env.operations.get(op_name).cloned() {
                    info
                } else {
                    // Operation not found - fall back to fresh type variable
                    // This allows gradual migration and handles unknown effects
                    let result_ty = self.fresh_var();
                    let mut combined_effects = Row::Empty;
                    for arg in args {
                        let arg_result = self.infer_expr_full(env, arg)?;
                        combined_effects = self.union_rows(&combined_effects, &arg_result.effects);
                    }
                    let effect_row = Row::Extend {
                        effect: Effect {
                            name: effect.clone(),
                            params: vec![],
                        },
                        rest: Rc::new(combined_effects),
                    };
                    return Ok(InferResult::with_effects(result_ty, effect_row));
                };

                // Verify the effect name matches
                if effect_name != *effect {
                    return Err(TypeError::Other(format!(
                        "Operation '{}' belongs to effect '{}', not '{}'",
                        op_name, effect_name, effect
                    )));
                }

                // Get the effect info to know type params
                let effect_info = self.effect_env.effects.get(&effect_name).cloned();
                let type_params = effect_info.map(|e| e.type_params).unwrap_or_default();

                // 1. Fresh vars for effect type params (e.g., 's' in State s)
                let effect_type_args: Vec<Type> = type_params.iter()
                    .map(|_| self.fresh_var())
                    .collect();

                // 2. Fresh vars for operation's own generics (e.g., 'a' in fail : String -> a)
                let op_type_args: Vec<Type> = op_info.generics.iter()
                    .map(|_| self.fresh_var())
                    .collect();

                // 3. Combine for substitution (effect params first, then operation generics)
                let all_type_args: Vec<Type> = effect_type_args.iter()
                    .chain(op_type_args.iter())
                    .cloned()
                    .collect();

                // Instantiate the operation's param types and result type
                // by substituting Generic(i) with all_type_args[i]
                let instantiated_params: Vec<Type> = op_info.param_types.iter()
                    .map(|t| substitute_generics(t, &all_type_args))
                    .collect();
                let instantiated_result = substitute_generics(&op_info.result_type, &all_type_args);

                // Check argument count
                if args.len() != instantiated_params.len() {
                    return Err(TypeError::Other(format!(
                        "Effect operation '{}' expects {} arguments, got {}",
                        op_name, instantiated_params.len(), args.len()
                    )));
                }

                // Infer types for arguments, unify with expected types, and collect effects
                let mut combined_effects = Row::Empty;
                for (arg, expected_ty) in args.iter().zip(instantiated_params.iter()) {
                    let arg_result = self.infer_expr_full(env, arg)?;
                    self.unify_at(&arg_result.ty, expected_ty, &arg.span)?;
                    combined_effects = self.union_rows(&combined_effects, &arg_result.effects);
                }

                // Add the effect to the result's effect row
                let effect_row = Row::Extend {
                    effect: Effect {
                        name: effect.clone(),
                        params: effect_type_args,
                    },
                    rest: Rc::new(combined_effects),
                };

                Ok(InferResult::with_effects(instantiated_result, effect_row))
            }

            ExprKind::Handle { body, return_clause, handlers } => {
                // Infer body type and collect its effects
                let body_result = self.infer_expr_full(env, body)?;

                // Collect which effects are handled by these handlers
                let mut handled_effects: HashSet<String> = HashSet::new();
                for handler in handlers {
                    // Look up which effect this operation belongs to
                    if let Some((effect_name, _)) = self.effect_env.operations.get(&handler.operation).cloned() {
                        handled_effects.insert(effect_name);
                    }
                    // If operation not found, we'll still handle it but can't verify types
                }

                // Infer return clause - this determines the result type
                let mut return_env = env.clone();
                let return_pattern_ty = self.fresh_var();
                self.bind_pattern(&mut return_env, &return_clause.pattern, &return_pattern_ty)?;
                self.unify_at(&body_result.ty, &return_pattern_ty, &return_clause.pattern.span)?;

                let return_body_result = self.infer_expr_full(&return_env, &return_clause.body)?;
                let result_ty = return_body_result.ty.clone();

                // Infer each handler arm and verify types
                for handler in handlers {
                    let mut handler_env = env.clone();

                    // Look up operation signature if available
                    let op_info = self.effect_env.operations.get(&handler.operation).cloned();

                    // Bind operation parameters with expected types if known
                    if let Some((effect_name, ref op)) = op_info {
                        // Get effect type params
                        let effect_info = self.effect_env.effects.get(&effect_name).cloned();
                        let num_type_params = effect_info.map(|e| e.type_params.len()).unwrap_or(0);

                        // Create fresh vars for effect type params
                        let mut type_args: Vec<Type> = Vec::new();
                        for _ in 0..num_type_params {
                            type_args.push(self.fresh_var());
                        }

                        // Also create fresh vars for operation-local generics
                        // These have IDs starting at num_type_params (see extract_operation_signature)
                        // We need to extend type_args to cover all Generic indices used in the operation
                        let max_generic_id = op.generics.iter().copied().max();
                        if let Some(max_id) = max_generic_id {
                            // Extend type_args to cover up to max_id (inclusive)
                            while type_args.len() <= max_id as usize {
                                type_args.push(self.fresh_var());
                            }
                        }

                        // Bind params with instantiated types
                        let instantiated_params: Vec<Type> = op.param_types.iter()
                            .map(|t| substitute_generics(t, &type_args))
                            .collect();

                        for (i, param) in handler.params.iter().enumerate() {
                            let param_ty = instantiated_params.get(i)
                                .cloned()
                                .unwrap_or_else(|| self.fresh_var());
                            self.bind_pattern(&mut handler_env, param, &param_ty)?;
                        }

                        // Continuation takes the operation's result type and returns the handle result
                        // For deep handlers, the continuation can re-perform the handled effects
                        // (they'll be caught by the handler again), so use the body's full effects
                        let cont_arg = substitute_generics(&op.result_type, &type_args);
                        let cont_type = Type::Arrow {
                            arg: Rc::new(cont_arg),
                            ret: Rc::new(result_ty.clone()),
                            effects: body_result.effects.clone(), // Continuation carries body's effects
                        };
                        handler_env.insert(handler.continuation.clone(), Scheme::mono(cont_type));
                    } else {
                        // Unknown operation - use fresh type variables
                        // TODO: Add warning diagnostic for unknown operations
                        // This is potentially unsound but we continue for error recovery

                        for param in &handler.params {
                            let param_ty = self.fresh_var();
                            self.bind_pattern(&mut handler_env, param, &param_ty)?;
                        }

                        let cont_arg = self.fresh_var();
                        let cont_type = Type::Arrow {
                            arg: Rc::new(cont_arg),
                            ret: Rc::new(result_ty.clone()),
                            effects: body_result.effects.clone(), // Continuation carries body's effects
                        };
                        handler_env.insert(handler.continuation.clone(), Scheme::mono(cont_type));
                    }

                    // Infer handler body - should unify with result type
                    let handler_body_result = self.infer_expr_full(&handler_env, &handler.body)?;
                    self.unify_at(&handler_body_result.ty, &result_ty, &handler.body.span)?;
                }

                // Subtract handled effects from body's effect row
                let remaining_effects = self.subtract_effects(&body_result.effects, &handled_effects);

                // Also combine with return clause's effects (should typically be pure)
                let combined = self.union_rows(&remaining_effects, &return_body_result.effects);

                Ok(InferResult::with_effects(result_ty, combined))
            }

            // ========================================================================
            // Records
            // ========================================================================
            ExprKind::Record { name, fields } => {
                // Look up the record type by name
                if let Some(info) = self.type_ctx.get_record(name).cloned() {
                    // Create fresh type variables for the type parameters
                    let mut type_args = Vec::new();
                    for _ in 0..info.type_params {
                        type_args.push(self.fresh_var());
                    }

                    // Build substitution from generic params to fresh type vars
                    let mut subst: HashMap<TypeVarId, Type> = HashMap::new();
                    for (i, ty) in type_args.iter().enumerate() {
                        subst.insert(i as TypeVarId, ty.clone());
                    }

                    // Check that all required fields are provided
                    let provided_fields: HashSet<&str> =
                        fields.iter().map(|(n, _)| n.as_str()).collect();
                    let required_fields: HashSet<&str> =
                        info.field_names.iter().map(|s| s.as_str()).collect();

                    for required in &required_fields {
                        if !provided_fields.contains(required) {
                            return Err(TypeError::MissingRecordField {
                                record_type: name.clone(),
                                field: required.to_string(),
                                span: expr.span.clone(),
                            });
                        }
                    }

                    // Check for unknown fields
                    for (field_name, _) in fields {
                        if !required_fields.contains(field_name.as_str()) {
                            return Err(TypeError::UnknownRecordField {
                                record_type: name.clone(),
                                field: field_name.clone(),
                                span: expr.span.clone(),
                            });
                        }
                    }

                    // Type check each field value
                    for (field_name, field_expr) in fields {
                        if let Some(expected_ty) = info.field_types.get(field_name) {
                            let expected = self.substitute(expected_ty, &subst);
                            let actual = self.infer_expr(env, field_expr)?;
                            self.unify_at(&expected, &actual, &field_expr.span)?;
                        }
                    }

                    // The result type is the record type with the inferred type args
                    let result_ty = Type::Constructor {
                        name: info.type_name.clone(),
                        args: type_args,
                    };
                    Ok(InferResult::pure(result_ty))
                } else {
                    Err(TypeError::UnknownRecordType {
                        name: name.clone(),
                        span: expr.span.clone(),
                    })
                }
            }

            ExprKind::FieldAccess { record, field } => {
                // Check if this is a module-qualified access (Module.name)
                if let ExprKind::Var(module_name) = &record.node {
                    // Check if this is a module name (possibly aliased)
                    let actual_module = self.module_aliases
                        .get(module_name)
                        .cloned();
                    let module_to_check = actual_module.as_deref().unwrap_or(module_name);

                    if let Some(module_env) = self.module_envs.get(module_to_check).cloned() {
                        if let Some(scheme) = module_env.get(field) {
                            // Mark module alias as used if one was used
                            if actual_module.is_some() {
                                self.used_module_aliases.insert(module_name.clone());
                            }
                            let ty = self.instantiate(scheme);
                            return Ok(InferResult::pure(ty));
                        } else {
                            // Module exists but field doesn't
                            return Err(TypeError::UnboundVariable {
                                name: format!("{}.{}", module_name, field),
                                span: expr.span.clone(),
                                suggestions: vec![],
                            });
                        }
                    }
                    // Not a module - fall through to record field access
                }

                // Regular record field access
                let record_ty = self.infer_expr(env, record)?;
                let record_ty_resolved = record_ty.resolve(&self.type_uf);

                // The record type must be a Constructor type
                match &record_ty_resolved {
                    Type::Constructor { name, args } => {
                        // Look up the record info
                        if let Some(info) = self.type_ctx.get_record(name).cloned() {
                            // Build substitution from the record's type args
                            let mut subst: HashMap<TypeVarId, Type> = HashMap::new();
                            for (i, ty) in args.iter().enumerate() {
                                subst.insert(i as TypeVarId, ty.clone());
                            }

                            // Look up the field type
                            if let Some(field_ty) = info.field_types.get(field) {
                                let result_ty = self.substitute(field_ty, &subst);
                                Ok(InferResult::pure(result_ty))
                            } else {
                                Err(TypeError::UnknownRecordField {
                                    record_type: name.clone(),
                                    field: field.clone(),
                                    span: expr.span.clone(),
                                })
                            }
                        } else {
                            Err(TypeError::NotARecordType {
                                ty: record_ty_resolved,
                                span: expr.span.clone(),
                            })
                        }
                    }
                    Type::Var(_) => {
                        // Record type not yet known - could add deferred constraint
                        // For now, require the type to be known
                        Err(TypeError::CannotInferRecordType {
                            span: expr.span.clone(),
                        })
                    }
                    _ => Err(TypeError::NotARecordType {
                        ty: record_ty_resolved,
                        span: expr.span.clone(),
                    }),
                }
            }

            ExprKind::RecordUpdate { base, updates } => {
                // Infer the type of the base record
                let base_ty = self.infer_expr(env, base)?;
                let base_ty_resolved = base_ty.resolve(&self.type_uf);

                match &base_ty_resolved {
                    Type::Constructor { name, args } => {
                        if let Some(info) = self.type_ctx.get_record(name).cloned() {
                            // Build substitution from the record's type args
                            let mut subst: HashMap<TypeVarId, Type> = HashMap::new();
                            for (i, ty) in args.iter().enumerate() {
                                subst.insert(i as TypeVarId, ty.clone());
                            }

                            // Type check each update field
                            for (field_name, field_expr) in updates {
                                if let Some(expected_ty) = info.field_types.get(field_name) {
                                    let expected = self.substitute(expected_ty, &subst);
                                    let actual = self.infer_expr(env, field_expr)?;
                                    self.unify_at(&expected, &actual, &field_expr.span)?;
                                } else {
                                    return Err(TypeError::UnknownRecordField {
                                        record_type: name.clone(),
                                        field: field_name.clone(),
                                        span: field_expr.span.clone(),
                                    });
                                }
                            }

                            // Result has same type as base
                            Ok(InferResult::pure(base_ty))
                        } else {
                            Err(TypeError::NotARecordType {
                                ty: base_ty_resolved,
                                span: expr.span.clone(),
                            })
                        }
                    }
                    Type::Var(_) => Err(TypeError::CannotInferRecordType {
                        span: expr.span.clone(),
                    }),
                    _ => Err(TypeError::NotARecordType {
                        ty: base_ty_resolved,
                        span: expr.span.clone(),
                    }),
                }
            }
        }
    }

    fn infer_literal(&self, lit: &Literal) -> Type {
        match lit {
            Literal::Int(_) => Type::Int,
            Literal::Float(_) => Type::Float,
            Literal::String(_) => Type::String,
            Literal::Char(_) => Type::Char,
            Literal::Bool(_) => Type::Bool,
            Literal::Unit => Type::Unit,
        }
    }

    /// Bind pattern variables to types in the environment
    fn bind_pattern(
        &mut self,
        env: &mut TypeEnv,
        pattern: &Pattern,
        ty: &Type,
    ) -> Result<(), TypeError> {
        match &pattern.node {
            PatternKind::Wildcard => Ok(()),

            PatternKind::Var(name) => {
                env.insert(name.clone(), Scheme::mono(ty.clone()));
                Ok(())
            }

            PatternKind::Lit(lit) => {
                let lit_ty = self.infer_literal(lit);
                self.unify_at(ty, &lit_ty, &pattern.span)?;
                Ok(())
            }

            PatternKind::Tuple(pats) => {
                let pat_types: Vec<_> = pats.iter().map(|_| self.fresh_var()).collect();
                self.unify_at(ty, &Type::Tuple(pat_types.clone()), &pattern.span)?;
                for (pat, pat_ty) in pats.iter().zip(&pat_types) {
                    self.bind_pattern(env, pat, pat_ty)?;
                }
                Ok(())
            }

            PatternKind::List(pats) => {
                let elem_ty = self.fresh_var();
                let list_ty = Type::list(elem_ty.clone());
                self.unify_at(ty, &list_ty, &pattern.span)?;
                for pat in pats {
                    self.bind_pattern(env, pat, &elem_ty)?;
                }
                Ok(())
            }

            PatternKind::Cons { head, tail } => {
                let elem_ty = self.fresh_var();
                let list_ty = Type::list(elem_ty.clone());
                self.unify_at(ty, &list_ty, &pattern.span)?;
                self.bind_pattern(env, head, &elem_ty)?;
                self.bind_pattern(env, tail, &list_ty)?;
                Ok(())
            }

            PatternKind::Constructor { name, args } => {
                if let Some(info) = self.type_ctx.get_constructor(name).cloned() {
                    let mut type_args = Vec::new();
                    for _ in 0..info.type_params {
                        type_args.push(self.fresh_var());
                    }

                    let constructor_ty = Type::Constructor {
                        name: info.type_name.clone(),
                        args: type_args.clone(),
                    };
                    self.unify_at(ty, &constructor_ty, &pattern.span)?;

                    let mut subst: HashMap<TypeVarId, Type> = HashMap::new();
                    for (i, ty) in type_args.iter().enumerate() {
                        subst.insert(i as TypeVarId, ty.clone());
                    }

                    for (pat, field_ty) in args.iter().zip(&info.field_types) {
                        let expected_ty = self.substitute(field_ty, &subst);
                        self.bind_pattern(env, pat, &expected_ty)?;
                    }
                    Ok(())
                } else {
                    let candidates: Vec<&str> = self.type_ctx.constructor_names().collect();
                    let suggestions = find_similar(name, candidates, 2);
                    Err(TypeError::UnknownConstructor {
                        name: name.clone(),
                        span: pattern.span.clone(),
                        suggestions,
                    })
                }
            }

            PatternKind::Record { name, fields } => {
                if let Some(info) = self.type_ctx.get_record(name).cloned() {
                    // Create fresh type variables for type parameters
                    let mut type_args = Vec::new();
                    for _ in 0..info.type_params {
                        type_args.push(self.fresh_var());
                    }

                    // The record type is represented as a Constructor type
                    let record_ty = Type::Constructor {
                        name: info.type_name.clone(),
                        args: type_args.clone(),
                    };
                    self.unify_at(ty, &record_ty, &pattern.span)?;

                    // Build substitution map for type parameters
                    let mut subst: HashMap<TypeVarId, Type> = HashMap::new();
                    for (i, ty) in type_args.iter().enumerate() {
                        subst.insert(i as TypeVarId, ty.clone());
                    }

                    // Bind each field pattern
                    for (field_name, sub_pat_opt) in fields {
                        let field_name_str = field_name.as_str();
                        if let Some(field_ty) = info.field_types.get(field_name_str) {
                            let expected_ty = self.substitute(field_ty, &subst);
                            match sub_pat_opt {
                                Some(sub_pat) => {
                                    // Explicit pattern: { method = m }
                                    self.bind_pattern(env, sub_pat, &expected_ty)?;
                                }
                                None => {
                                    // Punned field: { method } binds variable `method`
                                    env.insert(
                                        field_name_str.to_string(),
                                        Scheme::mono(expected_ty),
                                    );
                                }
                            }
                        } else {
                            return Err(TypeError::UnknownRecordField {
                                record_type: name.clone(),
                                field: field_name_str.to_string(),
                                span: pattern.span.clone(),
                            });
                        }
                    }
                    Ok(())
                } else {
                    Err(TypeError::UnknownRecordType {
                        name: name.clone(),
                        span: pattern.span.clone(),
                    })
                }
            }
        }
    }

    /// Bind pattern variables with a pre-computed scheme (for let-polymorphism)
    fn bind_pattern_scheme(
        &mut self,
        env: &mut TypeEnv,
        pattern: &Pattern,
        scheme: Scheme,
    ) -> Result<(), TypeError> {
        match &pattern.node {
            PatternKind::Var(name) => {
                env.insert(name.clone(), scheme);
                Ok(())
            }
            // For complex patterns in let bindings, we just use monomorphic types
            _ => {
                let ty = self.instantiate(&scheme);
                self.bind_pattern(env, pattern, &ty)
            }
        }
    }


    /// Collect all type variable names from a TypeExpr (lowercase identifiers)
    /// Used for implicitly quantified type variables in val declarations
    fn collect_type_vars(expr: &TypeExpr, vars: &mut Vec<String>) {
        match &expr.node {
            TypeExprKind::Var(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            TypeExprKind::Named(_) => {}
            TypeExprKind::App { constructor, args } => {
                Self::collect_type_vars(constructor, vars);
                for arg in args {
                    Self::collect_type_vars(arg, vars);
                }
            }
            TypeExprKind::Arrow { from, to, effects } => {
                Self::collect_type_vars(from, vars);
                Self::collect_type_vars(to, vars);
                // Collect type vars from effect parameters
                if let Some(eff_row) = effects {
                    for eff in &eff_row.effects {
                        for param in &eff.params {
                            Self::collect_type_vars(param, vars);
                        }
                    }
                    // Row variable is also a type var
                    if let Some(ref rest) = eff_row.rest {
                        if !vars.contains(rest) {
                            vars.push(rest.clone());
                        }
                    }
                }
            }
            TypeExprKind::Tuple(types) => {
                for t in types {
                    Self::collect_type_vars(t, vars);
                }
            }
            TypeExprKind::Channel(inner) => {
                Self::collect_type_vars(inner, vars);
            }
            TypeExprKind::List(inner) => {
                Self::collect_type_vars(inner, vars);
            }
        }
    }

    /// Convert surface TypeExpr to internal Type, using pre-created fresh type variables
    /// for type parameters. This is used for val declarations where we want proper
    /// Unbound type vars (that can be generalized) instead of Generic types.
    fn type_expr_to_type_with_fresh_vars(
        &mut self,
        expr: &TypeExpr,
        var_names: &[String],
        fresh_vars: &[Type],
    ) -> Result<Type, TypeError> {
        match &expr.node {
            TypeExprKind::Var(name) => {
                // Find the index of this variable name
                if let Some(idx) = var_names.iter().position(|n| n == name) {
                    Ok(fresh_vars[idx].clone())
                } else {
                    Err(TypeError::UnboundVariable {
                        name: name.clone(),
                        span: expr.span.clone(),
                        suggestions: vec![],
                    })
                }
            }
            TypeExprKind::Named(name) => match name.as_str() {
                "Int" => Ok(Type::Int),
                "Float" => Ok(Type::Float),
                "Bool" => Ok(Type::Bool),
                "String" => Ok(Type::String),
                "Char" => Ok(Type::Char),
                "Pid" => Ok(Type::Pid),
                _ => {
                    // Check if this is a type alias with no parameters
                    if let Some(alias_info) = self.type_ctx.get_type_alias(name).cloned() {
                        if alias_info.params.is_empty() {
                            return Ok(alias_info.body);
                        }
                    }
                    Ok(Type::Constructor {
                        name: name.clone(),
                        args: vec![],
                    })
                }
            },
            TypeExprKind::App { constructor, args } => {
                let arg_types: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.type_expr_to_type_with_fresh_vars(a, var_names, fresh_vars))
                    .collect();
                match &constructor.node {
                    TypeExprKind::Named(name) => {
                        let arg_types = arg_types?;
                        if name == "List" && arg_types.len() == 1 {
                            return Ok(Type::list(arg_types.into_iter().next().unwrap()));
                        }
                        // Canonicalize "Fiber a" to Type::Fiber(a)
                        if name == "Fiber" && arg_types.len() == 1 {
                            return Ok(Type::Fiber(Rc::new(arg_types.into_iter().next().unwrap())));
                        }
                        // Check if this is a type alias with parameters
                        if let Some(alias_info) = self.type_ctx.get_type_alias(name).cloned() {
                            if alias_info.params.len() == arg_types.len() {
                                return Ok(substitute_generics(&alias_info.body, &arg_types));
                            }
                        }
                        Ok(Type::Constructor {
                            name: name.clone(),
                            args: arg_types,
                        })
                    }
                    _ => Err(TypeError::PatternMismatch {
                        span: constructor.span.clone(),
                    }),
                }
            }
            TypeExprKind::Arrow { from, to, effects } => {
                let from_ty = self.type_expr_to_type_with_fresh_vars(from, var_names, fresh_vars)?;
                let to_ty = self.type_expr_to_type_with_fresh_vars(to, var_names, fresh_vars)?;
                let eff_row = self.effect_row_expr_to_row(effects.as_ref(), var_names, fresh_vars)?;
                Ok(Type::arrow_with_effects(from_ty, to_ty, eff_row))
            }
            TypeExprKind::Tuple(types) => {
                let tys: Result<Vec<_>, _> = types
                    .iter()
                    .map(|t| self.type_expr_to_type_with_fresh_vars(t, var_names, fresh_vars))
                    .collect();
                let tys = tys?;
                Ok(if tys.is_empty() {
                    Type::Unit
                } else {
                    Type::Tuple(tys)
                })
            }
            TypeExprKind::Channel(inner) => {
                let inner_ty = self.type_expr_to_type_with_fresh_vars(inner, var_names, fresh_vars)?;
                Ok(Type::Channel(Rc::new(inner_ty)))
            }
            TypeExprKind::List(inner) => {
                let elem_ty = self.type_expr_to_type_with_fresh_vars(inner, var_names, fresh_vars)?;
                Ok(Type::list(elem_ty))
            }
        }
    }

    /// Convert an EffectRowExpr to a Row type, using fresh type variables
    fn effect_row_expr_to_row(
        &mut self,
        expr: Option<&EffectRowExpr>,
        var_names: &[String],
        fresh_vars: &[Type],
    ) -> Result<Row, TypeError> {
        match expr {
            None => {
                // No effect annotation - use empty row (pure)
                Ok(Row::Empty)
            }
            Some(eff_row) => {
                // Build row from effects
                let mut row = if let Some(ref rest) = eff_row.rest {
                    // Has a row variable - look it up or create fresh
                    if let Some(idx) = var_names.iter().position(|n| n == rest) {
                        // Convert the type var to a row var
                        if let Type::Var(_) = &fresh_vars[idx] {
                            // Create a fresh row var with same level
                            Row::Var(self.row_uf.fresh(self.level))
                        } else {
                            Row::Empty
                        }
                    } else {
                        // Not in fresh_vars, create new row var
                        self.fresh_row_var()
                    }
                } else {
                    Row::Empty
                };

                // Add effects in reverse order (so first effect is outermost)
                for eff in eff_row.effects.iter().rev() {
                    let params: Result<Vec<_>, _> = eff
                        .params
                        .iter()
                        .map(|p| self.type_expr_to_type_with_fresh_vars(p, var_names, fresh_vars))
                        .collect();
                    row = Row::Extend {
                        effect: Effect {
                            name: eff.name.clone(),
                            params: params?,
                        },
                        rest: Rc::new(row),
                    };
                }

                Ok(row)
            }
        }
    }

    /// Convert an EffectRowExpr to a Row type, using param_map for generic vars
    fn effect_row_expr_to_row_with_param_map(
        &mut self,
        expr: Option<&EffectRowExpr>,
        param_map: &HashMap<String, TypeVarId>,
    ) -> Result<Row, TypeError> {
        match expr {
            None => {
                // No effect annotation - use empty row (pure)
                Ok(Row::Empty)
            }
            Some(eff_row) => {
                // Build row from effects
                let mut row = if let Some(ref rest) = eff_row.rest {
                    // Has a row variable - look it up
                    if let Some(&id) = param_map.get(rest) {
                        Row::Generic(id)
                    } else {
                        // Not in param_map, create fresh row var
                        self.fresh_row_var()
                    }
                } else {
                    Row::Empty
                };

                // Add effects in reverse order (so first effect is outermost)
                for eff in eff_row.effects.iter().rev() {
                    let params: Result<Vec<_>, _> = eff
                        .params
                        .iter()
                        .map(|p| self.type_expr_to_type(p, param_map))
                        .collect();
                    row = Row::Extend {
                        effect: Effect {
                            name: eff.name.clone(),
                            params: params?,
                        },
                        rest: Rc::new(row),
                    };
                }

                Ok(row)
            }
        }
    }

    /// Convert surface TypeExpr to internal Type, mapping type params to generic vars
    fn type_expr_to_type(
        &mut self,
        expr: &TypeExpr,
        param_map: &HashMap<String, TypeVarId>,
    ) -> Result<Type, TypeError> {
        match &expr.node {
            TypeExprKind::Var(name) => {
                if let Some(&id) = param_map.get(name) {
                    Ok(Type::new_generic(id))
                } else {
                    Err(TypeError::UnboundVariable {
                        name: name.clone(),
                        span: expr.span.clone(),
                        suggestions: vec![], // No suggestions for type variables
                    })
                }
            }
            TypeExprKind::Named(name) => match name.as_str() {
                "Int" => Ok(Type::Int),
                "Float" => Ok(Type::Float),
                "Bool" => Ok(Type::Bool),
                "String" => Ok(Type::String),
                "Char" => Ok(Type::Char),
                "Pid" => Ok(Type::Pid),
                _ => {
                    // Check if this is a type alias with no parameters
                    if let Some(alias_info) = self.type_ctx.get_type_alias(name).cloned() {
                        if alias_info.params.is_empty() {
                            // Zero-parameter alias: just return the body
                            return Ok(alias_info.body);
                        }
                    }
                    Ok(Type::Constructor {
                        name: name.clone(),
                        args: vec![],
                    })
                }
            },
            TypeExprKind::App { constructor, args } => {
                let arg_types: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.type_expr_to_type(a, param_map))
                    .collect();
                match &constructor.node {
                    TypeExprKind::Named(name) => {
                        let arg_types = arg_types?;
                        // Canonicalize "List a" to Type::list(a)
                        if name == "List" && arg_types.len() == 1 {
                            return Ok(Type::list(arg_types.into_iter().next().unwrap()));
                        }
                        // Canonicalize "Fiber a" to Type::Fiber(a)
                        if name == "Fiber" && arg_types.len() == 1 {
                            return Ok(Type::Fiber(Rc::new(arg_types.into_iter().next().unwrap())));
                        }
                        // Check if this is a type alias with parameters
                        if let Some(alias_info) = self.type_ctx.get_type_alias(name).cloned() {
                            if alias_info.params.len() == arg_types.len() {
                                // Substitute type arguments into the alias body
                                return Ok(substitute_generics(&alias_info.body, &arg_types));
                            }
                        }
                        Ok(Type::Constructor {
                            name: name.clone(),
                            args: arg_types,
                        })
                    }
                    _ => Err(TypeError::PatternMismatch {
                        span: constructor.span.clone(),
                    }),
                }
            }
            TypeExprKind::Arrow { from, to, effects } => {
                let from_ty = self.type_expr_to_type(from, param_map)?;
                let to_ty = self.type_expr_to_type(to, param_map)?;
                let eff_row = self.effect_row_expr_to_row_with_param_map(effects.as_ref(), param_map)?;
                Ok(Type::arrow_with_effects(from_ty, to_ty, eff_row))
            }
            TypeExprKind::Tuple(types) => {
                let tys: Result<Vec<_>, _> = types
                    .iter()
                    .map(|t| self.type_expr_to_type(t, param_map))
                    .collect();
                let tys = tys?;
                // Empty tuple normalizes to Unit (they are the same in type theory)
                Ok(if tys.is_empty() {
                    Type::Unit
                } else {
                    Type::Tuple(tys)
                })
            }
            TypeExprKind::Channel(inner) => {
                let inner_ty = self.type_expr_to_type(inner, param_map)?;
                Ok(Type::Channel(Rc::new(inner_ty)))
            }
            TypeExprKind::List(inner) => {
                let elem_ty = self.type_expr_to_type(inner, param_map)?;
                Ok(Type::list(elem_ty))
            }
        }
    }

    /// Register a type declaration
    pub fn register_type_decl(&mut self, decl: &Decl) {
        if let Decl::Type {
            name,
            params,
            constructors,
            ..
        } = decl
        {
            // Map type parameter names to generic variable IDs
            let param_map: HashMap<String, TypeVarId> = params
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), i as TypeVarId))
                .collect();

            for ctor in constructors {
                // Convert actual TypeExprs to Types with proper generic variables
                let field_types: Vec<Type> = ctor
                    .fields
                    .iter()
                    .filter_map(|type_expr| self.type_expr_to_type(type_expr, &param_map).ok())
                    .collect();

                let info = ConstructorInfo {
                    type_name: name.clone(),
                    type_params: params.len() as u32,
                    field_types,
                };

                self.type_ctx.add_constructor(ctor.name.clone(), info);
            }
        }
    }

    /// Register a record type declaration
    pub fn register_record_decl(&mut self, decl: &Decl) {
        if let Decl::Record {
            name,
            params,
            fields,
            ..
        } = decl
        {
            // Map type parameter names to generic variable IDs
            let param_map: HashMap<String, TypeVarId> = params
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), i as TypeVarId))
                .collect();

            let mut field_names = Vec::new();
            let mut field_types = HashMap::new();

            for field in fields {
                field_names.push(field.name.clone());
                if let Ok(ty) = self.type_expr_to_type(&field.ty, &param_map) {
                    field_types.insert(field.name.clone(), ty);
                }
            }

            let info = RecordInfo {
                type_name: name.clone(),
                type_params: params.len() as u32,
                field_names,
                field_types,
            };

            self.type_ctx.add_record(name.clone(), info);
        }
    }

    /// Register a type alias declaration
    pub fn register_type_alias_decl(&mut self, decl: &Decl) {
        if let Decl::TypeAlias {
            name,
            params,
            body,
            ..
        } = decl
        {
            // Map type parameter names to generic variable IDs
            let param_map: HashMap<String, TypeVarId> = params
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), i as TypeVarId))
                .collect();

            // Convert the body TypeExpr to a Type
            if let Ok(body_type) = self.type_expr_to_type(body, &param_map) {
                let info = TypeAliasInfo {
                    name: name.clone(),
                    params: params.clone(),
                    body: body_type,
                };
                self.type_ctx.add_type_alias(name.clone(), info);
            }
        }
    }

    /// Register a trait declaration
    pub fn register_trait_decl(&mut self, decl: &Decl) -> Result<(), TypeError> {
        if let Decl::Trait {
            name,
            type_param,
            supertraits,
            methods,
            ..
        } = decl
        {
            // Map the type parameter to Generic(0)
            let mut param_map: HashMap<String, TypeVarId> = HashMap::new();
            param_map.insert(type_param.clone(), 0);

            // Convert method signatures to internal Types
            let mut method_types: HashMap<String, Type> = HashMap::new();
            for method in methods {
                let method_ty = self.type_expr_to_type(&method.type_sig, &param_map)?;
                method_types.insert(method.name.clone(), method_ty);
            }

            let info = TraitInfo {
                name: name.clone(),
                type_param: type_param.clone(),
                supertraits: supertraits.clone(),
                methods: method_types,
            };

            self.class_env.add_trait(info);
        }
        Ok(())
    }

    /// Register an instance declaration
    pub fn register_instance_decl(&mut self, decl: &Decl) -> Result<(), TypeError> {
        if let Decl::Instance {
            trait_name,
            target_type,
            constraints,
            methods,
        } = decl
        {
            // Check that the trait exists
            if self.class_env.get_trait(trait_name).is_none() {
                return Err(TypeError::UnknownTrait {
                    name: trait_name.clone(),
                    span: None,
                });
            }

            // Build a param_map for type variables that might appear in the instance head
            // For now, use a simple approach: collect all type vars from constraints
            let mut param_map: HashMap<String, TypeVarId> = HashMap::new();
            for (i, constraint) in constraints.iter().enumerate() {
                param_map.insert(constraint.type_var.clone(), i as TypeVarId);
            }

            // Convert target type to internal Type
            let head = self.type_expr_to_type(target_type, &param_map)?;

            // Convert constraints to Preds
            let preds: Vec<Pred> = constraints
                .iter()
                .map(|c| {
                    let ty = param_map
                        .get(&c.type_var)
                        .map(|&id| Type::Generic(id))
                        .unwrap_or_else(|| Type::Generic(0)); // Fallback shouldn't happen
                    Pred::new(c.trait_name.clone(), ty)
                })
                .collect();

            let info = InstanceInfo {
                trait_name: trait_name.clone(),
                head: head.clone(),
                constraints: preds,
                method_impls: methods.clone(),
            };

            // Add instance with overlap checking
            self.class_env.add_instance(info).map_err(|e| match e {
                ClassError::UnknownTrait(name) => TypeError::UnknownTrait { name, span: None },
                ClassError::OverlappingInstance {
                    trait_name,
                    existing,
                    new,
                } => TypeError::OverlappingInstance {
                    trait_name,
                    existing,
                    new,
                    span: None,
                },
            })?;
        }
        Ok(())
    }

    /// Infer types for an entire program.
    /// Returns all accumulated type errors rather than stopping on the first one.
    /// On success, returns the type environment.
    /// On failure, returns a non-empty vector of type errors.
    pub fn infer_program(&mut self, program: &Program) -> Result<TypeEnv, Vec<TypeError>> {
        // Clear any previous errors
        self.errors.clear();
        let mut env = TypeEnv::new();

        // Add built-in functions
        // print : forall a t. a/t -> ()/t  (polymorphic in arg type and answer type)
        let print_scheme = Scheme {
            num_generics: 2,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::new_generic(0)), // Generic arg type
                ret: Rc::new(Type::Unit),
                effects: Row::Empty, // Pure for now - TODO: should have IO effect
            },
        };
        env.insert("print".into(), print_scheme);

        // int_to_string : Int -> String
        env.insert(
            "int_to_string".into(),
            Scheme::mono(Type::arrow(Type::Int, Type::String)),
        );

        // string_to_int : String -> Int (parses decimal string)
        env.insert(
            "string_to_int".into(),
            Scheme::mono(Type::arrow(Type::String, Type::Int)),
        );

        // string_length : String -> Int
        env.insert(
            "string_length".into(),
            Scheme::mono(Type::arrow(Type::String, Type::Int)),
        );

        // string_to_chars : String -> List Char
        env.insert(
            "string_to_chars".into(),
            Scheme::mono(Type::arrow(Type::String, Type::list(Type::Char))),
        );

        // chars_to_string : List Char -> String
        env.insert(
            "chars_to_string".into(),
            Scheme::mono(Type::arrow(Type::list(Type::Char), Type::String)),
        );

        // char_to_string : Char -> String
        env.insert(
            "char_to_string".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::String)),
        );

        // char_to_int : Char -> Int
        env.insert(
            "char_to_int".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::Int)),
        );

        // char_to_lower : Char -> Char
        env.insert(
            "char_to_lower".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::Char)),
        );

        // char_to_upper : Char -> Char
        env.insert(
            "char_to_upper".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::Char)),
        );

        // char_is_whitespace : Char -> Bool
        env.insert(
            "char_is_whitespace".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::Bool)),
        );

        // char_is_digit : Char -> Bool
        env.insert(
            "char_is_digit".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::Bool)),
        );

        // char_is_alpha : Char -> Bool
        env.insert(
            "char_is_alpha".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::Bool)),
        );

        // char_is_alphanumeric : Char -> Bool
        env.insert(
            "char_is_alphanumeric".into(),
            Scheme::mono(Type::arrow(Type::Char, Type::Bool)),
        );

        // string_index_of : String -> String -> Option Int
        env.insert(
            "string_index_of".into(),
            Scheme::mono(Type::arrow(
                Type::String,
                Type::arrow(Type::String, Type::option(Type::Int)),
            )),
        );

        // string_substring : Int -> Int -> String -> String
        env.insert(
            "string_substring".into(),
            Scheme::mono(Type::arrow(
                Type::Int,
                Type::arrow(Type::Int, Type::arrow(Type::String, Type::String)),
            )),
        );

        // Native string operations (replacing Gneiss implementations)
        // string_to_lower : String -> String
        env.insert(
            "string_to_lower".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // string_to_upper : String -> String
        env.insert(
            "string_to_upper".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // string_trim : String -> String
        env.insert(
            "string_trim".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // string_trim_start : String -> String
        env.insert(
            "string_trim_start".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // string_trim_end : String -> String
        env.insert(
            "string_trim_end".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // string_reverse : String -> String
        env.insert(
            "string_reverse".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // string_is_empty : String -> Bool
        env.insert(
            "string_is_empty".into(),
            Scheme::mono(Type::arrow(Type::String, Type::Bool)),
        );

        // string_split : String -> String -> List String
        env.insert(
            "string_split".into(),
            Scheme::mono(Type::arrow(
                Type::String,
                Type::arrow(Type::String, Type::list(Type::String)),
            )),
        );

        // string_join : String -> List String -> String
        env.insert(
            "string_join".into(),
            Scheme::mono(Type::arrow(
                Type::String,
                Type::arrow(Type::list(Type::String), Type::String),
            )),
        );

        // string_char_at : Int -> String -> Option Char
        env.insert(
            "string_char_at".into(),
            Scheme::mono(Type::arrow(
                Type::Int,
                Type::arrow(Type::String, Type::option(Type::Char)),
            )),
        );

        // string_concat : String -> String -> String
        env.insert(
            "string_concat".into(),
            Scheme::mono(Type::arrow(
                Type::String,
                Type::arrow(Type::String, Type::String),
            )),
        );

        // string_repeat : Int -> String -> String
        env.insert(
            "string_repeat".into(),
            Scheme::mono(Type::arrow(
                Type::Int,
                Type::arrow(Type::String, Type::String),
            )),
        );

        // Bytes builtins
        // bytes_to_string : Bytes -> String
        env.insert(
            "bytes_to_string".into(),
            Scheme::mono(Type::arrow(Type::Bytes, Type::String)),
        );

        // string_to_bytes : String -> Bytes
        env.insert(
            "string_to_bytes".into(),
            Scheme::mono(Type::arrow(Type::String, Type::Bytes)),
        );

        // bytes_length : Bytes -> Int
        env.insert(
            "bytes_length".into(),
            Scheme::mono(Type::arrow(Type::Bytes, Type::Int)),
        );

        // bytes_slice : Int -> Int -> Bytes -> Bytes
        env.insert(
            "bytes_slice".into(),
            Scheme::mono(Type::arrow(
                Type::Int,
                Type::arrow(Type::Int, Type::arrow(Type::Bytes, Type::Bytes)),
            )),
        );

        // bytes_concat : Bytes -> Bytes -> Bytes
        env.insert(
            "bytes_concat".into(),
            Scheme::mono(Type::arrow(Type::Bytes, Type::arrow(Type::Bytes, Type::Bytes))),
        );

        // I/O builtins
        // sleep_ms : Int -> ()
        env.insert(
            "sleep_ms".into(),
            Scheme::mono(Type::arrow(Type::Int, Type::Unit)),
        );

        // Helper: Result IoError T
        let io_error_type = Type::Constructor {
            name: "IoError".to_string(),
            args: vec![],
        };
        let result_io_error = |t: Type| -> Type {
            Type::Constructor {
                name: "Result".to_string(),
                args: vec![io_error_type.clone(), t],
            }
        };

        // File I/O builtins
        // file_open : String -> String -> Result IoError FileHandle
        env.insert(
            "file_open".into(),
            Scheme::mono(Type::arrow(
                Type::String,
                Type::arrow(Type::String, result_io_error(Type::FileHandle)),
            )),
        );

        // file_read : FileHandle -> Int -> Result IoError Bytes
        env.insert(
            "file_read".into(),
            Scheme::mono(Type::arrow(
                Type::FileHandle,
                Type::arrow(Type::Int, result_io_error(Type::Bytes)),
            )),
        );

        // file_write : FileHandle -> Bytes -> Result IoError Int
        env.insert(
            "file_write".into(),
            Scheme::mono(Type::arrow(
                Type::FileHandle,
                Type::arrow(Type::Bytes, result_io_error(Type::Int)),
            )),
        );

        // file_close : FileHandle -> Result IoError ()
        env.insert(
            "file_close".into(),
            Scheme::mono(Type::arrow(Type::FileHandle, result_io_error(Type::Unit))),
        );

        // TCP socket builtins
        // tcp_connect : String -> Int -> Result IoError TcpSocket
        env.insert(
            "tcp_connect".into(),
            Scheme::mono(Type::arrow(
                Type::String,
                Type::arrow(Type::Int, result_io_error(Type::TcpSocket)),
            )),
        );

        // tcp_listen : String -> Int -> Result IoError TcpListener
        env.insert(
            "tcp_listen".into(),
            Scheme::mono(Type::arrow(
                Type::String,
                Type::arrow(Type::Int, result_io_error(Type::TcpListener)),
            )),
        );

        // tcp_accept : TcpListener -> Result IoError TcpSocket
        env.insert(
            "tcp_accept".into(),
            Scheme::mono(Type::arrow(Type::TcpListener, result_io_error(Type::TcpSocket))),
        );

        // tcp_read : TcpSocket -> Int -> Result IoError Bytes
        env.insert(
            "tcp_read".into(),
            Scheme::mono(Type::arrow(
                Type::TcpSocket,
                Type::arrow(Type::Int, result_io_error(Type::Bytes)),
            )),
        );

        // tcp_write : TcpSocket -> Bytes -> Result IoError Int
        env.insert(
            "tcp_write".into(),
            Scheme::mono(Type::arrow(
                Type::TcpSocket,
                Type::arrow(Type::Bytes, result_io_error(Type::Int)),
            )),
        );

        // tcp_close : TcpSocket -> Result IoError ()
        env.insert(
            "tcp_close".into(),
            Scheme::mono(Type::arrow(Type::TcpSocket, result_io_error(Type::Unit))),
        );

        // spawn : forall a. (() -> a) -> Pid (backwards compatibility)
        // Note: This is the old spawn syntax, returns Pid not Fiber
        let spawn_scheme = Scheme {
            num_generics: 1, // Only need generic for return type now
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Unit),
                    ret: Rc::new(Type::new_generic(0)),
                    effects: Row::Empty,
                }),
                ret: Rc::new(Type::Pid),
                effects: Row::Empty, // TODO: should have Async effect
            },
        };
        env.insert("spawn".into(), spawn_scheme);

        // Fiber.spawn : forall a. (() -> a) -> Fiber a
        let fiber_spawn_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Unit),
                    ret: Rc::new(Type::new_generic(0)),
                    effects: Row::Empty,
                }),
                ret: Rc::new(Type::Fiber(Rc::new(Type::new_generic(0)))),
                effects: Row::Empty, // TODO: should have Async effect
            },
        };
        env.insert("Fiber.spawn".into(), fiber_spawn_scheme);

        // Fiber.join : forall a. Fiber a -> a
        let fiber_join_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Fiber(Rc::new(Type::new_generic(0)))),
                ret: Rc::new(Type::new_generic(0)),
                effects: Row::Empty, // TODO: should have Async effect
            },
        };
        env.insert("Fiber.join".into(), fiber_join_scheme);

        // Fiber.yield : () -> ()
        let fiber_yield_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Unit),
                ret: Rc::new(Type::Unit),
                effects: Row::Empty, // TODO: should have Async effect
            },
        };
        env.insert("Fiber.yield".into(), fiber_yield_scheme);

        // Dict.new : forall a. () -> Dict a
        let dict_new_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Unit),
                ret: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.new".into(), dict_new_scheme);

        // Dict.insert : forall a. String -> a -> Dict a -> Dict a
        let dict_insert_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::String),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::new_generic(0)),
                    ret: Rc::new(Type::Arrow {
                        arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                        ret: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                        effects: Row::Empty,
                    }),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.insert".into(), dict_insert_scheme);

        // Dict.get : forall a. String -> Dict a -> Option a
        let dict_get_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::String),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                    ret: Rc::new(Type::Constructor {
                        name: "Option".into(),
                        args: vec![Type::new_generic(0)],
                    }),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.get".into(), dict_get_scheme);

        // Dict.remove : forall a. String -> Dict a -> Dict a
        let dict_remove_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::String),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                    ret: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.remove".into(), dict_remove_scheme);

        // Dict.contains : forall a. String -> Dict a -> Bool
        let dict_contains_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::String),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                    ret: Rc::new(Type::Bool),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.contains".into(), dict_contains_scheme);

        // Dict.keys : forall a. Dict a -> [String]
        let dict_keys_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                ret: Rc::new(Type::list(Type::String)),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.keys".into(), dict_keys_scheme);

        // Dict.values : forall a. Dict a -> [a]
        let dict_values_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                ret: Rc::new(Type::list(Type::new_generic(0))),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.values".into(), dict_values_scheme);

        // Dict.size : forall a. Dict a -> Int
        let dict_size_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                ret: Rc::new(Type::Int),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.size".into(), dict_size_scheme);

        // Dict.isEmpty : forall a. Dict a -> Bool
        let dict_is_empty_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                ret: Rc::new(Type::Bool),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.isEmpty".into(), dict_is_empty_scheme);

        // Dict.toList : forall a. Dict a -> [(String, a)]
        let dict_to_list_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                ret: Rc::new(Type::list(Type::Tuple(vec![
                    Type::String,
                    Type::new_generic(0),
                ]))),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.toList".into(), dict_to_list_scheme);

        // Dict.fromList : forall a. [(String, a)] -> Dict a
        let dict_from_list_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::list(Type::Tuple(vec![
                    Type::String,
                    Type::new_generic(0),
                ]))),
                ret: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.fromList".into(), dict_from_list_scheme);

        // Dict.merge : forall a. Dict a -> Dict a -> Dict a
        let dict_merge_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                    ret: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.merge".into(), dict_merge_scheme);

        // Dict.getOrDefault : forall a. a -> String -> Dict a -> a
        let dict_get_or_default_scheme = Scheme {
            num_generics: 1,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::new_generic(0)),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::String),
                    ret: Rc::new(Type::Arrow {
                        arg: Rc::new(Type::Dict(Rc::new(Type::new_generic(0)))),
                        ret: Rc::new(Type::new_generic(0)),
                        effects: Row::Empty,
                    }),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Dict.getOrDefault".into(), dict_get_or_default_scheme);

        // html_escape : String -> String
        env.insert(
            "html_escape".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // json_escape_string : String -> String
        env.insert(
            "json_escape_string".into(),
            Scheme::mono(Type::arrow(Type::String, Type::String)),
        );

        // Set.new : () -> Set
        let set_new_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Unit),
                ret: Rc::new(Type::Set),
                effects: Row::Empty,
            },
        };
        env.insert("Set.new".into(), set_new_scheme);

        // Set.insert : String -> Set -> Set
        let set_insert_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::String),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Set),
                    ret: Rc::new(Type::Set),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Set.insert".into(), set_insert_scheme);

        // Set.contains : String -> Set -> Bool
        let set_contains_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::String),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Set),
                    ret: Rc::new(Type::Bool),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Set.contains".into(), set_contains_scheme);

        // Set.remove : String -> Set -> Set
        let set_remove_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::String),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Set),
                    ret: Rc::new(Type::Set),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Set.remove".into(), set_remove_scheme);

        // Set.union : forall t. Set -> Set -> Set
        let set_union_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Set),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Set),
                    ret: Rc::new(Type::Set),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Set.union".into(), set_union_scheme);

        // Set.intersect : Set -> Set -> Set
        let set_intersect_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Set),
                ret: Rc::new(Type::Arrow {
                    arg: Rc::new(Type::Set),
                    ret: Rc::new(Type::Set),
                    effects: Row::Empty,
                }),
                effects: Row::Empty,
            },
        };
        env.insert("Set.intersect".into(), set_intersect_scheme);

        // Set.size : Set -> Int
        let set_size_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Set),
                ret: Rc::new(Type::Int),
                effects: Row::Empty,
            },
        };
        env.insert("Set.size".into(), set_size_scheme);

        // Set.toList : Set -> [String]
        let set_tolist_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Set),
                ret: Rc::new(Type::list(Type::String)),
                effects: Row::Empty,
            },
        };
        env.insert("Set.toList".into(), set_tolist_scheme);

        // get_args : () -> [String]
        let get_args_scheme = Scheme {
            num_generics: 0,
            predicates: vec![],
            ty: Type::Arrow {
                arg: Rc::new(Type::Unit),
                ret: Rc::new(Type::list(Type::String)),
                effects: Row::Empty, // TODO: should have IO effect
            },
        };
        env.insert("get_args".into(), get_args_scheme);

        // Parse and inject prelude (Option, Result, id, const, flip)
        let prelude = parse_prelude().map_err(|e| vec![TypeError::Other(e)])?;

        // Collect all items: prelude first, then user program
        let all_items: Vec<&Item> = prelude
            .items
            .iter()
            .chain(program.items.iter())
            .collect();

        // First pass: register all type declarations (ADTs, records, and type aliases)
        for item in &all_items {
            if let Item::Decl(decl) = item {
                self.register_type_decl(decl);
                self.register_record_decl(decl);
                self.register_type_alias_decl(decl);
            }
        }

        // Second pass: register all trait declarations
        for item in &all_items {
            if let Item::Decl(decl @ Decl::Trait { .. }) = item {
                if let Err(e) = self.register_trait_decl(decl) {
                    self.record_error(e);
                }
            }
        }

        // Third pass: register all instance declarations
        for item in &all_items {
            if let Item::Decl(decl @ Decl::Instance { .. }) = item {
                if let Err(e) = self.register_instance_decl(decl) {
                    self.record_error(e);
                }
            }
        }

        // Fourth pass: infer types for let declarations and top-level expressions
        for item in &all_items {
            match item {
                Item::Decl(Decl::Let {
                    name, params, body, ..
                }) => {
                    // Use closure to allow early return on errors while continuing the loop
                    let result: Result<Scheme, TypeError> = (|| {
                        self.level += 1;

                        // Clear wanted predicates for this binding
                        self.wanted_preds.clear();

                        let mut local_env = env.clone();
                        let mut param_types = Vec::new();

                        // Check if there's a val declaration - if so, use its parameter types
                        let declared_param_types: Option<Vec<Type>> = env.get(name).map(|scheme| {
                            let ty = self.instantiate(scheme);
                            // Extract parameter types from the arrow chain
                            let mut types = Vec::new();
                            let mut current = ty;
                            for _ in 0..params.len() {
                                match current {
                                    Type::Arrow { arg, ret, .. } => {
                                        types.push((*arg).clone());
                                        current = (*ret).clone();
                                    }
                                    _ => break,
                                }
                            }
                            types
                        });

                        for (i, param) in params.iter().enumerate() {
                            let param_ty = if let Some(ref decl_types) = declared_param_types {
                                if i < decl_types.len() {
                                    decl_types[i].clone()
                                } else {
                                    self.fresh_var()
                                }
                            } else {
                                self.fresh_var()
                            };
                            self.bind_pattern(&mut local_env, param, &param_ty)?;
                            param_types.push(param_ty);
                        }

                        // For recursive functions, add the function to its own scope with a fresh type
                        let ret_ty = self.fresh_var();
                        let preliminary_func_ty = if param_types.is_empty() {
                            ret_ty.clone()
                        } else {
                            Type::arrows(param_types.clone(), ret_ty.clone())
                        };
                        local_env.insert(name.clone(), Scheme::mono(preliminary_func_ty));

                        // Use infer_expr_full to get answer type information
                        let body_result = self.infer_expr_full(&local_env, body)?;

                        // Unify the body type with what we predicted
                        self.unify_at(&ret_ty, &body_result.ty, &body.span)?;

                        // Build the function type with proper answer types
                        let func_ty = if param_types.is_empty() {
                            body_result.ty
                        } else {
                            // Build function type
                            // TODO: Track effects properly once effect inference is implemented
                            let mut result = Type::Arrow {
                                arg: Rc::new(param_types.pop().unwrap()),
                                ret: Rc::new(body_result.ty),
                                effects: Row::Empty, // Pure for now
                            };
                            // Wrap remaining params as pure arrows (returning a closure is pure)
                            for param_ty in param_types.into_iter().rev() {
                                result = Type::Arrow {
                                    arg: Rc::new(param_ty),
                                    ret: Rc::new(result),
                                    effects: Row::Empty, // Pure
                                };
                            }
                            result
                        };

                        self.level -= 1;

                        // TODO: In Phase 5, we'll discharge predicates here and check for unsatisfied constraints
                        // For now, predicates are collected but not checked

                        // Check against declared type signature if one exists (from `val` declaration)
                        if let Some(declared_scheme) = env.get(name) {
                            let declared_ty = self.instantiate(declared_scheme);
                            self.unify_at(&func_ty, &declared_ty, &body.span)?;
                        }

                        Ok(self.generalize(&func_ty))
                    })();

                    match result {
                        Ok(scheme) => {
                            env.insert(name.clone(), scheme);
                        }
                        Err(e) => {
                            self.record_error(e);
                            // Add error recovery type to prevent cascade errors
                            env.insert(name.clone(), Scheme::mono(self.fresh_var()));
                            // Ensure level is reset even on error
                            self.level = self.level.saturating_sub(1);
                        }
                    }
                }
                Item::Decl(Decl::Type { .. } | Decl::TypeAlias { .. } | Decl::Record { .. }) => {
                    // Already handled in first pass (type registration)
                }
                Item::Decl(Decl::Trait { .. } | Decl::Instance { .. }) => {
                    // Already handled in second/third pass
                }
                Item::Decl(Decl::EffectDecl { name, params, operations, .. }) => {
                    // Register the effect declaration in the effect environment
                    let param_strs: Vec<String> = params.to_vec();
                    if let Err(e) = self.register_effect(name, &param_strs, operations) {
                        self.record_error(e);
                    }
                }
                Item::Decl(Decl::Val { name, type_sig, constraints }) => {
                    // Register a type signature for the name
                    // The subsequent let declaration will be checked against this
                    let result: Result<Scheme, TypeError> = (|| {
                        self.level += 1;

                        // Collect type variable names and create fresh vars for each
                        let mut type_var_names = Vec::new();
                        Self::collect_type_vars(type_sig, &mut type_var_names);

                        let mut param_map: HashMap<String, TypeVarId> = HashMap::new();
                        let mut fresh_vars: Vec<Type> = Vec::new();
                        for name_str in &type_var_names {
                            let fresh = self.fresh_var();
                            // Extract the ID from the fresh var
                            if let Type::Var(id) = &fresh {
                                param_map.insert(name_str.clone(), *id);
                            }
                            fresh_vars.push(fresh);
                        }

                        let declared_ty =
                            self.type_expr_to_type_with_fresh_vars(type_sig, &type_var_names, &fresh_vars)?;

                        // Build predicates from constraints
                        let mut predicates = Vec::new();
                        for constraint in constraints {
                            // Look up the type var for this constraint
                            if let Some(idx) = type_var_names.iter().position(|n| n == &constraint.type_var) {
                                predicates.push(Pred {
                                    trait_name: constraint.trait_name.clone(),
                                    ty: fresh_vars[idx].clone(),
                                });
                            }
                        }

                        self.level -= 1;

                        // Generalize with predicates
                        let scheme = self.generalize(&declared_ty);
                        Ok(Scheme {
                            predicates,
                            ..scheme
                        })
                    })();

                    match result {
                        Ok(scheme) => {
                            env.insert(name.clone(), scheme);
                        }
                        Err(e) => {
                            self.record_error(e);
                            env.insert(name.clone(), Scheme::mono(self.fresh_var()));
                            self.level = self.level.saturating_sub(1);
                        }
                    }
                }
                Item::Decl(Decl::OperatorDef { op, params, body, .. }) => {
                    // Operator definitions work like function definitions
                    let result: Result<Scheme, TypeError> = (|| {
                        self.level += 1;
                        self.wanted_preds.clear();

                        let mut local_env = env.clone();
                        let mut param_types = Vec::new();

                        for param in params {
                            let param_ty = self.fresh_var();
                            self.bind_pattern(&mut local_env, param, &param_ty)?;
                            param_types.push(param_ty);
                        }

                        let body_result = self.infer_expr_full(&local_env, body)?;

                        let func_ty = if param_types.is_empty() {
                            body_result.ty
                        } else {
                            let mut result = Type::Arrow {
                                arg: Rc::new(param_types.pop().unwrap()),
                                ret: Rc::new(body_result.ty),
                                effects: Row::Empty, // Pure for now
                            };
                            for param_ty in param_types.into_iter().rev() {
                                result = Type::Arrow {
                                    arg: Rc::new(param_ty),
                                    ret: Rc::new(result),
                                    effects: Row::Empty, // Pure
                                };
                            }
                            result
                        };

                        self.level -= 1;
                        Ok(self.generalize(&func_ty))
                    })();

                    match result {
                        Ok(scheme) => {
                            env.insert(op.clone(), scheme);
                        }
                        Err(e) => {
                            self.record_error(e);
                            env.insert(op.clone(), Scheme::mono(self.fresh_var()));
                            self.level = self.level.saturating_sub(1);
                        }
                    }
                }
                Item::Decl(Decl::Fixity(_)) => {
                    // Fixity declarations are handled during parsing
                    // No type inference needed
                }
                Item::Decl(Decl::LetRec { bindings, .. }) => {
                    // Mutually recursive function definitions
                    // Wrap entire letrec in error handling
                    let result: Result<Vec<(String, Scheme)>, TypeError> = (|| {
                        self.level += 1;
                        self.wanted_preds.clear();

                        // Step 1: Create preliminary types for all bindings
                        // If there's a pre-existing val declaration, use that type to help break
                        // infinite type cycles in answer types
                        let mut local_env = env.clone();
                        let mut preliminary_types: Vec<Type> = Vec::new();

                        for binding in bindings {
                            // Check if there's an existing type signature from a val declaration
                            let preliminary_ty = if let Some(scheme) = env.get(&binding.name.node) {
                                // Use the declared type as the preliminary type
                                // This helps break infinite type cycles for recursive functions
                                self.instantiate(scheme)
                            } else {
                                // No declaration, use fresh variable
                                self.fresh_var()
                            };
                            preliminary_types.push(preliminary_ty.clone());
                            local_env.insert(binding.name.node.clone(), Scheme::mono(preliminary_ty));
                        }

                        // Step 2: Infer each binding's body
                        for (i, binding) in bindings.iter().enumerate() {
                            let mut body_env = local_env.clone();
                            let mut param_types = Vec::new();

                            // Check if there's a val declaration - if so, use its parameter types
                            let declared_param_types: Option<Vec<Type>> =
                                env.get(&binding.name.node).map(|scheme| {
                                    let ty = self.instantiate(scheme);
                                    let mut types = Vec::new();
                                    let mut current = ty;
                                    for _ in 0..binding.params.len() {
                                        match current {
                                            Type::Arrow { arg, ret, .. } => {
                                                types.push((*arg).clone());
                                                current = (*ret).clone();
                                            }
                                            _ => break,
                                        }
                                    }
                                    types
                                });

                            for (j, param) in binding.params.iter().enumerate() {
                                let param_ty = if let Some(ref decl_types) = declared_param_types {
                                    if j < decl_types.len() {
                                        decl_types[j].clone()
                                    } else {
                                        self.fresh_var()
                                    }
                                } else {
                                    self.fresh_var()
                                };
                                self.bind_pattern(&mut body_env, param, &param_ty)?;
                                param_types.push(param_ty);
                            }

                            let body_result = self.infer_expr_full(&body_env, &binding.body)?;

                            let func_ty = if param_types.is_empty() {
                                body_result.ty
                            } else {
                                let mut result = Type::Arrow {
                                    arg: Rc::new(param_types.pop().unwrap()),
                                    ret: Rc::new(body_result.ty),
                                    effects: Row::Empty, // Pure for now
                                };
                                for param_ty in param_types.into_iter().rev() {
                                    result = Type::Arrow {
                                        arg: Rc::new(param_ty),
                                        ret: Rc::new(result),
                                        effects: Row::Empty, // Pure
                                    };
                                }
                                result
                            };

                            self.unify_with_context(
                                &preliminary_types[i],
                                &func_ty,
                                &binding.name.span,
                                UnifyContext::RecursiveBinding {
                                    name: binding.name.node.clone(),
                                },
                            )?;
                        }

                        self.level -= 1;

                        // Generalize and return schemes
                        Ok(bindings.iter().enumerate().map(|(i, binding)| {
                            (binding.name.node.clone(), self.generalize(&preliminary_types[i]))
                        }).collect())
                    })();

                    match result {
                        Ok(schemes) => {
                            for (name, scheme) in schemes {
                                env.insert(name, scheme);
                            }
                        }
                        Err(e) => {
                            self.record_error(e);
                            // Add error recovery types for all bindings
                            for binding in bindings {
                                env.insert(binding.name.node.clone(), Scheme::mono(self.fresh_var()));
                            }
                            self.level = self.level.saturating_sub(1);
                        }
                    }
                }
                Item::Import(import_spec) => {
                    // Process imports to populate the import mappings
                    let module_name = &import_spec.node.module_path;

                    // Handle module alias: import Module as Alias
                    if let Some(alias) = &import_spec.node.alias {
                        self.add_module_alias(alias.clone(), module_name.clone());
                    }

                    // Handle selective imports: import Module (x, y as z)
                    if let Some(items) = &import_spec.node.items {
                        for (name, alias) in items {
                            let local_name = alias.as_ref().unwrap_or(name).clone();
                            self.add_import(local_name, module_name.clone(), name.clone());
                        }
                    }
                    // Note: For `import Module` without selective imports, qualified access
                    // (Module.name) is handled by lookup_name checking module_envs directly
                }
                Item::Expr(expr) => {
                    // Type-check top-level expressions
                    if let Err(e) = self.infer_expr(&env, expr) {
                        self.record_error(e);
                    }
                }
            }
        }

        // Return errors if any were accumulated
        if self.has_errors() {
            Err(self.take_errors())
        } else {
            Ok(env)
        }
    }

    /// Get the wanted predicates (for testing)
    #[cfg(test)]
    pub fn get_wanted_preds(&self) -> &[Pred] {
        &self.wanted_preds
    }
}

impl Default for Inferencer {
    fn default() -> Self {
        Self::new()
    }
}

// Public test helpers for property-based testing
impl Inferencer {
    /// Public wrapper for unify, for property testing
    pub fn unify_types(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        self.unify(t1, t2)
    }

    /// Get the class environment (for passing to the interpreter)
    pub fn take_class_env(&mut self) -> ClassEnv {
        std::mem::take(&mut self.class_env)
    }

    /// Get a reference to the class environment
    pub fn class_env(&self) -> &ClassEnv {
        &self.class_env
    }

    /// Get the type context (for passing to the interpreter)
    pub fn take_type_ctx(&mut self) -> TypeContext {
        std::mem::take(&mut self.type_ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn infer(input: &str) -> Result<Type, TypeError> {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expr().unwrap();
        let mut inferencer = Inferencer::new();
        let env = TypeEnv::new();
        inferencer.infer_expr(&env, &expr)
    }

    #[test]
    fn test_literal() {
        let ty = infer("42").unwrap();
        assert!(matches!(ty, Type::Int));
    }

    #[test]
    fn test_lambda() {
        let ty = infer("fun x -> x").unwrap();
        assert!(matches!(ty, Type::Arrow { .. }));
    }

    #[test]
    fn test_application() {
        let ty = infer("(fun x -> x + 1) 42").unwrap();
        assert!(matches!(ty, Type::Int));
    }

    #[test]
    fn test_if() {
        let ty = infer("if true then 1 else 2").unwrap();
        assert!(matches!(ty, Type::Int));
    }

    fn typecheck_program(input: &str) -> Result<TypeEnv, Vec<TypeError>> {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();
        let mut inferencer = Inferencer::new();
        inferencer.infer_program(&program)
    }

    #[test]
    fn test_select_type_inference_consistent() {
        // All arms must return same type - this should succeed
        let good = r#"
let f ch1 ch2 =
    select
    | x <- ch1 -> x + 1
    | y <- ch2 -> y + 2
    end
"#;
        assert!(typecheck_program(good).is_ok());
    }

    #[test]
    fn test_select_type_inference_mismatch() {
        // Mixed return types should fail
        let bad = r#"
let f ch1 ch2 =
    select
    | x <- ch1 -> x + 1
    | y <- ch2 -> true
    end
"#;
        assert!(typecheck_program(bad).is_err());
    }

    // Note: Delimited continuation (shift/reset) type tests were removed when shift/reset
    // was replaced by algebraic effects. Use handle/perform for effect-based control flow.

    // Tests for generalize_inner and substitute returning resolved types (bug4.md)
    #[test]
    fn test_generalize_follows_links() {
        // This test ensures that generalization properly resolves linked type variables
        let source = r#"
let f = fun x ->
    let y = x in
    y
"#;
        let result = typecheck_program(source);
        assert!(result.is_ok());

        if let Ok(env) = result {
            if let Some(scheme) = env.get("f") {
                // The type string should be clean, not contain internal var references (t0, t1, etc.)
                let ty_str = format!("{}", scheme);
                assert!(
                    !ty_str.contains("t"),
                    "Type should be generalized without linked vars: {}",
                    ty_str
                );
            }
        }
    }

    #[test]
    fn test_unification_then_generalize() {
        // Create a scenario where unification creates links, then generalize
        let source = r#"
let apply f x = f x
let id = fun x -> x
let result = apply id 42
"#;
        let result = typecheck_program(source);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Phase 4 Typeclass Tests
    // ========================================================================

    fn typecheck_program_with_inferencer(input: &str) -> (Result<TypeEnv, Vec<TypeError>>, Inferencer) {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();
        let mut inferencer = Inferencer::new();
        let result = inferencer.infer_program(&program);
        (result, inferencer)
    }

    #[test]
    fn test_trait_registration() {
        let source = r#"
trait Show a =
    val show : a -> String
end
"#;
        let (result, inferencer) = typecheck_program_with_inferencer(source);
        assert!(result.is_ok());
        assert!(inferencer.class_env.get_trait("Show").is_some());
    }

    #[test]
    fn test_trait_method_lookup() {
        let source = r#"
trait Show a =
    val show : a -> String
end
"#;
        let (_, inferencer) = typecheck_program_with_inferencer(source);
        let lookup = inferencer.class_env.lookup_method("show");
        assert!(lookup.is_some());
        let (trait_name, _) = lookup.unwrap();
        assert_eq!(trait_name, "Show");
    }

    #[test]
    fn test_instance_registration() {
        // Note: Use Display instead of Show since Show is now in prelude
        let source = r#"
trait Display a =
    val display : a -> String
end
impl Display for Int =
    let display n = int_to_string n
end
"#;
        let (result, inferencer) = typecheck_program_with_inferencer(source);
        assert!(result.is_ok());
        // Prelude adds Show instances, so we check Display was added
        let display_instances: Vec<_> = inferencer
            .class_env
            .instances
            .iter()
            .filter(|i| i.trait_name == "Display")
            .collect();
        assert_eq!(display_instances.len(), 1);
    }

    #[test]
    fn test_method_call_adds_predicate() {
        // When we call 'display x', it should add a Display predicate for x's type
        // Note: Use Display instead of Show since Show is now in prelude
        let source = r#"
trait Display a =
    val display : a -> String
end
let f x = display x
"#;
        let (result, inferencer) = typecheck_program_with_inferencer(source);
        assert!(result.is_ok());
        // The inferencer collects predicates during inference
        // Note: predicates are cleared per binding, so we can't directly check them here
        // This test mainly verifies the code path works
        assert!(inferencer.class_env.get_trait("Display").is_some());
    }

    #[test]
    fn test_trait_unknown_error() {
        // Instance for unknown trait should fail
        // Note: Use UnknownTrait instead of Show since Show is now in prelude
        let source = r#"
impl UnknownTrait for Int =
    let method n = int_to_string n
end
"#;
        let (result, _) = typecheck_program_with_inferencer(source);
        assert!(result.is_err());
        if let Err(errors) = result {
            assert!(!errors.is_empty());
            if let TypeError::UnknownTrait { name, .. } = &errors[0] {
                assert_eq!(name, "UnknownTrait");
            } else {
                panic!("Expected UnknownTrait error, got {:?}", errors[0]);
            }
        }
    }

    #[test]
    fn test_overlapping_instance_error() {
        // Two instances for the same type should fail
        // Note: Use Display instead of Show since Show is now in prelude
        let source = r#"
trait Display a =
    val display : a -> String
end
impl Display for Int =
    let display n = int_to_string n
end
impl Display for Int =
    let display n = "int"
end
"#;
        let (result, _) = typecheck_program_with_inferencer(source);
        assert!(result.is_err());
        if let Err(errors) = result {
            assert!(!errors.is_empty());
            assert!(matches!(errors[0], TypeError::OverlappingInstance { .. }));
        }
    }

    #[test]
    fn test_constrained_instance_registration() {
        // Note: Use Display instead of Show since Show is now in prelude
        let source = r#"
trait Display a =
    val display : a -> String
end
impl Display for Int =
    let display n = int_to_string n
end
type MyList a = | Nil | Cons a (MyList a)
impl Display for (MyList a) where a : Display =
    let display xs = "list"
end
"#;
        let (result, inferencer) = typecheck_program_with_inferencer(source);
        assert!(result.is_ok());
        // Filter for Display instances (prelude adds Show instances)
        let display_instances: Vec<_> = inferencer
            .class_env
            .instances
            .iter()
            .filter(|i| i.trait_name == "Display")
            .collect();
        // Should have 2 Display instances: Display Int and Display (MyList a)
        assert_eq!(display_instances.len(), 2);

        // Check that the MyList instance has constraints
        let list_instance = display_instances
            .iter()
            .find(|i| !i.constraints.is_empty())
            .expect("Should have constrained instance");
        assert_eq!(list_instance.constraints.len(), 1);
        assert_eq!(list_instance.constraints[0].trait_name, "Display");
    }

    // ========================================================================
    // Module / Import Tests
    // ========================================================================

    #[test]
    fn test_unused_import_tracking() {
        // Manually set up imports and check unused detection
        let mut inferencer = Inferencer::new();

        // Create a fake module environment
        let mut module_env = TypeEnv::new();
        module_env.insert("foo".to_string(), Scheme::mono(Type::Int));
        module_env.insert("bar".to_string(), Scheme::mono(Type::String));
        inferencer.register_module("TestModule".to_string(), module_env);

        // Add imports for both items
        inferencer.add_import("foo".to_string(), "TestModule".to_string(), "foo".to_string());
        inferencer.add_import("bar".to_string(), "TestModule".to_string(), "bar".to_string());

        // Look up only 'foo', leaving 'bar' unused
        let env = TypeEnv::new();
        let _ = inferencer.lookup_name(&env, "foo");

        // Check unused imports
        let unused = inferencer.get_unused_imports();
        assert_eq!(unused.len(), 1);
        assert!(unused.contains(&"bar".to_string()));
    }

    #[test]
    fn test_all_imports_used() {
        let mut inferencer = Inferencer::new();

        let mut module_env = TypeEnv::new();
        module_env.insert("x".to_string(), Scheme::mono(Type::Int));
        module_env.insert("y".to_string(), Scheme::mono(Type::Int));
        inferencer.register_module("M".to_string(), module_env);

        inferencer.add_import("x".to_string(), "M".to_string(), "x".to_string());
        inferencer.add_import("y".to_string(), "M".to_string(), "y".to_string());

        // Use both imports
        let env = TypeEnv::new();
        let _ = inferencer.lookup_name(&env, "x");
        let _ = inferencer.lookup_name(&env, "y");

        // No unused imports
        let unused = inferencer.get_unused_imports();
        assert!(unused.is_empty());
    }

    #[test]
    fn test_unused_module_alias() {
        let mut inferencer = Inferencer::new();

        let mut module_env = TypeEnv::new();
        module_env.insert("func".to_string(), Scheme::mono(Type::Int));
        inferencer.register_module("LongModuleName".to_string(), module_env);

        // Add an alias that is never used
        inferencer.add_module_alias("L".to_string(), "LongModuleName".to_string());

        // Don't use qualified access via alias at all
        let unused_aliases = inferencer.get_unused_module_aliases();
        assert_eq!(unused_aliases.len(), 1);
        assert!(unused_aliases.contains(&"L".to_string()));
    }

    #[test]
    fn test_module_alias_used() {
        let mut inferencer = Inferencer::new();

        let mut module_env = TypeEnv::new();
        module_env.insert("func".to_string(), Scheme::mono(Type::Int));
        inferencer.register_module("LongModuleName".to_string(), module_env);

        inferencer.add_module_alias("L".to_string(), "LongModuleName".to_string());

        // Use qualified access via alias
        let env = TypeEnv::new();
        let _ = inferencer.lookup_name(&env, "L.func");

        let unused_aliases = inferencer.get_unused_module_aliases();
        assert!(unused_aliases.is_empty());
    }

    #[test]
    fn test_clear_imports_resets_usage() {
        let mut inferencer = Inferencer::new();

        let mut module_env = TypeEnv::new();
        module_env.insert("x".to_string(), Scheme::mono(Type::Int));
        inferencer.register_module("M".to_string(), module_env);
        inferencer.add_import("x".to_string(), "M".to_string(), "x".to_string());

        let env = TypeEnv::new();
        let _ = inferencer.lookup_name(&env, "x");
        assert!(inferencer.get_unused_imports().is_empty());

        // Clear should reset everything
        inferencer.clear_imports();
        assert!(inferencer.get_unused_imports().is_empty()); // No imports anymore
    }

    // ========================================================================
    // Error Accumulation Tests
    // ========================================================================

    #[test]
    fn test_multiple_type_errors_accumulated() {
        // Program with multiple independent type errors
        // Each binding has an expression that doesn't match the declared type
        let source = r#"
val x : Int
let x = "hello"
val y : Int
let y = true
val z : String
let z = 42
"#;
        let result = typecheck_program(source);
        assert!(result.is_err());
        if let Err(errors) = result {
            // Should have 3 errors, one for each mismatched binding
            assert_eq!(errors.len(), 3, "Expected 3 errors, got {}: {:?}", errors.len(), errors);
        }
    }

    #[test]
    fn test_error_accumulation_continues_after_unbound_variable() {
        // Two different unbound variables - should get errors for both
        let source = r#"
let x = unknownVar1
let y = unknownVar2
"#;
        let result = typecheck_program(source);
        assert!(result.is_err());
        if let Err(errors) = result {
            // Should have 2 errors for the two unbound variables
            assert_eq!(errors.len(), 2, "Expected 2 errors, got {}: {:?}", errors.len(), errors);
            for err in &errors {
                assert!(matches!(err, TypeError::UnboundVariable { .. }));
            }
        }
    }

    #[test]
    fn test_error_recovery_prevents_cascades() {
        // After an error, subsequent bindings that use the errored name
        // should not cause cascade errors (we insert a fresh type)
        let source = r#"
let x = unknownVar
let y = x + 1
"#;
        let result = typecheck_program(source);
        assert!(result.is_err());
        if let Err(errors) = result {
            // Should only have 1 error for unknownVar
            // The use of x in y should not cause another error
            // because we add a recovery type for x
            assert_eq!(errors.len(), 1, "Expected 1 error (not cascade), got {}: {:?}", errors.len(), errors);
        }
    }

    #[test]
    fn test_single_error_still_works() {
        // Verify single errors still work as expected
        let source = "let x = 1 + \"hello\"";
        let result = typecheck_program(source);
        assert!(result.is_err());
        if let Err(errors) = result {
            assert_eq!(errors.len(), 1);
            assert!(matches!(errors[0], TypeError::TypeMismatch { .. }));
        }
    }
}
