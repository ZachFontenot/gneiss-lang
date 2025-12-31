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

    /// Effect operation: perform Effect.op args
    Perform {
        effect: Ident,
        op: Ident,
        args: Vec<MonoExpr>,
    },

    /// Effect handler: handle body with handlers
    Handle {
        body: Box<MonoExpr>,
        handler: MonoHandler,
    },

    // ========================================================================
    // Dictionary-based trait method dispatch
    // ========================================================================

    /// Call a trait method through a dictionary
    /// E.g., `show x` where Show dict is passed as parameter
    DictCall {
        trait_name: String,
        method: String,
        /// The dictionary to use (either DictParam or DictBuild)
        dict: Box<MonoExpr>,
        /// Arguments to the method
        args: Vec<MonoExpr>,
    },

    /// Reference to a dictionary parameter of the current function
    /// E.g., in `let f dict x = dict.show x`, this references `dict`
    DictParam {
        trait_name: String,
        /// Index in the function's dict_params list
        param_idx: usize,
    },

    /// Build a dictionary for a concrete type
    /// For simple instances: returns static dict (e.g., Show_Int)
    /// For parametric instances: calls factory with sub-dicts (e.g., make_Show_List(Show_Int))
    DictBuild {
        trait_name: String,
        /// The concrete type this dict is for
        for_type: MonoType,
        /// Sub-dictionaries needed (for parametric instances)
        /// E.g., Show (List Int) needs Show Int dict
        sub_dicts: Vec<MonoExpr>,
    },

    /// Error placeholder
    Error(String),
}

// ============================================================================
// Monomorphized Effect Handlers
// ============================================================================

/// Monomorphized effect handler
#[derive(Debug, Clone)]
pub struct MonoHandler {
    /// Effect being handled (None for catch-all)
    pub effect: Option<Ident>,
    /// Return clause: return x -> body
    pub return_clause: Option<MonoHandlerClause>,
    /// Operation handlers
    pub op_clauses: Vec<MonoOpClause>,
}

/// Handler clause for return
#[derive(Debug, Clone)]
pub struct MonoHandlerClause {
    pub pattern: MonoPattern,
    pub body: Box<MonoExpr>,
}

/// Handler clause for an operation
#[derive(Debug, Clone)]
pub struct MonoOpClause {
    pub op_name: Ident,
    pub params: Vec<MonoPattern>,
    pub continuation: Ident,
    pub body: Box<MonoExpr>,
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

/// A dictionary parameter for a function
#[derive(Debug, Clone)]
pub struct MonoDictParam {
    pub trait_name: String,
    /// The type variable index this dict is for (from the original scheme)
    pub type_var: u32,
}

#[derive(Debug, Clone)]
pub struct MonoFn {
    pub id: MonoFnId,
    /// Dictionary parameters (passed before regular params)
    pub dict_params: Vec<MonoDictParam>,
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
    /// Current function's dictionary parameters (for resolving DictRef)
    /// Maps type_var_id -> (trait_name, param_index)
    current_dict_params: HashMap<u32, (String, usize)>,
    /// When specializing an instance method, this maps method names to their
    /// concrete implementations. E.g., "show" -> "Show_Int_show"
    instance_method_subst: HashMap<String, String>,
}

impl<'a> MonoCtx<'a> {
    pub fn new(tprogram: &'a TProgram) -> Self {
        MonoCtx {
            tprogram,
            work_queue: VecDeque::new(),
            done: HashMap::new(),
            subst: HashMap::new(),
            in_progress: HashSet::new(),
            current_dict_params: HashMap::new(),
            instance_method_subst: HashMap::new(),
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

    /// Convert a MonoType back to a Type (for matching against instances)
    fn subst_to_type(&self, mono: &MonoType) -> Type {
        match mono {
            MonoType::Int => Type::Int,
            MonoType::Float => Type::Float,
            MonoType::Bool => Type::Bool,
            MonoType::String => Type::String,
            MonoType::Char => Type::Char,
            MonoType::Unit => Type::Unit,
            MonoType::Bytes => Type::Bytes,
            MonoType::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.subst_to_type(e)).collect())
            }
            MonoType::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args.iter().map(|a| self.subst_to_type(a)).collect(),
            },
            MonoType::Function { params, ret } => {
                let mut result = self.subst_to_type(ret);
                for p in params.iter().rev() {
                    result = Type::Arrow {
                        arg: std::rc::Rc::new(self.subst_to_type(p)),
                        ret: std::rc::Rc::new(result),
                        effects: crate::types::Row::Empty,
                    };
                }
                result
            }
        }
    }

    /// Build a dictionary expression for a concrete type.
    /// For simple instances (Show Int), returns DictBuild with no sub-dicts.
    /// For parametric instances (Show (List Int)), returns DictBuild with sub-dicts.
    fn build_dict_for_type(&mut self, trait_name: &str, instance_ty: &Type) -> MonoExpr {
        let mono_ty = self.mono_type(instance_ty);
        let span = Span::default();

        // Find the instance that matches this type
        for instance in &self.tprogram.instance_decls {
            if instance.trait_name != trait_name {
                continue;
            }

            // Try to match
            if let Some(type_subst) = match_instance_type(instance_ty, &instance.instance_ty) {
                // Found matching instance
                // For parametric instances, build sub-dicts for each type parameter
                let mut sub_dicts = Vec::new();

                // If instance has generics, we need sub-dicts and specialized methods
                if type_has_generics(&instance.instance_ty) {
                    // Get the constraints from the original instance
                    // For each Generic in instance_ty, we need a sub-dict
                    // The type_subst tells us what concrete type each generic maps to
                    for (_generic_id, concrete_mono_ty) in &type_subst {
                        // Find what trait constraint this generic has
                        // For now, assume same trait (Show (List a) where a: Show)
                        // TODO: look up actual constraints from instance decl
                        let concrete_ty = self.subst_to_type(concrete_mono_ty);
                        let sub_dict = self.build_dict_for_type(trait_name, &concrete_ty);
                        sub_dicts.push(sub_dict);
                    }

                    // Trigger specialization for all methods in this instance
                    for method in &instance.methods.clone() {
                        let mangled = format!("{}_{}_{}", trait_name, mono_ty.mangle(), method.name);
                        let fn_id = MonoFnId::new(&mangled);
                        if !self.done.contains_key(&fn_id) && !self.in_progress.contains(&fn_id) {
                            self.try_specialize_instance_method(
                                trait_name,
                                &method.name,
                                instance_ty,
                                &fn_id,
                            );
                        }
                    }
                }

                return MonoExpr::new(
                    MonoExprKind::DictBuild {
                        trait_name: trait_name.to_string(),
                        for_type: mono_ty,
                        sub_dicts,
                    },
                    MonoType::Unit, // dict type is opaque
                    span,
                );
            }
        }

        // No instance found - error
        MonoExpr::new(
            MonoExprKind::Error(format!(
                "No {} instance for {:?}",
                trait_name, instance_ty
            )),
            MonoType::Unit,
            span,
        )
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
            TExprKind::Var(name) => {
                // Check if this variable should be substituted (e.g., "show" -> "Show_Int_show")
                if let Some(concrete_name) = self.instance_method_subst.get(name) {
                    MonoExprKind::Var(concrete_name.clone())
                } else {
                    MonoExprKind::Var(name.clone())
                }
            }

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
                // Check if the argument is a dictionary node - if so, extract type info and skip it
                // Dictionary arguments provide type information for monomorphization
                match &arg.node {
                    TExprKind::DictValue { instance_ty, type_var, .. } => {
                        // The DictValue directly provides the type variable mapping
                        let concrete = self.mono_type(instance_ty);
                        self.subst.insert(*type_var, concrete.clone());

                        // Check if func is a reference to a top-level polymorphic function
                        // If so, we need to create a specialized version
                        // Note: func might be nested in App nodes if there are multiple dict args
                        let base_name = find_base_function_name(func);
                        if let Some(name) = base_name {
                            if let Some(binding) = self.find_binding(&name).cloned() {
                                // Check if this binding contains dictionary-based calls
                                // (i.e., is polymorphic and needs specialization)
                                if binding_needs_specialization(&binding) {
                                    // Create specialized function name
                                    let specialized_name = format!("{}_{}", name, concrete.mangle());
                                    let fn_id = MonoFnId::new(&specialized_name);

                                    // If not already processed, monomorphize it with current subst
                                    if !self.done.contains_key(&fn_id) && !self.in_progress.contains(&fn_id) {
                                        self.in_progress.insert(fn_id.clone());

                                        // Extract the actual type variable used in the binding's body
                                        // (may differ from scheme's Generic ID)
                                        if let Some(body_type_var) = extract_body_type_var(&binding.body) {
                                            self.subst.insert(body_type_var, concrete.clone());
                                        }

                                        // Monomorphize the binding's body with the type substitution
                                        let mono_params: Vec<_> = binding
                                            .params
                                            .iter()
                                            .map(|p| self.mono_pattern(p))
                                            .collect();
                                        let mono_body = self.mono_expr(&binding.body);
                                        let return_type = self.mono_type(&binding.ty);

                                        let mono_fn = MonoFn {
                                            id: fn_id.clone(),
                                            dict_params: vec![],
                                            params: mono_params,
                                            body: mono_body,
                                            return_type,
                                        };

                                        self.in_progress.remove(&fn_id);
                                        self.done.insert(fn_id, mono_fn);
                                    }

                                    // Return a reference to the specialized function
                                    return MonoExpr {
                                        node: MonoExprKind::Var(specialized_name),
                                        ty,
                                        span,
                                    };
                                }
                            }
                        }

                        // Fall through: just process func normally
                        self.mono_expr(func).node
                    }
                    TExprKind::DictRef { type_var, .. } => {
                        // DictRef references a dictionary from enclosing scope
                        // The type variable should already be in subst from an outer DictValue

                        // Look up the concrete type from subst
                        if let Some(concrete) = self.subst.get(type_var).cloned() {

                            // Check if func is a reference to a polymorphic function
                            if let TExprKind::Var(name) = &func.node {
                                if let Some(binding) = self.find_binding(name).cloned() {
                                    if binding_needs_specialization(&binding) {
                                        // Create specialized version
                                        let specialized_name = format!("{}_{}", name, concrete.mangle());
                                        let fn_id = MonoFnId::new(&specialized_name);

                                        if !self.done.contains_key(&fn_id) && !self.in_progress.contains(&fn_id) {
                                            self.in_progress.insert(fn_id.clone());

                                            // Extract the body's type var and map it
                                            if let Some(body_type_var) = extract_body_type_var(&binding.body) {
                                                self.subst.insert(body_type_var, concrete.clone());
                                            }

                                            let mono_params: Vec<_> = binding
                                                .params
                                                .iter()
                                                .map(|p| self.mono_pattern(p))
                                                .collect();
                                            let mono_body = self.mono_expr(&binding.body);
                                            let return_type = self.mono_type(&binding.ty);

                                            let mono_fn = MonoFn {
                                                id: fn_id.clone(),
                                                dict_params: vec![],
                                                params: mono_params,
                                                body: mono_body,
                                                return_type,
                                            };

                                            self.in_progress.remove(&fn_id);
                                            self.done.insert(fn_id, mono_fn);
                                        }

                                        return MonoExpr {
                                            node: MonoExprKind::Var(specialized_name),
                                            ty,
                                            span,
                                        };
                                    }
                                }
                            }
                        }

                        // Fall through: just process func normally
                        self.mono_expr(func).node
                    }
                    _ => {
                        let mono_func = self.mono_expr(func);
                        let mono_arg = self.mono_expr(arg);
                        MonoExprKind::App {
                            func: Box::new(mono_func),
                            arg: Box::new(mono_arg),
                        }
                    }
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
                // Concrete type is known - build dictionary and call through it
                let mono_ty = self.mono_type(instance_ty);
                let mono_args: Vec<_> = args.iter().map(|a| self.mono_expr(a)).collect();

                // Build the dictionary for this concrete type
                let dict = self.build_dict_for_type(trait_name, instance_ty);

                MonoExprKind::DictCall {
                    trait_name: trait_name.clone(),
                    method: method.clone(),
                    dict: Box::new(dict),
                    args: mono_args,
                }
            }

            TExprKind::DictMethodCall {
                trait_name,
                method,
                type_var,
                args,
            } => {
                eprintln!("DEBUG: DictMethodCall {}.{} type_var={} subst={:?}", trait_name, method, type_var, self.subst);
                // Type is a parameter - look up the dict param
                let mono_args: Vec<_> = args.iter().map(|a| self.mono_expr(a)).collect();

                // Check if we have a dict param for this type var
                if let Some((param_trait, param_idx)) = self.current_dict_params.get(type_var) {
                    MonoExprKind::DictCall {
                        trait_name: trait_name.clone(),
                        method: method.clone(),
                        dict: Box::new(MonoExpr::new(
                            MonoExprKind::DictParam {
                                trait_name: param_trait.clone(),
                                param_idx: *param_idx,
                            },
                            MonoType::Unit, // dict type is opaque
                            expr.span.clone(),
                        )),
                        args: mono_args,
                    }
                } else if let Some(mono_ty) = self.subst.get(type_var) {
                    // Type var is bound to a concrete type in current subst
                    // This happens when specializing a polymorphic function
                    let concrete_ty = self.subst_to_type(mono_ty);
                    let dict = self.build_dict_for_type(trait_name, &concrete_ty);
                    MonoExprKind::DictCall {
                        trait_name: trait_name.clone(),
                        method: method.clone(),
                        dict: Box::new(dict),
                        args: mono_args,
                    }
                } else {
                    MonoExprKind::Error(format!(
                        "Unresolved dict for type var {} in {}",
                        type_var, trait_name
                    ))
                }
            }

            TExprKind::DictValue {
                trait_name,
                instance_ty,
                ..
            } => {
                // Build a dictionary for the given type
                self.build_dict_for_type(trait_name, instance_ty).node
            }

            TExprKind::DictRef {
                trait_name,
                type_var,
            } => {
                // Reference to a dict param
                if let Some((param_trait, param_idx)) = self.current_dict_params.get(type_var) {
                    MonoExprKind::DictParam {
                        trait_name: param_trait.clone(),
                        param_idx: *param_idx,
                    }
                } else {
                    MonoExprKind::Error(format!(
                        "Unresolved dict ref for type var {} in {}",
                        type_var, trait_name
                    ))
                }
            }

            TExprKind::Perform { effect, op, args } => {
                // Preserve effect structure for later CPS transformation
                let mono_args: Vec<_> = args.iter().map(|a| self.mono_expr(a)).collect();
                MonoExprKind::Perform {
                    effect: effect.clone(),
                    op: op.clone(),
                    args: mono_args,
                }
            }

            TExprKind::Handle { body, handler } => {
                // Preserve handler structure for later CPS transformation
                let mono_body = Box::new(self.mono_expr(body));
                let mono_handler = self.mono_handler(handler);
                MonoExprKind::Handle {
                    body: mono_body,
                    handler: mono_handler,
                }
            }

            TExprKind::Error(msg) => MonoExprKind::Error(msg.clone()),
        };

        MonoExpr::new(node, ty, span)
    }

    /// Monomorphize an effect handler
    fn mono_handler(&mut self, handler: &crate::tast::THandler) -> MonoHandler {
        let return_clause = handler.return_clause.as_ref().map(|clause| {
            MonoHandlerClause {
                pattern: self.mono_pattern(&clause.pattern),
                body: Box::new(self.mono_expr(&clause.body)),
            }
        });

        let op_clauses = handler.op_clauses.iter().map(|clause| {
            MonoOpClause {
                op_name: clause.op_name.clone(),
                params: clause.params.iter().map(|p| self.mono_pattern(p)).collect(),
                continuation: clause.continuation.clone(),
                body: Box::new(self.mono_expr(&clause.body)),
            }
        }).collect();

        MonoHandler {
            effect: handler.effect.clone(),
            return_clause,
            op_clauses,
        }
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

    /// Try to specialize a generic instance method for a concrete type.
    /// E.g., for Show_List_Int_show, finds Show (List a) and specializes with a->Int.
    /// Returns true if successful (function added to done), false otherwise.
    fn try_specialize_instance_method(
        &mut self,
        trait_name: &str,
        method_name: &str,
        concrete_ty: &Type,
        fn_id: &MonoFnId,
    ) -> bool {
        // Find a matching generic instance
        for instance in &self.tprogram.instance_decls {
            if instance.trait_name != trait_name {
                continue;
            }
            // Only consider generic instances (those with type params)
            if !type_has_generics(&instance.instance_ty) {
                continue;
            }

            // Try to match the concrete type against the instance's pattern
            if let Some(type_subst) = match_instance_type(concrete_ty, &instance.instance_ty) {
                // Found a match! Now find the method
                for method in &instance.methods {
                    if method.name != method_name {
                        continue;
                    }

                    // Save current subst, apply new one for this specialization
                    let old_subst = std::mem::replace(&mut self.subst, type_subst.clone());

                    // Set up method name substitutions for constrained methods
                    // E.g., in Show (List a) where a : Show, "show" -> "Show_Int_show"
                    let old_method_subst = std::mem::take(&mut self.instance_method_subst);
                    for (_generic_id, mono_ty) in &type_subst {
                        // For each trait method from constraints, set up the substitution
                        // TODO: properly look up which methods come from constraints
                        // For now, we assume the same trait's method
                        let concrete_method = format!("{}_{}_{}", trait_name, mono_ty.mangle(), method_name);
                        self.instance_method_subst.insert(method_name.to_string(), concrete_method);
                    }

                    // For each type parameter, set up dict_param mapping
                    // This allows DictMethodCall in the body to resolve correctly
                    let old_dict_params = std::mem::take(&mut self.current_dict_params);
                    for (generic_id, _mono_ty) in &type_subst {
                        self.current_dict_params.insert(*generic_id, (trait_name.to_string(), 0));
                    }

                    self.in_progress.insert(fn_id.clone());

                    // Monomorphize the method with the new substitution
                    let mono_params: Vec<_> = method
                        .params
                        .iter()
                        .map(|p| self.mono_pattern(p))
                        .collect();
                    let mono_body = self.mono_expr(&method.body);
                    let return_type = self.mono_type(&method.body.ty);

                    // Restore context
                    self.current_dict_params = old_dict_params;
                    self.instance_method_subst = old_method_subst;

                    let mono_fn = MonoFn {
                        id: fn_id.clone(),
                        dict_params: vec![], // TODO: parametric instances need dict params
                        params: mono_params,
                        body: mono_body,
                        return_type,
                    };

                    // Restore old subst
                    self.subst = old_subst;
                    self.in_progress.remove(fn_id);
                    self.done.insert(fn_id.clone(), mono_fn);

                    return true;
                }
            }
        }
        false
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

/// Try to match a concrete type against a generic instance type pattern.
/// Returns a substitution map if successful (generic_id -> MonoType).
/// E.g., matching `List Int` against `List a` returns {0 -> Int}
fn match_instance_type(concrete: &Type, pattern: &Type) -> Option<HashMap<u32, MonoType>> {
    let mut subst = HashMap::new();
    if match_type_pattern(concrete, pattern, &mut subst) {
        Some(subst)
    } else {
        None
    }
}

/// Recursive helper for matching types
fn match_type_pattern(concrete: &Type, pattern: &Type, subst: &mut HashMap<u32, MonoType>) -> bool {
    match (concrete, pattern) {
        // Generic matches anything, binding the type variable
        (concrete, Type::Generic(id)) => {
            let mono = MonoType::from_type(concrete, &HashMap::new());
            if let Some(existing) = subst.get(id) {
                // Must match existing binding
                *existing == mono
            } else {
                subst.insert(*id, mono);
                true
            }
        }
        // Constructor types must match name and recursively match args
        (
            Type::Constructor { name: n1, args: a1 },
            Type::Constructor { name: n2, args: a2 },
        ) => {
            n1 == n2 && a1.len() == a2.len()
                && a1.iter().zip(a2.iter()).all(|(c, p)| match_type_pattern(c, p, subst))
        }
        // Arrow types
        (
            Type::Arrow { arg: a1, ret: r1, .. },
            Type::Arrow { arg: a2, ret: r2, .. },
        ) => {
            match_type_pattern(a1, a2, subst) && match_type_pattern(r1, r2, subst)
        }
        // Tuples
        (Type::Tuple(t1), Type::Tuple(t2)) => {
            t1.len() == t2.len()
                && t1.iter().zip(t2.iter()).all(|(c, p)| match_type_pattern(c, p, subst))
        }
        // Wrapper types
        (Type::Channel(c1), Type::Channel(c2)) => match_type_pattern(c1, c2, subst),
        (Type::Fiber(c1), Type::Fiber(c2)) => match_type_pattern(c1, c2, subst),
        (Type::Dict(c1), Type::Dict(c2)) => match_type_pattern(c1, c2, subst),
        // Primitive types must match exactly
        (Type::Int, Type::Int) => true,
        (Type::Float, Type::Float) => true,
        (Type::String, Type::String) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::Char, Type::Char) => true,
        (Type::Unit, Type::Unit) => true,
        (Type::Bytes, Type::Bytes) => true,
        (Type::FileHandle, Type::FileHandle) => true,
        (Type::TcpSocket, Type::TcpSocket) => true,
        (Type::TcpListener, Type::TcpListener) => true,
        (Type::Pid, Type::Pid) => true,
        (Type::Set, Type::Set) => true,
        // No match
        _ => false,
    }
}

/// Find the base function name by looking through nested App nodes with DictValue/DictRef args.
/// For `App(App(f, dict1), dict2)`, returns the name from `f` if it's a Var.
fn find_base_function_name(expr: &TExpr) -> Option<String> {
    match &expr.node {
        TExprKind::Var(name) => Some(name.clone()),
        TExprKind::App { func, arg } => {
            // If the arg is a dictionary node, recurse into func
            match &arg.node {
                TExprKind::DictValue { .. } | TExprKind::DictRef { .. } => {
                    find_base_function_name(func)
                }
                _ => None, // Not a dict arg, so this isn't a dict application
            }
        }
        _ => None,
    }
}

/// Extract the type variable used in dictionary-based calls in a binding's body.
/// This finds the first DictMethodCall or DictRef and returns its type_var.
fn extract_body_type_var(expr: &TExpr) -> Option<u32> {
    match &expr.node {
        TExprKind::DictMethodCall { type_var, .. } => Some(*type_var),
        TExprKind::DictRef { type_var, .. } => Some(*type_var),
        TExprKind::DictValue { instance_ty, .. } => {
            // If instance_ty is a type variable, extract its ID
            match instance_ty {
                Type::Var(id) | Type::Generic(id) => Some(*id),
                _ => None,
            }
        }
        TExprKind::Var(_) | TExprKind::Lit(_) => None,
        TExprKind::Lambda { body, .. } => extract_body_type_var(body),
        TExprKind::App { func, arg } => {
            extract_body_type_var(func).or_else(|| extract_body_type_var(arg))
        }
        TExprKind::Let { value, body, .. } => {
            extract_body_type_var(value).or_else(|| body.as_ref().and_then(|b| extract_body_type_var(b)))
        }
        TExprKind::LetRec { bindings, body } => {
            bindings.iter().find_map(|b| extract_body_type_var(&b.body))
                .or_else(|| body.as_ref().and_then(|b| extract_body_type_var(b)))
        }
        TExprKind::If { cond, then_branch, else_branch } => {
            extract_body_type_var(cond)
                .or_else(|| extract_body_type_var(then_branch))
                .or_else(|| extract_body_type_var(else_branch))
        }
        TExprKind::Match { scrutinee, arms } => {
            extract_body_type_var(scrutinee).or_else(|| {
                arms.iter().find_map(|a| {
                    extract_body_type_var(&a.body)
                        .or_else(|| a.guard.as_ref().and_then(|g| extract_body_type_var(g)))
                })
            })
        }
        TExprKind::BinOp { left, right, .. } => {
            extract_body_type_var(left).or_else(|| extract_body_type_var(right))
        }
        TExprKind::UnaryOp { operand, .. } => extract_body_type_var(operand),
        TExprKind::Tuple(elems) => elems.iter().find_map(|e| extract_body_type_var(e)),
        TExprKind::List(elems) => elems.iter().find_map(|e| extract_body_type_var(e)),
        TExprKind::Record { fields, .. } => {
            fields.iter().find_map(|(_, e)| extract_body_type_var(e))
        }
        TExprKind::FieldAccess { record, .. } => extract_body_type_var(record),
        TExprKind::MethodCall { args, .. } => args.iter().find_map(|a| extract_body_type_var(a)),
        TExprKind::Constructor { args, .. } => args.iter().find_map(|a| extract_body_type_var(a)),
        TExprKind::Perform { args, .. } => args.iter().find_map(|a| extract_body_type_var(a)),
        TExprKind::Handle { body, handler, .. } => {
            extract_body_type_var(body)
                .or_else(|| handler.op_clauses.iter().find_map(|op| extract_body_type_var(&op.body)))
                .or_else(|| handler.return_clause.as_ref().and_then(|r| extract_body_type_var(&r.body)))
        }
        TExprKind::RecordUpdate { base, updates, .. } => {
            extract_body_type_var(base)
                .or_else(|| updates.iter().find_map(|(_, e)| extract_body_type_var(e)))
        }
        TExprKind::Seq { first, second } => {
            extract_body_type_var(first).or_else(|| extract_body_type_var(second))
        }
        TExprKind::Error(_) => None,
    }
}

/// Check if a binding needs specialization (is polymorphic and contains dictionary-based calls)
/// A binding needs specialization only if:
/// 1. It contains dictionary-based calls (DictMethodCall, DictRef)
/// 2. These calls use type variables from the binding's own parameters
///
/// Functions like `main` that call polymorphic functions don't need specialization themselves.
fn binding_needs_specialization(binding: &TBinding) -> bool {
    // Check if the body contains dictionary method calls or refs
    // but NOT if it just contains DictValue (which is passing dictionaries to other functions)
    expr_uses_dict_method(&binding.body)
}

/// Check if an expression uses dictionary methods (DictMethodCall or DictRef)
/// or passes a polymorphic type to another function (DictValue with type variable instance).
/// This indicates the function itself is polymorphic and needs specialization.
fn expr_uses_dict_method(expr: &TExpr) -> bool {
    match &expr.node {
        TExprKind::DictMethodCall { .. } => true,
        TExprKind::DictRef { .. } => true,
        // DictValue with a type variable instance means we're forwarding a polymorphic type
        TExprKind::DictValue { instance_ty, .. } => {
            matches!(instance_ty, Type::Var(_) | Type::Generic(_))
        }
        TExprKind::Var(_) | TExprKind::Lit(_) => false,
        TExprKind::Lambda { body, .. } => expr_uses_dict_method(body),
        TExprKind::App { func, arg } => {
            expr_uses_dict_method(func) || expr_uses_dict_method(arg)
        }
        TExprKind::Let { value, body, .. } => {
            expr_uses_dict_method(value)
                || body.as_ref().map_or(false, |b| expr_uses_dict_method(b))
        }
        TExprKind::LetRec { bindings, body } => {
            bindings.iter().any(|b| expr_uses_dict_method(&b.body))
                || body.as_ref().map_or(false, |b| expr_uses_dict_method(b))
        }
        TExprKind::If { cond, then_branch, else_branch } => {
            expr_uses_dict_method(cond)
                || expr_uses_dict_method(then_branch)
                || expr_uses_dict_method(else_branch)
        }
        TExprKind::Match { scrutinee, arms } => {
            expr_uses_dict_method(scrutinee)
                || arms.iter().any(|a| {
                    expr_uses_dict_method(&a.body)
                        || a.guard.as_ref().map_or(false, |g| expr_uses_dict_method(g))
                })
        }
        TExprKind::BinOp { left, right, .. } => {
            expr_uses_dict_method(left) || expr_uses_dict_method(right)
        }
        TExprKind::UnaryOp { operand, .. } => expr_uses_dict_method(operand),
        TExprKind::Tuple(elems) => elems.iter().any(|e| expr_uses_dict_method(e)),
        TExprKind::List(elems) => elems.iter().any(|e| expr_uses_dict_method(e)),
        TExprKind::Record { fields, .. } => {
            fields.iter().any(|(_, e)| expr_uses_dict_method(e))
        }
        TExprKind::FieldAccess { record, .. } => expr_uses_dict_method(record),
        TExprKind::MethodCall { args, .. } => {
            args.iter().any(|a| expr_uses_dict_method(a))
        }
        TExprKind::Constructor { args, .. } => args.iter().any(|a| expr_uses_dict_method(a)),
        TExprKind::Perform { args, .. } => args.iter().any(|a| expr_uses_dict_method(a)),
        TExprKind::Handle { body, handler, .. } => {
            expr_uses_dict_method(body)
                || handler.op_clauses.iter().any(|op| expr_uses_dict_method(&op.body))
                || handler.return_clause.as_ref().map_or(false, |r| expr_uses_dict_method(&r.body))
        }
        TExprKind::RecordUpdate { base, updates, .. } => {
            expr_uses_dict_method(base)
                || updates.iter().any(|(_, e)| expr_uses_dict_method(e))
        }
        TExprKind::Seq { first, second } => {
            expr_uses_dict_method(first) || expr_uses_dict_method(second)
        }
        TExprKind::Error(_) => false,
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
    // Skip bindings that need specialization - they'll be created on demand when called
    for binding in &tprogram.bindings {
        // Skip polymorphic bindings that need specialization
        let needs_spec = binding_needs_specialization(binding);
        if needs_spec {
            continue;
        }

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

        // Convert dict_params from TBinding
        let dict_params: Vec<_> = binding
            .dict_params
            .iter()
            .map(|(trait_name, type_var)| MonoDictParam {
                trait_name: trait_name.clone(),
                type_var: *type_var,
            })
            .collect();

        let mono_fn = MonoFn {
            id: fn_id.clone(),
            dict_params,
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
                dict_params: vec![], // Simple instances have no dict params
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
