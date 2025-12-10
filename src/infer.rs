//! Hindley-Milner type inference

use crate::ast::*;
use crate::types::*;
use std::collections::HashMap;
use std::rc::Rc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("unbound variable: {0}")]
    UnboundVariable(String),
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: Type, found: Type },
    #[error("occurs check failed: {0} occurs in {1}")]
    OccursCheck(TypeVarId, Type),
    #[error("unknown constructor: {0}")]
    UnknownConstructor(String),
    #[error("pattern type mismatch")]
    PatternMismatch,
    #[error("non-exhaustive patterns")]
    NonExhaustivePatterns,
}

pub struct Inferencer {
    /// Counter for generating fresh type variables
    next_var: TypeVarId,
    /// Current let-nesting level (for polymorphism)
    level: u32,
    /// Type context for constructors
    type_ctx: TypeContext,
}

impl Inferencer {
    pub fn new() -> Self {
        Self {
            next_var: 0,
            level: 0,
            type_ctx: TypeContext::new(),
        }
    }

    /// Generate a fresh type variable
    fn fresh_var(&mut self) -> Type {
        let id = self.next_var;
        self.next_var += 1;
        Type::new_var(id, self.level)
    }

    /// Unify two types
    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = t1.resolve();
        let t2 = t2.resolve();

        match (&t1, &t2) {
            // Same primitive types
            (Type::Int, Type::Int) => Ok(()),
            (Type::Float, Type::Float) => Ok(()),
            (Type::Bool, Type::Bool) => Ok(()),
            (Type::String, Type::String) => Ok(()),
            (Type::Char, Type::Char) => Ok(()),
            (Type::Unit, Type::Unit) => Ok(()),
            (Type::Pid, Type::Pid) => Ok(()),

            // Type variables
            (Type::Var(v1), Type::Var(v2)) if Rc::ptr_eq(v1, v2) => Ok(()),
            (Type::Var(var), other) | (other, Type::Var(var)) => {
                let var_inner = var.borrow().clone();
                match var_inner {
                    TypeVar::Link(_) => unreachable!("resolve should have followed links"),
                    TypeVar::Unbound { id, level } => {
                        // Occurs check
                        if other.occurs(id) {
                            return Err(TypeError::OccursCheck(id, other.clone()));
                        }
                        // Update levels for let-polymorphism
                        self.update_levels(&other, level);
                        // Link the variable
                        *var.borrow_mut() = TypeVar::Link(other.clone());
                        Ok(())
                    }
                    TypeVar::Generic(_) => Err(TypeError::TypeMismatch {
                        expected: t1.clone(),
                        found: t2.clone(),
                    }),
                }
            }

            // Function types
            (Type::Arrow(a1, b1), Type::Arrow(a2, b2)) => {
                self.unify(a1, a2)?;
                self.unify(b1, b2)?;
                Ok(())
            }

            // Tuples
            (Type::Tuple(ts1), Type::Tuple(ts2)) if ts1.len() == ts2.len() => {
                for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                    self.unify(t1, t2)?;
                }
                Ok(())
            }

            // Lists
            (Type::List(t1), Type::List(t2)) => self.unify(t1, t2),

            // Channels
            (Type::Channel(t1), Type::Channel(t2)) => self.unify(t1, t2),

            // Named constructors
            (
                Type::Constructor {
                    name: n1,
                    args: a1,
                },
                Type::Constructor {
                    name: n2,
                    args: a2,
                },
            ) if n1 == n2 && a1.len() == a2.len() => {
                for (t1, t2) in a1.iter().zip(a2.iter()) {
                    self.unify(t1, t2)?;
                }
                Ok(())
            }

            _ => Err(TypeError::TypeMismatch {
                expected: t1,
                found: t2,
            }),
        }
    }

    /// Update type variable levels for let-polymorphism
    fn update_levels(&self, ty: &Type, level: u32) {
        match ty.resolve() {
            Type::Var(var) => {
                let mut var = var.borrow_mut();
                if let TypeVar::Unbound {
                    level: var_level, ..
                } = &mut *var
                {
                    *var_level = (*var_level).min(level);
                }
            }
            Type::Arrow(a, b) => {
                self.update_levels(&a, level);
                self.update_levels(&b, level);
            }
            Type::Tuple(ts) => {
                for t in ts {
                    self.update_levels(&t, level);
                }
            }
            Type::List(t) => self.update_levels(&t, level),
            Type::Channel(t) => self.update_levels(&t, level),
            Type::Constructor { args, .. } => {
                for t in args {
                    self.update_levels(&t, level);
                }
            }
            _ => {}
        }
    }

    /// Instantiate a polymorphic type scheme
    fn instantiate(&mut self, scheme: &Scheme) -> Type {
        if scheme.num_generics == 0 {
            return scheme.ty.clone();
        }

        let mut substitution: HashMap<TypeVarId, Type> = HashMap::new();
        for i in 0..scheme.num_generics {
            substitution.insert(i, self.fresh_var());
        }

        self.substitute(&scheme.ty, &substitution)
    }

    /// Substitute generic type variables
    fn substitute(&self, ty: &Type, subst: &HashMap<TypeVarId, Type>) -> Type {
        let resolved = ty.resolve();
        match &resolved {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Generic(id) => subst.get(id).cloned().unwrap_or_else(|| resolved.clone()),
                _ => resolved.clone(),
            },
            Type::Arrow(a, b) => Type::Arrow(
                Rc::new(self.substitute(a, subst)),
                Rc::new(self.substitute(b, subst)),
            ),
            Type::Tuple(ts) => Type::Tuple(ts.iter().map(|t| self.substitute(t, subst)).collect()),
            Type::List(t) => Type::List(Rc::new(self.substitute(t, subst))),
            Type::Channel(t) => Type::Channel(Rc::new(self.substitute(t, subst))),
            Type::Constructor { name, args } => Type::Constructor {
                name: name.clone(),
                args: args.iter().map(|t| self.substitute(t, subst)).collect(),
            },
            _ => resolved.clone(),
        }
    }

    /// Generalize a type to a polymorphic scheme
    fn generalize(&self, ty: &Type) -> Scheme {
        let mut generics: HashMap<TypeVarId, TypeVarId> = HashMap::new();
        let generalized = self.generalize_inner(ty, &mut generics);
        Scheme {
            num_generics: generics.len() as u32,
            ty: generalized,
        }
    }

    fn generalize_inner(
        &self,
        ty: &Type,
        generics: &mut HashMap<TypeVarId, TypeVarId>,
    ) -> Type {
        let resolved = ty.resolve();
        match &resolved {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Unbound { id, level } if *level > self.level => {
                    let gen_id = if let Some(&gen_id) = generics.get(id) {
                        gen_id
                    } else {
                        let gen_id = generics.len() as TypeVarId;
                        generics.insert(*id, gen_id);
                        gen_id
                    };
                    Type::new_generic(gen_id)
                }
                _ => resolved.clone(),
            },
            Type::Arrow(a, b) => Type::Arrow(
                Rc::new(self.generalize_inner(a, generics)),
                Rc::new(self.generalize_inner(b, generics)),
            ),
            Type::Tuple(ts) => Type::Tuple(
                ts.iter()
                    .map(|t| self.generalize_inner(t, generics))
                    .collect(),
            ),
            Type::List(t) => Type::List(Rc::new(self.generalize_inner(t, generics))),
            Type::Channel(t) => Type::Channel(Rc::new(self.generalize_inner(t, generics))),
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

    /// Infer the type of an expression
    pub fn infer_expr(&mut self, env: &TypeEnv, expr: &Expr) -> Result<Type, TypeError> {
        match &expr.node {
            ExprKind::Lit(lit) => Ok(self.infer_literal(lit)),

            ExprKind::Var(name) => {
                if let Some(scheme) = env.get(name) {
                    Ok(self.instantiate(scheme))
                } else {
                    Err(TypeError::UnboundVariable(name.clone()))
                }
            }

            ExprKind::Lambda { params, body } => {
                let mut current_env = env.clone();
                let mut param_types = Vec::new();

                for param in params {
                    let param_ty = self.fresh_var();
                    self.bind_pattern(&mut current_env, param, &param_ty)?;
                    param_types.push(param_ty);
                }

                let body_ty = self.infer_expr(&current_env, body)?;
                Ok(Type::arrows(param_types, body_ty))
            }

            ExprKind::App { func, arg } => {
                let func_ty = self.infer_expr(env, func)?;
                let arg_ty = self.infer_expr(env, arg)?;
                let ret_ty = self.fresh_var();

                self.unify(&func_ty, &Type::arrow(arg_ty, ret_ty.clone()))?;
                Ok(ret_ty)
            }

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

                let value_ty = if is_recursive_fn {
                    // For recursive functions, add the name to env with a fresh type
                    // before inferring the lambda body
                    if let PatternKind::Var(name) = &pattern.node {
                        let mut recursive_env = env.clone();
                        let preliminary_ty = self.fresh_var();
                        recursive_env.insert(name.clone(), Scheme::mono(preliminary_ty.clone()));

                        let inferred_ty = self.infer_expr(&recursive_env, value)?;
                        self.unify(&preliminary_ty, &inferred_ty)?;
                        inferred_ty
                    } else {
                        unreachable!()
                    }
                } else {
                    self.infer_expr(env, value)?
                };

                self.level -= 1;

                // Value restriction: only generalize syntactic values.
                // This prevents unsound polymorphism for effectful expressions
                // like Channel.new, where the type variable must be shared
                // across all uses rather than instantiated fresh each time.
                let scheme = if Self::is_syntactic_value(value) {
                    self.generalize(&value_ty)
                } else {
                    // Don't generalize - keep the monomorphic type
                    Scheme {
                        num_generics: 0,
                        ty: value_ty,
                    }
                };

                // Bind the pattern
                let mut new_env = env.clone();
                self.bind_pattern_scheme(&mut new_env, pattern, scheme)?;

                if let Some(body) = body {
                    self.infer_expr(&new_env, body)
                } else {
                    Ok(Type::Unit)
                }
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_ty = self.infer_expr(env, cond)?;
                self.unify(&cond_ty, &Type::Bool)?;

                let then_ty = self.infer_expr(env, then_branch)?;
                let else_ty = self.infer_expr(env, else_branch)?;
                self.unify(&then_ty, &else_ty)?;

                Ok(then_ty)
            }

            ExprKind::Match { scrutinee, arms } => {
                let scrutinee_ty = self.infer_expr(env, scrutinee)?;
                let result_ty = self.fresh_var();

                for arm in arms {
                    let mut arm_env = env.clone();
                    self.bind_pattern(&mut arm_env, &arm.pattern, &scrutinee_ty)?;

                    if let Some(guard) = &arm.guard {
                        let guard_ty = self.infer_expr(&arm_env, guard)?;
                        self.unify(&guard_ty, &Type::Bool)?;
                    }

                    let body_ty = self.infer_expr(&arm_env, &arm.body)?;
                    self.unify(&result_ty, &body_ty)?;
                }

                Ok(result_ty)
            }

            ExprKind::Tuple(exprs) => {
                let types: Result<Vec<_>, _> = exprs
                    .iter()
                    .map(|e| self.infer_expr(env, e))
                    .collect();
                Ok(Type::Tuple(types?))
            }

            ExprKind::List(exprs) => {
                let elem_ty = self.fresh_var();
                for expr in exprs {
                    let ty = self.infer_expr(env, expr)?;
                    self.unify(&elem_ty, &ty)?;
                }
                Ok(Type::List(Rc::new(elem_ty)))
            }

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
                        let mut ty = result_ty;
                        for field_ty in info.field_types.iter().rev() {
                            let param_ty = self.substitute(field_ty, &subst);
                            ty = Type::Arrow(Rc::new(param_ty), Rc::new(ty));
                        }
                        return Ok(ty);
                    }

                    // Check field types against provided arguments
                    if args.len() != info.field_types.len() {
                        return Err(TypeError::TypeMismatch {
                            expected: result_ty,
                            found: Type::Unit,
                        });
                    }

                    for (arg, field_ty) in args.iter().zip(&info.field_types) {
                        let arg_ty = self.infer_expr(env, arg)?;
                        let expected_ty = self.substitute(field_ty, &subst);
                        self.unify(&arg_ty, &expected_ty)?;
                    }

                    Ok(result_ty)
                } else {
                    // Unknown constructor - for now just create a generic type
                    Err(TypeError::UnknownConstructor(name.clone()))
                }
            }

            ExprKind::BinOp { op, left, right } => {
                let left_ty = self.infer_expr(env, left)?;
                let right_ty = self.infer_expr(env, right)?;

                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.unify(&left_ty, &Type::Int)?;
                        self.unify(&right_ty, &Type::Int)?;
                        Ok(Type::Int)
                    }
                    BinOp::Eq | BinOp::Neq => {
                        self.unify(&left_ty, &right_ty)?;
                        Ok(Type::Bool)
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        self.unify(&left_ty, &Type::Int)?;
                        self.unify(&right_ty, &Type::Int)?;
                        Ok(Type::Bool)
                    }
                    BinOp::And | BinOp::Or => {
                        self.unify(&left_ty, &Type::Bool)?;
                        self.unify(&right_ty, &Type::Bool)?;
                        Ok(Type::Bool)
                    }
                    BinOp::Cons => {
                        let elem_ty = left_ty;
                        self.unify(&right_ty, &Type::List(Rc::new(elem_ty.clone())))?;
                        Ok(Type::List(Rc::new(elem_ty)))
                    }
                    BinOp::Concat => {
                        self.unify(&left_ty, &right_ty)?;
                        // Works for lists and strings
                        Ok(left_ty)
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
                                self.unify(&left_ty, &Type::arrow(a.clone(), b.clone()))?;
                                self.unify(&right_ty, &Type::arrow(b, c.clone()))?;
                            }
                            BinOp::ComposeBack => {
                                self.unify(&left_ty, &Type::arrow(b.clone(), c.clone()))?;
                                self.unify(&right_ty, &Type::arrow(a.clone(), b))?;
                            }
                            _ => unreachable!(),
                        }
                        Ok(Type::arrow(a, c))
                    }
                }
            }

            ExprKind::UnaryOp { op, operand } => {
                let operand_ty = self.infer_expr(env, operand)?;
                match op {
                    UnaryOp::Neg => {
                        self.unify(&operand_ty, &Type::Int)?;
                        Ok(Type::Int)
                    }
                    UnaryOp::Not => {
                        self.unify(&operand_ty, &Type::Bool)?;
                        Ok(Type::Bool)
                    }
                }
            }

            ExprKind::Seq { first, second } => {
                self.infer_expr(env, first)?;
                self.infer_expr(env, second)
            }

            // Concurrency primitives
            ExprKind::Spawn(body) => {
                let body_ty = self.infer_expr(env, body)?;
                // Body should be a thunk: () -> a
                let ret_ty = self.fresh_var();
                self.unify(&body_ty, &Type::arrow(Type::Unit, ret_ty))?;
                Ok(Type::Pid)
            }

            ExprKind::NewChannel => {
                let elem_ty = self.fresh_var();
                Ok(Type::Channel(Rc::new(elem_ty)))
            }

            ExprKind::ChanSend { channel, value } => {
                let chan_ty = self.infer_expr(env, channel)?;
                let val_ty = self.infer_expr(env, value)?;
                self.unify(&chan_ty, &Type::Channel(Rc::new(val_ty)))?;
                Ok(Type::Unit)
            }

            ExprKind::ChanRecv(channel) => {
                let chan_ty = self.infer_expr(env, channel)?;
                let elem_ty = self.fresh_var();
                self.unify(&chan_ty, &Type::Channel(Rc::new(elem_ty.clone())))?;
                Ok(elem_ty)
            }

            ExprKind::Select { arms } => {
                // All arms must have compatible channel element types
                // All arm bodies must have the same result type
                let result_ty = self.fresh_var();

                for arm in arms {
                    // Infer channel type
                    let chan_ty = self.infer_expr(env, &arm.channel)?;
                    let elem_ty = self.fresh_var();
                    self.unify(&chan_ty, &Type::Channel(Rc::new(elem_ty.clone())))?;

                    // Bind pattern in arm's environment
                    let mut arm_env = env.clone();
                    self.bind_pattern(&mut arm_env, &arm.pattern, &elem_ty)?;

                    // Infer body type and unify with result
                    let body_ty = self.infer_expr(&arm_env, &arm.body)?;
                    self.unify(&result_ty, &body_ty)?;
                }

                Ok(result_ty)
            }

            // Delimited continuations
            ExprKind::Reset(body) => {
                // reset e : a  where e : a
                self.infer_expr(env, body)
            }

            ExprKind::Shift { param, body } => {
                // Simplified typing: k : a -> a, body : a, result : a
                // (Full answer-type polymorphism is complex, defer to future)
                let captured_ty = self.fresh_var();
                let cont_ty = Type::arrow(captured_ty.clone(), captured_ty.clone());

                let mut body_env = env.clone();
                self.bind_pattern(&mut body_env, param, &cont_ty)?;

                let body_ty = self.infer_expr(&body_env, body)?;
                self.unify(&body_ty, &captured_ty)?;

                Ok(captured_ty)
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
                self.unify(ty, &lit_ty)?;
                Ok(())
            }

            PatternKind::Tuple(pats) => {
                let pat_types: Vec<_> = pats.iter().map(|_| self.fresh_var()).collect();
                self.unify(ty, &Type::Tuple(pat_types.clone()))?;
                for (pat, pat_ty) in pats.iter().zip(&pat_types) {
                    self.bind_pattern(env, pat, pat_ty)?;
                }
                Ok(())
            }

            PatternKind::List(pats) => {
                let elem_ty = self.fresh_var();
                self.unify(ty, &Type::List(Rc::new(elem_ty.clone())))?;
                for pat in pats {
                    self.bind_pattern(env, pat, &elem_ty)?;
                }
                Ok(())
            }

            PatternKind::Cons { head, tail } => {
                let elem_ty = self.fresh_var();
                let list_ty = Type::List(Rc::new(elem_ty.clone()));
                self.unify(ty, &list_ty)?;
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
                    self.unify(ty, &constructor_ty)?;

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
                    Err(TypeError::UnknownConstructor(name.clone()))
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
                    Err(TypeError::UnboundVariable(name.clone()))
                }
            }
            TypeExprKind::Named(name) => match name.as_str() {
                "Int" => Ok(Type::Int),
                "Float" => Ok(Type::Float),
                "Bool" => Ok(Type::Bool),
                "String" => Ok(Type::String),
                "Char" => Ok(Type::Char),
                "Pid" => Ok(Type::Pid),
                _ => Ok(Type::Constructor {
                    name: name.clone(),
                    args: vec![],
                }),
            },
            TypeExprKind::App { constructor, args } => {
                let arg_types: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.type_expr_to_type(a, param_map))
                    .collect();
                match &constructor.node {
                    TypeExprKind::Named(name) => Ok(Type::Constructor {
                        name: name.clone(),
                        args: arg_types?,
                    }),
                    _ => Err(TypeError::PatternMismatch),
                }
            }
            TypeExprKind::Arrow { from, to } => {
                let from_ty = self.type_expr_to_type(from, param_map)?;
                let to_ty = self.type_expr_to_type(to, param_map)?;
                Ok(Type::arrow(from_ty, to_ty))
            }
            TypeExprKind::Tuple(types) => {
                let tys: Result<Vec<_>, _> = types
                    .iter()
                    .map(|t| self.type_expr_to_type(t, param_map))
                    .collect();
                Ok(Type::Tuple(tys?))
            }
            TypeExprKind::Channel(inner) => {
                let inner_ty = self.type_expr_to_type(inner, param_map)?;
                Ok(Type::Channel(Rc::new(inner_ty)))
            }
        }
    }

    /// Register a type declaration
    pub fn register_type_decl(&mut self, decl: &Decl) {
        if let Decl::Type {
            name,
            params,
            constructors,
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

    /// Infer types for an entire program
    pub fn infer_program(&mut self, program: &Program) -> Result<TypeEnv, TypeError> {
        let mut env = TypeEnv::new();

        // Add built-in functions
        // print : a -> ()
        let print_ty = {
            let a = self.fresh_var();
            Type::arrow(a, Type::Unit)
        };
        env.insert("print".into(), self.generalize(&print_ty));

        // int_to_string : Int -> String
        env.insert("int_to_string".into(), Scheme::mono(Type::arrow(Type::Int, Type::String)));

        // string_length : String -> Int
        env.insert("string_length".into(), Scheme::mono(Type::arrow(Type::String, Type::Int)));

        // First pass: register all type declarations
        for item in &program.items {
            if let Item::Decl(decl) = item {
                self.register_type_decl(decl);
            }
        }

        // Second pass: infer types for let declarations and top-level expressions
        for item in &program.items {
            match item {
                Item::Decl(Decl::Let {
                    name,
                    params,
                    body,
                    ..
                }) => {
                    self.level += 1;

                    let mut local_env = env.clone();
                    let mut param_types = Vec::new();

                    for param in params {
                        let param_ty = self.fresh_var();
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

                    let body_ty = self.infer_expr(&local_env, body)?;

                    // Unify the body type with what we predicted
                    if param_types.is_empty() {
                        self.unify(&ret_ty, &body_ty)?;
                    } else {
                        self.unify(&ret_ty, &body_ty)?;
                    }

                    let func_ty = if param_types.is_empty() {
                        body_ty
                    } else {
                        Type::arrows(param_types, body_ty)
                    };

                    self.level -= 1;

                    let scheme = self.generalize(&func_ty);
                    env.insert(name.clone(), scheme);
                }
                Item::Decl(Decl::Type { .. } | Decl::TypeAlias { .. }) => {
                    // Already handled in first pass
                }
                Item::Expr(expr) => {
                    // Type-check top-level expressions
                    self.infer_expr(&env, expr)?;
                }
            }
        }

        Ok(env)
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
        assert!(matches!(ty, Type::Arrow(_, _)));
    }

    #[test]
    fn test_application() {
        let ty = infer("(fun x -> x + 1) 42").unwrap();
        assert!(matches!(ty.resolve(), Type::Int));
    }

    #[test]
    fn test_if() {
        let ty = infer("if true then 1 else 2").unwrap();
        assert!(matches!(ty, Type::Int));
    }

    fn typecheck_program(input: &str) -> Result<TypeEnv, TypeError> {
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

    // Delimited continuation type tests
    #[test]
    fn test_reset_type() {
        let ty = infer("reset 42").unwrap();
        assert!(matches!(ty, Type::Int));
    }

    #[test]
    fn test_shift_type() {
        let ty = infer("reset (1 + shift (fun k -> k 10))").unwrap();
        assert!(matches!(ty.resolve(), Type::Int));
    }

    #[test]
    fn test_shift_type_mismatch() {
        // Body returns String, but context expects Int
        let result = infer("reset (1 + shift (fun k -> \"hello\"))");
        // With simplified typing, this should be a type error
        assert!(result.is_err());
    }

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
}
