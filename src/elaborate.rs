//! Elaboration Pass: AST + Types → TAST
//!
//! This pass converts the untyped AST into a fully-typed TAST (Typed AST).
//! Every expression node in the output carries its resolved, concrete type.
//!
//! Key responsibilities:
//! - Attach resolved types to every expression
//! - Resolve trait method calls to concrete instances (monomorphization)
//! - Convert operators to their type-specific variants

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::{
    BinOp, Decl, Expr, ExprKind, HandlerArm, HandlerReturn, Item, MatchArm, Pattern, PatternKind,
    Program, RecBinding, TypeExpr, TypeExprKind, UnaryOp,
};
use crate::infer::Inferencer;
use crate::tast::{
    TBinOp, TBinding, TConstructor, TEffectDecl, TExpr, TExprKind, THandler, THandlerClause,
    TInstanceDecl, TMatchArm, TMethodImpl, TMethodSig, TOpClause, TOperation, TPattern,
    TPatternKind, TProgram, TRecBinding, TTraitDecl, TTypeDecl, TUnaryOp,
};
use crate::types::{Type, TypeEnv};

/// Context for elaboration
pub struct ElaborateCtx<'a> {
    /// The type inferencer with resolved types
    infer: &'a Inferencer,
    /// Type environment for looking up variable types (reserved for future use)
    #[allow(dead_code)]
    type_env: &'a TypeEnv,
    /// Known trait instances for monomorphization
    /// Maps (trait_name, concrete_type_string) -> mangled_name
    instances: HashMap<(String, String), String>,
    /// Maps method_name -> trait_name for method lookup
    method_to_trait: HashMap<String, String>,
}

impl<'a> ElaborateCtx<'a> {
    pub fn new(infer: &'a Inferencer, type_env: &'a TypeEnv) -> Self {
        let mut method_to_trait = HashMap::new();

        // Build method->trait lookup from ClassEnv
        for (trait_name, trait_info) in &infer.class_env_ref().traits {
            for method_name in trait_info.methods.keys() {
                method_to_trait.insert(method_name.clone(), trait_name.clone());
            }
        }

        ElaborateCtx {
            infer,
            type_env,
            instances: HashMap::new(),
            method_to_trait,
        }
    }

    /// Get the type of an expression from the inferencer
    fn get_expr_type(&self, expr: &Expr) -> Type {
        self.infer
            .get_expr_type(&expr.span)
            .map(|ty| self.infer.resolve_type(&ty))
            .unwrap_or(Type::Unit)
    }

    /// Resolve all type variables in a type
    fn resolve_type(&self, ty: &Type) -> Type {
        self.infer.resolve_type(ty)
    }

    /// Generate a mangled name for a trait instance
    fn mangle_instance(&self, trait_name: &str, ty: &Type) -> String {
        let type_suffix = self.mangle_type(ty);
        format!("{}_{}", trait_name, type_suffix)
    }

    /// Mangle a type for use in function names
    fn mangle_type(&self, ty: &Type) -> String {
        match ty {
            Type::Int => "Int".to_string(),
            Type::Float => "Float".to_string(),
            Type::String => "String".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::Unit => "Unit".to_string(),
            Type::Tuple(elems) if elems.is_empty() => "Unit".to_string(), // () is Unit
            Type::Tuple(elems) => {
                let parts: Vec<_> = elems.iter().map(|t| self.mangle_type(t)).collect();
                format!("Tuple_{}", parts.join("_"))
            }
            Type::Constructor { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let parts: Vec<_> = args.iter().map(|t| self.mangle_type(t)).collect();
                    format!("{}_{}", name, parts.join("_"))
                }
            }
            Type::Var(id) => {
                // Try to resolve the variable
                let resolved = self.resolve_type(ty);
                if let Type::Var(_) = resolved {
                    format!("Var{}", id)
                } else {
                    self.mangle_type(&resolved)
                }
            }
            Type::Arrow { .. } => "Fn".to_string(),
            Type::Channel(_) => "Channel".to_string(),
            Type::Dict(_) => "Dict".to_string(),
            Type::Set => "Set".to_string(),
            Type::Pid => "Pid".to_string(),
            Type::Fiber(_) => "Fiber".to_string(),
            Type::Char => "Char".to_string(),
            Type::Bytes => "Bytes".to_string(),
            Type::FileHandle => "FileHandle".to_string(),
            Type::TcpSocket => "TcpSocket".to_string(),
            Type::TcpListener => "TcpListener".to_string(),
            Type::Generic(id) => format!("Gen{}", id),
        }
    }

    /// Check if a variable is a trait method and return the trait name
    fn lookup_trait_method(&self, name: &str) -> Option<&str> {
        self.method_to_trait.get(name).map(|s| s.as_str())
    }

    /// Convert a TypeExpr to a Type (simplified conversion)
    #[allow(clippy::only_used_in_recursion)]
    fn type_expr_to_type(&self, te: &TypeExpr) -> Type {
        match &te.node {
            TypeExprKind::Named(name) => {
                match name.as_str() {
                    "Int" => Type::Int,
                    "Float" => Type::Float,
                    "String" => Type::String,
                    "Bool" => Type::Bool,
                    "Char" => Type::Char,
                    "()" | "Unit" => Type::Unit,
                    _ => Type::Constructor {
                        name: name.clone(),
                        args: Vec::new(),
                    },
                }
            }
            TypeExprKind::Var(name) => {
                // Type variable - use Generic for now
                Type::Generic(name.chars().next().map(|c| c as u32 - 'a' as u32).unwrap_or(0))
            }
            TypeExprKind::App { constructor, args } => {
                // Extract the constructor name and build a type with args
                let base = self.type_expr_to_type(constructor);
                let resolved_args: Vec<_> = args.iter().map(|a| self.type_expr_to_type(a)).collect();
                match base {
                    Type::Constructor { name, args: _ } => Type::Constructor {
                        name,
                        args: resolved_args,
                    },
                    _ => base, // Just return the base if it's a primitive
                }
            }
            TypeExprKind::Arrow { from, to, effects: _ } => Type::Arrow {
                arg: Rc::new(self.type_expr_to_type(from)),
                ret: Rc::new(self.type_expr_to_type(to)),
                effects: crate::types::Row::Empty,
            },
            TypeExprKind::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.type_expr_to_type(e)).collect())
            }
            TypeExprKind::List(elem) => Type::Constructor {
                name: "List".to_string(),
                args: vec![self.type_expr_to_type(elem)],
            },
            TypeExprKind::Channel(elem) => Type::Channel(Rc::new(self.type_expr_to_type(elem))),
        }
    }
}

/// Elaborate a program from AST to TAST
pub fn elaborate(
    program: &Program,
    infer: &Inferencer,
    type_env: &TypeEnv,
) -> Result<TProgram, Vec<String>> {
    let mut ctx = ElaborateCtx::new(infer, type_env);
    let mut errors = Vec::new();

    let mut type_decls = Vec::new();
    let mut effect_decls = Vec::new();
    let mut trait_decls = Vec::new();
    let mut instance_decls = Vec::new();
    let mut bindings = Vec::new();
    let mut main_expr = None;

    // Process all items
    for item in &program.items {
        match item {
            Item::Decl(decl) => match decl {
                Decl::Type {
                    name,
                    params,
                    constructors,
                    ..
                } => {
                    type_decls.push(TTypeDecl {
                        name: name.clone(),
                        params: params.clone(),
                        constructors: constructors
                            .iter()
                            .map(|c| TConstructor {
                                name: c.name.clone(),
                                fields: c.fields.iter().map(|te| ctx.type_expr_to_type(te)).collect(),
                            })
                            .collect(),
                    });
                }

                Decl::EffectDecl {
                    name, operations, ..
                } => {
                    effect_decls.push(TEffectDecl {
                        name: name.clone(),
                        operations: operations
                            .iter()
                            .map(|op| {
                                // Parse the type_sig to extract param types and return type
                                let (param_tys, return_ty) =
                                    parse_effect_sig(&ctx, &op.type_sig);
                                TOperation {
                                    name: op.name.clone(),
                                    param_tys,
                                    return_ty,
                                }
                            })
                            .collect(),
                    });
                }

                Decl::Trait {
                    name,
                    type_param,
                    methods,
                    ..
                } => {
                    trait_decls.push(TTraitDecl {
                        name: name.clone(),
                        param: type_param.clone(),
                        methods: methods
                            .iter()
                            .map(|m| TMethodSig {
                                name: m.name.clone(),
                                ty: ctx.type_expr_to_type(&m.type_sig),
                            })
                            .collect(),
                    });
                }

                Decl::Instance {
                    trait_name,
                    target_type,
                    methods,
                    ..
                } => {
                    let instance_ty = ctx.type_expr_to_type(target_type);
                    let resolved_ty = ctx.resolve_type(&instance_ty);
                    let mangled_name = ctx.mangle_instance(trait_name, &resolved_ty);

                    // Register this instance
                    ctx.instances.insert(
                        (trait_name.clone(), format!("{:?}", resolved_ty)),
                        mangled_name.clone(),
                    );

                    let tmethods: Vec<_> = methods
                        .iter()
                        .filter_map(|m| {
                            let method_mangled = format!("{}_{}", mangled_name, m.name);
                            match elaborate_expr(&ctx, &m.body) {
                                Ok(body) => {
                                    let params: Vec<_> = m
                                        .params
                                        .iter()
                                        .map(|p| elaborate_pattern(&ctx, p))
                                        .collect();
                                    Some(TMethodImpl {
                                        name: m.name.clone(),
                                        mangled_name: method_mangled,
                                        params,
                                        body,
                                    })
                                }
                                Err(e) => {
                                    errors.push(e);
                                    None
                                }
                            }
                        })
                        .collect();

                    instance_decls.push(TInstanceDecl {
                        trait_name: trait_name.clone(),
                        instance_ty: resolved_ty,
                        mangled_name,
                        methods: tmethods,
                    });
                }

                Decl::Let {
                    name, params, body, ..
                } => {
                    match elaborate_expr(&ctx, body) {
                        Ok(tbody) => {
                            let (ty, dict_params) = if let Some(scheme) = type_env.get(name) {
                                let dicts: Vec<_> = scheme.predicates.iter()
                                    .filter_map(|pred| {
                                        // Extract type var ID from predicate type
                                        if let Type::Generic(id) = &pred.ty {
                                            Some((pred.trait_name.clone(), *id))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                (ctx.resolve_type(&scheme.ty), dicts)
                            } else {
                                (ctx.get_expr_type(body), vec![])
                            };
                            let tparams: Vec<_> =
                                params.iter().map(|p| elaborate_pattern(&ctx, p)).collect();
                            bindings.push(TBinding {
                                name: name.clone(),
                                params: tparams,
                                body: tbody,
                                ty,
                                dict_params,
                            });
                        }
                        Err(e) => errors.push(e),
                    }
                }

                Decl::LetRec { bindings: recs, .. } => {
                    for rec in recs {
                        match elaborate_expr(&ctx, &rec.body) {
                            Ok(tbody) => {
                                let (ty, dict_params) = if let Some(scheme) = type_env.get(&rec.name.node) {
                                    let dicts: Vec<_> = scheme.predicates.iter()
                                        .filter_map(|pred| {
                                            if let Type::Generic(id) = &pred.ty {
                                                Some((pred.trait_name.clone(), *id))
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    (ctx.resolve_type(&scheme.ty), dicts)
                                } else {
                                    (ctx.get_expr_type(&rec.body), vec![])
                                };
                                let tparams: Vec<_> =
                                    rec.params.iter().map(|p| elaborate_pattern(&ctx, p)).collect();
                                bindings.push(TBinding {
                                    name: rec.name.node.clone(),
                                    params: tparams,
                                    body: tbody,
                                    ty,
                                    dict_params,
                                });
                            }
                            Err(e) => errors.push(e),
                        }
                    }
                }

                Decl::OperatorDef {
                    op, params, body, ..
                } => {
                    match elaborate_expr(&ctx, body) {
                        Ok(tbody) => {
                            let ty = ctx.get_expr_type(body);
                            let tparams: Vec<_> =
                                params.iter().map(|p| elaborate_pattern(&ctx, p)).collect();
                            bindings.push(TBinding {
                                name: op.clone(),
                                params: tparams,
                                body: tbody,
                                ty,
                                dict_params: vec![], // Operators don't have class constraints
                            });
                        }
                        Err(e) => errors.push(e),
                    }
                }

                // Skip other declarations for now
                _ => {}
            },

            Item::Expr(expr) => {
                match elaborate_expr(&ctx, expr) {
                    Ok(texpr) => {
                        main_expr = Some(texpr);
                    }
                    Err(e) => errors.push(e),
                }
            }

            Item::Import(_) => {
                // Imports don't produce TAST nodes
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(TProgram {
        type_decls,
        effect_decls,
        trait_decls,
        instance_decls,
        bindings,
        main: main_expr,
    })
}

/// Parse an effect operation type signature into param types and return type
fn parse_effect_sig(ctx: &ElaborateCtx, sig: &TypeExpr) -> (Vec<Type>, Type) {
    match &sig.node {
        TypeExprKind::Arrow { from, to, effects: _ } => {
            let param = ctx.type_expr_to_type(from);
            let (mut rest_params, ret) = parse_effect_sig(ctx, to);
            let mut params = vec![param];
            params.append(&mut rest_params);
            (params, ret)
        }
        _ => (Vec::new(), ctx.type_expr_to_type(sig)),
    }
}

/// Elaborate an expression from AST to TAST
fn elaborate_expr(ctx: &ElaborateCtx, expr: &Expr) -> Result<TExpr, String> {
    let span = expr.span.clone();
    let ty = ctx.get_expr_type(expr);

    let node = match &expr.node {
        ExprKind::Var(name) => TExprKind::Var(name.clone()),

        ExprKind::Lit(lit) => TExprKind::Lit(lit.clone()),

        ExprKind::Lambda { params, body } => {
            let tparams: Vec<_> = params.iter().map(|p| elaborate_pattern(ctx, p)).collect();
            let tbody = elaborate_expr(ctx, body)?;
            TExprKind::Lambda {
                params: tparams,
                body: Rc::new(tbody),
            }
        }

        ExprKind::App { func, arg } => {
            // Check if this is a trait method call
            if let ExprKind::Var(name) = &func.node {
                if let Some(trait_name) = ctx.lookup_trait_method(name) {
                    // This is a trait method call
                    let targ = elaborate_expr(ctx, arg)?;
                    let arg_ty = targ.ty.clone();

                    // Check if the argument type is polymorphic (Generic or unbound Var)
                    // If so, we need dictionary-based dispatch at runtime
                    let type_var_id = match &arg_ty {
                        Type::Generic(id) => Some(*id),
                        Type::Var(id) if !ctx.infer.is_type_var_bound(*id) => Some(*id),
                        _ => None,
                    };
                    if let Some(type_var_id) = type_var_id {
                        return Ok(TExpr::new(
                            TExprKind::DictMethodCall {
                                trait_name: trait_name.to_string(),
                                method: name.clone(),
                                type_var: type_var_id,
                                args: vec![targ],
                            },
                            ty,
                            span,
                        ));
                    }

                    // Concrete type - use monomorphized method call
                    return Ok(TExpr::new(
                        TExprKind::MethodCall {
                            trait_name: trait_name.to_string(),
                            method: name.clone(),
                            instance_ty: arg_ty,
                            args: vec![targ],
                        },
                        ty,
                        span,
                    ));
                }
            }

            // Check if calling a function with class constraints
            // If so, we need to pass dictionary arguments
            if let ExprKind::Var(name) = &func.node {
                if let Some(scheme) = ctx.type_env.get(name) {
                    if !scheme.predicates.is_empty() {
                        // Function has class constraints - need to pass dictionaries
                        let targ = elaborate_expr(ctx, arg)?;
                        let arg_ty = targ.ty.clone();

                        // Build dictionary expressions for each predicate
                        let mut dict_apps: Vec<TExpr> = Vec::new();
                        for pred in &scheme.predicates {
                            // The predicate's type is Generic(id).
                            // We need to figure out what concrete type it maps to.
                            // For simple cases like `print x` where x : Int,
                            // the argument type IS the instantiation.
                            let dict_expr = match &pred.ty {
                                Type::Generic(_id) => {
                                    // The concrete type is the argument type
                                    // (for single-parameter functions with single constraint)
                                    match &arg_ty {
                                        Type::Generic(type_var) => {
                                            // Still polymorphic - need to pass through dict from scope
                                            TExpr::new(
                                                TExprKind::DictRef {
                                                    trait_name: pred.trait_name.clone(),
                                                    type_var: *type_var,
                                                },
                                                Type::Unit, // Dict type placeholder
                                                span.clone(),
                                            )
                                        }
                                        _ => {
                                            // Concrete type - use static dictionary
                                            // Get the type variable ID from the Generic in the predicate
                                            let type_var_id = if let Type::Generic(id) = &pred.ty {
                                                *id
                                            } else {
                                                0 // Fallback - shouldn't happen
                                            };
                                            TExpr::new(
                                                TExprKind::DictValue {
                                                    trait_name: pred.trait_name.clone(),
                                                    instance_ty: arg_ty.clone(),
                                                    type_var: type_var_id,
                                                },
                                                Type::Unit, // Dict type placeholder
                                                span.clone(),
                                            )
                                        }
                                    }
                                }
                                _ => {
                                    // Non-generic predicate type - shouldn't happen in well-formed schemes
                                    continue;
                                }
                            };
                            dict_apps.push(dict_expr);
                        }

                        // Build: func dict1 dict2 ... arg
                        let tfunc = elaborate_expr(ctx, func)?;
                        let mut result = tfunc;

                        // Apply dictionary arguments first
                        for dict_expr in dict_apps {
                            let app_ty = ctx.get_expr_type(expr); // Approximate
                            result = TExpr::new(
                                TExprKind::App {
                                    func: Rc::new(result),
                                    arg: Rc::new(dict_expr),
                                },
                                app_ty.clone(),
                                span.clone(),
                            );
                        }

                        // Then apply the actual argument
                        return Ok(TExpr::new(
                            TExprKind::App {
                                func: Rc::new(result),
                                arg: Rc::new(targ),
                            },
                            ty,
                            span,
                        ));
                    }
                }
            }

            // Check if func is a constructor - if so, merge arg into constructor args
            if let ExprKind::Constructor { name, args } = &func.node {
                let targ = elaborate_expr(ctx, arg)?;
                let mut targs: Vec<_> = args
                    .iter()
                    .map(|e| elaborate_expr(ctx, e))
                    .collect::<Result<_, _>>()?;
                targs.push(targ);
                return Ok(TExpr::new(
                    TExprKind::Constructor {
                        name: name.clone(),
                        args: targs,
                    },
                    ty,
                    span,
                ));
            }

            // Check if func is an App of a constructor - recursively flatten
            let tfunc = elaborate_expr(ctx, func)?;
            let targ = elaborate_expr(ctx, arg)?;

            // If tfunc is a Constructor, merge targ into its args
            if let TExprKind::Constructor { name, args } = &tfunc.node {
                let mut new_args = args.clone();
                new_args.push(targ);
                return Ok(TExpr::new(
                    TExprKind::Constructor {
                        name: name.clone(),
                        args: new_args,
                    },
                    ty,
                    span,
                ));
            }

            TExprKind::App {
                func: Rc::new(tfunc),
                arg: Rc::new(targ),
            }
        }

        ExprKind::Let {
            pattern,
            value,
            body,
        } => {
            let tpat = elaborate_pattern(ctx, pattern);
            let tval = elaborate_expr(ctx, value)?;
            let tbody = body
                .as_ref()
                .map(|b| elaborate_expr(ctx, b))
                .transpose()?
                .map(Rc::new);
            TExprKind::Let {
                pattern: tpat,
                value: Rc::new(tval),
                body: tbody,
            }
        }

        ExprKind::LetRec { bindings, body } => {
            let tbindings: Vec<_> = bindings
                .iter()
                .map(|b| elaborate_rec_binding(ctx, b))
                .collect::<Result<_, _>>()?;
            let tbody = body
                .as_ref()
                .map(|b| elaborate_expr(ctx, b))
                .transpose()?
                .map(Rc::new);
            TExprKind::LetRec {
                bindings: tbindings,
                body: tbody,
            }
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let tcond = elaborate_expr(ctx, cond)?;
            let tthen = elaborate_expr(ctx, then_branch)?;
            let telse = elaborate_expr(ctx, else_branch)?;
            TExprKind::If {
                cond: Rc::new(tcond),
                then_branch: Rc::new(tthen),
                else_branch: Rc::new(telse),
            }
        }

        ExprKind::Match { scrutinee, arms } => {
            let tscrutinee = elaborate_expr(ctx, scrutinee)?;
            let tarms: Vec<_> = arms
                .iter()
                .map(|a| elaborate_match_arm(ctx, a))
                .collect::<Result<_, _>>()?;
            TExprKind::Match {
                scrutinee: Rc::new(tscrutinee),
                arms: tarms,
            }
        }

        ExprKind::Tuple(elems) => {
            let telems: Vec<_> = elems
                .iter()
                .map(|e| elaborate_expr(ctx, e))
                .collect::<Result<_, _>>()?;
            TExprKind::Tuple(telems)
        }

        ExprKind::List(elems) => {
            let telems: Vec<_> = elems
                .iter()
                .map(|e| elaborate_expr(ctx, e))
                .collect::<Result<_, _>>()?;
            TExprKind::List(telems)
        }

        ExprKind::Record { name, fields } => {
            let tfields: Vec<_> = fields
                .iter()
                .map(|(n, e)| elaborate_expr(ctx, e).map(|te| (n.clone(), te)))
                .collect::<Result<_, _>>()?;
            TExprKind::Record {
                name: name.clone(),
                fields: tfields,
            }
        }

        ExprKind::FieldAccess { record, field } => {
            let trecord = elaborate_expr(ctx, record)?;
            TExprKind::FieldAccess {
                record: Rc::new(trecord),
                field: field.clone(),
            }
        }

        ExprKind::RecordUpdate { base, updates } => {
            let tbase = elaborate_expr(ctx, base)?;
            let tupdates: Vec<_> = updates
                .iter()
                .map(|(n, e)| elaborate_expr(ctx, e).map(|te| (n.clone(), te)))
                .collect::<Result<_, _>>()?;
            TExprKind::RecordUpdate {
                base: Rc::new(tbase),
                updates: tupdates,
            }
        }

        ExprKind::Constructor { name, args } => {
            let targs: Vec<_> = args
                .iter()
                .map(|e| elaborate_expr(ctx, e))
                .collect::<Result<_, _>>()?;
            TExprKind::Constructor {
                name: name.clone(),
                args: targs,
            }
        }

        ExprKind::BinOp { op, left, right } => {
            let tleft = elaborate_expr(ctx, left)?;
            let tright = elaborate_expr(ctx, right)?;
            let left_ty = &tleft.ty;

            let top = elaborate_binop_with_ctx(op.clone(), left_ty);
            TExprKind::BinOp {
                op: top,
                left: Rc::new(tleft),
                right: Rc::new(tright),
            }
        }

        ExprKind::UnaryOp { op, operand } => {
            let toperand = elaborate_expr(ctx, operand)?;
            let operand_ty = &toperand.ty;

            let top = elaborate_unaryop(*op, operand_ty);
            TExprKind::UnaryOp {
                op: top,
                operand: Rc::new(toperand),
            }
        }

        ExprKind::Seq { first, second } => {
            let tfirst = elaborate_expr(ctx, first)?;
            let tsecond = elaborate_expr(ctx, second)?;
            TExprKind::Seq {
                first: Rc::new(tfirst),
                second: Rc::new(tsecond),
            }
        }

        ExprKind::Perform {
            effect,
            operation,
            args,
        } => {
            let targs: Vec<_> = args
                .iter()
                .map(|e| elaborate_expr(ctx, e))
                .collect::<Result<_, _>>()?;
            TExprKind::Perform {
                effect: effect.clone(),
                op: operation.clone(),
                args: targs,
            }
        }

        ExprKind::Handle {
            body,
            return_clause,
            handlers,
        } => {
            let tbody = elaborate_expr(ctx, body)?;
            let thandler = elaborate_handler(ctx, return_clause, handlers)?;
            TExprKind::Handle {
                body: Rc::new(tbody),
                handler: thandler,
            }
        }

        ExprKind::Hole => {
            TExprKind::Error(format!("Typed hole at {:?}", span))
        }

        // Concurrency primitives
        ExprKind::Spawn(body) => {
            let tbody = elaborate_expr(ctx, body)?;
            // For now, represent spawn as a constructor-like call
            TExprKind::Constructor {
                name: "__spawn".to_string(),
                args: vec![tbody],
            }
        }
        ExprKind::NewChannel => {
            TExprKind::Constructor {
                name: "__new_channel".to_string(),
                args: vec![],
            }
        }
        ExprKind::ChanSend { channel, value } => {
            let tchan = elaborate_expr(ctx, channel)?;
            let tval = elaborate_expr(ctx, value)?;
            TExprKind::Constructor {
                name: "__chan_send".to_string(),
                args: vec![tchan, tval],
            }
        }
        ExprKind::ChanRecv(channel) => {
            let tchan = elaborate_expr(ctx, channel)?;
            TExprKind::Constructor {
                name: "__chan_recv".to_string(),
                args: vec![tchan],
            }
        }
        ExprKind::Select { arms: _ } => {
            // Represent select as an error for now
            TExprKind::Error("select expressions not yet supported in codegen".to_string())
        }
    };

    Ok(TExpr::new(node, ty, span))
}

fn elaborate_rec_binding(ctx: &ElaborateCtx, binding: &RecBinding) -> Result<TRecBinding, String> {
    let tparams: Vec<_> = binding
        .params
        .iter()
        .map(|p| elaborate_pattern(ctx, p))
        .collect();
    let tbody = elaborate_expr(ctx, &binding.body)?;
    let ty = ctx.get_expr_type(&binding.body);

    Ok(TRecBinding {
        name: binding.name.node.clone(),
        params: tparams,
        body: tbody,
        ty,
    })
}

fn elaborate_match_arm(ctx: &ElaborateCtx, arm: &MatchArm) -> Result<TMatchArm, String> {
    let tpat = elaborate_pattern(ctx, &arm.pattern);
    let tguard = arm
        .guard
        .as_ref()
        .map(|g| elaborate_expr(ctx, g).map(Rc::new))
        .transpose()?;
    let tbody = elaborate_expr(ctx, &arm.body)?;

    Ok(TMatchArm {
        pattern: tpat,
        guard: tguard,
        body: tbody,
    })
}

#[allow(clippy::only_used_in_recursion)]
fn elaborate_pattern(ctx: &ElaborateCtx, pattern: &Pattern) -> TPattern {
    let span = pattern.span.clone();
    // Try to get pattern type from context if available
    let ty = Type::Unit; // Pattern types would need separate tracking

    let node = match &pattern.node {
        PatternKind::Wildcard => TPatternKind::Wildcard,
        PatternKind::Var(name) => TPatternKind::Var(name.clone()),
        PatternKind::Lit(lit) => TPatternKind::Lit(lit.clone()),
        PatternKind::Tuple(pats) => {
            let tpats: Vec<_> = pats.iter().map(|p| elaborate_pattern(ctx, p)).collect();
            TPatternKind::Tuple(tpats)
        }
        PatternKind::List(pats) => {
            let tpats: Vec<_> = pats.iter().map(|p| elaborate_pattern(ctx, p)).collect();
            TPatternKind::List(tpats)
        }
        PatternKind::Cons { head, tail } => {
            let thead = elaborate_pattern(ctx, head.as_ref());
            let ttail = elaborate_pattern(ctx, tail.as_ref());
            TPatternKind::Cons {
                head: Rc::new(thead),
                tail: Rc::new(ttail),
            }
        }
        PatternKind::Constructor { name, args } => {
            let targs: Vec<_> = args.iter().map(|p| elaborate_pattern(ctx, p)).collect();
            TPatternKind::Constructor {
                name: name.clone(),
                args: targs,
            }
        }
        PatternKind::Record { name, fields } => {
            let tfields: Vec<_> = fields
                .iter()
                .map(|(n, p)| (n.clone(), p.as_ref().map(|pat| elaborate_pattern(ctx, pat))))
                .collect();
            TPatternKind::Record {
                name: name.clone(),
                fields: tfields,
            }
        }
    };

    TPattern::new(node, ty, span)
}

fn elaborate_handler(
    ctx: &ElaborateCtx,
    return_clause: &HandlerReturn,
    handlers: &[HandlerArm],
) -> Result<THandler, String> {
    let return_tclause = {
        let tpat = elaborate_pattern(ctx, &return_clause.pattern);
        let tbody = elaborate_expr(ctx, &return_clause.body)?;
        THandlerClause {
            pattern: tpat,
            body: Box::new(tbody),
        }
    };

    let op_clauses: Vec<_> = handlers
        .iter()
        .map(|h| {
            let tparams: Vec<_> = h.params.iter().map(|p| elaborate_pattern(ctx, p)).collect();
            let tbody = elaborate_expr(ctx, &h.body)?;
            Ok(TOpClause {
                op_name: h.operation.clone(),
                params: tparams,
                continuation: h.continuation.clone(),
                body: Box::new(tbody),
            })
        })
        .collect::<Result<_, String>>()?;

    Ok(THandler {
        effect: None, // Effect name inferred from operations
        return_clause: Some(return_tclause),
        op_clauses,
    })
}

/// Convert AST binary operator to typed binary operator based on operand type.
/// For equality operators, always use PolyEq/PolyNe which will be resolved to
/// the correct type-specific operator at monomorphization time.
fn elaborate_binop_with_ctx(op: BinOp, left_ty_resolved: &Type) -> TBinOp {
    // For equality operators, always use PolyEq/PolyNe
    // These will be resolved to the correct operator at monomorphization time
    // when we know the actual concrete types (handles polymorphic functions correctly)
    if matches!(op, BinOp::Eq | BinOp::Neq) {
        return match op {
            BinOp::Eq => TBinOp::PolyEq,
            BinOp::Neq => TBinOp::PolyNe,
            _ => unreachable!(),
        };
    }

    // For other operators, use the type-specific variant
    elaborate_binop(op, left_ty_resolved)
}

/// Convert AST binary operator to typed binary operator based on operand type
fn elaborate_binop(op: BinOp, left_ty: &Type) -> TBinOp {
    match op {
        BinOp::Add => match left_ty {
            Type::Float => TBinOp::FloatAdd,
            Type::String => TBinOp::StringConcat,
            _ => TBinOp::IntAdd,
        },
        BinOp::Sub => match left_ty {
            Type::Float => TBinOp::FloatSub,
            _ => TBinOp::IntSub,
        },
        BinOp::Mul => match left_ty {
            Type::Float => TBinOp::FloatMul,
            _ => TBinOp::IntMul,
        },
        BinOp::Div => match left_ty {
            Type::Float => TBinOp::FloatDiv,
            _ => TBinOp::IntDiv,
        },
        BinOp::Mod => TBinOp::IntMod,

        BinOp::Eq => match left_ty {
            Type::Float => TBinOp::FloatEq,
            Type::String => TBinOp::StringEq,
            Type::Bool => TBinOp::BoolEq,
            _ => TBinOp::IntEq,
        },
        BinOp::Neq => match left_ty {
            Type::Float => TBinOp::FloatNe,
            Type::String => TBinOp::StringNe,
            Type::Bool => TBinOp::BoolNe,
            _ => TBinOp::IntNe,
        },
        BinOp::Lt => match left_ty {
            Type::Float => TBinOp::FloatLt,
            _ => TBinOp::IntLt,
        },
        BinOp::Lte => match left_ty {
            Type::Float => TBinOp::FloatLe,
            _ => TBinOp::IntLe,
        },
        BinOp::Gt => match left_ty {
            Type::Float => TBinOp::FloatGt,
            _ => TBinOp::IntGt,
        },
        BinOp::Gte => match left_ty {
            Type::Float => TBinOp::FloatGe,
            _ => TBinOp::IntGe,
        },

        BinOp::And => TBinOp::BoolAnd,
        BinOp::Or => TBinOp::BoolOr,

        BinOp::Cons => TBinOp::Cons,
        BinOp::Concat => TBinOp::StringConcat, // ++ only for strings; use concat for lists

        BinOp::Pipe => TBinOp::Pipe,
        BinOp::PipeBack => TBinOp::PipeLeft,
        BinOp::Compose => TBinOp::Compose,
        BinOp::ComposeBack => TBinOp::ComposeLeft,

        BinOp::UserDefined(_) => TBinOp::IntAdd, // Placeholder for user-defined ops
    }
}

/// Convert AST unary operator to typed unary operator based on operand type
fn elaborate_unaryop(op: UnaryOp, operand_ty: &Type) -> TUnaryOp {
    match op {
        UnaryOp::Neg => match operand_ty {
            Type::Float => TUnaryOp::FloatNeg,
            _ => TUnaryOp::IntNeg,
        },
        UnaryOp::Not => TUnaryOp::BoolNot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Program;
    use crate::prelude::parse_prelude;
    use crate::{Lexer, Parser};

    fn parse_and_elaborate(source: &str) -> TProgram {
        // Parse prelude
        let prelude = parse_prelude().expect("prelude should parse");

        // Parse user program
        let tokens = Lexer::new(source).tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let user_program = parser.parse_program().unwrap();

        // Combine prelude + user program
        let mut combined_items = prelude.items;
        combined_items.extend(user_program.items);
        let program = Program {
            exports: user_program.exports,
            items: combined_items,
        };

        let mut inferencer = Inferencer::new();
        let type_env = inferencer.infer_program(&program, TypeEnv::new()).unwrap();

        elaborate(&program, &inferencer, &type_env).unwrap()
    }

    /// Helper to find a binding by name in the TProgram
    fn find_binding<'a>(tprog: &'a TProgram, name: &str) -> Option<&'a TBinding> {
        tprog.bindings.iter().find(|b| b.name == name)
    }

    #[test]
    fn test_mangle_type() {
        let infer = Inferencer::new();
        let type_env = TypeEnv::new();
        let ctx = ElaborateCtx::new(&infer, &type_env);

        assert_eq!(ctx.mangle_type(&Type::Int), "Int");
        assert_eq!(
            ctx.mangle_type(&Type::Constructor {
                name: "List".to_string(),
                args: vec![Type::Int]
            }),
            "List_Int"
        );
    }

    #[test]
    fn test_elaborate_simple_let() {
        let tprog = parse_and_elaborate("let x = 42");

        let binding = find_binding(&tprog, "x").expect("binding 'x' should exist");
        assert!(matches!(binding.ty, Type::Int));
    }

    #[test]
    fn test_elaborate_function() {
        let tprog = parse_and_elaborate("let add x y = x + y");

        let binding = find_binding(&tprog, "add").expect("binding 'add' should exist");
        // Function type: a -> a -> a (or Int -> Int -> Int after resolution)
        match &binding.ty {
            Type::Arrow { ret, .. } => {
                // First arrow should have another arrow as return
                match ret.as_ref() {
                    Type::Arrow { .. } => {
                        // Nested function type - correct shape
                    }
                    _ => panic!("Expected nested function type, got {:?}", ret),
                }
            }
            _ => panic!("Expected function type, got {:?}", binding.ty),
        }
    }

    #[test]
    fn test_elaborate_show_call() {
        let source = r#"
            let result = show 42
        "#;
        let tprog = parse_and_elaborate(source);

        let binding = find_binding(&tprog, "result").expect("binding 'result' should exist");
        assert!(matches!(binding.ty, Type::String));

        // Check that the body is a MethodCall
        match &binding.body.node {
            TExprKind::MethodCall {
                trait_name,
                method,
                ..
            } => {
                assert_eq!(trait_name, "Show");
                assert_eq!(method, "show");
                // instance_ty should be Int or a resolved type variable
            }
            _ => panic!(
                "Expected MethodCall, got {:?}",
                binding.body.node
            ),
        }
    }

    #[test]
    fn test_elaborate_type_decl() {
        // Note: prelude already includes Option, so we use a different type name
        let source = r#"
            type MyOption a = | MySome a | MyNone
            let x = MySome 1
        "#;
        let tprog = parse_and_elaborate(source);

        // Find our custom type (not prelude's Option)
        let type_decl = tprog.type_decls.iter().find(|t| t.name == "MyOption");
        assert!(type_decl.is_some(), "MyOption type should exist");
        let type_decl = type_decl.unwrap();
        assert_eq!(type_decl.constructors.len(), 2);
    }

    #[test]
    fn test_elaborate_match() {
        // Use a simpler match that doesn't trigger inference edge cases
        let source = r#"
            let classify n =
                match n with
                | 0 -> "zero"
                | _ -> "other"
                end
        "#;
        let tprog = parse_and_elaborate(source);

        let binding = find_binding(&tprog, "classify").expect("binding 'classify' should exist");

        // Unwrap lambda to get to the match
        let mut body = &binding.body;
        while let TExprKind::Lambda { body: inner, .. } = &body.node {
            body = inner;
        }

        match &body.node {
            TExprKind::Match { arms, .. } => {
                assert_eq!(arms.len(), 2);
            }
            _ => panic!("Expected Match expression, got {:?}", body.node),
        }
    }
}
