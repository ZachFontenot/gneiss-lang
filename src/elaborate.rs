//! Elaboration: Convert untyped AST to typed AST
//!
//! After type inference succeeds, this module walks the AST and builds a typed
//! AST (TExpr, TProgram) where each node is annotated with its inferred type.
//!
//! This enables:
//! 1. Tests that inspect types of sub-expressions
//! 2. Verification that constraints are properly attached
//! 3. Foundation for future compilation passes (codegen needs types)

use crate::ast::*;
use crate::infer::Inferencer;
use crate::typed_ast::*;
use crate::types::{Scheme, Type, TypeEnv};
use std::rc::Rc;

impl Inferencer {
    /// Get the recorded type for an expression span
    pub fn get_expr_type(&self, span: &Span) -> Option<&Type> {
        self.expr_types.get(span)
    }

    /// Elaborate a program into a typed AST.
    /// Must be called after successful type inference.
    pub fn elaborate_program(&self, program: &Program, env: &TypeEnv) -> TProgram {
        let items = program
            .items
            .iter()
            .filter_map(|item| self.elaborate_item(item, env))
            .collect();
        TProgram { items }
    }

    fn elaborate_item(&self, item: &Item, env: &TypeEnv) -> Option<TItem> {
        match item {
            Item::Decl(decl) => self.elaborate_decl(decl, env).map(TItem::Decl),
            Item::Import(_) => None, // Import statements don't appear in typed AST
            Item::Expr(expr) => {
                // Top-level expression as a declaration
                let texpr = self.elaborate_expr(expr, env);
                Some(TItem::Decl(TDecl::Let {
                    pattern: TPattern {
                        kind: TPatternKind::Wildcard,
                        ty: texpr.ty.clone(),
                        span: expr.span.clone(),
                    },
                    scheme: Scheme::mono(texpr.ty.clone()),
                    value: texpr,
                }))
            }
        }
    }

    fn elaborate_decl(&self, decl: &Decl, env: &TypeEnv) -> Option<TDecl> {
        match decl {
            Decl::Let {
                pattern,
                params,
                body,
                ..
            } => {
                // Get the scheme for this binding from the environment
                let scheme = if let PatternKind::Var(name) = &pattern.node {
                    env.get(name)
                        .cloned()
                        .unwrap_or_else(|| Scheme::mono(Type::Unit))
                } else {
                    Scheme::mono(Type::Unit)
                };

                // Build the typed value
                let tvalue = if params.is_empty() {
                    self.elaborate_expr(body, env)
                } else {
                    // Convert params + body into a lambda
                    self.elaborate_function(params, body, &scheme.ty, env)
                };

                let tpattern = self.elaborate_pattern(pattern, &tvalue.ty);

                Some(TDecl::Let {
                    pattern: tpattern,
                    scheme,
                    value: tvalue,
                })
            }

            Decl::LetRec { bindings, .. } => {
                let tbindings = bindings
                    .iter()
                    .map(|binding| {
                        // Get scheme from environment
                        let scheme = env
                            .get(&binding.name.node)
                            .cloned()
                            .unwrap_or_else(|| Scheme::mono(Type::Unit));

                        // Elaborate body
                        let tbody = self.elaborate_expr(&binding.body, env);

                        // Elaborate parameters
                        let param_types = self.infer_param_types(&binding.params, &scheme.ty);
                        let tparams: Vec<_> = binding
                            .params
                            .iter()
                            .zip(param_types.iter())
                            .map(|(pat, ty)| self.elaborate_pattern(pat, ty))
                            .collect();

                        TRecBinding {
                            name: binding.name.node.clone(),
                            scheme,
                            params: tparams,
                            body: tbody,
                        }
                    })
                    .collect();

                Some(TDecl::LetRec { bindings: tbindings })
            }

            Decl::Type {
                name,
                params,
                constructors,
                ..
            } => Some(TDecl::Let {
                // Wrap type decl in a dummy let for now
                // Type declarations pass through as TItem::TypeDecl in real use
                pattern: TPattern {
                    kind: TPatternKind::Var(format!("__type_{}", name)),
                    ty: Type::Unit,
                    span: Span::default(),
                },
                scheme: Scheme::mono(Type::Unit),
                value: TExpr::new(TExprKind::Lit(Literal::Unit), Type::Unit, Span::default()),
            }),

            Decl::Trait { name, type_param, methods, .. } => {
                // Convert trait to TItem::TraitDecl
                // For now, return None as these are handled specially
                None
            }

            Decl::Instance { trait_name, target_type, methods, .. } => {
                // Instance declarations are handled specially
                None
            }

            _ => None, // Val, OperatorDef, Fixity, TypeAlias, Effect
        }
    }

    /// Elaborate a function definition (params + body) into a typed expression
    fn elaborate_function(
        &self,
        params: &[Pattern],
        body: &Expr,
        func_ty: &Type,
        env: &TypeEnv,
    ) -> TExpr {
        let param_types = self.infer_param_types(params, func_ty);
        let tparams: Vec<_> = params
            .iter()
            .zip(param_types.iter())
            .map(|(pat, ty)| self.elaborate_pattern(pat, ty))
            .collect();

        let tbody = self.elaborate_expr(body, env);

        TExpr::new(
            TExprKind::Lambda {
                params: tparams,
                body: Rc::new(tbody),
            },
            func_ty.clone(),
            body.span.clone(),
        )
    }

    /// Elaborate an expression into a typed expression.
    pub fn elaborate_expr(&self, expr: &Expr, env: &TypeEnv) -> TExpr {
        // Look up the recorded type for this expression
        let ty = self
            .expr_types
            .get(&expr.span)
            .cloned()
            .unwrap_or(Type::Unit);

        let kind = match &expr.node {
            ExprKind::Lit(lit) => TExprKind::Lit(lit.clone()),

            ExprKind::Var(name) => {
                let scheme = env.get(name).cloned();
                let instantiated_preds = self
                    .var_preds
                    .get(&expr.span)
                    .cloned()
                    .unwrap_or_default();
                TExprKind::Var {
                    name: name.clone(),
                    scheme,
                    instantiated_preds,
                }
            }

            ExprKind::Lambda { params, body } => {
                let param_types = self.infer_param_types_from_arrow(&ty);
                let tparams: Vec<_> = params
                    .iter()
                    .zip(param_types.iter())
                    .map(|(pat, pty)| self.elaborate_pattern(pat, pty))
                    .collect();
                let tbody = self.elaborate_expr(body, env);
                TExprKind::Lambda {
                    params: tparams,
                    body: Rc::new(tbody),
                }
            }

            ExprKind::App { func, arg } => {
                let tfunc = self.elaborate_expr(func, env);
                let targ = self.elaborate_expr(arg, env);
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
                let tvalue = self.elaborate_expr(value, env);
                let tpattern = self.elaborate_pattern(pattern, &tvalue.ty);

                // Get the scheme for this binding
                let scheme = if let PatternKind::Var(name) = &pattern.node {
                    self.binding_schemes
                        .get(&pattern.span)
                        .cloned()
                        .or_else(|| env.get(name).cloned())
                        .unwrap_or_else(|| Scheme::mono(tvalue.ty.clone()))
                } else {
                    Scheme::mono(tvalue.ty.clone())
                };

                let tbody = body.as_ref().map(|b| Rc::new(self.elaborate_expr(b, env)));

                TExprKind::Let {
                    pattern: tpattern,
                    scheme,
                    value: Rc::new(tvalue),
                    body: tbody,
                }
            }

            ExprKind::LetRec { bindings, body } => {
                let tbindings = bindings
                    .iter()
                    .map(|binding| {
                        let tbody = self.elaborate_expr(&binding.body, env);
                        let scheme = env
                            .get(&binding.name.node)
                            .cloned()
                            .unwrap_or_else(|| Scheme::mono(tbody.ty.clone()));

                        let param_types = self.infer_param_types(&binding.params, &scheme.ty);
                        let tparams: Vec<_> = binding
                            .params
                            .iter()
                            .zip(param_types.iter())
                            .map(|(pat, pty)| self.elaborate_pattern(pat, pty))
                            .collect();

                        TRecBinding {
                            name: binding.name.node.clone(),
                            scheme,
                            params: tparams,
                            body: tbody,
                        }
                    })
                    .collect();

                let tbody = body.as_ref().map(|b| Rc::new(self.elaborate_expr(b, env)));

                TExprKind::LetRec {
                    bindings: tbindings,
                    body: tbody,
                }
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => TExprKind::If {
                cond: Rc::new(self.elaborate_expr(cond, env)),
                then_branch: Rc::new(self.elaborate_expr(then_branch, env)),
                else_branch: Rc::new(self.elaborate_expr(else_branch, env)),
            },

            ExprKind::Match { scrutinee, arms } => {
                let tscrutinee = self.elaborate_expr(scrutinee, env);
                let tarms = arms
                    .iter()
                    .map(|arm| {
                        let tpattern = self.elaborate_pattern(&arm.pattern, &tscrutinee.ty);
                        let tguard = arm.guard.as_ref().map(|g| self.elaborate_expr(g, env));
                        let tbody = self.elaborate_expr(&arm.body, env);
                        TMatchArm {
                            pattern: tpattern,
                            guard: tguard,
                            body: tbody,
                        }
                    })
                    .collect();
                TExprKind::Match {
                    scrutinee: Rc::new(tscrutinee),
                    arms: tarms,
                }
            }

            ExprKind::Tuple(exprs) => {
                let texprs = exprs.iter().map(|e| self.elaborate_expr(e, env)).collect();
                TExprKind::Tuple(texprs)
            }

            ExprKind::List(exprs) => {
                let texprs = exprs.iter().map(|e| self.elaborate_expr(e, env)).collect();
                TExprKind::List(texprs)
            }

            ExprKind::Constructor { name, args } => {
                let targs = args.iter().map(|e| self.elaborate_expr(e, env)).collect();
                TExprKind::Constructor {
                    name: name.clone(),
                    args: targs,
                }
            }

            ExprKind::BinOp { op, left, right } => TExprKind::BinOp {
                op: op.clone(),
                left: Rc::new(self.elaborate_expr(left, env)),
                right: Rc::new(self.elaborate_expr(right, env)),
            },

            ExprKind::UnaryOp { op, operand } => TExprKind::UnaryOp {
                op: op.clone(),
                operand: Rc::new(self.elaborate_expr(operand, env)),
            },

            ExprKind::Seq { first, second } => TExprKind::Seq {
                first: Rc::new(self.elaborate_expr(first, env)),
                second: Rc::new(self.elaborate_expr(second, env)),
            },

            // Concurrency primitives
            ExprKind::Spawn(e) => TExprKind::Spawn(Rc::new(self.elaborate_expr(e, env))),

            ExprKind::NewChannel => TExprKind::NewChannel,

            ExprKind::ChanSend { channel, value } => TExprKind::ChanSend {
                channel: Rc::new(self.elaborate_expr(channel, env)),
                value: Rc::new(self.elaborate_expr(value, env)),
            },

            ExprKind::ChanRecv(channel) => {
                TExprKind::ChanRecv(Rc::new(self.elaborate_expr(channel, env)))
            }

            ExprKind::Select { arms } => {
                let tarms = arms
                    .iter()
                    .map(|arm| {
                        let tchannel = self.elaborate_expr(&arm.channel, env);
                        let elem_ty = self.channel_elem_type(&tchannel.ty);
                        let tpattern = self.elaborate_pattern(&arm.pattern, &elem_ty);
                        let tbody = self.elaborate_expr(&arm.body, env);
                        TSelectArm {
                            channel: tchannel,
                            pattern: tpattern,
                            body: tbody,
                        }
                    })
                    .collect();
                TExprKind::Select { arms: tarms }
            }

            // Algebraic effects
            ExprKind::Perform {
                effect,
                operation,
                args,
            } => TExprKind::Perform {
                effect: effect.clone(),
                operation: operation.clone(),
                args: args.iter().map(|e| self.elaborate_expr(e, env)).collect(),
            },

            ExprKind::Handle {
                body,
                return_clause,
                handlers,
            } => {
                let tbody = self.elaborate_expr(body, env);
                let treturn = THandlerReturn {
                    pattern: self.elaborate_pattern(&return_clause.pattern, &tbody.ty),
                    body: Box::new(self.elaborate_expr(&return_clause.body, env)),
                };
                let thandlers = handlers
                    .iter()
                    .map(|h| {
                        // Look up effect operation parameter types from EffectEnv
                        let param_types = self.effect_operation_param_types(&h.operation, h.params.len());
                        let tparams: Vec<_> = h
                            .params
                            .iter()
                            .zip(param_types.iter())
                            .map(|(p, pty)| self.elaborate_pattern(p, pty))
                            .collect();
                        THandlerArm {
                            operation: h.operation.clone(),
                            params: tparams,
                            continuation: h.continuation.clone(),
                            body: Box::new(self.elaborate_expr(&h.body, env)),
                        }
                    })
                    .collect();
                TExprKind::Handle {
                    body: Rc::new(tbody),
                    return_clause: treturn,
                    handlers: thandlers,
                }
            }

            // Records
            ExprKind::Record { name, fields } => {
                let tfields = fields
                    .iter()
                    .map(|(fname, fexpr)| (fname.clone(), self.elaborate_expr(fexpr, env)))
                    .collect();
                TExprKind::Record {
                    name: name.clone(),
                    fields: tfields,
                }
            }

            ExprKind::FieldAccess { record, field } => TExprKind::FieldAccess {
                record: Rc::new(self.elaborate_expr(record, env)),
                field: field.clone(),
            },

            ExprKind::RecordUpdate { base, updates } => {
                let tupdates = updates
                    .iter()
                    .map(|(fname, fexpr)| (fname.clone(), self.elaborate_expr(fexpr, env)))
                    .collect();
                TExprKind::RecordUpdate {
                    base: Rc::new(self.elaborate_expr(base, env)),
                    updates: tupdates,
                }
            }

            ExprKind::Hole => TExprKind::Hole,
        };

        TExpr::new(kind, ty, expr.span.clone())
    }

    fn elaborate_pattern(&self, pattern: &Pattern, ty: &Type) -> TPattern {
        let kind = match &pattern.node {
            PatternKind::Var(name) => TPatternKind::Var(name.clone()),
            PatternKind::Wildcard => TPatternKind::Wildcard,
            PatternKind::Lit(lit) => TPatternKind::Lit(lit.clone()),
            PatternKind::Constructor { name, args } => {
                // For constructor patterns, look up arg types from TypeContext
                let arg_types = self.constructor_arg_types(ty, name, args.len());
                let targs = args
                    .iter()
                    .zip(arg_types.iter())
                    .map(|(pat, pty)| self.elaborate_pattern(pat, pty))
                    .collect();
                TPatternKind::Constructor {
                    name: name.clone(),
                    args: targs,
                }
            }
            PatternKind::Tuple(pats) => {
                let elem_types = self.tuple_elem_types(ty);
                let tpats = pats
                    .iter()
                    .zip(elem_types.iter())
                    .map(|(pat, pty)| self.elaborate_pattern(pat, pty))
                    .collect();
                TPatternKind::Tuple(tpats)
            }
            PatternKind::List(pats) => {
                let elem_ty = self.list_elem_type(ty);
                let tpats = pats
                    .iter()
                    .map(|pat| self.elaborate_pattern(pat, &elem_ty))
                    .collect();
                TPatternKind::List(tpats)
            }
            PatternKind::Cons { head, tail } => {
                let elem_ty = self.list_elem_type(ty);
                TPatternKind::Cons {
                    head: Box::new(self.elaborate_pattern(head, &elem_ty)),
                    tail: Box::new(self.elaborate_pattern(tail, ty)),
                }
            }
            PatternKind::Record { name, fields } => {
                // For record patterns, look up field types from TypeContext
                let tfields = fields
                    .iter()
                    .map(|(field_name, opt_pat)| {
                        let field_ty = self.record_field_type(name, field_name);
                        let tpat = opt_pat.as_ref().map(|p| self.elaborate_pattern(p, &field_ty));
                        (field_name.clone(), tpat)
                    })
                    .collect();
                TPatternKind::Record {
                    name: name.clone(),
                    fields: tfields,
                }
            }
        };
        TPattern {
            kind,
            ty: ty.clone(),
            span: pattern.span.clone(),
        }
    }

    // Helper methods for type extraction

    fn infer_param_types(&self, params: &[Pattern], func_ty: &Type) -> Vec<Type> {
        let mut types = Vec::new();
        let mut current_ty = func_ty.clone();
        for _ in params {
            if let Type::Arrow { arg, ret, .. } = &current_ty {
                types.push((**arg).clone());
                current_ty = (**ret).clone();
            } else {
                types.push(Type::Unit); // Fallback
            }
        }
        types
    }

    fn infer_param_types_from_arrow(&self, ty: &Type) -> Vec<Type> {
        let mut types = Vec::new();
        let mut current_ty = ty.clone();
        while let Type::Arrow { arg, ret, .. } = &current_ty {
            types.push((**arg).clone());
            current_ty = (**ret).clone();
        }
        types
    }

    fn tuple_elem_types(&self, ty: &Type) -> Vec<Type> {
        if let Type::Tuple(types) = ty {
            types.clone()
        } else {
            vec![]
        }
    }

    fn list_elem_type(&self, ty: &Type) -> Type {
        if let Type::Constructor { name, args } = ty {
            if name == "List" && !args.is_empty() {
                return args[0].clone();
            }
        }
        Type::Unit
    }

    fn channel_elem_type(&self, ty: &Type) -> Type {
        if let Type::Channel(inner) = ty {
            (**inner).clone()
        } else {
            Type::Unit
        }
    }

    /// Get the argument types for a constructor pattern.
    /// Uses TypeContext to look up the constructor's field types.
    fn constructor_arg_types(&self, _ty: &Type, name: &str, count: usize) -> Vec<Type> {
        // Look up the constructor in type_ctx
        if let Some(info) = self.type_ctx.get_constructor(name) {
            // If the constructor is generic, we need to instantiate it
            // For now, just return the field types directly
            // TODO: Apply type substitution for generic constructors
            if info.field_types.len() == count {
                return info.field_types.clone();
            }
        }
        // Fallback to Unit placeholders if not found
        vec![Type::Unit; count]
    }

    /// Get the field types for a record pattern.
    /// Uses TypeContext to look up the record's field types.
    fn record_field_type(&self, record_name: &str, field_name: &str) -> Type {
        if let Some(info) = self.type_ctx.get_record(record_name) {
            if let Some(field_ty) = info.field_types.get(field_name) {
                return field_ty.clone();
            }
        }
        Type::Unit
    }

    /// Get the parameter types for an effect operation.
    /// Uses EffectEnv to look up the operation's parameter types.
    fn effect_operation_param_types(&self, operation_name: &str, count: usize) -> Vec<Type> {
        if let Some((_, op_info)) = self.effect_env.operations.get(operation_name) {
            if op_info.param_types.len() == count {
                return op_info.param_types.clone();
            }
        }
        // Fallback to Unit placeholders if not found
        vec![Type::Unit; count]
    }
}
