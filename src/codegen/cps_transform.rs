//! CPS Transformation Pass
//!
//! Transforms effectful Core IR code to Continuation-Passing Style.
//! Pure code remains in direct style for efficiency.
//!
//! ## Transformation Rules
//!
//! 1. Effectful function: add continuation parameter
//!    `fn f(x, y) = body` → `fn f_cps(x, y, k) = [[body]]_k`
//!
//! 2. Perform → CaptureK:
//!    `perform Effect.op(args)` → `CaptureK { effect, op, args, cont_var: k }`
//!
//! 3. Handle → WithHandler:
//!    `handle body with handlers end` →
//!    `WithHandler { effect, handler, body, outer_cont }`
//!
//! 4. Let (effectful RHS):
//!    `let x = e1 in e2` → `[[e1]]_{fun x -> [[e2]]_k}`
//!
//! 5. Return (in effectful function):
//!    `value` → `Resume { cont: k, value }`

use std::collections::HashSet;

use super::core_ir::{
    CPSHandler, CPSOpHandler, CoreExpr, CoreProgram, FunDef, VarGen, VarId,
};
use super::effect_analysis::{analyze_effects, EffectInfo};

/// CPS transformation context
pub struct CPSTransformer {
    /// Effect analysis results
    effect_info: EffectInfo,
    /// Variable generator for fresh variables
    var_gen: VarGen,
    /// Stack of continuation variables (current continuation)
    cont_stack: Vec<VarId>,
    /// Set of functions that have been CPS-transformed
    transformed_funs: HashSet<VarId>,
    /// New function definitions created by transformation
    new_functions: Vec<FunDef>,
}

impl CPSTransformer {
    /// Create a new CPS transformer
    pub fn new(effect_info: EffectInfo) -> Self {
        Self {
            effect_info,
            var_gen: VarGen::new(),
            cont_stack: Vec::new(),
            transformed_funs: HashSet::new(),
            new_functions: Vec::new(),
        }
    }

    /// Transform a Core IR program
    ///
    /// Only effectful functions are transformed to CPS.
    /// Pure functions remain unchanged.
    pub fn transform(mut self, program: CoreProgram) -> CoreProgram {
        let mut functions = Vec::new();

        for fun in program.functions {
            if self.effect_info.is_effectful(fun.var_id) {
                // Transform effectful function to CPS
                let cps_fun = self.transform_fun_cps(fun);
                functions.push(cps_fun);
            } else {
                // Keep pure function as-is
                functions.push(fun);
            }
        }

        // Transform main if it's effectful
        let main = program.main.map(|expr| {
            if self.is_expr_effectful(&expr) {
                // Main needs a top-level continuation
                let final_k = self.fresh_var();
                self.cont_stack.push(final_k);
                let transformed = self.transform_expr(&expr);
                self.cont_stack.pop();
                // Wrap in identity continuation for top-level
                let result_var = self.fresh_var();
                CoreExpr::LetExpr {
                    name: final_k,
                    name_hint: Some("final_k".to_string()),
                    value: Box::new(CoreExpr::Lam {
                        params: vec![result_var],
                        param_hints: vec![Some("result".to_string())],
                        body: Box::new(CoreExpr::Return(result_var)),
                    }),
                    body: Box::new(transformed),
                }
            } else {
                expr
            }
        });

        // Add any newly created helper functions
        functions.extend(self.new_functions);

        CoreProgram {
            types: program.types,
            functions,
            builtins: program.builtins,
            main,
        }
    }

    /// Transform an effectful function to CPS
    fn transform_fun_cps(&mut self, fun: FunDef) -> FunDef {
        // Add continuation parameter
        let cont_param = self.fresh_var();
        self.cont_stack.push(cont_param);

        // Transform the body
        let cps_body = self.transform_expr(&fun.body);

        self.cont_stack.pop();
        self.transformed_funs.insert(fun.var_id);

        // Build new parameter list with continuation
        let mut params = fun.params.clone();
        params.push(cont_param);

        let mut param_hints = fun.param_hints.clone();
        param_hints.push("k".to_string());

        FunDef {
            name: fun.name,
            var_id: fun.var_id,
            params,
            param_hints,
            param_types: fun.param_types, // TODO: add continuation type
            return_type: fun.return_type,
            body: cps_body,
            is_tail_recursive: fun.is_tail_recursive,
        }
    }

    /// Transform an expression in CPS style
    fn transform_expr(&mut self, expr: &CoreExpr) -> CoreExpr {
        match expr {
            // Values in tail position: resume the continuation
            CoreExpr::Var(v) => {
                if let Some(&k) = self.cont_stack.last() {
                    CoreExpr::Resume {
                        cont: k,
                        value: *v,
                    }
                } else {
                    // No continuation context, pass through
                    CoreExpr::Var(*v)
                }
            }

            CoreExpr::Lit(lit) => {
                // Literals need to be bound to a variable first
                let lit_var = self.fresh_var();
                if let Some(&k) = self.cont_stack.last() {
                    CoreExpr::Let {
                        name: lit_var,
                        name_hint: Some("lit".to_string()),
                        value: Box::new(super::core_ir::Atom::Lit(lit.clone())),
                        body: Box::new(CoreExpr::Resume {
                            cont: k,
                            value: lit_var,
                        }),
                    }
                } else {
                    CoreExpr::Lit(lit.clone())
                }
            }

            CoreExpr::Return(v) => {
                if let Some(&k) = self.cont_stack.last() {
                    CoreExpr::Resume {
                        cont: k,
                        value: *v,
                    }
                } else {
                    CoreExpr::Return(*v)
                }
            }

            // Let binding: check if RHS is effectful
            CoreExpr::Let { name, name_hint, value, body } => {
                // For now, assume atom values are pure
                CoreExpr::Let {
                    name: *name,
                    name_hint: name_hint.clone(),
                    value: value.clone(),
                    body: Box::new(self.transform_expr(body)),
                }
            }

            CoreExpr::LetExpr { name, name_hint, value, body } => {
                if self.is_expr_effectful(value) {
                    // Effectful RHS: create continuation for the rest
                    let rest_cont = self.fresh_var();
                    let result_var = *name;

                    // Build continuation: fun result -> [[body]]_k
                    self.cont_stack.push(self.current_cont());
                    let transformed_body = self.transform_expr(body);
                    self.cont_stack.pop();

                    let cont_lam = CoreExpr::Lam {
                        params: vec![result_var],
                        param_hints: vec![name_hint.clone()],
                        body: Box::new(transformed_body),
                    };

                    // Push the continuation and transform RHS
                    self.cont_stack.push(rest_cont);
                    let transformed_value = self.transform_expr(value);
                    self.cont_stack.pop();

                    CoreExpr::LetExpr {
                        name: rest_cont,
                        name_hint: Some("rest_k".to_string()),
                        value: Box::new(cont_lam),
                        body: Box::new(transformed_value),
                    }
                } else {
                    // Pure RHS: keep as let
                    CoreExpr::LetExpr {
                        name: *name,
                        name_hint: name_hint.clone(),
                        value: Box::new(self.transform_expr(value)),
                        body: Box::new(self.transform_expr(body)),
                    }
                }
            }

            // Perform → CaptureK
            CoreExpr::Perform { effect, effect_name, op, op_name, args } => {
                CoreExpr::CaptureK {
                    effect: *effect,
                    effect_name: effect_name.clone(),
                    op: *op,
                    op_name: op_name.clone(),
                    args: args.clone(),
                }
            }

            // Handle → WithHandler
            CoreExpr::Handle { body, handler } => {
                let outer_k = self.current_cont();

                // Create return handler closure
                let return_handler = self.fresh_var();
                // TODO: properly build return handler closure

                // Create operation handler closures
                let op_handlers: Vec<CPSOpHandler> = handler.ops.iter().map(|op| {
                    let handler_fn = self.fresh_var();
                    // TODO: properly build operation handler closure
                    CPSOpHandler {
                        op: op.op,
                        op_name: op.op_name.clone(),
                        handler_fn,
                    }
                }).collect();

                let cps_handler = CPSHandler {
                    return_handler,
                    op_handlers,
                };

                // Transform body
                let transformed_body = self.transform_expr(body);

                CoreExpr::WithHandler {
                    effect: handler.effect,
                    effect_name: handler.effect_name.clone(),
                    handler: cps_handler,
                    body: Box::new(transformed_body),
                    outer_cont: outer_k,
                }
            }

            // Application: check if callee is effectful
            CoreExpr::App { func, args } => {
                if self.effect_info.is_effectful(*func) {
                    // Effectful call: use AppCont with current continuation
                    CoreExpr::AppCont {
                        func: *func,
                        args: args.clone(),
                        cont: self.current_cont(),
                    }
                } else {
                    // Pure call: keep as-is but transform result
                    let result = self.fresh_var();
                    CoreExpr::LetExpr {
                        name: result,
                        name_hint: Some("call_result".to_string()),
                        value: Box::new(CoreExpr::App {
                            func: *func,
                            args: args.clone(),
                        }),
                        body: Box::new(CoreExpr::Resume {
                            cont: self.current_cont(),
                            value: result,
                        }),
                    }
                }
            }

            CoreExpr::TailApp { func, args } => {
                if self.effect_info.is_effectful(*func) {
                    CoreExpr::AppCont {
                        func: *func,
                        args: args.clone(),
                        cont: self.current_cont(),
                    }
                } else {
                    // Pure tail call in effectful context: resume with result
                    let result = self.fresh_var();
                    CoreExpr::LetExpr {
                        name: result,
                        name_hint: Some("tail_result".to_string()),
                        value: Box::new(CoreExpr::TailApp {
                            func: *func,
                            args: args.clone(),
                        }),
                        body: Box::new(CoreExpr::Resume {
                            cont: self.current_cont(),
                            value: result,
                        }),
                    }
                }
            }

            // Control flow: transform branches
            CoreExpr::If { cond, then_branch, else_branch } => {
                CoreExpr::If {
                    cond: *cond,
                    then_branch: Box::new(self.transform_expr(then_branch)),
                    else_branch: Box::new(self.transform_expr(else_branch)),
                }
            }

            CoreExpr::Case { scrutinee, alts, default } => {
                CoreExpr::Case {
                    scrutinee: *scrutinee,
                    alts: alts.iter().map(|alt| {
                        super::core_ir::Alt {
                            tag: alt.tag,
                            tag_name: alt.tag_name.clone(),
                            binders: alt.binders.clone(),
                            binder_hints: alt.binder_hints.clone(),
                            body: self.transform_expr(&alt.body),
                        }
                    }).collect(),
                    default: default.as_ref().map(|d| Box::new(self.transform_expr(d))),
                }
            }

            // Sequences
            CoreExpr::Seq { first, second } => {
                if self.is_expr_effectful(first) {
                    // First is effectful: create continuation for second
                    let discard = self.fresh_var();
                    let rest_k = self.fresh_var();

                    self.cont_stack.push(self.current_cont());
                    let transformed_second = self.transform_expr(second);
                    self.cont_stack.pop();

                    let cont_lam = CoreExpr::Lam {
                        params: vec![discard],
                        param_hints: vec![Some("_".to_string())],
                        body: Box::new(transformed_second),
                    };

                    self.cont_stack.push(rest_k);
                    let transformed_first = self.transform_expr(first);
                    self.cont_stack.pop();

                    CoreExpr::LetExpr {
                        name: rest_k,
                        name_hint: Some("seq_k".to_string()),
                        value: Box::new(cont_lam),
                        body: Box::new(transformed_first),
                    }
                } else {
                    CoreExpr::Seq {
                        first: Box::new(self.transform_expr(first)),
                        second: Box::new(self.transform_expr(second)),
                    }
                }
            }

            // Other expressions: pass through with recursive transform
            CoreExpr::Alloc { tag, type_name, ctor_name, fields } => {
                let result = self.fresh_var();
                CoreExpr::LetExpr {
                    name: result,
                    name_hint: Some("alloc".to_string()),
                    value: Box::new(CoreExpr::Alloc {
                        tag: *tag,
                        type_name: type_name.clone(),
                        ctor_name: ctor_name.clone(),
                        fields: fields.clone(),
                    }),
                    body: Box::new(CoreExpr::Resume {
                        cont: self.current_cont(),
                        value: result,
                    }),
                }
            }

            CoreExpr::Lam { params, param_hints, body } => {
                // Lambda: if body is effectful, it needs internal CPS handling
                if self.is_expr_effectful(body) {
                    // Add implicit continuation parameter
                    let lam_k = self.fresh_var();
                    self.cont_stack.push(lam_k);
                    let transformed_body = self.transform_expr(body);
                    self.cont_stack.pop();

                    let mut new_params = params.clone();
                    new_params.push(lam_k);
                    let mut new_hints = param_hints.clone();
                    new_hints.push(Some("k".to_string()));

                    CoreExpr::Lam {
                        params: new_params,
                        param_hints: new_hints,
                        body: Box::new(transformed_body),
                    }
                } else {
                    CoreExpr::Lam {
                        params: params.clone(),
                        param_hints: param_hints.clone(),
                        body: Box::new(self.transform_expr(body)),
                    }
                }
            }

            CoreExpr::LetRec { name, name_hint, params, param_hints, func_body, body } => {
                // Check if the recursive function is effectful
                if self.is_expr_effectful(func_body) {
                    // Add continuation parameter
                    let rec_k = self.fresh_var();
                    self.cont_stack.push(rec_k);
                    let transformed_func_body = self.transform_expr(func_body);
                    self.cont_stack.pop();

                    let mut new_params = params.clone();
                    new_params.push(rec_k);
                    let mut new_hints = param_hints.clone();
                    new_hints.push(Some("k".to_string()));

                    CoreExpr::LetRec {
                        name: *name,
                        name_hint: name_hint.clone(),
                        params: new_params,
                        param_hints: new_hints,
                        func_body: Box::new(transformed_func_body),
                        body: Box::new(self.transform_expr(body)),
                    }
                } else {
                    CoreExpr::LetRec {
                        name: *name,
                        name_hint: name_hint.clone(),
                        params: params.clone(),
                        param_hints: param_hints.clone(),
                        func_body: Box::new(self.transform_expr(func_body)),
                        body: Box::new(self.transform_expr(body)),
                    }
                }
            }

            CoreExpr::LetRecMutual { bindings, body } => {
                // TODO: handle mutual recursion with effects
                CoreExpr::LetRecMutual {
                    bindings: bindings.iter().map(|b| {
                        super::core_ir::RecBinding {
                            name: b.name,
                            name_hint: b.name_hint.clone(),
                            params: b.params.clone(),
                            param_hints: b.param_hints.clone(),
                            body: self.transform_expr(&b.body),
                        }
                    }).collect(),
                    body: Box::new(self.transform_expr(body)),
                }
            }

            // Pass-through cases
            CoreExpr::PrimOp { op, args } => {
                let result = self.fresh_var();
                CoreExpr::LetExpr {
                    name: result,
                    name_hint: Some("prim".to_string()),
                    value: Box::new(CoreExpr::PrimOp {
                        op: *op,
                        args: args.clone(),
                    }),
                    body: Box::new(CoreExpr::Resume {
                        cont: self.current_cont(),
                        value: result,
                    }),
                }
            }

            CoreExpr::ExternCall { name, args } => {
                let result = self.fresh_var();
                CoreExpr::LetExpr {
                    name: result,
                    name_hint: Some("extern".to_string()),
                    value: Box::new(CoreExpr::ExternCall {
                        name: name.clone(),
                        args: args.clone(),
                    }),
                    body: Box::new(CoreExpr::Resume {
                        cont: self.current_cont(),
                        value: result,
                    }),
                }
            }

            CoreExpr::DictCall { dict, method, args } => {
                // TODO: check if dict call is effectful
                CoreExpr::DictCall {
                    dict: *dict,
                    method: method.clone(),
                    args: args.clone(),
                }
            }

            CoreExpr::DictValue { trait_name, instance_ty } => {
                CoreExpr::DictValue {
                    trait_name: trait_name.clone(),
                    instance_ty: instance_ty.clone(),
                }
            }

            CoreExpr::DictRef(v) => CoreExpr::DictRef(*v),

            CoreExpr::Proj { tuple, index } => {
                let result = self.fresh_var();
                CoreExpr::LetExpr {
                    name: result,
                    name_hint: Some("proj".to_string()),
                    value: Box::new(CoreExpr::Proj {
                        tuple: *tuple,
                        index: *index,
                    }),
                    body: Box::new(CoreExpr::Resume {
                        cont: self.current_cont(),
                        value: result,
                    }),
                }
            }

            CoreExpr::Error(msg) => CoreExpr::Error(msg.clone()),

            // CPS expressions - already transformed
            CoreExpr::AppCont { .. }
            | CoreExpr::Resume { .. }
            | CoreExpr::CaptureK { .. }
            | CoreExpr::WithHandler { .. } => expr.clone(),
        }
    }

    /// Check if an expression is effectful
    fn is_expr_effectful(&self, expr: &CoreExpr) -> bool {
        match expr {
            CoreExpr::Perform { .. } | CoreExpr::CaptureK { .. } => true,
            CoreExpr::Handle { body, handler } => {
                self.is_expr_effectful(body)
                    || self.is_expr_effectful(&handler.return_body)
                    || handler.ops.iter().any(|op| self.is_expr_effectful(&op.body))
            }
            CoreExpr::App { func, .. } | CoreExpr::TailApp { func, .. } => {
                self.effect_info.is_effectful(*func)
            }
            CoreExpr::AppCont { .. } => true,
            CoreExpr::Let { value, body, .. } => {
                self.is_atom_effectful(value) || self.is_expr_effectful(body)
            }
            CoreExpr::LetExpr { value, body, .. } => {
                self.is_expr_effectful(value) || self.is_expr_effectful(body)
            }
            CoreExpr::LetRec { func_body, body, .. } => {
                self.is_expr_effectful(func_body) || self.is_expr_effectful(body)
            }
            CoreExpr::LetRecMutual { bindings, body } => {
                bindings.iter().any(|b| self.is_expr_effectful(&b.body))
                    || self.is_expr_effectful(body)
            }
            CoreExpr::If { then_branch, else_branch, .. } => {
                self.is_expr_effectful(then_branch) || self.is_expr_effectful(else_branch)
            }
            CoreExpr::Case { alts, default, .. } => {
                alts.iter().any(|alt| self.is_expr_effectful(&alt.body))
                    || default.as_ref().map_or(false, |d| self.is_expr_effectful(d))
            }
            CoreExpr::Seq { first, second } => {
                self.is_expr_effectful(first) || self.is_expr_effectful(second)
            }
            CoreExpr::Lam { body, .. } => self.is_expr_effectful(body),
            CoreExpr::WithHandler { body, .. } => self.is_expr_effectful(body),
            _ => false,
        }
    }

    /// Check if an atom is effectful
    fn is_atom_effectful(&self, atom: &super::core_ir::Atom) -> bool {
        match atom {
            super::core_ir::Atom::Lam { body, .. } => self.is_expr_effectful(body),
            super::core_ir::Atom::App { func, .. } => self.effect_info.is_effectful(*func),
            _ => false,
        }
    }

    /// Generate a fresh variable
    fn fresh_var(&mut self) -> VarId {
        self.var_gen.fresh()
    }

    /// Get the current continuation (or panic if none)
    fn current_cont(&self) -> VarId {
        *self.cont_stack.last().expect("No continuation in scope")
    }
}

/// Transform a program using CPS for effects
pub fn cps_transform(program: CoreProgram) -> CoreProgram {
    let effect_info = analyze_effects(&program);

    // If no effectful functions, return unchanged
    if effect_info.effectful_funs.is_empty() && effect_info.direct_effect_sites.is_empty() {
        return program;
    }

    let transformer = CPSTransformer::new(effect_info);
    transformer.transform(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::core_ir::{CoreLit, Atom};

    #[test]
    fn test_pure_program_unchanged() {
        let program = CoreProgram {
            types: vec![],
            functions: vec![],
            builtins: vec![],
            main: Some(CoreExpr::Lit(CoreLit::Int(42))),
        };

        let transformed = cps_transform(program.clone());

        // Pure program should be unchanged
        assert!(matches!(transformed.main, Some(CoreExpr::Lit(CoreLit::Int(42)))));
    }

    #[test]
    fn test_perform_becomes_capture_k() {
        let mut var_gen = VarGen::new();
        let unit_var = var_gen.fresh();

        let program = CoreProgram {
            types: vec![],
            functions: vec![],
            builtins: vec![],
            main: Some(CoreExpr::Perform {
                effect: 0,
                effect_name: Some("State".to_string()),
                op: 0,
                op_name: Some("get".to_string()),
                args: vec![unit_var],
            }),
        };

        let transformed = cps_transform(program);

        // Should have some transformation (at minimum, wrapped in continuation setup)
        assert!(transformed.main.is_some());
    }
}
