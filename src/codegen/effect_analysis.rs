//! Effect Analysis Pass
//!
//! Analyzes the Core IR program to identify which functions are effectful.
//! A function is effectful if it:
//! 1. Contains a Perform expression directly
//! 2. Calls another effectful function
//!
//! This information is used by the CPS transformation to decide which
//! functions need to be transformed to continuation-passing style.

use std::collections::{HashMap, HashSet};

use super::core_ir::{CoreExpr, CoreProgram, VarId};

/// Result of effect analysis
#[derive(Debug, Clone, Default)]
pub struct EffectInfo {
    /// Functions that may perform effects (VarId -> function name)
    pub effectful_funs: HashSet<VarId>,
    /// Map from function name to VarId for lookups
    pub fun_name_to_id: HashMap<String, VarId>,
    /// Map from VarId to function name
    pub fun_id_to_name: HashMap<VarId, String>,
    /// Direct effect sites: functions that directly contain Perform
    pub direct_effect_sites: HashSet<VarId>,
}

impl EffectInfo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a function is effectful
    pub fn is_effectful(&self, var: VarId) -> bool {
        self.effectful_funs.contains(&var)
    }

    /// Check if a function is effectful by name
    pub fn is_effectful_by_name(&self, name: &str) -> bool {
        self.fun_name_to_id
            .get(name)
            .map(|id| self.effectful_funs.contains(id))
            .unwrap_or(false)
    }
}

/// Analyze a Core IR program for effects
pub fn analyze_effects(program: &CoreProgram) -> EffectInfo {
    let mut info = EffectInfo::new();

    // Build name <-> VarId mappings
    for fun in &program.functions {
        info.fun_name_to_id.insert(fun.name.clone(), fun.var_id);
        info.fun_id_to_name.insert(fun.var_id, fun.name.clone());
    }

    // Phase 1: Find functions with direct Perform expressions
    for fun in &program.functions {
        if contains_perform(&fun.body) {
            info.direct_effect_sites.insert(fun.var_id);
            info.effectful_funs.insert(fun.var_id);
        }
    }

    // Also check main expression
    if let Some(ref main) = program.main {
        if contains_perform(main) {
            // Main is effectful, but we don't have a VarId for it
            // This is handled specially in CPS transform
        }
    }

    // Phase 2: Propagate effectfulness through the call graph
    // Keep iterating until we reach a fixed point
    let mut changed = true;
    while changed {
        changed = false;

        for fun in &program.functions {
            if info.effectful_funs.contains(&fun.var_id) {
                // Already marked as effectful
                continue;
            }

            // Check if this function calls any effectful functions
            if calls_effectful(&fun.body, &info) {
                info.effectful_funs.insert(fun.var_id);
                changed = true;
            }
        }
    }

    info
}

/// Check if an expression directly contains a Perform
fn contains_perform(expr: &CoreExpr) -> bool {
    match expr {
        CoreExpr::Perform { .. } => true,

        // Handle is effectful but the handler body handles the effects
        // The body inside handle might perform effects that are caught
        CoreExpr::Handle { body, handler } => {
            // Check if body performs effects not handled by this handler
            // For simplicity, we treat Handle as not directly effectful
            // since effects inside are supposed to be handled
            contains_perform(body)
                || contains_perform(&handler.return_body)
                || handler.ops.iter().any(|op| contains_perform(&op.body))
        }

        CoreExpr::Var(_) | CoreExpr::Lit(_) | CoreExpr::Return(_) => false,

        CoreExpr::Let { value, body, .. } => contains_perform_atom(value) || contains_perform(body),

        CoreExpr::LetExpr { value, body, .. } => {
            contains_perform(value) || contains_perform(body)
        }

        CoreExpr::App { .. } | CoreExpr::TailApp { .. } => {
            // App itself isn't effectful; effectfulness depends on the callee
            false
        }

        CoreExpr::Alloc { .. } => false,

        CoreExpr::Case {
            alts, default, ..
        } => {
            alts.iter().any(|alt| contains_perform(&alt.body))
                || default.as_ref().map_or(false, |d| contains_perform(d))
        }

        CoreExpr::If {
            then_branch,
            else_branch,
            ..
        } => contains_perform(then_branch) || contains_perform(else_branch),

        CoreExpr::Lam { body, .. } => contains_perform(body),

        CoreExpr::LetRec {
            func_body, body, ..
        } => contains_perform(func_body) || contains_perform(body),

        CoreExpr::LetRecMutual { bindings, body } => {
            bindings.iter().any(|b| contains_perform(&b.body)) || contains_perform(body)
        }

        CoreExpr::Seq { first, second } => contains_perform(first) || contains_perform(second),

        CoreExpr::PrimOp { .. } | CoreExpr::ExternCall { .. } => false,

        CoreExpr::DictCall { .. } | CoreExpr::DictValue { .. } | CoreExpr::DictRef(_) => false,

        CoreExpr::Proj { .. } => false,

        CoreExpr::Error(_) => false,

        // CPS-transformed expressions - these are produced after CPS transform
        // and indicate the code has already been processed for effects
        CoreExpr::AppCont { .. } => false, // CPS call, effectfulness handled
        CoreExpr::Resume { .. } => false,  // Resume is not itself a perform
        CoreExpr::CaptureK { .. } => true, // CaptureK IS the CPS version of Perform
        CoreExpr::WithHandler { body, .. } => contains_perform(body),
    }
}

/// Check if an atom contains a Perform (only Lam can)
fn contains_perform_atom(atom: &super::core_ir::Atom) -> bool {
    match atom {
        super::core_ir::Atom::Lam { body, .. } => contains_perform(body),
        super::core_ir::Atom::App { .. } => false, // App effectfulness depends on callee
        _ => false,
    }
}

/// Check if an expression calls any effectful functions
fn calls_effectful(expr: &CoreExpr, info: &EffectInfo) -> bool {
    match expr {
        CoreExpr::Perform { .. } => true, // Already counted in direct_effect_sites

        CoreExpr::Handle { body, handler } => {
            calls_effectful(body, info)
                || calls_effectful(&handler.return_body, info)
                || handler.ops.iter().any(|op| calls_effectful(&op.body, info))
        }

        CoreExpr::Var(_) | CoreExpr::Lit(_) | CoreExpr::Return(_) => false,

        CoreExpr::Let { value, body, .. } => {
            calls_effectful_atom(value, info) || calls_effectful(body, info)
        }

        CoreExpr::LetExpr { value, body, .. } => {
            calls_effectful(value, info) || calls_effectful(body, info)
        }

        CoreExpr::App { func, .. } | CoreExpr::TailApp { func, .. } => {
            // Check if the callee is effectful
            info.is_effectful(*func)
        }

        CoreExpr::Alloc { .. } => false,

        CoreExpr::Case {
            alts, default, ..
        } => {
            alts.iter().any(|alt| calls_effectful(&alt.body, info))
                || default.as_ref().map_or(false, |d| calls_effectful(d, info))
        }

        CoreExpr::If {
            then_branch,
            else_branch,
            ..
        } => calls_effectful(then_branch, info) || calls_effectful(else_branch, info),

        CoreExpr::Lam { body, .. } => calls_effectful(body, info),

        CoreExpr::LetRec {
            func_body, body, ..
        } => calls_effectful(func_body, info) || calls_effectful(body, info),

        CoreExpr::LetRecMutual { bindings, body } => {
            bindings.iter().any(|b| calls_effectful(&b.body, info)) || calls_effectful(body, info)
        }

        CoreExpr::Seq { first, second } => {
            calls_effectful(first, info) || calls_effectful(second, info)
        }

        CoreExpr::PrimOp { .. } | CoreExpr::ExternCall { .. } => false,

        CoreExpr::DictCall { .. } | CoreExpr::DictValue { .. } | CoreExpr::DictRef(_) => false,

        CoreExpr::Proj { .. } => false,

        CoreExpr::Error(_) => false,

        // CPS-transformed expressions
        CoreExpr::AppCont { func, .. } => info.is_effectful(*func),
        CoreExpr::Resume { .. } => false,
        CoreExpr::CaptureK { .. } => true, // CaptureK is effectful
        CoreExpr::WithHandler { body, .. } => calls_effectful(body, info),
    }
}

/// Check if an atom calls effectful functions
fn calls_effectful_atom(atom: &super::core_ir::Atom, info: &EffectInfo) -> bool {
    match atom {
        super::core_ir::Atom::Lam { body, .. } => calls_effectful(body, info),
        super::core_ir::Atom::App { func, .. } => info.is_effectful(*func),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::core_ir::{Atom, CoreLit, FunDef, VarGen};

    fn make_simple_program(main_expr: CoreExpr) -> CoreProgram {
        CoreProgram {
            types: vec![],
            functions: vec![],
            builtins: vec![],
            main: Some(main_expr),
        }
    }

    #[test]
    fn test_pure_program_has_no_effects() {
        let program = make_simple_program(CoreExpr::Lit(CoreLit::Int(42)));
        let info = analyze_effects(&program);
        assert!(info.effectful_funs.is_empty());
        assert!(info.direct_effect_sites.is_empty());
    }

    #[test]
    fn test_perform_detected() {
        let program = make_simple_program(CoreExpr::Perform {
            effect: 0,
            effect_name: Some("State".to_string()),
            op: 0,
            op_name: Some("get".to_string()),
            args: vec![],
        });

        let info = analyze_effects(&program);
        // Main expression is effectful, but we don't track it with VarId
        // The test just verifies the analysis doesn't crash
        assert!(info.effectful_funs.is_empty()); // No named functions
    }

    #[test]
    fn test_function_with_perform_is_effectful() {
        let mut var_gen = VarGen::new();
        let func_id = var_gen.fresh();
        let param = var_gen.fresh();

        let func = FunDef {
            name: "get_state".to_string(),
            var_id: func_id,
            params: vec![param],
            param_hints: vec!["_".to_string()],
            param_types: vec![],
            return_type: super::super::core_ir::CoreType::Int,
            body: CoreExpr::Perform {
                effect: 0,
                effect_name: Some("State".to_string()),
                op: 0,
                op_name: Some("get".to_string()),
                args: vec![],
            },
            is_tail_recursive: false,
        };

        let program = CoreProgram {
            types: vec![],
            functions: vec![func],
            builtins: vec![],
            main: Some(CoreExpr::Lit(CoreLit::Int(0))),
        };

        let info = analyze_effects(&program);
        assert!(info.effectful_funs.contains(&func_id));
        assert!(info.direct_effect_sites.contains(&func_id));
    }

    #[test]
    fn test_effectfulness_propagates() {
        let mut var_gen = VarGen::new();

        // Function A: directly performs effect
        let func_a_id = var_gen.fresh();
        let func_a_param = var_gen.fresh();
        let func_a = FunDef {
            name: "effectful_a".to_string(),
            var_id: func_a_id,
            params: vec![func_a_param],
            param_hints: vec!["_".to_string()],
            param_types: vec![],
            return_type: super::super::core_ir::CoreType::Int,
            body: CoreExpr::Perform {
                effect: 0,
                effect_name: Some("State".to_string()),
                op: 0,
                op_name: Some("get".to_string()),
                args: vec![],
            },
            is_tail_recursive: false,
        };

        // Function B: calls A (should become effectful)
        let func_b_id = var_gen.fresh();
        let func_b_param = var_gen.fresh();
        let func_b = FunDef {
            name: "calls_a".to_string(),
            var_id: func_b_id,
            params: vec![func_b_param],
            param_hints: vec!["_".to_string()],
            param_types: vec![],
            return_type: super::super::core_ir::CoreType::Int,
            body: CoreExpr::App {
                func: func_a_id,
                args: vec![func_b_param],
            },
            is_tail_recursive: false,
        };

        // Function C: calls B (should also become effectful)
        let func_c_id = var_gen.fresh();
        let func_c_param = var_gen.fresh();
        let func_c = FunDef {
            name: "calls_b".to_string(),
            var_id: func_c_id,
            params: vec![func_c_param],
            param_hints: vec!["_".to_string()],
            param_types: vec![],
            return_type: super::super::core_ir::CoreType::Int,
            body: CoreExpr::App {
                func: func_b_id,
                args: vec![func_c_param],
            },
            is_tail_recursive: false,
        };

        let program = CoreProgram {
            types: vec![],
            functions: vec![func_a, func_b, func_c],
            builtins: vec![],
            main: Some(CoreExpr::Lit(CoreLit::Int(0))),
        };

        let info = analyze_effects(&program);

        // All three functions should be marked effectful
        assert!(info.effectful_funs.contains(&func_a_id));
        assert!(info.effectful_funs.contains(&func_b_id));
        assert!(info.effectful_funs.contains(&func_c_id));

        // Only A directly performs effects
        assert!(info.direct_effect_sites.contains(&func_a_id));
        assert!(!info.direct_effect_sites.contains(&func_b_id));
        assert!(!info.direct_effect_sites.contains(&func_c_id));
    }

    #[test]
    fn test_pure_function_not_effectful() {
        let mut var_gen = VarGen::new();
        let func_id = var_gen.fresh();
        let param = var_gen.fresh();

        let func = FunDef {
            name: "pure_fn".to_string(),
            var_id: func_id,
            params: vec![param],
            param_hints: vec!["x".to_string()],
            param_types: vec![],
            return_type: super::super::core_ir::CoreType::Int,
            body: CoreExpr::Let {
                name: var_gen.fresh(),
                name_hint: Some("result".to_string()),
                value: Box::new(Atom::Lit(CoreLit::Int(42))),
                body: Box::new(CoreExpr::Return(param)),
            },
            is_tail_recursive: false,
        };

        let program = CoreProgram {
            types: vec![],
            functions: vec![func],
            builtins: vec![],
            main: Some(CoreExpr::Lit(CoreLit::Int(0))),
        };

        let info = analyze_effects(&program);
        assert!(!info.effectful_funs.contains(&func_id));
    }
}
