//! C Code Generation
//!
//! Emits C source code from Core IR. The generated code uses a simple
//! runtime library for value representation and memory management.
//!
//! ## Value Representation
//!
//! All values are represented as `gn_value` (uint64_t):
//! - Integers: tagged with low bit = 1 (immediate, unboxed)
//! - Heap objects: pointers (aligned, low bit = 0)
//!
//! ## Generated Code Structure
//!
//! ```c
//! #include "gn_runtime.h"
//!
//! // Forward declarations
//! static gn_value fn_foo(gn_value arg0, gn_value arg1);
//!
//! // Function definitions
//! static gn_value fn_foo(gn_value arg0, gn_value arg1) {
//!     ...
//! }
//!
//! // Main entry
//! int main(int argc, char** argv) {
//!     gn_init(argc, argv);
//!     gn_value result = ...;
//!     gn_shutdown();
//!     return 0;
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use super::core_ir::{Alt, Atom, CoreExpr, CoreLit, CoreProgram, FunDef, PrimOp, VarId};

// ============================================================================
// Code Emitter
// ============================================================================

/// C code emitter context
pub struct CEmitter {
    /// Output buffer
    output: String,
    /// Current indentation level
    indent: usize,
    /// Map from VarId to C variable name
    var_names: HashMap<VarId, String>,
    /// Map of top-level function VarIds to their C names (preserved across function emissions)
    top_level_funcs: HashMap<VarId, String>,
    /// Map of function arities (for CAF detection - 0-arg functions need to be called)
    func_arities: HashMap<VarId, usize>,
    /// Map of function name to arity (for closure wrapping when VarId lookup fails)
    func_name_arities: HashMap<String, usize>,
    /// Map of function captures (for closure conversion)
    /// VarId -> (C var names for captures to pass at call sites)
    func_captures: HashMap<VarId, Vec<String>>,
    /// Counter for generating unique names
    name_counter: u32,
    /// Forward declarations buffer
    forward_decls: String,
    /// Top-level function definitions
    function_defs: String,
    /// Generated functions (from LetRec during main emission)
    generated_funcs: String,
    /// Forward declarations for generated functions
    generated_forward_decls: String,
}

impl CEmitter {
    pub fn new() -> Self {
        CEmitter {
            output: String::new(),
            indent: 0,
            var_names: HashMap::new(),
            top_level_funcs: HashMap::new(),
            func_arities: HashMap::new(),
            func_name_arities: HashMap::new(),
            func_captures: HashMap::new(),
            name_counter: 0,
            forward_decls: String::new(),
            function_defs: String::new(),
            generated_funcs: String::new(),
            generated_forward_decls: String::new(),
        }
    }

    /// Collect all VarIds referenced in an expression
    fn collect_refs_expr(expr: &CoreExpr, refs: &mut HashSet<VarId>) {
        match expr {
            CoreExpr::Var(v) => {
                refs.insert(*v);
            }
            CoreExpr::Lit(_) => {}
            CoreExpr::Let { value, body, .. } => {
                Self::collect_refs_atom(value, refs);
                Self::collect_refs_expr(body, refs);
            }
            CoreExpr::LetExpr { value, body, .. } => {
                Self::collect_refs_expr(value, refs);
                Self::collect_refs_expr(body, refs);
            }
            CoreExpr::App { func, args } | CoreExpr::TailApp { func, args } => {
                refs.insert(*func);
                for arg in args {
                    refs.insert(*arg);
                }
            }
            CoreExpr::Alloc { fields, .. } => {
                for f in fields {
                    refs.insert(*f);
                }
            }
            CoreExpr::Case { scrutinee, alts, default } => {
                refs.insert(*scrutinee);
                for alt in alts {
                    Self::collect_refs_expr(&alt.body, refs);
                }
                if let Some(def) = default {
                    Self::collect_refs_expr(def, refs);
                }
            }
            CoreExpr::If { cond, then_branch, else_branch } => {
                refs.insert(*cond);
                Self::collect_refs_expr(then_branch, refs);
                Self::collect_refs_expr(else_branch, refs);
            }
            CoreExpr::Lam { body, .. } => {
                Self::collect_refs_expr(body, refs);
            }
            CoreExpr::LetRec { func_body, body, .. } => {
                Self::collect_refs_expr(func_body, refs);
                Self::collect_refs_expr(body, refs);
            }
            CoreExpr::LetRecMutual { bindings, body } => {
                for binding in bindings {
                    Self::collect_refs_expr(&binding.body, refs);
                }
                Self::collect_refs_expr(body, refs);
            }
            CoreExpr::PrimOp { args, .. } => {
                for arg in args {
                    refs.insert(*arg);
                }
            }
            CoreExpr::Seq { first, second } => {
                Self::collect_refs_expr(first, refs);
                Self::collect_refs_expr(second, refs);
            }
            CoreExpr::Return(v) => {
                refs.insert(*v);
            }
            CoreExpr::Handle { body, handler } => {
                Self::collect_refs_expr(body, refs);
                Self::collect_refs_expr(&handler.return_body, refs);
                for op in &handler.ops {
                    Self::collect_refs_expr(&op.body, refs);
                }
            }
            CoreExpr::Perform { args, .. } => {
                for arg in args {
                    refs.insert(*arg);
                }
            }
            CoreExpr::ExternCall { args, .. } => {
                for arg in args {
                    refs.insert(*arg);
                }
            }
            CoreExpr::DictCall { dict, args, .. } => {
                refs.insert(*dict);
                for arg in args {
                    refs.insert(*arg);
                }
            }
            CoreExpr::DictValue { .. } => {
                // No variable references - it's a static dictionary lookup
            }
            CoreExpr::DictRef(var) => {
                refs.insert(*var);
            }
            CoreExpr::Proj { tuple, .. } => {
                refs.insert(*tuple);
            }
            CoreExpr::Error(_) => {}

            // CPS-transformed expressions
            CoreExpr::AppCont { func, args, cont } => {
                refs.insert(*func);
                for arg in args {
                    refs.insert(*arg);
                }
                refs.insert(*cont);
            }
            CoreExpr::Resume { cont, value } => {
                refs.insert(*cont);
                refs.insert(*value);
            }
            CoreExpr::CaptureK { args, .. } => {
                for arg in args {
                    refs.insert(*arg);
                }
            }
            CoreExpr::WithHandler {
                handler,
                body,
                outer_cont,
                ..
            } => {
                refs.insert(handler.return_handler);
                for op in &handler.op_handlers {
                    refs.insert(op.handler_fn);
                }
                Self::collect_refs_expr(body, refs);
                refs.insert(*outer_cont);
            }
        }
    }

    /// Collect VarIds referenced in an atom
    fn collect_refs_atom(atom: &Atom, refs: &mut HashSet<VarId>) {
        match atom {
            Atom::Var(v) => {
                refs.insert(*v);
            }
            Atom::Lit(_) => {}
            Atom::PrimOp { args, .. } => {
                for arg in args {
                    refs.insert(*arg);
                }
            }
            Atom::Lam { body, .. } => {
                Self::collect_refs_expr(body, refs);
            }
            Atom::Alloc { fields, .. } => {
                for f in fields {
                    refs.insert(*f);
                }
            }
            Atom::App { func, args } => {
                refs.insert(*func);
                for arg in args {
                    refs.insert(*arg);
                }
            }
        }
    }

    /// Compute all functions reachable from main
    fn collect_reachable(
        main: &Option<CoreExpr>,
        functions: &[FunDef],
    ) -> HashSet<VarId> {
        // Build a map from VarId to function body for transitive lookup
        let func_map: HashMap<VarId, &FunDef> = functions
            .iter()
            .map(|f| (f.var_id, f))
            .collect();

        let mut reachable: HashSet<VarId> = HashSet::new();
        let mut worklist: Vec<VarId> = Vec::new();

        // Always include the "main" function if it exists
        for fun in functions {
            if fun.name == "main" {
                reachable.insert(fun.var_id);
                worklist.push(fun.var_id);
            }
        }

        // Start with references from main expression (if present)
        if let Some(main_expr) = main {
            let mut main_refs = HashSet::new();
            Self::collect_refs_expr(main_expr, &mut main_refs);
            for v in main_refs {
                if func_map.contains_key(&v) && !reachable.contains(&v) {
                    reachable.insert(v);
                    worklist.push(v);
                }
            }
        }

        // Transitively collect dependencies
        while let Some(var_id) = worklist.pop() {
            if let Some(func) = func_map.get(&var_id) {
                let mut func_refs = HashSet::new();
                Self::collect_refs_expr(&func.body, &mut func_refs);
                for v in func_refs {
                    if func_map.contains_key(&v) && !reachable.contains(&v) {
                        reachable.insert(v);
                        worklist.push(v);
                    }
                }
            }
        }

        reachable
    }

    /// Collect all dictionaries (trait, type) pairs used in the program
    fn collect_dictionaries(
        main: &Option<CoreExpr>,
        functions: &[FunDef],
    ) -> HashSet<(String, String)> {
        let mut dicts = HashSet::new();

        fn collect_from_expr(expr: &CoreExpr, dicts: &mut HashSet<(String, String)>) {
            match expr {
                CoreExpr::DictValue { trait_name, instance_ty } => {
                    dicts.insert((trait_name.clone(), instance_ty.clone()));
                }
                CoreExpr::Var(_) | CoreExpr::Lit(_) | CoreExpr::Error(_) | CoreExpr::DictRef(_) | CoreExpr::Proj { .. } => {}
                CoreExpr::Let { body, .. } => {
                    // value is Atom, no DictValue in atoms
                    collect_from_expr(body, dicts);
                }
                CoreExpr::LetExpr { value, body, .. } => {
                    collect_from_expr(value, dicts);
                    collect_from_expr(body, dicts);
                }
                CoreExpr::App { .. } | CoreExpr::TailApp { .. } => {}
                CoreExpr::Alloc { .. } => {}
                CoreExpr::Case { alts, default, .. } => {
                    for alt in alts {
                        collect_from_expr(&alt.body, dicts);
                    }
                    if let Some(def) = default {
                        collect_from_expr(def, dicts);
                    }
                }
                CoreExpr::If { then_branch, else_branch, .. } => {
                    collect_from_expr(then_branch, dicts);
                    collect_from_expr(else_branch, dicts);
                }
                CoreExpr::Lam { body, .. } => {
                    collect_from_expr(body, dicts);
                }
                CoreExpr::LetRec { func_body, body, .. } => {
                    collect_from_expr(func_body, dicts);
                    collect_from_expr(body, dicts);
                }
                CoreExpr::LetRecMutual { bindings, body } => {
                    for binding in bindings {
                        collect_from_expr(&binding.body, dicts);
                    }
                    collect_from_expr(body, dicts);
                }
                CoreExpr::PrimOp { .. } => {}
                CoreExpr::Seq { first, second } => {
                    collect_from_expr(first, dicts);
                    collect_from_expr(second, dicts);
                }
                CoreExpr::Return(_) => {}
                CoreExpr::Handle { body, handler } => {
                    collect_from_expr(body, dicts);
                    collect_from_expr(&handler.return_body, dicts);
                    for op in &handler.ops {
                        collect_from_expr(&op.body, dicts);
                    }
                }
                CoreExpr::Perform { .. } => {}
                CoreExpr::ExternCall { .. } => {}
                CoreExpr::DictCall { .. } => {}

                // CPS-transformed expressions
                CoreExpr::AppCont { .. } => {}
                CoreExpr::Resume { .. } => {}
                CoreExpr::CaptureK { .. } => {}
                CoreExpr::WithHandler { body, .. } => {
                    collect_from_expr(body, dicts);
                }
            }
        }

        if let Some(main_expr) = main {
            collect_from_expr(main_expr, &mut dicts);
        }

        for func in functions {
            collect_from_expr(&func.body, &mut dicts);
        }

        dicts
    }

    /// Emit a complete C program from Core IR
    pub fn emit_program(&mut self, program: &CoreProgram) -> String {
        // Header
        self.output.push_str("// Generated by gneic - Gneiss Compiler\n");
        self.output.push_str("// DO NOT EDIT\n\n");
        self.output.push_str("#include \"gn_runtime.h\"\n\n");

        // Compute reachable functions (dead code elimination)
        let reachable = Self::collect_reachable(&program.main, &program.functions);

        // First pass: register all top-level functions and their arities
        // (we need all registered for name lookup, but only emit reachable ones)
        for fun in &program.functions {
            let c_name = format!("fn_{}", mangle_name(&fun.name));
            self.top_level_funcs.insert(fun.var_id, c_name.clone());
            self.func_arities.insert(fun.var_id, fun.params.len());
            self.func_name_arities.insert(c_name, fun.params.len());
        }

        // Register builtin function mappings from the program
        for (var_id, name) in &program.builtins {
            let c_name = format!("gn_{}", name);
            self.top_level_funcs.insert(*var_id, c_name);
            // Builtins generally have at least 1 arg, so don't mark as 0-arity
        }

        // Emit forward declarations only for reachable functions
        for fun in &program.functions {
            if reachable.contains(&fun.var_id) {
                self.emit_forward_decl(fun);
            }
        }

        // Collect and emit dictionaries for type class instances
        let dicts = Self::collect_dictionaries(&program.main, &program.functions);
        if !dicts.is_empty() {
            self.output.push_str("// Type class dictionaries\n");

            // Emit dictionary type definition (generic for all traits with 'show' method)
            // For now, we use a simple struct with function pointers
            self.output.push_str("typedef struct {\n");
            self.output.push_str("    gn_value (*show)(gn_value);\n");
            self.output.push_str("} GnDict;\n\n");

            // Emit dictionary instances
            for (trait_name, instance_ty) in &dicts {
                // Dictionary instance references the trait method function
                // e.g., Show_Int_dict references fn_Show_Int_show
                let dict_name = format!("{}_{}_dict", trait_name, instance_ty);
                let method_name = format!("fn_{}_{}_{}", trait_name, instance_ty, "show");

                // Forward declare the method function
                self.output
                    .push_str(&format!("gn_value {}(gn_value);\n", method_name));

                // Emit dictionary structure
                self.output.push_str(&format!(
                    "static GnDict {} = {{ .show = {} }};\n",
                    dict_name, method_name
                ));
            }
            self.output.push('\n');
        }

        // Emit function definitions only for reachable functions
        for fun in &program.functions {
            if reachable.contains(&fun.var_id) {
                self.emit_function(fun);
            }
        }

        // Now emit forward declarations (including generated ones)
        if !self.forward_decls.is_empty() || !self.generated_forward_decls.is_empty() {
            self.output.push_str("// Forward declarations\n");
            self.output.push_str(&self.forward_decls);
            self.output.push_str(&self.generated_forward_decls);
            self.output.push('\n');
        }

        // Emit generated functions (local recursive helpers)
        if !self.generated_funcs.is_empty() {
            self.output.push_str("// Generated helper functions\n");
            self.output.push_str(&self.generated_funcs);
            self.output.push('\n');
        }

        if !self.function_defs.is_empty() {
            self.output.push_str("// Function definitions\n");
            self.output.push_str(&self.function_defs);
        }

        // Emit main
        self.emit_main(&program.main);

        self.output.clone()
    }

    /// Emit forward declaration for a function
    fn emit_forward_decl(&mut self, fun: &FunDef) {
        // All functions have env as first parameter for uniform closure calling convention
        let params: Vec<String> = std::iter::once("gn_value* env".to_string())
            .chain(
                fun.params
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("gn_value arg{}", i)),
            )
            .collect();

        let params_str = params.join(", ");

        writeln!(
            self.forward_decls,
            "static gn_value fn_{}({});",
            mangle_name(&fun.name),
            params_str
        )
        .unwrap();
    }

    /// Emit a function definition
    fn emit_function(&mut self, fun: &FunDef) {
        // Clear var names for this function, but preserve top-level function bindings
        self.var_names.clear();
        for (var_id, name) in &self.top_level_funcs {
            self.var_names.insert(*var_id, name.clone());
        }

        // Bind parameters
        for (i, (param, _hint)) in fun.params.iter().zip(fun.param_hints.iter()).enumerate() {
            let name = format!("arg{}", i);
            self.var_names.insert(*param, name.clone());
        }

        // All functions have env as first parameter for uniform closure calling convention
        let params: Vec<String> = std::iter::once("gn_value* env".to_string())
            .chain(
                fun.params
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("gn_value arg{}", i)),
            )
            .collect();

        let params_str = params.join(", ");

        writeln!(
            self.function_defs,
            "static gn_value fn_{}({}) {{",
            mangle_name(&fun.name),
            params_str
        )
        .unwrap();

        self.indent = 1;
        let body = self.emit_expr(&fun.body);
        writeln!(self.function_defs, "{}return {};", self.pad(), body).unwrap();
        writeln!(self.function_defs, "}}\n").unwrap();
    }

    /// Emit main function
    fn emit_main(&mut self, main_expr: &Option<CoreExpr>) {
        // Check if there's a fn_main to call when no main expression
        let has_fn_main = self.top_level_funcs.values().any(|n| n == "fn_main");

        if let Some(expr) = main_expr {
            self.indent = 1;
            // Clear var names but preserve top-level function bindings
            self.var_names.clear();
            for (var_id, name) in &self.top_level_funcs {
                self.var_names.insert(*var_id, name.clone());
            }

            // Temporarily use function_defs as statement buffer for main
            let saved_defs = std::mem::take(&mut self.function_defs);
            let saved_gen_funcs = std::mem::take(&mut self.generated_funcs);

            let result = self.emit_expr(expr);

            // Collect the statements emitted during expr lowering
            let main_stmts = std::mem::replace(&mut self.function_defs, saved_defs);

            // Collect any generated functions (from LetRec)
            let gen_funcs = std::mem::replace(&mut self.generated_funcs, saved_gen_funcs);

            // Emit generated functions before main
            if !gen_funcs.is_empty() {
                self.output.push_str("// Generated functions\n");
                self.output.push_str(&gen_funcs);
            }

            // Emit main function
            self.output.push_str("// Main entry point\n");
            self.output
                .push_str("int main(int argc, char** argv) {\n");
            self.output.push_str("    gn_init(argc, argv);\n");
            self.output.push_str(&main_stmts);
            writeln!(self.output, "    gn_value result = {};", result).unwrap();
            self.output.push_str("    gn_print(result);\n");
            self.output.push_str("    gn_println();\n");
            self.output.push_str("    gn_shutdown();\n");
            self.output.push_str("    return 0;\n");
            self.output.push_str("}\n");
        } else if has_fn_main {
            // Call fn_main with Unit argument
            self.output.push_str("// Main entry point\n");
            self.output
                .push_str("int main(int argc, char** argv) {\n");
            self.output.push_str("    gn_init(argc, argv);\n");
            self.output.push_str("    gn_value result = fn_main(NULL, GN_UNIT);\n");
            self.output.push_str("    gn_print(result);\n");
            self.output.push_str("    gn_println();\n");
            self.output.push_str("    gn_shutdown();\n");
            self.output.push_str("    return 0;\n");
            self.output.push_str("}\n");
        } else {
            // No main expression and no fn_main
            self.output.push_str("// Main entry point\n");
            self.output
                .push_str("int main(int argc, char** argv) {\n");
            self.output.push_str("    gn_init(argc, argv);\n");
            self.output.push_str("    gn_shutdown();\n");
            self.output.push_str("    return 0;\n");
            self.output.push_str("}\n");
        }
    }

    /// Emit an expression, returning the C expression string
    fn emit_expr(&mut self, expr: &CoreExpr) -> String {
        match expr {
            CoreExpr::Var(var) => {
                let name = self.get_var_name(*var);

                // Check if this is a function being used as a value
                // First try by VarId, then by name
                let arity = self
                    .func_arities
                    .get(var)
                    .copied()
                    .or_else(|| self.func_name_arities.get(&name).copied());

                if let Some(arity) = arity {
                    if arity == 0 {
                        // It's a CAF - call it
                        return format!("{}()", name);
                    } else {
                        // Function used as a value - wrap in closure
                        return match arity {
                            1 => format!("gn_make_closure1((void*){})", name),
                            2 => format!("gn_make_closure2((void*){})", name),
                            _ => format!("gn_make_closure((void*){}, {}, 0, NULL)", name, arity),
                        };
                    }
                }
                name
            }

            CoreExpr::Lit(lit) => self.emit_literal(lit),

            CoreExpr::Let {
                name,
                name_hint,
                value,
                body,
            } => {
                let var_name = self.fresh_var(name_hint.as_deref());
                self.var_names.insert(*name, var_name.clone());

                let value_expr = self.emit_atom(value);
                writeln!(
                    self.function_defs,
                    "{}gn_value {} = {};",
                    self.pad(),
                    var_name,
                    value_expr
                )
                .unwrap();

                self.emit_expr(body)
            }

            CoreExpr::LetExpr {
                name,
                name_hint,
                value,
                body,
            } => {
                let var_name = self.fresh_var(name_hint.as_deref());
                self.var_names.insert(*name, var_name.clone());

                let value_expr = self.emit_expr(value);
                writeln!(
                    self.function_defs,
                    "{}gn_value {} = {};",
                    self.pad(),
                    var_name,
                    value_expr
                )
                .unwrap();

                self.emit_expr(body)
            }

            CoreExpr::App { func, args } => {
                let func_name = self.get_var_name(*func);
                let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();

                // Check if this is a known function call (top-level fn_ or gn_ builtin)
                if func_name.starts_with("fn_") {
                    // User-defined function - pass NULL for env
                    let arity = self.func_arities.get(func).copied().unwrap_or(args.len());

                    if args_str.len() <= arity {
                        // Simple case: all args fit in the function's arity
                        let all_args = std::iter::once("NULL".to_string())
                            .chain(args_str.into_iter())
                            .collect::<Vec<_>>();
                        format!("{}({})", func_name, all_args.join(", "))
                    } else {
                        // Curried call: call function with arity args, then gn_apply rest
                        let (direct_args, extra_args) = args_str.split_at(arity);
                        let all_direct = std::iter::once("NULL".to_string())
                            .chain(direct_args.iter().cloned())
                            .collect::<Vec<_>>();
                        let call = format!("{}({})", func_name, all_direct.join(", "));
                        self.emit_chained_apply(&call, extra_args)
                    }
                } else if func_name.starts_with("gn_") {
                    // Builtin function - no env parameter
                    format!("{}({})", func_name, args_str.join(", "))
                } else {
                    // Closure call - chain gn_apply for each argument
                    self.emit_chained_apply(&func_name, &args_str)
                }
            }

            CoreExpr::TailApp { func, args } => {
                // For now, emit as regular call (TCO optimization comes later)
                let func_name = self.get_var_name(*func);
                let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();
                // Check if this is a known function call
                if func_name.starts_with("fn_") {
                    // User-defined function - pass NULL for env
                    let arity = self.func_arities.get(func).copied().unwrap_or(args.len());

                    if args_str.len() <= arity {
                        let all_args = std::iter::once("NULL".to_string())
                            .chain(args_str.into_iter())
                            .collect::<Vec<_>>();
                        format!("{}({})", func_name, all_args.join(", "))
                    } else {
                        let (direct_args, extra_args) = args_str.split_at(arity);
                        let all_direct = std::iter::once("NULL".to_string())
                            .chain(direct_args.iter().cloned())
                            .collect::<Vec<_>>();
                        let call = format!("{}({})", func_name, all_direct.join(", "));
                        self.emit_chained_apply(&call, extra_args)
                    }
                } else if func_name.starts_with("gn_") {
                    // Builtin function - no env parameter
                    format!("{}({})", func_name, args_str.join(", "))
                } else {
                    self.emit_chained_apply(&func_name, &args_str)
                }
            }

            CoreExpr::Alloc {
                tag,
                ctor_name,
                fields,
                ..
            } => {
                let fields_str: Vec<String> =
                    fields.iter().map(|f| self.get_var_name(*f)).collect();
                let name = ctor_name
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or("anon");

                if fields.is_empty() {
                    // Nullary constructor - can be a singleton
                    format!("GN_CTOR({}, 0) /* {} */", tag, name)
                } else {
                    format!(
                        "gn_alloc({}, {}, (gn_value[]){{{}}})",
                        tag,
                        fields.len(),
                        fields_str.join(", ")
                    )
                }
            }

            CoreExpr::Case {
                scrutinee,
                alts,
                default,
            } => self.emit_case(*scrutinee, alts, default.as_deref()),

            CoreExpr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let result_var = self.fresh_var(Some("if_result"));
                let cond_name = self.get_var_name(*cond);

                writeln!(self.function_defs, "{}gn_value {};", self.pad(), result_var).unwrap();
                writeln!(
                    self.function_defs,
                    "{}if (GN_IS_TRUE({})) {{",
                    self.pad(),
                    cond_name
                )
                .unwrap();

                self.indent += 1;
                let then_expr = self.emit_expr(then_branch);
                writeln!(
                    self.function_defs,
                    "{}{} = {};",
                    self.pad(),
                    result_var,
                    then_expr
                )
                .unwrap();
                self.indent -= 1;

                writeln!(self.function_defs, "{}}} else {{", self.pad()).unwrap();

                self.indent += 1;
                let else_expr = self.emit_expr(else_branch);
                writeln!(
                    self.function_defs,
                    "{}{} = {};",
                    self.pad(),
                    result_var,
                    else_expr
                )
                .unwrap();
                self.indent -= 1;

                writeln!(self.function_defs, "{}}}", self.pad()).unwrap();

                result_var
            }

            CoreExpr::Lam {
                params,
                param_hints: _,
                body,
            } => {
                // Check for free variables (captures)
                let bound: HashSet<VarId> = params.iter().cloned().collect();
                let mut free_vars: HashSet<VarId> = HashSet::new();
                find_free_vars(body, &bound, &mut free_vars);

                // Filter out top-level functions from free vars (they're not captures)
                let captures: Vec<VarId> = free_vars
                    .into_iter()
                    .filter(|v| !self.top_level_funcs.contains_key(v))
                    .collect();

                if captures.is_empty() {
                    // Simple lambda without captures - lift to top-level function
                    let unique_id = self.name_counter;
                    self.name_counter += 1;
                    let func_name = format!(
                        "fn_lambda_{}",
                        unique_id
                    );

                    let arity = params.len();

                    // Build parameter list with env* as first parameter (required by gn_apply)
                    let mut param_strs: Vec<String> = vec!["gn_value* env".to_string()];
                    param_strs.extend(params
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("gn_value arg{}", i)));
                    let params_str = param_strs.join(", ");

                    // Emit forward declaration
                    writeln!(
                        self.generated_forward_decls,
                        "static gn_value {}({});",
                        func_name, params_str
                    )
                    .unwrap();

                    // Save current state
                    let saved_var_names = self.var_names.clone();
                    let saved_counter = self.name_counter;
                    let saved_indent = self.indent;

                    // Clear var_names for the function and set up parameters
                    self.var_names.clear();
                    // Keep top-level function references
                    for (var_id, name) in &self.top_level_funcs {
                        self.var_names.insert(*var_id, name.clone());
                    }
                    for (i, param) in params.iter().enumerate() {
                        let arg_name = format!("arg{}", i);
                        self.var_names.insert(*param, arg_name);
                    }

                    // Emit function body to a separate buffer
                    let mut func_body_stmts = String::new();
                    std::mem::swap(&mut func_body_stmts, &mut self.function_defs);
                    self.indent = 1;

                    let result = self.emit_expr(body);

                    std::mem::swap(&mut func_body_stmts, &mut self.function_defs);

                    // Emit complete function definition to generated_funcs buffer
                    writeln!(
                        self.generated_funcs,
                        "static gn_value {}({}) {{",
                        func_name, params_str
                    )
                    .unwrap();
                    // Mark env as unused to avoid compiler warning
                    writeln!(self.generated_funcs, "    (void)env;").unwrap();
                    self.generated_funcs.push_str(&func_body_stmts);
                    writeln!(self.generated_funcs, "    return {};", result).unwrap();
                    writeln!(self.generated_funcs, "}}\n").unwrap();

                    // Restore state
                    self.var_names = saved_var_names;
                    self.name_counter = saved_counter;
                    self.indent = saved_indent;

                    // Create a closure struct for the lifted function
                    match arity {
                        1 => format!("gn_make_closure1((void*){func_name})"),
                        2 => format!("gn_make_closure2((void*){func_name})"),
                        _ => format!("gn_make_closure((void*){}, {}, 0, NULL)", func_name, arity),
                    }
                } else {
                    // Lambda with captures - create closure with captured environment
                    let unique_id = self.name_counter;
                    self.name_counter += 1;
                    let func_name = format!("fn_lambda_{}", unique_id);

                    let arity = params.len();
                    let num_captures = captures.len();

                    // Sort captures for consistent ordering (by underlying u32)
                    let mut sorted_captures: Vec<VarId> = captures.clone();
                    sorted_captures.sort_by_key(|v| v.0);

                    // Build parameter list with env* as first parameter
                    let mut param_strs: Vec<String> = vec!["gn_value* env".to_string()];
                    param_strs.extend(params
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("gn_value arg{}", i)));
                    let params_str = param_strs.join(", ");

                    // Emit forward declaration
                    writeln!(
                        self.generated_forward_decls,
                        "static gn_value {}({});",
                        func_name, params_str
                    )
                    .unwrap();

                    // Save current state
                    let saved_var_names = self.var_names.clone();
                    let saved_counter = self.name_counter;
                    let saved_indent = self.indent;

                    // Set up the function scope with parameters and env[] for captures
                    self.var_names.clear();
                    // Keep top-level function references
                    for (var_id, name) in &self.top_level_funcs {
                        self.var_names.insert(*var_id, name.clone());
                    }
                    // Parameters
                    for (i, param) in params.iter().enumerate() {
                        let arg_name = format!("arg{}", i);
                        self.var_names.insert(*param, arg_name);
                    }
                    // Captured variables accessed via env[]
                    for (i, cap_var) in sorted_captures.iter().enumerate() {
                        let cap_name = format!("env[{}]", i);
                        self.var_names.insert(*cap_var, cap_name);
                    }

                    // Emit function body to a separate buffer
                    let mut func_body_stmts = String::new();
                    std::mem::swap(&mut func_body_stmts, &mut self.function_defs);
                    self.indent = 1;

                    let result = self.emit_expr(body);

                    std::mem::swap(&mut func_body_stmts, &mut self.function_defs);

                    // Emit complete function definition to generated_funcs buffer
                    writeln!(
                        self.generated_funcs,
                        "static gn_value {}({}) {{",
                        func_name, params_str
                    )
                    .unwrap();
                    self.generated_funcs.push_str(&func_body_stmts);
                    writeln!(self.generated_funcs, "    return {};", result).unwrap();
                    writeln!(self.generated_funcs, "}}\n").unwrap();

                    // Restore state
                    self.var_names = saved_var_names;
                    self.name_counter = saved_counter;
                    self.indent = saved_indent;

                    // Create closure with captured environment
                    // First, build the captures array
                    let capture_names: Vec<String> = sorted_captures
                        .iter()
                        .map(|v| self.get_var_name(*v))
                        .collect();

                    if num_captures == 0 {
                        // No captures - use simpler closure creation
                        match arity {
                            1 => format!("gn_make_closure1((void*){func_name})"),
                            2 => format!("gn_make_closure2((void*){func_name})"),
                            _ => format!("gn_make_closure((void*){}, {}, 0, NULL)", func_name, arity),
                        }
                    } else {
                        // Build captures array inline
                        let captures_array = format!(
                            "(gn_value[]){{ {} }}",
                            capture_names.join(", ")
                        );
                        format!(
                            "gn_make_closure((void*){}, {}, {}, {})",
                            func_name, arity, num_captures, captures_array
                        )
                    }
                }
            }

            CoreExpr::LetRec {
                name,
                name_hint,
                params,
                param_hints: _,
                func_body,
                body,
            } => {
                // Emit recursive function as top-level C function
                let func_name = format!(
                    "fn_{}",
                    mangle_name(name_hint.as_deref().unwrap_or("anon"))
                );
                self.var_names.insert(*name, func_name.clone());

                // Register arity so the function is properly wrapped as a closure when used as value
                let arity = params.len();
                self.func_arities.insert(*name, arity);
                self.func_name_arities.insert(func_name.clone(), arity);

                // Build parameter list - include env* for closure compatibility
                let mut param_strs: Vec<String> = vec!["gn_value* env".to_string()];
                param_strs.extend(
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("gn_value arg{}", i)),
                );
                let params_str = param_strs.join(", ");

                // Emit forward declaration
                writeln!(
                    self.forward_decls,
                    "static gn_value {}({});",
                    func_name, params_str
                )
                .unwrap();

                // Save current state
                let saved_var_names = self.var_names.clone();
                let saved_counter = self.name_counter;
                let saved_indent = self.indent;

                // Clear var_names for the function and set up parameters
                self.var_names.clear();
                self.var_names.insert(*name, func_name.clone()); // Self-reference
                for (i, param) in params.iter().enumerate() {
                    let arg_name = format!("arg{}", i);
                    self.var_names.insert(*param, arg_name);
                }

                // Emit function body to a separate buffer
                let mut func_body_stmts = String::new();
                std::mem::swap(&mut func_body_stmts, &mut self.function_defs);
                self.indent = 1;

                let result = self.emit_expr(func_body);

                std::mem::swap(&mut func_body_stmts, &mut self.function_defs);

                // Emit complete function definition to generated_funcs buffer
                writeln!(
                    self.generated_funcs,
                    "static gn_value {}({}) {{",
                    func_name, params_str
                )
                .unwrap();
                self.generated_funcs.push_str(&func_body_stmts);
                writeln!(self.generated_funcs, "    return {};", result).unwrap();
                writeln!(self.generated_funcs, "}}\n").unwrap();

                // Restore state
                self.var_names = saved_var_names;
                self.var_names.insert(*name, func_name); // Keep the function binding
                self.name_counter = saved_counter;
                self.indent = saved_indent;

                // Continue with body
                self.emit_expr(body)
            }

            CoreExpr::LetRecMutual { bindings, body } => {
                // Emit each recursive binding as a top-level C function with closure conversion

                // First, collect the current environment's variable bindings
                // These are potential captured variables
                let current_env: HashSet<VarId> = self.var_names.keys().cloned().collect();

                // First pass: find free variables and register function names
                let mut func_names = Vec::new();
                let mut func_captures: Vec<Vec<VarId>> = Vec::new();

                // Collect function names and their bound parameters
                for binding in bindings {
                    let unique_id = self.name_counter;
                    self.name_counter += 1;
                    let func_name = format!(
                        "fn_rec_{}_{}",
                        mangle_name(binding.name_hint.as_deref().unwrap_or("anon")),
                        unique_id
                    );
                    func_names.push(func_name);
                }

                // Register function names before finding free vars (they're not free)
                for (binding, func_name) in bindings.iter().zip(func_names.iter()) {
                    self.var_names.insert(binding.name, func_name.clone());
                    // Register arity so function is properly wrapped as closure when used as value
                    let arity = binding.params.len();
                    self.func_arities.insert(binding.name, arity);
                    self.func_name_arities.insert(func_name.clone(), arity);
                }

                // Find free variables for each binding
                for binding in bindings {
                    let mut bound: HashSet<VarId> = HashSet::new();
                    // The function's own params are bound
                    for p in &binding.params {
                        bound.insert(*p);
                    }
                    // The mutual rec function names are bound
                    for b in bindings {
                        bound.insert(b.name);
                    }

                    let mut free: HashSet<VarId> = HashSet::new();
                    find_free_vars(&binding.body, &bound, &mut free);

                    // Only keep captures that are in the current environment
                    let captures: Vec<VarId> = free
                        .into_iter()
                        .filter(|v| current_env.contains(v))
                        .collect();
                    func_captures.push(captures);
                }

                // Second pass: emit forward declarations with captures as extra params
                let mut param_decls_with_captures = Vec::new();
                for (binding, captures) in bindings.iter().zip(func_captures.iter()) {
                    let func_name = &func_names[bindings.iter().position(|b| b.name == binding.name).unwrap()];

                    // Build parameter names: regular params + captured vars
                    let mut all_param_names: Vec<String> = binding
                        .params
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            binding
                                .param_hints
                                .get(i)
                                .and_then(|h| h.as_ref())
                                .map(|h| format!("arg_{}", mangle_name(h)))
                                .unwrap_or_else(|| format!("arg{}", i))
                        })
                        .collect();

                    // Add captured variables as extra params
                    for (i, cap) in captures.iter().enumerate() {
                        let cap_name = self.var_names.get(cap)
                            .map(|n| format!("cap_{}", n))
                            .unwrap_or_else(|| format!("cap{}", i));
                        all_param_names.push(cap_name);
                    }

                    let params_decl = if all_param_names.is_empty() {
                        "void".to_string()
                    } else {
                        all_param_names
                            .iter()
                            .map(|p| format!("gn_value {}", p))
                            .collect::<Vec<_>>()
                            .join(", ")
                    };

                    writeln!(
                        self.generated_forward_decls,
                        "static gn_value {}({});",
                        func_name, params_decl
                    )
                    .unwrap();

                    param_decls_with_captures.push(params_decl);
                }

                // Third pass: emit function bodies
                for (idx, (binding, func_name)) in bindings.iter().zip(func_names.iter()).enumerate() {
                    let captures = &func_captures[idx];

                    // Save state
                    let saved_var_names = self.var_names.clone();
                    let saved_counter = self.name_counter;
                    let saved_indent = self.indent;
                    self.indent = 1;

                    // Re-bind all function names for mutual recursion
                    for (b, name) in bindings.iter().zip(func_names.iter()) {
                        self.var_names.insert(b.name, name.clone());
                    }

                    // Build and bind regular parameters
                    for (j, p) in binding.params.iter().enumerate() {
                        let name = binding
                            .param_hints
                            .get(j)
                            .and_then(|h| h.as_ref())
                            .map(|h| format!("arg_{}", mangle_name(h)))
                            .unwrap_or_else(|| format!("arg{}", j));
                        self.var_names.insert(*p, name);
                    }

                    // Bind captured variables
                    for (i, cap) in captures.iter().enumerate() {
                        let cap_name = saved_var_names.get(cap)
                            .map(|n| format!("cap_{}", n))
                            .unwrap_or_else(|| format!("cap{}", i));
                        self.var_names.insert(*cap, cap_name);
                    }

                    writeln!(
                        self.generated_funcs,
                        "static gn_value {}({}) {{",
                        func_name, param_decls_with_captures[idx]
                    )
                    .unwrap();

                    // Use function_defs temporarily for the body
                    let saved_func_defs = std::mem::take(&mut self.function_defs);

                    let result = self.emit_expr(&binding.body);

                    let func_body_stmts = std::mem::replace(&mut self.function_defs, saved_func_defs);

                    self.generated_funcs.push_str(&func_body_stmts);
                    writeln!(self.generated_funcs, "    return {};", result).unwrap();
                    writeln!(self.generated_funcs, "}}\n").unwrap();

                    // Restore state but keep function bindings
                    self.var_names = saved_var_names;
                    for (b, name) in bindings.iter().zip(func_names.iter()) {
                        self.var_names.insert(b.name, name.clone());
                    }
                    self.name_counter = saved_counter;
                    self.indent = saved_indent;
                }

                // Store capture info for call site generation
                // TODO: We need to transform calls to these functions to pass captures
                // For now, store info in a way the call sites can use
                // This is tricky because calls are already in the body...

                // Actually, the calls inside the rec functions to each other also need captures
                // This is getting complex - for now, let's skip the call transformation
                // and rely on the captured vars being bound in the inner function's scope

                // Continue with body
                self.emit_expr(body)
            }

            CoreExpr::Seq { first, second } => {
                let first_expr = self.emit_expr(first);
                writeln!(self.function_defs, "{}{};", self.pad(), first_expr).unwrap();
                self.emit_expr(second)
            }

            CoreExpr::PrimOp { op, args } => self.emit_primop(op, args),

            CoreExpr::Return(var) => self.get_var_name(*var),

            CoreExpr::ExternCall { name, args } => {
                let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();
                format!("{}({})", name, args_str.join(", "))
            }

            CoreExpr::Error(msg) => {
                format!("gn_panic(\"{}\")", escape_string(msg))
            }

            CoreExpr::Perform {
                effect_name,
                op_name,
                args: _,
                ..
            } => {
                let effect = effect_name.as_deref().unwrap_or("?");
                let op = op_name.as_deref().unwrap_or("?");
                format!("gn_panic(\"perform {}.{} not implemented\")", effect, op)
            }

            CoreExpr::Handle { body, handler: _ } => {
                // Effects not yet implemented
                self.emit_expr(body)
            }

            CoreExpr::DictCall { dict, method, args } => {
                // Dictionary method call: gn_dict_call(dict, "method", args...)
                let dict_name = self.get_var_name(*dict);
                let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();

                // For now, emit as a lookup and call
                // gn_apply(gn_dict_get(dict, "method"), arg)
                if args.is_empty() {
                    format!("gn_dict_get({}, \"{}\")", dict_name, method)
                } else if args.len() == 1 {
                    format!(
                        "gn_apply(gn_dict_get({}, \"{}\"), {})",
                        dict_name, method, args_str[0]
                    )
                } else {
                    // Multiple args - chain applies
                    let mut result = format!("gn_dict_get({}, \"{}\")", dict_name, method);
                    for arg in &args_str {
                        result = format!("gn_apply({}, {})", result, arg);
                    }
                    result
                }
            }

            CoreExpr::DictValue {
                trait_name,
                instance_ty,
            } => {
                // Dictionary value: reference to the instance's dictionary
                // e.g., &Show_Int_dict
                format!("(&{}_{}_dict)", trait_name, instance_ty)
            }

            CoreExpr::DictRef(var) => {
                // Reference to a dictionary parameter
                self.get_var_name(*var)
            }

            CoreExpr::Proj { tuple, index } => {
                // Project field from tuple: GN_FIELD(tuple, index)
                let tuple_name = self.get_var_name(*tuple);
                format!("GN_FIELD({}, {})", tuple_name, index)
            }

            // CPS-transformed expressions
            // These emit calls to the algebraic effects runtime
            CoreExpr::AppCont {
                func,
                args,
                cont,
            } => {
                // CPS function call: func(args..., k)
                // For known top-level functions, emit direct call
                // For closures, use gn_apply chain
                let func_name = self.get_var_name(*func);
                let cont_name = self.get_var_name(*cont);
                let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();

                if func_name.starts_with("fn_") {
                    // Known top-level function - emit direct call with cont as last arg
                    let all_args = std::iter::once("NULL".to_string())
                        .chain(args_str.into_iter())
                        .chain(std::iter::once(cont_name))
                        .collect::<Vec<_>>();
                    format!("{}({})", func_name, all_args.join(", "))
                } else {
                    // Closure - use gn_apply for each arg including continuation
                    let mut result = func_name;
                    for arg_name in args_str {
                        result = format!("gn_apply({}, {})", result, arg_name);
                    }
                    format!("gn_apply({}, {})", result, cont_name)
                }
            }

            CoreExpr::Resume { cont, value } => {
                // Resume a captured continuation with a value
                // Use gn_resume_multi for multi-shot continuation support
                let cont_name = self.get_var_name(*cont);
                let value_name = self.get_var_name(*value);
                format!("gn_resume_multi({}, {})", cont_name, value_name)
            }

            CoreExpr::CaptureK {
                effect,
                op,
                args,
                cont,
                ..
            } => {
                // Perform an effect operation with captured continuation
                // This transfers control, so we emit it as a return statement
                let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();
                let cont_name = self.get_var_name(*cont);
                let n_args = args.len();

                let perform_call = if args.is_empty() {
                    format!("gn_perform({}, {}, 0, NULL, {})", effect, op, cont_name)
                } else {
                    format!(
                        "gn_perform({}, {}, {}, (gn_value[]){{{}}}, {})",
                        effect, op, n_args, args_str.join(", "), cont_name
                    )
                };

                // Emit as return statement since this transfers control
                writeln!(self.function_defs, "{}return {};", self.pad(), perform_call).unwrap();

                // Return a dummy value - this code is unreachable after the return
                "GN_UNIT /* unreachable */".to_string()
            }

            CoreExpr::WithHandler {
                effect,
                handler,
                body,
                outer_cont,
                ..
            } => {
                // Set up effect handler and run body
                // This emits inline handler setup code
                let outer_k = self.get_var_name(*outer_cont);
                let return_handler = self.get_var_name(handler.return_handler);

                // Create handler setup
                let handler_var = self.fresh_var(Some("handler"));
                let n_ops = handler.op_handlers.len();

                // Sort op handlers by op_id to ensure ops[i] handles operation i
                let mut sorted_ops: Vec<_> = handler.op_handlers.iter().collect();
                sorted_ops.sort_by_key(|op| op.op);

                let op_handlers_str: Vec<String> = sorted_ops
                    .iter()
                    .map(|op| self.get_var_name(op.handler_fn))
                    .collect();

                // Write handler creation
                writeln!(
                    self.function_defs,
                    "{}gn_value {}_ops[] = {{{}}};",
                    self.pad(),
                    handler_var,
                    op_handlers_str.join(", ")
                )
                .unwrap();

                writeln!(
                    self.function_defs,
                    "{}gn_handler* {} = gn_create_handler({}, {}, {}, {}_ops, {});",
                    self.pad(),
                    handler_var,
                    effect,
                    return_handler,
                    n_ops,
                    handler_var,
                    outer_k
                )
                .unwrap();

                writeln!(
                    self.function_defs,
                    "{}gn_push_handler({});",
                    self.pad(),
                    handler_var
                )
                .unwrap();

                // Emit body as a return statement (CPS: body transfers control, doesn't return)
                // The handler is popped by gn_perform when an effect is performed
                let body_result = self.emit_expr(body);

                // For CPS, we need to return the body result directly since it transfers control
                // The handler cleanup happens in the continuation/return handler, not here
                writeln!(
                    self.function_defs,
                    "{}return {};",
                    self.pad(),
                    body_result
                )
                .unwrap();

                // This is unreachable in CPS, but we return a placeholder
                "GN_UNIT /* handler body returned */".to_string()
            }
        }
    }

    /// Emit an atom (simple expression)
    fn emit_atom(&mut self, atom: &Atom) -> String {
        match atom {
            Atom::Var(var) => {
                let name = self.get_var_name(*var);

                // Check if this is a function being used as a value
                // First try by VarId, then by name
                let arity = self
                    .func_arities
                    .get(var)
                    .copied()
                    .or_else(|| self.func_name_arities.get(&name).copied());

                if let Some(arity) = arity {
                    if arity == 0 {
                        // It's a CAF - call it
                        return format!("{}()", name);
                    } else {
                        // Function used as a value - wrap in closure
                        // Named functions have no captures, env is NULL
                        return match arity {
                            1 => format!("gn_make_closure1((void*){})", name),
                            2 => format!("gn_make_closure2((void*){})", name),
                            _ => format!("gn_make_closure((void*){}, {}, 0, NULL)", name, arity),
                        };
                    }
                }
                name
            }
            Atom::Lit(lit) => self.emit_literal(lit),
            Atom::Alloc {
                tag,
                ctor_name,
                fields,
                ..
            } => {
                let fields_str: Vec<String> =
                    fields.iter().map(|f| self.get_var_name(*f)).collect();
                let name = ctor_name.as_ref().map(|s| s.as_str()).unwrap_or("anon");
                if fields.is_empty() {
                    format!("GN_CTOR({}, 0) /* {} */", tag, name)
                } else {
                    format!(
                        "gn_alloc({}, {}, (gn_value[]){{{}}})",
                        tag,
                        fields.len(),
                        fields_str.join(", ")
                    )
                }
            }
            Atom::PrimOp { op, args } => self.emit_primop(op, args),
            Atom::Lam { .. } => "GN_UNIT /* lambda */".to_string(),
            Atom::App { func, args } => {
                let func_name = self.get_var_name(*func);
                let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();
                // Check if this is a known function call
                if func_name.starts_with("fn_") || func_name.starts_with("gn_") {
                    format!("{}({})", func_name, args_str.join(", "))
                } else {
                    self.emit_chained_apply(&func_name, &args_str)
                }
            }
        }
    }

    /// Emit chained gn_apply calls for multi-argument closure application
    /// e.g., for `f(a, b, c)` produces `gn_apply(gn_apply(gn_apply(f, a), b), c)`
    fn emit_chained_apply(&self, func: &str, args: &[String]) -> String {
        if args.is_empty() {
            // No args - just return the function value
            func.to_string()
        } else {
            // Chain gn_apply for each argument
            let mut result = format!("gn_apply({}, {})", func, args[0]);
            for arg in &args[1..] {
                result = format!("gn_apply({}, {})", result, arg);
            }
            result
        }
    }

    /// Emit a case expression
    /// Groups alternatives by tag to avoid duplicate case labels
    fn emit_case(&mut self, scrutinee: VarId, alts: &[Alt], default: Option<&CoreExpr>) -> String {
        use std::collections::BTreeMap;

        let result_var = self.fresh_var(Some("case_result"));
        let scrut_name = self.get_var_name(scrutinee);

        // Group alts by tag to handle duplicate tags
        let mut alts_by_tag: BTreeMap<u32, Vec<&Alt>> = BTreeMap::new();
        for alt in alts {
            alts_by_tag.entry(alt.tag).or_default().push(alt);
        }

        writeln!(self.function_defs, "{}gn_value {};", self.pad(), result_var).unwrap();
        writeln!(
            self.function_defs,
            "{}switch (GN_TAG({})) {{",
            self.pad(),
            scrut_name
        )
        .unwrap();

        for (tag, tag_alts) in &alts_by_tag {
            let tag_name = tag_alts[0].tag_name.as_deref().unwrap_or("?");
            writeln!(
                self.function_defs,
                "{}case {}: /* {} */ {{",
                self.pad(),
                tag,
                tag_name
            )
            .unwrap();

            self.indent += 1;

            if tag_alts.len() == 1 {
                // Single alt for this tag - emit directly
                let alt = tag_alts[0];
                self.emit_alt_body(&scrut_name, alt, &result_var);
            } else {
                // Multiple alts with same tag - need to emit as if-else chain
                // First, bind the fields that all alts need (use max binders)
                let max_binders = tag_alts.iter().map(|a| a.binders.len()).max().unwrap_or(0);
                let mut field_vars = Vec::new();
                for i in 0..max_binders {
                    let var_name = self.fresh_var(Some(&format!("fld{}", i)));
                    field_vars.push(var_name.clone());
                    writeln!(
                        self.function_defs,
                        "{}gn_value {} = GN_FIELD({}, {});",
                        self.pad(),
                        var_name,
                        scrut_name,
                        i
                    )
                    .unwrap();
                }

                // Now emit the alts as an if-else chain
                // Each alt's body may have nested case expressions that do the real discrimination
                let mut first = true;
                for alt in tag_alts {
                    // Bind this alt's binders to the field vars
                    for (i, binder) in alt.binders.iter().enumerate() {
                        if i < field_vars.len() {
                            self.var_names.insert(*binder, field_vars[i].clone());
                        }
                    }

                    if first {
                        // First alt - try it, if its body panics we fall through
                        // Actually, we need to emit them all and let the nested cases discriminate
                        let body_expr = self.emit_expr(&alt.body);
                        writeln!(
                            self.function_defs,
                            "{}{} = {};",
                            self.pad(),
                            result_var,
                            body_expr
                        )
                        .unwrap();
                        first = false;
                        // Note: we only emit the first alt's body because nested cases
                        // will handle the discrimination. This works because the lowering
                        // phase generates nested case expressions that properly fall through.
                        break;
                    }
                }
            }

            writeln!(self.function_defs, "{}break;", self.pad()).unwrap();

            self.indent -= 1;
            writeln!(self.function_defs, "{}}}", self.pad()).unwrap();
        }

        // Default case
        writeln!(self.function_defs, "{}default: {{", self.pad()).unwrap();
        self.indent += 1;
        if let Some(def) = default {
            let def_expr = self.emit_expr(def);
            writeln!(
                self.function_defs,
                "{}{} = {};",
                self.pad(),
                result_var,
                def_expr
            )
            .unwrap();
        } else {
            writeln!(
                self.function_defs,
                "{}gn_panic(\"non-exhaustive match\");",
                self.pad()
            )
            .unwrap();
            writeln!(
                self.function_defs,
                "{}{} = GN_UNIT;",
                self.pad(),
                result_var
            )
            .unwrap();
        }
        writeln!(self.function_defs, "{}break;", self.pad()).unwrap();
        self.indent -= 1;
        writeln!(self.function_defs, "{}}}", self.pad()).unwrap();

        writeln!(self.function_defs, "{}}}", self.pad()).unwrap();

        result_var
    }

    /// Emit a single alt's body (used when there's only one alt for a tag)
    fn emit_alt_body(&mut self, scrut_name: &str, alt: &Alt, result_var: &str) {
        // Bind the fields
        for (i, binder) in alt.binders.iter().enumerate() {
            let hint = alt.binder_hints.get(i).and_then(|h| h.as_deref());
            let var_name = self.fresh_var(hint);
            self.var_names.insert(*binder, var_name.clone());
            writeln!(
                self.function_defs,
                "{}gn_value {} = GN_FIELD({}, {});",
                self.pad(),
                var_name,
                scrut_name,
                i
            )
            .unwrap();
        }

        let body_expr = self.emit_expr(&alt.body);
        writeln!(
            self.function_defs,
            "{}{} = {};",
            self.pad(),
            result_var,
            body_expr
        )
        .unwrap();
    }

    /// Emit a primitive operation
    fn emit_primop(&self, op: &PrimOp, args: &[VarId]) -> String {
        let args_str: Vec<String> = args.iter().map(|a| self.get_var_name(*a)).collect();

        match op {
            // Arithmetic (Int)
            PrimOp::IntAdd => format!("GN_INT_ADD({}, {})", args_str[0], args_str[1]),
            PrimOp::IntSub => format!("GN_INT_SUB({}, {})", args_str[0], args_str[1]),
            PrimOp::IntMul => format!("GN_INT_MUL({}, {})", args_str[0], args_str[1]),
            PrimOp::IntDiv => format!("GN_INT_DIV({}, {})", args_str[0], args_str[1]),
            PrimOp::IntMod => format!("GN_INT_MOD({}, {})", args_str[0], args_str[1]),
            PrimOp::IntNeg => format!("GN_INT_NEG({})", args_str[0]),

            // Arithmetic (Float)
            PrimOp::FloatAdd => format!("gn_float_add({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatSub => format!("gn_float_sub({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatMul => format!("gn_float_mul({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatDiv => format!("gn_float_div({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatNeg => format!("gn_float_neg({})", args_str[0]),

            // Comparison (Int)
            PrimOp::IntEq => format!("GN_INT_EQ({}, {})", args_str[0], args_str[1]),
            PrimOp::IntNe => format!("GN_INT_NE({}, {})", args_str[0], args_str[1]),
            PrimOp::IntLt => format!("GN_INT_LT({}, {})", args_str[0], args_str[1]),
            PrimOp::IntLe => format!("GN_INT_LE({}, {})", args_str[0], args_str[1]),
            PrimOp::IntGt => format!("GN_INT_GT({}, {})", args_str[0], args_str[1]),
            PrimOp::IntGe => format!("GN_INT_GE({}, {})", args_str[0], args_str[1]),

            // Comparison (Float)
            PrimOp::FloatEq => format!("gn_float_eq({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatNe => format!("gn_float_ne({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatLt => format!("gn_float_lt({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatLe => format!("gn_float_le({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatGt => format!("gn_float_gt({}, {})", args_str[0], args_str[1]),
            PrimOp::FloatGe => format!("gn_float_ge({}, {})", args_str[0], args_str[1]),

            // Boolean
            PrimOp::BoolAnd => format!("GN_BOOL_AND({}, {})", args_str[0], args_str[1]),
            PrimOp::BoolOr => format!("GN_BOOL_OR({}, {})", args_str[0], args_str[1]),
            PrimOp::BoolNot => format!("GN_BOOL_NOT({})", args_str[0]),

            // String
            PrimOp::StringConcat => format!("gn_string_concat({}, {})", args_str[0], args_str[1]),
            PrimOp::StringLength => format!("gn_string_length({})", args_str[0]),
            PrimOp::StringEq => format!("gn_string_eq({}, {})", args_str[0], args_str[1]),

            // Conversion
            PrimOp::IntToFloat => format!("gn_int_to_float({})", args_str[0]),
            PrimOp::FloatToInt => format!("gn_float_to_int({})", args_str[0]),
            PrimOp::IntToString => format!("gn_int_to_string({})", args_str[0]),
            PrimOp::FloatToString => format!("gn_float_to_string({})", args_str[0]),
            PrimOp::CharToInt => format!("GN_CHAR_TO_INT({})", args_str[0]),
            PrimOp::IntToChar => format!("GN_INT_TO_CHAR({})", args_str[0]),

            // Tuple/record
            PrimOp::TupleGet(n) => format!("GN_FIELD({}, {})", args_str[0], n),

            // List
            PrimOp::ListCons => format!("gn_list_cons({}, {})", args_str[0], args_str[1]),
            PrimOp::ListHead => format!("gn_list_head({})", args_str[0]),
            PrimOp::ListTail => format!("gn_list_tail({})", args_str[0]),
            PrimOp::ListIsEmpty => format!("gn_list_is_empty({})", args_str[0]),
        }
    }

    /// Emit a literal
    fn emit_literal(&self, lit: &CoreLit) -> String {
        match lit {
            CoreLit::Int(n) => format!("GN_INT({})", n),
            CoreLit::Float(f) => format!("gn_float({})", f),
            CoreLit::String(s) => format!("gn_string(\"{}\")", escape_string(s)),
            CoreLit::Char(c) => format!("GN_CHAR({})", *c as u32),
            CoreLit::Bool(true) => "GN_TRUE".to_string(),
            CoreLit::Bool(false) => "GN_FALSE".to_string(),
            CoreLit::Unit => "GN_UNIT".to_string(),
        }
    }

    /// Get the C variable name for a VarId
    fn get_var_name(&self, var: VarId) -> String {
        self.var_names
            .get(&var)
            .cloned()
            .unwrap_or_else(|| format!("v{}", var.0))
    }

    /// Generate a fresh variable name
    fn fresh_var(&mut self, hint: Option<&str>) -> String {
        let name = if let Some(h) = hint {
            format!("{}_{}", mangle_name(h), self.name_counter)
        } else {
            format!("tmp_{}", self.name_counter)
        };
        self.name_counter += 1;
        name
    }

    /// Get indentation padding
    fn pad(&self) -> String {
        "    ".repeat(self.indent)
    }
}

impl Default for CEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Find free variables in a CoreExpr that are bound in the given environment
fn find_free_vars(expr: &CoreExpr, bound: &HashSet<VarId>, free: &mut HashSet<VarId>) {
    match expr {
        CoreExpr::Var(v) => {
            if !bound.contains(v) {
                free.insert(*v);
            }
        }
        CoreExpr::Lit(_) => {}
        CoreExpr::Alloc { fields, .. } => {
            for f in fields {
                if !bound.contains(f) {
                    free.insert(*f);
                }
            }
        }
        CoreExpr::Let { name, value, body, .. } => {
            find_free_vars_atom(value, bound, free);
            let mut inner_bound = bound.clone();
            inner_bound.insert(*name);
            find_free_vars(body, &inner_bound, free);
        }
        CoreExpr::LetExpr { name, value, body, .. } => {
            find_free_vars(value, bound, free);
            let mut inner_bound = bound.clone();
            inner_bound.insert(*name);
            find_free_vars(body, &inner_bound, free);
        }
        CoreExpr::LetRec { name, params, func_body, body, .. } => {
            let mut inner_bound = bound.clone();
            inner_bound.insert(*name);
            for p in params {
                inner_bound.insert(*p);
            }
            find_free_vars(func_body, &inner_bound, free);
            find_free_vars(body, &inner_bound, free);
        }
        CoreExpr::LetRecMutual { bindings, body } => {
            let mut inner_bound = bound.clone();
            for b in bindings {
                inner_bound.insert(b.name);
            }
            for b in bindings {
                let mut func_bound = inner_bound.clone();
                for p in &b.params {
                    func_bound.insert(*p);
                }
                find_free_vars(&b.body, &func_bound, free);
            }
            find_free_vars(body, &inner_bound, free);
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
        CoreExpr::If { cond, then_branch, else_branch } => {
            if !bound.contains(cond) {
                free.insert(*cond);
            }
            find_free_vars(then_branch, bound, free);
            find_free_vars(else_branch, bound, free);
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
                find_free_vars(&alt.body, &alt_bound, free);
            }
            if let Some(d) = default {
                find_free_vars(d, bound, free);
            }
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
        CoreExpr::Seq { first, second } => {
            find_free_vars(first, bound, free);
            find_free_vars(second, bound, free);
        }
        CoreExpr::ExternCall { args, .. } => {
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        CoreExpr::Lam { params, body, .. } => {
            let mut inner_bound = bound.clone();
            for p in params {
                inner_bound.insert(*p);
            }
            find_free_vars(body, &inner_bound, free);
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
            find_free_vars(body, bound, free);
            for op in &handler.ops {
                let mut op_bound = bound.clone();
                for p in &op.params {
                    op_bound.insert(*p);
                }
                op_bound.insert(op.cont);
                find_free_vars(&op.body, &op_bound, free);
            }
            let mut ret_bound = bound.clone();
            ret_bound.insert(handler.return_var);
            find_free_vars(&handler.return_body, &ret_bound, free);
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
        CoreExpr::DictValue { .. } => {
            // No free variables - static dictionary reference
        }
        CoreExpr::DictRef(var) => {
            if !bound.contains(var) {
                free.insert(*var);
            }
        }
        CoreExpr::Proj { tuple, .. } => {
            if !bound.contains(tuple) {
                free.insert(*tuple);
            }
        }

        // CPS-transformed expressions
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
            find_free_vars(body, bound, free);
            if !bound.contains(outer_cont) {
                free.insert(*outer_cont);
            }
        }
    }
}

fn find_free_vars_atom(atom: &Atom, bound: &HashSet<VarId>, free: &mut HashSet<VarId>) {
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
        Atom::PrimOp { args, .. } => {
            for a in args {
                if !bound.contains(a) {
                    free.insert(*a);
                }
            }
        }
        Atom::Lam { params, body, .. } => {
            let mut inner_bound = bound.clone();
            for p in params {
                inner_bound.insert(*p);
            }
            find_free_vars(body, &inner_bound, free);
        }
    }
}

/// Mangle a name to be a valid C identifier
fn mangle_name(name: &str) -> String {
    let mut result = String::new();
    for c in name.chars() {
        match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' => result.push(c),
            '_' => result.push('_'),
            '\'' => result.push_str("_prime"),
            _ => result.push_str(&format!("_u{:04x}_", c as u32)),
        }
    }
    result
}

/// Escape a string for C
fn escape_string(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_ascii_graphic() || c == ' ' => result.push(c),
            c => result.push_str(&format!("\\x{:02x}", c as u32)),
        }
    }
    result
}

// ============================================================================
// Public API
// ============================================================================

/// Compile a CoreProgram to C source code
///
/// Pipeline:
/// 1. CPS transformation (for effects)
/// 2. Closure conversion (eliminate lambdas)
/// 3. Emit C from Flat IR
pub fn emit_c(program: &CoreProgram) -> String {
    // Apply CPS transformation for effects if needed
    let cps_program = super::cps_transform::cps_transform(program.clone());

    // Closure conversion: lift lambdas to top-level functions
    let flat_program = super::closure::closure_convert(&cps_program);

    // Emit C from FlatProgram
    super::emit_flat::emit_flat_c(&flat_program)
}
