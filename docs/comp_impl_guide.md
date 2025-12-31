# Gneiss Compiler: Implementation Guide

## Executive Summary

After analyzing the codebase, **the project is salvageable but needs architectural corrections**. The core issue is that the compiler pipeline has a disconnect: type inference produces rich typed information (TAST), but the lowering phase largely ignores it and works from the untyped AST.

**Verdict: Refactor, don't rewrite.**

The interpreter is excellent and serves as a semantic reference. The type system is sound. The problem is in the middle of the pipeline.

---

## Current State Analysis

### What Works Well ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer/Parser | ✅ Solid | Complete coverage of syntax |
| AST | ✅ Good | Well-designed with spans, comprehensive |
| Type Inference | ✅ Good | Union-find, effect rows, polymorphism |
| TAST Design | ✅ Good | Has `MethodCall`, `DictMethodCall`, `DictValue` |
| Interpreter | ✅ Excellent | Defunctionalized CPS, full effect handling |
| Runtime Header | ✅ Basic | Value representation defined |

### What's Broken or Missing ❌

| Component | Status | Issue |
|-----------|--------|-------|
| Elaboration | ⚠️ Partial | Produces TAST but trait resolution incomplete |
| Lower | ❌ Wrong input | Works from AST, not TAST! Types are lost |
| Monomorphization | ❌ Missing | No pass to instantiate polymorphic code |
| Closure Conversion | ❌ Missing | CoreIR has `Lam`, C emitter hacks around it |
| CPS/Effect Lowering | ❌ Missing | No evidence passing or continuation representation |
| Perceus | ❌ Missing | No ownership analysis or RC insertion |

### The Core Architectural Bug

```
CURRENT (Broken):
┌─────────┐    ┌─────────┐    ┌─────────┐
│   AST   │───▶│  Infer  │───▶│  TAST   │ (types computed, then ignored!)
└─────────┘    └─────────┘    └─────────┘
     │
     │ (Lower uses AST directly!)
     ▼
┌─────────┐    ┌─────────┐
│ CoreIR  │───▶│  C Emit │
└─────────┘    └─────────┘

CORRECT:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│   AST   │───▶│  Infer  │───▶│  TAST   │───▶│  Mono   │───▶│ MonoIR  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                                  │
     ┌────────────────────────────────────────────────────────────┘
     ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Lower  │───▶│ Closure │───▶│ Perceus │───▶│  C Emit │
│ (ANF)   │    │  Conv   │    │   RC    │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

---

## Corrected Pipeline Design

### Phase 1: Frontend (Existing, Keep)
```
Source → Lexer → Parser → AST
```
No changes needed.

### Phase 2: Type Inference (Existing, Keep)
```
AST → Inferencer → TypedAST + TypeEnv
```
The inferencer is solid. Keep it.

### Phase 3: Elaboration (Needs Work)
```
AST + Types → Elaborator → TAST
```

**Current issue:** Elaboration exists but doesn't fully resolve traits.

**Fix:** Ensure every `Var` that refers to a trait method becomes either:
- `MethodCall` (when type is concrete)
- `DictMethodCall` (when type is polymorphic)

```rust
// In elaborate.rs, when encountering a Var:
fn elaborate_var(&self, name: &str, ty: &Type, span: &Span) -> TExpr {
    // Check if this is a trait method
    if let Some(trait_name) = self.lookup_trait_method(name) {
        let resolved_ty = self.resolve_type(ty);
        
        if self.is_concrete(&resolved_ty) {
            // Concrete type → direct method call
            TExprKind::MethodCall {
                trait_name: trait_name.to_string(),
                method: name.to_string(),
                instance_ty: resolved_ty,
                args: vec![], // Args added when App is elaborated
            }
        } else if let Type::Var(id) = &resolved_ty {
            // Polymorphic → dictionary lookup
            TExprKind::DictMethodCall {
                trait_name: trait_name.to_string(),
                method: name.to_string(),
                type_var: *id,
                args: vec![],
            }
        } else {
            // Fallback to regular var
            TExprKind::Var(name.to_string())
        }
    } else {
        TExprKind::Var(name.to_string())
    }
}
```

### Phase 4: Monomorphization (NEW - Must Add)

**Purpose:** Eliminate all polymorphism. Every function becomes specialized to concrete types.

```rust
// src/mono.rs (new file)

/// Monomorphized program - no type variables remain
pub struct MonoProgram {
    pub functions: HashMap<MonoFnId, MonoFn>,
    pub types: Vec<MonoTypeDef>,
    pub main: Option<MonoFnId>,
}

/// A function ID includes its type instantiation
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct MonoFnId {
    pub base_name: String,
    pub type_args: Vec<MonoType>,
}

/// Monomorphized type (no variables)
pub enum MonoType {
    Int,
    Float,
    Bool,
    String,
    Unit,
    Tuple(Vec<MonoType>),
    Constructor { name: String, args: Vec<MonoType> },
    Function { args: Vec<MonoType>, ret: Box<MonoType> },
}

/// Monomorphization context
pub struct MonoCtx {
    /// Work queue of functions to monomorphize
    work_queue: VecDeque<(String, Vec<MonoType>)>,
    /// Already processed functions
    done: HashMap<MonoFnId, MonoFn>,
    /// Type substitution for current function
    subst: HashMap<TypeVarId, MonoType>,
}

pub fn monomorphize(tprogram: &TProgram) -> MonoProgram {
    let mut ctx = MonoCtx::new();
    
    // Start with main
    if let Some(main) = &tprogram.main {
        ctx.work_queue.push_back(("main".to_string(), vec![]));
    }
    
    // Process work queue
    while let Some((name, type_args)) = ctx.work_queue.pop_front() {
        let fn_id = MonoFnId { base_name: name.clone(), type_args: type_args.clone() };
        if ctx.done.contains_key(&fn_id) {
            continue;
        }
        
        let binding = tprogram.find_binding(&name)
            .expect(&format!("Unknown function: {}", name));
        
        // Set up type substitution
        ctx.subst = build_substitution(&binding.ty, &type_args);
        
        // Monomorphize the function body
        let mono_body = ctx.mono_expr(&binding.body);
        
        ctx.done.insert(fn_id, MonoFn {
            params: mono_params(&binding.params, &ctx.subst),
            body: mono_body,
        });
    }
    
    MonoProgram {
        functions: ctx.done,
        // ...
    }
}

impl MonoCtx {
    fn mono_expr(&mut self, expr: &TExpr) -> MonoExpr {
        match &expr.node {
            TExprKind::MethodCall { trait_name, method, instance_ty, args } => {
                // Substitute type variables in instance_ty
                let concrete_ty = self.substitute_type(instance_ty);
                
                // Generate mangled name: show_Int, show_List_Int, etc.
                let mangled = mangle_method(trait_name, method, &concrete_ty);
                
                // Queue the instance method for monomorphization if needed
                self.ensure_instance(trait_name, &concrete_ty);
                
                MonoExpr::Call {
                    func: mangled,
                    args: args.iter().map(|a| self.mono_expr(a)).collect(),
                }
            }
            
            TExprKind::App { func, arg } => {
                let mono_func = self.mono_expr(func);
                let mono_arg = self.mono_expr(arg);
                
                // If calling a polymorphic function, instantiate it
                if let MonoExpr::Var(name) = &mono_func {
                    let arg_ty = self.get_mono_type(&arg.ty);
                    self.work_queue.push_back((name.clone(), vec![arg_ty]));
                }
                
                MonoExpr::App {
                    func: Box::new(mono_func),
                    arg: Box::new(mono_arg),
                }
            }
            
            // ... other cases
        }
    }
}
```

### Phase 5: ANF/Core Lowering (Exists, Needs Rewrite)

**Current issue:** `lower.rs` takes AST, should take MonoProgram.

```rust
// src/codegen/lower.rs - REWRITE to use MonoProgram

pub fn lower_mono(program: &MonoProgram) -> CoreProgram {
    let mut ctx = LowerCtx::new();
    
    for (fn_id, mono_fn) in &program.functions {
        let core_fn = lower_function(&mut ctx, fn_id, mono_fn);
        ctx.functions.push(core_fn);
    }
    
    CoreProgram {
        functions: ctx.functions,
        main: program.main.as_ref().map(|id| lower_main(&ctx, id)),
        // ...
    }
}

fn lower_function(ctx: &mut LowerCtx, id: &MonoFnId, f: &MonoFn) -> FunDef {
    // All types are now concrete - no polymorphism to worry about
    ctx.push_scope();
    
    let param_vars: Vec<VarId> = f.params.iter()
        .map(|p| {
            let var = ctx.fresh();
            ctx.bind(&p.name, var);
            var
        })
        .collect();
    
    let body = lower_expr(ctx, &f.body, true); // true = tail position
    
    ctx.pop_scope();
    
    FunDef {
        name: mangle_fn_id(id),
        params: param_vars,
        body,
        // ...
    }
}

fn lower_expr(ctx: &mut LowerCtx, expr: &MonoExpr, in_tail: bool) -> CoreExpr {
    match expr {
        // ANF: bind intermediate values
        MonoExpr::App { func, arg } => {
            let func_var = ctx.fresh();
            let arg_var = ctx.fresh();
            
            let func_core = lower_expr(ctx, func, false);
            let arg_core = lower_expr(ctx, arg, false);
            
            CoreExpr::Let {
                name: func_var,
                value: Box::new(func_core),
                body: Box::new(CoreExpr::Let {
                    name: arg_var,
                    value: Box::new(arg_core),
                    body: Box::new(if in_tail {
                        CoreExpr::TailApp { func: func_var, args: vec![arg_var] }
                    } else {
                        CoreExpr::App { func: func_var, args: vec![arg_var] }
                    }),
                }),
            }
        }
        // ...
    }
}
```

### Phase 6: Closure Conversion (NEW - Must Add)

**Purpose:** Eliminate nested lambdas. Every function becomes top-level with explicit environment.

```rust
// src/codegen/closure_convert.rs (new file)

/// After closure conversion, all functions are top-level
pub struct FlatProgram {
    pub functions: Vec<FlatFn>,
    pub env_structs: Vec<EnvStruct>,
}

pub struct FlatFn {
    pub name: String,
    pub env_param: Option<EnvStructId>,  // None if no captures
    pub params: Vec<VarId>,
    pub body: FlatExpr,
}

pub struct EnvStruct {
    pub id: EnvStructId,
    pub fields: Vec<(String, FlatType)>,
}

/// Expressions after closure conversion
pub enum FlatExpr {
    Var(VarId),
    Lit(CoreLit),
    
    // No more Lam! Instead:
    MakeClosure {
        func: String,           // Top-level function name
        captures: Vec<VarId>,   // Values to capture
    },
    
    CallClosure {
        closure: VarId,
        args: Vec<VarId>,
    },
    
    CallDirect {  // For known functions with no closure
        func: String,
        args: Vec<VarId>,
    },
    
    // ... Let, Case, If, etc. stay similar
}

pub fn closure_convert(program: &CoreProgram) -> FlatProgram {
    let mut ctx = ClosureCtx::new();
    
    for fun in &program.functions {
        convert_function(&mut ctx, fun);
    }
    
    if let Some(main) = &program.main {
        let main_body = convert_expr(&mut ctx, main, &HashSet::new());
        ctx.add_main(main_body);
    }
    
    ctx.into_program()
}

fn convert_expr(
    ctx: &mut ClosureCtx, 
    expr: &CoreExpr, 
    in_scope: &HashSet<VarId>
) -> FlatExpr {
    match expr {
        CoreExpr::Lam { params, body, .. } => {
            // Find free variables
            let body_vars = free_vars(body);
            let captured: Vec<VarId> = body_vars
                .difference(&params.iter().copied().collect())
                .filter(|v| in_scope.contains(v))
                .copied()
                .collect();
            
            // Create environment struct if needed
            let env_struct = if captured.is_empty() {
                None
            } else {
                Some(ctx.create_env_struct(&captured))
            };
            
            // Lift lambda to top-level
            let lifted_name = ctx.fresh_fn_name();
            let mut new_scope = in_scope.clone();
            new_scope.extend(params.iter().copied());
            
            let lifted_body = convert_expr(ctx, body, &new_scope);
            
            ctx.add_function(FlatFn {
                name: lifted_name.clone(),
                env_param: env_struct,
                params: params.clone(),
                body: lifted_body,
            });
            
            // Replace lambda with MakeClosure
            FlatExpr::MakeClosure {
                func: lifted_name,
                captures: captured,
            }
        }
        
        CoreExpr::App { func, args } => {
            // Determine if this is a known function or closure call
            if ctx.is_known_function(*func) {
                FlatExpr::CallDirect {
                    func: ctx.get_function_name(*func),
                    args: args.clone(),
                }
            } else {
                FlatExpr::CallClosure {
                    closure: *func,
                    args: args.clone(),
                }
            }
        }
        
        // ... other cases recursively convert
    }
}

fn free_vars(expr: &CoreExpr) -> HashSet<VarId> {
    let mut vars = HashSet::new();
    collect_free_vars(expr, &mut HashSet::new(), &mut vars);
    vars
}

fn collect_free_vars(
    expr: &CoreExpr, 
    bound: &mut HashSet<VarId>, 
    free: &mut HashSet<VarId>
) {
    match expr {
        CoreExpr::Var(v) => {
            if !bound.contains(v) {
                free.insert(*v);
            }
        }
        CoreExpr::Let { name, value, body, .. } => {
            collect_free_vars_atom(value, bound, free);
            bound.insert(*name);
            collect_free_vars(body, bound, free);
            bound.remove(name);
        }
        CoreExpr::Lam { params, body, .. } => {
            for p in params {
                bound.insert(*p);
            }
            collect_free_vars(body, bound, free);
            for p in params {
                bound.remove(p);
            }
        }
        // ... etc
    }
}
```

### Phase 7: Effect Lowering (NEW - Critical for Effects)

For algebraic effects, you have two main strategies:

#### Option A: Evidence Passing (Koka-style, Recommended)

Effects become extra parameters threaded through:

```rust
// src/codegen/effect_lower.rs

/// Lower effects to evidence passing
pub fn lower_effects(program: &FlatProgram) -> EvidenceProgram {
    let mut ctx = EffectCtx::new();
    
    for func in &program.functions {
        let lowered = lower_function_effects(&mut ctx, func);
        ctx.add_function(lowered);
    }
    
    ctx.into_program()
}

/// Effect evidence parameter
pub struct Evidence {
    pub effect_name: String,
    pub handlers: HashMap<String, VarId>,  // op_name -> handler function
}

fn lower_function_effects(ctx: &mut EffectCtx, func: &FlatFn) -> EvidenceFn {
    // Analyze which effects this function uses
    let effects_used = collect_effects(&func.body);
    
    // Add evidence parameters for each effect
    let evidence_params: Vec<EvidenceParam> = effects_used.iter()
        .map(|eff| EvidenceParam {
            effect: eff.clone(),
            var: ctx.fresh(),
        })
        .collect();
    
    let body = lower_expr_effects(ctx, &func.body, &evidence_params);
    
    EvidenceFn {
        name: func.name.clone(),
        evidence_params,
        regular_params: func.params.clone(),
        body,
    }
}

fn lower_expr_effects(
    ctx: &mut EffectCtx, 
    expr: &FlatExpr,
    evidence: &[EvidenceParam]
) -> EvidenceExpr {
    match expr {
        // perform Effect.op args  →  evidence.handlers["op"](args, k)
        FlatExpr::Perform { effect, op, args } => {
            let ev = evidence.iter()
                .find(|e| e.effect == *effect)
                .expect("Effect not in scope");
            
            EvidenceExpr::InvokeHandler {
                evidence: ev.var,
                op: op.clone(),
                args: args.clone(),
            }
        }
        
        // handle body with handlers  →  create evidence, call body with it
        FlatExpr::Handle { body, handlers, effect } => {
            let ev_var = ctx.fresh();
            
            // Create handler functions for each operation
            let handler_fns: HashMap<String, VarId> = handlers.iter()
                .map(|h| {
                    let fn_var = ctx.fresh();
                    ctx.emit_handler_fn(fn_var, h);
                    (h.op_name.clone(), fn_var)
                })
                .collect();
            
            // Create evidence struct
            let create_ev = EvidenceExpr::CreateEvidence {
                effect: effect.clone(),
                handlers: handler_fns,
                result: ev_var,
            };
            
            // Add evidence to scope for body
            let mut new_evidence = evidence.to_vec();
            new_evidence.push(EvidenceParam {
                effect: effect.clone(),
                var: ev_var,
            });
            
            let body_lowered = lower_expr_effects(ctx, body, &new_evidence);
            
            EvidenceExpr::Seq(Box::new(create_ev), Box::new(body_lowered))
        }
        
        // ... other cases thread evidence through
    }
}
```

#### Option B: CPS + Delimited Continuations

Transform everything to CPS, making continuations explicit:

```rust
// This is closer to your interpreter's approach
pub enum CpsExpr {
    LetCont {
        name: ContId,
        param: VarId,
        body: Box<CpsExpr>,
        rest: Box<CpsExpr>,
    },
    AppCont {
        cont: ContId,
        arg: VarId,
    },
    AppFn {
        func: VarId,
        args: Vec<VarId>,
        ret_cont: ContId,
    },
    // Effects use the continuation
    Perform {
        effect: String,
        op: String,
        args: Vec<VarId>,
        ret_cont: ContId,  // The delimited continuation!
    },
    // ...
}
```

**Recommendation:** Start with evidence passing. It's simpler and matches Koka's approach.

### Phase 8: Perceus RC Insertion (Exists as Stub, Needs Implementation)

```rust
// src/codegen/perceus.rs

pub fn insert_rc(program: &EvidenceProgram) -> RcProgram {
    let mut ctx = PerceusCtx::new();
    
    for func in &program.functions {
        let rc_body = analyze_and_insert(&mut ctx, &func.body, Ownership::Owned);
        ctx.add_function(RcFn {
            name: func.name.clone(),
            params: func.params.clone(),
            body: rc_body,
        });
    }
    
    ctx.into_program()
}

#[derive(Clone, Copy)]
enum Ownership {
    Owned,      // We own it, must drop or transfer
    Borrowed,   // Someone else owns it, must dup if keeping
}

fn analyze_and_insert(
    ctx: &mut PerceusCtx,
    expr: &EvidenceExpr,
    ownership: Ownership,
) -> RcExpr {
    match expr {
        EvidenceExpr::Let { name, value, body } => {
            // Determine last use of `name` in body
            let uses = count_uses(*name, body);
            
            let value_rc = analyze_and_insert(ctx, value, Ownership::Owned);
            
            // Insert drops for variables that go out of scope
            let body_rc = analyze_and_insert(ctx, body, ownership);
            
            // If name is never used, drop immediately
            if uses == 0 {
                RcExpr::Let {
                    name: *name,
                    value: Box::new(value_rc),
                    body: Box::new(RcExpr::Drop {
                        var: *name,
                        body: Box::new(body_rc),
                    }),
                }
            } else {
                RcExpr::Let {
                    name: *name,
                    value: Box::new(value_rc),
                    body: Box::new(body_rc),
                }
            }
        }
        
        EvidenceExpr::Var(v) => {
            match ownership {
                Ownership::Borrowed => {
                    // Need to dup if we're borrowing
                    RcExpr::Dup { var: *v, body: Box::new(RcExpr::Var(*v)) }
                }
                Ownership::Owned => RcExpr::Var(*v),
            }
        }
        
        // Pattern matching enables REUSE
        EvidenceExpr::Case { scrutinee, alts, default } => {
            // If we own scrutinee and destructure it, we can reuse its memory
            // ...
        }
        
        // ...
    }
}
```

### Phase 9: C Code Generation (Exists, Needs Updates)

The C emitter needs to handle:
1. Closure structs (from closure conversion)
2. Evidence structs (from effect lowering)  
3. RC operations (from Perceus)

```c
// Generated code structure:

// ==================== TYPES ====================

// Environment struct for closure
typedef struct env_f_123 {
    gn_value captured_x;
    gn_value captured_y;
} env_f_123;

// Closure representation
typedef struct gn_closure {
    uint32_t rc;
    uint32_t arity;
    gn_value (*code)(struct gn_closure*, gn_value*, int);
    // Followed by environment
} gn_closure;

// Evidence for State effect
typedef struct ev_State {
    gn_value (*get)(struct ev_State*, gn_value k);
    gn_value (*put)(struct ev_State*, gn_value v, gn_value k);
} ev_State;

// ==================== FUNCTIONS ====================

// Top-level function (no closure needed)
static gn_value fn_add(gn_value a, gn_value b) {
    return GN_INT_ADD(a, b);
}

// Lifted lambda with environment
static gn_value fn_lifted_123(gn_closure* self, gn_value* args, int nargs) {
    env_f_123* env = (env_f_123*)(self + 1);
    gn_value x = env->captured_x;
    gn_value arg = args[0];
    return GN_INT_ADD(x, arg);
}

// Function using effects
static gn_value fn_stateful(ev_State* ev, gn_value x) {
    gn_value current = ev->get(ev, /* continuation */);
    gn_value new_val = GN_INT_ADD(current, x);
    ev->put(ev, new_val, /* continuation */);
    return new_val;
}

// ==================== MAIN ====================

int main(int argc, char** argv) {
    gn_init(argc, argv);
    
    gn_value result = fn_main();
    gn_print(result);
    
    gn_shutdown();
    return 0;
}
```

---

## Corrected File Structure

```
src/
├── lib.rs
├── main.rs
│
├── frontend/           # KEEP AS-IS
│   ├── lexer.rs
│   ├── parser/
│   └── ast.rs
│
├── typing/             # KEEP AS-IS  
│   ├── types.rs
│   ├── infer.rs
│   └── tast.rs
│
├── middle/             # NEW ORGANIZATION
│   ├── elaborate.rs    # AST + Types → TAST (fix trait resolution)
│   ├── mono.rs         # NEW: TAST → MonoProgram
│   └── mono_ir.rs      # NEW: Monomorphized IR types
│
├── codegen/            # REFACTOR
│   ├── core_ir.rs      # Keep (minor updates)
│   ├── lower.rs        # REWRITE: MonoProgram → CoreIR
│   ├── closure.rs      # NEW: Closure conversion
│   ├── effects.rs      # NEW: Effect lowering
│   ├── perceus.rs      # NEW: RC insertion
│   └── c_emit.rs       # Update for new IR
│
├── runtime/            # KEEP AS-IS
│   ├── gn_runtime.h
│   └── gn_runtime.c
│
└── eval.rs             # KEEP: Reference interpreter
```

---

## Implementation Plan

### Week 1-2: Fix the Plumbing

**Goal:** Get types flowing through the pipeline correctly.

1. **Day 1-2:** Create `mono.rs` with basic `MonoProgram` types
2. **Day 3-4:** Implement monomorphization for simple cases (no traits)
3. **Day 5-7:** Rewrite `lower.rs` to consume `MonoProgram`
4. **Day 8-10:** Test with simple programs: `1 + 2`, `let x = 1 in x + 2`

**Milestone Test:**
```
let add x y = x + y
let main _ = add 1 2
```
Should compile to C that outputs `3`.

### Week 3-4: Closure Conversion

**Goal:** Eliminate lambdas, make closures explicit.

1. **Day 1-2:** Implement `free_vars` analysis
2. **Day 3-4:** Create `FlatProgram` types
3. **Day 5-7:** Implement closure conversion pass
4. **Day 8-10:** Update C emitter for closures

**Milestone Test:**
```
let apply f x = f x
let add1 x = x + 1
let main _ = apply add1 5
```

### Week 5-6: Trait Monomorphization

**Goal:** `show 42` becomes `show_Int(42)`.

1. **Day 1-3:** Fix elaboration to emit `MethodCall` nodes
2. **Day 4-6:** Implement trait instance collection in mono pass
3. **Day 7-10:** Generate monomorphized trait methods

**Milestone Test:**
```
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

let main _ = show 42
```

### Week 7-8: Effect Lowering

**Goal:** `perform State.get ()` works.

1. **Day 1-3:** Design evidence passing representation
2. **Day 4-6:** Implement effect analysis
3. **Day 7-10:** Transform handlers to evidence

**Milestone Test:**
```
effect State s =
    | get : () -> s
    | put : s -> ()
end

let main _ =
    handle
        let x = perform State.get () in
        perform State.put (x + 1)
    with
        | return v -> v
        | get () k -> fun s -> k s s
        | put s' k -> fun _ -> k () s'
    end 0
```

### Week 9-10: Perceus

**Goal:** Memory safe, no leaks.

1. **Day 1-4:** Implement ownership analysis
2. **Day 5-7:** Insert dup/drop
3. **Day 8-10:** Basic reuse optimization

**Milestone Test:**
```
let map f xs = 
    match xs with
    | [] -> []
    | x :: rest -> f x :: map f rest
    end

let main _ = map (fun x -> x + 1) [1, 2, 3]
```
Should not leak memory.

---

## Key Refactoring Steps

### Step 1: Make Lower Use TAST

```rust
// OLD (broken)
pub fn lower_program(program: &Program) -> Result<CoreProgram, Vec<String>>

// NEW (correct)
pub fn lower_mono(program: &MonoProgram) -> CoreProgram
```

### Step 2: Add MonoProgram Between TAST and CoreIR

```rust
// In main.rs or lib.rs
pub fn compile(source: &str) -> Result<String, CompileError> {
    let ast = parse(source)?;
    let (tast, type_env) = infer(&ast)?;
    let mono = monomorphize(&tast)?;      // NEW
    let core = lower(&mono)?;             // Changed
    let flat = closure_convert(&core)?;   // NEW
    let ev = lower_effects(&flat)?;       // NEW
    let rc = insert_rc(&ev)?;             // NEW  
    let c_code = emit_c(&rc)?;            // Changed
    Ok(c_code)
}
```

### Step 3: Fix Elaboration Trait Resolution

The TAST already has `MethodCall` and `DictMethodCall` - ensure elaboration produces them:

```rust
// In elaborate.rs
TExprKind::Var(name) if is_trait_method(name) => {
    let ty = get_type(expr);
    if is_concrete(ty) {
        TExprKind::MethodCall { ... }
    } else {
        TExprKind::DictMethodCall { ... }
    }
}
```

---

## Testing Strategy

### Unit Tests Per Phase

```rust
#[cfg(test)]
mod mono_tests {
    #[test]
    fn mono_identity() {
        // let id x = x
        // id 42
        // Should produce: id_Int(42)
    }
    
    #[test]
    fn mono_show_int() {
        // show 42
        // Should produce: show_Int(42)
    }
}

#[cfg(test)]
mod closure_tests {
    #[test]
    fn closure_captures_one() {
        // let x = 1 in (fun y -> x + y)
        // Should lift lambda and create env struct
    }
}
```

### Integration Tests

Keep the existing `codegen_minimal.rs` tests and add more as features are implemented.

---

## Appendix: Data Structure Summary

### Input to Each Phase

| Phase | Input | Output |
|-------|-------|--------|
| Parse | Source code | `AST` |
| Infer | `AST` | `TAST` + `TypeEnv` |
| Elaborate | `AST` + `TypeEnv` | `TAST` |
| Mono | `TAST` | `MonoProgram` |
| Lower | `MonoProgram` | `CoreIR` |
| Closure | `CoreIR` | `FlatIR` |
| Effects | `FlatIR` | `EvidenceIR` |
| Perceus | `EvidenceIR` | `RcIR` |
| Emit | `RcIR` | C source |

### Type Flow

```
Type (with vars)  →  MonoType (concrete)  →  CoreType  →  C type
     TAST                  Mono               Core        Emit
```

---

## Conclusion

The Gneiss compiler has solid foundations:
- Good parser and AST
- Working type inference with effects
- Excellent interpreter as semantic reference
- Runtime value representation designed

The critical fix is ensuring **types flow through to codegen**. The current disconnect where `lower.rs` uses AST instead of TAST is the root cause of many issues.

**Do not start over.** Instead:
1. Add monomorphization pass
2. Rewrite `lower.rs` to use monomorphized IR
3. Add closure conversion
4. Add effect lowering
5. Implement Perceus

Each phase can be tested incrementally. The interpreter provides a semantic oracle for correctness.
