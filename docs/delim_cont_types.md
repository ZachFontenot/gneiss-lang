# Gneiss Answer-Type Modification: Implementation Plan

## Overview

This document provides a step-by-step implementation plan for extending Gneiss's type system to support **answer-type modification** for delimited continuations. The key change is generalizing function types from `σ → τ` to `σ/α → τ/β`, enabling proper typing of `shift/reset`.

---

## Part 1: Type Representation

### 1.1 Current State (Assumed)

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Int,
    Bool,
    Unit,
    Var(TypeVarId),
    Arrow(Box<Type>, Box<Type>),  // σ → τ
    List(Box<Type>),
    // ... other types
}
```

### 1.2 Target State

```rust
pub type TypeVarId = u32;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    // Primitive types (unchanged)
    Int,
    Bool,
    Unit,
    
    // Type variable (unchanged)
    Var(TypeVarId),
    
    // CHANGED: Generalized function type with answer types
    // Represents: σ/α → τ/β
    // "Function from σ to τ that changes answer type from α to β"
    Fun {
        arg: Box<Type>,      // σ: argument type
        ret: Box<Type>,      // τ: return type  
        ans_in: Box<Type>,   // α: answer type before application
        ans_out: Box<Type>,  // β: answer type after application
    },
    
    // Polymorphic type (for let-polymorphism and polymorphic continuations)
    Forall(TypeVarId, Box<Type>),
    
    // Other types (unchanged)
    List(Box<Type>),
    Tuple(Vec<Type>),
    // ...
}
```

### 1.3 Type Constructors

```rust
impl Type {
    /// Create a fresh type variable
    pub fn fresh(ctx: &mut InferCtx) -> Type {
        Type::Var(ctx.fresh_var())
    }
    
    /// Pure function: σ → τ (answer type unchanged)
    /// Internally: σ/α → τ/α for fresh α
    pub fn pure_fun(ctx: &mut InferCtx, arg: Type, ret: Type) -> Type {
        let ans = Type::fresh(ctx);
        Type::Fun {
            arg: Box::new(arg),
            ret: Box::new(ret),
            ans_in: Box::new(ans.clone()),
            ans_out: Box::new(ans),
        }
    }
    
    /// Effectful function: σ/α → τ/β (explicit answer types)
    pub fn effectful_fun(arg: Type, ret: Type, ans_in: Type, ans_out: Type) -> Type {
        Type::Fun {
            arg: Box::new(arg),
            ret: Box::new(ret),
            ans_in: Box::new(ans_in),
            ans_out: Box::new(ans_out),
        }
    }
    
    /// Check if this is a pure function type (answer types unify)
    pub fn is_pure_fun(&self, subst: &Substitution) -> bool {
        match self {
            Type::Fun { ans_in, ans_out, .. } => {
                let in_resolved = subst.apply(ans_in);
                let out_resolved = subst.apply(ans_out);
                in_resolved == out_resolved
            }
            _ => false,
        }
    }
}
```

### 1.4 Type Schemes (for Polymorphism)

```rust
#[derive(Clone, Debug)]
pub enum TypeScheme {
    Mono(Type),
    Poly {
        vars: Vec<TypeVarId>,
        body: Type,
    },
}

impl TypeScheme {
    /// Instantiate a type scheme with fresh variables
    pub fn instantiate(&self, ctx: &mut InferCtx) -> Type {
        match self {
            TypeScheme::Mono(ty) => ty.clone(),
            TypeScheme::Poly { vars, body } => {
                let subst: HashMap<TypeVarId, Type> = vars
                    .iter()
                    .map(|&v| (v, Type::fresh(ctx)))
                    .collect();
                body.substitute(&subst)
            }
        }
    }
}
```

---

## Part 2: Inference Context

### 2.1 Core Structure

```rust
pub struct InferCtx {
    /// Type environment: variable -> type scheme
    env: HashMap<String, TypeScheme>,
    
    /// Current substitution (mutable during inference)
    subst: Substitution,
    
    /// Next fresh variable ID
    next_var: TypeVarId,
}

pub struct Substitution {
    mapping: HashMap<TypeVarId, Type>,
}

impl InferCtx {
    pub fn new() -> Self {
        InferCtx {
            env: HashMap::new(),
            subst: Substitution::new(),
            next_var: 0,
        }
    }
    
    pub fn fresh_var(&mut self) -> TypeVarId {
        let v = self.next_var;
        self.next_var += 1;
        v
    }
    
    pub fn fresh_type(&mut self) -> Type {
        Type::Var(self.fresh_var())
    }
}
```

### 2.2 Inference Result

The inference function now returns **three types**:

```rust
/// Result of type inference
/// - `ty`: the type of the expression
/// - `ans_in`: answer type before evaluation
/// - `ans_out`: answer type after evaluation
pub struct InferResult {
    pub ty: Type,
    pub ans_in: Type,
    pub ans_out: Type,
}

impl InferResult {
    /// Expression is pure if answer types are equal
    pub fn is_pure(&self, ctx: &InferCtx) -> bool {
        let in_resolved = ctx.subst.apply(&self.ans_in);
        let out_resolved = ctx.subst.apply(&self.ans_out);
        in_resolved == out_resolved
    }
}
```

---

## Part 3: Unification

### 3.1 Extended Unification

```rust
impl InferCtx {
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = self.subst.apply(t1);
        let t2 = self.subst.apply(t2);
        
        match (&t1, &t2) {
            // Identical types
            (Type::Int, Type::Int) => Ok(()),
            (Type::Bool, Type::Bool) => Ok(()),
            (Type::Unit, Type::Unit) => Ok(()),
            
            // Variable binding
            (Type::Var(v), ty) | (ty, Type::Var(v)) => {
                if let Type::Var(v2) = ty {
                    if v == v2 {
                        return Ok(()); // Same variable
                    }
                }
                if self.occurs_check(*v, ty) {
                    return Err(TypeError::InfiniteType(*v, ty.clone()));
                }
                self.subst.insert(*v, ty.clone());
                Ok(())
            }
            
            // Function types - unify ALL four components
            (
                Type::Fun { arg: a1, ret: r1, ans_in: ai1, ans_out: ao1 },
                Type::Fun { arg: a2, ret: r2, ans_in: ai2, ans_out: ao2 },
            ) => {
                self.unify(a1, a2)?;
                self.unify(r1, r2)?;
                self.unify(ai1, ai2)?;  // Answer types too!
                self.unify(ao1, ao2)?;
                Ok(())
            }
            
            // List types
            (Type::List(e1), Type::List(e2)) => self.unify(e1, e2),
            
            // Mismatch
            _ => Err(TypeError::Mismatch(t1, t2)),
        }
    }
    
    fn occurs_check(&self, var: TypeVarId, ty: &Type) -> bool {
        match ty {
            Type::Var(v) => *v == var,
            Type::Fun { arg, ret, ans_in, ans_out } => {
                self.occurs_check(var, arg)
                    || self.occurs_check(var, ret)
                    || self.occurs_check(var, ans_in)
                    || self.occurs_check(var, ans_out)
            }
            Type::List(elem) => self.occurs_check(var, elem),
            Type::Forall(_, body) => self.occurs_check(var, body),
            _ => false,
        }
    }
}
```

---

## Part 4: Answer Type Threading

Understanding answer type threading is **critical** for correct implementation. This section explains the pattern before diving into specific rules.

### 4.0 The Core Principle

Answer types flow through the **continuation** - what happens *after* an expression evaluates. 

For an expression `e` with `ans_in = α` and `ans_out = β`:
- `α` = answer type of `e`'s continuation (what type the rest of the program produces)
- `β` = answer type after `e`'s effects modify the continuation

**Critical insight**: Answer types flow through continuations, which is **backwards from evaluation order** for nested expressions!

### 4.0.1 Sequential Composition Pattern

For `e₁; e₂` (evaluate e₁ then e₂), you might expect `e₁.ans_out = e₂.ans_in`.

But let's trace through what the continuation sees:
- When evaluating `e₁`, its continuation is "evaluate e₂, then rest"
- When evaluating `e₂`, its continuation is "rest"

So `e₁.ans_in` (what e₁'s continuation produces) = `e₂.ans_out` (what e₂ leaves behind for "rest").

**Wait, that's still confusing.** Let me clarify with the actual rule interpretation:

In the Danvy-Filinski formulation, for simple sequential `e₁; e₂`:
```
e₁: ans_in = α, ans_out = β
e₂: ans_in = β, ans_out = γ
─────────────────────────────
e₁; e₂: ans_in = α, ans_out = γ

Threading: e₁.ans_out = e₂.ans_in
```

This IS left-to-right! The "output" of e₁ feeds into the "input" of e₂.

```rust
// Pattern for sequential composition
ctx.unify(&e1_result.ans_out, &e2_result.ans_in)?;
Ok(InferResult {
    ty: ...,
    ans_in: e1_result.ans_in,
    ans_out: e2_result.ans_out,
})
```

### 4.0.2 Alternative Branches Pattern

For `if cond then e₁ else e₂` (evaluate cond, then ONE branch):

```
cond: α → β
e₁:   β → γ    ← both branches start where cond ends
e₂:   β → γ    ← and must have same effect
─────────────────
if-then-else: α → γ

Threading constraints:
  - cond.ans_out = e₁.ans_in = e₂.ans_in
  - e₁.ans_out = e₂.ans_out
```

### 4.0.3 Application Pattern (COMPLEX!)

For `e₁ e₂` where `e₁ : σ/α → τ/β`, the rule is:

```
Γ; γ ⊢ e₁ : (σ/α → τ/β); δ    -- e₁ has ans_in=γ, ans_out=δ  
Γ; β ⊢ e₂ : σ; γ               -- e₂ has ans_in=β, ans_out=γ
────────────────────────────────
Γ; α ⊢ e₁ e₂ : τ; δ
```

**HERE IS THE TRICKY PART**: Notice that `e₂.ans_out = γ = e₁.ans_in`!

This seems backwards from evaluation order (e₁ is evaluated before e₂). The reason:
- e₁'s continuation INCLUDES evaluating e₂, applying, and continuing
- So e₁'s ans_in (γ) reflects what happens after all of that
- e₂'s ans_out (γ) is what e₂ leaves behind for "apply and continue"
- They're the same point in the continuation!

Flow (reading continuation right-to-left):
```
α ← β ← γ ← δ
│   │   │   └── after e₁ finishes (overall output)
│   │   └────── after e₂ finishes = before e₁ starts
│   └────────── after function body = before e₂ starts  
└────────────── start of application (overall input)
```

Threading:
```rust
ctx.unify(&arg_result.ans_out, &fun_result.ans_in)?;  // γ = γ
ctx.unify(&arg_result.ans_in, &beta)?;                 // β = β
```

### 4.0.4 Visual Summary

```
Sequential (e₁; e₂) - straightforward:
    ┌───┐   ┌───┐
α → │e₁ │ → │e₂ │ → γ
    └───┘ β └───┘
    e₁.ans_out = e₂.ans_in
    
Alternative (if c then t else f):
         ┌───────┐
    β → │ then  │ → γ
   ╱     └───────┘
┌───┐
│ c │ β
└───┘
α  ╲     ┌───────┐
    β → │ else  │ → γ
         └───────┘
    cond.ans_out = both.ans_in

Application (e₁ e₂) - continuation flows backwards!
    δ ← e₁ ← γ ← e₂ ← β ← apply ← α
    
    e₂.ans_out (γ) = e₁.ans_in (γ)
    e₂.ans_in (β) = function's latent ans_out
```

---

## Part 5: Typing Rules (Core)

### 5.1 Main Inference Function

```rust
pub fn infer(ctx: &mut InferCtx, expr: &Expr) -> Result<InferResult, TypeError> {
    match expr {
        Expr::Lit(n) => infer_lit(ctx, *n),
        Expr::Bool(b) => infer_bool(ctx, *b),
        Expr::Unit => infer_unit(ctx),
        Expr::Var(x) => infer_var(ctx, x),
        Expr::Lambda(x, body) => infer_lambda(ctx, x, body),
        Expr::App(fun, arg) => infer_app(ctx, fun, arg),
        Expr::Let(x, e1, e2) => infer_let(ctx, x, e1, e2),
        Expr::If(cond, then_br, else_br) => infer_if(ctx, cond, then_br, else_br),
        Expr::BinOp(op, l, r) => infer_binop(ctx, *op, l, r),
        Expr::Reset(body) => infer_reset(ctx, body),
        Expr::Shift { param, body } => infer_shift(ctx, param, body),
    }
}
```

### 5.2 Pure Expressions (Constants, Variables)

```rust
/// Literals are pure: Γ; α ⊢ n : Int; α
fn infer_lit(ctx: &mut InferCtx, _n: i64) -> Result<InferResult, TypeError> {
    let ans = ctx.fresh_type();
    Ok(InferResult {
        ty: Type::Int,
        ans_in: ans.clone(),
        ans_out: ans,  // Same! Pure expression.
    })
}

fn infer_bool(ctx: &mut InferCtx, _b: bool) -> Result<InferResult, TypeError> {
    let ans = ctx.fresh_type();
    Ok(InferResult {
        ty: Type::Bool,
        ans_in: ans.clone(),
        ans_out: ans,
    })
}

fn infer_unit(ctx: &mut InferCtx) -> Result<InferResult, TypeError> {
    let ans = ctx.fresh_type();
    Ok(InferResult {
        ty: Type::Unit,
        ans_in: ans.clone(),
        ans_out: ans,
    })
}

/// Variables are pure: Γ; α ⊢ x : τ; α  where (x : τ) ∈ Γ
fn infer_var(ctx: &mut InferCtx, name: &str) -> Result<InferResult, TypeError> {
    let scheme = ctx.env.get(name)
        .ok_or_else(|| TypeError::UnboundVariable(name.to_string()))?;
    let ty = scheme.instantiate(ctx);
    let ans = ctx.fresh_type();
    Ok(InferResult {
        ty,
        ans_in: ans.clone(),
        ans_out: ans,
    })
}
```

### 5.3 Lambda (Pure - It's a Value)

```rust
/// Lambda is pure:
///   Γ, x : σ; α ⊢ e : τ; β
///   ─────────────────────────
///   Γ ⊢ₚ λx.e : (σ/α → τ/β)
fn infer_lambda(
    ctx: &mut InferCtx,
    param: &str,
    body: &Expr,
) -> Result<InferResult, TypeError> {
    // Fresh type for parameter
    let param_ty = ctx.fresh_type();
    
    // Extend environment
    let old_binding = ctx.env.insert(param.to_string(), TypeScheme::Mono(param_ty.clone()));
    
    // Infer body
    let body_result = infer(ctx, body)?;
    
    // Restore environment
    match old_binding {
        Some(scheme) => ctx.env.insert(param.to_string(), scheme),
        None => ctx.env.remove(param),
    };
    
    // Build function type with answer types from body
    let fun_ty = Type::Fun {
        arg: Box::new(param_ty),
        ret: Box::new(body_result.ty),
        ans_in: Box::new(body_result.ans_in),
        ans_out: Box::new(body_result.ans_out),
    };
    
    // Lambda itself is pure (it's a value)
    let outer_ans = ctx.fresh_type();
    Ok(InferResult {
        ty: fun_ty,
        ans_in: outer_ans.clone(),
        ans_out: outer_ans,
    })
}
```

### 5.4 Application (CRITICAL - Answer Type Threading)

```rust
/// Application: e₁ e₂  where e₁ : σ/α → τ/β
///
/// Formal rule (from Danvy-Filinski/Asai-Kameyama):
///   Γ; γ ⊢ e₁ : (σ/α → τ/β); δ    -- e₁ has ans_in=γ, ans_out=δ
///   Γ; β ⊢ e₂ : σ; γ               -- e₂ has ans_in=β, ans_out=γ
///   ────────────────────────────────
///   Γ; α ⊢ e₁ e₂ : τ; δ            -- result has ans_in=α, ans_out=δ
///
/// KEY INSIGHT: Answer types flow through CONTINUATIONS, not evaluation order!
/// 
/// e₁'s continuation includes: "evaluate e₂, then apply, then rest"
/// So e₁.ans_in (γ) = e₂.ans_out (γ) -- e₁ "sees" what e₂ will leave behind
///
/// Flow diagram (continuation order, right to left):
///   α ← β ← γ ← δ
///   │   │   │   └── e₁ finishes (outer continuation)
///   │   │   └────── e₂ finishes, e₁ starts (arg_out = fun_in)
///   │   └────────── e₂ starts (function's latent effect ends)
///   └────────────── application happens (function's latent effect starts)
fn infer_app(
    ctx: &mut InferCtx,
    fun_expr: &Expr,
    arg_expr: &Expr,
) -> Result<InferResult, TypeError> {
    // Infer function expression
    // Per rule: Γ; γ ⊢ e₁ : (σ/α → τ/β); δ
    let fun_result = infer(ctx, fun_expr)?;
    
    // Infer argument expression
    // Per rule: Γ; β ⊢ e₂ : σ; γ  
    let arg_result = infer(ctx, arg_expr)?;
    
    // Fresh variables for function type components
    let param_ty = ctx.fresh_type();    // σ
    let ret_ty = ctx.fresh_type();      // τ
    let alpha = ctx.fresh_type();       // α: function's latent ans_in
    let beta = ctx.fresh_type();        // β: function's latent ans_out
    
    // Function must have type (σ/α → τ/β)
    let expected_fun = Type::Fun {
        arg: Box::new(param_ty.clone()),
        ret: Box::new(ret_ty.clone()),
        ans_in: Box::new(alpha.clone()),
        ans_out: Box::new(beta.clone()),
    };
    ctx.unify(&fun_result.ty, &expected_fun)?;
    
    // Argument must have type σ
    ctx.unify(&arg_result.ty, &param_ty)?;
    
    // CRITICAL THREADING:
    // 1. e₂.ans_out = e₁.ans_in (= γ)
    //    "e₂'s output answer type = e₁'s input answer type"
    ctx.unify(&arg_result.ans_out, &fun_result.ans_in)?;
    
    // 2. e₂.ans_in = β (function's latent ans_out)
    //    "e₂ starts where function body will end"
    ctx.unify(&arg_result.ans_in, &beta)?;
    
    Ok(InferResult {
        ty: ctx.subst.apply(&ret_ty),
        ans_in: ctx.subst.apply(&alpha),           // α: overall input
        ans_out: ctx.subst.apply(&fun_result.ans_out), // δ: overall output
    })
}
```

### 5.5 Reset (Delimiter - Makes Expression Pure)

```rust
/// Reset:
///   Γ; σ ⊢ e : σ; τ
///   ─────────────────
///   Γ ⊢ₚ ⟨e⟩ : τ
///
/// Key insight: reset is PURE because it delimits all control effects
fn infer_reset(ctx: &mut InferCtx, body: &Expr) -> Result<InferResult, TypeError> {
    // Infer body
    let body_result = infer(ctx, body)?;
    
    // CRITICAL: body's initial answer type = body's expression type
    // This is what makes reset a delimiter
    ctx.unify(&body_result.ans_in, &body_result.ty)?;
    
    // Reset produces body's final answer type
    let result_ty = ctx.subst.apply(&body_result.ans_out);
    
    // Reset itself is pure
    let outer_ans = ctx.fresh_type();
    Ok(InferResult {
        ty: result_ty,
        ans_in: outer_ans.clone(),
        ans_out: outer_ans,
    })
}
```

### 5.6 Shift (The Control Operator)

```rust
/// Shift:
///   Γ, k : ∀t.(τ/t → α/t); σ ⊢ e : σ; β
///   ─────────────────────────────────────
///   Γ; α ⊢ Sk.e : τ; β
///
/// The continuation k is polymorphic in its answer type!
fn infer_shift(
    ctx: &mut InferCtx,
    k_name: &str,
    body: &Expr,
) -> Result<InferResult, TypeError> {
    // Fresh variables
    let tau = ctx.fresh_type();    // Type of the "hole" (what k receives)
    let alpha = ctx.fresh_type();  // Original answer type
    let sigma = ctx.fresh_type();  // Body's initial answer type
    
    // k : ∀t.(τ/t → α/t)
    // The continuation takes the hole value and returns to original answer type
    // It's polymorphic in its own answer type because of the implicit reset
    let k_ans_var = ctx.fresh_var();
    let k_body_type = Type::Fun {
        arg: Box::new(tau.clone()),
        ret: Box::new(alpha.clone()),
        ans_in: Box::new(Type::Var(k_ans_var)),
        ans_out: Box::new(Type::Var(k_ans_var)),  // Pure! (wrapped in reset)
    };
    let k_type = TypeScheme::Poly {
        vars: vec![k_ans_var],
        body: k_body_type,
    };
    
    // Extend environment with k
    let old_binding = ctx.env.insert(k_name.to_string(), k_type);
    
    // Infer body
    let body_result = infer(ctx, body)?;
    
    // Restore environment
    match old_binding {
        Some(scheme) => ctx.env.insert(k_name.to_string(), scheme),
        None => ctx.env.remove(k_name),
    };
    
    // Body's type = body's initial answer type (shift body goes to reset)
    ctx.unify(&body_result.ty, &body_result.ans_in)?;
    
    // sigma from rule = body's initial answer type
    ctx.unify(&sigma, &body_result.ans_in)?;
    
    Ok(InferResult {
        ty: ctx.subst.apply(&tau),
        ans_in: ctx.subst.apply(&alpha),
        ans_out: ctx.subst.apply(&body_result.ans_out),  // β
    })
}
```

### 5.7 Let (With Purity Restriction)

```rust
/// Let: let x = e₁ in e₂
///
/// Evaluation: e₁ first, then e₂ with x bound
///
/// Answer type threading (sequential):
///   e₁: ans_in = α, ans_out = β
///   e₂: ans_in = β, ans_out = γ
///   ────────────────────────────
///   let x = e₁ in e₂: ans_in = α, ans_out = γ
///
/// Purity restriction:
///   - If e₁ is pure (α = β), generalize its type
///   - If e₁ is effectful (α ≠ β), keep monomorphic
///
/// Formal rule:
///   Γ ⊢ₚ e₁ : σ              (if pure, can generalize)
///   Γ, x : Gen(σ; Γ); α ⊢ e₂ : τ; β
///   ─────────────────────────────────
///   Γ; α ⊢ let x = e₁ in e₂ : τ; β
fn infer_let(
    ctx: &mut InferCtx,
    name: &str,
    binding: &Expr,
    body: &Expr,
) -> Result<InferResult, TypeError> {
    // Infer binding
    let bind_result = infer(ctx, binding)?;
    
    // Check if binding is pure (answer types can unify)
    let is_pure = ctx.unify(&bind_result.ans_in, &bind_result.ans_out).is_ok();
    
    // Generalize only if pure
    let scheme = if is_pure {
        generalize(ctx, &bind_result.ty)
    } else {
        TypeScheme::Mono(ctx.subst.apply(&bind_result.ty))
    };
    
    // Extend environment
    let old_binding = ctx.env.insert(name.to_string(), scheme);
    
    // Infer body
    let body_result = infer(ctx, body)?;
    
    // Restore environment
    match old_binding {
        Some(scheme) => ctx.env.insert(name.to_string(), scheme),
        None => ctx.env.remove(name),
    };
    
    // Thread: binding's output feeds into body's input
    ctx.unify(&bind_result.ans_out, &body_result.ans_in)?;
    
    Ok(InferResult {
        ty: body_result.ty,
        ans_in: bind_result.ans_in,   // Start where binding starts
        ans_out: body_result.ans_out, // End where body ends
    })
}

/// Generalize a type over variables not free in the environment
fn generalize(ctx: &InferCtx, ty: &Type) -> TypeScheme {
    let ty = ctx.subst.apply(ty);
    let ty_vars = ty.free_vars();
    let env_vars = ctx.env_free_vars();
    
    let generalizable: Vec<_> = ty_vars
        .difference(&env_vars)
        .copied()
        .collect();
    
    if generalizable.is_empty() {
        TypeScheme::Mono(ty)
    } else {
        TypeScheme::Poly {
            vars: generalizable,
            body: ty,
        }
    }
}
```

### 5.8 Binary Operations

```rust
/// Binary operations: e₁ op e₂
/// 
/// Evaluation order: e₁ first, then e₂, then pure operation
/// 
/// Answer type threading (sequential):
///   e₁: ans_in = α, ans_out = β
///   e₂: ans_in = β, ans_out = γ
///   ─────────────────────────────
///   e₁ op e₂: ans_in = α, ans_out = γ
///
/// Key: e₁.ans_out = e₂.ans_in (left feeds into right)
fn infer_binop(
    ctx: &mut InferCtx,
    op: BinOp,
    left: &Expr,
    right: &Expr,
) -> Result<InferResult, TypeError> {
    let left_result = infer(ctx, left)?;
    let right_result = infer(ctx, right)?;
    
    // CRITICAL: Thread answer types left-to-right
    // Left's output becomes right's input
    ctx.unify(&left_result.ans_out, &right_result.ans_in)?;
    
    // Type check operands based on operator
    let result_ty = match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
            ctx.unify(&left_result.ty, &Type::Int)?;
            ctx.unify(&right_result.ty, &Type::Int)?;
            Type::Int
        }
        BinOp::Eq => {
            ctx.unify(&left_result.ty, &right_result.ty)?;
            Type::Bool
        }
        BinOp::Lt => {
            ctx.unify(&left_result.ty, &Type::Int)?;
            ctx.unify(&right_result.ty, &Type::Int)?;
            Type::Bool
        }
    };
    
    Ok(InferResult {
        ty: result_ty,
        ans_in: left_result.ans_in,   // Start where left starts
        ans_out: right_result.ans_out, // End where right ends
    })
}
```

### 5.9 If-Then-Else

```rust
/// If-then-else: if cond then e₁ else e₂
///
/// Evaluation: cond first, then ONE of the branches
/// Branches are alternatives, not sequential - they must have same answer types
///
/// Answer type threading:
///   cond: ans_in = α, ans_out = β
///   then: ans_in = β, ans_out = γ
///   else: ans_in = β, ans_out = γ  (same as then!)
///   ────────────────────────────────
///   if-then-else: ans_in = α, ans_out = γ
///
/// Key constraints:
///   - cond.ans_out = then.ans_in = else.ans_in
///   - then.ans_out = else.ans_out (branches unify)
fn infer_if(
    ctx: &mut InferCtx,
    cond: &Expr,
    then_br: &Expr,
    else_br: &Expr,
) -> Result<InferResult, TypeError> {
    let cond_result = infer(ctx, cond)?;
    ctx.unify(&cond_result.ty, &Type::Bool)?;
    
    let then_result = infer(ctx, then_br)?;
    let else_result = infer(ctx, else_br)?;
    
    // Branches must have same type
    ctx.unify(&then_result.ty, &else_result.ty)?;
    
    // Branches must have same answer type effects (they're alternatives)
    ctx.unify(&then_result.ans_in, &else_result.ans_in)?;
    ctx.unify(&then_result.ans_out, &else_result.ans_out)?;
    
    // Thread: cond feeds into branches
    ctx.unify(&cond_result.ans_out, &then_result.ans_in)?;
    
    Ok(InferResult {
        ty: ctx.subst.apply(&then_result.ty),
        ans_in: cond_result.ans_in,    // Start at condition
        ans_out: then_result.ans_out,  // End at branch (both same)
    })
}
```

---

## Part 6: Pretty Printing

### 6.1 Type Display

```rust
impl Type {
    pub fn display(&self, subst: &Substitution) -> String {
        self.display_impl(subst, false)
    }
    
    fn display_impl(&self, subst: &Substitution, in_fun_arg: bool) -> String {
        match subst.apply(self) {
            Type::Int => "Int".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::Unit => "()".to_string(),
            Type::Var(v) => format!("'{}", var_name(v)),
            Type::Fun { arg, ret, ans_in, ans_out } => {
                let ans_in_resolved = subst.apply(&ans_in);
                let ans_out_resolved = subst.apply(&ans_out);
                
                // Check if pure (answer types equal)
                let is_pure = ans_in_resolved == ans_out_resolved;
                
                let arg_str = arg.display_impl(subst, true);
                let ret_str = ret.display_impl(subst, false);
                
                let fun_str = if is_pure {
                    // Pure function: show as σ → τ
                    format!("{} -> {}", arg_str, ret_str)
                } else {
                    // Effectful: show as σ/α → τ/β
                    let ai = ans_in_resolved.display_impl(subst, false);
                    let ao = ans_out_resolved.display_impl(subst, false);
                    format!("{}/{} -> {}/{}", arg_str, ai, ret_str, ao)
                };
                
                if in_fun_arg {
                    format!("({})", fun_str)
                } else {
                    fun_str
                }
            }
            Type::List(elem) => format!("[{}]", elem.display_impl(subst, false)),
            Type::Forall(v, body) => {
                format!("∀{}. {}", var_name(v), body.display_impl(subst, false))
            }
            _ => "?".to_string(),
        }
    }
}

fn var_name(id: TypeVarId) -> String {
    let mut id = id as usize;
    let mut name = String::new();
    loop {
        name.insert(0, (b'a' + (id % 26) as u8) as char);
        id /= 26;
        if id == 0 { break; }
        id -= 1;
    }
    name
}
```

---

## Part 7: Test Suite

### 7.1 Property: Pure Expressions Have Equal Answer Types

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    fn assert_pure(expr: &str) {
        let ast = parse(expr).unwrap();
        let mut ctx = InferCtx::new();
        let result = infer(&mut ctx, &ast).unwrap();
        assert!(
            result.is_pure(&ctx),
            "Expected {} to be pure, got ans_in={:?}, ans_out={:?}",
            expr,
            ctx.subst.apply(&result.ans_in),
            ctx.subst.apply(&result.ans_out)
        );
    }
    
    fn assert_effectful(expr: &str) {
        let ast = parse(expr).unwrap();
        let mut ctx = InferCtx::new();
        let result = infer(&mut ctx, &ast).unwrap();
        assert!(
            !result.is_pure(&ctx),
            "Expected {} to be effectful",
            expr
        );
    }

    #[test]
    fn test_literals_are_pure() {
        assert_pure("42");
        assert_pure("true");
        assert_pure("()");
    }
    
    #[test]
    fn test_lambda_is_pure() {
        assert_pure("fun x -> x");
        assert_pure("fun x -> x + 1");
    }
    
    #[test]
    fn test_reset_is_pure() {
        assert_pure("reset (1 + 2)");
        assert_pure("reset (shift (fun k -> k 1))");
        assert_pure("reset (shift (fun k -> 42))");  // Even with effectful body!
    }
    
    #[test]
    fn test_shift_is_effectful() {
        assert_effectful("shift (fun k -> k 1)");
        assert_effectful("shift (fun k -> 42)");
    }
    
    #[test]
    fn test_pure_application_is_pure() {
        assert_pure("(fun x -> x) 42");
        assert_pure("(fun x -> x + 1) 5");
    }
}
```

### 7.2 Property: Shift Has Correct Continuation Type

```rust
#[test]
fn test_continuation_type_in_shift() {
    // In: shift (fun k -> k 1)
    // k should have type: ∀t. (Int/t -> α/t) where α is answer type
    
    let ast = parse("reset (shift (fun k -> k 1) + 2)").unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    assert_eq!(ctx.subst.apply(&result.ty), Type::Int);
}

#[test]
fn test_continuation_can_be_called_multiple_times() {
    // k is called twice with different values
    let ast = parse("reset (shift (fun k -> k 1 + k 2) * 10)").unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    assert_eq!(ctx.subst.apply(&result.ty), Type::Int);
}

#[test]
fn test_continuation_polymorphic_in_answer_type() {
    // k used in two different answer type contexts
    // This requires polymorphic k
    let code = r#"
        reset (
            let k = shift (fun k -> k) in
            let _ = reset (k 1) in   // k used with answer type Int
            reset (k true)           // k used with answer type Bool  
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    // Should type check successfully
    let result = infer(&mut ctx, &ast);
    assert!(result.is_ok(), "Polymorphic k should type check");
}
```

### 7.3 Property: Answer Type Modification Works

```rust
#[test]
fn test_answer_type_modification_simple() {
    // reset (1 + shift (fun k -> "hello"))
    // Answer type changes from Int to String
    let ast = parse(r#"reset (1 + shift (fun k -> "hello"))"#).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    // Result type is String (the modified answer type)
    assert_eq!(ctx.subst.apply(&result.ty), Type::String);
}

#[test]
fn test_append_example() {
    // The classic append example
    // append : 'a list / 'b -> 'a list / ('a list -> 'b)
    let code = r#"
        let rec append = fun lst ->
            match lst with
            | [] -> shift (fun k -> k)
            | x :: xs -> x :: append xs
        in
        let append123 = reset (append [1, 2, 3]) in
        append123 [4, 5]
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    // Result is a list
    assert!(matches!(ctx.subst.apply(&result.ty), Type::List(_)));
}
```

### 7.4 Property: Let-Polymorphism Respects Purity

```rust
#[test]
fn test_pure_let_is_polymorphic() {
    // let id = fun x -> x should be polymorphic
    let code = r#"
        let id = fun x -> x in
        let _ = id 1 in
        id true
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    assert_eq!(ctx.subst.apply(&result.ty), Type::Bool);
}

#[test]
fn test_reset_enables_polymorphism() {
    // reset makes effectful code pure, enabling polymorphism
    let code = r#"
        let f = reset (shift (fun k -> fun x -> k x)) in
        let _ = f 1 in
        f true
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast);
    
    // Should succeed because reset makes f pure
    assert!(result.is_ok());
}

#[test]
fn test_effectful_let_is_monomorphic() {
    // Without reset, shift makes binding effectful -> monomorphic
    let code = r#"
        reset (
            let f = shift (fun k -> k (fun x -> x)) in
            let _ = f 1 in
            f true
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast);
    
    // Should fail: f is monomorphic, can't use at both Int and Bool
    assert!(result.is_err());
}
```

### 7.5 Property: Type Soundness (Evaluation Preserves Types)

```rust
#[test]
fn test_well_typed_programs_dont_get_stuck() {
    let programs = vec![
        "reset (1 + 2)",
        "reset (shift (fun k -> k 1) + 2)",
        "reset (shift (fun k -> k 1 + k 2) * 3)",
        "reset (1 + reset (shift (fun k -> k 10) + 5))",
        "reset (shift (fun k -> 42))",
        "reset (let x = shift (fun k -> k 1) in x + 1)",
    ];
    
    for prog in programs {
        let ast = parse(prog).unwrap();
        
        // Type check
        let mut ctx = InferCtx::new();
        let ty_result = infer(&mut ctx, &ast);
        assert!(ty_result.is_ok(), "Type check failed for: {}", prog);
        
        // Evaluate
        let eval_result = eval(&ast);
        assert!(eval_result.is_ok(), "Evaluation failed for: {}", prog);
    }
}
```

### 7.6 Property: Type Inference Finds Principal Types

```rust
#[test]
fn test_identity_has_polymorphic_type() {
    let ast = parse("fun x -> x").unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    // Should be: ∀a b. a/b -> a/b (or displayed as a -> a)
    match ctx.subst.apply(&result.ty) {
        Type::Fun { arg, ret, ans_in, ans_out } => {
            // arg and ret should be same variable
            assert_eq!(arg, ret);
            // ans_in and ans_out should be same (pure)
            assert_eq!(ans_in, ans_out);
        }
        _ => panic!("Expected function type"),
    }
}

#[test]
fn test_shift_k_k_has_correct_type() {
    // shift (fun k -> k) : τ/α -> α/(τ -> α)
    // Returns the continuation itself, changing answer type
    let ast = parse("shift (fun k -> k)").unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    // Result should be a function type
    assert!(matches!(ctx.subst.apply(&result.ty), Type::Fun { .. }));
}
```

### 7.7 Regression Tests (From Runtime Bugs)

```rust
#[test]
fn test_nested_shift_in_continuation_arg() {
    // This was a runtime bug - ensure types are consistent
    let code = r#"
        reset (
            shift (fun k1 -> k1 (shift (fun k2 -> k2 100))) + 1
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let ty_result = infer(&mut ctx, &ast);
    assert!(ty_result.is_ok());
    
    let eval_result = eval(&ast);
    assert_eq!(eval_result.unwrap(), Value::Int(101));
}

#[test]
fn test_multiple_continuation_invocations_type() {
    let code = r#"
        reset (
            let x = shift (fun k -> k 1 + k 10 + k 100) in
            x * 2
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    assert_eq!(ctx.subst.apply(&result.ty), Type::Int);
}
```

### 7.8 Application Threading Tests

```rust
/// These tests verify the tricky application threading:
/// e₂.ans_out = e₁.ans_in (not e₁.ans_out = e₂.ans_in!)

#[test]
fn test_effectful_function_application() {
    // f is effectful (has shift), applied to pure argument
    let code = r#"
        reset (
            let f = fun x -> x + shift (fun k -> k 0) in
            f 10
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast);
    assert!(result.is_ok());
}

#[test]
fn test_effectful_argument() {
    // Pure function applied to effectful argument
    let code = r#"
        reset (
            let f = fun x -> x + 1 in
            f (shift (fun k -> k 10))
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast);
    assert!(result.is_ok());
}

#[test]
fn test_both_effectful() {
    // Both function and argument have effects
    let code = r#"
        reset (
            let f = fun x -> x + shift (fun k -> k 0) in
            f (shift (fun k -> k 10))
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast);
    assert!(result.is_ok());
}

#[test]
fn test_answer_type_flows_through_application() {
    // This tests that the function's latent effect is properly threaded
    // The shift inside f changes the answer type
    let code = r#"
        reset (
            let f = fun x -> shift (fun k -> "changed") in
            f 1 + 2
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    
    // This should fail to type-check: f returns to string context,
    // but we try to add 2 to the result
    let result = infer(&mut ctx, &ast);
    assert!(result.is_err(), "Should fail: can't add Int to String answer");
}

#[test]  
fn test_answer_type_modification_through_app() {
    // Correct version: use the modified answer type properly
    let code = r#"
        reset (
            let f = fun x -> shift (fun k -> k x ^ " world") in
            f "hello"
        )
    "#;
    let ast = parse(code).unwrap();
    let mut ctx = InferCtx::new();
    let result = infer(&mut ctx, &ast).unwrap();
    
    // Result type should be String
    assert_eq!(ctx.subst.apply(&result.ty), Type::String);
}
```

---

## Part 8: Migration Guide

### 8.1 Backwards Compatibility

Existing code using simple `σ → τ` function types should continue to work:

```rust
impl Type {
    /// Migrate old Arrow type to new Fun type
    pub fn from_legacy_arrow(arg: Type, ret: Type) -> Type {
        // Treat as pure function with fresh answer type variable
        Type::Fun {
            arg: Box::new(arg),
            ret: Box::new(ret),
            ans_in: Box::new(Type::Var(DUMMY_VAR)),  // Will be unified
            ans_out: Box::new(Type::Var(DUMMY_VAR)), // Same var = pure
        }
    }
}
```

### 8.2 Gradual Rollout

1. **Phase 1**: Add new type representation, make old Arrow sugar for pure Fun
2. **Phase 2**: Update unification to handle answer types
3. **Phase 3**: Update inference for lambda, app (without shift/reset)
4. **Phase 4**: Add reset inference (makes things pure)
5. **Phase 5**: Add shift inference (the hard part)
6. **Phase 6**: Add let-polymorphism with purity restriction
7. **Phase 7**: Update error messages and pretty printing

### 8.3 Error Messages

```rust
pub enum TypeError {
    Mismatch(Type, Type),
    UnboundVariable(String),
    InfiniteType(TypeVarId, Type),
    
    // New error types for answer type system
    AnswerTypeMismatch {
        expected: Type,
        got: Type,
        context: String,
    },
    EffectfulLetBinding {
        name: String,
        hint: String,
    },
}

impl TypeError {
    pub fn display(&self) -> String {
        match self {
            TypeError::EffectfulLetBinding { name, hint } => {
                format!(
                    "Cannot generalize '{}': binding has control effects.\n\
                     Hint: {}",
                    name, hint
                )
            }
            // ...
        }
    }
}
```

---

## Part 9: Implementation Checklist

### Phase 1: Type Representation
- [ ] Add `ans_in`, `ans_out` fields to `Fun` variant
- [ ] Add `Forall` variant for polymorphic types
- [ ] Add `TypeScheme` enum
- [ ] Add helper constructors (`pure_fun`, `effectful_fun`)
- [ ] Update `Clone`, `Debug`, `PartialEq` derives

### Phase 2: Unification
- [ ] Extend `unify` to handle 4-field `Fun` type
- [ ] Update `occurs_check` for new fields
- [ ] Update `Substitution::apply` for new fields
- [ ] Add tests for function type unification

### Phase 3: Inference Infrastructure  
- [ ] Change `infer` return type to `InferResult` (ty, ans_in, ans_out)
- [ ] Add `is_pure` helper method
- [ ] Update all existing inference cases to return 3 types
- [ ] Add `generalize` function

### Phase 4: Core Typing Rules
- [ ] Literals (pure)
- [ ] Variables (pure)
- [ ] Lambda (pure, builds function type)
- [ ] Application (threads answer types)
- [ ] Let (with purity restriction)
- [ ] If-then-else
- [ ] Binary operations

### Phase 5: Control Operators
- [ ] Reset (makes body pure)
- [ ] Shift (polymorphic continuation)

### Phase 6: Testing
- [ ] Purity property tests
- [ ] Continuation type tests
- [ ] Answer type modification tests
- [ ] Let-polymorphism tests
- [ ] Type soundness tests
- [ ] Regression tests

### Phase 7: Polish
- [ ] Pretty printing for answer types
- [ ] Error messages
- [ ] Documentation

---

## References

1. Asai & Kameyama, "Polymorphic Delimited Continuations" (APLAS 2007)
2. Danvy & Filinski, "A Functional Abstraction of Typed Contexts" (1989)
3. Asai & Kiselyov, "Introduction to Programming with Shift and Reset" (2011)
