# Answer-Type Polymorphism: Research and Implementation Plan

## Executive Summary

The current Gneiss implementation has a **working** answer-type modification system but **lacks full answer-type polymorphism**. The key issue is that continuations captured by `shift` are given monomorphic types instead of polymorphic ones, which prevents patterns like printf from type-checking correctly.

---

## 1. What is Answer-Type Polymorphism?

### The Problem it Solves

In a language with delimited continuations (`shift`/`reset`), the "answer type" is the type of the final result produced by the enclosing `reset`. Consider:

```gneiss
reset (1 + shift (fun k -> k 2))  -- Returns 3 : Int
reset (1 + shift (fun k -> "hello"))  -- Returns "hello" : String (!)
```

In the second case, `shift` **modifies** the answer type from `Int` to `String` by discarding the continuation.

### Why Polymorphism is Needed

The continuation `k` captured by `shift` should be **polymorphic in its own answer type**. This is critical for patterns like:

```gneiss
-- printf: Each %s changes the answer type from String to (String -> String)
let % to_str = shift (fun k -> fun x -> k (to_str x))
sprintf (FStr (FLit "!" FEnd))  -- : String -> String
```

Here, `k` is used in a context where it must return `String -> String`, not just `String`. The continuation must be usable at different answer types within the same shift body.

### The "prefix" Example (from Asai & Kameyama)

```gneiss
let rec visit lst = match lst with
  | [] -> shift (fun h -> [])
  | a :: rest -> a :: shift (fun k ->
      (k []) :: reset (k (visit rest)))
```

Here `k` is used **twice** with different answer types:
- `k []` at type `'a list / 'a list list -> 'a list / 'a list list`  
- `k (visit rest)` at type `'a list / 'a list -> 'a list / 'a list`

Without answer-type polymorphism, this cannot type-check.

---

## 2. The Typing Rule

From Asai & Kameyama's "Polymorphic Delimited Continuations" (APLAS 2007):

```
Γ, k : ∀t.(τ/t → α/t); σ ⊢ e : σ; β
─────────────────────────────────────
      Γ; α ⊢ Sk.e : τ; β           (shift)
```

Key insights:
- `τ` is the "hole type" (what shift returns to the context)
- `α` is the captured answer type (what the context originally expected)
- `k : ∀t.(τ/t → α/t)` means k is **polymorphic** in its answer type `t`
- The body has answer type `σ → β`, and `σ = body.ty` (body's type equals its initial answer type)
- `β` is the final answer type after all transformations

### Why k is Pure

The continuation `k` is typed as a **pure** function (same `t` for both ans_in and ans_out) because invoking the continuation is semantically equivalent to wrapping in `reset`:

```
k v  ≡  reset (F[v])  -- where F is the captured context
```

Since `reset` delimits effects, the continuation invocation is pure from the perspective of its own answer type.

---

## 3. Current Implementation Analysis

### What Gneiss Has (Correct)

1. **Answer type fields in Arrow**: ✅
   ```rust
   Type::Arrow { arg, ret, ans_in, ans_out }
   ```

2. **Answer type threading**: ✅
   ```rust
   // In application: e₂.ans_out = e₁.ans_in
   self.unify(&arg_result.ans_out, &fun_result.ans_in)?;
   ```

3. **Reset constraint**: ✅
   ```rust
   // body.ans_in = body.ty
   self.unify(&body_result.answer_before, &body_result.ty)?;
   ```

4. **Type schemes for let-polymorphism**: ✅

### What's Wrong (The Bug)

In `src/infer.rs`, the shift implementation:

```rust
// k : ∀t.(τ/t → α/t)
// For now, we approximate this by using a pure function type
// with fresh answer type variable that can unify with any context
let k_ans = self.fresh_var();
let k_type = Type::Arrow {
    arg: Rc::new(tau.clone()),
    ret: Rc::new(alpha.clone()),
    ans_in: Rc::new(k_ans.clone()),
    ans_out: Rc::new(k_ans), // Pure! (same ans_in and ans_out)
};

// Bind k in body's environment
let mut body_env = env.clone();
self.bind_pattern(&mut body_env, param, &k_type)?;  // ← MONOMORPHIC binding!
```

The problem: `k_type` is a **monomorphic** Arrow type. When `k` is used multiple times in the shift body, they all share the same `k_ans` variable, which gets unified to a single concrete type.

**What should happen**: `k` should be bound with a **polymorphic type scheme** `∀t.(τ/t → α/t)`, so each use of `k` gets fresh answer type variables.

---

## 4. The Fix

### Step 1: Create Polymorphic Continuation Type

```rust
// In shift inference:

// τ = type of the "hole" (what shift returns to the context)
let tau = self.fresh_var();
// α = the captured answer type (what the context expected)
let alpha = self.fresh_var();

// Create a type scheme for k: ∀t.(τ/t → α/t)
// We need to explicitly represent the polymorphism
let k_scheme = {
    // The answer type variable that will be generalized
    let t_id = self.next_var_id;
    self.next_var_id += 1;
    
    let k_body_type = Type::Arrow {
        arg: Rc::new(tau.clone()),
        ret: Rc::new(alpha.clone()),
        ans_in: Rc::new(Type::new_generic(0)),   // Generic 't'
        ans_out: Rc::new(Type::new_generic(0)),  // Same 't' (pure)
    };
    
    Scheme {
        num_generics: 1,  // One generic: 't' (the answer type)
        ty: k_body_type,
    }
};

// Bind k with its POLYMORPHIC scheme
let mut body_env = env.clone();
body_env.insert(k_name.clone(), k_scheme);
```

### Step 2: Instantiation at Each Use

When `k` is referenced in the body, the `instantiate` function will automatically replace `Generic(0)` with a fresh type variable. Each use of `k` gets its own answer type variable.

This is already implemented correctly in `Inferencer::instantiate()`:

```rust
fn instantiate(&mut self, scheme: &Scheme) -> Type {
    if scheme.num_generics == 0 {
        return scheme.ty.clone();
    }

    let mut substitution: HashMap<TypeVarId, Type> = HashMap::new();
    for i in 0..scheme.num_generics {
        substitution.insert(i, self.fresh_var());  // Fresh var for each generic
    }

    self.substitute(&scheme.ty, &substitution)
}
```

### Step 3: Modify bind_pattern or Add bind_pattern_scheme

Currently `bind_pattern` creates monomorphic bindings. For shift, we need to directly insert a scheme:

```rust
// Option A: Direct insertion
body_env.insert(k_name.clone(), k_scheme);

// Option B: Use existing bind_pattern_scheme if available
self.bind_pattern_scheme(&mut body_env, param, k_scheme)?;
```

---

## 5. Complete Implementation

Here's the corrected `infer_shift` function:

```rust
ExprKind::Shift { param, body } => {
    // τ = type of the "hole" (what shift returns to the context)
    let tau = self.fresh_var();
    // α = the captured answer type (what the context expected)
    let alpha = self.fresh_var();

    // k : ∀t.(τ/t → α/t)
    // The continuation is POLYMORPHIC in its answer type
    // This is essential for answer-type polymorphism
    let k_type_body = Type::Arrow {
        arg: Rc::new(tau.clone()),
        ret: Rc::new(alpha.clone()),
        ans_in: Rc::new(Type::new_generic(0)),   // Generic answer type 't'
        ans_out: Rc::new(Type::new_generic(0)),  // Same 't' (pure function)
    };
    
    let k_scheme = Scheme {
        num_generics: 1,  // ∀t. ...
        ty: k_type_body,
    };

    // Bind k in body's environment with its POLYMORPHIC type
    let mut body_env = env.clone();
    
    // Handle the pattern (usually just a variable name)
    match &param.node {
        PatternKind::Var(k_name) => {
            body_env.insert(k_name.clone(), k_scheme);
        }
        PatternKind::Wildcard => {
            // k is not used, nothing to bind
        }
        _ => {
            return Err(TypeError::InvalidPattern(
                "shift parameter must be a variable or wildcard".to_string()
            ));
        }
    }

    // Infer body with its own answer type tracking
    let body_result = self.infer_expr_full(&body_env, body)?;

    // Body type must equal body's initial answer type
    // This ensures the shift body produces a value compatible with the reset
    self.unify(&body_result.ty, &body_result.answer_before)?;

    // Shift returns τ (the hole type) with answer type α → β
    // α = original answer type (before shift)
    // β = body's final answer type (after shift)
    Ok(InferResult {
        ty: tau,
        answer_before: alpha,
        answer_after: body_result.answer_after,
    })
}
```

---

## 6. Additional Considerations

### Purity Restriction for Let-Polymorphism

The current implementation already has this partially:

```rust
// Only pure expressions can be generalized
let is_pure = types_equal(&ans_in, &ans_out);
let scheme = if Self::is_syntactic_value(value) && is_pure {
    self.generalize(&value_result.ty)
} else {
    Scheme::mono(value_result.ty.clone())
};
```

This is correct! Expressions with control effects (ans_in ≠ ans_out) cannot be generalized.

### Reset Makes Things Pure

The reset implementation correctly makes the body pure from the outside:

```rust
// Reset itself is PURE from the outside - it delimits all effects
Ok(InferResult::pure(body_result.answer_after, ans))
```

### Display/Pretty-Printing

The existing display logic for Arrow types already handles showing effectful vs pure functions:

```rust
if is_pure {
    // Pure function: show as σ → τ
    write!(f, "{} -> {}", arg_str, ret)
} else {
    // Effectful: show as σ/α → τ/β
    write!(f, "{}/{} -> {}/{}", arg_str, ans_in, ret, ans_out)
}
```

---

## 7. Test Cases to Verify

After implementing, these should type-check:

### Basic Answer-Type Modification
```gneiss
reset (shift (fun k -> "hello"))  -- : String (not Int!)
reset (1 + shift (fun k -> "hello"))  -- : String
```

### Multiple K Uses (Same Answer Type)
```gneiss
reset (shift (fun k -> k 1 + k 2))  -- : Int
```

### Multiple K Uses (Different Answer Types) - THE KEY TEST
```gneiss
-- This is the prefix pattern
reset (shift (fun k -> (k []) :: reset (k [1,2,3])))
```

### Printf-Style Patterns
```gneiss
let format_string rest = shift (fun k -> fun s -> k (s ^ rest))
reset (format_string "world")  -- : String -> String
```

---

## 8. References

1. **Asai & Kameyama**, "Polymorphic Delimited Continuations" (APLAS 2007)
   - The foundational paper for this type system
   - Available at: https://www.cs.tsukuba.ac.jp/~kam/paper/aplas07.pdf

2. **Asai & Kiselyov**, "Introduction to Programming with Shift and Reset" (2011)
   - Tutorial with practical examples
   - Available at: http://pllab.is.ocha.ac.jp/~asai/cw2011tutorial/main-e.pdf

3. **Oleg Kiselyov's Implementation Page**
   - Haskell implementation with answer-type modification and polymorphism
   - Available at: https://okmij.org/ftp/continuations/implementations.html

4. **Scala Delimited Continuations** (Rompf et al., ICFP 2009)
   - Type-directed selective CPS transform approach
   - Shows how to implement in a real language

5. **Koka Language** (Leijen)
   - Row-polymorphic effects with effect handlers
   - Modern approach to similar problems

---

## 9. Implementation Checklist

- [ ] Modify shift inference to create polymorphic scheme for k
- [ ] Ensure k is bound with Scheme, not monomorphic Type
- [ ] Add test cases for multiple k uses with different answer types
- [ ] Test printf-style patterns
- [ ] Test prefix-style patterns
- [ ] Verify purity restriction still works correctly
- [ ] Update documentation

---

## 10. Summary of the Fix

**The one-line summary**: Change how `k` is bound in shift from a monomorphic type to a polymorphic type scheme with `∀t.(τ/t → α/t)`.

The infrastructure is already there - type schemes, instantiation, generalization all work. The only missing piece is making shift use a polymorphic scheme instead of a monomorphic type for the continuation parameter.
