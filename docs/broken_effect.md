# Gneiss Effect System: Analysis and Recommendations

A comprehensive review of the algebraic effect implementation in your language.

---

## Executive Summary

Your effect system has a solid foundation but several issues that could cause unsoundness or surprising behavior:

1. **Continuation effect typing is incorrect** — Continuations in handlers are typed as pure, but they should carry the remaining effect row
2. **Row unification is incomplete** — Missing cases for complex row interactions
3. **Effect type parameters work but have subtle instantiation issues**
4. **State threading requires function-returning handlers** — Your current examples don't demonstrate this properly

---

## Part 1: What's Working

### 1.1 Effect Declaration Parsing ✓

The AST correctly captures effect declarations with type parameters:

```rust
// ast.rs:706-712
EffectDecl {
    visibility: Visibility,
    name: Ident,
    params: Vec<Ident>,        // ✓ Type parameters supported
    operations: Vec<EffectOperation>,
}
```

### 1.2 Effect Registration ✓

`register_effect` correctly builds a param map and converts operation signatures:

```rust
// infer.rs:322-326
let mut param_map: HashMap<String, TypeVarId> = HashMap::new();
for (i, param) in params.iter().enumerate() {
    param_map.insert(param.clone(), i as TypeVarId);
}
```

This means `effect State s = | get : () -> s | put : s -> () end` correctly maps `s` to `Generic(0)`.

### 1.3 Perform Instantiation ✓

When you write `perform State.get ()`, the type checker:
1. Looks up `get` → finds it belongs to `State`
2. Creates fresh type variable for `s` (say `?t42`)
3. Substitutes `Generic(0)` → `?t42` in the operation signature
4. Returns `?t42` with effect row `{ State ?t42 | ε }`

This is correct.

### 1.4 Row Types ✓

The row representation is standard and correct:

```rust
// types.rs:84-94
pub enum Row {
    Empty,
    Extend { effect: Effect, rest: Rc<Row> },
    Var(Rc<RefCell<RowVar>>),
}
```

---

## Part 2: Critical Issues

### 2.1 CRITICAL: Continuation Effect Typing Bug

**Location:** `infer.rs:2136-2141`

**The Problem:**
```rust
let cont_type = Type::Arrow {
    arg: Rc::new(cont_arg),
    ret: Rc::new(result_ty.clone()),
    effects: Row::Empty,  // ← BUG: Should NOT be Empty!
};
```

The continuation `k` in a handler should have the *remaining* effects after handling, not an empty row.

**Why This Matters:**

Consider:
```gneiss
effect State s = | get : () -> s | put : s -> () end
effect Log = | log : String -> () end

let example () =
    perform Log.log "before";
    let x = perform State.get () in
    perform Log.log "after";
    x

-- In handler:
handle example () with
| return x -> x
| get () k -> k 0  -- k should have type: Int -> a ! {Log | ε}, NOT Int -> a
end
```

When `k` is invoked, the continuation still needs to perform `Log` effects. If `k` is typed as pure, the type system won't track these effects.

**The Fix:**
```rust
// The continuation should have the remaining effects (body effects minus handled effects)
let remaining_effects = self.subtract_effects(&body_result.effects, &handled_effects);
let cont_type = Type::Arrow {
    arg: Rc::new(cont_arg),
    ret: Rc::new(result_ty.clone()),
    effects: remaining_effects.clone(),  // ← Use remaining effects
};
```

**But Wait:** This is still incomplete. In a *deep handler* (which yours is), the continuation includes the handler itself. So effects performed inside the continuation that match the handler should be caught again. This is actually handled by your runtime (the `HandleScope` is included in captured frames), but the type system should reflect this.

For a simpler approximation that's still sound:
```rust
// Continuation can perform any effects the body can, minus the one being handled NOW
// But since we're a deep handler, the handled effect is also available again
// So really, the continuation has the same effects as the body
let cont_type = Type::Arrow {
    arg: Rc::new(cont_arg),
    ret: Rc::new(result_ty.clone()),
    effects: body_result.effects.clone(),  // Same as body - simplification
};
```

---

### 2.2 CRITICAL: Row Unification Incomplete

**Location:** `infer.rs:634-720`

**Missing Cases:**

1. **Var vs Extend with rewriting:**
```rust
// When unifying Row::Var(r) with Row::Extend { effect: E, rest: ... }
// You need to instantiate r to { E | r' } for fresh r'
```

Current code handles some cases but may fail on:
```gneiss
let f : () -> a ! { State s | r } = ...
let g : () -> a ! { Log, State s } = f  -- Need to unify {State s | r} with {Log, State s}
```

2. **Effect parameter unification during row rewriting:**
```rust
// row_rewrite at line 724 unifies effect params, but...
if e1.params.len() != e2.params.len() {
    return Err(...);  // This is correct, but error message could be better
}
for (p1, p2) in e1.params.iter().zip(e2.params.iter()) {
    self.unify(p1, p2)?;  // ← Make sure this propagates correctly
}
```

---

### 2.3 HIGH: Handler Return Clause Effects Not Checked

**Location:** `infer.rs:2166-2167`

```rust
// Also combine with return clause's effects (should typically be pure)
let combined = self.union_rows(&remaining_effects, &return_body_result.effects);
```

The comment says "should typically be pure" but this isn't enforced. If the return clause performs effects, they'll be silently included in the result.

**Recommendation:** Either enforce purity or document that return clause effects are allowed.

---

### 2.4 MEDIUM: Effect Name Not Verified in Handle

**Location:** `infer.rs:2086-2092`

```rust
for handler in handlers {
    if let Some((effect_name, _)) = self.effect_env.operations.get(&handler.operation).cloned() {
        handled_effects.insert(effect_name);
    }
    // If operation not found, we'll still handle it but can't verify types
}
```

If you write:
```gneiss
handle expr with
| nonexistent_op x k -> ...
end
```

This silently creates an unverified handler. Should at least warn.

---

## Part 3: Effect Type Parameters

### 3.1 Yes, Type Parameters Work

Your syntax supports:
```gneiss
effect State s =
    | get : () -> s
    | put : s -> ()
end
```

And the implementation handles this correctly. The flow is:

1. **Declaration:** `params = ["s"]`
2. **Registration:** `param_map = {"s" -> 0}`
3. **Operation types:** `get: () -> Generic(0)`, `put: Generic(0) -> ()`
4. **Perform:** Fresh var `?t42` created, substituted for `Generic(0)`
5. **Effect in row:** `State ?t42`

### 3.2 Potential Issue: Multiple Type Parameters

```gneiss
effect RW r w =
    | read : () -> r
    | write : w -> ()
end
```

This should work, but verify:
- `read` gets type `() -> Generic(0)`
- `write` gets type `Generic(1) -> ()`
- When performing, both `r` and `w` get fresh variables

### 3.3 Row Variable in Effect Type Parameter Position

What about:
```gneiss
effect Eff e =
    | op : () -> () ! e  -- e is a row variable, not a type!
end
```

This currently won't work because effect type parameters are mapped to `Type`, not `Row`. If you need this, you'd need to extend the system to support row-kinded parameters.

---

## Part 4: State Effect Implementation Pattern

Your examples show a broken State handler:
```gneiss
handle ... with
| get () k -> k 0  -- Always returns 0!
| put s k -> k ()  -- Doesn't update anything!
end
```

**The Correct Pattern Requires Function-Returning Handlers:**

```gneiss
-- Handler returns a function: state -> result
let run_state initial comp =
    let go = 
        handle comp () with
        | return x -> fun s -> x              -- Result ignores final state (or return (x, s))
        | get () k -> fun s -> k s s          -- Pass state to k, thread state through
        | put new_s k -> fun s -> k () new_s  -- Ignore old state, continue with new
        end
    in
    go initial

-- Usage
let result = run_state 0 (fun () ->
    let x = perform State.get () in
    perform State.put (x + 1);
    perform State.get ()
)
-- result = 1
```

**The Type of This Handler:**

For `run_state : s -> (() -> a ! {State s | ε}) -> a`:
- The `handle` expression has type `s -> a` (function from state to result)
- Each clause must produce `s -> a`:
  - `return x -> fun s -> x` : Takes result, returns `s -> a`
  - `get () k -> fun s -> k s s` : Takes nothing, returns `s -> a` where `k : s -> (s -> a)` internally

**Your Type System May Not Support This!**

The challenge is that the handler body type depends on the return type in a complex way. In Koka, this works because of row-polymorphic effect types. In your system, you may need explicit annotations.

---

## Part 5: Verification Recommendations

### 5.1 Type Soundness Property Tests

Add these to your test suite:

```gneiss
-- Test 1: Effect escapes handler (should type error)
-- This should FAIL to type check:
let bad () =
    perform State.get ()  -- No handler!

-- Test 2: Wrong effect type parameter
let wrong_param () =
    handle (perform State.put "string") with  -- put expects the state type
    | put (n : Int) k -> k ()                  -- Handler expects Int
    end
-- Should be a type error

-- Test 3: Handler result type consistency
let inconsistent () =
    handle (...) with
    | return x -> x           -- Returns a
    | op () k -> "string"     -- Returns String!
    end
-- Should error if a ≠ String

-- Test 4: Effect row polymorphism
let polymorphic_handler (f : () -> a ! {State Int | ε}) : a ! ε =
    handle f () with
    | return x -> x
    | get () k -> k 0
    | put _ k -> k ()
    end
-- ε should be preserved in result

-- Test 5: Nested handlers
let nested () =
    handle (
        handle (...) with
        | get () k -> k 0
        end
    ) with
    | log msg k -> k ()
    end
-- Effects should be discharged correctly
```

### 5.2 Effect Law Tests

Add property tests for effect laws:

```gneiss
-- State laws
property put_get s =
    run_state 0 (fun () -> perform State.put s; perform State.get ())
    == s

property get_get =
    run_state 0 (fun () ->
        let x = perform State.get () in
        let y = perform State.get () in
        x == y
    )

property put_put s1 s2 =
    run_state 0 (fun () -> perform State.put s1; perform State.put s2; perform State.get ())
    == s2
```

---

## Part 6: Specific Code Fixes

### Fix 1: Continuation Effect Row

```rust
// infer.rs, in Handle case, around line 2135

// Calculate remaining effects properly
let remaining_effects = self.subtract_effects(&body_result.effects, &handled_effects);

// For deep handlers, the continuation can re-perform handled effects
// So use the body's full effect row
let cont_effects = body_result.effects.clone();

let cont_type = Type::Arrow {
    arg: Rc::new(cont_arg),
    ret: Rc::new(result_ty.clone()),
    effects: cont_effects,  // FIXED
};
```

### Fix 2: Warn on Unknown Operations

```rust
// infer.rs, around line 2087

for handler in handlers {
    if let Some((effect_name, _)) = self.effect_env.operations.get(&handler.operation).cloned() {
        handled_effects.insert(effect_name);
    } else {
        // ADD WARNING
        self.warnings.push(Warning::UnknownOperation {
            operation: handler.operation.clone(),
            span: handler.body.span.clone(),
        });
    }
}
```

### Fix 3: Row Unification Edge Case

```rust
// infer.rs, unify_rows function

// Add case for Var vs Extend when var could be instantiated
(Row::Var(v), Row::Extend { effect, rest }) | (Row::Extend { effect, rest }, Row::Var(v)) => {
    match &*v.borrow() {
        RowVar::Unbound { id, level } => {
            // Occurs check
            if rest.occurs(*id) {
                return Err(TypeError::OccursCheck { ... });
            }
            // Instantiate: v = { effect | fresh_row_var }
            let fresh_rest = Row::new_var(self.next_var, *level);
            self.next_var += 1;
            let new_row = Row::Extend {
                effect: effect.clone(),
                rest: Rc::new(fresh_rest.clone()),
            };
            *v.borrow_mut() = RowVar::Link(new_row);
            self.unify_rows(&fresh_rest, rest)
        }
        RowVar::Link(linked) => {
            self.unify_rows(linked, &Row::Extend { effect: effect.clone(), rest: rest.clone() })
        }
        RowVar::Generic(_) => {
            Err(TypeError::Other("Cannot unify generic row var".into()))
        }
    }
}
```

---

## Part 7: Summary of Issues by Severity

### Critical (Affects Soundness)
1. **Continuation typed as pure** — Effects inside continuations won't be tracked
2. **Row unification gaps** — Some valid programs will fail to type check

### High (Affects Usability)
3. **State handlers can't actually track state** — Documentation/examples needed
4. **Unknown operations silently ignored** — Should warn

### Medium (Code Quality)
5. **Handler return clause effects not enforced** — Should document or enforce
6. **Error messages for effect mismatches** — Could be improved

### Low (Nice to Have)
7. **Row-kinded type parameters** — For advanced patterns
8. **Effect aliases** — `effect alias Stateful s = State s, Log`

---

## Appendix: Complete Corrected Handler Typing

Here's what the full Handle inference should look like:

```rust
ExprKind::Handle { body, return_clause, handlers } => {
    // 1. Infer body
    let body_result = self.infer_expr_full(env, body)?;

    // 2. Collect handled effects
    let mut handled_effects: HashSet<String> = HashSet::new();
    let mut handler_effect_params: HashMap<String, Vec<Type>> = HashMap::new();
    
    for handler in handlers {
        if let Some((effect_name, op_info)) = self.effect_env.operations.get(&handler.operation).cloned() {
            // Get effect info for type params
            let effect_info = self.effect_env.effects.get(&effect_name).cloned();
            let num_params = effect_info.map(|e| e.type_params.len()).unwrap_or(0);
            
            // Create fresh vars for this effect's type params
            let effect_type_args: Vec<Type> = (0..num_params).map(|_| self.fresh_var()).collect();
            
            handled_effects.insert(effect_name.clone());
            handler_effect_params.insert(effect_name, effect_type_args);
        } else {
            return Err(TypeError::UnknownOperation { 
                operation: handler.operation.clone() 
            });
        }
    }

    // 3. Compute remaining effect row
    let remaining_effects = self.subtract_effects(&body_result.effects, &handled_effects);

    // 4. Determine result type from return clause
    let mut return_env = env.clone();
    let return_pattern_ty = body_result.ty.clone();  // Return pattern matches body result
    self.bind_pattern(&mut return_env, &return_clause.pattern, &return_pattern_ty)?;
    
    let return_body_result = self.infer_expr_full(&return_env, &return_clause.body)?;
    let result_ty = return_body_result.ty.clone();

    // 5. Type check each handler clause
    for handler in handlers {
        let mut handler_env = env.clone();
        let (effect_name, op_info) = self.effect_env.operations.get(&handler.operation).unwrap().clone();
        let effect_type_args = handler_effect_params.get(&effect_name).unwrap();

        // Bind operation parameters
        let instantiated_params: Vec<Type> = op_info.param_types.iter()
            .map(|t| substitute_generics(t, effect_type_args))
            .collect();
        
        for (param, param_ty) in handler.params.iter().zip(instantiated_params.iter()) {
            self.bind_pattern(&mut handler_env, param, param_ty)?;
        }

        // Continuation type: op_result -> handler_result ! remaining_effects
        // For deep handlers, continuation can also re-perform handled effects
        let cont_arg = substitute_generics(&op_info.result_type, effect_type_args);
        let cont_ret = result_ty.clone();
        let cont_effects = body_result.effects.clone();  // Full effects for deep handler
        
        let cont_type = Type::Arrow {
            arg: Rc::new(cont_arg),
            ret: Rc::new(cont_ret),
            effects: cont_effects,
        };
        handler_env.insert(handler.continuation.clone(), Scheme::mono(cont_type));

        // Handler body must produce result type
        let handler_body_result = self.infer_expr_full(&handler_env, &handler.body)?;
        self.unify_at(&handler_body_result.ty, &result_ty, &handler.body.span)?;
    }

    // 6. Result effects = remaining + return clause effects
    let final_effects = self.union_rows(&remaining_effects, &return_body_result.effects);

    Ok(InferResult::with_effects(result_ty, final_effects))
}
```

---

*Analysis generated for Gneiss language implementation*
