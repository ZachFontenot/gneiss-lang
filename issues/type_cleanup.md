# Type System Improvements

This document outlines ergonomic and cleanup improvements for the Gneiss type inference system, particularly around delimited continuations and answer types.

---

## 1. Error Message Improvements

### 1.1 Normalize Type Variables

**Problem:** Error messages show internal variable IDs like `t732`, `t985` which are meaningless to users.

**Solution:** Normalize variables to `a`, `b`, `c`, etc. when displaying types.

```rust
impl Type {
    pub fn display_normalized(&self) -> String {
        let mut var_map: HashMap<TypeVarId, char> = HashMap::new();
        let mut next_var = 'a';
        self.display_with_map(&mut var_map, &mut next_var)
    }
    
    fn display_with_map(
        &self, 
        var_map: &mut HashMap<TypeVarId, char>,
        next_var: &mut char
    ) -> String {
        match self.resolve() {
            Type::Var(var) => match &*var.borrow() {
                TypeVar::Unbound { id, .. } | TypeVar::Generic(id) => {
                    let c = *var_map.entry(*id).or_insert_with(|| {
                        let c = *next_var;
                        *next_var = (c as u8 + 1) as char;
                        c
                    });
                    format!("{}", c)
                }
                _ => unreachable!()
            },
            // ... handle other variants, recursively using display_with_map
        }
    }
}
```

**Before:**
```
I expected:  t732/t734 -> t733/t735
But found:   [Char]
```

**After:**
```
I expected:  a -> b
But found:   [Char]
```

---

### 1.2 Add Unification Context

**Problem:** Errors say "expected X, found Y" but don't explain *why* or *where*.

**Solution:** Track context during unification and include it in errors.

```rust
enum UnifyContext {
    FunctionArgument { func_name: Option<String>, param_num: usize },
    FunctionReturn { func_name: Option<String> },
    LetBinding { name: String },
    IfBranches,
    IfCondition,
    MatchArms,
    MatchScrutinee,
    BinOp { op: String, side: &'static str },
    ListElement { index: usize },
    TupleElement { index: usize },
}

// Update TypeError
enum TypeError {
    TypeMismatch { 
        expected: Type, 
        found: Type, 
        span: Option<Span>,
        context: Option<UnifyContext>,  // NEW
    },
    // ...
}

// New helper method
fn unify_with_context(
    &mut self, 
    expected: &Type, 
    found: &Type, 
    context: UnifyContext,
    span: &Span
) -> Result<(), TypeError> {
    self.unify_inner(expected, found, Some(span))
        .map_err(|e| e.with_context(context))
}
```

**Usage in App:**
```rust
self.unify_with_context(
    &param_ty, 
    &arg_result.ty,
    UnifyContext::FunctionArgument { 
        func_name: extract_func_name(func),
        param_num: 1 
    },
    &arg.span
)?;
```

**Before:**
```
I found a type mismatch.
  I expected:  () -> Char
  But found:   Char
```

**After:**
```
Type mismatch in argument 1 of `many1`

Expected:
    () -> Char

But found:
    Char

Hint: `many1` expects a parser (a thunk), but you passed 
a bare value. Try wrapping it: `(fun () -> digit ())`
```

---

### 1.3 Hide Answer Types from Users

**Problem:** Answer types like `()/[Char] -> ParseResult Int -> Int/[Char] -> ParseResult Int` are confusing implementation details.

**Solution:** Hide answer types in user-facing output unless they're "interesting" (i.e., indicate an effect mismatch).

```rust
impl Type {
    /// User-friendly display that hides answer types
    pub fn display_for_user(&self) -> String {
        // Like display_normalized, but for Arrow types:
        // - If ans_in == ans_out (pure), show as `arg -> ret`
        // - If ans_in != ans_out, still show as `arg -> ret` but 
        //   maybe add a note about effects
    }
    
    /// Check if this type has "interesting" answer types worth showing
    pub fn has_visible_effects(&self) -> bool {
        // Returns true if answer types are concrete (not just variables)
        // and differ from each other
    }
}
```

**Before:**
```
test : ()/ParseState -> ParseResult Int -> Int/ParseState -> ParseResult Int
```

**After:**
```
test : () -> Int  (requires parser context)
```

---

### 1.4 Improve Occurs Check Error

**Problem:** Occurs check errors are cryptic, especially when caused by recursion + shift.

**Solution:** Detect the common case and give a targeted error.

```rust
TypeError::OccursCheck { var_id, ty, span } => {
    let ty_str = ty.display_normalized();
    
    if ty.contains_answer_type_arrow() {
        // Likely the recursion + shift pattern
        write!(f, 
            "Infinite type detected.\n\n\
             This usually happens when a recursive function contains `shift`.\n\n\
             The type checker tried to create: {} = <type containing {}>\n\n\
             Consider using one of these patterns instead:\n\
             • Move recursion inside the shift body (loop-inside-shift)\n\
             • Use effect handlers (put recursion in the handler)\n\n\
             See: https://your-docs/patterns/recursion-with-effects",
            var_id, ty_str)
    } else {
        write!(f, "Infinite type: {} occurs in {}", var_id, ty_str)
    }
}
```

---

## 2. Forbid Shift in Recursive Bindings

**Problem:** Recursive functions containing `shift` cause infinite type errors that are confusing. This pattern fundamentally requires rank-2 answer-type polymorphism which isn't supported.

**Solution:** Detect this pattern early and give a clear error.

### 2.1 Add Syntactic Check

```rust
fn contains_shift(expr: &Expr) -> bool {
    match &expr.node {
        ExprKind::Shift { .. } => true,
        ExprKind::Lambda { body, .. } => contains_shift(body),
        ExprKind::App { func, arg } => contains_shift(func) || contains_shift(arg),
        ExprKind::Let { value, body, .. } => {
            contains_shift(value) || body.as_ref().map_or(false, contains_shift)
        }
        ExprKind::LetRec { bindings, body } => {
            bindings.iter().any(|b| contains_shift(&b.body)) ||
            body.as_ref().map_or(false, contains_shift)
        }
        ExprKind::If { cond, then_branch, else_branch } => {
            contains_shift(cond) || contains_shift(then_branch) || contains_shift(else_branch)
        }
        ExprKind::Match { scrutinee, arms } => {
            contains_shift(scrutinee) || arms.iter().any(|a| contains_shift(&a.body))
        }
        ExprKind::Reset(body) => false,  // Reset delimits - shift inside is OK
        ExprKind::Seq { first, second } => contains_shift(first) || contains_shift(second),
        ExprKind::Tuple(exprs) | ExprKind::List(exprs) => exprs.iter().any(contains_shift),
        ExprKind::BinOp { left, right, .. } => contains_shift(left) || contains_shift(right),
        ExprKind::UnaryOp { operand, .. } => contains_shift(operand),
        ExprKind::Constructor { args, .. } => args.iter().any(contains_shift),
        _ => false,
    }
}
```

### 2.2 Check in LetRec

```rust
ExprKind::LetRec { bindings, body } => {
    // Check for shift in recursive bindings BEFORE inference
    for binding in bindings {
        if contains_shift(&binding.body) {
            return Err(TypeError::ShiftInRecursiveBinding {
                name: binding.name.node.clone(),
                span: binding.name.span.clone(),
            });
        }
    }
    
    // ... rest of inference
}
```

### 2.3 New Error Type

```rust
#[error("recursive function with shift")]
ShiftInRecursiveBinding { name: String, span: Span },
```

### 2.4 Error Message

```
Error: Recursive function `many` contains `shift`

   |
42 | let rec many p =
   |         ^^^^ this recursive function
43 |     (fun () ->
44 |         let x = p () in
45 |         let xs = many p () in
   |                  ^^^^^^^^ recursive call
46 |         x :: xs
47 |     ) <|> (fun () -> [])
   |       ^^^ contains shift (via <|>)

This pattern requires answer-type polymorphism which isn't supported.

Instead, use one of these patterns:

1. Loop-inside-shift: Put recursion inside the shift body

   let many p =
       fun () ->
           shift (fun k ->
               let rec loop acc input =
                   match run_parser p input with
                   | Parsed v rest -> loop (v :: acc) rest
                   | ParseFail _ -> k (reverse acc) input
               in
               fun input -> loop [] input
           )

2. Effect handlers: Put recursion in the handler, not the effectful code

   See: https://your-docs/patterns/effect-handlers
```

---

## 3. Refactor: Unify Decl and Expr Inference

**Problem:** Let-binding inference logic is duplicated between `infer_decl` (for top-level) and `infer_expr_full` (for expressions).

**Solution:** Extract shared helpers.

### 3.1 Core Let-Binding Helper

```rust
/// Infer a let binding, returning the extended environment and inference result.
/// Shared between top-level declarations and let expressions.
fn infer_let_binding(
    &mut self,
    env: &TypeEnv,
    pattern: &Pattern,
    value: &Expr,
) -> Result<(TypeEnv, InferResult), TypeError> {
    self.level += 1;
    
    let is_recursive_fn = matches!(
        (&pattern.node, &value.node),
        (PatternKind::Var(_), ExprKind::Lambda { .. })
    );
    
    let value_result = if is_recursive_fn {
        if let PatternKind::Var(name) = &pattern.node {
            let mut recursive_env = env.clone();
            let preliminary_ty = self.fresh_var();
            recursive_env.insert(name.clone(), Scheme::mono(preliminary_ty.clone()));
            let inferred = self.infer_expr_full(&recursive_env, value)?;
            self.unify_at(&preliminary_ty, &inferred.ty, &value.span)?;
            inferred
        } else {
            unreachable!()
        }
    } else {
        self.infer_expr_full(env, value)?
    };
    
    self.level -= 1;
    
    let is_pure = value_result.is_pure();
    let scheme = if Self::is_syntactic_value(value) && is_pure {
        self.generalize(&value_result.ty)
    } else {
        Scheme::mono(value_result.ty.clone())
    };
    
    let mut new_env = env.clone();
    self.bind_pattern_scheme(&mut new_env, pattern, scheme)?;
    
    Ok((new_env, value_result))
}
```

### 3.2 Core Let-Rec Helper

```rust
/// Infer mutually recursive bindings, returning extended environment.
/// Shared between top-level let-rec and let-rec expressions.
fn infer_let_rec_bindings(
    &mut self,
    env: &TypeEnv,
    bindings: &[RecBinding],
) -> Result<(TypeEnv, Vec<InferResult>), TypeError> {
    // Check for shift in recursive bindings
    for binding in bindings {
        if contains_shift(&binding.body) {
            return Err(TypeError::ShiftInRecursiveBinding {
                name: binding.name.node.clone(),
                span: binding.name.span.clone(),
            });
        }
    }
    
    self.level += 1;
    
    // Step 1: Create preliminary types
    let mut rec_env = env.clone();
    let mut preliminary_types: Vec<Type> = Vec::new();
    
    for binding in bindings {
        let preliminary_ty = self.fresh_var();
        preliminary_types.push(preliminary_ty.clone());
        rec_env.insert(binding.name.node.clone(), Scheme::mono(preliminary_ty));
    }
    
    // Step 2: Infer each binding body
    let mut results = Vec::new();
    for binding in bindings {
        let result = self.infer_rec_binding_body(&rec_env, binding)?;
        results.push(result);
    }
    
    // Step 3: Unify with preliminary types
    for (i, (binding, result)) in bindings.iter().zip(results.iter()).enumerate() {
        self.unify_at(&preliminary_types[i], &result.ty, &binding.name.span)?;
    }
    
    self.level -= 1;
    
    // Step 4: Generalize and build final environment
    let mut new_env = env.clone();
    for (i, binding) in bindings.iter().enumerate() {
        let scheme = self.generalize(&preliminary_types[i]);
        new_env.insert(binding.name.node.clone(), scheme);
    }
    
    Ok((new_env, results))
}

/// Helper to infer a single recursive binding's body (builds lambda type from params)
fn infer_rec_binding_body(
    &mut self,
    env: &TypeEnv,
    binding: &RecBinding,
) -> Result<InferResult, TypeError> {
    let mut body_env = env.clone();
    let mut param_types = Vec::new();
    
    for param in &binding.params {
        let param_ty = self.fresh_var();
        self.bind_pattern(&mut body_env, param, &param_ty)?;
        param_types.push(param_ty);
    }
    
    let body_result = self.infer_expr_full(&body_env, &binding.body)?;
    
    // Build function type from params
    let func_ty = if param_types.is_empty() {
        body_result.ty.clone()
    } else {
        let mut ty = Type::Arrow {
            arg: Rc::new(param_types.pop().unwrap()),
            ret: Rc::new(body_result.ty.clone()),
            ans_in: Rc::new(body_result.answer_before.clone()),
            ans_out: Rc::new(body_result.answer_after.clone()),
        };
        
        while let Some(param_ty) = param_types.pop() {
            let pure_ans = self.fresh_var();
            ty = Type::Arrow {
                arg: Rc::new(param_ty),
                ret: Rc::new(ty),
                ans_in: Rc::new(pure_ans.clone()),
                ans_out: Rc::new(pure_ans),
            };
        }
        ty
    };
    
    Ok(InferResult {
        ty: func_ty,
        answer_before: body_result.answer_before,
        answer_after: body_result.answer_after,
    })
}
```

### 3.3 Simplified Expression Cases

```rust
ExprKind::Let { pattern, value, body } => {
    let (new_env, value_result) = self.infer_let_binding(env, pattern, value)?;
    
    match body {
        Some(body) => {
            let body_result = self.infer_expr_full(&new_env, body)?;
            self.unify_at(
                &value_result.answer_after, 
                &body_result.answer_before, 
                &body.span
            )?;
            Ok(InferResult {
                ty: body_result.ty,
                answer_before: value_result.answer_before,
                answer_after: body_result.answer_after,
            })
        }
        None => Ok(value_result),
    }
}

ExprKind::LetRec { bindings, body } => {
    let (new_env, _results) = self.infer_let_rec_bindings(env, bindings)?;
    
    match body {
        Some(body) => self.infer_expr_full(&new_env, body),
        None => Ok(InferResult::pure(Type::Unit, self.fresh_var())),
    }
}
```

### 3.4 Simplified Declaration Cases

```rust
Decl::Let { pattern, value, .. } => {
    let (new_env, _) = self.infer_let_binding(env, pattern, value)?;
    *env = new_env;
    Ok(())
}

Decl::LetRec { bindings, .. } => {
    let (new_env, _) = self.infer_let_rec_bindings(env, bindings)?;
    *env = new_env;
    Ok(())
}
```

---

## 4. Type Alias Handling

**Problem:** `type ParseState = List Char` creates a distinct type constructor instead of an alias, causing `[Char]` vs `ParseState` mismatches.

**Solution:** Either expand aliases during inference, or don't allow this syntax for aliases.

### Option A: Expand Aliases

```rust
// In TypeContext, store aliases separately
struct TypeContext {
    constructors: HashMap<String, ConstructorInfo>,
    type_aliases: HashMap<String, Type>,  // NEW
}

// When encountering a type name, check for alias first
fn resolve_type_name(&self, name: &str) -> Type {
    if let Some(aliased) = self.type_aliases.get(name) {
        aliased.clone()
    } else {
        Type::Constructor { name: name.to_string(), args: vec![] }
    }
}
```

### Option B: Different Syntax for Aliases

Require `type alias` keyword:

```
type alias ParseState = List Char    -- true alias, expands to List Char
type ParseResult a = Parsed a | Fail -- ADT, distinct type
```

---

## 5. Add InferResult Helper

**Problem:** Checking if an expression is pure requires comparing answer types manually.

**Solution:** Add helper method.

```rust
impl InferResult {
    pub fn pure(ty: Type, answer: Type) -> Self {
        InferResult {
            ty,
            answer_before: answer.clone(),
            answer_after: answer,
        }
    }
    
    /// Check if this result represents a pure expression (no effect on answer type)
    pub fn is_pure(&self) -> bool {
        let before = self.answer_before.resolve();
        let after = self.answer_after.resolve();
        match (&before, &after) {
            (Type::Var(v1), Type::Var(v2)) => Rc::ptr_eq(v1, v2),
            _ => types_equal(&before, &after),
        }
    }
}
```

---

## Summary: Priority Order

1. **High Priority (UX)**
   - Normalize type variables in errors
   - Hide answer types from users
   - Add unification context to errors

2. **High Priority (Safety)**
   - Forbid shift in recursive bindings with clear error

3. **Medium Priority (Code Quality)**
   - Refactor decl/expr to use shared helpers
   - Add `InferResult::is_pure()` helper

4. **Lower Priority (Nice to Have)**
   - Type alias handling
   - Improve occurs-check error for answer types
   - Add hints/suggestions to common errors

---

## Documentation to Write

1. **User Guide: Effect Patterns**
   - Effect handlers pattern (recommended)
   - Loop-inside-shift pattern
   - Why recursive shift doesn't work

2. **Error Reference**
   - "Recursive function contains shift" — what it means, how to fix
   - Common type mismatches with effects

3. **Internal Docs**
   - How answer types work
   - Why we don't have full answer-type polymorphism
