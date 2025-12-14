# Issue #006: REPL Never Shows Inferred Types for Definitions

## Summary
The REPL's `TypeEnvExt::bindings()` method is a stub that always returns an empty iterator, so the REPL never displays the inferred types when you define new bindings.

## Severity
**LOW** - Cosmetic/UX issue only. Type inference still works; it just doesn't display.

## Location
- **File:** `src/main.rs`
- **Trait impl:** `TypeEnvExt for TypeEnv` (approximately lines 259-268)

## Current Broken Code
```rust
// Helper trait to access TypeEnv internals for REPL
trait TypeEnvExt {
    fn bindings(&self) -> impl Iterator<Item = (&String, &gneiss::types::Scheme)>;
}

impl TypeEnvExt for TypeEnv {
    fn bindings(&self) -> impl Iterator<Item = (&String, &gneiss::types::Scheme)> {
        // This is a bit of a hack - in a real implementation we'd expose this properly
        std::iter::empty()  // BUG: Always returns empty!
    }
}
```

## The Problem

In the REPL loop:
```rust
match inferencer.infer_program(&program) {
    Ok(new_env) => {
        for (name, scheme) in new_env.bindings() {  // Always empty!
            type_env.insert(name.clone(), scheme.clone());
            println!("{} : {}", name, scheme);  // Never prints
        }
    }
    // ...
}
```

When you define something in the REPL:
```
gneiss> let add x y = x + y
gneiss>   <-- No type shown!
```

Expected:
```
gneiss> let add x y = x + y
add : Int -> Int -> Int
gneiss>
```

## Required Fix

### Option 1: Add a `bindings()` method to `TypeEnv` in `types.rs`

In `src/types.rs`, add a method to `TypeEnv`:

```rust
impl TypeEnv {
    // ... existing methods ...

    /// Iterate over all bindings in this environment
    pub fn bindings(&self) -> impl Iterator<Item = (&String, &Scheme)> {
        self.bindings.iter()
    }
}
```

Then in `src/main.rs`, simplify:

```rust
impl TypeEnvExt for TypeEnv {
    fn bindings(&self) -> impl Iterator<Item = (&String, &gneiss::types::Scheme)> {
        TypeEnv::bindings(self)  // Or just remove the trait entirely
    }
}
```

Or just remove the `TypeEnvExt` trait and call `new_env.bindings()` directly if you make the method public.

### Option 2: Fix the trait impl directly in main.rs

If you don't want to modify `types.rs`, you can access the private field through the module system by re-exporting it, but that's messier. The clean solution is Option 1.

### Complete Fix (Option 1)

**File: `src/types.rs`**

Add this method to the `impl TypeEnv` block:

```rust
impl TypeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: String, scheme: Scheme) {
        self.bindings.insert(name, scheme);
    }

    pub fn get(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name)
    }

    pub fn extend(&self, name: String, scheme: Scheme) -> TypeEnv {
        let mut new_env = self.clone();
        new_env.insert(name, scheme);
        new_env
    }

    // ADD THIS METHOD:
    /// Iterate over all bindings in this environment
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Scheme)> {
        self.bindings.iter()
    }
}
```

**File: `src/main.rs`**

Update the trait implementation:

```rust
impl TypeEnvExt for TypeEnv {
    fn bindings(&self) -> impl Iterator<Item = (&String, &gneiss::types::Scheme)> {
        self.iter()
    }
}
```

Or remove the trait entirely and just use `new_env.iter()` directly:

```rust
match inferencer.infer_program(&program) {
    Ok(new_env) => {
        for (name, scheme) in new_env.iter() {
            type_env.insert(name.clone(), scheme.clone());
            println!("{} : {}", name, scheme);
        }
    }
    Err(e) => {
        eprintln!("Type error: {}", e);
        continue;
    }
}
```

## Test

Manual testing in the REPL:

```bash
cargo run
```

```
gneiss> let id x = x
id : forall a. a -> a
gneiss> let add x y = x + y
add : Int -> Int -> Int
gneiss> let always x y = x
always : forall a b. a -> b -> a
gneiss>
```

## Verification
```bash
cargo build
cargo run
# Then test in REPL:
# let f x = x
# Should print: f : forall a. a -> a
```
