# Gneiss Compiler Design: Perceus-based Codegen

## Overview

This document describes the design for `gneic` - the Gneiss compiler that produces native executables via C codegen with Perceus reference counting.

**Key references:**
- [Perceus: Garbage Free Reference Counting with Reuse](https://www.microsoft.com/en-us/research/publication/perceus-garbage-free-reference-counting-with-reuse/) (PLDI 2021)
- [Reference Counting with Frame Limited Reuse](https://www.microsoft.com/en-us/research/publication/reference-counting-with-frame-limited-reuse/) (ICFP 2022)
- [Koka compiler](https://github.com/koka-lang/koka) - reference implementation

## Compilation Pipeline

```
Source (.gn)
    │
    ▼
┌─────────────────────────────────────┐
│  Frontend (existing)                │
│  - Lexer                            │
│  - Parser → AST                     │
│  - Type Inference → Typed AST       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  IR Lowering (new)                  │
│  - Desugar: pattern match → cases   │
│  - Desugar: let rec → fix           │
│  - ANF conversion                   │
│  - Lambda lifting                   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Core IR                            │
│  - Explicit allocation/tag          │
│  - Tail call annotations            │
│  - Effect evidence passing          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Perceus Transformation             │
│  - Ownership analysis               │
│  - Insert dup/drop                  │
│  - Reuse analysis                   │
│  - Drop specialization              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  C Codegen                          │
│  - Emit C source                    │
│  - Runtime library calls            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  C Compiler (gcc/clang)             │
└─────────────────────────────────────┘
    │
    ▼
Native executable
```

---

## Phase 1: Core IR Design

### Goals

1. **Explicit control flow** - No implicit sequencing
2. **Explicit allocation** - All heap allocation is visible
3. **ANF (A-Normal Form)** - All intermediate values are named
4. **Tail call annotation** - Mark tail positions for TCO

### IR Syntax

```rust
/// Core IR - A-Normal Form with explicit allocation
pub enum CoreExpr {
    /// Variable reference
    Var(VarId),

    /// Literal value (unboxed)
    Lit(Literal),

    /// Let binding: let x = e1 in e2
    Let {
        name: VarId,
        value: Box<CoreExpr>,
        body: Box<CoreExpr>,
    },

    /// Function application (non-tail)
    App {
        func: VarId,
        args: Vec<VarId>,
    },

    /// Tail call (tail position only)
    TailApp {
        func: VarId,
        args: Vec<VarId>,
    },

    /// Allocate constructor: Con tag [fields]
    Alloc {
        tag: Tag,
        fields: Vec<VarId>,
    },

    /// Pattern match / case analysis
    Case {
        scrutinee: VarId,
        alts: Vec<Alt>,
        default: Option<Box<CoreExpr>>,
    },

    /// Lambda (will be lambda-lifted to top-level)
    Lam {
        params: Vec<VarId>,
        body: Box<CoreExpr>,
    },

    /// Fixed-point for recursion
    Fix {
        name: VarId,
        params: Vec<VarId>,
        body: Box<CoreExpr>,
    },

    /// Effect perform
    Perform {
        effect: EffectId,
        op: OpId,
        args: Vec<VarId>,
    },

    /// Effect handler
    Handle {
        body: Box<CoreExpr>,
        handler: Handler,
    },
}

/// Case alternative
pub struct Alt {
    pub tag: Tag,
    pub binders: Vec<VarId>,
    pub body: CoreExpr,
}

/// Constructor tag (numeric, for fast dispatch)
pub type Tag = u32;
```

### Type Representation in IR

```rust
/// Types preserved through IR for size/layout info
pub enum CoreType {
    /// Unboxed integer
    Int,
    /// Unboxed float
    Float,
    /// Boxed reference (pointer to heap object)
    Box(Box<CoreType>),
    /// Function type
    Fun(Vec<CoreType>, Box<CoreType>),
    /// Sum type with variant tags
    Sum(Vec<(Tag, Vec<CoreType>)>),
    /// Product type (tuple/record)
    Prod(Vec<CoreType>),
}
```

---

## Phase 2: Perceus Transformation

### Overview

The Perceus algorithm inserts reference counting operations:
- **`dup(x)`** - Increment reference count
- **`drop(x)`** - Decrement reference count, free if zero

Key principles:
1. **Delay dup** - Push dup operations as late as possible
2. **Early drop** - Insert drop immediately after last use
3. **Ownership transfer** - Pass ownership instead of copying when possible

### Perceus IR (with RC ops)

```rust
pub enum PerceusExpr {
    // ... same as CoreExpr, plus:

    /// Duplicate (increment refcount)
    Dup {
        var: VarId,
        body: Box<PerceusExpr>,
    },

    /// Drop (decrement refcount, may free)
    Drop {
        var: VarId,
        body: Box<PerceusExpr>,
    },

    /// Reuse: allocate reusing memory from dropped object
    Reuse {
        reuse_token: VarId,  // from a drop
        tag: Tag,
        fields: Vec<VarId>,
    },
}
```

### Ownership Analysis

For each variable use, determine:
1. **Owned** - We have the only reference, can consume
2. **Borrowed** - We're using someone else's reference, must dup
3. **Returned** - We're returning it, transfer ownership to caller

```rust
pub enum Ownership {
    Owned,      // We own this, must drop or transfer
    Borrowed,   // We borrowed this, must dup if keeping
    Returned,   // Transfer to caller
}
```

### Reuse Analysis

When we drop a value and immediately allocate one of the same size, we can reuse the memory:

```rust
// Before reuse analysis:
let xs = drop xs in
let ys = Alloc Cons [y, ys'] in ...

// After reuse analysis:
let token = drop_reuse xs in
let ys = Reuse token Cons [y, ys'] in ...
```

---

## Phase 3: C Codegen

### Value Representation

```c
// All values are tagged pointers or immediates
typedef uint64_t gn_value;

// Immediate integers (tagged with low bit = 1)
#define GN_INT(n)      (((gn_value)(n) << 1) | 1)
#define GN_UNINT(v)    ((int64_t)(v) >> 1)
#define GN_IS_INT(v)   ((v) & 1)

// Heap objects (aligned, so low bit = 0)
typedef struct gn_object {
    uint32_t rc;      // Reference count
    uint32_t tag;     // Constructor tag
    gn_value fields[]; // Flexible array of fields
} gn_object;
```

### Reference Counting Primitives

```c
static inline gn_value gn_dup(gn_value v) {
    if (!GN_IS_INT(v)) {
        gn_object* obj = (gn_object*)v;
        obj->rc++;
    }
    return v;
}

static inline void gn_drop(gn_value v) {
    if (!GN_IS_INT(v)) {
        gn_object* obj = (gn_object*)v;
        if (--obj->rc == 0) {
            gn_free_recursive(obj);
        }
    }
}

// Reuse: check if unique and same size
static inline gn_object* gn_drop_reuse(gn_value v, size_t fields) {
    if (GN_IS_INT(v)) return NULL;
    gn_object* obj = (gn_object*)v;
    if (obj->rc == 1 && gn_obj_fields(obj) == fields) {
        // Unique! Can reuse
        return obj;
    } else {
        gn_drop(v);
        return NULL;
    }
}
```

### Function Representation

```c
// Closures: function pointer + captured environment
typedef struct gn_closure {
    uint32_t rc;
    uint32_t arity;
    gn_value (*func)(gn_value* env, gn_value* args);
    gn_value env[];  // Captured variables
} gn_closure;

// Top-level functions (no captures)
gn_value gn_add(gn_value* env, gn_value* args) {
    int64_t a = GN_UNINT(args[0]);
    int64_t b = GN_UNINT(args[1]);
    return GN_INT(a + b);
}
```

### Effect Handlers in C

Effects use setjmp/longjmp for control flow:

```c
typedef struct gn_handler {
    jmp_buf resume_point;
    gn_value result;
    // Handler closures for each operation
    gn_closure* ops[];
} gn_handler;

// Perform jumps to innermost handler
gn_value gn_perform(int effect, int op, gn_value arg) {
    gn_handler* h = gn_current_handler(effect);
    h->op_arg = arg;
    h->op_id = op;
    longjmp(h->resume_point, 1);
}
```

---

## Phase 4: Runtime Library

### Core Components

| Module | Purpose |
|--------|---------|
| `gn_alloc.c` | Memory allocation, GC-free allocator |
| `gn_rc.c` | Reference counting primitives |
| `gn_value.c` | Value representation, tagging |
| `gn_effect.c` | Effect handler machinery |
| `gn_io.c` | IO operations |
| `gn_string.c` | String operations |
| `gn_panic.c` | Runtime errors |

### Memory Allocator

Options:
1. **System malloc** - Simple, portable
2. **Bump allocator + arenas** - Fast allocation
3. **Size-segregated pools** - Good for reuse

Start with system malloc, optimize later.

---

## Implementation Plan

### Milestone 1: Minimal Pipeline (Weeks 1-2)
- [ ] Core IR data structures
- [ ] AST → Core IR lowering (subset: Int, +, let, functions)
- [ ] Core IR → C codegen (no RC yet)
- [ ] Basic runtime (value representation)
- [ ] End-to-end test: `1 + 2` compiles and runs

### Milestone 2: Data Types (Weeks 3-4)
- [ ] ADT lowering (Option, List, etc.)
- [ ] Pattern matching → Case
- [ ] Tag-based dispatch in C
- [ ] Recursive types work

### Milestone 3: Perceus (Weeks 5-6)
- [ ] Ownership analysis
- [ ] Dup/drop insertion
- [ ] Verify garbage-free property
- [ ] Basic reuse analysis

### Milestone 4: Effects (Weeks 7-8)
- [ ] Effect evidence passing
- [ ] setjmp/longjmp handler implementation
- [ ] Multi-shot continuations (if needed)

### Milestone 5: Optimization (Week 9+)
- [ ] Lambda lifting optimization
- [ ] Inlining
- [ ] Drop specialization
- [ ] Frame-limited reuse

---

## File Structure

```
src/
  codegen/
    mod.rs          -- Module re-exports
    core_ir.rs      -- Core IR definitions
    lower.rs        -- AST → Core IR
    perceus.rs      -- RC insertion
    reuse.rs        -- Reuse analysis
    c_emit.rs       -- C code generation

runtime/
  kklib/            -- (or gnlib/) C runtime library
    gn_value.h
    gn_rc.h
    gn_alloc.c
    gn_effect.c
    gn_io.c
    gn_main.c       -- Entry point wrapper
```

---

## Open Questions

1. **Effect representation**: Koka uses "evidence passing" for effects. Should we copy this or use simpler approach?

2. **Multi-shot continuations**: Do we need them? Simpler to start without.

3. **Closures**: Lambda-lift everything, or support heap-allocated closures?

4. **Interop**: Call C functions from Gneiss? FFI design?

5. **Modules**: Separate compilation? Link-time optimization?

---

## References

- [Koka compiler source](https://github.com/koka-lang/koka)
- [Perceus paper](https://xnning.github.io/papers/perceus.pdf)
- [Frame Limited Reuse](https://www.microsoft.com/en-us/research/publication/reference-counting-with-frame-limited-reuse/)
- [Compiling with Continuations](https://www.cs.princeton.edu/~appel/papers/cwc.html) - CPS/ANF background
