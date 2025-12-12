# Delimited Continuation Patterns in Gneiss

This directory contains examples demonstrating powerful programming patterns enabled by delimited continuations (`shift`/`reset`).

## Quick Reference

| Pattern | File | Key Insight |
|---------|------|-------------|
| **Generators** | `pattern_generators.gn` | `yield` captures "rest of generator" |
| **Nondeterminism** | `pattern_nondeterminism.gn` | `choose` runs continuation multiple times |
| **State** | `pattern_state.gn` | Thread state through continuation |
| **Coroutines** | `pattern_coroutines.gn` | `suspend` captures "rest of coroutine" |
| **Exceptions** | `pattern_exceptions.gn` | `throw` discards continuation |
| **Backtracking** | `pattern_backtracking.gn` | `cut` commits, `fail` backtracks |
| **Parsers** | `pattern_parsers.gn` | Backtracking + state = parser combinators |
| **Effect Handlers** | `pattern_effect_handlers.gn` | Unified view of all patterns |
| **Answer Types** | `pattern_answer_types.gn` | Type-changing continuations (printf) |

## The Core Mechanism

All patterns use the same fundamental mechanism:

```gneiss
reset (
    ...
    shift (fun k -> ...)  -- k = "rest of the reset body"
    ...
)
```

- `shift` captures the continuation `k` (everything after the shift, up to the enclosing reset)
- The shift body decides what to do: call `k`, ignore it, call it multiple times, etc.
- `reset` delimits the scope of the captured continuation

## Pattern Categories

### 1. Value Production Patterns

**Generators** (`yield`): Produce values incrementally
```gneiss
let yield x = shift (fun k -> x :: k ())
reset (yield 1; yield 2; yield 3; [])  -- [1, 2, 3]
```

**Nondeterminism** (`choose`): Explore all alternatives
```gneiss
let choose xs = shift (fun k -> concat_map k xs)
reset (
    let x = choose [1, 2] in
    let y = choose [10, 20] in
    [x + y]
)  -- [11, 21, 12, 22]
```

### 2. Control Flow Patterns

**Exceptions** (`throw`): Abort computation
```gneiss
let throw err = shift (fun _k -> Err err)  -- discard k!
try_catch (fun () -> if x < 0 then throw "negative" else x) handle
```

**Early Exit**: Return from nested context
```gneiss
let escape result = shift (fun _k -> result)
reset (
    let x = compute () in
    if x < 0 then escape 0 else continue x
)
```

### 3. State Patterns

**Mutable State**: Thread state through computation
```gneiss
let get () = shift (fun k -> fun s -> k s s)
let put s' = shift (fun k -> fun _ -> k () s')
run_state (fun () -> let x = get () in put (x + 1)) initial
```

**Coroutines**: Suspend and resume
```gneiss
let suspend () = shift (fun k -> Suspended k)
-- Later: resume by calling the continuation
```

### 4. Search Patterns

**Backtracking**: Explore and prune
```gneiss
let fail () = shift (fun _ -> [])
let cut () = shift (fun k -> reset (k ()))  -- commit!
```

**Parsers**: Backtracking + input state
```gneiss
let (<|>) p1 p2 = shift (fun k -> 
    match p1 () with 
    | Success -> k result 
    | Fail -> p2 ()
)
```

### 5. Advanced Type Patterns

**Printf**: Answer type changes per format specifier
```gneiss
-- %s changes answer from String to (String -> String)
let format_string rest = shift (fun k -> fun s -> k (s ^ rest))
printf (FStr (FLit "!" FEnd))  -- : String -> String
```

**Continuation Composition**: Build pipelines
```gneiss
let double () = shift (fun k -> fun xs -> k (map (*2) xs))
let filter_pos () = shift (fun k -> fun xs -> k (filter (>0) xs))
reset (double (); filter_pos (); id)  -- composed pipeline
```

## Effect Handlers: The Unified View

Effect handlers generalize ALL these patterns:

```gneiss
-- Any effect is just: identify operation, capture continuation
let perform op = shift (fun k -> Op op k)

-- Any handler decides how to interpret the operation
let handle comp { op -> handler_code }
```

| Pattern | Effect Operation | Handler Behavior |
|---------|------------------|------------------|
| Generators | `Yield x` | Collect x, resume k |
| Exceptions | `Throw e` | Return error, discard k |
| State | `Get` / `Put s` | Thread state through k |
| Nondeterminism | `Choose xs` | Run k for each x |
| Async | `Await promise` | Register k as callback |

## Type System Notes

Gneiss uses **answer-type modification** to type these patterns correctly:

- Function type: `σ/α → τ/β` means "takes σ, returns τ, changes answer type from α to β"
- Pure functions: `α = β` (answer type unchanged)
- Continuations are polymorphic: `k : ∀t. τ/t → α/t`

This allows:
- `reset (1 + shift (fun k -> "hello"))` : String (not Int!)
- Printf to return different function types based on format
- Generators to have different yield and final types

## Running the Examples

```bash
gneiss run examples/pattern_generators.gn
gneiss run examples/pattern_nondeterminism.gn
# etc.
```

## Further Reading

- Danvy & Filinski, "Abstracting Control" (1990) - foundational paper
- Asai & Kameyama, "Polymorphic Delimited Continuations" (2007) - type system
- Plotkin & Pretnar, "Handlers of Algebraic Effects" (2009) - effect handlers
- Leijen, "Type Directed Compilation of Row-typed Algebraic Effects" (2017) - Koka language
