# Algebraic Effect Parser Combinators: Implementation Guide

A comprehensive guide to building a type-sound parser combinator library using algebraic effects, including formal foundations, implementation patterns, and verification strategies.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Algebraic Effects Foundation](#2-algebraic-effects-foundation)
3. [Parser Effect Design](#3-parser-effect-design)
4. [Type System](#4-type-system)
5. [Implementation](#5-implementation)
6. [Proving Type Soundness](#6-proving-type-soundness)
7. [Effect Laws and Verification](#7-effect-laws-and-verification)
8. [Property Testing](#8-property-testing)
9. [Advanced Topics](#9-advanced-topics)
10. [Complete Reference Implementation](#10-complete-reference-implementation)

---

## 1. Introduction

### 1.1 Why Algebraic Effects for Parsing?

Traditional parser combinator libraries face a fundamental tension: they must handle multiple concerns (state management, failure, backtracking, error reporting) while maintaining composability. Algebraic effects provide an elegant solution by separating the *syntax* of effectful operations from their *semantics*.

**Benefits:**
- **Modularity**: Effects are declared separately from their interpretation
- **Composability**: Handlers can be layered and combined
- **Backtracking control**: Fine-grained control over when to commit or retry
- **Multiple interpretations**: Same parser can be run with different handlers (e.g., with/without tracing)

### 1.2 What This Guide Covers

This guide will help you:
1. Design a clean effect signature for parsing
2. Implement handlers that respect algebraic laws
3. Prove type soundness for your implementation
4. Verify correctness through property testing
5. Extend the library with advanced features

### 1.3 Prerequisites

Familiarity with:
- Basic parsing concepts (grammars, combinators)
- Algebraic data types and pattern matching
- Effect systems (conceptually)
- Basic type theory (helpful but not required)

---

## 2. Algebraic Effects Foundation

### 2.1 What Are Algebraic Effects?

An algebraic effect consists of:
1. **A signature Σ** — A set of *operations* with specified arities
2. **An equational theory E** — Laws the operations must satisfy
3. **Handlers** — Interpretations that give meaning to operations

The key insight: operations are *syntax*, handlers provide *semantics*.

### 2.2 The Free Monad Perspective

Given a signature Σ, computations form a *free monad*:

```
type Comp ε a =
    | Pure a
    | Op (op : Operation) (args : Args op) (k : Result op → Comp ε a)
```

A computation is either:
- A pure value, or
- An operation invocation with a continuation

**Handlers** are Σ-algebras that interpret operations:

```
handler : {
    return : a → r,
    op₁ : Args₁ → (Result₁ → r) → r,
    op₂ : Args₂ → (Result₂ → r) → r,
    ...
} → Comp Σ a → r
```

### 2.3 Effect Rows

Modern effect systems use *row polymorphism* to track which effects a computation may perform:

```
-- This function may perform State and Fail effects, plus any others in ε
get_and_check : () → Int ! {State Int, Fail | ε}
```

The row variable `ε` enables effect polymorphism—functions can be generic over additional effects.

---

## 3. Parser Effect Design

### 3.1 Core Effects

A parser combinator library needs three fundamental capabilities:

```
-- 1. STATE: Track position in input
effect State s =
    | get : () → s
    | put : s → ()
end

-- 2. FAILURE: Signal parse failures
effect Fail =
    | fail : String → a
end

-- 3. CHOICE: Try alternatives (implicit in handler design)
```

### 3.2 Parser State

Define what state the parser tracks:

```
type ParserState = {
    input  : String,      -- Original input (immutable)
    pos    : Int,         -- Current position
    line   : Int,         -- Current line (for error messages)
    column : Int          -- Current column
}
```

### 3.3 Derived Parser Effect

Build a higher-level parser effect from primitives:

```
effect Parser =
    -- Input observation
    | peek      : () → String       -- Remaining input from current position
    | peek_char : () → Option Char  -- Next character, if any
    
    -- Input consumption
    | advance   : Int → ()          -- Move position forward
    | consume   : () → Char         -- Consume and return one character
    
    -- Failure
    | fail_parse : String → a       -- Fail with message
    
    -- Position tracking
    | get_pos   : () → Int          -- Current position
    | get_loc   : () → (Int, Int)   -- (line, column)
    
    -- Backtracking control
    | mark      : () → Int          -- Save current position
    | restore   : Int → ()          -- Restore to saved position
    | commit    : () → ()           -- Prevent backtracking past this point
end
```

### 3.4 Design Rationale

**Why separate `peek` from `consume`?**
- `peek` is pure observation—calling it twice returns the same result
- `consume` advances state—has a visible side effect
- This distinction matters for laws and optimization

**Why explicit `mark`/`restore`?**
- Makes backtracking explicit and controllable
- Enables `try` combinator implementation
- Clearer semantics than implicit state restoration

**Why `commit`?**
- Prevents exponential backtracking in ambiguous grammars
- Essential for efficient PEG-style parsing
- Gives programmer control over error recovery points

---

## 4. Type System

### 4.1 Syntax

```
Types:
    τ ::= α                         -- Type variable
        | τ → τ ! ε                 -- Function type with effects
        | τ × τ                     -- Product
        | τ + τ                     -- Sum
        | T τ₁ ... τₙ               -- Type constructor application
        | ∀α. τ                     -- Universal quantification
        | ∀ε. τ                     -- Effect quantification

Effect Rows:
    ε ::= {}                        -- Empty row (pure)
        | {op | ε}                  -- Row extension
        | ε                         -- Row variable

Expressions:
    e ::= x                         -- Variable
        | λx. e                     -- Abstraction
        | e e                       -- Application
        | perform op e              -- Effect operation
        | handle e with h           -- Handler application
        | return e                  -- Pure value injection

Handlers:
    h ::= { return x → e; op₁ x k → e₁; ... }
```

### 4.2 Typing Rules

#### Variables and Abstraction

```
    x : τ ∈ Γ
    ─────────── (Var)
    Γ ⊢ x : τ ! ε


    Γ, x : τ₁ ⊢ e : τ₂ ! ε
    ─────────────────────────────── (Abs)
    Γ ⊢ λx. e : (τ₁ → τ₂ ! ε) ! {}


    Γ ⊢ e₁ : (τ₁ → τ₂ ! ε) ! ε    Γ ⊢ e₂ : τ₁ ! ε
    ──────────────────────────────────────────────── (App)
    Γ ⊢ e₁ e₂ : τ₂ ! ε
```

#### Effect Operations

```
    op : τ_arg → τ_ret ∈ E    Γ ⊢ e : τ_arg ! ε    E ∈ ε
    ────────────────────────────────────────────────────── (Perform)
    Γ ⊢ perform op e : τ_ret ! ε
```

#### Handlers

```
    Γ ⊢ e : τ ! {E | ε}
    
    Γ, x : τ ⊢ e_ret : τ' ! ε'
    
    For each op : τ_arg → τ_ret in E:
        Γ, x : τ_arg, k : (τ_ret → τ' ! ε') ⊢ e_op : τ' ! ε'
    ────────────────────────────────────────────────────────── (Handle)
    Γ ⊢ handle e with {
            return x → e_ret;
            op x k → e_op; ...
        } : τ' ! ε'
```

### 4.3 Effect Subtyping

Effects support subtyping via row inclusion:

```
    ε ⊆ ε'    Γ ⊢ e : τ ! ε
    ─────────────────────── (Sub)
    Γ ⊢ e : τ ! ε'
```

This allows a pure computation to be used where an effectful one is expected.

### 4.4 Parser-Specific Types

```
-- A parser that produces values of type a
type Parser a = () → a ! {Parser}

-- Alternative: make the effect row explicit
type Parser ε a = () → a ! {Parser | ε}

-- Result type for running parsers
type ParseResult a =
    | ParseOk a String Int        -- value, remaining input, final position
    | ParseFail String Int Int    -- message, position, line
```

---

## 5. Implementation

### 5.1 Core Primitives

```
-- Peek at remaining input without consuming
let peek () : String ! {Parser} =
    perform Parser.peek ()

-- Peek at next character
let peek_char () : Option Char ! {Parser} =
    let remaining = peek () in
    if string_length remaining == 0 then
        None
    else
        Some (string_head remaining)

-- Consume one character
let consume () : Char ! {Parser} =
    match peek_char () with
    | None → perform Parser.fail_parse "Unexpected end of input"
    | Some c →
        perform Parser.advance 1;
        c
    end

-- Get current position
let position () : Int ! {Parser} =
    perform Parser.get_pos ()

-- Fail with message
let fail msg : a ! {Parser} =
    perform Parser.fail_parse msg
```

### 5.2 Basic Combinators

```
-- Parse a character satisfying a predicate
let satisfy (pred : Char → Bool) (expected : String) : Char ! {Parser} =
    match peek_char () with
    | None → fail ("Expected " ++ expected ++ ", got end of input")
    | Some c →
        if pred c then
            (perform Parser.advance 1; c)
        else
            fail ("Expected " ++ expected ++ ", got '" ++ char_to_string c ++ "'")
    end

-- Parse a specific character
let char (c : Char) : Char ! {Parser} =
    satisfy (fun x → x == c) ("'" ++ char_to_string c ++ "'")

-- Parse a specific string
let string (s : String) : String ! {Parser} =
    let len = string_length s in
    let remaining = peek () in
    if string_take len remaining == s then
        (perform Parser.advance len; s)
    else
        fail ("Expected \"" ++ s ++ "\"")

-- Parse end of input
let eof () : () ! {Parser} =
    let remaining = peek () in
    if string_length remaining == 0 then
        ()
    else
        fail "Expected end of input"
```

### 5.3 Sequencing Combinators

```
-- Monadic bind
let (>>=) (p : Parser a) (f : a → Parser b) : Parser b =
    fun () →
        let x = p () in
        f x ()

-- Sequence, keep right
let (>>) (p1 : Parser a) (p2 : Parser b) : Parser b =
    p1 >>= (fun _ → p2)

-- Sequence, keep left
let (<<) (p1 : Parser a) (p2 : Parser b) : Parser a =
    p1 >>= (fun x → p2 >> return_parser x)

-- Pure value in parser context
let return_parser (x : a) : Parser a =
    fun () → x

-- Applicative operations
let (<$>) (f : a → b) (p : Parser a) : Parser b =
    p >>= (fun x → return_parser (f x))

let (<*>) (pf : Parser (a → b)) (pa : Parser a) : Parser b =
    pf >>= (fun f → pa >>= (fun a → return_parser (f a)))
```

### 5.4 Choice and Backtracking

```
-- Try parser, restore position on failure
let try (p : Parser a) : Parser a =
    fun () →
        let saved = perform Parser.mark () in
        handle p () with
        | return x → x
        | Parser.fail_parse msg k →
            perform Parser.restore saved;
            perform Parser.fail_parse msg
        end

-- Choice: try first parser, if it fails without consuming, try second
let (<|>) (p1 : Parser a) (p2 : Parser a) : Parser a =
    fun () →
        let start_pos = perform Parser.get_pos () in
        handle p1 () with
        | return x → x
        | Parser.fail_parse msg k →
            let current_pos = perform Parser.get_pos () in
            if current_pos == start_pos then
                -- No input consumed, try alternative
                p2 ()
            else
                -- Input was consumed, propagate failure
                perform Parser.fail_parse msg
        end

-- Choice with explicit backtracking
let (<|?>) (p1 : Parser a) (p2 : Parser a) : Parser a =
    try p1 <|> p2

-- Commit: prevent backtracking past this point
let commit () : () ! {Parser} =
    perform Parser.commit ()

-- Parse with commit on success
let committed (p : Parser a) : Parser a =
    p >>= (fun x → commit (); return_parser x)
```

### 5.5 Repetition Combinators

```
-- Zero or more
let rec many (p : Parser a) : Parser (List a) =
    (p >>= fun x →
     many p >>= fun xs →
     return_parser (x :: xs))
    <|>
    return_parser []

-- One or more
let many1 (p : Parser a) : Parser (List a) =
    p >>= fun x →
    many p >>= fun xs →
    return_parser (x :: xs)

-- Exactly n times
let rec count (n : Int) (p : Parser a) : Parser (List a) =
    if n <= 0 then
        return_parser []
    else
        p >>= fun x →
        count (n - 1) p >>= fun xs →
        return_parser (x :: xs)

-- Optional
let optional (p : Parser a) : Parser (Option a) =
    (p >>= fun x → return_parser (Some x)) <|> return_parser None

-- Separated by
let sep_by (p : Parser a) (sep : Parser b) : Parser (List a) =
    (p >>= fun x →
     many (sep >> p) >>= fun xs →
     return_parser (x :: xs))
    <|>
    return_parser []

let sep_by1 (p : Parser a) (sep : Parser b) : Parser (List a) =
    p >>= fun x →
    many (sep >> p) >>= fun xs →
    return_parser (x :: xs)
```

### 5.6 The Main Handler

```
-- Run a parser on input
let run_parser (p : Parser a) (input : String) : ParseResult a =
    let initial_state = {
        input = input,
        pos = 0,
        line = 1,
        column = 1
    } in
    
    -- State handler (inner)
    let with_state comp =
        handle comp with
        | return x → fun st → (x, st)
        | Parser.peek () k → fun st →
            k (string_drop st.pos st.input) st
        | Parser.advance n k → fun st →
            let new_pos = st.pos + n in
            let (new_line, new_col) = update_position st n in
            k () { st with pos = new_pos, line = new_line, column = new_col }
        | Parser.get_pos () k → fun st →
            k st.pos st
        | Parser.get_loc () k → fun st →
            k (st.line, st.column) st
        | Parser.mark () k → fun st →
            k st.pos st
        | Parser.restore saved_pos k → fun st →
            k () { st with pos = saved_pos }  -- Note: line/col would need recalculation
        | Parser.commit () k → fun st →
            k () st  -- Commit is handled by choice combinator
        | Parser.fail_parse msg k → fun st →
            -- Propagate failure with position info
            (Error (msg, st.pos, st.line), st)
        end
    in
    
    -- Run with initial state
    match with_state (p ()) initial_state with
    | (Error (msg, pos, line), _) → ParseFail msg pos line
    | (x, final_state) → 
        ParseOk x (string_drop final_state.pos input) final_state.pos
    end

-- Helper: update line/column after advancing
let update_position (st : ParserState) (n : Int) : (Int, Int) =
    let consumed = string_substring st.pos (st.pos + n) st.input in
    let newlines = string_count '\n' consumed in
    if newlines == 0 then
        (st.line, st.column + n)
    else
        let last_newline = string_last_index_of '\n' consumed in
        (st.line + newlines, n - last_newline)
```

---

## 6. Proving Type Soundness

### 6.1 Overview

Type soundness means "well-typed programs don't go wrong." We prove this via:
1. **Progress**: A well-typed term either is a value, can step, or is waiting for a handler
2. **Preservation**: Reduction preserves typing

### 6.2 Operational Semantics

Define small-step reduction `e ↦ e'`:

```
-- Values
v ::= λx. e | (v, v) | C v | ...

-- Evaluation contexts
E ::= [] 
    | E e                           -- Application, left
    | v E                           -- Application, right
    | perform op E                  -- Effect argument
    | handle E with h               -- Handle body

-- Core reduction rules
(λx. e) v                           ↦  e[v/x]                    (β)
handle (return v) with h            ↦  e_ret[v/x]                (Ret)
handle E[perform op v] with h       ↦  e_op[v/x, λy. handle E[y] with h / k]  (Op)
    where h = { return x → e_ret; op x k → e_op; ... }
    and op ∈ h
```

The `(Op)` rule is crucial: when an operation reaches its handler:
1. The argument `v` is bound to `x` in the handler clause
2. The *delimited continuation* `E` (everything between the perform and the handle) is captured
3. This continuation is wrapped as `λy. handle E[y] with h` and bound to `k`

### 6.3 Type Preservation Theorem

**Theorem (Preservation)**: If `Γ ⊢ e : τ ! ε` and `e ↦ e'`, then `Γ ⊢ e' : τ ! ε`.

**Proof sketch** (by cases on the reduction rule):

**Case (β)**: `(λx. e) v ↦ e[v/x]`
- By inversion on application typing: `Γ ⊢ λx. e : (τ₁ → τ₂ ! ε) ! ε'` and `Γ ⊢ v : τ₁ ! ε'`
- By inversion on abstraction typing: `Γ, x : τ₁ ⊢ e : τ₂ ! ε`
- By substitution lemma: `Γ ⊢ e[v/x] : τ₂ ! ε` ∎

**Case (Ret)**: `handle (return v) with h ↦ e_ret[v/x]`
- By inversion: `Γ ⊢ v : τ ! {}` and `Γ, x : τ ⊢ e_ret : τ' ! ε'`
- By substitution lemma: `Γ ⊢ e_ret[v/x] : τ' ! ε'` ∎

**Case (Op)**: `handle E[perform op v] with h ↦ e_op[v/x, (λy. handle E[y] with h)/k]`
- This requires showing the continuation has the right type
- Key lemma: If `Γ ⊢ E[e] : τ ! ε` and `Γ ⊢ e : τ_e ! ε`, then there exists `τ_hole` such that `Γ ⊢ E : τ_hole → τ` (context typing)
- The continuation `λy. handle E[y] with h` has type `τ_ret → τ' ! ε'` ∎

### 6.4 Progress Theorem

**Theorem (Progress)**: If `∅ ⊢ e : τ ! ε`, then either:
1. `e` is a value, or
2. `e ↦ e'` for some `e'`, or
3. `e = E[perform op v]` where `op ∉ handled(E)` (unhandled effect)

**Proof sketch** (by induction on typing derivation):

- **Variable**: Impossible (empty context)
- **Abstraction**: Already a value
- **Application**: By IH on subterms; if both values and left is lambda, then β-reduce
- **Perform**: Either inside a handler (can reduce) or not (case 3)
- **Handle**: By IH on body; if body is value, use (Ret); if body performs handled op, use (Op)

### 6.5 Effect Safety

**Theorem (Effect Safety)**: If `∅ ⊢ e : τ ! {}`, then `e` cannot get stuck on an unhandled effect.

**Proof**: By Progress, case 3 requires `op ∉ handled(E)` for some effect op. But the typing `! {}` means no effects are permitted, so the perform would be ill-typed. ∎

### 6.6 Key Lemmas

You'll need these supporting lemmas:

**Substitution Lemma**: If `Γ, x : τ₁ ⊢ e : τ₂ ! ε` and `Γ ⊢ v : τ₁ ! {}`, then `Γ ⊢ e[v/x] : τ₂ ! ε`.

**Weakening**: If `Γ ⊢ e : τ ! ε` and `x ∉ dom(Γ)`, then `Γ, x : τ' ⊢ e : τ ! ε`.

**Context Typing**: Evaluation contexts can be assigned "function" types from hole type to result type.

**Canonical Forms**: Values of function type are lambdas; values of sum type are injections; etc.

---

## 7. Effect Laws and Verification

### 7.1 State Effect Laws

The state operations must satisfy these equations:

```
-- Get-Get: Two consecutive gets return the same value
get (); (fun s → get (); k s)  ≡  get (); (fun s → k s s)

-- Put-Get: Get after put returns the put value
put s; get (); k  ≡  put s; k s

-- Put-Put: Consecutive puts, only last matters
put s; put s'; k  ≡  put s'; k

-- Get-Put: Getting then putting same value is identity
get (); (fun s → put s; k)  ≡  k
```

### 7.2 Fail Effect Laws

```
-- Fail-Discard: Failure discards its continuation
fail msg; k  ≡  fail msg

-- Fail-Handle: Handler catches failure
handle (fail msg) with { fail m k' → e }  ≡  e[msg/m]
```

### 7.3 Parser-Specific Laws

```
-- Peek-Peek: Peek is idempotent (pure observation)
peek (); (fun s → peek (); k s)  ≡  peek (); (fun s → k s s)

-- Advance-Peek: Advance affects subsequent peek
advance n; peek (); k  ≡  advance n; (fun _ → let remaining = drop n input in k remaining)

-- Mark-Restore: Restore returns to marked position
let p = mark () in advance n; restore p; peek ()  ≡  peek ()

-- Choice-Left-Fail: Left failure without consumption tries right
(fail msg <|> p2) input  ≡  p2 input    -- when at same position

-- Choice-Success: Success doesn't try alternative
(return x <|> p2)  ≡  return x

-- Try-Backtrack: Try enables backtracking even after consumption
try (advance n; fail msg) <|> p  ≡  p
```

### 7.4 Proving Handler Correctness

**Goal**: Show your handler interprets operations in a way that satisfies the laws.

**Method**: Equational reasoning on handler semantics.

**Example**: Proving Put-Get law

```
Claim: put s; get (); k  ≡  put s; k s

Proof:
  handle (perform put s; perform get (); k _) with state_handler
  
= -- By (Op) rule for put
  (fun st → handle (perform get (); k _) with state_handler) s
  
= -- By (Op) rule for get  
  (fun st → (fun st' → handle (k st') with state_handler) st) s
  
= -- β-reduction
  (fun st' → handle (k st') with state_handler) s
  
= -- This equals...
  handle (perform put s; k s) with state_handler

  (fun st → handle (k s) with state_handler) s
  
= -- Same result ∎
```

### 7.5 Contextual Equivalence

Two expressions are contextually equivalent if they behave the same in all contexts:

```
e₁ ≈ e₂  ⟺  ∀C. C[e₁] terminates ⟺ C[e₂] terminates
             ∧ C[e₁] ⇓ v ⟺ C[e₂] ⇓ v
```

For parser laws, this means:
- Two parsers are equivalent if they produce the same results on all inputs
- They consume the same amount of input
- They fail on the same inputs with the same error positions

---

## 8. Property Testing

### 8.1 Monad Laws

```
-- Left identity: return a >>= f  ≡  f a
property left_identity (a : α) (f : α → Parser β) (input : String) =
    run_parser (return_parser a >>= f) input 
    == 
    run_parser (f a) input

-- Right identity: p >>= return  ≡  p
property right_identity (p : Parser α) (input : String) =
    run_parser (p >>= return_parser) input 
    == 
    run_parser p input

-- Associativity: (p >>= f) >>= g  ≡  p >>= (λx. f x >>= g)
property associativity (p : Parser α) (f : α → Parser β) (g : β → Parser γ) (input : String) =
    run_parser ((p >>= f) >>= g) input 
    == 
    run_parser (p >>= (fun x → f x >>= g)) input
```

### 8.2 Choice Laws

```
-- Left zero: fail <|> p  ≡  p (when fail doesn't consume)
property left_zero (p : Parser α) (input : String) =
    run_parser (fail "x" <|> p) input 
    == 
    run_parser p input

-- Right zero: return x <|> p  ≡  return x
property right_zero (x : α) (p : Parser α) (input : String) =
    run_parser (return_parser x <|> p) input 
    == 
    run_parser (return_parser x) input

-- Associativity of choice
property choice_assoc (p1 p2 p3 : Parser α) (input : String) =
    run_parser ((p1 <|> p2) <|> p3) input 
    == 
    run_parser (p1 <|> (p2 <|> p3)) input
```

### 8.3 Backtracking Properties

```
-- Try enables backtracking after consumption
property try_backtracks (s1 s2 : String) (input : String) =
    -- If s1 is prefix of input but parsing s1 then failing should backtrack
    is_prefix s1 input && not (is_prefix (s1 ++ "!") input) ==>
    run_parser (try (string s1 >> fail "x") <|> string s2) input
    ==
    run_parser (string s2) input

-- Without try, consumption prevents backtracking
property no_backtrack_after_consume (input : String) =
    string_length input >= 2 ==>
    let c1 = string_head input in
    is_fail (run_parser (char c1 >> fail "x" <|> char c1) input)
```

### 8.4 Position Tracking

```
-- Position increases correctly
property position_advances (s : String) (input : String) =
    is_prefix s input ==>
    match run_parser (string s >> position ()) input with
    | ParseOk pos _ _ → pos == string_length s
    | _ → False
    end

-- Mark/restore returns to correct position
property mark_restore (n : Int) (input : String) =
    n >= 0 && n <= string_length input ==>
    run_parser (
        let p = mark () in
        advance n;
        restore p;
        position ()
    ) input
    ==
    ParseOk 0 input 0
```

### 8.5 Combinator Properties

```
-- many never fails
property many_never_fails (p : Parser α) (input : String) =
    not (is_fail (run_parser (many p) input))

-- many1 fails iff first parse fails
property many1_first (p : Parser α) (input : String) =
    is_fail (run_parser (many1 p) input)
    ==
    is_fail (run_parser p input)

-- sep_by with no input gives empty list
property sep_by_empty (p : Parser α) (sep : Parser β) =
    run_parser (sep_by p sep) ""
    ==
    ParseOk [] "" 0

-- optional always succeeds
property optional_succeeds (p : Parser α) (input : String) =
    not (is_fail (run_parser (optional p) input))
```

### 8.6 Roundtrip Properties

```
-- If we can parse it, we parsed something sensible
property parse_string_roundtrip (s : String) (input : String) =
    is_prefix s input ==>
    match run_parser (string s) input with
    | ParseOk result rest _ → 
        result == s && rest == string_drop (string_length s) input
    | _ → False
    end

-- Consumption is correct
property consumption_correct (p : Parser α) (input : String) =
    match run_parser p input with
    | ParseOk _ rest pos → 
        pos == string_length input - string_length rest
    | _ → True
    end
```

### 8.7 Adversarial Tests

```
-- Empty input handling
property empty_input_char =
    is_fail (run_parser (char 'x') "")

property empty_input_eof =
    run_parser (eof ()) "" == ParseOk () "" 0

-- Very long input
property long_input (n : Int) =
    n > 0 ==>
    let input = string_repeat n 'a' in
    run_parser (many (char 'a') >> eof ()) input
    ==
    ParseOk () "" n

-- Deep nesting
property deep_nesting (depth : Int) =
    depth > 0 && depth < 1000 ==>
    let input = string_repeat depth '(' ++ string_repeat depth ')' in
    let parens = char '(' >> parens >> char ')' <|> return_parser () in
    not (is_fail (run_parser parens input))

-- Unicode handling
property unicode_handling =
    run_parser (string "héllo") "héllo world"
    ==
    ParseOk "héllo" " world" 5
```

---

## 9. Advanced Topics

### 9.1 Error Recovery

Add error recovery to continue parsing after failures:

```
effect Parser =
    | ... -- previous operations
    | recover : () → ()           -- Mark recovery point
    | sync : (Char → Bool) → ()   -- Skip to synchronization token
end

-- Recover combinator: try to parse, on failure skip to recovery point
let recovering (sync_pred : Char → Bool) (p : Parser a) : Parser (Option a) =
    handle p () with
    | return x → Some x
    | Parser.fail_parse msg k →
        perform Parser.sync sync_pred;
        None
    end
```

### 9.2 Error Messages with Context

```
type ParseError = {
    message  : String,
    position : Int,
    line     : Int,
    column   : Int,
    context  : List String,     -- Stack of "currently parsing X"
    expected : List String      -- What was expected
}

effect Parser =
    | ...
    | label : String → a → a      -- Add context label
    | expect : String → ()        -- Record expectation
end

let labelled (name : String) (p : Parser a) : Parser a =
    perform Parser.label name (p ())
```

### 9.3 Incremental Parsing

For parsing streaming input:

```
type Partial a =
    | Done a String              -- Finished with remaining input
    | Fail String Int            -- Failed
    | NeedMore (String → Partial a)  -- Need more input

effect Parser =
    | ...
    | demand : Int → String      -- Demand at least n more characters
end
```

### 9.4 Memoization (Packrat Parsing)

```
effect Memo =
    | memo : (() → a) → a        -- Memoize a parser at current position
end

-- Handler maintains memo table keyed by (parser_id, position)
let with_memoization (p : Parser a) (input : String) =
    let table = ref empty_map in
    handle run_parser p input with
    | Memo.memo parser k →
        let pos = current_position () in
        let key = (parser_id parser, pos) in
        match Map.get key !table with
        | Some result → k result
        | None →
            let result = parser () in
            table := Map.set key result !table;
            k result
        end
    end
```

### 9.5 Left Recursion

Direct left recursion doesn't work with parser combinators. Solutions:

```
-- 1. Rewrite grammar to right recursion
let rec expr () = term () >>= expr_rest
and expr_rest left = 
    (char '+' >> term () >>= fun right → expr_rest (Add left right))
    <|> return_parser left

-- 2. Use explicit iteration
let chainl1 (p : Parser a) (op : Parser (a → a → a)) : Parser a =
    p >>= fun x →
    many (op >>= fun f → p >>= fun y → return_parser (f, y)) >>= fun rest →
    return_parser (List.foldl (fun acc (f, y) → f acc y) x rest)

-- Usage:
let expr = chainl1 term (char '+' >> return_parser (fun a b → Add a b))
```

---

## 10. Complete Reference Implementation

### 10.1 Full Source

```
-- ============================================================
-- Algebraic Effect Parser Combinator Library
-- ============================================================

-- ------------------------------------------------------------
-- Types
-- ------------------------------------------------------------

type Option a =
    | None
    | Some a
end

type List a =
    | Nil
    | Cons a (List a)
end

type ParseResult a =
    | ParseOk a String Int        -- value, remaining, position
    | ParseFail String Int Int    -- message, position, line
end

type ParserState = {
    input  : String,
    pos    : Int,
    line   : Int,
    column : Int
}

-- ------------------------------------------------------------
-- Effects
-- ------------------------------------------------------------

effect Parser =
    | peek      : () → String
    | advance   : Int → ()
    | get_pos   : () → Int
    | get_loc   : () → (Int, Int)
    | mark      : () → Int
    | restore   : Int → ()
    | fail_parse : String → a
end

-- ------------------------------------------------------------
-- Parser Type
-- ------------------------------------------------------------

-- A parser is a suspended effectful computation
type Parser a = () → a ! {Parser}

-- ------------------------------------------------------------
-- Running Parsers
-- ------------------------------------------------------------

let run_parser (p : Parser a) (input : String) : ParseResult a =
    let handler = fun comp →
        handle comp with
        | return x → fun st → 
            ParseOk x (string_drop st.pos st.input) st.pos
        | Parser.peek () k → fun st →
            k (string_drop st.pos st.input) st
        | Parser.advance n k → fun st →
            k () { st with pos = st.pos + n }
        | Parser.get_pos () k → fun st →
            k st.pos st
        | Parser.get_loc () k → fun st →
            k (st.line, st.column) st
        | Parser.mark () k → fun st →
            k st.pos st
        | Parser.restore saved k → fun st →
            k () { st with pos = saved }
        | Parser.fail_parse msg k → fun st →
            ParseFail msg st.pos st.line
        end
    in
    handler (p ()) { input = input, pos = 0, line = 1, column = 1 }

-- ------------------------------------------------------------
-- Core Combinators
-- ------------------------------------------------------------

let return_parser (x : a) : Parser a =
    fun () → x

let (>>=) (p : Parser a) (f : a → Parser b) : Parser b =
    fun () → f (p ()) ()

let (>>) (p : Parser a) (q : Parser b) : Parser b =
    p >>= fun _ → q

let (<<) (p : Parser a) (q : Parser b) : Parser a =
    p >>= fun x → q >> return_parser x

let (<$>) (f : a → b) (p : Parser a) : Parser b =
    p >>= fun x → return_parser (f x)

let (<*>) (pf : Parser (a → b)) (pa : Parser a) : Parser b =
    pf >>= fun f → f <$> pa

-- ------------------------------------------------------------
-- Failure and Choice
-- ------------------------------------------------------------

let fail (msg : String) : Parser a =
    fun () → perform Parser.fail_parse msg

let try (p : Parser a) : Parser a =
    fun () →
        let saved = perform Parser.mark () in
        handle p () with
        | return x → x
        | Parser.fail_parse msg k →
            perform Parser.restore saved;
            perform Parser.fail_parse msg
        end

let (<|>) (p : Parser a) (q : Parser a) : Parser a =
    fun () →
        let start = perform Parser.get_pos () in
        handle p () with
        | return x → x
        | Parser.fail_parse msg k →
            if perform Parser.get_pos () == start then
                q ()
            else
                perform Parser.fail_parse msg
        end

-- ------------------------------------------------------------
-- Primitive Parsers
-- ------------------------------------------------------------

let peek () : Parser String =
    fun () → perform Parser.peek ()

let position () : Parser Int =
    fun () → perform Parser.get_pos ()

let satisfy (pred : Char → Bool) (desc : String) : Parser Char =
    fun () →
        let remaining = perform Parser.peek () in
        if string_length remaining == 0 then
            perform Parser.fail_parse ("Expected " ++ desc ++ ", got end of input")
        else
            let c = string_head remaining in
            if pred c then
                (perform Parser.advance 1; c)
            else
                perform Parser.fail_parse ("Expected " ++ desc)

let char (c : Char) : Parser Char =
    satisfy (fun x → x == c) (char_to_string c)

let string (s : String) : Parser String =
    fun () →
        let len = string_length s in
        let remaining = perform Parser.peek () in
        if string_take len remaining == s then
            (perform Parser.advance len; s)
        else
            perform Parser.fail_parse ("Expected \"" ++ s ++ "\"")

let eof () : Parser () =
    fun () →
        let remaining = perform Parser.peek () in
        if string_length remaining == 0 then
            ()
        else
            perform Parser.fail_parse "Expected end of input"

-- ------------------------------------------------------------
-- Character Classes
-- ------------------------------------------------------------

let any_char : Parser Char =
    satisfy (fun _ → True) "any character"

let digit : Parser Char =
    satisfy is_digit "digit"

let letter : Parser Char =
    satisfy is_alpha "letter"

let alphanumeric : Parser Char =
    satisfy is_alphanumeric "alphanumeric"

let whitespace : Parser Char =
    satisfy is_space "whitespace"

let one_of (chars : String) : Parser Char =
    satisfy (fun c → string_contains chars c) ("one of \"" ++ chars ++ "\"")

let none_of (chars : String) : Parser Char =
    satisfy (fun c → not (string_contains chars c)) ("none of \"" ++ chars ++ "\"")

-- ------------------------------------------------------------
-- Repetition
-- ------------------------------------------------------------

let rec many (p : Parser a) : Parser (List a) =
    (p >>= fun x → many p >>= fun xs → return_parser (Cons x xs))
    <|>
    return_parser Nil

let many1 (p : Parser a) : Parser (List a) =
    p >>= fun x → many p >>= fun xs → return_parser (Cons x xs)

let optional (p : Parser a) : Parser (Option a) =
    (Some <$> p) <|> return_parser None

let skip_many (p : Parser a) : Parser () =
    many p >> return_parser ()

let skip_many1 (p : Parser a) : Parser () =
    p >> skip_many p

-- ------------------------------------------------------------
-- Separators and Chains
-- ------------------------------------------------------------

let sep_by (p : Parser a) (sep : Parser b) : Parser (List a) =
    sep_by1 p sep <|> return_parser Nil

let sep_by1 (p : Parser a) (sep : Parser b) : Parser (List a) =
    p >>= fun x → 
    many (sep >> p) >>= fun xs → 
    return_parser (Cons x xs)

let end_by (p : Parser a) (sep : Parser b) : Parser (List a) =
    many (p << sep)

let chainl1 (p : Parser a) (op : Parser (a → a → a)) : Parser a =
    p >>= fun x →
    let rec go acc =
        (op >>= fun f → p >>= fun y → go (f acc y))
        <|> return_parser acc
    in go x

let chainr1 (p : Parser a) (op : Parser (a → a → a)) : Parser a =
    p >>= fun x →
    (op >>= fun f → chainr1 p op >>= fun y → return_parser (f x y))
    <|> return_parser x

-- ------------------------------------------------------------
-- Combinators
-- ------------------------------------------------------------

let between (open : Parser a) (close : Parser b) (p : Parser c) : Parser c =
    open >> p << close

let parens (p : Parser a) : Parser a =
    between (char '(') (char ')') p

let braces (p : Parser a) : Parser a =
    between (char '{') (char '}') p

let brackets (p : Parser a) : Parser a =
    between (char '[') (char ']') p

let lexeme (p : Parser a) : Parser a =
    p << skip_many whitespace

let symbol (s : String) : Parser String =
    lexeme (string s)

-- ------------------------------------------------------------
-- Numbers
-- ------------------------------------------------------------

let natural : Parser Int =
    many1 digit >>= fun ds → 
    return_parser (chars_to_int ds)

let integer : Parser Int =
    (char '-' >> natural >>= fun n → return_parser (0 - n))
    <|>
    (optional (char '+') >> natural)

-- ------------------------------------------------------------
-- Utility
-- ------------------------------------------------------------

let look_ahead (p : Parser a) : Parser a =
    fun () →
        let saved = perform Parser.mark () in
        let result = p () in
        perform Parser.restore saved;
        result

let not_followed_by (p : Parser a) : Parser () =
    fun () →
        let saved = perform Parser.mark () in
        handle p () with
        | return _ →
            perform Parser.restore saved;
            perform Parser.fail_parse "Unexpected"
        | Parser.fail_parse _ k →
            perform Parser.restore saved;
            ()
        end

-- ============================================================
-- End of Library
-- ============================================================
```

### 10.2 Example Usage

```
-- A simple expression parser

type Expr =
    | Num Int
    | Add Expr Expr
    | Mul Expr Expr
    | Neg Expr
    | Var String
end

let spaces = skip_many whitespace

let ident : Parser String =
    letter >>= fun c →
    many alphanumeric >>= fun cs →
    return_parser (chars_to_string (Cons c cs))

let number : Parser Expr =
    Num <$> lexeme integer

let variable : Parser Expr =
    Var <$> lexeme ident

let rec expr () : Parser Expr =
    chainl1 (term ()) add_op

and term () : Parser Expr =
    chainl1 (factor ()) mul_op

and factor () : Parser Expr =
    lexeme (parens (expr ()))
    <|> number
    <|> variable
    <|> (symbol "-" >> Neg <$> factor ())

and add_op : Parser (Expr → Expr → Expr) =
    (symbol "+" >> return_parser Add)
    <|> (symbol "-" >> return_parser (fun a b → Add a (Neg b)))

and mul_op : Parser (Expr → Expr → Expr) =
    symbol "*" >> return_parser Mul

let parse_expr (input : String) : ParseResult Expr =
    run_parser (spaces >> expr () << eof ()) input

-- Test it
let main () =
    match parse_expr "1 + 2 * 3" with
    | ParseOk e _ _ → print_expr e   -- Add (Num 1) (Mul (Num 2) (Num 3))
    | ParseFail msg pos line → print ("Error at " ++ int_to_string pos ++ ": " ++ msg)
    end
```

---

## Appendix A: Checklist for Implementation

### Type System
- [ ] Define syntax for types, effects, expressions
- [ ] Implement typing rules for all expression forms
- [ ] Implement effect row operations (union, membership)
- [ ] Implement effect subtyping

### Operational Semantics
- [ ] Define values and evaluation contexts
- [ ] Define reduction rules (β, Ret, Op)
- [ ] Implement small-step evaluator (for testing)

### Type Soundness
- [ ] Prove substitution lemma
- [ ] Prove weakening lemma
- [ ] Prove context typing lemma
- [ ] Prove preservation theorem
- [ ] Prove progress theorem
- [ ] Prove effect safety

### Handler Correctness
- [ ] State state effect laws
- [ ] State failure effect laws
- [ ] State parser-specific laws
- [ ] Prove handlers satisfy laws (equational reasoning)

### Property Tests
- [ ] Monad laws
- [ ] Choice laws
- [ ] Backtracking properties
- [ ] Position tracking
- [ ] Combinator properties
- [ ] Adversarial tests

---

## Appendix B: Further Reading

1. **Plotkin & Power** — "Algebraic Operations and Generic Effects" (2003)
2. **Plotkin & Pretnar** — "Handlers of Algebraic Effects" (2013)
3. **Bauer & Pretnar** — "Programming with Algebraic Effects and Handlers" (2015)
4. **Leijen** — "Type Directed Compilation of Row-Typed Algebraic Effects" (2017)
5. **Hillerström & Lindley** — "Shallow Effect Handlers" (2018)
6. **Biernacki et al.** — "Abstracting Algebraic Effects" (2019)

---

*Document Version 1.0 — Generated for algebraic effect parser combinator implementation*
