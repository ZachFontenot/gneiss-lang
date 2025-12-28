# Gneiss Language Grammar

This document defines the formal grammar for Gneiss using Extended Backus-Naur Form (EBNF).
The parser implementation in `src/parser.rs` must conform to this specification.

## Notation

- `|` alternation
- `?` optional (zero or one)
- `*` zero or more
- `+` one or more
- `( )` grouping
- `'...'` literal tokens
- `UPPER_IDENT` uppercase-start identifier (constructors, modules, traits)
- `IDENT` lowercase-start identifier (variables, functions)
- `OP` operator symbol

---

## Program Structure

```ebnf
program     ::= export? item* EOF

item        ::= import
              | decl
              | expr ';;'?
```

---

## Exports

Export declarations must appear at the top of a module, before any other items.
If no export declaration is present, all declarations are public (backward compatible).

```ebnf
export      ::= 'export' '(' export_item (',' export_item)* ','? ')'

export_item ::= IDENT                                   (* value/function *)
              | UPPER_IDENT                             (* type only, constructors private *)
              | UPPER_IDENT '(' '..' ')'                (* type + all constructors *)
              | UPPER_IDENT '(' UPPER_IDENT (',' UPPER_IDENT)* ')' (* type + specific constructors *)
```

**Examples**:
```gneiss
export (foo, bar)                    -- export values foo and bar
export (MyType)                      -- export type, constructors private
export (MyType(..))                  -- export type and all constructors
export (Either(Left, Right))         -- export type and specific constructors
export (Result(..), map, flatMap)    -- mix of types and values
```

---

## Imports

```ebnf
import      ::= 'import' module_path import_alias? selective_imports?

module_path ::= UPPER_IDENT ('/' UPPER_IDENT)*

import_alias ::= 'as' UPPER_IDENT

selective_imports ::= '(' import_item (',' import_item)* ')'

import_item ::= IDENT ('as' IDENT)?
              | UPPER_IDENT ('as' UPPER_IDENT)?
```

---

## Declarations

Visibility is controlled by the module's export list, not per-declaration keywords.

```ebnf
decl        ::= let_decl
              | type_decl
              | effect_decl
              | trait_decl
              | impl_decl
              | fixity_decl
              | val_decl
```

### Let Declarations

```ebnf
let_decl    ::= 'let' 'rec'? binding ('and' binding)*

binding     ::= pattern '=' expr                    (* simple binding *)
              | IDENT pattern+ '=' expr             (* function syntax *)
              | qualified_name pattern* '=' expr    (* module function *)
              | '(' OP ')' pattern* '=' expr        (* operator prefix *)
              | pattern OP pattern '=' expr         (* operator infix *)
```

### Type Declarations

```ebnf
type_decl   ::= 'type' UPPER_IDENT type_param* '=' type_body

type_param  ::= IDENT

type_body   ::= variant_type
              | record_type
              | type_expr                           (* type alias *)

variant_type ::= '|'? constructor ('|' constructor)*

constructor ::= UPPER_IDENT type_atom*

record_type ::= '{' field_decl (',' field_decl)* ','? '}'

field_decl  ::= IDENT ':' type_expr
```

### Effect Declarations

```ebnf
effect_decl ::= 'effect' UPPER_IDENT type_param* '=' effect_body 'end'

effect_body ::= effect_op*

effect_op   ::= '|' IDENT ':' type_expr
```

**Examples**:
```gneiss
-- Simple effect
effect Ask =
    | ask : () -> Int
end

-- Parameterized effect
effect State s =
    | get : () -> s
    | put : s -> ()
end

-- Effect with polymorphic operations
effect Choice =
    | choose : [a] -> a
    | fail : () -> a
end
```

### Trait Declarations

```ebnf
trait_decl  ::= 'trait' UPPER_IDENT IDENT supertrait? '=' trait_body 'end'

supertrait  ::= ':' UPPER_IDENT (',' UPPER_IDENT)*

trait_body  ::= val_sig*

val_sig     ::= 'val' IDENT ':' type_expr
```

### Implementation Declarations

```ebnf
impl_decl   ::= 'impl' UPPER_IDENT 'for' type_expr constraints? '=' impl_body 'end'

constraints ::= 'where' constraint (',' constraint)*

constraint  ::= IDENT ':' UPPER_IDENT

impl_body   ::= impl_method*

impl_method ::= 'let' IDENT pattern* '=' expr
```

### Fixity Declarations

```ebnf
fixity_decl ::= fixity_kind INT op_list

fixity_kind ::= 'infixl' | 'infixr' | 'infix'

op_list     ::= OP+
```

### Value Declarations

```ebnf
val_decl    ::= 'val' qualified_name ':' type_expr constraints?
```

---

## Expressions

Expressions are parsed with context-awareness to handle ambiguous constructs.

```ebnf
expr        ::= expr_seq

expr_seq    ::= expr_let (';' expr_let)*

expr_let    ::= 'let' 'rec'? binding ('and' binding)* 'in' expr
              | expr_if

expr_if     ::= 'if' expr 'then' expr 'else' expr
              | expr_compound

expr_compound ::= expr_match
                | expr_lambda
                | expr_select
                | expr_binary
```

### Match Expressions

**Key rule**: Nested match expressions require explicit delimiters.

```ebnf
expr_match  ::= 'match' expr 'with' match_arm+ match_end

match_end   ::= 'end'                               (* explicit terminator *)
              | (* empty - allowed at top level only *)

match_arm   ::= '|'? pattern guard? '->' expr

guard       ::= 'if' expr_binary                    (* restricted context *)
```

When a match arm body contains another `match`, the inner match MUST use
either parentheses `(match ... with ...)` or terminate with `end`.

### Lambda Expressions

```ebnf
expr_lambda ::= 'fun' pattern+ '->' expr
```

### Select Expressions (CSP-style)

```ebnf
expr_select ::= 'select' select_arm+ 'end'

select_arm  ::= '|' pattern '<-' expr '->' expr
```

### Binary Expressions

Binary expressions use precedence climbing (Pratt parsing).

```ebnf
expr_binary ::= expr_unary (OP expr_unary)*
```

See [Operator Precedence](#operator-precedence) for precedence table.

### Unary Expressions

```ebnf
expr_unary  ::= 'not' expr_unary
              | '-' expr_unary
              | expr_app
```

### Application Expressions

```ebnf
expr_app    ::= expr_postfix expr_postfix*
```

Function application is left-associative and binds tighter than all operators.

### Postfix Expressions

```ebnf
expr_postfix ::= expr_atom ('.' IDENT)*
```

### Atomic Expressions

```ebnf
expr_atom   ::= literal
              | IDENT
              | qualified_name
              | UPPER_IDENT record_literal?         (* constructor *)
              | '(' expr ')'                        (* parenthesized *)
              | '(' expr (',' expr)+ ')'            (* tuple *)
              | '(' ')'                             (* unit *)
              | '[' (expr (',' expr)*)? ','? ']'    (* list *)
              | '{' expr 'with' record_update '}'   (* record update *)
              | '{' record_literal_fields '}'       (* record literal *)
              | expr_perform
              | expr_handle
              | 'spawn' expr_atom

expr_perform ::= 'perform' qualified_name expr_atom*

expr_handle ::= 'handle' expr 'with' handler_arm+ 'end'

handler_arm ::= '|' 'return' IDENT '->' expr
              | '|' IDENT pattern* IDENT '->' expr

record_literal ::= '{' record_literal_fields '}'

record_literal_fields ::= field_init (',' field_init)* ','?

field_init  ::= IDENT '=' expr

record_update ::= field_init (',' field_init)* ','?
```

---

## Patterns

```ebnf
pattern     ::= pattern_cons

pattern_cons ::= simple_pattern ('::' pattern_cons)?    (* right-associative *)

simple_pattern ::= pattern_atom
                 | UPPER_IDENT pattern_atom*            (* constructor with args *)

pattern_atom ::= '_'                                    (* wildcard *)
               | IDENT                                  (* variable *)
               | literal
               | UPPER_IDENT                            (* constructor no args *)
               | UPPER_IDENT record_pattern             (* record constructor *)
               | '(' pattern ')'                        (* parenthesized *)
               | '(' pattern (',' pattern)+ ')'         (* tuple *)
               | '(' ')'                                (* unit *)
               | '[' (pattern (',' pattern)*)? ','? ']' (* list *)

record_pattern ::= '{' record_pattern_fields '}'

record_pattern_fields ::= field_pattern (',' field_pattern)* ','?

field_pattern ::= IDENT ('=' pattern)?                  (* punning allowed *)
                | '_'                                   (* ignore remaining *)
```

---

## Types

```ebnf
type_expr   ::= type_app ('->' type_expr)?              (* function type *)

type_app    ::= type_atom type_atom*                    (* type application *)

type_atom   ::= IDENT                                   (* type variable *)
              | UPPER_IDENT                             (* type constructor *)
              | '(' type_expr ')'                       (* parenthesized *)
              | '(' ')'                                 (* unit type *)
              | '(' type_expr (',' type_expr)+ ')'      (* tuple type *)
              | '[' type_expr ']'                       (* list type sugar *)
```

---

## Helpers

```ebnf
qualified_name ::= UPPER_IDENT '.' IDENT
                 | IDENT

literal     ::= INT
              | FLOAT
              | STRING
              | CHAR
              | 'true'
              | 'false'
```

---

## Operator Precedence

From lowest to highest precedence:

| Level | Operators | Associativity | Description |
|-------|-----------|---------------|-------------|
| 1 | `\|>` `<\|` | Left | Pipe operators |
| 2 | `\|\|` | Left | Boolean or |
| 3 | `&&` | Left | Boolean and |
| 4 | `==` `!=` `<` `>` `<=` `>=` | None | Comparison |
| 5 | `::` `++` | Right | Cons and append |
| 6 | `+` `-` | Left | Additive |
| 7 | `*` `/` `%` | Left | Multiplicative |
| 8 | `>>` `<<` | Left | Composition |
| 9 | (application) | Left | Function application |
| 10 | (unary) | Right | `not`, `-` |

**Pipe operator semantics**:
- `x |> f` desugars to `f x`
- `f <| x` desugars to `f x`

**Non-associative operators**: Comparison operators cannot be chained.
`a == b == c` is a parse error.

---

## Disambiguation Rules

### Nested Match Expressions

**Problem**: Without delimiters, nested match is ambiguous:
```
match x with
| 1 -> match y with
       | a -> a
       | b -> b      -- Is this arm of inner or outer match?
| 2 -> 0
```

**Solution**: Inner match requires explicit delimiter:
```
-- Option 1: Use 'end' keyword
match x with
| 1 -> match y with
       | a -> a
       | b -> b
       end
| 2 -> 0

-- Option 2: Use parentheses
match x with
| 1 -> (match y with | a -> a | b -> b)
| 2 -> 0
```

**Rule**: When parsing a match arm body, if the body starts with `match`,
the inner match must terminate with `end` or be wrapped in parentheses.

### Let-in vs Let Declaration

- `let x = 1 in x + 1` is a let expression (has body)
- `let x = 1` at top level is a let declaration

### Sequences in Branches

Sequences (`;`) are not allowed directly in if/match branches.
Use parentheses for multi-statement branches:
```
if cond then (print "a"; 1) else (print "b"; 2)
```

---

## Reserved Words

```
let rec and in fun match with if then else
type trait impl for where val end
effect handle perform return
import export as
spawn select
true false not
infixl infixr infix
```

---

## Comments

```
-- Single line comment (to end of line)

{- Multi-line
   block comment -}
```

Block comments nest properly: `{- outer {- inner -} outer -}`
