# Gneiss Syntax Specification

Gneiss is a statically-typed functional language with OCaml-inspired syntax, a trait system for ad-hoc polymorphism, and first-class concurrency primitives.

**Key design choices:**
- File = Module (no in-language module syntax)
- Comma-separated lists
- No object-oriented features
- Bare lowercase type variables (no tick prefix)
- Trait-based polymorphism with constraints

---

## Lexical Conventions

### Comments
```gneiss
-- Single line comment

{-
   Multi-line
   comment
-}
```

### Identifiers
- Value identifiers: lowercase start, alphanumeric + underscores (`foo`, `my_value`, `x1`)
- Type identifiers: lowercase start (`int`, `string`, `my_type`)
- Type variables: bare lowercase letters (`a`, `b`, `key`)
- Constructor identifiers: uppercase start (`Some`, `None`, `Ok`, `Error`)
- Trait identifiers: uppercase start (`Display`, `Eq`, `Ord`)

### Literals
```gneiss
42              -- Int
3.14            -- Float
"hello"         -- String
'c'             -- Char
true, false     -- Bool
()              -- Unit
```

---

## Basic Expressions

### Let Bindings
```gneiss
let x = 42

let add a b = a + b

-- With type annotation
let add : Int -> Int -> Int
let add a b = a + b

-- Local binding
let x =
  let y = 10 in
  y + 1
```

### Function Application
```gneiss
add 1 2
List.map f xs      -- List is a file/module
```

### Lambda Expressions
```gneiss
fun x -> x + 1
fun x y -> x + y
fun (x, y) -> x + y   -- Pattern in lambda
```

### Operators
```gneiss
-- Arithmetic
+, -, *, /, %

-- Comparison
==, !=, <, >, <=, >=

-- Boolean
&&, ||, not

-- Function composition
f >> g    -- (f >> g) x = g (f x)
f << g    -- (f << g) x = f (g x)

-- Pipe
x |> f    -- f x
f <| x    -- f x
```

---

## Lists and Tuples

### Lists (Comma-Separated)
```gneiss
[]                      -- Empty list
[1, 2, 3]               -- List of integers
[1, 2, 3,]              -- Trailing comma allowed
["a", "b", "c"]         -- List of strings

-- Cons operator
1 :: [2, 3]             -- [1, 2, 3]
1 :: 2 :: 3 :: []       -- [1, 2, 3]

-- List concatenation
[1, 2] ++ [3, 4]        -- [1, 2, 3, 4]
```

### Tuples
```gneiss
(1, "hello")            -- Tuple of Int and String
(1, 2, 3)               -- Triple
(x, y, z)               -- Tuple pattern
```

---

## Type Definitions

### Type Aliases
```gneiss
type UserId = Int
type Point = (Float, Float)
type StringList = List String
```

### Algebraic Data Types (Variants)
```gneiss
type Option a =
  | Some a
  | None

type Result a e =
  | Ok a
  | Error e

type List a =
  | Nil
  | Cons a (List a)

type Expr =
  | Lit Int
  | Add Expr Expr
  | Mul Expr Expr
  | Var String
```

### Records
```gneiss
type Person = {
  name : String,
  age : Int,
  email : String,
}

-- Record construction
let p = { name = "Alice", age = 30, email = "alice@example.com" }

-- Field access
p.name

-- Record update
{ p with age = 31 }

-- Record pattern matching
let { name, age, _ } = p
```

---

## Pattern Matching

### Match Expressions
```gneiss
match x with
| 0 -> "zero"
| 1 -> "one"
| n -> "many"

match opt with
| Some x -> x
| None -> default

match list with
| [] -> "empty"
| [x] -> "singleton"
| [x, y] -> "pair"
| x :: xs -> "at least one"

-- Guards
match n with
| x when x < 0 -> "negative"
| x when x == 0 -> "zero"
| x -> "positive"
```

### Let Patterns
```gneiss
let (x, y) = point
let { name, age, _ } = person
let Some value = maybe_value   -- Partial, warns or errors
```

---

## Traits and Implementations LATER PHASE

### Trait Definitions 
```gneiss
trait Display a =
  val display : a -> String
end

trait Eq a =
  val eq : a -> a -> Bool
  val neq : a -> a -> Bool
end

-- Trait with default implementations
trait Ord a =
  val compare : a -> a -> Ordering

  -- Default implementations
  let lt a b = compare a b == Lt
  let gt a b = compare a b == Gt
  let lte a b = compare a b != Gt
  let gte a b = compare a b != Lt
end
```

### Trait Inheritance
```gneiss
trait Ord a : Eq =
  val compare : a -> a -> Ordering
end

-- Multiple supertraits
trait Num a : Eq, Display =
  val add : a -> a -> a
  val mul : a -> a -> a
  val neg : a -> a
end
```

### Implementations
```gneiss
impl Display for Int =
  let display n = Int.to_string n
end

impl Display for String =
  let display s = s
end

impl Display for (List a) where a : Display =
  let display xs =
    let items = List.map display xs in
    "[" ++ String.join ", " items ++ "]"
end

-- Implementing for a specific type
impl Eq for Person =
  let eq a b = a.name == b.name && a.age == b.age
  let neq a b = not (eq a b)
end
```

### Constrained Type Signatures
```gneiss
-- Single constraint
val show : a -> String where a : Display
let show x = display x

-- Multiple constraints on one type variable
val show_eq : a -> a -> String where a : Display, a : Eq
let show_eq x y =
  if eq x y then "equal: " ++ display x
  else "not equal"

-- Multiple type variables with constraints
val combine : a -> b -> String where a : Display, b : Display
let combine x y = display x ++ " and " ++ display y

-- Complex constraints
val sort_and_show : List a -> String where a : Ord, a : Display
let sort_and_show xs =
  xs |> List.sort |> List.map display |> String.join ", "
```

---

## File = Module System

Each `.gn` file is implicitly a module. The filename (capitalized) becomes the module name.

### File: `list.gn`
```gneiss
-- This file defines the List module
-- Accessed as List.map, List.filter, etc.

type t a =
  | Nil
  | Cons a (t a)

let empty = Nil

let singleton x = Cons x Nil

let map f xs =
  match xs with
  | Nil -> Nil
  | Cons x rest -> Cons (f x) (map f rest)

let fold f init xs =
  match xs with
  | Nil -> init
  | Cons x rest -> fold f (f init x) rest
```

### Imports
```gneiss
-- Import specific items from a module
import List (map, filter, fold)

-- Import all exports
import List

-- Qualified access (always available)
List.map f xs
String.length s
```

### Project Structure
```
my_project/
  src/
    main.gn          -- Main module
    list.gn          -- List module
    option.gn        -- Option module
    utils/
      string.gn      -- Utils.String module
      math.gn        -- Utils.Math module
```

---

## Concurrency Primitives

### Channels (CML-style)
```gneiss
-- Create a channel
let ch : Channel Int = Channel.new ()

-- Send and receive (synchronous)
Channel.send ch 42
let value = Channel.recv ch

-- As events
let send_evt = Channel.send_evt ch 42
let recv_evt = Channel.recv_evt ch

-- Synchronize on an event
let value = Event.sync recv_evt
```

### Events
```gneiss
-- Choose between events (non-deterministic)
let evt = Event.choose [recv_evt1, recv_evt2, timeout_evt]

-- Wrap an event with a handler
let evt = Event.wrap recv_evt (fun x -> x + 1)

-- Guard (delayed event creation)
let evt = Event.guard (fun () ->
  let ch = get_current_channel () in
  Channel.recv_evt ch
)

-- Select (sync on choice)
let result = Event.select [
  Event.wrap (Channel.recv_evt ch1) (fun x -> Left x),
  Event.wrap (Channel.recv_evt ch2) (fun x -> Right x),
]
```

---

## Control Flow

### If Expressions
```gneiss
if condition then
  expr1
else
  expr2

-- Single line
if x > 0 then "positive" else "non-positive"

-- Chained
if x < 0 then "negative"
else if x == 0 then "zero"
else "positive"
```

### Sequencing
```gneiss
-- Use semicolon for sequencing side effects
let main () =
  print "Hello";
  print "World";
  0
```

---

## Reserved Keywords

```
let, in, fun, match, with, if, then, else,
type, trait, impl, for, where, val, end,
import,
spawn, send, receive, timeout,
true, false, not
```

---

## Operator Precedence (Highest to Lowest)

| Precedence | Operators | Associativity |
|------------|-----------|---------------|
| 9 | Function application | Left |
| 8 | `>>`, `<<` | Left |
| 7 | `*`, `/`, `%` | Left |
| 6 | `+`, `-` | Left |
| 5 | `::`, `++` | Right |
| 4 | `==`, `!=`, `<`, `>`, `<=`, `>=` | None |
| 3 | `&&` | Left |
| 2 | `\|\|` | Left |
| 1 | `\|>`, `<\|` | Left |
| 0 | `=` (binding) | - |

---

## Example Program

### File: `counter.gn`
```gneiss
type Message =
  | Increment
  | Decrement
  | Get (Channel Int)

let start initial =
  spawn (fun () -> loop initial)

let loop count =
  receive
  | Increment -> loop (count + 1)
  | Decrement -> loop (count - 1)
  | Get reply_ch ->
      Channel.send reply_ch count;
      loop count

let increment counter = send counter Increment

let decrement counter = send counter Decrement

let get counter =
  let ch = Channel.new () in
  send counter (Get ch);
  Channel.recv ch
```

### File: `main.gn`
```gneiss
import Counter

let main () =
  let counter = Counter.start 0 in
  Counter.increment counter;
  Counter.increment counter;
  Counter.decrement counter;
  let value = Counter.get counter in
  print ("Counter value: " ++ Int.to_string value)
```
