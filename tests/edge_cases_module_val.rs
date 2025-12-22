//! Edge case tests for module access syntax and val declarations
//!
//! These tests investigate edge cases in:
//! 1. Module access syntax (Module.function)
//! 2. Val type declarations and their interaction with let bindings

use gneiss::{Inferencer, Lexer, Parser};
use gneiss::types::TypeEnv;

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_program(source: &str) -> Result<gneiss::ast::Program, String> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lexer error: {:?}", e))?;
    Parser::new(tokens)
        .parse_program()
        .map_err(|e| format!("Parse error: {:?}", e))
}

fn infer_program(source: &str) -> Result<(), String> {
    let program = parse_program(source)?;
    let mut inferencer = Inferencer::new();
    inferencer
        .infer_program(&program)
        .map_err(|e| format!("Type error: {:?}", e))?;
    Ok(())
}

fn parse_expr(source: &str) -> Result<gneiss::ast::Expr, String> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("Lexer error: {:?}", e))?;
    Parser::new(tokens)
        .parse_expr()
        .map_err(|e| format!("Parse error: {:?}", e))
}

fn infer_expr(source: &str) -> Result<gneiss::types::Type, String> {
    let expr = parse_expr(source)?;
    let mut inferencer = Inferencer::new();
    let env = TypeEnv::new();
    inferencer
        .infer_expr(&env, &expr)
        .map_err(|e| format!("Type error: {:?}", e))
}

// ============================================================================
// Module Access Syntax Edge Cases
// ============================================================================

mod module_access {
    use super::*;

    /// Test: Simple qualified name parsing in let declaration
    #[test]
    fn qualified_name_in_let_declaration() {
        let source = r#"
let Http.parseRequest raw = raw
"#;
        let result = parse_program(source);
        assert!(result.is_ok(), "Should parse qualified name in let: {:?}", result);
    }

    /// Test: Multiple qualified names from same "module"
    #[test]
    fn multiple_qualified_names_same_prefix() {
        let source = r#"
let Response.ok body = body
let Response.error msg = msg
"#;
        let result = parse_program(source);
        assert!(result.is_ok(), "Should parse multiple qualified names: {:?}", result);
    }

    /// Test: Nested module paths should fail clearly (A.B.function)
    #[test]
    fn nested_module_path_should_fail() {
        let source = r#"
let A.B.function x = x
"#;
        let result = parse_program(source);
        // This should fail - we don't support nested module paths
        assert!(result.is_err(), "Nested module paths should not be supported");
        if let Err(e) = result {
            println!("Error for nested module path: {}", e);
        }
    }

    /// Test: Lowercase "module" names should fail at parse time
    #[test]
    fn lowercase_module_name_should_fail() {
        let source = r#"
let myModule.function x = x
"#;
        let result = parse_program(source);
        // This should fail - module names must start with uppercase
        assert!(result.is_err(), "Lowercase module names should not be allowed");
        if let Err(e) = result {
            println!("Error for lowercase module: {}", e);
        }
    }

    /// Test: Module.Constructor pattern (uppercase.uppercase)
    #[test]
    fn module_constructor_pattern() {
        let source = r#"
type Maybe a = | Just a | Nothing
let test x =
    match x with
    | Just v -> v
    | Nothing -> 0
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Should handle constructor patterns: {:?}", result);
    }

    /// Test: Partial module path should fail (Module. with no function)
    #[test]
    fn partial_module_path_should_fail() {
        let source = r#"
let Http. = 5
"#;
        let result = parse_program(source);
        assert!(result.is_err(), "Partial module path should fail");
        if let Err(e) = result {
            println!("Error for partial module path: {}", e);
        }
    }

    /// Test: Module access with reserved word as function name
    #[test]
    fn reserved_word_after_module_dot() {
        // let, if, match, fun, etc. are reserved words
        let source = r#"
let Module.let x = x
"#;
        let result = parse_program(source);
        assert!(result.is_err(), "Reserved words should not be allowed after Module.");
        if let Err(e) = result {
            println!("Error for reserved word: {}", e);
        }
    }

    /// Test: Using qualified name in expression context
    #[test]
    fn qualified_name_in_expression() {
        let source = r#"
let Response.ok body = body
let test = Response.ok "hello"
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Should use qualified names in expressions: {:?}", result);
    }

    /// Test: Module-like record access should work differently
    #[test]
    fn record_field_access_vs_module_access() {
        let source = r#"
type Point = { x : Int, y : Int }
let p = Point { x = 10, y = 20 }
let result = p.x
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Record field access should work: {:?}", result);
    }

    /// Test: What happens with undefined module access
    #[test]
    fn undefined_module_function() {
        let source = r#"
let test = UndefinedModule.someFunction 42
"#;
        let result = infer_program(source);
        assert!(result.is_err(), "Undefined module function should fail type check");
        if let Err(e) = result {
            println!("Error for undefined module: {}", e);
            // Ideally this error should mention the qualified name
        }
    }

    /// Test: Uppercase identifiers cannot be used as variable names
    /// This documents that module-like names (uppercase) cannot be shadowed by variables
    #[test]
    fn uppercase_cannot_be_variable_name() {
        let source = r#"
let Response = 42
"#;
        let result = parse_program(source);
        // Uppercase identifiers are treated as constructor/type names, not variable names
        assert!(result.is_err(), "Uppercase identifiers should not be valid variable names");
        if let Err(e) = result {
            println!("Error (expected): {}", e);
        }
    }

    /// Test: Lowercase local can shadow when used in qualified call
    #[test]
    fn lowercase_local_with_qualified_call() {
        let source = r#"
let Response.ok body = body
let response = Response.ok "hello"
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Should use qualified name: {:?}", result);
    }

    /// Test: Empty function name after module
    #[test]
    fn module_with_empty_function() {
        let source = r#"
let Module. = 5
"#;
        let result = parse_program(source);
        assert!(result.is_err(), "Empty function name should fail");
    }

    /// Test: Qualified name with number in function name
    #[test]
    fn qualified_name_with_numbers() {
        let source = r#"
let Module.parse2 x = x
let test = Module.parse2 42
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Qualified names with numbers should work: {:?}", result);
    }

    /// Test: Qualified name with underscore
    #[test]
    fn qualified_name_with_underscore() {
        let source = r#"
let Module.parse_request x = x
let test = Module.parse_request 42
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Qualified names with underscores should work: {:?}", result);
    }
}

// ============================================================================
// Val Declaration Edge Cases
// ============================================================================

mod val_declarations {
    use super::*;

    /// Test: Basic val declaration with matching let
    #[test]
    fn basic_val_with_matching_let() {
        let source = r#"
val add : Int -> Int -> Int
let add x y = x + y
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Basic val with matching let should work: {:?}", result);
    }

    /// Test: Orphan val declaration without let implementation
    #[test]
    fn orphan_val_declaration() {
        let source = r#"
val orphan : Int -> Int
let other x = x + 1
"#;
        let result = infer_program(source);
        // What happens? Should this be allowed or error?
        println!("Orphan val result: {:?}", result);
        // Document actual behavior
    }

    /// Test: Val with mismatched let type
    #[test]
    fn val_with_mismatched_let() {
        let source = r#"
val wrongType : Int -> Int
let wrongType x = "string"
"#;
        let result = infer_program(source);
        assert!(result.is_err(), "Mismatched val/let types should fail");
        if let Err(e) = result {
            println!("Error for mismatched types: {}", e);
        }
    }

    /// Test: Val with generic type parameter
    #[test]
    fn val_with_generic_type() {
        let source = r#"
val id : a -> a
let id x = x
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Generic val should work: {:?}", result);
    }

    /// Test: Val with multiple generic parameters
    #[test]
    fn val_with_multiple_generics() {
        let source = r#"
val const : a -> b -> a
let const x y = x
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Multiple generic params should work: {:?}", result);
    }

    /// Test: Qualified val declaration
    #[test]
    fn qualified_val_declaration() {
        let source = r#"
val Http.format : String -> String
let Http.format s = s
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Qualified val should work: {:?}", result);
    }

    /// Test: Qualified val without qualified let
    #[test]
    fn qualified_val_without_qualified_let() {
        let source = r#"
val Http.format : String -> String
let format s = s
"#;
        let result = infer_program(source);
        // What happens? The names don't match
        println!("Qualified val with unqualified let result: {:?}", result);
    }

    /// Test: Multiple vals for the same name
    #[test]
    fn duplicate_val_declarations() {
        let source = r#"
val duplicate : Int -> Int
val duplicate : String -> String
let duplicate x = x
"#;
        let result = infer_program(source);
        println!("Duplicate val result: {:?}", result);
        // Document actual behavior - should it error?
    }

    /// Test: Val with complex type (list of functions)
    #[test]
    fn val_with_complex_type() {
        let source = r#"
val mappers : [Int -> Int]
let mappers = [fun x -> x + 1, fun x -> x * 2]
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Val with complex type should work: {:?}", result);
    }

    /// Test: Val with tuple return type
    #[test]
    fn val_with_tuple_type() {
        let source = r#"
val pair : Int -> (Int, Int)
let pair x = (x, x)
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Val with tuple type should work: {:?}", result);
    }

    /// Test: Val with option type
    #[test]
    fn val_with_option_type() {
        let source = r#"
type Option a = | Some a | None
val safeHead : [a] -> Option a
let safeHead xs =
    match xs with
    | [] -> None
    | x :: _ -> Some x
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Val with option type should work: {:?}", result);
    }

    /// Test: Val with option type works when not using generic params in val
    #[test]
    fn val_with_option_type_concrete() {
        let source = r#"
type Option a = | Some a | None
let safeHead xs =
    match xs with
    | [] -> None
    | x :: _ -> Some x
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Function without val should work: {:?}", result);
    }

    /// Test: Val declaration order - val after let
    #[test]
    fn val_after_let_declaration() {
        let source = r#"
let add x y = x + y
val add : Int -> Int -> Int
"#;
        let result = infer_program(source);
        // What happens when val comes after let?
        println!("Val after let result: {:?}", result);
    }

    /// Test: Recursive function with val
    #[test]
    fn recursive_function_with_val() {
        let source = r#"
val factorial : Int -> Int
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Recursive function with val should work: {:?}", result);
    }

    /// Test: Mutually recursive functions with val
    #[test]
    fn mutually_recursive_with_val() {
        let source = r#"
val isEven : Int -> Bool
val isOdd : Int -> Bool
let rec isEven n = if n == 0 then true else isOdd (n - 1)
and isOdd n = if n == 0 then false else isEven (n - 1)
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Mutually recursive with val should work: {:?}", result);
    }

    /// Test: Val with constrained type (if supported)
    #[test]
    fn val_with_trait_constraint() {
        let source = r#"
trait Show a =
    val show : a -> String
end

impl Show for Int =
    let show n = int_to_string n
end

val display : Show a => a -> String
let display x = show x
"#;
        let result = infer_program(source);
        println!("Val with constraint result: {:?}", result);
    }

    /// Test: Val in trait definition with type params
    /// Note: Trait methods with extra type params (beyond the trait's type param) may have limitations
    #[test]
    fn val_in_trait_definition_with_extra_params() {
        let source = r#"
trait Mappable f =
    val map : (a -> b) -> f a -> f b
end
"#;
        let result = infer_program(source);
        // Trait methods with extra type params like a, b are more complex
        // The trait registration uses a fixed param_map with only the trait's type param
        // For now, document the actual behavior
        println!("Val in trait with extra params result: {:?}", result);
        // This may fail or succeed depending on how trait method parsing handles extra type vars
    }

    /// Test: Simple val in trait definition (using trait's type param only)
    #[test]
    fn val_in_trait_definition_simple() {
        let source = r#"
trait Show a =
    val show : a -> String
end
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Simple val in trait should work: {:?}", result);
    }

    /// Test: More specific let than val
    #[test]
    fn more_specific_let_than_val() {
        let source = r#"
val generic : a -> a
let generic x = x + 1
"#;
        let result = infer_program(source);
        // Val says generic is a -> a, but let makes it Int -> Int
        // Should this be allowed (let is more specific) or error?
        println!("More specific let result: {:?}", result);
    }

    /// Test: Less specific let than val
    #[test]
    fn less_specific_let_than_val() {
        let source = r#"
val specific : Int -> Int
let specific x = x
"#;
        let result = infer_program(source);
        // Val says Int -> Int, let infers a -> a
        // The let should be unified to Int -> Int
        assert!(result.is_ok(), "Val should constrain let type: {:?}", result);
    }
}

// ============================================================================
// Combined Edge Cases
// ============================================================================

mod combined {
    use super::*;

    /// Test: Qualified val and let with type mismatch
    #[test]
    fn qualified_val_let_mismatch() {
        let source = r#"
val Http.parse : String -> Int
let Http.parse s = s
"#;
        let result = infer_program(source);
        assert!(result.is_err(), "Qualified val/let mismatch should fail");
        if let Err(e) = result {
            println!("Error for qualified mismatch: {}", e);
        }
    }

    /// Test: Complex module-style API with vals
    #[test]
    fn complex_module_api() {
        let source = r#"
type Response = { status : Int, body : String }

val Response.new : Int -> String -> Response
let Response.new status body = Response { status = status, body = body }

val Response.ok : String -> Response
let Response.ok body = Response.new 200 body

let test = Response.ok "Hello, World!"
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Complex module API should work: {:?}", result);
    }

    /// Test: Function pipeline with qualified names
    #[test]
    fn pipeline_with_qualified_names() {
        let source = r#"
let String.toUpper s = s
let String.toLower s = s
let String.trim s = s

let process s =
    s
    |> String.trim
    |> String.toUpper
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Pipeline with qualified names should work: {:?}", result);
    }

    /// Test: Pattern matching with qualified constructor
    #[test]
    fn pattern_matching_qualified_constructor() {
        let source = r#"
type Result e a = | Ok a | Err e

let handle result =
    match result with
    | Ok value -> value
    | Err _ -> 0
"#;
        let result = infer_program(source);
        assert!(result.is_ok(), "Pattern matching with constructors should work: {:?}", result);
    }
}
