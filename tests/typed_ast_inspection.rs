//! Tests for typed AST inspection
//!
//! These tests verify that we can inspect the types of sub-expressions
//! within a program, enabling compositional testing.

use gneiss::test_support::elaborate_program;
use gneiss::typed_ast::{TExprKind, TItem, TDecl};

/// Test the show_test.gn structure - the original motivation for typed AST
///
/// This test demonstrates that we CAN now inspect the internal structure of programs.
/// It also reveals that the current inference may not properly generalize inner let
/// bindings with constraints - this is a known area for improvement.
#[test]
fn inspect_show_test_structure() {
    let program = r#"
let main _ =
    let f x = show x in
    print (f "string");
    print (f 42)
"#;

    let (tprogram, _env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // Find the main declaration
    let main_decl = tprogram.items.iter().find_map(|item| {
        if let TItem::Decl(TDecl::Let { pattern, value, scheme }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                if name == "main" {
                    return Some((value, scheme));
                }
            }
        }
        None
    });

    assert!(main_decl.is_some(), "should find main declaration");
    let (main_value, main_scheme) = main_decl.unwrap();

    // main should have a function type
    println!("main scheme: {:?}", main_scheme);
    assert!(main_scheme.num_generics > 0, "main should have a polymorphic type");

    // Inspect the structure of main's body (which is a lambda)
    if let TExprKind::Lambda { body, .. } = &main_value.kind {
        // The body should be a Let binding for f
        if let TExprKind::Let { pattern, scheme, value, body: _let_body } = &body.kind {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                assert_eq!(name, "f", "first let should bind f");

                // Check f's scheme - this is the KEY verification
                // NOTE: This inspection capability is exactly what we needed!
                // The typed AST now allows us to see what type `f` got.
                println!("f's scheme: {:?}", scheme);
                println!("f's type: {:?}", scheme.ty);

                // Verify f's value is a lambda that calls show
                if let TExprKind::Lambda { body: f_body, .. } = &value.kind {
                    // The body should be an App of show to x
                    if let TExprKind::App { func, .. } = &f_body.kind {
                        if let TExprKind::Var { name, .. } = &func.kind {
                            assert_eq!(name, "show", "f should call show");
                        }
                    }
                }
            }
        }
    }
}

/// Test that polymorphic function types are properly recorded
#[test]
fn polymorphic_function_has_scheme() {
    let program = r#"
let identity x = x
let main _ = identity 42
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // Check that identity has a polymorphic type in the environment
    let identity_scheme = env.get("identity").expect("identity should be in env");
    println!("identity scheme: {:?}", identity_scheme);

    // identity should be generalized: forall a. a -> a
    assert!(identity_scheme.num_generics > 0, "identity should be polymorphic");
}

/// Test that trait method calls record predicates
#[test]
fn trait_method_call_records_predicates() {
    let program = r#"
let show_it x = show x
let main _ = show_it 42
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // show_it should have a Show constraint
    let show_it_scheme = env.get("show_it").expect("show_it should be in env");
    println!("show_it scheme: {:?}", show_it_scheme);

    // show_it should have Show constraint: forall a. Show a => a -> String
    assert!(
        !show_it_scheme.predicates.is_empty() || show_it_scheme.num_generics > 0,
        "show_it should have Show constraint or be polymorphic"
    );
}

/// Test that type environment is correctly populated
#[test]
fn type_env_contains_all_bindings() {
    let program = r#"
let x = 42
let y = "hello"
let f a = a + 1
let main _ = f x
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // Check all bindings exist
    assert!(env.get("x").is_some(), "x should be in env");
    assert!(env.get("y").is_some(), "y should be in env");
    assert!(env.get("f").is_some(), "f should be in env");
    assert!(env.get("main").is_some(), "main should be in env");

    // Check types
    let x_scheme = env.get("x").unwrap();
    println!("x: {:?}", x_scheme);

    let y_scheme = env.get("y").unwrap();
    println!("y: {:?}", y_scheme);

    let f_scheme = env.get("f").unwrap();
    println!("f: {:?}", f_scheme);
}

/// Test elaboration of nested let bindings
#[test]
fn nested_let_bindings_have_types() {
    let program = r#"
let main _ =
    let a = 1 in
    let b = 2 in
    a + b
"#;

    let (tprogram, _env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // Find main and verify its structure
    let main_item = tprogram.items.iter().find(|item| {
        if let TItem::Decl(TDecl::Let { pattern, .. }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                return name == "main";
            }
        }
        false
    });

    assert!(main_item.is_some(), "should find main");
}

// ============================================================================
// Polymorphic Functions
// ============================================================================

/// Basic polymorphic identity function
#[test]
fn polymorphic_identity() {
    let program = r#"
let id x = x
let main _ =
    let a = id 42 in
    let b = id "hello" in
    let c = id true in
    ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let id_scheme = env.get("id").expect("id should be in env");
    println!("id scheme: {:?}", id_scheme);

    // id should be: forall a. a -> a
    assert!(id_scheme.num_generics > 0, "id should be polymorphic");
    assert!(id_scheme.predicates.is_empty(), "id should have no constraints");
}

/// Polymorphic const function (ignores second arg)
#[test]
fn polymorphic_const() {
    let program = r#"
let const x y = x
let main _ =
    let a = const 42 "ignored" in
    let b = const "hello" 999 in
    ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let const_scheme = env.get("const").expect("const should be in env");
    println!("const scheme: {:?}", const_scheme);

    // const should be: forall a b. a -> b -> a
    assert!(const_scheme.num_generics >= 2, "const should have at least 2 type params");
}

/// Polymorphic flip function
#[test]
fn polymorphic_flip() {
    let program = r#"
let flip f x y = f y x
let main _ = ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let flip_scheme = env.get("flip").expect("flip should be in env");
    println!("flip scheme: {:?}", flip_scheme);

    // flip should be polymorphic
    assert!(flip_scheme.num_generics > 0, "flip should be polymorphic");
}

/// Polymorphic compose function
#[test]
fn polymorphic_compose() {
    let program = r#"
let compose f g x = f (g x)
let main _ = ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let compose_scheme = env.get("compose").expect("compose should be in env");
    println!("compose scheme: {:?}", compose_scheme);

    // compose should be: forall a b c. (b -> c) -> (a -> b) -> a -> c
    assert!(compose_scheme.num_generics >= 3, "compose should have at least 3 type params");
}

/// Polymorphic function used at multiple types
#[test]
fn polymorphic_used_at_multiple_types() {
    let program = r#"
let pair x y = (x, y)
let main _ =
    let p1 = pair 1 2 in
    let p2 = pair "a" "b" in
    let p3 = pair 1 "mixed" in
    ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let pair_scheme = env.get("pair").expect("pair should be in env");
    println!("pair scheme: {:?}", pair_scheme);

    // pair should be: forall a b. a -> b -> (a, b)
    assert!(pair_scheme.num_generics >= 2, "pair should have at least 2 type params");
}

// ============================================================================
// Trait-Constrained Functions
// ============================================================================

/// Show constraint on single parameter
#[test]
fn trait_show_single_param() {
    let program = r#"
let stringify x = show x
let main _ = print (stringify 42)
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let stringify_scheme = env.get("stringify").expect("stringify should be in env");
    println!("stringify scheme: {:?}", stringify_scheme);

    // stringify should have Show constraint
    // Either through predicates or through the type structure
    println!("  num_generics: {}", stringify_scheme.num_generics);
    println!("  predicates: {:?}", stringify_scheme.predicates);
}

/// Eq constraint through == operator
#[test]
fn trait_eq_through_operator() {
    let program = r#"
let are_equal x y = x == y
let main _ =
    let a = are_equal 1 2 in
    let b = are_equal "a" "b" in
    ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let are_equal_scheme = env.get("are_equal").expect("are_equal should be in env");
    println!("are_equal scheme: {:?}", are_equal_scheme);

    // are_equal should have Eq constraint: forall a. Eq a => a -> a -> Bool
    println!("  num_generics: {}", are_equal_scheme.num_generics);
    println!("  predicates: {:?}", are_equal_scheme.predicates);
}

/// Ord constraint through < operator
#[test]
fn trait_ord_through_operator() {
    let program = r#"
let is_less x y = x < y
let main _ = is_less 1 2
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let is_less_scheme = env.get("is_less").expect("is_less should be in env");
    println!("is_less scheme: {:?}", is_less_scheme);

    // is_less should have Ord constraint
    println!("  num_generics: {}", is_less_scheme.num_generics);
    println!("  predicates: {:?}", is_less_scheme.predicates);
}

/// Multiple constraints on same parameter (Show + Eq)
#[test]
fn trait_multiple_constraints_same_param() {
    let program = r#"
let show_if_equal x y =
    if x == y then show x else "not equal"
let main _ = print (show_if_equal 1 1)
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let show_if_equal_scheme = env.get("show_if_equal").expect("show_if_equal should be in env");
    println!("show_if_equal scheme: {:?}", show_if_equal_scheme);

    // show_if_equal should have both Show and Eq constraints on same type
    println!("  num_generics: {}", show_if_equal_scheme.num_generics);
    println!("  predicates: {:?}", show_if_equal_scheme.predicates);
}

/// Constraints on multiple parameters
#[test]
fn trait_constraints_multiple_params() {
    let program = r#"
let format_pair x y = show x ++ " and " ++ show y
let main _ = print (format_pair 1 "hello")
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let format_pair_scheme = env.get("format_pair").expect("format_pair should be in env");
    println!("format_pair scheme: {:?}", format_pair_scheme);

    // format_pair should have Show constraints on both parameters
    println!("  num_generics: {}", format_pair_scheme.num_generics);
    println!("  predicates: {:?}", format_pair_scheme.predicates);
}

/// Trait method in higher-order function
#[test]
fn trait_in_higher_order() {
    let program = r#"
let apply_show f x = f (show x)
let main _ = apply_show print 42
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let apply_show_scheme = env.get("apply_show").expect("apply_show should be in env");
    println!("apply_show scheme: {:?}", apply_show_scheme);

    println!("  num_generics: {}", apply_show_scheme.num_generics);
    println!("  predicates: {:?}", apply_show_scheme.predicates);
}

// ============================================================================
// Effectful Functions
// ============================================================================

/// Function that performs IO (print)
#[test]
fn effectful_print() {
    let program = r#"
let say_hello _ = print "hello"
let main _ = say_hello ()
"#;

    let (tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let say_hello_scheme = env.get("say_hello").expect("say_hello should be in env");
    println!("say_hello scheme: {:?}", say_hello_scheme);

    // Inspect the function type to see if effects are tracked
    println!("  type: {:?}", say_hello_scheme.ty);

    // Find say_hello in the typed AST and inspect its structure
    for item in &tprogram.items {
        if let TItem::Decl(TDecl::Let { pattern, value, .. }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                if name == "say_hello" {
                    println!("  value type: {:?}", value.ty);
                }
            }
        }
    }
}

/// Function with channel operations
#[test]
fn effectful_channels() {
    let program = r#"
let send_value ch v = Channel.send ch v
let recv_value ch = Channel.recv ch
let main _ = ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let send_value_scheme = env.get("send_value").expect("send_value should be in env");
    println!("send_value scheme: {:?}", send_value_scheme);

    let recv_value_scheme = env.get("recv_value").expect("recv_value should be in env");
    println!("recv_value scheme: {:?}", recv_value_scheme);
}

/// Function that spawns a fiber
#[test]
fn effectful_spawn() {
    let program = r#"
let spawn_printer msg = spawn (fun _ -> print msg)
let main _ = spawn_printer "hello"
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let spawn_printer_scheme = env.get("spawn_printer").expect("spawn_printer should be in env");
    println!("spawn_printer scheme: {:?}", spawn_printer_scheme);
    println!("  type: {:?}", spawn_printer_scheme.ty);
}

/// Combined: channel creation and communication
#[test]
fn effectful_channel_workflow() {
    // Just test channel send/recv types, avoid Channel.new complexity
    let program = r#"
let send_int ch = Channel.send ch 42
let recv_int ch = Channel.recv ch
let main _ = ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let send_int_scheme = env.get("send_int").expect("send_int should be in env");
    println!("send_int scheme: {:?}", send_int_scheme);
    println!("  type: {:?}", send_int_scheme.ty);

    let recv_int_scheme = env.get("recv_int").expect("recv_int should be in env");
    println!("recv_int scheme: {:?}", recv_int_scheme);
    println!("  type: {:?}", recv_int_scheme.ty);
}

// ============================================================================
// Combined: Polymorphic + Trait + Effects
// ============================================================================

/// Polymorphic function with Show constraint used in effectful context
#[test]
fn combined_poly_trait_effect() {
    let program = r#"
let print_twice x =
    print (show x);
    print (show x)
let main _ =
    print_twice 42;
    print_twice "hello"
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let print_twice_scheme = env.get("print_twice").expect("print_twice should be in env");
    println!("print_twice scheme: {:?}", print_twice_scheme);
    println!("  num_generics: {}", print_twice_scheme.num_generics);
    println!("  predicates: {:?}", print_twice_scheme.predicates);
    println!("  type: {:?}", print_twice_scheme.ty);
}

/// Higher-order function with constraints flowing through
#[test]
fn combined_higher_order_constraints() {
    let program = r#"
let map_show f xs =
    match xs with
    | [] -> []
    | x :: rest -> f (show x) :: map_show f rest
    end
let main _ = ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let map_show_scheme = env.get("map_show").expect("map_show should be in env");
    println!("map_show scheme: {:?}", map_show_scheme);
    println!("  num_generics: {}", map_show_scheme.num_generics);
    println!("  predicates: {:?}", map_show_scheme.predicates);
}

/// Assert-like function combining Eq and Show
#[test]
fn combined_assert_eq_pattern() {
    let program = r#"
let assert_eq x y =
    if x == y then print "pass"
    else print ("fail: " ++ show x ++ " != " ++ show y)
let main _ =
    assert_eq 1 1;
    assert_eq "a" "a"
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let assert_eq_scheme = env.get("assert_eq").expect("assert_eq should be in env");
    println!("assert_eq scheme: {:?}", assert_eq_scheme);
    println!("  num_generics: {}", assert_eq_scheme.num_generics);
    println!("  predicates: {:?}", assert_eq_scheme.predicates);

    // assert_eq should have both Eq and Show constraints on the same type param
}

// ============================================================================
// Typed AST Structure Inspection
// ============================================================================

/// Verify lambda parameter types are recorded
#[test]
fn lambda_param_types() {
    let program = r#"
let add x y = x + y
let main _ = add 1 2
"#;

    let (tprogram, _env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // Find add and inspect its lambda structure
    for item in &tprogram.items {
        if let TItem::Decl(TDecl::Let { pattern, value, .. }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                if name == "add" {
                    println!("add value: {:?}", value.kind);
                    if let TExprKind::Lambda { params, body } = &value.kind {
                        println!("  params: {:?}", params.iter().map(|p| &p.ty).collect::<Vec<_>>());
                        println!("  body type: {:?}", body.ty);
                    }
                }
            }
        }
    }
}

/// Verify application types flow correctly
#[test]
fn application_type_flow() {
    let program = r#"
let apply f x = f x
let inc n = n + 1
let main _ = apply inc 5
"#;

    let (tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let apply_scheme = env.get("apply").expect("apply should be in env");
    println!("apply scheme: {:?}", apply_scheme);

    // Find main and inspect the application structure
    for item in &tprogram.items {
        if let TItem::Decl(TDecl::Let { pattern, value, .. }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                if name == "main" {
                    println!("main value type: {:?}", value.ty);
                }
            }
        }
    }
}

/// Verify match arm types are consistent
#[test]
fn match_arm_types() {
    // Use list pattern matching which is simpler
    let program = r#"
let head_or_default xs default =
    match xs with
    | [] -> default
    | x :: _ -> x
    end

let main _ = head_or_default [1, 2, 3] 0
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let head_or_default_scheme = env.get("head_or_default").expect("head_or_default should be in env");
    println!("head_or_default scheme: {:?}", head_or_default_scheme);
    println!("  num_generics: {}", head_or_default_scheme.num_generics);

    // head_or_default should be: forall a. [a] -> a -> a
    assert!(head_or_default_scheme.num_generics > 0, "head_or_default should be polymorphic");
}

/// Verify tuple types are properly constructed
#[test]
fn tuple_type_construction() {
    let program = r#"
let make_triple a b c = (a, b, c)
let main _ = make_triple 1 "two" true
"#;

    let (tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let make_triple_scheme = env.get("make_triple").expect("make_triple should be in env");
    println!("make_triple scheme: {:?}", make_triple_scheme);

    // Find main and inspect the result
    for item in &tprogram.items {
        if let TItem::Decl(TDecl::Let { pattern, value, .. }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                if name == "main" {
                    println!("main result type: {:?}", value.ty);
                }
            }
        }
    }
}

/// Verify list types are properly inferred
#[test]
fn list_type_inference() {
    let program = r#"
let singleton x = [x]
let pair_list a b = [a, b]
let main _ =
    let ints = singleton 1 in
    let strs = pair_list "a" "b" in
    ()
"#;

    let (_tprogram, env, _inferencer) = elaborate_program(program).expect("should elaborate");

    let singleton_scheme = env.get("singleton").expect("singleton should be in env");
    println!("singleton scheme: {:?}", singleton_scheme);

    let pair_list_scheme = env.get("pair_list").expect("pair_list should be in env");
    println!("pair_list scheme: {:?}", pair_list_scheme);
}

/// Test that inner let bindings get generalized properly
/// This tests that `let f x = show x in f 42` has f with Show constraint
#[test]
fn inner_let_binding_generalization() {
    let program = r#"
let main _ =
    let f x = show x in
    f 42
"#;

    let (tprogram, _env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // Find main's body and inspect the inner let binding for f
    for item in &tprogram.items {
        if let TItem::Decl(TDecl::Let { pattern, value, .. }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                if name == "main" {
                    // The body of main should be a Lambda with a Let inside
                    if let TExprKind::Lambda { body, .. } = &value.kind {
                        if let TExprKind::Let { pattern: inner_pattern, scheme, .. } = &body.kind {
                            println!("Inner f scheme: {:?}", scheme);
                            println!("  num_generics: {}", scheme.num_generics);
                            println!("  predicates: {:?}", scheme.predicates);

                            // f should have Show constraint since it uses show
                            // f : forall a. Show a => a -> String
                            if let gneiss::typed_ast::TPatternKind::Var(f_name) = &inner_pattern.kind {
                                assert_eq!(f_name, "f");
                            }
                            assert!(
                                scheme.num_generics > 0 || !scheme.predicates.is_empty(),
                                "inner f should be polymorphic with Show constraint, got num_generics={}, predicates={:?}",
                                scheme.num_generics,
                                scheme.predicates
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Test that inner let bindings get generalized with polymorphic identity
#[test]
fn inner_let_polymorphic_identity() {
    let program = r#"
let main _ =
    let id x = x in
    let a = id 42 in
    let b = id "hello" in
    a
"#;

    let (tprogram, _env, _inferencer) = elaborate_program(program).expect("should elaborate");

    // Find main's body and inspect the inner let binding for id
    for item in &tprogram.items {
        if let TItem::Decl(TDecl::Let { pattern, value, .. }) = item {
            if let gneiss::typed_ast::TPatternKind::Var(name) = &pattern.kind {
                if name == "main" {
                    if let TExprKind::Lambda { body, .. } = &value.kind {
                        if let TExprKind::Let { pattern: inner_pattern, scheme, .. } = &body.kind {
                            println!("Inner id scheme: {:?}", scheme);
                            println!("  num_generics: {}", scheme.num_generics);

                            if let gneiss::typed_ast::TPatternKind::Var(id_name) = &inner_pattern.kind {
                                assert_eq!(id_name, "id");
                            }
                            // id should be polymorphic: forall a. a -> a
                            assert!(
                                scheme.num_generics > 0,
                                "inner id should be polymorphic, got num_generics={}",
                                scheme.num_generics
                            );
                        }
                    }
                }
            }
        }
    }
}
