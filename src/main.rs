//! Gneiss CLI - REPL and file execution

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

use gneiss::{Inferencer, Interpreter, Lexer, Parser, TypeEnv};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Run a file
        run_file(&args[1]);
    } else {
        // Start REPL
        repl();
    }
}

fn run_file(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return;
        }
    };

    // Tokenize
    let tokens = match Lexer::new(&source).tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            return;
        }
    };

    // Parse
    let program = match Parser::new(tokens).parse_program() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

    // Type check
    let mut inferencer = Inferencer::new();
    match inferencer.infer_program(&program) {
        Ok(env) => {
            // Print inferred types
            println!("=== Types ===");
            for item in &program.items {
                if let gneiss::ast::Item::Decl(gneiss::ast::Decl::Let { name, .. }) = item {
                    if let Some(scheme) = env.get(name) {
                        println!("  {} : {}", name, scheme);
                    }
                }
            }
            println!();
        }
        Err(e) => {
            eprintln!("Type error: {}", e);
            return;
        }
    }

    // Evaluate
    println!("=== Running ===");
    let mut interpreter = Interpreter::new();
    match interpreter.run(&program) {
        Ok(result) => {
            println!("\n=== Result ===");
            print_value(&result);
            println!();
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
        }
    }
}

fn repl() {
    println!("Gneiss v0.1.0 - Type :help for help, :quit to exit");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut interpreter = Interpreter::new();
    let mut type_env = TypeEnv::new();
    let mut inferencer = Inferencer::new();

    loop {
        print!("gneiss> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Handle commands
        if line.starts_with(':') {
            match line {
                ":quit" | ":q" => break,
                ":help" | ":h" => {
                    println!("Commands:");
                    println!("  :quit, :q    Exit the REPL");
                    println!("  :help, :h    Show this help");
                    println!("  :type <expr> Show the type of an expression");
                    println!();
                    continue;
                }
                cmd if cmd.starts_with(":type ") || cmd.starts_with(":t ") => {
                    let expr_str = cmd.split_once(' ').unwrap().1;
                    match eval_type(expr_str, &type_env, &mut inferencer) {
                        Ok(ty) => println!("{}", ty),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                    continue;
                }
                _ => {
                    eprintln!("Unknown command: {}", line);
                    continue;
                }
            }
        }

        // Try to parse as a declaration first
        let tokens = match Lexer::new(line).tokenize() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Lexer error: {}", e);
                continue;
            }
        };

        // Try as declaration
        let mut parser = Parser::new(tokens.clone());
        if let Ok(program) = parser.parse_program() {
            if !program.items.is_empty() {
                // Type check and add to environment
                match inferencer.infer_program(&program) {
                    Ok(new_env) => {
                        for (name, scheme) in new_env.bindings() {
                            type_env.insert(name.clone(), scheme.clone());
                            println!("{} : {}", name, scheme);
                        }
                    }
                    Err(e) => {
                        eprintln!("Type error: {}", e);
                        continue;
                    }
                }

                // Evaluate
                match interpreter.run(&program) {
                    Ok(_) => {}
                    Err(e) => eprintln!("Runtime error: {}", e),
                }
                continue;
            }
        }

        // Try as expression
        let mut parser = Parser::new(tokens);
        match parser.parse_expr() {
            Ok(expr) => {
                // Type check
                match inferencer.infer_expr(&type_env, &expr) {
                    Ok(ty) => {
                        // Evaluate
                        let env = gneiss::eval::EnvInner::new();
                        match interpreter.eval_expr(&env, &expr) {
                            Ok(val) => {
                                print_value(&val);
                                println!(" : {}", ty);
                            }
                            Err(e) => eprintln!("Runtime error: {}", e),
                        }
                    }
                    Err(e) => eprintln!("Type error: {}", e),
                }
            }
            Err(e) => eprintln!("Parse error: {}", e),
        }
    }

    println!("Goodbye!");
}

fn eval_type(
    expr_str: &str,
    env: &TypeEnv,
    inferencer: &mut Inferencer,
) -> Result<String, String> {
    let tokens = Lexer::new(expr_str)
        .tokenize()
        .map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr().map_err(|e| e.to_string())?;
    let ty = inferencer
        .infer_expr(env, &expr)
        .map_err(|e| e.to_string())?;
    Ok(ty.to_string())
}

fn print_value(val: &gneiss::Value) {
    match val {
        gneiss::Value::Int(n) => print!("{}", n),
        gneiss::Value::Float(f) => print!("{}", f),
        gneiss::Value::Bool(b) => print!("{}", b),
        gneiss::Value::String(s) => print!("\"{}\"", s),
        gneiss::Value::Char(c) => print!("'{}'", c),
        gneiss::Value::Unit => print!("()"),
        gneiss::Value::List(items) => {
            print!("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print_value(item);
            }
            print!("]");
        }
        gneiss::Value::Tuple(items) => {
            print!("(");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print_value(item);
            }
            print!(")");
        }
        gneiss::Value::Closure { .. } => print!("<function>"),
        gneiss::Value::Constructor { name, fields } => {
            print!("{}", name);
            for field in fields {
                print!(" ");
                print_value(field);
            }
        }
        gneiss::Value::Pid(pid) => print!("<pid:{}>", pid),
        gneiss::Value::Channel(id) => print!("<channel:{}>", id),
        gneiss::Value::Builtin(name) => print!("<builtin:{}>", name),
        gneiss::Value::Continuation { .. } => print!("<continuation>"),
    }
}

// Helper trait to access TypeEnv internals for REPL
trait TypeEnvExt {
    fn bindings(&self) -> impl Iterator<Item = (&String, &gneiss::types::Scheme)>;
}

impl TypeEnvExt for TypeEnv {
    fn bindings(&self) -> impl Iterator<Item = (&String, &gneiss::types::Scheme)> {
        // This is a bit of a hack - in a real implementation we'd expose this properly
        std::iter::empty()
    }
}
