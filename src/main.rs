//! Gneiss CLI - REPL and file execution

use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};

use gneiss::errors::{Colors, format_header, format_snippet, format_suggestions};
use gneiss::infer::TypeError;
use gneiss::parser::ParseError;
use gneiss::{Inferencer, Interpreter, Lexer, Parser, SourceMap, TypeEnv};

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

    // Create SourceMap for error formatting
    let source_map = SourceMap::new(&source);
    let colors = Colors::new(std::io::stderr().is_terminal());

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
            eprint!("{}", format_parse_error(&e, &source_map, Some(path), &colors));
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
            eprint!("{}", format_type_error(&e, &source_map, Some(path), &colors));
            return;
        }
    }

    // Evaluate
    println!("=== Running ===");
    let mut interpreter = Interpreter::new();
    // Pass the class environment and type context from type inference to the interpreter
    interpreter.set_class_env(inferencer.take_class_env());
    interpreter.set_type_ctx(inferencer.take_type_ctx());
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
        gneiss::Value::Dict { trait_name, .. } => print!("<dict:{}>", trait_name),
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

// ============================================================================
// Elm-style Error Formatting
// ============================================================================

fn format_type_error(err: &TypeError, source_map: &SourceMap, filename: Option<&str>, colors: &Colors) -> String {
    let mut out = String::new();

    // Header
    let (header, msg, span, suggestions) = match err {
        TypeError::UnboundVariable { name, span, suggestions } => {
            ("NAME ERROR", format!("I cannot find a variable named `{}`.", name), Some(span), suggestions.clone())
        }
        TypeError::TypeMismatch { expected, found, span } => {
            let msg = format!(
                "I found a type mismatch.\n\n  I expected:  {}{}{}\n  But found:   {}{}{}",
                colors.cyan(), expected, colors.reset(),
                colors.red(), found, colors.reset()
            );
            ("TYPE ERROR", msg, span.as_ref(), vec![])
        }
        TypeError::OccursCheck { var_id, ty, span } => {
            let msg = format!("I detected an infinite type: type variable {} occurs in {}.", var_id, ty);
            ("TYPE ERROR", msg, span.as_ref(), vec![])
        }
        TypeError::UnknownConstructor { name, span, suggestions } => {
            ("NAME ERROR", format!("I cannot find a constructor named `{}`.", name), Some(span), suggestions.clone())
        }
        TypeError::PatternMismatch { span } => {
            ("PATTERN ERROR", "I found a pattern that doesn't make sense here.".to_string(), Some(span), vec![])
        }
        TypeError::NonExhaustivePatterns { span } => {
            ("PATTERN ERROR", "This match expression doesn't cover all possible cases.".to_string(), Some(span), vec![])
        }
        TypeError::UnknownTrait { name, span } => {
            ("NAME ERROR", format!("I cannot find a trait named `{}`.", name), span.as_ref(), vec![])
        }
        TypeError::OverlappingInstance { trait_name, existing, new, span } => {
            let msg = format!(
                "I found overlapping instances for trait `{}`.\n\n  Existing instance: {}\n  Conflicting:      {}",
                trait_name, existing, new
            );
            ("INSTANCE ERROR", msg, span.as_ref(), vec![])
        }
        TypeError::NoInstance { trait_name, ty, span } => {
            let msg = format!("I cannot find an instance of `{}` for type `{}`.", trait_name, ty);
            ("TYPE ERROR", msg, span.as_ref(), vec![])
        }
    };

    out.push_str(&format_header(header, colors));
    out.push('\n');
    out.push('\n');

    // Location
    if let Some(span) = span {
        let pos = source_map.position(span.start);
        let file = filename.unwrap_or("<input>");
        out.push_str(&format!("{}{}:{}{}\n\n", colors.bold(), file, pos, colors.reset()));
    }

    // Message
    out.push_str(&msg);

    // Suggestions
    if !suggestions.is_empty() {
        out.push_str(&format_suggestions(&suggestions, colors));
    }

    out.push('\n');

    // Source snippet
    if let Some(span) = span {
        out.push('\n');
        out.push_str(&format_snippet(source_map, span, colors));
    }

    out.push('\n');
    out
}

fn format_parse_error(err: &ParseError, source_map: &SourceMap, filename: Option<&str>, colors: &Colors) -> String {
    let mut out = String::new();

    out.push_str(&format_header("PARSE ERROR", colors));
    out.push('\n');
    out.push('\n');

    let (msg, span) = match err {
        ParseError::UnexpectedToken { expected, found, span } => {
            (format!("I was expecting {} but found {:?} instead.", expected, found), Some(span))
        }
        ParseError::UnexpectedEof { expected, last_span } => {
            (format!("I reached the end of the file but was expecting {}.", expected), Some(last_span))
        }
        ParseError::InvalidPattern { span } => {
            ("I found an invalid pattern here.".to_string(), Some(span))
        }
    };

    // Location
    if let Some(span) = span {
        let pos = source_map.position(span.start);
        let file = filename.unwrap_or("<input>");
        out.push_str(&format!("{}{}:{}{}\n\n", colors.bold(), file, pos, colors.reset()));
    }

    // Message
    out.push_str(&msg);
    out.push('\n');

    // Source snippet
    if let Some(span) = span {
        out.push('\n');
        out.push_str(&format_snippet(source_map, span, colors));
    }

    out.push('\n');
    out
}
