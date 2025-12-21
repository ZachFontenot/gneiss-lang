//! Gneiss CLI - REPL and file execution

use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use gneiss::ast::{ImportSpec, Item};
use gneiss::errors::{format_header, format_snippet, format_suggestions, Colors};
use gneiss::infer::TypeError;
use gneiss::lexer::LexError;
use gneiss::module::ModuleResolver;
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
    let entry_path = Path::new(path);
    let colors = Colors::new(std::io::stderr().is_terminal());

    // Build search paths: entry file's directory + stdlib path (if found)
    let mut search_paths = Vec::new();
    if let Some(parent) = entry_path.parent() {
        search_paths.push(parent.to_path_buf());
    } else {
        search_paths.push(PathBuf::from("."));
    }

    // Add stdlib path if found (Phase 3 will make this more robust)
    if let Some(stdlib_path) = find_stdlib_path() {
        search_paths.push(stdlib_path);
    }

    // Load entry module and all dependencies
    let mut resolver = ModuleResolver::new(search_paths.clone());
    let entry_id = match resolver.load_module(entry_path) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Module error: {}", e);
            return;
        }
    };

    // Get all modules in load order (dependencies first)
    let load_order = resolver.load_order.clone();

    // Type check and evaluate all modules
    let mut inferencer = Inferencer::new();
    let mut interpreter = Interpreter::new();

    // Type-check each module in dependency order (dependencies first)
    let mut module_type_envs: Vec<(String, TypeEnv)> = Vec::new();

    for &module_id in &load_order {
        let module = resolver.graph.get(module_id).unwrap();
        let module_path_str = module.path.display().to_string();

        // Read source for error formatting
        let source = match fs::read_to_string(&module.path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {}", module_path_str, e);
                return;
            }
        };
        let source_map = SourceMap::new(&source);

        // Set up imports for this module in the inferencer
        setup_inferencer_imports(&mut inferencer, &module.imports);

        // Type check this module
        let type_env = match inferencer.infer_program(&module.program) {
            Ok(env) => env,
            Err(e) => {
                eprint!(
                    "{}",
                    format_type_error(&e, &source_map, Some(&module_path_str), &colors)
                );
                return;
            }
        };

        // Collect exported types and register this module's exports
        let exported_types = collect_exported_types(&type_env, &module.exports);
        inferencer.register_module(module.name.clone(), exported_types);

        module_type_envs.push((module.name.clone(), type_env));
    }

    // Get the entry module's type env for display
    let entry_module = resolver.graph.get(entry_id).unwrap();
    let type_env = module_type_envs
        .iter()
        .find(|(name, _)| name == &entry_module.name)
        .map(|(_, env)| env.clone())
        .unwrap_or_default();

    // Pass class env and type ctx to interpreter
    interpreter.set_class_env(inferencer.take_class_env());
    interpreter.set_type_ctx(inferencer.take_type_ctx());

    // Print types for entry module before running
    let entry_module = resolver.graph.get(entry_id).unwrap();
    println!("=== Types ===");
    for item in &entry_module.program.items {
        if let Item::Decl(gneiss::ast::Decl::Let { name, .. }) = item {
            if let Some(scheme) = type_env.get(name) {
                println!("  {} : {}", name, scheme);
            }
        }
    }
    println!();
    println!("=== Running ===");

    // Second pass: evaluate all modules (dependencies first)
    for &module_id in &load_order {
        let module = resolver.graph.get(module_id).unwrap();
        let is_entry = module_id == entry_id;

        // Set up imports for this module
        setup_imports(&mut interpreter, &module.imports, &resolver);

        // Run the module
        match interpreter.run(&module.program) {
            Ok(result) => {
                // Register this module's exports for other modules to import
                register_module_exports(&mut interpreter, &module.name, &module.exports);

                // Only print result for entry module
                if is_entry {
                    println!("\n=== Result ===");
                    print_value(&result);
                    println!();
                }
            }
            Err(e) => {
                eprintln!("Runtime error in {}: {}", module.name, e);
                return;
            }
        }
    }
}

/// Find the stdlib path (MVP: just check relative to executable)
fn find_stdlib_path() -> Option<PathBuf> {
    // Check GNEISS_STDLIB env var first
    if let Ok(path) = std::env::var("GNEISS_STDLIB") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    // Check relative to executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // Try ../stdlib (for development)
            let dev_stdlib = exe_dir.join("../stdlib");
            if dev_stdlib.exists() {
                return Some(dev_stdlib);
            }

            // Try ../../stdlib (for release builds in target/release)
            let dev_stdlib2 = exe_dir.join("../../stdlib");
            if dev_stdlib2.exists() {
                return Some(dev_stdlib2);
            }
        }
    }

    // Check current working directory
    let cwd_stdlib = PathBuf::from("stdlib");
    if cwd_stdlib.exists() {
        return Some(cwd_stdlib);
    }

    None
}

/// Set up imports in the interpreter for a module
fn setup_imports(interpreter: &mut Interpreter, imports: &[ImportSpec], _resolver: &ModuleResolver) {
    interpreter.clear_imports();

    for import in imports {
        let module_name = import_path_to_module_name(&import.module_path);

        if let Some(alias) = &import.alias {
            // import Module as Alias
            interpreter.add_module_alias(alias.clone(), module_name.clone());
        }

        if let Some(items) = &import.items {
            // Selective import: import Module (a, b as c)
            for (name, alias) in items {
                let local_name = alias.as_ref().unwrap_or(name).clone();
                interpreter.add_import(local_name, module_name.clone(), name.clone());
            }
        } else if import.alias.is_none() {
            // Whole module import without alias: import Module
            // This makes Module.x available via qualified access
            // The module is already registered, so nothing more needed here
        }
    }
}

/// Convert import path to module name (e.g., "Collections/HashMap" -> "CollectionsHashMap")
fn import_path_to_module_name(path: &str) -> String {
    // For now, just replace / with nothing to get a flat module name
    // The module resolver registers modules by their file-derived name
    path.replace("/", "")
}

/// Register a module's exports so other modules can import them
fn register_module_exports(interpreter: &mut Interpreter, module_name: &str, exports: &HashSet<String>) {
    // Get the current global environment which has the module's bindings
    // and create a new env with just the exports
    let exports_env = interpreter.create_exports_env(exports);
    interpreter.register_module(module_name.to_string(), exports_env);
}

/// Set up imports in the inferencer for type checking a module
fn setup_inferencer_imports(inferencer: &mut Inferencer, imports: &[ImportSpec]) {
    inferencer.clear_imports();

    for import in imports {
        let module_name = import_path_to_module_name(&import.module_path);

        if let Some(alias) = &import.alias {
            // import Module as Alias
            inferencer.add_module_alias(alias.clone(), module_name.clone());
        }

        if let Some(items) = &import.items {
            // Selective import: import Module (a, b as c)
            for (name, alias) in items {
                let local_name = alias.as_ref().unwrap_or(name).clone();
                inferencer.add_import(local_name, module_name.clone(), name.clone());
            }
        }
    }
}

/// Collect exported types from a type environment
fn collect_exported_types(type_env: &TypeEnv, exports: &HashSet<String>) -> TypeEnv {
    let mut exported = TypeEnv::new();
    for name in exports {
        if let Some(scheme) = type_env.get(name) {
            exported.insert(name.clone(), scheme.clone());
        }
    }
    exported
}

fn repl() {
    println!("Gneiss v0.1.0 - Type :help for help, :quit to exit");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let colors = Colors::new(std::io::stderr().is_terminal());

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
                let source_map = SourceMap::new(line);
                eprint!("{}", format_lex_error(&e, &source_map, None, &colors));
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
                        let source_map = SourceMap::new(line);
                        eprint!("{}", format_type_error(&e, &source_map, None, &colors));
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
                    Err(e) => {
                        let source_map = SourceMap::new(line);
                        eprint!("{}", format_type_error(&e, &source_map, None, &colors));
                    }
                }
            }
            Err(e) => {
                let source_map = SourceMap::new(line);
                eprint!("{}", format_parse_error(&e, &source_map, None, &colors));
            }
        }
    }

    println!("Goodbye!");
}

fn eval_type(expr_str: &str, env: &TypeEnv, inferencer: &mut Inferencer) -> Result<String, String> {
    let tokens = Lexer::new(expr_str).tokenize().map_err(|e| e.to_string())?;
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
        gneiss::Value::ComposedFn { .. } => print!("<function>"),
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
        gneiss::Value::BuiltinPartial { name, args } => {
            print!("<builtin-partial:{}({} args)>", name, args.len())
        }
        gneiss::Value::Continuation { .. } => print!("<continuation>"),
        gneiss::Value::Dict { trait_name, .. } => print!("<dict:{}>", trait_name),
        gneiss::Value::Fiber(id) => print!("<fiber:{}>", id),
        gneiss::Value::FiberEffect(effect) => print!("<fiber-effect:{:?}>", effect),
        gneiss::Value::Record { type_name, fields } => {
            print!("{} {{ ", type_name);
            let mut first = true;
            for (field_name, field_value) in fields {
                if !first {
                    print!(", ");
                }
                first = false;
                print!("{} = ", field_name);
                print_value(field_value);
            }
            print!(" }}");
        }
        gneiss::Value::Dictionary(dict) => {
            print!("{{");
            let mut first = true;
            for (key, value) in dict {
                if !first {
                    print!(", ");
                }
                first = false;
                print!("\"{}\": ", key);
                print_value(value);
            }
            print!("}}");
        }
        gneiss::Value::Set(set) => {
            print!("Set {{");
            let mut first = true;
            for elem in set {
                if !first {
                    print!(", ");
                }
                first = false;
                print!("\"{}\"", elem);
            }
            print!("}}");
        }
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

fn format_type_error(
    err: &TypeError,
    source_map: &SourceMap,
    filename: Option<&str>,
    colors: &Colors,
) -> String {
    let mut out = String::new();

    // Header
    let (header, msg, span, suggestions) = match err {
        TypeError::UnboundVariable {
            name,
            span,
            suggestions,
        } => (
            "NAME ERROR",
            format!("I cannot find a variable named `{}`.", name),
            Some(span),
            suggestions.clone(),
        ),
        TypeError::TypeMismatch {
            expected,
            found,
            span,
            context,
        } => {
            // Use user-friendly display with normalized variable names and hidden answer types
            let expected_str = expected.display_user_friendly();
            let found_str = found.display_user_friendly();

            let msg = match context {
                Some(ctx) => {
                    // Context-aware message that explains what's happening
                    format!(
                        "Type mismatch {}.\n\n\
                         The context expects:  {}{}{}\n\
                         But this has type:    {}{}{}",
                        ctx,
                        colors.cyan(),
                        expected_str,
                        colors.reset(),
                        colors.red(),
                        found_str,
                        colors.reset()
                    )
                }
                None => {
                    // Generic message when no context available
                    format!(
                        "I found a type mismatch.\n\n\
                         One part has type:    {}{}{}\n\
                         Another part has:     {}{}{}\n\n\
                         These types are not compatible.",
                        colors.cyan(),
                        expected_str,
                        colors.reset(),
                        colors.red(),
                        found_str,
                        colors.reset()
                    )
                }
            };
            ("TYPE ERROR", msg, span.as_ref(), vec![])
        }
        TypeError::OccursCheck {
            var_id: _,
            ty,
            span,
        } => {
            // Use normalized display for occurs check too
            let ty_str = ty.display_user_friendly();
            let msg = format!(
                "I detected an infinite type.\n\n\
                 The type `{}` refers to itself, which would create an infinitely nested type.\n\n\
                 This often happens when:\n\
                 - A function is applied to itself\n\
                 - A recursive function uses `shift` (which requires answer-type polymorphism)",
                ty_str
            );
            ("TYPE ERROR", msg, span.as_ref(), vec![])
        }
        TypeError::UnknownConstructor {
            name,
            span,
            suggestions,
        } => (
            "NAME ERROR",
            format!("I cannot find a constructor named `{}`.", name),
            Some(span),
            suggestions.clone(),
        ),
        TypeError::PatternMismatch { span } => (
            "PATTERN ERROR",
            "I found a pattern that doesn't make sense here.".to_string(),
            Some(span),
            vec![],
        ),
        TypeError::NonExhaustivePatterns { span } => (
            "PATTERN ERROR",
            "This match expression doesn't cover all possible cases.".to_string(),
            Some(span),
            vec![],
        ),
        TypeError::UnknownTrait { name, span } => (
            "NAME ERROR",
            format!("I cannot find a trait named `{}`.", name),
            span.as_ref(),
            vec![],
        ),
        TypeError::OverlappingInstance {
            trait_name,
            existing,
            new,
            span,
        } => {
            let msg = format!(
                "I found overlapping instances for trait `{}`.\n\n  Existing instance: {}\n  Conflicting:      {}",
                trait_name, existing.display_user_friendly(), new.display_user_friendly()
            );
            ("INSTANCE ERROR", msg, span.as_ref(), vec![])
        }
        TypeError::NoInstance {
            trait_name,
            ty,
            span,
        } => {
            let msg = format!(
                "I cannot find an instance of `{}` for type `{}`.",
                trait_name,
                ty.display_user_friendly()
            );
            ("TYPE ERROR", msg, span.as_ref(), vec![])
        }
        TypeError::UnknownRecordType { name, span } => (
            "NAME ERROR",
            format!("I cannot find a record type named `{}`.", name),
            Some(span),
            vec![],
        ),
        TypeError::MissingRecordField {
            record_type,
            field,
            span,
        } => (
            "RECORD ERROR",
            format!(
                "The record type `{}` requires a field named `{}`.",
                record_type, field
            ),
            Some(span),
            vec![],
        ),
        TypeError::UnknownRecordField {
            record_type,
            field,
            span,
        } => (
            "RECORD ERROR",
            format!(
                "The record type `{}` has no field named `{}`.",
                record_type, field
            ),
            Some(span),
            vec![],
        ),
        TypeError::NotARecordType { ty, span } => (
            "TYPE ERROR",
            format!(
                "I expected a record type, but found `{}`.",
                ty.display_user_friendly()
            ),
            Some(span),
            vec![],
        ),
        TypeError::CannotInferRecordType { span } => (
            "TYPE ERROR",
            "I cannot infer the record type for this update expression.".to_string(),
            Some(span),
            vec![],
        ),
        TypeError::Other(msg) => (
            "ERROR",
            msg.clone(),
            None,
            vec![],
        ),
    };

    out.push_str(&format_header(header, colors));
    out.push('\n');
    out.push('\n');

    // Location
    if let Some(span) = span {
        let pos = source_map.position(span.start);
        let file = filename.unwrap_or("<input>");
        out.push_str(&format!(
            "{}{}:{}{}\n\n",
            colors.bold(),
            file,
            pos,
            colors.reset()
        ));
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

fn format_parse_error(
    err: &ParseError,
    source_map: &SourceMap,
    filename: Option<&str>,
    colors: &Colors,
) -> String {
    let mut out = String::new();

    out.push_str(&format_header("PARSE ERROR", colors));
    out.push('\n');
    out.push('\n');

    let (msg, span) = match err {
        ParseError::UnexpectedToken {
            expected,
            found,
            span,
        } => (
            format!(
                "I was expecting {} but found {:?} instead.",
                expected, found
            ),
            Some(span),
        ),
        ParseError::UnexpectedEof {
            expected,
            last_span,
        } => (
            format!(
                "I reached the end of the file but was expecting {}.",
                expected
            ),
            Some(last_span),
        ),
        ParseError::InvalidPattern { span } => {
            ("I found an invalid pattern here.".to_string(), Some(span))
        }
    };

    // Location
    if let Some(span) = span {
        let pos = source_map.position(span.start);
        let file = filename.unwrap_or("<input>");
        out.push_str(&format!(
            "{}{}:{}{}\n\n",
            colors.bold(),
            file,
            pos,
            colors.reset()
        ));
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

fn format_lex_error(
    err: &LexError,
    source_map: &SourceMap,
    filename: Option<&str>,
    colors: &Colors,
) -> String {
    let mut out = String::new();

    out.push_str(&format_header("SYNTAX ERROR", colors));
    out.push('\n');
    out.push('\n');

    let (msg, span) = match err {
        LexError::UnexpectedChar(c, span) => {
            (format!("I found an unexpected character: '{}'", c), span)
        }
        LexError::UnterminatedString(span) => {
            ("I found a string that was never closed.".to_string(), span)
        }
        LexError::UnterminatedChar(span) => (
            "I found a character literal that was never closed.".to_string(),
            span,
        ),
        LexError::InvalidEscape(c, span) => {
            (format!("I found an invalid escape sequence: \\{}", c), span)
        }
        LexError::InvalidNumber(s, span) => {
            (format!("I could not parse this as a number: {}", s), span)
        }
    };

    // Location
    let pos = source_map.position(span.start);
    let file = filename.unwrap_or("<input>");
    out.push_str(&format!(
        "{}{}:{}{}\n\n",
        colors.bold(),
        file,
        pos,
        colors.reset()
    ));

    // Message
    out.push_str(&msg);
    out.push('\n');

    // Source snippet
    out.push('\n');
    out.push_str(&format_snippet(source_map, span, colors));

    out.push('\n');
    out
}
