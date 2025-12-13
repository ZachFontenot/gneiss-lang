//! Snapshot tests for error message formatting
//!
//! These tests verify that error messages maintain their Elm-style format.
//! If the format changes intentionally, update the .expected files by running:
//!
//!     UPDATE_SNAPSHOTS=1 cargo test error_snapshots
//!
//! The snapshot files are in tests/snapshots/

use std::fs;

use gneiss::ast::SourceMap;
use gneiss::errors::{Colors, format_header, format_snippet, format_suggestions};
use gneiss::infer::TypeError;
use gneiss::lexer::LexError;
use gneiss::parser::ParseError;
use gneiss::{Inferencer, Lexer, Parser};

/// Assert that actual output matches snapshot file, or create/update if UPDATE_SNAPSHOTS=1
fn assert_snapshot(actual: &str, snapshot_name: &str) {
    let snapshot_path = format!("tests/snapshots/{}.expected", snapshot_name);

    if std::env::var("UPDATE_SNAPSHOTS").is_ok() {
        fs::write(&snapshot_path, actual).expect("Failed to write snapshot");
        println!("Updated snapshot: {}", snapshot_path);
        return;
    }

    let expected = fs::read_to_string(&snapshot_path).unwrap_or_else(|_| {
        panic!(
            "Snapshot not found: {}\n\
             Run with UPDATE_SNAPSHOTS=1 to create it.\n\
             Actual output:\n{}",
            snapshot_path, actual
        )
    });

    assert_eq!(
        actual, expected,
        "\nSnapshot mismatch for {}\n\
         Run with UPDATE_SNAPSHOTS=1 to update.\n\
         Actual:\n{}\n\
         Expected:\n{}",
        snapshot_name, actual, expected
    );
}

// ============================================================================
// Type Error Snapshots
// ============================================================================

#[test]
fn snapshot_type_error_mismatch() {
    let source = "let x = 1 + \"hello\"";
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse_program().unwrap();

    let mut inferencer = Inferencer::new();
    let err = inferencer.infer_program(&program).unwrap_err();

    let source_map = SourceMap::new(source);
    let colors = Colors::new(false); // No colors for deterministic output

    let formatted = format_type_error_for_test(&err, &source_map, Some("test.gn"), &colors);
    assert_snapshot(&formatted, "type_error_mismatch");
}

#[test]
fn snapshot_type_error_unbound_variable() {
    let source = "let x = unknownVar";
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse_program().unwrap();

    let mut inferencer = Inferencer::new();
    let err = inferencer.infer_program(&program).unwrap_err();

    let source_map = SourceMap::new(source);
    let colors = Colors::new(false);

    let formatted = format_type_error_for_test(&err, &source_map, Some("test.gn"), &colors);
    assert_snapshot(&formatted, "type_error_unbound");
}

#[test]
fn snapshot_type_error_with_suggestion() {
    // First define 'print' in a program, then try to use 'prnt'
    let source = "let prnt = print";  // This will use the built-in print
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let _ = parser.parse_program();

    // Now try using an undefined variable similar to a known one
    let source2 = "let result = prnt 42";
    let tokens2 = Lexer::new(source2).tokenize().unwrap();
    let program2 = Parser::new(tokens2).parse_program().unwrap();

    let mut inferencer = Inferencer::new();
    let err = inferencer.infer_program(&program2).unwrap_err();

    let source_map = SourceMap::new(source2);
    let colors = Colors::new(false);

    let formatted = format_type_error_for_test(&err, &source_map, Some("test.gn"), &colors);
    assert_snapshot(&formatted, "type_error_suggestion");
}

// ============================================================================
// Parse Error Snapshots
// ============================================================================

#[test]
fn snapshot_parse_error_unexpected_token() {
    let source = "let x = let";
    let tokens = Lexer::new(source).tokenize().unwrap();
    let err = Parser::new(tokens).parse_program().unwrap_err();

    let source_map = SourceMap::new(source);
    let colors = Colors::new(false);

    let formatted = format_parse_error_for_test(&err, &source_map, Some("test.gn"), &colors);
    assert_snapshot(&formatted, "parse_error_unexpected");
}

#[test]
fn snapshot_parse_error_unexpected_eof() {
    let source = "let x =";
    let tokens = Lexer::new(source).tokenize().unwrap();
    let err = Parser::new(tokens).parse_program().unwrap_err();

    let source_map = SourceMap::new(source);
    let colors = Colors::new(false);

    let formatted = format_parse_error_for_test(&err, &source_map, Some("test.gn"), &colors);
    assert_snapshot(&formatted, "parse_error_eof");
}

// ============================================================================
// Lexer Error Snapshots
// ============================================================================

#[test]
fn snapshot_lex_error_unexpected_char() {
    let source = "let x = @invalid";
    let err = Lexer::new(source).tokenize().unwrap_err();

    let source_map = SourceMap::new(source);
    let colors = Colors::new(false);

    let formatted = format_lex_error_for_test(&err, &source_map, Some("test.gn"), &colors);
    assert_snapshot(&formatted, "lex_error_unexpected");
}

#[test]
fn snapshot_lex_error_unterminated_string() {
    let source = "let x = \"hello";
    let err = Lexer::new(source).tokenize().unwrap_err();

    let source_map = SourceMap::new(source);
    let colors = Colors::new(false);

    let formatted = format_lex_error_for_test(&err, &source_map, Some("test.gn"), &colors);
    assert_snapshot(&formatted, "lex_error_unterminated_string");
}

// ============================================================================
// Helper Functions (duplicated from main.rs to avoid binary dependency)
// ============================================================================

fn format_type_error_for_test(
    err: &TypeError,
    source_map: &SourceMap,
    filename: Option<&str>,
    colors: &Colors,
) -> String {
    let mut out = String::new();

    let (header, msg, span, suggestions) = match err {
        TypeError::UnboundVariable { name, span, suggestions } => {
            ("NAME ERROR", format!("I cannot find a variable named `{}`.", name), Some(span), suggestions.clone())
        }
        TypeError::TypeMismatch { expected, found, span } => {
            let msg = format!(
                "I found a type mismatch.\n\n  I expected:  {}\n  But found:   {}",
                expected, found
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

    if let Some(span) = span {
        let pos = source_map.position(span.start);
        let file = filename.unwrap_or("<input>");
        out.push_str(&format!("{}:{}\n\n", file, pos));
    }

    out.push_str(&msg);

    if !suggestions.is_empty() {
        out.push_str(&format_suggestions(&suggestions, colors));
    }

    out.push('\n');

    if let Some(span) = span {
        out.push('\n');
        out.push_str(&format_snippet(source_map, span, colors));
    }

    out.push('\n');
    out
}

fn format_parse_error_for_test(
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

    if let Some(span) = span {
        let pos = source_map.position(span.start);
        let file = filename.unwrap_or("<input>");
        out.push_str(&format!("{}:{}\n\n", file, pos));
    }

    out.push_str(&msg);
    out.push('\n');

    if let Some(span) = span {
        out.push('\n');
        out.push_str(&format_snippet(source_map, span, colors));
    }

    out.push('\n');
    out
}

fn format_lex_error_for_test(
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
        LexError::UnterminatedChar(span) => {
            ("I found a character literal that was never closed.".to_string(), span)
        }
        LexError::InvalidEscape(c, span) => {
            (format!("I found an invalid escape sequence: \\{}", c), span)
        }
        LexError::InvalidNumber(s, span) => {
            (format!("I could not parse this as a number: {}", s), span)
        }
    };

    let pos = source_map.position(span.start);
    let file = filename.unwrap_or("<input>");
    out.push_str(&format!("{}:{}\n\n", file, pos));

    out.push_str(&msg);
    out.push('\n');

    out.push('\n');
    out.push_str(&format_snippet(source_map, span, colors));

    out.push('\n');
    out
}
