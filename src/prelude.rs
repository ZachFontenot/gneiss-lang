//! Gneiss Prelude - Auto-imported core types and functions
//!
//! The prelude provides:
//! - `Option a` = `Some a | None`
//! - `Result e a` = `Ok a | Err e`
//! - `id`, `const`, `flip` combinators

use crate::ast::Program;
use crate::lexer::Lexer;
use crate::parser::Parser;

/// The prelude source code, embedded in the binary
pub const PRELUDE_SOURCE: &str = include_str!("../stdlib/prelude.gn");

/// Parse the prelude source code
///
/// Called fresh each time since Program contains Rc (not thread-safe for static caching).
/// The prelude is small, so this is fast.
pub fn parse_prelude() -> Result<Program, String> {
    let lexer = Lexer::new(PRELUDE_SOURCE);
    let tokens = lexer.tokenize().map_err(|e| format!("Prelude lex error: {:?}", e))?;
    let mut parser = Parser::new(tokens);
    parser.parse_program().map_err(|e| format!("Prelude parse error: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_parses() {
        let program = parse_prelude().expect("Prelude should parse");
        // Should have Option, Result types + id, const, flip functions
        assert!(!program.items.is_empty(), "Prelude should have items");
    }

    #[test]
    fn test_prelude_has_option_type() {
        let program = parse_prelude().expect("Prelude should parse");
        let has_option = program.items.iter().any(|item| {
            if let crate::ast::Item::Decl(decl) = item {
                if let crate::ast::Decl::Type { name, .. } = decl {
                    return name.as_str() == "Option";
                }
            }
            false
        });
        assert!(has_option, "Prelude should define Option type");
    }

    #[test]
    fn test_prelude_has_result_type() {
        let program = parse_prelude().expect("Prelude should parse");
        let has_result = program.items.iter().any(|item| {
            if let crate::ast::Item::Decl(decl) = item {
                if let crate::ast::Decl::Type { name, .. } = decl {
                    return name.as_str() == "Result";
                }
            }
            false
        });
        assert!(has_result, "Prelude should define Result type");
    }
}
