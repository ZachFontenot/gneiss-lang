//! Unified record field parsing
//!
//! This module provides a generic abstraction for parsing record fields,
//! which eliminates the 4x duplication in record type fields, record literal fields,
//! record update fields, and record pattern fields.

use crate::ast::{Expr, Ident, Pattern, RecordField, TypeExpr};
use crate::lexer::Token;

use super::cursor::TokenCursor;
use super::error::ParseResult;
use super::pattern::PatternParser;
use super::types::TypeParser;

/// The separator used between field name and field value
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldSeparator {
    /// Colon separator: field : Type (for type definitions)
    Colon,
    /// Equals separator: field = value (for literals and patterns)
    Equals,
}

/// Extension trait for record field parsing
pub trait RecordParser {
    /// Parse record type fields: { field1 : Type1, field2 : Type2, ... }
    fn parse_record_type_fields(&mut self) -> ParseResult<Vec<RecordField>>;

    /// Parse record literal fields: { field1 = expr1, field2 = expr2, ... }
    fn parse_record_literal_fields(&mut self, parse_expr: impl FnMut(&mut Self) -> ParseResult<Expr>) -> ParseResult<Vec<(Ident, Expr)>>;

    /// Parse record update fields (after 'with'): field1 = expr1, field2 = expr2, ...
    fn parse_record_update_fields(&mut self, parse_expr: impl FnMut(&mut Self) -> ParseResult<Expr>) -> ParseResult<Vec<(Ident, Expr)>>;

    /// Parse record pattern fields: { field1, field2 = pat, ... }
    fn parse_record_pattern_fields_unified(&mut self) -> ParseResult<Vec<(Ident, Option<Pattern>)>>;
}

impl RecordParser for TokenCursor {
    fn parse_record_type_fields(&mut self) -> ParseResult<Vec<RecordField>> {
        self.consume(Token::LBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = parse_ident(self)?;
                self.consume(Token::Colon)?;
                let field_ty = self.parse_type_expr()?;
                fields.push(RecordField {
                    name: field_name,
                    ty: field_ty,
                });

                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
    }

    fn parse_record_literal_fields(&mut self, mut parse_expr: impl FnMut(&mut Self) -> ParseResult<Expr>) -> ParseResult<Vec<(Ident, Expr)>> {
        self.consume(Token::LBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = parse_ident(self)?;
                self.consume(Token::Eq)?;
                let field_value = parse_expr(self)?;
                fields.push((field_name, field_value));

                if !self.match_token(&Token::Comma) {
                    break;
                }
                // Allow trailing comma
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
    }

    fn parse_record_update_fields(&mut self, mut parse_expr: impl FnMut(&mut Self) -> ParseResult<Expr>) -> ParseResult<Vec<(Ident, Expr)>> {
        let mut updates = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = parse_ident(self)?;
                self.consume(Token::Eq)?;
                let field_value = parse_expr(self)?;
                updates.push((field_name, field_value));

                if !self.match_token(&Token::Comma) {
                    break;
                }
                // Allow trailing comma
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        Ok(updates)
    }

    fn parse_record_pattern_fields_unified(&mut self) -> ParseResult<Vec<(Ident, Option<Pattern>)>> {
        self.consume(Token::LBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = parse_ident(self)?;
                let pattern = if self.match_token(&Token::Eq) {
                    // Explicit pattern binding: field = pattern
                    Some(self.parse_pattern()?)
                } else {
                    // Shorthand: just field name, binds to same name
                    None
                };
                fields.push((field_name, pattern));

                if !self.match_token(&Token::Comma) {
                    break;
                }
                // Allow trailing comma
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
    }
}

/// Helper to parse an identifier
fn parse_ident(cursor: &mut TokenCursor) -> ParseResult<Ident> {
    match cursor.peek().clone() {
        Token::Ident(name) => {
            cursor.advance();
            Ok(name)
        }
        Token::Underscore => {
            cursor.advance();
            Ok("_".to_string())
        }
        _ => Err(cursor.unexpected("identifier")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::ast::{ExprKind, Spanned, Literal, Span};

    fn cursor(input: &str) -> TokenCursor {
        let tokens = Lexer::new(input).tokenize().unwrap();
        TokenCursor::new(tokens)
    }

    fn dummy_parse_expr(cursor: &mut TokenCursor) -> ParseResult<Expr> {
        // Simple expression parser for testing - just parse integers
        if let Token::Int(n) = cursor.peek().clone() {
            cursor.advance();
            Ok(Spanned::new(ExprKind::Lit(Literal::Int(n)), Span::default()))
        } else {
            Err(cursor.unexpected("expression"))
        }
    }

    #[test]
    fn test_record_type_fields() {
        let mut c = cursor("{ name : String, age : Int }");
        let fields = c.parse_record_type_fields().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "name");
        assert_eq!(fields[1].name, "age");
    }

    #[test]
    fn test_record_literal_fields() {
        let mut c = cursor("{ x = 1, y = 2 }");
        let fields = c.parse_record_literal_fields(dummy_parse_expr).unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].0, "x");
        assert_eq!(fields[1].0, "y");
    }

    #[test]
    fn test_record_pattern_fields() {
        let mut c = cursor("{ name, age = a }");
        let fields = c.parse_record_pattern_fields_unified().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].0, "name");
        assert!(fields[0].1.is_none()); // shorthand
        assert_eq!(fields[1].0, "age");
        assert!(fields[1].1.is_some()); // explicit pattern
    }

    #[test]
    fn test_empty_record() {
        let mut c = cursor("{ }");
        let fields = c.parse_record_type_fields().unwrap();
        assert!(fields.is_empty());
    }

    #[test]
    fn test_trailing_comma() {
        let mut c = cursor("{ x = 1, y = 2, }");
        let fields = c.parse_record_literal_fields(dummy_parse_expr).unwrap();
        assert_eq!(fields.len(), 2);
    }
}
