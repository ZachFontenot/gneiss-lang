//! Pattern parsing with unified hierarchy

use std::rc::Rc;

use crate::ast::{Ident, Literal, Pattern, PatternKind, Spanned};
use crate::lexer::Token;

use super::cursor::TokenCursor;
use super::error::ParseResult;

/// Mode for pattern parsing - controls whether constructors can take arguments
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PatternMode {
    /// Full patterns: constructors can take arguments (e.g., "Some x")
    Full,
    /// Atom patterns: no constructor arguments (e.g., just "Some")
    Atom,
}

/// Extension trait for pattern parsing
pub trait PatternParser {
    /// Parse a full pattern (cons patterns allowed)
    fn parse_pattern(&mut self) -> ParseResult<Pattern>;

    /// Parse a pattern with cons operator (right-associative)
    fn parse_pattern_cons(&mut self) -> ParseResult<Pattern>;

    /// Parse a simple pattern (constructors with args allowed)
    fn parse_simple_pattern(&mut self) -> ParseResult<Pattern>;

    /// Parse a pattern atom (no constructor args)
    fn parse_pattern_atom(&mut self) -> ParseResult<Pattern>;

    /// Parse a pattern in the given mode
    fn parse_pattern_in(&mut self, mode: PatternMode) -> ParseResult<Pattern>;

    /// Parse record pattern fields: { field1, field2 = pat, ... }
    fn parse_record_pattern_fields(&mut self) -> ParseResult<Vec<(Ident, Option<Pattern>)>>;
}

impl PatternParser for TokenCursor {
    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        self.parse_pattern_cons()
    }

    fn parse_pattern_cons(&mut self) -> ParseResult<Pattern> {
        let start = self.current_span();
        let left = self.parse_simple_pattern()?;

        if self.match_token(&Token::Cons) {
            let right = self.parse_pattern_cons()?; // Right associative
            let span = start.merge(&right.span);
            Ok(Spanned::new(
                PatternKind::Cons {
                    head: Rc::new(left),
                    tail: Rc::new(right),
                },
                span,
            ))
        } else {
            Ok(left)
        }
    }

    fn parse_simple_pattern(&mut self) -> ParseResult<Pattern> {
        self.parse_pattern_in(PatternMode::Full)
    }

    fn parse_pattern_atom(&mut self) -> ParseResult<Pattern> {
        self.parse_pattern_in(PatternMode::Atom)
    }

    fn parse_pattern_in(&mut self, mode: PatternMode) -> ParseResult<Pattern> {
        let start = self.current_span();

        match self.peek().clone() {
            Token::Underscore => {
                self.advance();
                Ok(Spanned::new(PatternKind::Wildcard, start))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Var(name), start))
            }
            Token::UpperIdent(name) => {
                self.advance();
                // Check for record pattern: TypeName { field1, field2 = pat }
                if self.check(&Token::LBrace) {
                    let fields = self.parse_record_pattern_fields()?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(PatternKind::Record { name, fields }, span))
                } else if mode == PatternMode::Full {
                    // Constructor with args: "Some x" or "Pair a b"
                    let mut args = Vec::new();
                    while self.is_pattern_start() {
                        args.push(self.parse_pattern_atom()?);
                    }
                    let span = if args.is_empty() {
                        start
                    } else {
                        start.merge(&args.last().unwrap().span)
                    };
                    Ok(Spanned::new(PatternKind::Constructor { name, args }, span))
                } else {
                    // Nullary constructor as atom: "Get" in "StateOp Get k"
                    Ok(Spanned::new(
                        PatternKind::Constructor { name, args: vec![] },
                        start,
                    ))
                }
            }
            Token::Int(n) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Int(n)), start))
            }
            Token::String(s) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::String(s)), start))
            }
            Token::Char(c) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Char(c)), start))
            }
            Token::True => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Bool(true)), start))
            }
            Token::False => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Bool(false)), start))
            }
            Token::LParen => {
                self.advance();
                if self.match_token(&Token::RParen) {
                    let span = start.merge(&self.current_span());
                    return Ok(Spanned::new(PatternKind::Lit(Literal::Unit), span));
                }

                let first = self.parse_pattern()?;

                if self.match_token(&Token::Comma) {
                    // Tuple pattern
                    let mut pats = vec![first];
                    loop {
                        if self.check(&Token::RParen) {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                    self.consume(Token::RParen)?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(PatternKind::Tuple(pats), span))
                } else {
                    // Parenthesized pattern
                    self.consume(Token::RParen)?;
                    Ok(first)
                }
            }
            Token::LBracket => {
                // List pattern
                self.advance();
                let mut pats = Vec::new();

                if !self.check(&Token::RBracket) {
                    loop {
                        pats.push(self.parse_pattern()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                        if self.check(&Token::RBracket) {
                            break;
                        }
                    }
                }

                self.consume(Token::RBracket)?;
                let span = start.merge(&self.current_span());
                Ok(Spanned::new(PatternKind::List(pats), span))
            }
            _ => Err(self.unexpected("pattern")),
        }
    }

    fn parse_record_pattern_fields(&mut self) -> ParseResult<Vec<(Ident, Option<Pattern>)>> {
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

    fn parse_pat(input: &str) -> Pattern {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let mut cursor = TokenCursor::new(tokens);
        cursor.parse_pattern().unwrap()
    }

    #[test]
    fn test_wildcard() {
        let pat = parse_pat("_");
        assert!(matches!(pat.node, PatternKind::Wildcard));
    }

    #[test]
    fn test_var() {
        let pat = parse_pat("x");
        assert!(matches!(pat.node, PatternKind::Var(ref s) if s == "x"));
    }

    #[test]
    fn test_int_literal() {
        let pat = parse_pat("42");
        assert!(matches!(pat.node, PatternKind::Lit(Literal::Int(42))));
    }

    #[test]
    fn test_constructor_nullary() {
        let pat = parse_pat("None");
        if let PatternKind::Constructor { name, args } = &pat.node {
            assert_eq!(name, "None");
            assert!(args.is_empty());
        } else {
            panic!("expected constructor pattern");
        }
    }

    #[test]
    fn test_constructor_with_arg() {
        let pat = parse_pat("Some x");
        if let PatternKind::Constructor { name, args } = &pat.node {
            assert_eq!(name, "Some");
            assert_eq!(args.len(), 1);
        } else {
            panic!("expected constructor pattern");
        }
    }

    #[test]
    fn test_cons_pattern() {
        let pat = parse_pat("x :: xs");
        assert!(matches!(pat.node, PatternKind::Cons { .. }));
    }

    #[test]
    fn test_tuple_pattern() {
        let pat = parse_pat("(x, y, z)");
        if let PatternKind::Tuple(pats) = &pat.node {
            assert_eq!(pats.len(), 3);
        } else {
            panic!("expected tuple pattern");
        }
    }

    #[test]
    fn test_list_pattern() {
        let pat = parse_pat("[x, y]");
        if let PatternKind::List(pats) = &pat.node {
            assert_eq!(pats.len(), 2);
        } else {
            panic!("expected list pattern");
        }
    }

    #[test]
    fn test_empty_list_pattern() {
        let pat = parse_pat("[]");
        if let PatternKind::List(pats) = &pat.node {
            assert!(pats.is_empty());
        } else {
            panic!("expected empty list pattern");
        }
    }

    #[test]
    fn test_unit_pattern() {
        let pat = parse_pat("()");
        assert!(matches!(pat.node, PatternKind::Lit(Literal::Unit)));
    }
}
