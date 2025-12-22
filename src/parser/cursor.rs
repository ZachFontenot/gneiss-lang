//! Token stream cursor with lookahead and span tracking

use crate::ast::Span;
use crate::lexer::{SpannedToken, Token};

use super::error::{ParseError, ParseResult};

/// Token stream cursor providing lookahead and span tracking
pub struct TokenCursor {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl TokenCursor {
    /// Create a new cursor over a token stream
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Self { tokens, pos: 0 }
    }

    // ========================================================================
    // Position and lookahead
    // ========================================================================

    /// Get the current token without consuming it
    pub fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .map(|t| &t.token)
            .unwrap_or(&Token::Eof)
    }

    /// Peek at a token n positions ahead (0 = current)
    pub fn peek_nth(&self, n: usize) -> &Token {
        self.tokens
            .get(self.pos + n)
            .map(|t| &t.token)
            .unwrap_or(&Token::Eof)
    }

    /// Get the span of the current token
    pub fn current_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|t| t.span.clone())
            .unwrap_or_default()
    }

    /// Check if we've reached the end of the token stream
    pub fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    /// Get the current position (for backtracking if needed)
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Restore to a previous position (for backtracking)
    pub fn restore(&mut self, pos: usize) {
        self.pos = pos;
    }

    // ========================================================================
    // Token consumption
    // ========================================================================

    /// Advance to the next token and return the previous one
    pub fn advance(&mut self) -> &SpannedToken {
        if !self.is_at_end() {
            self.pos += 1;
        }
        &self.tokens[self.pos - 1]
    }

    /// Check if the current token matches the expected token
    pub fn check(&self, token: &Token) -> bool {
        self.peek() == token
    }

    /// If the current token matches, consume it and return true
    pub fn match_token(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume the expected token or return an error
    pub fn consume(&mut self, expected: Token) -> ParseResult<&SpannedToken> {
        if self.check(&expected) {
            Ok(self.advance())
        } else {
            Err(ParseError::unexpected(
                format!("{:?}", expected),
                self.peek().clone(),
                self.current_span(),
            ))
        }
    }

    /// Create an error for unexpected token at current position
    pub fn unexpected(&self, expected: &str) -> ParseError {
        ParseError::unexpected(expected, self.peek().clone(), self.current_span())
    }

    // ========================================================================
    // Operator helpers
    // ========================================================================

    /// Check if the current token is an operator and return its string representation
    pub fn peek_operator_symbol(&self) -> Option<(String, Span)> {
        let span = self.current_span();
        let op_str = self.peek().operator_symbol()?;
        Some((op_str, span))
    }

    /// Try to peek for an operator token
    pub fn try_peek_operator(&self) -> Option<String> {
        self.peek().operator_symbol()
    }

    /// Check if the next tokens are `( <op> )` and return the operator symbol
    pub fn peek_operator_in_parens(&self) -> Option<String> {
        if !matches!(self.peek(), Token::LParen) {
            return None;
        }
        if self.pos + 2 >= self.tokens.len() {
            return None;
        }
        let op_str = self.tokens[self.pos + 1].token.operator_symbol()?;
        if !matches!(self.tokens[self.pos + 2].token, Token::RParen) {
            return None;
        }
        Some(op_str)
    }

    // ========================================================================
    // Token predicates
    // ========================================================================

    /// Check if the current token could start an expression atom
    pub fn is_atom_start(&self) -> bool {
        matches!(
            self.peek(),
            Token::Int(_)
                | Token::Float(_)
                | Token::String(_)
                | Token::Char(_)
                | Token::True
                | Token::False
                | Token::Ident(_)
                | Token::UpperIdent(_)
                | Token::LParen
                | Token::LBracket
                | Token::Reset
                | Token::Shift
        )
    }

    /// Check if the current token could start a pattern
    pub fn is_pattern_start(&self) -> bool {
        matches!(
            self.peek(),
            Token::Ident(_)
                | Token::UpperIdent(_)
                | Token::Int(_)
                | Token::String(_)
                | Token::Char(_)
                | Token::True
                | Token::False
                | Token::LParen
                | Token::LBracket
                | Token::Underscore
        )
    }

    /// Check if the current token could start a type atom
    pub fn is_type_atom_start(&self) -> bool {
        matches!(
            self.peek(),
            Token::Ident(_) | Token::UpperIdent(_) | Token::LParen
        )
    }

    /// Check if the current token is an uppercase identifier
    pub fn is_upper_ident(&self) -> bool {
        matches!(self.peek(), Token::UpperIdent(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn cursor(input: &str) -> TokenCursor {
        let tokens = Lexer::new(input).tokenize().unwrap();
        TokenCursor::new(tokens)
    }

    #[test]
    fn test_basic_navigation() {
        let mut c = cursor("let x = 42");
        assert!(c.check(&Token::Let));
        c.advance();
        assert!(matches!(c.peek(), Token::Ident(s) if s == "x"));
        c.advance();
        assert!(c.check(&Token::Eq));
    }

    #[test]
    fn test_lookahead() {
        let c = cursor("a b c");
        assert!(matches!(c.peek(), Token::Ident(s) if s == "a"));
        assert!(matches!(c.peek_nth(1), Token::Ident(s) if s == "b"));
        assert!(matches!(c.peek_nth(2), Token::Ident(s) if s == "c"));
        assert!(matches!(c.peek_nth(3), Token::Eof));
    }

    #[test]
    fn test_match_and_consume() {
        let mut c = cursor("let x");
        assert!(c.match_token(&Token::Let));
        assert!(!c.match_token(&Token::Let));
        assert!(c.consume(Token::Ident("x".to_string())).is_ok());
    }

    #[test]
    fn test_backtracking() {
        let mut c = cursor("a b c");
        let pos = c.position();
        c.advance();
        c.advance();
        assert!(matches!(c.peek(), Token::Ident(s) if s == "c"));
        c.restore(pos);
        assert!(matches!(c.peek(), Token::Ident(s) if s == "a"));
    }
}
