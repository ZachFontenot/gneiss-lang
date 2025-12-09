//! Handwritten lexer for Gneiss

use crate::ast::Span;
use std::iter::Peekable;
use std::str::Chars;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
    True,
    False,

    // Identifiers
    Ident(String),      // lowercase start
    UpperIdent(String), // uppercase start (constructors, traits)

    // Keywords
    Let,
    In,
    Fun,
    Match,
    With,
    If,
    Then,
    Else,
    Type,
    Spawn,
    Select,
    End,

    // Delimiters
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]
    LBrace,   // {
    RBrace,   // }
    Comma,    // ,
    Semicolon,// ;
    Colon,    // :

    // Operators
    Arrow,    // ->
    FatArrow, // =>
    LArrow,   // <-
    Pipe,     // |
    Eq,       // =
    EqEq,     // ==
    Neq,      // !=
    Lt,       // <
    Gt,       // >
    Lte,      // <=
    Gte,      // >=
    Plus,     // +
    Minus,    // -
    Star,     // *
    Slash,    // /
    Percent,  // %
    AndAnd,   // &&
    OrOr,     // ||
    Not,      // not (keyword)
    Cons,     // ::
    Concat,   // ++
    PipeOp,   // |>
    PipeBack, // <|
    Compose,  // >>
    ComposeBack, // <<
    Underscore,  // _
    Dot,      // .

    // Special
    Eof,
}

#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

#[derive(Error, Debug)]
pub enum LexError {
    #[error("unexpected character: {0}")]
    UnexpectedChar(char),
    #[error("unterminated string")]
    UnterminatedString,
    #[error("unterminated char literal")]
    UnterminatedChar,
    #[error("invalid escape sequence: \\{0}")]
    InvalidEscape(char),
    #[error("invalid number: {0}")]
    InvalidNumber(String),
}

pub struct Lexer<'a> {
    input: &'a str, // That's fine
    chars: Peekable<Chars<'a>>,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().peekable(),
            pos: 0,
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<SpannedToken>, LexError> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.token == Token::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.advance() {
            if c == '\n' {
                break;
            }
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), LexError> {
        let mut depth = 1;
        while depth > 0 {
            match self.advance() {
                Some('{') if self.peek() == Some('-') => {
                    self.advance();
                    depth += 1;
                }
                Some('-') if self.peek() == Some('}') => {
                    self.advance();
                    depth -= 1;
                }
                Some(_) => {}
                None => break, // Unterminated, but we'll just end
            }
        }
        Ok(())
    }

    fn next_token(&mut self) -> Result<SpannedToken, LexError> {
        loop {
            self.skip_whitespace();

            let start = self.pos;

            // Check for comments
            if self.peek() == Some('-') {
                let pos = self.pos;
                self.advance();
                if self.peek() == Some('-') {
                    self.advance();
                    self.skip_line_comment();
                    continue;
                } else {
                    // It was just a minus
                    self.pos = pos + 1; // Already advanced once
                    // Check for arrow
                    if self.peek() == Some('>') {
                        self.advance();
                        return Ok(SpannedToken {
                            token: Token::Arrow,
                            span: Span::new(start, self.pos),
                        });
                    }
                    return Ok(SpannedToken {
                        token: Token::Minus,
                        span: Span::new(start, self.pos),
                    });
                }
            }

            if self.peek() == Some('{') {
                let pos = self.pos;
                self.advance();
                if self.peek() == Some('-') {
                    self.advance();
                    self.skip_block_comment()?;
                    continue;
                } else {
                    return Ok(SpannedToken {
                        token: Token::LBrace,
                        span: Span::new(pos, self.pos),
                    });
                }
            }

            break;
        }

        let start = self.pos;

        let Some(c) = self.advance() else {
            return Ok(SpannedToken {
                token: Token::Eof,
                span: Span::new(start, start),
            });
        };

        let token = match c {
            // Single-char delimiters
            '(' => Token::LParen,
            ')' => Token::RParen,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            '}' => Token::RBrace,
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            '_' if !self.peek().map_or(false, is_ident_continue) => Token::Underscore,
            '.' => Token::Dot,

            // Operators that might be multi-char
            ':' => {
                if self.peek() == Some(':') {
                    self.advance();
                    Token::Cons
                } else {
                    Token::Colon
                }
            }
            '=' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Token::EqEq
                } else if self.peek() == Some('>') {
                    self.advance();
                    Token::FatArrow
                } else {
                    Token::Eq
                }
            }
            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Token::Neq
                } else {
                    return Err(LexError::UnexpectedChar('!'));
                }
            }
            '<' => {
                if self.peek() == Some('-') {
                    self.advance();
                    Token::LArrow
                } else if self.peek() == Some('=') {
                    self.advance();
                    Token::Lte
                } else if self.peek() == Some('|') {
                    self.advance();
                    Token::PipeBack
                } else if self.peek() == Some('<') {
                    self.advance();
                    Token::ComposeBack
                } else {
                    Token::Lt
                }
            }
            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Token::Gte
                } else if self.peek() == Some('>') {
                    self.advance();
                    Token::Compose
                } else {
                    Token::Gt
                }
            }
            '+' => {
                if self.peek() == Some('+') {
                    self.advance();
                    Token::Concat
                } else {
                    Token::Plus
                }
            }
            '*' => Token::Star,
            '/' => Token::Slash,
            '%' => Token::Percent,
            '&' => {
                if self.peek() == Some('&') {
                    self.advance();
                    Token::AndAnd
                } else {
                    return Err(LexError::UnexpectedChar('&'));
                }
            }
            '|' => {
                if self.peek() == Some('|') {
                    self.advance();
                    Token::OrOr
                } else if self.peek() == Some('>') {
                    self.advance();
                    Token::PipeOp
                } else {
                    Token::Pipe
                }
            }

            // String literal
            '"' => self.lex_string()?,

            // Char literal
            '\'' => self.lex_char()?,

            // Number
            c if c.is_ascii_digit() => self.lex_number(c)?,

            // Identifier or keyword
            c if c.is_alphabetic() || c == '_' => self.lex_ident(c),

            _ => return Err(LexError::UnexpectedChar(c)),
        };

        Ok(SpannedToken {
            token,
            span: Span::new(start, self.pos),
        })
    }

    fn lex_string(&mut self) -> Result<Token, LexError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('"') => break,
                Some('\\') => {
                    let escaped = match self.advance() {
                        Some('n') => '\n',
                        Some('t') => '\t',
                        Some('r') => '\r',
                        Some('\\') => '\\',
                        Some('"') => '"',
                        Some(c) => return Err(LexError::InvalidEscape(c)),
                        None => return Err(LexError::UnterminatedString),
                    };
                    s.push(escaped);
                }
                Some(c) => s.push(c),
                None => return Err(LexError::UnterminatedString),
            }
        }
        Ok(Token::String(s))
    }

    fn lex_char(&mut self) -> Result<Token, LexError> {
        let c = match self.advance() {
            Some('\\') => match self.advance() {
                Some('n') => '\n',
                Some('t') => '\t',
                Some('r') => '\r',
                Some('\\') => '\\',
                Some('\'') => '\'',
                Some(c) => return Err(LexError::InvalidEscape(c)),
                None => return Err(LexError::UnterminatedChar),
            },
            Some(c) => c,
            None => return Err(LexError::UnterminatedChar),
        };
        if self.advance() != Some('\'') {
            return Err(LexError::UnterminatedChar);
        }
        Ok(Token::Char(c))
    }

    fn lex_number(&mut self, first: char) -> Result<Token, LexError> {
        let mut s = String::new();
        s.push(first);

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Check for float
        if self.peek() == Some('.') {
            // Look ahead to make sure it's not something like `1.method`
            let mut chars = self.chars.clone();
            chars.next(); // skip the dot
            if chars.peek().map_or(false, |c| c.is_ascii_digit()) {
                s.push('.');
                self.advance(); // consume the dot
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        s.push(c);
                        self.advance();
                    } else {
                        break;
                    }
                }
                let f: f64 = s
                    .parse()
                    .map_err(|_| LexError::InvalidNumber(s.clone()))?;
                return Ok(Token::Float(f));
            }
        }

        let n: i64 = s
            .parse()
            .map_err(|_| LexError::InvalidNumber(s.clone()))?;
        Ok(Token::Int(n))
    }

    fn lex_ident(&mut self, first: char) -> Token {
        let mut s = String::new();
        s.push(first);

        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Check for keywords
        match s.as_str() {
            "let" => Token::Let,
            "in" => Token::In,
            "fun" => Token::Fun,
            "match" => Token::Match,
            "with" => Token::With,
            "if" => Token::If,
            "then" => Token::Then,
            "else" => Token::Else,
            "type" => Token::Type,
            "true" => Token::True,
            "false" => Token::False,
            "not" => Token::Not,
            "spawn" => Token::Spawn,
            "select" => Token::Select,
            "end" => Token::End,
            _ => {
                if s.chars().next().unwrap().is_uppercase() {
                    Token::UpperIdent(s)
                } else {
                    Token::Ident(s)
                }
            }
        }
    }
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(input: &str) -> Vec<Token> {
        Lexer::new(input)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.token)
            .collect()
    }

    #[test]
    fn test_basic() {
        assert_eq!(
            tokens("let x = 42"),
            vec![
                Token::Let,
                Token::Ident("x".into()),
                Token::Eq,
                Token::Int(42),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(
            tokens("x |> f >> g"),
            vec![
                Token::Ident("x".into()),
                Token::PipeOp,
                Token::Ident("f".into()),
                Token::Compose,
                Token::Ident("g".into()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_comments() {
        assert_eq!(
            tokens("x -- comment\ny"),
            vec![
                Token::Ident("x".into()),
                Token::Ident("y".into()),
                Token::Eof
            ]
        );
    }
}
