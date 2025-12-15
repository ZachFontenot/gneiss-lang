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
    Rec, // for 'let rec'
    And, // for mutual recursion: 'let rec f = ... and g = ...'
    In,
    Fun,
    Match,
    With,
    If,
    Then,
    Else,
    Type,
    Select,
    End,
    Reset,
    Shift,

    // Typeclass keywords
    Trait,
    Impl,
    For,
    Where,
    Val,

    // Delimiters
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    LBrace,     // {
    RBrace,     // }
    Comma,      // ,
    Semicolon,  // ;
    DoubleSemi, // ;;
    Colon,      // :

    // Operators
    Arrow,       // ->
    FatArrow,    // =>
    LArrow,      // <-
    Pipe,        // |
    Eq,          // =
    EqEq,        // ==
    Neq,         // !=
    Lt,          // <
    Gt,          // >
    Lte,         // <=
    Gte,         // >=
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    AndAnd,      // &&
    OrOr,        // ||
    Not,         // not (keyword)
    Cons,        // ::
    Concat,      // ++
    PipeOp,      // |>
    PipeBack,    // <|
    Compose,     // >>
    ComposeBack, // <<
    Underscore,  // _
    Dot,         // .

    // User-defined operator symbol (e.g., <|>, >>=, ***)
    OpSymbol(String),

    // Special
    Eof,
}

impl Token {
    /// If this token represents an operator, return its symbol string.
    /// This is the single source of truth for token-to-operator conversion.
    pub fn operator_symbol(&self) -> Option<String> {
        match self {
            Token::Plus => Some("+".to_string()),
            Token::Minus => Some("-".to_string()),
            Token::Star => Some("*".to_string()),
            Token::Slash => Some("/".to_string()),
            Token::Percent => Some("%".to_string()),
            Token::EqEq => Some("==".to_string()),
            Token::Neq => Some("!=".to_string()),
            Token::Lt => Some("<".to_string()),
            Token::Gt => Some(">".to_string()),
            Token::Lte => Some("<=".to_string()),
            Token::Gte => Some(">=".to_string()),
            Token::AndAnd => Some("&&".to_string()),
            Token::OrOr => Some("||".to_string()),
            Token::Cons => Some("::".to_string()),
            Token::Concat => Some("++".to_string()),
            Token::PipeOp => Some("|>".to_string()),
            Token::PipeBack => Some("<|".to_string()),
            Token::Compose => Some(">>".to_string()),
            Token::ComposeBack => Some("<<".to_string()),
            Token::OpSymbol(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Returns true if this token represents an operator
    pub fn is_operator(&self) -> bool {
        self.operator_symbol().is_some()
    }
}

#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

#[derive(Error, Debug)]
pub enum LexError {
    #[error("unexpected character: {0}")]
    UnexpectedChar(char, Span),
    #[error("unterminated string")]
    UnterminatedString(Span),
    #[error("unterminated char literal")]
    UnterminatedChar(Span),
    #[error("invalid escape sequence: \\{0}")]
    InvalidEscape(char, Span),
    #[error("invalid number: {0}")]
    InvalidNumber(String, Span),
}

impl LexError {
    /// Get the source span where this error occurred
    pub fn span(&self) -> &Span {
        match self {
            LexError::UnexpectedChar(_, span) => span,
            LexError::UnterminatedString(span) => span,
            LexError::UnterminatedChar(span) => span,
            LexError::InvalidEscape(_, span) => span,
            LexError::InvalidNumber(_, span) => span,
        }
    }
}

pub struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
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
            ';' => {
                if self.peek() == Some(';') {
                    self.advance();
                    Token::DoubleSemi
                } else {
                    Token::Semicolon
                }
            }
            '_' if !self.peek().map_or(false, is_ident_continue) => Token::Underscore,
            '.' => Token::Dot,

            // Colon: either `:` (type annotation) or `::` (cons)
            ':' => {
                if self.peek() == Some(':') {
                    self.advance();
                    Token::Cons
                } else {
                    Token::Colon
                }
            }
            // Left angle: special case for `<-` (left arrow), otherwise operator
            '<' => {
                if self.peek() == Some('-') {
                    self.advance();
                    Token::LArrow
                } else {
                    self.lex_operator(c)
                }
            }
            // All other operator characters: use generic operator lexing
            c if is_operator_char(c) => self.lex_operator(c),

            // String literal
            '"' => self.lex_string(start)?,

            // Char literal
            '\'' => self.lex_char(start)?,

            // Number
            c if c.is_ascii_digit() => self.lex_number(c, start)?,

            // Identifier or keyword
            c if c.is_alphabetic() || c == '_' => self.lex_ident(c),

            _ => return Err(LexError::UnexpectedChar(c, Span::new(start, self.pos))),
        };

        Ok(SpannedToken {
            token,
            span: Span::new(start, self.pos),
        })
    }

    fn lex_string(&mut self, start: usize) -> Result<Token, LexError> {
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
                        Some(c) => {
                            return Err(LexError::InvalidEscape(c, Span::new(start, self.pos)))
                        }
                        None => {
                            return Err(LexError::UnterminatedString(Span::new(start, self.pos)))
                        }
                    };
                    s.push(escaped);
                }
                Some(c) => s.push(c),
                None => return Err(LexError::UnterminatedString(Span::new(start, self.pos))),
            }
        }
        Ok(Token::String(s))
    }

    fn lex_char(&mut self, start: usize) -> Result<Token, LexError> {
        let c = match self.advance() {
            Some('\\') => match self.advance() {
                Some('n') => '\n',
                Some('t') => '\t',
                Some('r') => '\r',
                Some('\\') => '\\',
                Some('\'') => '\'',
                Some(c) => return Err(LexError::InvalidEscape(c, Span::new(start, self.pos))),
                None => return Err(LexError::UnterminatedChar(Span::new(start, self.pos))),
            },
            Some(c) => c,
            None => return Err(LexError::UnterminatedChar(Span::new(start, self.pos))),
        };
        if self.advance() != Some('\'') {
            return Err(LexError::UnterminatedChar(Span::new(start, self.pos)));
        }
        Ok(Token::Char(c))
    }

    fn lex_number(&mut self, first: char, start: usize) -> Result<Token, LexError> {
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
                    .map_err(|_| LexError::InvalidNumber(s.clone(), Span::new(start, self.pos)))?;
                return Ok(Token::Float(f));
            }
        }

        let n: i64 = s
            .parse()
            .map_err(|_| LexError::InvalidNumber(s.clone(), Span::new(start, self.pos)))?;
        Ok(Token::Int(n))
    }

    /// Lex an operator sequence starting with the given character.
    /// Collects all consecutive operator characters and classifies the result.
    fn lex_operator(&mut self, first: char) -> Token {
        let mut s = String::new();
        s.push(first);

        while let Some(c) = self.peek() {
            if is_operator_char(c) {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        classify_operator(&s)
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
            "rec" => Token::Rec,
            "and" => Token::And,
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
            "select" => Token::Select,
            "end" => Token::End,
            "reset" => Token::Reset,
            "shift" => Token::Shift,
            // Typeclass keywords
            "trait" => Token::Trait,
            "impl" => Token::Impl,
            "for" => Token::For,
            "where" => Token::Where,
            "val" => Token::Val,
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

/// Characters that can appear in operator symbols.
/// Note: `-` is handled specially due to `--` comments.
fn is_operator_char(c: char) -> bool {
    matches!(
        c,
        '!' | '$' | '%' | '&' | '*' | '+' | '/' | '<' | '=' | '>' | '?' | '@' | '^' | '|' | '~'
    )
}

/// Classify an operator string into a specific token or OpSymbol
fn classify_operator(s: &str) -> Token {
    match s {
        // Comparison
        "==" => Token::EqEq,
        "!=" => Token::Neq,
        "<" => Token::Lt,
        ">" => Token::Gt,
        "<=" => Token::Lte,
        ">=" => Token::Gte,
        // Arithmetic
        "+" => Token::Plus,
        "*" => Token::Star,
        "/" => Token::Slash,
        "%" => Token::Percent,
        // Boolean
        "&&" => Token::AndAnd,
        "||" => Token::OrOr,
        // List
        "++" => Token::Concat,
        // Pipe
        "|>" => Token::PipeOp,
        "<|" => Token::PipeBack,
        // Compose
        ">>" => Token::Compose,
        "<<" => Token::ComposeBack,
        // Fat arrow (used in match)
        "=>" => Token::FatArrow,
        // Single pipe is pattern separator, not an operator
        "|" => Token::Pipe,
        // Single equals is assignment
        "=" => Token::Eq,
        // Everything else is a user-defined operator
        _ => Token::OpSymbol(s.to_string()),
    }
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

    #[test]
    fn test_typeclass_tokens() {
        assert_eq!(
            tokens("trait impl for where val end"),
            vec![
                Token::Trait,
                Token::Impl,
                Token::For,
                Token::Where,
                Token::Val,
                Token::End,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_trait_not_ident() {
        // "trait" should be keyword, not identifier
        let toks = tokens("trait");
        assert!(!matches!(toks[0], Token::Ident(_)));
        assert!(matches!(toks[0], Token::Trait));
    }

    #[test]
    fn test_user_defined_operators() {
        // User-defined operators should be lexed as OpSymbol
        assert_eq!(
            tokens("<|>"),
            vec![Token::OpSymbol("<|>".into()), Token::Eof]
        );
        assert_eq!(
            tokens(">>="),
            vec![Token::OpSymbol(">>=".into()), Token::Eof]
        );
        assert_eq!(
            tokens("***"),
            vec![Token::OpSymbol("***".into()), Token::Eof]
        );
        assert_eq!(tokens("$"), vec![Token::OpSymbol("$".into()), Token::Eof]);
        // Built-ins should still work
        assert_eq!(
            tokens("+ - * /"),
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Eof
            ]
        );
        // Mixed: built-in followed by user operator
        assert_eq!(
            tokens("x <|> y"),
            vec![
                Token::Ident("x".into()),
                Token::OpSymbol("<|>".into()),
                Token::Ident("y".into()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_arrows_preserved() {
        // Arrows should not become operators
        assert_eq!(tokens("->"), vec![Token::Arrow, Token::Eof]);
        assert_eq!(tokens("<-"), vec![Token::LArrow, Token::Eof]);
        assert_eq!(tokens("=>"), vec![Token::FatArrow, Token::Eof]);
    }

    #[test]
    fn test_rec_and_keywords() {
        // 'rec' and 'and' should be keywords
        assert_eq!(
            tokens("let rec f = 1 and g = 2"),
            vec![
                Token::Let,
                Token::Rec,
                Token::Ident("f".into()),
                Token::Eq,
                Token::Int(1),
                Token::And,
                Token::Ident("g".into()),
                Token::Eq,
                Token::Int(2),
                Token::Eof
            ]
        );
        // 'record' should be ident, not 'rec' + 'ord'
        assert_eq!(
            tokens("record"),
            vec![Token::Ident("record".into()), Token::Eof]
        );
        // 'android' should be ident
        assert_eq!(
            tokens("android"),
            vec![Token::Ident("android".into()), Token::Eof]
        );
    }
}
