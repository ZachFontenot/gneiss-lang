//! Parser error types with enhanced context

use crate::ast::Span;
use crate::lexer::Token;
use thiserror::Error;

/// Parser error types with source location information
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("unexpected token: expected {expected}, found {found:?}")]
    UnexpectedToken {
        expected: String,
        found: Token,
        span: Span,
    },

    #[error("unexpected end of file")]
    UnexpectedEof { expected: String, last_span: Span },

    #[error("invalid pattern")]
    InvalidPattern { span: Span },
}

impl ParseError {
    /// Get the span associated with this error
    pub fn span(&self) -> &Span {
        match self {
            ParseError::UnexpectedToken { span, .. } => span,
            ParseError::UnexpectedEof { last_span, .. } => last_span,
            ParseError::InvalidPattern { span } => span,
        }
    }

    /// Create an UnexpectedToken error
    pub fn unexpected(expected: impl Into<String>, found: Token, span: Span) -> Self {
        ParseError::UnexpectedToken {
            expected: expected.into(),
            found,
            span,
        }
    }

    /// Create an UnexpectedEof error
    pub fn eof(expected: impl Into<String>, last_span: Span) -> Self {
        ParseError::UnexpectedEof {
            expected: expected.into(),
            last_span,
        }
    }
}

pub type ParseResult<T> = Result<T, ParseError>;
