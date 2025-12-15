//! Gneiss - A statically-typed functional language with actors and channels

pub mod ast;
pub mod errors;
pub mod eval;
pub mod infer;
pub mod lexer;
pub mod operators;
pub mod parser;
pub mod runtime;
pub mod test_support;
pub mod types;

pub use ast::{LocatedSpan, Position, Program, SourceMap, Span};
pub use errors::{
    find_similar, format_header, format_location, format_snippet, format_suggestions,
    levenshtein_distance, Colors, ErrorConfig,
};
pub use eval::{Interpreter, Value};
pub use infer::Inferencer;
pub use lexer::Lexer;
pub use parser::Parser;
pub use types::{Type, TypeEnv};
