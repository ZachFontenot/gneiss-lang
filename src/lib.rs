//! Gneiss - A statically-typed functional language with actors and channels

pub mod ast;
pub mod eval;
pub mod infer;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod types;

pub use ast::Program;
pub use eval::{Interpreter, Value};
pub use infer::Inferencer;
pub use lexer::Lexer;
pub use parser::Parser;
pub use types::{Type, TypeEnv};
