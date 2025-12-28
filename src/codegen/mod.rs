//! Gneiss compiler backend: AST → Core IR → C
//!
//! This module implements the Perceus-based compilation pipeline:
//! 1. Lower AST to Core IR (ANF with explicit allocation)
//! 2. Perceus transformation (insert dup/drop)
//! 3. Reuse analysis (in-place updates)
//! 4. C code generation

pub mod c_emit;
pub mod core_ir;
pub mod lower;
// pub mod perceus;  // TODO
// pub mod reuse;    // TODO

pub use c_emit::*;
pub use core_ir::*;
pub use lower::*;
