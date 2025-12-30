//! Gneiss compiler backend: TAST → Mono → Core IR → C
//!
//! This module implements the Perceus-based compilation pipeline:
//! 1. Monomorphize TAST to MonoProgram (eliminate polymorphism)
//! 2. Lower MonoProgram to Core IR (ANF with explicit allocation)
//! 3. Perceus transformation (insert dup/drop)
//! 4. Reuse analysis (in-place updates)
//! 5. C code generation

pub mod c_emit;
pub mod core_ir;
pub mod cps_transform;
pub mod effect_analysis;
pub mod lower;
pub mod lower_mono;
// pub mod perceus;  // TODO
// pub mod reuse;    // TODO

pub use c_emit::*;
pub use core_ir::*;
pub use cps_transform::*;
pub use effect_analysis::*;
pub use lower::*;
pub use lower_mono::lower_mono;
