//! Gneiss compiler backend: TAST → Mono → Core IR → FlatIR → C
//!
//! This module implements the Perceus-based compilation pipeline:
//! 1. Monomorphize TAST to MonoProgram (eliminate polymorphism)
//! 2. Lower MonoProgram to Core IR (ANF with explicit allocation)
//! 3. CPS transformation (for effects)
//! 4. Closure conversion (Core IR → Flat IR, eliminate lambdas)
//! 5. Perceus transformation (insert dup/drop) [TODO]
//! 6. Reuse analysis (in-place updates) [TODO]
//! 7. C code generation from Flat IR

pub mod c_emit;
pub mod closure;
pub mod core_ir;
pub mod cps_transform;
pub mod effect_analysis;
pub mod emit_flat;
pub mod lower_mono;
// pub mod perceus;  // TODO
// pub mod reuse;    // TODO

pub use c_emit::emit_c;
pub use closure::{closure_convert, FlatProgram, FlatExpr, FlatFn, FlatAtom};
pub use core_ir::*;
pub use cps_transform::*;
pub use effect_analysis::*;
pub use emit_flat::emit_flat_c;
pub use lower_mono::lower_mono;
