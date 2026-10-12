//! Control-flow abstractions and passes.
//
// FIXME(eddyb) consider moving more definitions into this module.

use crate::spv;

// NOTE(eddyb) all the modules are declared here, but they're documented "inside"
// (i.e. using inner doc comments).
pub mod cfgssa;
pub mod structurize;
pub mod unstructured;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum SelectionKind {
    /// Two-case selection based on boolean condition, i.e. `if`-`else`, with
    /// the two cases being "then" and "else" (in that order).
    BoolCond,

    SpvInst(spv::Inst),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ExitInvocationKind {
    SpvInst(spv::Inst),
}
