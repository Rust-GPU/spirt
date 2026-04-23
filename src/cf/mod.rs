//! Control-flow abstractions and passes.
//
// FIXME(eddyb) consider moving more definitions into this module.

use crate::{scalar, spv};

// NOTE(eddyb) all the modules are declared here, but they're documented "inside"
// (i.e. using inner doc comments).
pub mod cfgssa;
pub mod structurize;
pub mod unstructured;

// FIXME(eddyb) consider interning this.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum SelectionKind {
    /// Two-case selection based on boolean condition, i.e. `if`-`else`, with
    /// the two cases being "then" and "else" (in that order).
    BoolCond,

    /// `N+1`-case selection based on comparing an integer scrutinee against
    /// `N` constants, i.e. `switch`, with the last case being the "default"
    /// (making it the only case without a matching entry in `case_consts`).
    Switch {
        // FIXME(eddyb) avoid some of the `scalar::Const` overhead here, as there
        // is only a single type and we shouldn't need to store more bits per case,
        // than the actual width of the integer type.
        // FIXME(eddyb) consider storing this more like sorted compressed keyset,
        // as there can be no duplicates, and in many cases it may be contiguous.
        case_consts: Vec<scalar::Const>,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ExitInvocationKind {
    SpvInst(spv::Inst),
}
