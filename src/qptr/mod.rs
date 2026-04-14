//! [`QPtr`](crate::TypeKind::QPtr)-related type definitions and passes.
//
// FIXME(eddyb) consider `#[cfg(doc)] use crate::TypeKind::QPtr;` for doc comments.
// FIXME(eddyb) PR description of https://github.com/EmbarkStudios/spirt/pull/24
// has more useful docs that could be copied here.
//
// FIXME(eddyb) fully update post-`mem`-split.

use crate::{AddrSpace, OrdAssertEq, Type};
use std::num::NonZeroU32;
use std::ops::Range;

// NOTE(eddyb) all the modules are declared here, but they're documented "inside"
// (i.e. using inner doc comments).
pub mod lift;
pub mod lower;

/// `QPtr`-specific attributes ([`Attr::QPtr`]).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QPtrAttr {
    /// When applied to a `DataInst` with a `QPtr`-typed `inputs[input_idx]`,
    /// this describes the original `OpTypePointer` consumed by an unknown
    /// SPIR-V instruction (which may, or may not, access memory, at all).
    ///
    /// Assumes the original SPIR-V `StorageClass` is redundant (i.e. can be
    /// deduced from the pointer's provenance), and that any accesses performed
    /// through the pointer (or any pointers derived from it) stay within bounds
    /// (i.e. logical pointer semantics, unsuited for e.g. `OpPtrAccessChain`).
    //
    // FIXME(eddyb) reduce usage by modeling more of SPIR-V inside SPIR-T.
    ToSpvPtrInput { input_idx: u32, pointee: OrdAssertEq<Type> },

    /// When applied to a `DataInst` with a `QPtr`-typed output value,
    /// this describes the original `OpTypePointer` produced by an unknown
    /// SPIR-V instruction (likely creating it, without deriving from an input).
    ///
    /// Assumes the original SPIR-V `StorageClass` is significant (e.g. fresh
    /// provenance being created on the fly via `OpConvertUToPtr`, or derived
    /// internally by the implementation via `OpImageTexelPointer`).
    //
    // FIXME(eddyb) reduce usage by modeling more of SPIR-V inside SPIR-T, or
    // at least using some kind of bitcast instead of `QPtr` + this attribute.
    // FIXME(eddyb) `OpConvertUToPtr` creates a physical pointer, could we avoid
    // dealing with those at all in `QPtr`? (as its focus is logical legalization)
    FromSpvPtrOutput {
        // FIXME(eddyb) should this use a special `spv::StorageClass` type?
        addr_space: OrdAssertEq<AddrSpace>,
        pointee: OrdAssertEq<Type>,
    },
}

/// `QPtr`-specific operations ([`DataInstKind::QPtr`]).
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum QPtrOp {
    /// Adjust a **handle array** `QPtr` (`inputs[0]`), by selecting the handle
    /// at the index (`inputs[1]`) from the handle array (i.e. the resulting
    /// `QPtr` is limited to that one handle and can't be further "moved around").
    //
    // FIXME(eddyb) this could maybe use `DynOffset`, if `stride` is changed to
    // be `enum { Handle, Bytes(u32) }`, but that feels a bit too much?
    HandleArrayIndex,

    /// Get a **memory** `QPtr` pointing at the contents of the buffer whose
    /// handle is (implicitly) loaded from a **handle** `QPtr` (`inputs[0]`).
    //
    // FIXME(eddyb) should buffers be a `Type` of their own, that can be loaded
    // from a handle `QPtr`, and then has data pointer / length ops *on that*?
    BufferData,

    /// Get the length of the buffer whose handle is (implicitly) loaded from a
    /// **handle** `QPtr` (`inputs[0]`), converted to a count of "dynamic units"
    /// (as per [`shapes::MaybeDynMemLayout`]) by subtracting `fixed_base_size`,
    /// then dividing by `dyn_unit_stride`.
    //
    // FIXME(eddyb) should this handle _only_ "length in bytes", with additional
    // integer subtraction+division operations on lowering to `QPtr`, and then
    // multiplication+addition on lifting back to SPIR-V, followed by simplifying
    // the redundant `(x * a + b - b) / a` to just `x`?
    //
    // FIXME(eddyb) actually lower `OpArrayLength` to this!
    BufferDynLen { fixed_base_size: u32, dyn_unit_stride: NonZeroU32 },

    /// Adjust a **memory** `QPtr` (`inputs[0]`), by adding a (signed) immediate
    /// amount of bytes to its "address" (whether physical or conceptual).
    //
    // FIXME(eddyb) some kind of `inbounds` would be very useful here, up to and
    // including "capability slicing" to limit the usable range of the output.
    Offset(i32),

    /// Adjust a **memory** `QPtr` (`inputs[0]`), by adding a (signed) dynamic
    /// "index" (`inputs[1]`), multiplied by `stride` (bytes per element),
    /// to its "address" (whether physical or conceptual).
    DynOffset {
        stride: NonZeroU32,

        /// Bounds on the dynamic "index" (`inputs[1]`).
        //
        // FIXME(eddyb) should this be an attribute/refinement?
        index_bounds: Option<Range<i32>>,
    },
}
