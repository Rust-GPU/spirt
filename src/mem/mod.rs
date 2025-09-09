//! Memory operations, analyses and transformations.
//
// FIXME(eddyb) document at least these aspects:
// - "memory" = indirect storage of data and/or resources
// - (untyped) "data" = mix of plain bytes and pointers (as per RalfJ blog post)
//   (does "non-data memory" actually make sense? could be considered typed?)
//
// FIXME(eddyb) dig into past notes (e.g. `qptr::legalize`, Rust-GPU release notes,
// https://github.com/EmbarkStudios/spirt/pull/24, etc.) for useful docs.
//
// FIXME(eddyb) consider taking this into a more (R)VSDG "state type" direction.

use crate::{OrdAssertEq, Type};
use bitflags::bitflags;
use std::collections::BTreeMap;
use std::num::{NonZeroI32, NonZeroU32};
use std::rc::Rc;

// NOTE(eddyb) all the modules are declared here, but they're documented "inside"
// (i.e. using inner doc comments).
pub mod analyze;
pub mod const_data;
// FIXME(eddyb) make this public?
// HACK(eddyb) `pub` so that `spirti` can also rely on this.
pub mod layout;
pub mod shapes;

pub use layout::LayoutConfig;

/// Memory-specific attributes ([`Attr::Mem`]).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemAttr {
    /// When applied to a `GlobalVar` or `FuncLocalVar`, this tracks all possible
    /// access patterns its memory may be subjected to (see [`MemAccesses`]).
    //
    // FIXME(eddyb) either document that `qptr` offsetting ops also get this
    // attribute applied to them, or undo that change entirely.
    Accesses(OrdAssertEq<MemAccesses>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum MemAccesses {
    /// Accesses to one or more handles (i.e. optionally indexed by
    /// [`crate::qptr::QPtrOp::HandleArrayIndex`]), which can be:
    /// - `Handle::Opaque(handle_type)`: all accesses involve [`MemOp::Load`] or
    ///   [`crate::qptr::QPtrAttr::ToSpvPtrInput`], with the common type `handle_type`
    /// - `Handle::Buffer(data_happ)`: carries with it `data_happ`,
    ///   i.e. the access patterns for the memory that is reached through
    ///   [`crate::qptr::QPtrOp::BufferData`]
    Handles(shapes::Handle<DataHapp>),

    Data(DataHapp),
}

/// Data HAPP ("Hierarchical Access Pattern Partitioning"): all access patterns
/// for some memory, structured by disjoint offset ranges ("partitions").
///
/// This is the core of "type recovery" (inferring typed memory from untyped),
/// with "struct/"array" equivalents (i.e. as `DataHappKind` variants), but
/// while it can be mapped to an explicitly laid out data type, it also tracks
/// distinctions only needed during merging (e.g. [`DataHappKind::StrictlyTyped`]).
///
/// **Note**: the only reason for the recursive/"hierarchical" aspect is that
/// (array-like) dynamic indexing allows for compact representation of repeated
/// patterns, which can non-trivially nest for 3+ levels without losing the need
/// for efficient representation - in fact, one can construct a worst-case like:
/// ```ignore
/// [(A, [(B, [(C, [(D, [T; N], X); 2], Y); 2], Z); 2], W); 2]
/// ```
/// (with only `N * 2**4` leaf `T`s because of the 4 `[(_, ..., _); 2]` levels,
/// and the potential for dozens of such levels while remaining a plausible size)
//
// FIXME(eddyb) reconsider the name (acronym was picked to avoid harder decisions).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DataHapp {
    /// If present, this is a worst-case upper bound on the offsets at which
    /// accesses may be perfomed.
    //
    // FIXME(eddyb) use proper newtypes for byte amounts.
    //
    // FIXME(eddyb) suboptimal naming choice, but other options are too verbose,
    // including maybe using `RangeTo<_>` to explicitly indicate "exclusive".
    //
    // FIXME(eddyb) consider renaming such information to "extent", but that might
    // be ambiguous with an offset range (as opposed to using the maximum of all
    // *possible* `offset_range.end` values to describe a "maximum size").
    pub max_size: Option<u32>,

    pub flags: DataHappFlags,

    pub kind: DataHappKind,
}

impl DataHapp {
    pub const DEAD: Self =
        Self { max_size: Some(0), flags: DataHappFlags::empty(), kind: DataHappKind::Dead };
}

bitflags! {
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct DataHappFlags: u8 {
        /// Used as a source for `mem.copy` (of `0..max_size`), reading:
        /// - alongside `COPY_DST`: all bytes (see `COPY_SRC_AND_DST` for details)
        /// - without `COPY_DST`: leaf fields which could have only been written
        ///   by direct accesses, already tracked by (nested) `DataHapp`s
        ///
        /// **Note**: as this flag semantically covers an offset range, it's
        /// implicitly inherited by any (transitively) nested `DataHapp`s,
        /// and must be considered effectively present throughout all of them
        /// (e.g. for detecting and handling the `COPY_SRC_AND_DST` combination),
        /// even if not explicitly propagated inwards ahead of time.
        const COPY_SRC = 1 << 0;

        /// Used as a destination for `mem.copy` (of `0..max_size`), writing:
        /// - alongside `COPY_SRC`: all bytes (see `COPY_SRC_AND_DST` for details)
        /// - without `COPY_SRC`: leaf fields which could only be later read
        ///   by direct accesses, already tracked by (nested) `DataHapp`s
        ///
        /// **Note**: as this flag semantically covers an offset range, it's
        /// implicitly inherited by any (transitively) nested `DataHapp`s,
        /// and must be considered effectively present throughout all of them
        /// (e.g. for detecting and handling the `COPY_SRC_AND_DST` combination),
        /// even if not explicitly propagated inwards ahead of time.
        const COPY_DST = 1 << 1;

        /// Used as a source for some `mem.copy`s and a destination for others,
        /// making it unsound to rely solely on the offset ranges covered by the
        /// (nested) `DataHapp`s, as any contained bytes may be implicitly
        /// expected to have been preserved by the copy, regardless of whether
        /// direct accesses were observed by this `DataHapp` (so all gaps
        /// between those known accesses must be fully filled with leaf fields).
        ///
        /// **Note**: not a flag of its own, but simply the combination of
        /// `COPY_SRC` and `COPY_DST`, used to document their combined effect.
        const COPY_SRC_AND_DST = Self::COPY_SRC.bits() | Self::COPY_DST.bits();

        /// Mask for [`DataHappFlags::propagate_outwards`] (see its docs).
        const PROPAGATE_OUTWARDS_MASK = Self::empty().bits();
    }
}

impl DataHappFlags {
    /// Return the subset of flags in `self` that should remain present on an
    /// outer `DataHapp`, in which the `DataHapp` with `self` as flags will
    /// become nested (e.g. through `Disjoint`/`Repeated`).
    pub fn propagate_outwards(self) -> Self {
        self & Self::PROPAGATE_OUTWARDS_MASK
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum DataHappKind {
    /// Not actually accessed (only an intermediary state, during access analysis).
    //
    // FIXME(eddyb) use `Option<DataHapp>` instead? or an empty `Disjoint`?
    Dead,

    // FIXME(eddyb) replace the two leaves with e.g. `Leaf(Type, LeafAccessKind)`.
    //
    //
    /// Accesses through typed pointers (e.g. via unknown SPIR-V instructions),
    /// requiring a specific choice of pointee type which cannot be modified,
    /// and has to be reused as-is, when lifting to typed memory.
    ///
    /// Other overlapping accesses can be merged into this one as long as they
    /// can be fully expressed using the (transitive) components of this type.
    StrictlyTyped(Type),

    /// Direct accesses (e.g. [`MemOp::Load`], [`MemOp::Store`]), which can be
    /// decomposed as necessary (down to individual scalar leaves), to allow
    /// maximal merging opportunities.
    //
    // FIXME(eddyb) track whether accesses are `Load`s and/or `Store`s, to allow
    // inferring `NonWritable`/`NonReadable` annotations, as well.
    Direct(Type),

    /// Partitioning into disjoint offset ranges (the map is keyed by the start
    /// of the offset range, while the end is implied by its corresponding value),
    /// requiring a "struct" type, when lifting to typed memory.
    //
    // FIXME(eddyb) make this non-nestable and the fundamental basis of "HAPP".
    Disjoint(Rc<BTreeMap<u32, DataHapp>>),

    /// `Disjoint` counterpart for dynamic offsetting, requiring an "array" type,
    /// when lifting to typed memory, with one single element type being repeated
    /// across the entire size, at all offsets that are a multiple of `stride`.
    Repeated {
        // FIXME(eddyb) this feels inefficient.
        element: Rc<DataHapp>,
        stride: NonZeroU32,
    },
}

/// Memory-specific operations ([`DataInstKind::Mem`]).
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum MemOp {
    // HACK(eddyb) `OpVariable` replacement, which itself should not be kept as
    // a `SpvInst` - once fn-local variables are lowered, this should go there.
    FuncLocalVar(shapes::MemLayout),

    /// Read a single value from a pointer (`inputs[0]`) at `offset`.
    //
    // FIXME(eddyb) limit this to data - and scalars, maybe vectors at most.
    Load {
        // FIXME(eddyb) make this an `enum`, with another variant encoding some
        // GEP-like (aka SPIR-V `OpAccessChain`) "field path", and/or even one
        // (or more) stride(s) to allow direct array indexing at access time.
        offset: Option<NonZeroI32>,
    },

    /// Write a single value (`inputs[1]`) to a pointer (`inputs[0]`) at `offset`.
    //
    // FIXME(eddyb) limit this to data - and scalars, maybe vectors at most.
    Store {
        // FIXME(eddyb) make this an `enum`, with another variant encoding some
        // GEP-like (aka SPIR-V `OpAccessChain`) "field path", and/or even one
        // (or more) stride(s) to allow direct array indexing at access time.
        offset: Option<NonZeroI32>,
    },

    /// Copy `size` bytes to a "destination" pointer (`inputs[0]`), from a "source"
    /// pointer (`inputs[1]`), similar to a set of `Load`s and `Store`s, but with
    /// the actual value types left undetermined for as long as possible.
    Copy {
        // FIXME(eddyb) add two immediate offsets? allow dynamic length?
        size: NonZeroU32,
    },
    // FIXME(eddyb) implement more ops (e.g. atomics).
}
