//! [`QPtr`](crate::TypeKind::QPtr) legalization (e.g. for SPIR-V "logical addressing").
//!
//! # Memory vs pointers: semantics and limitations
//!
//! Memory semantics can usually be placed into one of two groups:
//! - "typed memory": accesses must strictly fit allocation ("object") types
//!   - often associated with GC-oriented languages/runtimes (including wasm GC),
//!     where the types tend to be limited to `struct`s and arrays (no `union`s),
//!     and serve to enforce memory safety, aid in precise GC, etc.
//!   - **SPIR-V "logical addressing"** is an unusual entry in this category,
//!     as other than resource handles (image/sampler/etc. "descriptors"),
//!     memory types are limited to plain data, buffers even using explicit layout
//!     (may eventually be rectified by extensions like `SPV_KHR_untyped_pointers`)
//! - "untyped memory": free-form accesses without any predeclared structure
//!   - underpins languages like C and Rust, IRs like LLVM and wasm, etc.
//!   - sometimes described as "array of bytes", though `[u8]` isn't accurate
//!     (see <https://www.ralfj.de/blog/2018/07/24/pointers-and-bytes.html>)
//!   - can be emulated on top of typed memory, if all values written to memory
//!     are representable through common types (i.e. integers, `u8` or larger),
//!     optionally combined with best-effort static analysis aka "type recovery"
//!
//! Pointer *values*, on top of accessing memory, can also be categorized by:
//! - granularity: from whole objects (GC refs) down to byte-level offsetting
//! - bitwise encoding: from hidden/read-only (GC refs) to mutably exposed bits
//!   - the actual choice of encoding is often a memory address, even if hidden
//!     (or e.g. a 32-bit offset from common base, aka "pointer compression")
//! - dynamic dataflow (`if`-`else`/loops/etc.): typically supported if types match
//!   (and even GC refs often allow type mixing through e.g. subtyping casts)
//! - indirect storage: like dynamic dataflow, but writing to compatible memory
//!   (at least for GC runtimes, this is *the* reason memory is typed at all)
//!
//! **SPIR-V "logical addressing"** is also an outlier here, disallowing what it
//! calls "variable pointers" (i.e. dynamic choice that is not array indexing),
//! even between same-address-space pointers *or pointing into the same memory*,
//! and offering no memory storage of pointers, not even relying on typed memory.
//!
//! # Legalization vs type recovery
//!
//! The worst-case semantic mismatch arises in SPIR-T between `qptr`-style
//! untyped memory and pointers (where even address spaces are erased), and
//! **SPIR-V "logical addressing"**, and that is tackled in two steps:
//! - [`legalize`](super::legalize) handles dynamic dataflow/indirect storage
//!   - chooses representations for both dynamic dataflow (locally, very flexible)
//!     and indirect storage (global encoding space, limited by pointer width)
//!   - the result is static usage: all pointers are global/local bindings + offsets
//! - [`analyze`](super::analyze) and [`lift`](super::lift) handle untyped memory
//!   - may become (partially) unnecessary with e.g. `SPV_KHR_untyped_pointers`
//!   - greatly limited in scope/difficulty by complete legalization
// FIXME(eddyb) ^^ refactor `analyze`+`lift` to take advantage of this?
//!   - "type recovery" done on a best-effort basis, as most bindings could use
//!     e.g. flat `[u32; N]` arrays (at some potential/unknown performance cost)
//!   - the result is typed memory accessed through SPIR-V typed pointers
//
// FIXME(eddyb) should the above docs go somewhere more general than `legalize`?

use crate::cf::{self, SelectionKind};
use crate::func_at::{FuncAt, FuncAtMut};
use crate::mem::LayoutConfig;
use crate::mem::MemOp;
use crate::qptr::QPtrOp;
use crate::transform::{InnerInPlaceTransform as _, Transformed, Transformer};
use crate::{
    AddrSpace, Attr, AttrSet, Const, ConstDef, ConstKind, Context, DeclDef, Diag, DiagLevel,
    EntityOrientedDenseMap, Func, FuncDecl, FuncDefBody, FuncParam, FxIndexMap, FxIndexSet,
    GlobalVar, GlobalVarDecl, GlobalVarDefBody, GlobalVarInit, Module, Node, NodeDef, NodeKind,
    Region, RegionDef, Type, TypeDef, TypeKind, Value, Var, VarDecl, VarKind, scalar, spv,
};
use crate::{EntityDefs, visit};
use itertools::{Either, EitherOrBoth, Itertools};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cell::Cell;
use std::collections::VecDeque;
use std::hash::Hash;
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::RangeTo;
use std::rc::Rc;
use std::{iter, mem};

// NOTE(eddyb) the current implementation of this pass relies on gathering a
// single set of "escaped pointers" (or rather, their shapes), then injecting
// them back elsewhere, which can be treated as a single fixpoint analysis
// around the whole module,
//
// i.e. in a similar vein to how "loop state variables" can be analyzed as
// e.g. `µX. initial_inputs[i] | loop_body({ .inputs[i] = X }).outputs[i]`
// (for the `i`th loop body input/output), but instead of relying on cycles
//
// (like recursion, though realistically you can't *just* use the call graph,
// it'd need to be a dependency graph across all call sites, and that can easily
// have a lot of cycles, e.g. `f(); g();` vs `g(); f();` in different functions
// implies a cyclic dep graph with a very similar shape to e.g. MEME loop CFGs),
//
// the current analysis looks more like `µE. scan_module_for_escapes(M, E)`,
// implemented like `scan_module_for_escapes(M, AnyEscapedBase)`, where the
// `AnyEscapedBase` ZST carries no information and is a stand-in for a single
// existential/inference variable, whose concrete result (`EscapedBaseMap`)
// cannot be meaningfully observed by `scan_module_for_escapes` *through* the
// `AnyEscapedBase` type (`ScanBasesInFunc` is accumulating an `EscapedBaseMap`,
// so in it theory it could get "the current state of its fixpoint result",
// it's not really considered "complete" in the same way `BaseChoice::Many` is).
//
// (also, `scan_module_for_escapes` isn't a method in the code, but rather refers
// to all the `ScanBasesInFunc::scan_func` calls - one per function in the module)
//
// FIXME(eddyb) the summary of all that is that reasoning in terms of dep graphs
// of analysis information, and always handling cycles as computing fixpoints
// (usually written e.g. `µX. f(X)`, or at least here, i.e. above in this comment),
// might be generally beneficial (see also the `flow` pass), but in order to
// make something like that foolproof, it would have to be designed around some
// kind of highly compositional system, where e.g. `MyAnalysis[op(x)]` would be
// represented more like "a derivative wrt `x`", arguably `MyAnalysis[op]`, and
// that information composing across dataflow, i.e. `MyAnalysis[g(f(x))]` being
// obtained by composing `MyAnalysis[x] |> MyAnalysis[f] |> MyAnalysis[g]`, even
// if `x` was known all along (this likely maps to some specific concept in e.g.
// category theory, but that may depend on the properties of choice), and then
// fixpoints like `µX. MyAnalysis[initialX] | MyAnalysis[loop_body(X)]` would
// become `exists N: Nat. MyAnalysis[initialX] |> MyAnalysis[loop_body]**N`,
// i.e. "union of all possible iteration counts", but, *crucially*, "counting"
// never happens, `**N` is just a conceptual tool here, and ideally alway `O(1)`
// (also, the derivative of `loop_body`'s `i`th output wrt its `i`th input would
// be more accurate to use but the syntax used above is verbose enough as it is).

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum GlobalBase {
    // HACK(eddyb) this is never added to base sets, and gets represented as:
    // - in `BaseChoice::Many`: an `undef` value in `base_index_selector`
    // - when encoding an escaping pointer: an `undef` integer result
    // (in fact, escaping pointer encoding/decoding mapping `undef` to `undef`,
    // instead of causing immediate UB, enabling e.g. reading pointers from
    // uninit memory, is the main motivation behind this and related choices)
    //
    // FIXME(eddyb) could this instead be e.g. another `BaseChoice` variant?
    Undef,

    // FIXME(eddyb) support other (invalid) constant integer address pointers.
    //
    // FIXME(eddyb) reserve the `0` encoding, and even up to some amount, maybe
    // `0..max(align_of(...all_globals_and_locals))` (and that doesn't even
    // take into account dynamic allocation!), for `Null`, so that the bits
    // themselves, when read as non-pointer, match up.
    //
    // FIXME(eddyb) consider introducing "byte types" to replace integer types
    // in memory interactions, reducing ambiguity between "exposed provenance"
    // and "pointer address" - sadly, roundtripping invalid pointers is trick
    // even after adding "byte types", as they can wildly overlap valid pointers,
    // and unlike CHERI,  there is no place to hide out-of-band "validity" bits
    // (at least for buffers - "module-owned" globals can theoretically have
    // a "shadow" version which stores validity bits like this, at some cost).
    Null,

    // TODO(eddyb) rename `GlobalVar` to use the new "binding" terminology.
    GlobalVar {
        // HACK(eddyb) `ConstKind::PtrToGlobalVar` is a more convenient key,
        // and just as cheap wrt hashing, than the `GlobalVar` it references.
        ptr_to_binding: Const,
    },

    // FIXME(eddyb) support `HandleArrayIndex`.
    BufferData {
        // HACK(eddyb) `ConstKind::PtrToGlobalVar` is a more convenient key,
        // and just as cheap wrt hashing, than the `GlobalVar` it references.
        ptr_to_buffer: Const,
    },
}

// FIXME(eddyb) should this be somewhere else?
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum AliasGroup {
    /// Anything that *might* reside in (and/or be backed by, for resources)
    /// the most general form of device (aka "global") memory, i.e.:
    /// - `OpenCL` `CrossWorkgroup` / Vulkan `PhysicalStorageBuffer`, which
    ///   are, strictly speaking, the only ones *required* to be device-scoped,
    ///   because they offer device-wide virtual addressing (on top of an MMU)
    /// - Vulkan `StorageBuffer` (buffer form of `PhysicalStorageBuffer`)
    /// - "constant" memory (`OpenCL` `UniformConstant` / Vulkan `Uniform` buffers),
    ///   allowed, *but not required*, to be implemented differently in hardware
    ///   (host APIs may impose additional limitations, but it's up to the driver)
    /// - texel memory (`Image`), almost(?) always the same as "constant" memory
    ///   (historically, GPU shader memory access evolved from "texture caches")
    /// - Vulkan resource handles (`UniformConstant`), almost(?) always using
    ///   one of the above flavors of memory for the actual storage of their
    ///   underlying data (e.g. `OpTypeImage` handles point to texel memory)
    //
    // FIXME(eddyb) reframe as "imported"/"external" memory, contrasting with
    // memory owned by the module itself.
    DeviceScopedMemory,

    /// Any `Workgroup` global binding having a `Block`-decorated type, i.e.
    /// relying on the feature enabled by `WorkgroupMemoryExplicitLayoutKHR`,
    /// where all such global bindings become *views* (of different types),
    /// into a single `Workgroup`-scoped memory allocation, as opposed to
    /// the default (each `Workgroup` global referring to disjoint memory).
    //
    // FIXME(eddyb) remove by lowering `WorkgroupMemoryExplicitLayoutKHR` to
    // a single workspace global binding definition, in `qptr::lower`.
    WorkgroupSingletonMemory,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct FuncLocal {
    // FIXME(eddyb) needs a better name (name due to `Var` refactor).
    qptr_output: Var,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Base {
    Global(GlobalBase),
    FuncLocal(FuncLocal),
    // FIXME(eddyb) support `OpImageTexelPointer` and any other exotic pointers.
}

/// A pointer offset (relative to some [`Base`]), generic over the value type `V`,
/// so that e.g. `Offset<Type>` can be used as an "offset shape".
//
// FIXME(eddyb) does this need a better name?
// FIXME(eddyb) support constant offsets and/or track offset ranges as well.
// HACK(eddyb) `Offset` methods are elsewhere, to keep these definitions readable.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
enum Offset<V> {
    #[default]
    Zero,

    /// When applied to some pointer `ptr`, equivalent to
    /// `QPtrOp::DynOffset { stride }` with `[ptr, index]` as inputs.
    //
    // FIXME(eddyb) track index bounds and/or a constant offset, as well?
    // FIXME(eddyb) maybe use `Scaled<V>` to avoid repeating the word "stride"?
    Strided { stride: NonZeroU32, index: V },
}

/// Shallow categorization of a `Value`'s definition, in terms of how it should
/// be legalized (if applicable, i.e. `qptr`-typed, or derived from `qptr`s).
//
// HACK(eddyb) the `Law` suffix is a shorthand for "legalization rule/step",
// and picked to stop bikeshedding over other mostly-too-long name choices.
// TODO(eddyb) rename `Law` to `Legality`.
#[derive(Copy, Clone)]
enum DefLaw {
    /// Any `qptr`-typed `Value` definition, (see also [`QPtrDefLaw`]).
    QPtr(QPtrDefLaw),

    /// Boolean output of a pointer `==`/`!=` (determined by `not_eq`) comparison,
    /// legalizable by comparing `BaseChoice::Many` and/or dynamic index, directly,
    /// without case-splitting (see [`QPtrExpansions::instantiate_by_multi_base_qptr_inputs`]),
    /// which would otherwise be quadratic (due to having two pointer inputs).
    ///
    /// In order for this approach to be sound, disjoint input bases are needed,
    /// i.e. every pair of "syntactically" different bases (`base1 != base2`, at
    /// the `Base: Eq` level, e.g. different `GlobalVar`s), from either input,
    /// **must** have zero bytes of overlap (i.e. *never* alias) - e.g. if the
    /// bases are both buffers, their annotations (or lack thereof), should imply
    /// it's UB for the host to dynamically bind the same memory to both of them.
    ///
    /// On top of not generating nested `switch`es full of pointer comparisons
    /// (many of which would be trivially `!=`), special-casing pointer `==`/`!=`
    /// comparisons means they can be fully legalized away, which helps whenever
    /// they wouldn't have been legal even for trivial (and legal) pointers,
    /// not to mention typed pointers require equal pointee types, which would
    /// complicate `mem::analyze` (requiring it to do global type inference).
    CmpQPtrEqOrNe { not_eq: bool },

    /// Integer output of a pointer-to-integer cast, already of the right type
    /// (i.e. the `qptr`-sized uint used for encoding/decoding escaped pointers),
    /// legalizable via integer ops using the `EscapedBaseMap` (i.e. combining
    /// the base choice and the index), which replace the original cast entirely.
    ///
    /// Currently includes:
    /// - `OpBitcast` from a pointer to the encoded integer type
    /// - `OpConvertPtrToU` with no implied narrowing/widening of the output
    ///   (i.e. only the encoded integer type is supported, nothing smaller/larger)
    ///
    /// This is also used (specifically through `OpConvertPtrToU`), as part of
    /// the implementation of [`QPtrExpansions::pre_encode_escaping_input`],
    /// by splitting out the encoding, e.g.:
    /// `mem.store(m, p)` -> `addr = spv.OpConvertPtrToU(p); mem.store(m, addr)`
    /// (the `spv.OpConvertPtrToU(p)` node then gets legalized separately).
    //
    // FIXME(eddyb) `qptr`-native `OpConvertPtrToU` replacement?
    //
    // FIXME(eddyb) consider introducing "byte types" to replace integer types
    // in encoding/decoding, reducing ambiguity between "exposing provenance"
    // and "get address from pointer".
    EncodeEscapingQPtr,
}

/// Shallow categorization of a `qptr`-typed `Value`'s definition, in terms of
/// how it should be legalized (i.e. [`DefLaw::QPtr`], see also [`DefLaw`]).
//
// HACK(eddyb) the `Law` suffix is a shorthand for "legalization rule/step",
// and picked to stop bikeshedding over other mostly-too-long name choices.
// TODO(eddyb) rename `Law` to `Legality`.
// FIXME(eddyb) legalize the integer output of same-base pointer subtractions.
#[derive(Copy, Clone)]
enum QPtrDefLaw {
    /// Any of the supported pointer [`Base`]s (i.e. global/local bindings),
    /// already legal on their own, but also the only pointers not derived from
    /// other pointers, therefore forming the basis of all pointer legalization.
    Base(Base),

    /// Pointer output of `QPtrOp::Offset` or `QPtrOp::DynOffset`, legalizable
    /// if the input pointer is (i.e. by summing up offsets into dynamic indices,
    /// while simply passing through the choice of base, even `BaseChoice::Many`).
    Offset(Offset<IndexValue>),

    /// Pointer output of an integer-to-pointer cast, already taking the right type
    /// (i.e. the `qptr`-sized uint used for encoding/decoding escaped pointers),
    /// legalizable via integer ops using the `EscapedBaseMap` (i.e. extracting
    /// the base choice and the index), which replace the original cast entirely.
    ///
    /// Currently includes:
    /// - `OpBitcast` from the encoded integer type to a pointer
    /// - `OpConvertUToPtr` with no implied narrowing/widening
    ///   (i.e. only the encoded integer type is supported, nothing smaller/larger)
    ///
    /// This is also used (specifically through `OpConvertUToPtr`), as part of
    /// the implementation of [`QPtrExpansions::post_decode_escaped_output`],
    /// by splitting out the decoding, e.g.:
    /// - `p = mem.load(m)` -> `addr = mem.load(m); p = spv.OpConvertUToPtr(addr)`
    ///   (the `spv.OpConvertUToPtr(addr)` node then gets legalized separately)
    //
    // FIXME(eddyb) `qptr`-native `OpConvertUToPtr` replacement?
    //
    // FIXME(eddyb) consider introducing "byte types" to replace integer types
    // in encoding/decoding, reducing ambiguity between "exposing provenance"
    // and "get address from pointer".
    DecodeEscaped,

    /// Dynamic pointer (flow-dependent, similar to "variable pointer" in SPIR-V),
    /// legalizable if all the "source" pointers themselves are (i.e. through
    /// merging their `BaseChoice::Many` and/or dynamic indices).
    ///
    /// Currently includes:
    /// - `Select`/`Loop` node outputs (merging child `Region` outputs)
    /// - loop body region inputs (fixpoint-merging body outputs,
    ///   with initial inputs provided as part of the `Loop` node)
    ///
    /// In the common case of a pointer being advanced each loop iteration,
    /// the dynamic index (replacing it) is also known as an "induction variable"
    /// (rewriting loops from e.g. iterating over a `p..p.add(n)` pointer range,
    /// like Rust's slice iterators, to iterating over the `0..n` integer range).
    //
    // FIXME(eddyb) consider including function parameters/returns?
    // (gets complicated if trying to treat functions as "polymorphic", and it's
    // probably hard to do much better than treating them as always escaping)
    Dyn,

    /// Any other pointer-producing operation that hasn't been implemented yet.
    //
    // FIXME(eddyb) resolve all such cases.
    Unsupported,
}

// HACK(eddyb) helper type for `QPtrDefLaw::Offset`, to defer interning a `Const`
// in the `QPtrOp::Offset` case (not just for efficiency, but also because of
// some unfortunate limitations around index types width and/or signedness).
#[derive(Copy, Clone)]
enum IndexValue {
    Dyn(Value),
    One,
    MinusOne,
}

/// All `NodeKind`-agnostic pointer input/output legalizations.
///
/// Once `QPtrExpansions` is computed from a `NodeDef`, it will be applied:
/// - in field order (i.e. `instantiate_by_multi_base_qptr_inputs` done last)
/// - without consulting the `NodeKind` (beyond potentially cloning it)
struct QPtrExpansions {
    /// If present, the index of a pointer input of a `Node`, either the value
    /// being stored for an `MemOp::Store` (i.e. writing a pointer to memory),
    /// or the sole input of a bitcast (implying a pointer-to-integer cast,
    /// combined with bitcasting still, to the original bitcast's output type),
    /// legalizable by splitting out the encoding into a new `OpConvertPtrToU`
    /// node (i.e. [`DefLaw::EncodeEscapingQPtr`], see also its own docs),
    /// and replacing this input's value with that new node's integer output.
    pre_encode_escaping_input: Option<usize>,

    /// If present, the index of a pointer output of a `Node`, either the sole
    /// output of `MemOp::Load` (i.e. reading a pointer from memory),
    /// or the sole output of a bitcast (implying an integer-to-pointer cast,
    /// combined with bitcasting still, from the original bitcast's input type),
    /// legalizable by splitting out the decoding into a new `OpConvertUToPtr`
    /// node (i.e. [`QPtrDefLaw::DecodeEscaped`], see also its own docs),
    /// and moving this output `Var` to that new node's pointer output.
    post_decode_escaped_output: Option<usize>,

    /// If `true`, this `Node` lacks any special legalization needs/options
    /// (beyond `pre_encode_escaping_input` / `post_decode_escaped_output`),
    /// and also lacks child `Region`s (i.e. it's not a `Select` or a `Loop`),
    /// so pointer inputs (other than `pre_encode_escaping_input`) with a
    /// dynamic choice of base are only legalizable by case-splitting the base.
    ///
    /// When some pointer input needs case-splitting for its base, the entire
    /// `Node` definition gets duplicated for each case, and this "instantiation"
    /// process can repeat for several such inputs, resulting in nested `Select`s
    /// (bounded by a lack of many-pointer-taking instructions in e.g. SPIR-V,
    /// with binary ops being the expected worst-case, themselves mitigated by
    /// e.g. [`DefLaw::CmpQPtrEqOrNe`]).
    //
    // FIXME(eddyb) this doesn't take into account the interactions with any
    // so-called "tangled execution" (e.g. SPIR-V instructions which care about
    // convergence) - thankfully, pointers are almost never used in such cases,
    // but they could easily creep up in the future (in SPIR-V extensions etc.).
    instantiate_by_multi_base_qptr_inputs: bool,
}

pub struct LegalizePtrs<'a> {
    cx: Rc<Context>,
    wk: &'static spv::spec::WellKnown,

    config: &'a LayoutConfig,

    cached_u32_type: Cell<Option<Type>>,

    cached_qptr_type: Cell<Option<Type>>,
    cached_undef_qptr_const: Cell<Option<Const>>,
    cached_null_qptr_const: Cell<Option<Const>>,

    cached_qptr_sized_uint_type: Cell<Option<Type>>,

    // HACK(eddyb) pre-filled from (all reachable `GlobalVar`s in) the `Module`,
    // so `base_alias_group` can avoid needing access to the `GlobalVarDecl`,
    // but also this makes it trivially `O(1)` so perhaps it should be kept.
    global_var_to_alias_group: EntityOrientedDenseMap<GlobalVar, Option<AliasGroup>>,
}

impl<'a> LegalizePtrs<'a> {
    pub fn new(cx: Rc<Context>, config: &'a LayoutConfig) -> Self {
        Self {
            cx,
            wk: &spv::spec::Spec::get().well_known,

            config,

            cached_u32_type: Default::default(),

            cached_qptr_type: Default::default(),
            cached_undef_qptr_const: Default::default(),
            cached_null_qptr_const: Default::default(),

            cached_qptr_sized_uint_type: Default::default(),

            global_var_to_alias_group: Default::default(),
        }
    }

    pub fn legalize_module(
        &mut self,
        module: &mut Module,
        // FIXME(eddyb) automate this and/or make it a `Module` wrapper.
        all_uses_from_module: &visit::AllUses,
    ) {
        // HACK(eddyb) shared helper.
        let try_concat_mem_layouts = {
            use crate::mem::shapes::MemLayout;
            |MemLayout { align, legacy_align, size }, layout: MemLayout| {
                let offset_in_fused = size.checked_next_multiple_of(layout.align)?;

                let max_align = align.max(layout.align);
                let total_size = offset_in_fused.checked_add(layout.size)?;

                // HACK(eddyb) this is expected to not panic when computed
                // after everything has been fused together.
                let _total_aligned_size = total_size.checked_next_multiple_of(max_align)?;

                Some((
                    offset_in_fused,
                    MemLayout {
                        align: max_align,
                        legacy_align: legacy_align.max(layout.legacy_align),
                        size: total_size,
                    },
                ))
            }
        };

        // FIXME(eddyb) make these configurable.
        let fuse_all_private_globals = true;
        let reuse_fused_private_global_for_escaped_func_locals = true;

        // HACK(eddyb) fuse as many `GlobalVar` as possible (in theory, only
        // one `GlobalVar` should be needed for `Private` and `Workgroup` each,
        // but this only handles `Private` w/ initializers).
        // FIXME(eddyb) this should definitely take into account what's "escaped",
        // but doing this rewrite later on causes all sorts of complications.
        // FIXME(eddyb) there's not much stopping `all_uses_from_module` from
        // being accidentally used instead of the correct thing, sadly.
        let mut fused_private_global = None;
        let all_global_vars = {
            use crate::mem::shapes::{GlobalVarShape, MemLayout};

            let mut fused_gv_remap = EntityOrientedDenseMap::new();
            for &gv in &all_uses_from_module.global_vars {
                if !fuse_all_private_globals {
                    continue;
                }

                let GlobalVarDecl {
                    attrs,
                    type_of_ptr_to: _,
                    shape: Some(GlobalVarShape::UntypedData(layout)),
                    addr_space,
                    def:
                        DeclDef::Present(GlobalVarDefBody {
                            // FIXME(eddyb) also accept `None` and treat it as if
                            // it's `undef`, i.e. `ConstData::new(layout.size)`.
                            initializer: Some(GlobalVarInit::Data(ref init)),
                        }),
                } = module.global_vars[gv]
                else {
                    continue;
                };

                // FIXME(eddyb) remove some of these restrictions.
                if attrs != AttrSet::default()
                    || layout.size != init.size()
                    || addr_space != AddrSpace::SpvStorageClass(self.wk.Private)
                {
                    continue;
                }

                // HACK(eddyb) approximating a `try {...}` block.
                let Some((gv_offset_in_fused, new_fused_layout)) = try_concat_mem_layouts(
                    fused_private_global.map_or(
                        MemLayout { align: 1, legacy_align: 1, size: 0 },
                        |fused_gv| match module.global_vars[fused_gv].shape.unwrap() {
                            GlobalVarShape::UntypedData(layout) => layout,
                            _ => unreachable!(),
                        },
                    ),
                    layout,
                ) else {
                    continue;
                };

                // HACK(eddyb) "hollow out" the old `GlobalVar`.
                let init = {
                    let GlobalVarDecl {
                        shape: Some(GlobalVarShape::UntypedData(layout)),
                        def: DeclDef::Present(GlobalVarDefBody { initializer }),
                        ..
                    } = &mut module.global_vars[gv]
                    else {
                        unreachable!();
                    };
                    layout.size = 0;
                    let Some(GlobalVarInit::Data(init)) = initializer.take() else {
                        unreachable!();
                    };
                    init
                };

                let fused_gv = if let Some(fused_private_gv) = fused_private_global {
                    let GlobalVarDecl {
                        shape: Some(GlobalVarShape::UntypedData(fused_layout)),
                        def:
                            DeclDef::Present(GlobalVarDefBody {
                                initializer: Some(GlobalVarInit::Data(fused_init)),
                            }),
                        ..
                    } = &mut module.global_vars[fused_private_gv]
                    else {
                        unreachable!();
                    };

                    *fused_layout = new_fused_layout;

                    let fused_init_old_size = fused_init.size();
                    fused_init.grow(new_fused_layout.size);

                    // HACK(eddyb) zero out any padding, instead of leaving it
                    // `undef`, to increase `qptr::lift`'s chances of success.
                    for offset in fused_init_old_size..gv_offset_in_fused {
                        fused_init.write_bytes(offset, &[0]).unwrap();
                    }

                    let mut offset = gv_offset_in_fused;
                    for part in init.read(0..init.size()) {
                        use crate::mem::const_data::Part;

                        match part {
                            Part::Uninit { size } => {
                                // HACK(eddyb) avoids issues with uninitialized
                                // bytes partly overlapping with leaf scalars.
                                if true {
                                    for offset in offset..(offset + size.get()) {
                                        fused_init.write_bytes(offset, &[0]).unwrap();
                                    }
                                }
                            }
                            Part::Bytes(bytes) => {
                                fused_init.write_bytes(offset, bytes).unwrap();
                            }
                            Part::Symbolic { size, maybe_partial_slice: _, value } => {
                                fused_init.write_symbolic(offset, size, value).unwrap();
                            }
                        }
                        offset = offset.checked_add(part.size().get()).unwrap();
                    }

                    fused_private_gv
                } else {
                    let fused_gv = module.global_vars.define(
                        &self.cx,
                        GlobalVarDecl {
                            attrs: AttrSet::default(),
                            type_of_ptr_to: self.qptr_type(),
                            shape: Some(GlobalVarShape::UntypedData(new_fused_layout)),
                            addr_space: AddrSpace::SpvStorageClass(self.wk.Private),
                            def: DeclDef::Present(GlobalVarDefBody {
                                initializer: Some(GlobalVarInit::Data(init)),
                            }),
                        },
                    );
                    fused_private_global = Some(fused_gv);
                    fused_gv
                };

                fused_gv_remap.insert(gv, (fused_gv, gv_offset_in_fused));
            }

            if let Some(fused_private_gv) = fused_private_global {
                let GlobalVarDecl {
                    shape: Some(GlobalVarShape::UntypedData(fused_layout)),
                    def:
                        DeclDef::Present(GlobalVarDefBody {
                            initializer: Some(GlobalVarInit::Data(fused_init)),
                        }),
                    ..
                } = &mut module.global_vars[fused_private_gv]
                else {
                    unreachable!();
                };

                let aligned_size =
                    fused_layout.size.checked_next_multiple_of(fused_layout.align).unwrap();

                if aligned_size > fused_layout.size {
                    let fused_init_old_size = fused_init.size();
                    fused_init.grow(aligned_size);
                    fused_layout.size = aligned_size;

                    // HACK(eddyb) zero out any padding, instead of leaving it
                    // `undef`, to increase `qptr::lift`'s chances of success.
                    for offset in fused_init_old_size..aligned_size {
                        fused_init.write_bytes(offset, &[0]).unwrap();
                    }
                }
            }

            struct FusedGlobalVarRewriter<'a> {
                cx: &'a Context,

                fused_gv_remap: &'a EntityOrientedDenseMap<GlobalVar, (GlobalVar, u32)>,

                // FIXME(eddyb) build some automation to avoid ever repeating these.
                transformed_consts: FxHashMap<Const, Transformed<Const>>,
            }

            impl Transformer for FusedGlobalVarRewriter<'_> {
                // FIXME(eddyb) build some automation to avoid ever repeating these.
                fn transform_type_use(&mut self, _ty: Type) -> Transformed<Type> {
                    // FIXME(eddyb) this doesn't detect misuse.
                    Transformed::Unchanged
                }
                fn transform_const_use(&mut self, ct: Const) -> Transformed<Const> {
                    if let Some(&cached) = self.transformed_consts.get(&ct) {
                        return cached;
                    }
                    let transformed =
                        self.transform_const_def(&self.cx[ct]).map(|ct_def| self.cx.intern(ct_def));
                    self.transformed_consts.insert(ct, transformed);
                    transformed
                }

                fn transform_global_var_use(&mut self, _gv: GlobalVar) -> Transformed<GlobalVar> {
                    unreachable!("non-`ConstKind::PtrToGlobalVar` use of `GlobalVar`")
                }
                fn transform_func_use(&mut self, _func: Func) -> Transformed<Func> {
                    Transformed::Unchanged
                }

                fn transform_const_def(&mut self, ct_def: &ConstDef) -> Transformed<ConstDef> {
                    if let ConstKind::PtrToGlobalVar { global_var, offset } = ct_def.kind
                        && let Some(&(fused_gv, gv_offset_in_fused)) =
                            self.fused_gv_remap.get(global_var)
                    {
                        let mut attrs = ct_def.attrs;

                        let offset = offset.map_or(0, |o| o.get());
                        let total_offset =
                            gv_offset_in_fused.checked_add(offset).unwrap_or_else(|| {
                                attrs.push_diag(
                                    self.cx,
                                    Diag::bug([format!(
                                        "{gv_offset_in_fused} + {offset} overflowed"
                                    )
                                    .into()]),
                                );
                                // HACK(eddyb) `const_qptr_to_base_and_offset` will
                                // return `None` because of this.
                                // FIXME(eddyb) need better ways to systematically
                                // prevent diagnostics from being accidentally ignored.
                                u32::MAX
                            });
                        Transformed::Changed(ConstDef {
                            attrs,
                            ty: ct_def.ty,
                            kind: ConstKind::PtrToGlobalVar {
                                global_var: fused_gv,
                                offset: NonZeroU32::new(total_offset),
                            },
                        })
                    } else {
                        Transformed::Unchanged
                    }
                }
            }

            let mut rewriter = FusedGlobalVarRewriter {
                cx: &self.cx,
                fused_gv_remap: &fused_gv_remap,
                transformed_consts: FxHashMap::default(),
            };

            // FIXME(eddyb) consider filtering out all the fused globals, and
            // maybe even updating `AllUses`?
            let all_global_vars =
                all_uses_from_module.global_vars.iter().copied().chain(fused_private_global);

            for gv in all_global_vars.clone() {
                rewriter.in_place_transform_global_var_decl(&mut module.global_vars[gv]);
            }
            for &func in &all_uses_from_module.funcs {
                rewriter.in_place_transform_func_decl(&mut module.funcs[func]);
            }
            for export_idx in 0..module.exports.len() {
                let (crate::ExportKey::SpvEntryPoint { imms, interface_global_vars }, &exportee) =
                    module.exports.get_index(export_idx).unwrap()
                else {
                    continue;
                };

                // FIXME(eddyb) avoid the cloning cost here.
                let mut interface_global_vars = interface_global_vars.clone();

                let mut uses_fused_private_gv = false;
                interface_global_vars.retain_mut(|gv| {
                    if let Some(&(fused_gv, _)) = fused_gv_remap.get(*gv) {
                        *gv = fused_gv;

                        assert!(Some(fused_gv) == fused_private_global);
                        !mem::replace(&mut uses_fused_private_gv, true)
                    } else {
                        true
                    }
                });

                if uses_fused_private_gv {
                    // HACK(eddyb) because the key needs to change, the old entry
                    // needs to be removed, and a new one added, but moving the
                    // entries around can be avoided by using `splice` for this,
                    // at the cost of not getting to steal the old key first.
                    module.exports.splice(
                        export_idx..=export_idx,
                        [(
                            crate::ExportKey::SpvEntryPoint {
                                imms: imms.clone(),
                                interface_global_vars,
                            },
                            exportee,
                        )],
                    );
                }
            }

            all_global_vars
        };

        // HACK(eddyb) pre-filling `global_var_to_alias_group` before it's read.
        // NOTE(eddyb) in theory, `Legalizer` could hold `module.global_vars`,
        // via `Option<&EntityDefs<GlobalVar>>`.
        {
            let wk = self.wk;

            // FIXME(eddyb) these are taken from the SPIR-V spec, but `Image`
            // and `PhysicalStorageBuffer` can't be buffers themselves, and
            // `UniformConstant` ends up only as `AddrSpace::Handles` anyway.
            let device_scoped_storage_classes = [
                wk.Image,
                wk.StorageBuffer,
                wk.PhysicalStorageBuffer,
                wk.Uniform,
                wk.UniformConstant,
            ];

            let workgroup_singleton_storage_class;
            let restrict_by_default;

            match &module.dialect {
                crate::ModuleDialect::Spv(dialect) => {
                    restrict_by_default =
                        [wk.Simple, wk.GLSL450, wk.Vulkan].contains(&dialect.memory_model);
                    workgroup_singleton_storage_class = dialect
                        .capabilities
                        .contains(&wk.WorkgroupMemoryExplicitLayoutKHR)
                        .then_some(wk.Workgroup);
                }
            }
            let attr_flipping_default = Attr::SpvAnnotation(spv::Inst {
                opcode: wk.OpDecorate,
                imms: [spv::Imm::Short(
                    wk.Decoration,
                    if restrict_by_default { wk.Aliased } else { wk.Restrict },
                )]
                .into_iter()
                .collect(),
            });

            for gv in all_global_vars.clone() {
                let gv_decl = &module.global_vars[gv];

                let addr_space_of_var_or_buffer =
                    if let Some(crate::mem::shapes::GlobalVarShape::Handles {
                        handle: crate::mem::shapes::Handle::Buffer(addr_space, _),
                        ..
                    }) = gv_decl.shape
                    {
                        assert!(gv_decl.addr_space == AddrSpace::Handles);
                        assert!(addr_space != AddrSpace::Handles);
                        addr_space
                    } else {
                        gv_decl.addr_space
                    };
                let intrinsic_alias_group = match addr_space_of_var_or_buffer {
                    AddrSpace::Handles => Some(AliasGroup::DeviceScopedMemory),
                    AddrSpace::SpvStorageClass(sc) => {
                        if device_scoped_storage_classes.contains(&sc) {
                            Some(AliasGroup::DeviceScopedMemory)
                        } else if Some(sc) == workgroup_singleton_storage_class {
                            Some(AliasGroup::WorkgroupSingletonMemory)
                        } else {
                            None
                        }
                    }
                };

                self.global_var_to_alias_group.insert(
                    gv,
                    intrinsic_alias_group.filter(|_| {
                        !restrict_by_default
                            ^ self.cx[gv_decl.attrs].attrs.contains(&attr_flipping_default)
                    }),
                );
            }
        }

        let mut base_maps = BaseMaps::default();

        let mut anything_ever_encodes_or_decodes_escaped_ptrs = false;

        // FIXME(eddyb) consider using a `Visitor` to process all consts, but it
        // would lack the context necessary to do the additional error-checking.
        for gv in all_global_vars.clone() {
            let gv_decl = &mut module.global_vars[gv];
            let mut visit_const_use = |ct: Const, untyped_size: Option<u32>| {
                if !matches!(self.cx[self.cx[ct].ty].kind, TypeKind::QPtr) {
                    return;
                }

                let qptr_size = self.config.logical_ptr_size_align.0;
                let base_and_offset_or_err = match untyped_size {
                    Some(size) if size == qptr_size => {
                        self.const_qptr_to_base_and_offset(ct).ok_or_else(|| {
                            Diag::bug(["unsupported pointer `".into(), ct.into(), "`".into()])
                        })
                    }
                    Some(size) => Err(Diag::bug([
                        "`qptr` constant `".into(),
                        ct.into(),
                        format!("` occupies {size} bytes (expected {qptr_size} bytes)").into(),
                    ])),
                    None => Err(Diag::bug([
                        "cannot legalize `".into(),
                        ct.into(),
                        "` in unlowered initializer".into(),
                    ])),
                };
                match base_and_offset_or_err {
                    // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
                    Ok((GlobalBase::Undef, _)) => {}

                    Ok((global_base, offset)) => {
                        anything_ever_encodes_or_decodes_escaped_ptrs = true;

                        let offset_shape = offset.map_index_value(|_| self.u32_type());
                        let merged_offset_shape =
                            base_maps.escaped.bases.entry(global_base).or_insert(Offset::Zero);
                        self.in_place_merge_offset_shapes(merged_offset_shape, offset_shape);
                    }

                    Err(err) => gv_decl.attrs.push_diag(&self.cx, err),
                }
            };
            if let DeclDef::Present(gv_def) = &gv_decl.def
                && let Some(init) = &gv_def.initializer
            {
                match init {
                    GlobalVarInit::Data(const_data) => {
                        for part in const_data.read(0..const_data.size()) {
                            use crate::mem::const_data::Part;
                            match part {
                                Part::Uninit { .. } | Part::Bytes(_) => {}
                                Part::Symbolic { size, maybe_partial_slice: _, value } => {
                                    visit_const_use(value, Some(size.get()));
                                }
                            }
                        }
                    }
                    &GlobalVarInit::Direct(ct) => {
                        let untyped_size = match gv_decl.shape {
                            Some(crate::mem::shapes::GlobalVarShape::UntypedData(mem_layout)) => {
                                Some(mem_layout.size)
                            }
                            Some(
                                crate::mem::shapes::GlobalVarShape::Handles { .. }
                                | crate::mem::shapes::GlobalVarShape::TypedInterface(_),
                            )
                            | None => None,
                        };
                        visit_const_use(ct, untyped_size);
                    }
                    GlobalVarInit::SpvAggregate { ty: _, leaves } => {
                        for &leaf in leaves {
                            visit_const_use(leaf, None);
                        }
                    }
                }
            }
        }

        // See `ScanBasesInFunc` description of the same field.
        let mut write_back_escaped_ptr_offset_shape = None;

        // FIXME(eddyb) this might be too much state to keep around at once.
        struct FuncScanResults {
            // HACK(eddyb) least effort so far seems to be to just keep around
            // the map from during summarization, at the cost of `PtrSummary`'s
            // (larger) size (i.e. a time-vs-space tradeoff).
            summarized_ptrs: FxIndexMap<Var, PtrSummary>,

            // HACK(eddyb) tracks whether escaping pointers are ever interacted with,
            // as both directions require e.g. adjusting `FuncLocalBaseMap` later.
            ever_encodes_or_decodes_escaped_ptrs: bool,
        }

        let mut fused_private_global_for_escaped_func_locals =
            if reuse_fused_private_global_for_escaped_func_locals {
                fused_private_global
            } else {
                None
            };

        let mut per_func_scan_results = EntityOrientedDenseMap::new();
        for &func in &all_uses_from_module.funcs {
            if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
                let loop_map = LoopMap::new(func_def_body);
                let mut scanner = ScanBasesInFunc {
                    legalizer: self,
                    loop_map: &loop_map,

                    escaped_base_map: &mut base_maps.escaped,
                    func_base_map: FuncLocalBaseMap::default(),

                    ever_encodes_or_decodes_escaped_ptrs: false,
                    write_back_escaped_ptr_offset_shape: None,
                    escaping_func_locals: FxIndexMap::default(),
                    summarized_ptrs: FxIndexMap::default(),
                };
                scanner.scan_func(func_def_body);

                if let Some((AnyEscapedBase, offset_shape)) =
                    scanner.write_back_escaped_ptr_offset_shape
                {
                    self.in_place_merge_offset_shapes(
                        &mut write_back_escaped_ptr_offset_shape
                            .get_or_insert((AnyEscapedBase, Offset::Zero))
                            .1,
                        offset_shape,
                    );
                }

                let ScanBasesInFunc {
                    mut func_base_map,
                    ever_encodes_or_decodes_escaped_ptrs,
                    escaping_func_locals,
                    mut summarized_ptrs,
                    ..
                } = scanner;

                for PtrSummary { bases, .. } in summarized_ptrs.values() {
                    if let Some(AnyEscapedBase) = bases.as_ref().right().and_then(|u| u.any_escaped)
                    {
                        assert!(ever_encodes_or_decodes_escaped_ptrs);

                        // Account for this pointer needing `BaseChoice::Many`,
                        // once escaped bases are also included, while `base`
                        // might otherwise not end up in `many_choice_bases`.
                        if let Some(&BaseChoice::Single(base)) = bases.as_ref().left() {
                            // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
                            if !matches!(base, Base::Global(GlobalBase::Undef)) {
                                func_base_map.many_choice_bases.insert(base);
                            }
                        }
                    }
                }

                let last_func_local_var =
                    (!escaping_func_locals.is_empty()).then_some(()).and_then(|()| {
                        func_def_body
                            .at_body()
                            .at_children()
                            .into_iter()
                            .take_while(|func_at_node| {
                                matches!(
                                    func_at_node.def().kind,
                                    NodeKind::Mem(MemOp::FuncLocalVar(_))
                                )
                            })
                            .map(|func_at_node| func_at_node.position)
                            .last()
                    });

                // HACK(eddyb) replace escaping `Base::FuncLocal`s with `Private`
                // globals, so that they can be referenced across functions.
                // FIXME(eddyb) enforce the lack of reentrance at this point
                // (in practice, e.g. callstack emulation will run first).
                let mut cursor = func_def_body.at_mut(InsertPosition {
                    parent_region: func_def_body.body,
                    anchor: last_func_local_var.map_or(InsertAnchor::First, InsertAnchor::After),
                });
                for (local, escaping_offset_shape) in escaping_func_locals {
                    use crate::mem::shapes::{GlobalVarShape, MemLayout};

                    let func = cursor.reborrow().at(());
                    let local_def =
                        &mut func.nodes[func.vars[local.qptr_output].def_parent.right().unwrap()];
                    let NodeKind::Mem(MemOp::FuncLocalVar(layout)) = local_def.kind else {
                        unreachable!();
                    };

                    // TODO(eddyb) support (by injecting a `mem.store`
                    // just after the original declaration position).
                    assert_eq!(local_def.inputs.len(), 0);

                    let mk_gv_decl = |layout| GlobalVarDecl {
                        attrs: Default::default(),
                        type_of_ptr_to: self.qptr_type(),
                        shape: Some(GlobalVarShape::UntypedData(layout)),
                        addr_space: AddrSpace::SpvStorageClass(self.wk.Private),

                        def: DeclDef::Present(GlobalVarDefBody { initializer: None }),
                    };
                    // FIXME(eddyb) make this configurable.
                    let (gv, offset_in_gv) = if true {
                        // FIXME(eddyb) this is the opposite of elegant.
                        let fused_gv = *fused_private_global_for_escaped_func_locals
                            .get_or_insert_with(|| {
                                module.global_vars.define(
                                    &self.cx,
                                    mk_gv_decl(MemLayout { align: 1, legacy_align: 1, size: 0 }),
                                )
                            });
                        let Some(GlobalVarShape::UntypedData(fused_layout)) =
                            &mut module.global_vars[fused_gv].shape
                        else {
                            unreachable!();
                        };
                        try_concat_mem_layouts(*fused_layout, layout).map(
                            |(offset_in_gv, new_fused_layout)| {
                                *fused_layout = new_fused_layout;
                                (fused_gv, offset_in_gv)
                            },
                        )
                    } else {
                        None
                    }
                    .unwrap_or_else(|| {
                        (module.global_vars.define(&self.cx, mk_gv_decl(layout)), 0)
                    });
                    self.global_var_to_alias_group.entry(gv).get_or_insert(None);

                    // HACK(eddyb) avoid losing diagnostics that were attached on
                    // the `mem.func_local_var` node.
                    module.global_vars[gv]
                        .attrs
                        .push_diags(&self.cx, local_def.attrs.diags(&self.cx).iter().cloned());

                    // FIXME(eddyb) cache this when possible.
                    let ptr_to_gv = self.cx.intern(ConstDef {
                        attrs: Default::default(),
                        ty: self.qptr_type(),
                        // FIXME(eddyb) merge all `Private` globals together.
                        kind: ConstKind::PtrToGlobalVar { global_var: gv, offset: None },
                    });

                    // HACK(eddyb) avoid removing the old node - while it should
                    // be a child of the function's body region, assuming that
                    // without checking it could lead to very subtle bugs.
                    (local_def.kind, local_def.inputs, local_def.outputs) = (
                        NodeKind::SpvInst(self.wk.OpNop.into(), spv::InstLowering::default()),
                        [].into_iter().collect(),
                        [].into_iter().collect(),
                    );
                    cursor.insert_node(
                        &self.cx,
                        NodeDef {
                            attrs: Default::default(),
                            kind: NodeKind::QPtr(QPtrOp::Offset(
                                i32::try_from(offset_in_gv).unwrap(),
                            )),
                            inputs: [Value::Const(ptr_to_gv)].into_iter().collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [local.qptr_output].into_iter().collect(),
                        },
                    );

                    let offset_in_gv =
                        NonZeroU32::new(offset_in_gv).map_or(Offset::Zero, |offset| {
                            Offset::Strided { stride: offset, index: IndexValue::One }
                        });

                    // HACK(eddyb) `summary.bases` fields of the
                    // summary get updated later on (along with any renumbering
                    // all `BaseChoice::Many`s may need, for escaping bases).
                    let summary = &mut summarized_ptrs[&local.qptr_output];
                    summary.def_law = QPtrDefLaw::Offset(offset_in_gv);
                    summary.offset_shape = offset_in_gv.map_index_value(|_| self.u32_type());

                    let escaping_offset_shape =
                        self.merge_offset_shapes(escaping_offset_shape, summary.offset_shape);

                    let global_base = GlobalBase::GlobalVar { ptr_to_binding: ptr_to_gv };
                    self.in_place_merge_offset_shapes(
                        base_maps.escaped.bases.entry(global_base).or_insert(Offset::Zero),
                        escaping_offset_shape,
                    );
                }
                // HACK(eddyb) remove any tombstones left around from the
                // loop above, as they could cause other problems later on.
                if let Some(last_func_local_var) = last_func_local_var {
                    let body = func_def_body.body;
                    let func = func_def_body.at_mut(());

                    let mut func_at_locals = func.at(body).at_children().into_iter();
                    func_at_locals.position.last = Some(last_func_local_var);
                    while let Some(mut func_at_node) = func_at_locals.next() {
                        let node = func_at_node.position;
                        if let NodeKind::SpvInst(spv_inst, _) = &func_at_node.reborrow().def().kind
                            && spv_inst.opcode == self.wk.OpNop
                        {
                            let func = func_at_node.at(());
                            func.regions[body].children.remove(node, func.nodes);
                        }
                    }
                }

                anything_ever_encodes_or_decodes_escaped_ptrs |=
                    ever_encodes_or_decodes_escaped_ptrs;

                base_maps.per_func.insert(func, func_base_map);
                per_func_scan_results.insert(
                    func,
                    FuncScanResults { summarized_ptrs, ever_encodes_or_decodes_escaped_ptrs },
                );
            }
        }

        if let Some(fused_private_gv) = fused_private_global_for_escaped_func_locals {
            use crate::mem::shapes::GlobalVarShape;

            let GlobalVarDecl {
                shape: Some(GlobalVarShape::UntypedData(fused_layout)),
                def: DeclDef::Present(GlobalVarDefBody { initializer }),
                ..
            } = &mut module.global_vars[fused_private_gv]
            else {
                unreachable!();
            };

            fused_layout.size =
                fused_layout.size.checked_next_multiple_of(fused_layout.align).unwrap();

            match initializer {
                Some(GlobalVarInit::Data(init)) => init.grow(fused_layout.size),
                None => {}
                _ => unreachable!(),
            }
        }

        // HACK(eddyb) make sure that, worst case, the `Null` base will
        // be usable to express the result of decoding arbitrary integers.
        if anything_ever_encodes_or_decodes_escaped_ptrs {
            let mut needs_null = base_maps.escaped.bases.is_empty();

            // HACK(eddyb) always include `Null` if supporting it is requested.
            if self.config.logical_ptr_null_is_zero {
                needs_null = true;
            }

            if needs_null {
                base_maps.escaped.bases.entry(GlobalBase::Null).or_insert(Offset::Zero);
            }
        }

        // FIXME(eddyb) take advantage of this sorting step to give large bases
        // more offset bits (see also Huffman coding mentions elsewhere).
        base_maps.escaped.bases.sort_by(|a, _, b, _| {
            let [a_sort_key, b_sort_key] = [a, b].map(|base| {
                // HACK(eddyb) always order `Null` first, effectively reserving
                // the `0` encoding for it.
                !matches!(base, GlobalBase::Null)
            });
            a_sort_key.cmp(&b_sort_key)
        });

        // NOTE(eddyb) there is still no interaction *between* escaped bases here,
        // e.g. if an escaped base already has a lower stride, that will not end
        // up propagating to other bases, as `write_back_escaped_ptr_offset_shape`
        // (and in fact all possible offsetting) behaves equivalently to mapping
        // offsets separately for each possible base, potentially shrinking each
        // base's stride to the common divisor needed to apply that offset.
        // FIXME(eddyb) figure out how much of the above comment is tied to the
        // choice of using byte offsets in the encoding itself, with the actual
        // index<->offset using multiply/divide (ideally shifts w/ pow2 stride)
        // as needed (crucially, decoding always ends up using a common divisor
        // of all strides, via the `any_escaped_offset_shape` computed below).
        if let Some((AnyEscapedBase, offset_shape)) = write_back_escaped_ptr_offset_shape {
            for escaped_offset_shape in base_maps.escaped.bases.values_mut() {
                *escaped_offset_shape =
                    self.merge_offset_shapes(*escaped_offset_shape, offset_shape);
            }
        }

        let mut any_escaped_offset_shape = Offset::Zero;
        for &escaped_offset_shape in base_maps.escaped.bases.values() {
            any_escaped_offset_shape =
                self.merge_offset_shapes(any_escaped_offset_shape, escaped_offset_shape);
        }

        // TODO(eddyb) in theory, everything in `base_maps.per_func` could be
        // attached as attributes on the function/loop body inputs/etc.

        // FIXME(eddyb) consider using a `Transformer` to process all consts, but
        // it would lack the context necessary to do any additional error-checking
        // (or the ability to).
        for gv in all_global_vars.clone() {
            let gv_decl = &mut module.global_vars[gv];
            let transform_const_use = |ct: Const, untyped_size: u32| -> Transformed<Const> {
                // HACK(eddyb) all errors will have been reported earlier.
                let maybe_global_base_and_offset =
                    if matches!(self.cx[self.cx[ct].ty].kind, TypeKind::QPtr)
                        && self.config.logical_ptr_size_align.0 == untyped_size
                    {
                        self.const_qptr_to_base_and_offset(ct)
                    } else {
                        None
                    };
                let encoded_ty = self.qptr_sized_uint_type();
                let kind = match maybe_global_base_and_offset {
                    // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
                    Some((GlobalBase::Undef, _)) => ConstKind::Undef,

                    Some((base, offset)) => {
                        let base_idx = base_maps.escaped.bases.get_index_of(&base).unwrap();
                        let offset = match offset {
                            Offset::Zero => 0,
                            Offset::Strided { stride, index } => match index {
                                // FIXME(eddyb) maybe avoid using `IndexValue` since it's
                                // always going to be `IndexValue::One` for const offsets.
                                IndexValue::One => stride.get(),

                                IndexValue::Dyn(_) | IndexValue::MinusOne => unreachable!(),
                            },
                        };

                        let encoded_scalar_ty = encoded_ty.as_scalar(&self.cx).unwrap();

                        // FIXME(eddyb) deduplicate the logic here, even if simple,
                        // maybe by creating a type for "encoding scheme".
                        let encoded_base_bit_width = base_maps.escaped.bits_needed_for_base_idx();
                        let encoded_offset_bit_width = encoded_scalar_ty
                            .bit_width()
                            .checked_sub(encoded_base_bit_width)
                            .unwrap_or_else(|| {
                                // FIXME(eddyb) how should this be handled?
                                // (it should be impossible for wide `qptr`s,
                                // but an e.g. 16-bit `qptr` could hit this)
                                unreachable!(
                                    "a choice of {} bases do not fit in {} bits",
                                    base_maps.escaped.bases.len(),
                                    encoded_scalar_ty.bit_width()
                                )
                            });
                        // FIXME(eddyb) introduce UB ("poison" consts?) when the
                        // offset is out of bounds, instead of having this panic.
                        let offset_mask = ((1u64 << encoded_offset_bit_width) - 1) as u32;
                        assert_eq!(offset & offset_mask, offset);

                        ConstKind::Scalar(scalar::Const::from_bits(
                            encoded_scalar_ty,
                            u128::try_from(base_idx)
                                .unwrap()
                                .checked_shl(encoded_offset_bit_width)
                                .unwrap()
                                .checked_add(offset.into())
                                .unwrap(),
                        ))
                    }
                    None => return Transformed::Unchanged,
                };

                Transformed::Changed(self.cx.intern(ConstDef {
                    attrs: Default::default(),
                    ty: encoded_ty,
                    kind,
                }))
            };
            if let DeclDef::Present(gv_def) = &mut gv_decl.def
                && let Some(init) = &mut gv_def.initializer
            {
                match init {
                    GlobalVarInit::Data(const_data) => {
                        use crate::mem::const_data::Part;

                        let mut next_offset = 0;
                        while next_offset < const_data.size() {
                            let part =
                                const_data.read(next_offset..const_data.size()).next().unwrap();

                            let offset = next_offset;
                            next_offset += part.size().get();

                            let encoded_part = match part {
                                Part::Uninit { .. } | Part::Bytes(_) => Transformed::Unchanged,
                                Part::Symbolic { size, maybe_partial_slice: _, value } => {
                                    transform_const_use(value, size.get())
                                }
                            };

                            if let Transformed::Changed(encoded) = encoded_part {
                                match self.cx[encoded].kind {
                                    // TODO(eddyb) `ConstData` needs a `write_undef`
                                    // method (maybe named "wipe"/"clear"/"erase"?).
                                    ConstKind::Undef => {
                                        const_data
                                            .write_scalar(
                                                offset,
                                                scalar::Const::from_bits(
                                                    self.qptr_sized_uint_type()
                                                        .as_scalar(&self.cx)
                                                        .unwrap(),
                                                    0,
                                                ),
                                                self.config,
                                            )
                                            .unwrap();
                                    }
                                    ConstKind::Scalar(scalar) => {
                                        let written_range = const_data
                                            .write_scalar(offset, scalar, self.config)
                                            .unwrap();

                                        // HACK(eddyb) only perfectly overwriting is allowed.
                                        assert_eq!(written_range, offset..next_offset);
                                    }
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }
                    GlobalVarInit::Direct(ct) => {
                        let untyped_size = match gv_decl.shape {
                            Some(crate::mem::shapes::GlobalVarShape::UntypedData(mem_layout)) => {
                                Some(mem_layout.size)
                            }
                            Some(
                                crate::mem::shapes::GlobalVarShape::Handles { .. }
                                | crate::mem::shapes::GlobalVarShape::TypedInterface(_),
                            )
                            | None => None,
                        };
                        if let Some(size) = untyped_size {
                            transform_const_use(*ct, size).apply_to(ct);
                        }
                    }
                    // HACK(eddyb) all errors will have been reported earlier.
                    GlobalVarInit::SpvAggregate { .. } => {}
                }
            }
        }

        for &func in &all_uses_from_module.funcs {
            // FIXME(eddyb) consider fusing locals from `func_base_map.bases`
            // into one big single local, to reduce the need for control-flow
            // (using `summarized_ptrs` would allow grouping them together for
            // more granularity).
            // NOTE(eddyb) this might make sense to almost always do, using
            // some kind of "capability narrowing" operation when a local
            // is used for anything other than in-bound direct accesss, this
            // could potentially also simplify "partition & propagate" logic,
            // and also makes sense for fusing *globals* as well!
            let DeclDef::Present(func_def_body) = &mut module.funcs[func].def else {
                continue;
            };

            let FuncScanResults { mut summarized_ptrs, ever_encodes_or_decodes_escaped_ptrs } =
                per_func_scan_results.remove(func).unwrap();

            let mut func_base_map = base_maps.per_func.remove(func).unwrap();

            let mut escaped_base_choices = None;

            if ever_encodes_or_decodes_escaped_ptrs {
                // As all functions have been scanned already, `base_maps.escaped`
                // should be complete, and can be added back to all bases that need it.
                let escaped_bases = base_maps.escaped.bases.keys().copied().map(Base::Global);

                // HACK(eddyb) escaping `Base::FuncLocal`s were rewritten above
                // to `Private` globals instead, but in order to avoid needing
                // a 1:1 remapping of of `Base::FuncLocal` -> `Base::Global`,
                // this now needs to be able to replace N locals with 1 global.
                let remap_base_if_needed = |base: Base| {
                    let func = func_def_body.at(());
                    if let Base::FuncLocal(local) = base {
                        let node_def =
                            &func.nodes[func.vars[local.qptr_output].def_parent.right().unwrap()];
                        match (&node_def.kind, &node_def.inputs[..]) {
                            (NodeKind::Mem(MemOp::FuncLocalVar(_)), _) => {}

                            (
                                NodeKind::QPtr(QPtrOp::Offset(_)),
                                &[Value::Const(ptr_to_binding)],
                            ) => return Base::Global(GlobalBase::GlobalVar { ptr_to_binding }),

                            _ => unreachable!(),
                        }
                    }
                    base
                };

                // HACK(eddyb) there might be more efficient ways to do this,
                // but the most straightforward one to ensure a simple mapping
                // for encoding escaping pointers, is to place the escaped bases
                // first, then remapping all pointer summaries' base index sets.
                let old_func_base_map = mem::take(&mut func_base_map);
                func_base_map.many_choice_bases.extend(escaped_bases.clone().chain(
                    old_func_base_map.many_choice_bases.iter().copied().map(remap_base_if_needed),
                ));
                let base_to_base_idx =
                    |base| func_base_map.many_choice_bases.get_index_of(&base).unwrap();

                for summary_idx in 0..summarized_ptrs.len() {
                    let summary = &summarized_ptrs[summary_idx];
                    let Some(bases) = summary.bases.as_ref().left() else {
                        continue;
                    };

                    // HACK(eddyb) this is pretty inefficient, but accuracy is
                    // needed later on to avoid several different failure modes.
                    let mut extra_offset_shape = None;
                    let remap_base_if_needed = |base| {
                        let remapped_base = remap_base_if_needed(base);
                        if let (Base::FuncLocal(local), Base::Global(_)) = (base, remapped_base) {
                            self.in_place_merge_offset_shapes(
                                extra_offset_shape.get_or_insert(Offset::Zero),
                                summarized_ptrs[&local.qptr_output].offset_shape,
                            );
                        }
                        remapped_base
                    };

                    let new_bases = bases
                        .iter(&old_func_base_map)
                        .map(remap_base_if_needed)
                        .map(BaseChoice::Single)
                        .reduce(|a, b| a.merge(b, base_to_base_idx))
                        .unwrap();

                    let summary = &mut summarized_ptrs[summary_idx];
                    *summary.bases.as_mut().left().unwrap() = new_bases;
                    if let Some(extra_offset_shape) = extra_offset_shape {
                        self.in_place_merge_offset_shapes(
                            &mut summary.offset_shape,
                            extra_offset_shape,
                        );
                    }
                }

                // Precompute a `BaseChoice` that already includes all escaped bases,
                // to cheaply combine with each summarized pointer's own `BaseChoice`.
                escaped_base_choices = escaped_bases
                    .map(BaseChoice::Single)
                    .reduce(|a, b| a.merge(b, base_to_base_idx));
            }

            let ptrs = summarized_ptrs.into_iter().map(|(ptr, summary)| {
                let PtrSummary { def_law, mut bases, mut offset_shape } = summary;

                // HACK(eddyb) completely filter out pointers that have any
                // unsupported bases, even when there's supported ones too,
                // to minimize error handling needs during legalization.
                let supported_bases = bases
                    .as_mut()
                    .right()
                    .and_then(|u| u.unsupported.take())
                    .map_or(Ok(()), Err)
                    .map(|()| {
                        let any_escaped = bases.as_ref().right().and_then(|u| u.any_escaped);
                        (bases.left(), any_escaped)
                    });

                if let Ok((_, Some(AnyEscapedBase))) = supported_bases {
                    offset_shape = self.merge_offset_shapes(offset_shape, any_escaped_offset_shape);
                }

                let get_bases_or_unsupported = supported_bases.map(|(known_bases, any_escaped)| {
                    // FIXME(eddyb) ideally this could avoid cloning.
                    let maybe_escaped =
                        any_escaped.map(|AnyEscapedBase| escaped_base_choices.clone().unwrap());
                    move |func_base_map: &FuncLocalBaseMap| {
                        match (known_bases, maybe_escaped) {
                            (Some(bases), Some(escaped)) => bases.merge(escaped, |base| {
                                func_base_map.many_choice_bases.get_index_of(&base).unwrap()
                            }),
                            (None, Some(bases)) | (Some(bases), None) => bases,

                            // HACK(eddyb) only `unsupported: Some(_)` can justify
                            // having no `bases.left()`, and that was handled earlier.
                            (None, None) => unreachable!(),
                        }
                    }
                });

                (ptr, def_law, get_bases_or_unsupported, offset_shape)
            });

            // FIXME(eddyb) move `&module.global_vars`, maybe even `escaped``,
            // into `self`.
            LegalizePtrsInFunc::new(
                self,
                &module.global_vars,
                &base_maps.escaped,
                (
                    escaped_base_choices
                        .clone()
                        .unwrap_or(BaseChoice::Single(Base::Global(GlobalBase::Null))),
                    any_escaped_offset_shape,
                ),
                func_base_map,
                func_def_body,
                ptrs,
            )
            .in_place_transform_func_decl(&mut module.funcs[func]);
        }
    }

    // FIXME(eddyb) are all of these caches necessary? (make interning faster?)
    fn u32_type(&self) -> Type {
        if let Some(cached) = self.cached_u32_type.get() {
            return cached;
        }
        let ty = self.cx.intern(scalar::Type::U32);
        self.cached_u32_type.set(Some(ty));
        ty
    }

    /// Get the (likely cached) `QPtr` type.
    fn qptr_type(&self) -> Type {
        if let Some(cached) = self.cached_qptr_type.get() {
            return cached;
        }
        let ty = self.cx.intern(TypeKind::QPtr);
        self.cached_qptr_type.set(Some(ty));
        ty
    }

    fn undef_qptr_const(&self) -> Const {
        if let Some(cached) = self.cached_undef_qptr_const.get() {
            return cached;
        }
        let ct = self.cx.intern(ConstDef {
            attrs: Default::default(),
            ty: self.qptr_type(),
            kind: ConstKind::Undef,
        });
        self.cached_undef_qptr_const.set(Some(ct));
        ct
    }

    fn null_qptr_const(&self) -> Const {
        if let Some(cached) = self.cached_null_qptr_const.get() {
            return cached;
        }
        let ct = self.cx.intern(ConstDef {
            attrs: Default::default(),
            ty: self.qptr_type(),
            // FIXME(eddyb) maybe `qptr` should
            // have its own null constant?
            kind: ConstKind::SpvInst {
                spv_inst_and_const_inputs: Rc::new((
                    self.wk.OpConstantNull.into(),
                    [].into_iter().collect(),
                )),
            },
        });
        self.cached_null_qptr_const.set(Some(ct));
        ct
    }

    // FIXME(eddyb) give easier access to the `scalar::Type` within, too.
    fn qptr_sized_uint_type(&self) -> Type {
        if let Some(cached) = self.cached_qptr_sized_uint_type.get() {
            return cached;
        }

        let (qptr_size, _qptr_align) = self.config.logical_ptr_size_align;
        let qptr_width =
            qptr_size.checked_mul(8).and_then(scalar::IntWidth::try_from_bits).unwrap_or_else(
                || unreachable!("qptr cannot be {qptr_size} bytes: not a valid integer size"),
            );

        let ty = self.cx.intern(scalar::Type::UInt(qptr_width));
        self.cached_qptr_sized_uint_type.set(Some(ty));
        ty
    }

    /// Returns the [`AliasGroup`] `base` belongs to, *if* at all aliasable,
    /// or `None` otherwise.
    ///
    /// The [`AliasGroup`] is a trivial (`O(1)`) predictor for `may_alias`,
    /// i.e. `may_alias(a, b)` (as understood by this analysis, see heading below)
    /// can be determined from only `base_alias_group(a)` and `base_alias_group(b)`:
    /// - both must be `Some(_)`, as `None` indicates impossibility of aliasing
    /// - both must be equal, as different [`AliasGroup`]s don't overlap
    ///
    /// This approach scales up to sets of [`Base`]s, allowing `O(N)` partitioning
    /// of such a set into one set per possible [`AliasGroup`] choice, plus the
    /// remaining [`Base`]s that lack the ability to be aliased (or rather, it
    /// would be **Undefined Behavior** to interact with the memory they reference
    /// without going through them - see also the `may_alias` description below).
    ///
    /// ## `may_alias` relation
    ///
    /// As used earlier above, `may_alias(a, b)` is a (hypothetical) symmetrical
    /// relation between [`Base`]s (and/or between the memory regions they reference),
    /// which conseratively approximates (i.e. can/will have false positives)
    /// whether there can ever exist any overlap (even as small as a single byte)
    /// between the memory regions referenced through `a` and `b`, even when
    /// `a != b`, "syntactically" (via `Base: Eq`), e.g. different `GlobalVar`s.
    ///
    /// For example, if `a` and `b` are both [`GlobalBase::BufferData`], their
    /// annotations (or lack thereof), could imply it's **Undefined Behavior**
    /// for the host side to bind the same memory region to both of them.
    ///
    /// **Note**: the only reason `may_alias` isn't a boolean predicate of its
    /// own in the code, is that unrestricted aliasing in rare in practice, and
    /// it helps to be able to use `AliasGroup` to partition e.g. sets of `Base`s,
    /// removing any need for quadratic queries (as within an `AliasGroup` it's
    /// effectively impossible to disprove aliasing through more static analysis,
    /// while between `AliasGroup`s aliasing is definitely impossible).
    fn base_alias_group(&self, base: Base) -> Option<AliasGroup> {
        match base {
            Base::Global(GlobalBase::Undef | GlobalBase::Null) | Base::FuncLocal(_) => None,

            Base::Global(
                GlobalBase::GlobalVar { ptr_to_binding: ptr }
                | GlobalBase::BufferData { ptr_to_buffer: ptr },
            ) => {
                let ConstKind::PtrToGlobalVar { global_var, offset: None } = self.cx[ptr].kind
                else {
                    unreachable!()
                };
                self.global_var_to_alias_group[global_var]
            }
        }
    }

    // FIXME(eddyb) document.
    fn const_qptr_to_base_and_offset(&self, ct: Const) -> Option<(GlobalBase, Offset<IndexValue>)> {
        // FIXME(eddyb) implement more constant address pointers.
        match &self.cx[ct].kind {
            ConstKind::Undef => Some((GlobalBase::Undef, Offset::Zero)),
            ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                (spv_inst.opcode == self.wk.OpConstantNull && const_inputs.is_empty())
                    .then_some((GlobalBase::Null, Offset::Zero))
            }
            ConstKind::PtrToGlobalVar { global_var: _, offset: None } => {
                Some((GlobalBase::GlobalVar { ptr_to_binding: ct }, Offset::Zero))
            }
            // HACK(eddyb) "fits in `i32`" check helps with anything using `u32::MAX`
            // to indicate erroring pointers that shouldn't be accidentally ignored.
            // FIXME(eddyb) need better ways to systematically
            // prevent diagnostics from being accidentally ignored.
            &ConstKind::PtrToGlobalVar { global_var, offset: Some(offset) }
                if i32::try_from(offset.get()).is_ok() =>
            {
                Some((
                    GlobalBase::GlobalVar {
                        ptr_to_binding: self.cx.intern(ConstDef {
                            attrs: Default::default(),
                            ty: self.qptr_type(),
                            kind: ConstKind::PtrToGlobalVar { global_var, offset: None },
                        }),
                    },
                    Offset::Strided { stride: offset, index: IndexValue::One },
                ))
            }
            _ => None,
        }
    }

    /// Determine the `DefLaw` (see also its docs) for a value's definition,
    /// returning `None` for unrelated definitions (i.e. not involving pointers),
    /// and invoking `qptr_def_law()` when the value is `qptr`-typed.
    //
    // FIXME(eddyb) `QPtrDefLaw::Unsupported` should probably be `Err`?
    // FIXME(eddyb) try refactoring this further, now that recomputation is rarer.
    fn maybe_def_law(
        &self,
        func_at_v: FuncAt<'_, Value>,
        qptr_def_law: impl FnOnce() -> QPtrDefLaw,
    ) -> Option<DefLaw> {
        let func = func_at_v.at(());

        let is_qptr =
            |v: Value| matches!(self.cx[func.at(v).type_of(&self.cx)].kind, TypeKind::QPtr);

        let ty_kind = &self.cx[func_at_v.type_of(&self.cx)].kind;

        match (ty_kind, func_at_v.position) {
            (TypeKind::QPtr, _) => return Some(DefLaw::QPtr(qptr_def_law())),
            (&TypeKind::Scalar(ty), Value::Var(v)) => {
                if let VarKind::NodeOutput { node, output_idx: 0 } = func.at(v).decl().kind() {
                    let node_def = func.at(node).def();
                    match &node_def.kind {
                        NodeKind::SpvInst(spv_inst, lowering)
                            if [self.wk.OpPtrEqual, self.wk.OpPtrNotEqual]
                                .contains(&spv_inst.opcode)
                                && lowering.disaggregated_output.is_none()
                                && lowering.disaggregated_inputs.is_empty()
                                && node_def.inputs.len() == 2
                                && is_qptr(node_def.inputs[0])
                                && is_qptr(node_def.inputs[1])
                                && ty == scalar::Type::Bool =>
                        {
                            return Some(DefLaw::CmpQPtrEqOrNe {
                                not_eq: spv_inst.opcode == self.wk.OpPtrNotEqual,
                            });
                        }

                        NodeKind::SpvInst(spv_inst, lowering)
                            if [self.wk.OpConvertPtrToU, self.wk.OpBitcast]
                                .contains(&spv_inst.opcode)
                                && lowering.disaggregated_output.is_none()
                                && lowering.disaggregated_inputs.is_empty()
                                && node_def.inputs.len() == 1
                                && is_qptr(node_def.inputs[0])
                                && ty
                                    == self.qptr_sized_uint_type().as_scalar(&self.cx).unwrap() =>
                        {
                            return Some(DefLaw::EncodeEscapingQPtr);
                        }

                        _ => {}
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Determine the `QPtrExpansions` (see also its docs) required by a node.
    fn node_qptr_expansions(&self, func_at_node: FuncAt<'_, Node>) -> QPtrExpansions {
        let func = func_at_node.at(());
        let node_def = func_at_node.def();

        let is_qptr =
            |v: Value| matches!(self.cx[func.at(v).type_of(&self.cx)].kind, TypeKind::QPtr);

        let mut expand = QPtrExpansions {
            pre_encode_escaping_input: None,
            post_decode_escaped_output: None,

            // Default for most instructions, overriden below where necessary.
            instantiate_by_multi_base_qptr_inputs: true,
        };
        match &node_def.kind {
            // Never instantiate nested regions (`Select` cases/`Loop` body),
            // which keeps IR duplication down - also, legalization is already
            // handled def-side via `QPtrDefLaw::Dyn` (i.e. for `Loop` inputs).
            _ if !node_def.child_regions.is_empty() => {
                expand.instantiate_by_multi_base_qptr_inputs = false;
            }

            NodeKind::Select(_) | NodeKind::Loop { .. } => unreachable!(),

            // Legalization already handled def-side (e.g. via `QPtrDefLaw::Offset`).
            NodeKind::QPtr(QPtrOp::Offset(_) | QPtrOp::DynOffset { .. })
            | NodeKind::FuncCall(_)
            | NodeKind::ThunkBind(_) => {
                expand.instantiate_by_multi_base_qptr_inputs = false;
            }

            NodeKind::Mem(MemOp::Load { .. }) => {
                if is_qptr(Value::Var(node_def.outputs[0])) {
                    expand.post_decode_escaped_output = Some(0);
                }
            }

            NodeKind::Mem(MemOp::Store { .. }) => {
                if is_qptr(node_def.inputs[1]) {
                    expand.pre_encode_escaping_input = Some(1);
                }
            }

            NodeKind::SpvInst(spv_inst, lowering)
                if [self.wk.OpConvertUToPtr, self.wk.OpConvertPtrToU, self.wk.OpBitcast]
                    .contains(&spv_inst.opcode)
                    && lowering.disaggregated_output.is_none()
                    && lowering.disaggregated_inputs.is_empty()
                    && node_def.inputs.len() == 1 =>
            {
                let in_ty = func.at(node_def.inputs[0]).type_of(&self.cx);
                let out_ty = func.at(node_def.outputs[0]).decl().ty;

                let is_exact_cast = |int_ty: scalar::Type| {
                    int_ty == self.qptr_sized_uint_type().as_scalar(&self.cx).unwrap()
                };

                match (&self.cx[in_ty].kind, &self.cx[out_ty].kind) {
                    // Legalization in the `is_exact_cast` case already
                    // handled def-side via `QPtrDefLaw::DecodeEscaped`.
                    (&TypeKind::Scalar(in_ty), TypeKind::QPtr) if !is_exact_cast(in_ty) => {
                        expand.post_decode_escaped_output = Some(0);
                    }

                    (TypeKind::QPtr, &TypeKind::Scalar(out_ty)) => {
                        if is_exact_cast(out_ty) {
                            // Legalization in the `is_exact_cast` case already
                            // handled def-side via `DefLaw::EncodeEscapingQPtr`.
                            expand.instantiate_by_multi_base_qptr_inputs = false;
                        } else {
                            expand.pre_encode_escaping_input = Some(0);
                        }
                    }

                    _ => {}
                }
            }

            NodeKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(_))
            | NodeKind::Scalar(_)
            | NodeKind::Vector(_)
            | NodeKind::Mem(MemOp::FuncLocalVar(_) | MemOp::Copy { .. })
            | NodeKind::QPtr(
                QPtrOp::HandleArrayIndex | QPtrOp::BufferData | QPtrOp::BufferDynLen { .. },
            )
            | NodeKind::SpvInst(..)
            | NodeKind::SpvExtInst { .. } => {}
        }

        // HACK(eddyb) this avoids replicating the decision logic.
        if let [output_var] = node_def.outputs[..]
            && let Some(DefLaw::CmpQPtrEqOrNe { .. }) =
                self.maybe_def_law(func.at(Value::Var(output_var)), || QPtrDefLaw::Unsupported)
        {
            // Legalization already handled def-side via `DefLaw::CmpQPtrEqOrNe`.
            expand.instantiate_by_multi_base_qptr_inputs = false;
        }

        expand
    }

    // HACK(eddyb) computes a kind of "integer promotion" merge of two integer
    // types (which most of the time would match, but this is "just in case"),
    // with `Scaled` allowing this to be used for `Offset<Type>` merging.
    #[must_use]
    fn merge_scaled_index_types(
        &self,
        Scaled { value: a_ty, multiplier: a_mul }: Scaled<Type>,
        Scaled { value: b_ty, multiplier: b_mul }: Scaled<Type>,
    ) -> Type {
        if a_ty == b_ty && a_mul.get() == 1 && b_mul.get() == 1 {
            // HACK(eddyb) even the fast path needs to check for integer types.
            if a_ty
                .as_scalar(&self.cx)
                .is_some_and(|ty| matches!(ty, scalar::Type::SInt(_) | scalar::Type::UInt(_)))
            {
                return a_ty;
            }
        }

        let ty_as_int_width_and_signedness = |ty: Type| {
            ty.as_scalar(&self.cx)
                .and_then(|scalar_ty| match scalar_ty {
                    scalar::Type::SInt(int_width) => Some((int_width, true)),
                    scalar::Type::UInt(int_width) => Some((int_width, false)),
                    scalar::Type::Bool | scalar::Type::Float(_) => None,
                })
                .ok_or_else(|| {
                    Diag::bug(["non-integer index type `".into(), ty.into(), "`".into()])
                })
        };

        ty_as_int_width_and_signedness(a_ty)
            .and_then(|(a_width, a_signed)| {
                let (b_width, b_signed) = ty_as_int_width_and_signedness(b_ty)?;

                // HACK(eddyb) this is UNSOUND in the general case, see also the
                // "UNSOUNDNESS HAZARD" warning on `QPtrDefLaw::DecodeEscaped`'s
                // implementation in `legalize_vars_defined_by`, but it's easier
                // to rely on it for now.
                // TODO(eddyb) actually track whether indices are in-bounds.
                let is_in_bounds = true;

                // FIXME(eddyb) support more combinations.
                let width = a_width.max(b_width);
                let signed = if a_signed == b_signed {
                    a_signed
                } else if is_in_bounds {
                    // HACK(eddyb) an in-bounds index will always be positive,
                    // and signed integers are either positive, or `-x` for a
                    // subtraction of `x` from a larger positive starting index.
                    false
                } else {
                    return Err(Diag::bug([
                        "incompatible index types `".into(),
                        a_ty.into(),
                        "` and `".into(),
                        b_ty.into(),
                        "`".into(),
                    ]));
                };

                let merged_scalar_ty =
                    if signed { scalar::Type::SInt(width) } else { scalar::Type::UInt(width) };
                let merged_ty: Type = self.cx.intern(merged_scalar_ty);

                // FIXME(eddyb) widen the type to fit the multiplier, instead.
                let max_mul = a_mul.max(b_mul).get().into();
                scalar::Const::int_try_from_i128(merged_scalar_ty, max_mul).ok_or_else(|| {
                    Diag::bug([
                        "index type `".into(),
                        merged_ty.into(),
                        format!("` can't represent `{max_mul}`").into(),
                    ])
                })?;

                Ok(merged_ty)
            })
            .unwrap_or_else(|diag| {
                let mut attrs = AttrSet::default();
                attrs.push_diag(&self.cx, diag);
                self.cx.intern(TypeDef { attrs, kind: TypeKind::Scalar(scalar::Type::U32) })
            })
    }

    #[must_use]
    fn merge_offset_shapes(&self, a: Offset<Type>, b: Offset<Type>) -> Offset<Type> {
        a.merge(
            b,
            |index_ty| index_ty,
            |index_ty| index_ty,
            |a, b| self.merge_scaled_index_types(a, b),
        )
    }

    fn in_place_merge_offset_shapes(&self, a: &mut Offset<Type>, b: Offset<Type>) {
        *a = self.merge_offset_shapes(*a, b);
    }
}

// FIXME(eddyb) move this into some common utilities (or even obsolete its need).
#[derive(Default)]
struct LoopMap {
    loop_body_region_to_loop_node: EntityOrientedDenseMap<Region, Node>,
}

impl LoopMap {
    fn new(func_def_body: &FuncDefBody) -> Self {
        let mut loop_map = Self::default();

        // FIXME(eddyb) adopt this style of queue-based visiting in more places.
        let mut queue = VecDeque::new();
        queue.push_back(func_def_body.body);
        while let Some(region) = queue.pop_front() {
            for func_at_node in func_def_body.at(region).at_children() {
                let node_def = func_at_node.def();
                if let NodeKind::Loop { .. } = node_def.kind {
                    loop_map
                        .loop_body_region_to_loop_node
                        .insert(node_def.child_regions[0], func_at_node.position);
                }
                queue.extend(node_def.child_regions.iter().copied());
            }
        }

        loop_map
    }
}

// FIXME(eddyb) this type doesn't do very much by itself.
#[derive(Default)]
struct BaseMaps {
    per_func: EntityOrientedDenseMap<Func, FuncLocalBaseMap>,
    escaped: EscapedBaseMap,
}

// FIXME(eddyb) in theory, something like union-find (or a much more complicated
// directed graph of "set inclusion" style relations) could be used to track
// disjoint "groups" of escaped bases, based on where they were written etc.
#[derive(Default)]
struct EscapedBaseMap {
    /// All bases of pointers that were ever written to memory, with their
    /// corresponding "offset shape" (`Offset<Type>`) *of the written pointers*.
    ///
    /// The reason only writes are relevant is that encoding escaped pointers
    /// (into integers) only requires knowing the values that will be encoded,
    /// *not* how they might be used *after* being decoded into the per-function
    /// set of bases and appropriate common-strided offset - which may require
    /// multiplication (i.e. complicating decoding to allow better encodings).
    //
    // FIXME(eddyb) optimize this definition, taking into account disjoint
    // uses of memory, pointers passed between functions, etc.
    //
    // TODO(eddyb) update the above comment, as the new approach is to actually
    // use byte offsets, "simply" multiplying/dividing before/after encode/decode
    // (which hopefully turns into just left/right shifting), and the recorded
    // "offset shape" (`Offset<Type`) for each base is more of an indication of
    // what is *expected* (e.g. that the index *will happen to be* a multiple
    // of the stride, dynamically) - for now, that information isn't really used.
    // TODO(eddyb) enforce a pow2 common stride, guaranteeing only shifts.
    //
    // FIXME(eddyb) make this somewhat hierarchical, more like huffman/arithmetic
    // coding, so that some bases have more bits left for the offset, comparable
    // to reserving 2**N consecutive base indices for one base, though decoding
    // that away may require using some bits of the the base index to indicate
    // how much to shift/mask (while remaining compatible with UB-less undef).
    bases: FxIndexMap<GlobalBase, Offset<Type>>,
}

impl EscapedBaseMap {
    // HACK(eddyb) this is the only common place where this information is found.
    fn bits_needed_for_base_idx(&self) -> u32 {
        self.bases.len().checked_next_power_of_two().unwrap().checked_ilog2().unwrap()
    }
}

#[derive(Default)]
struct FuncLocalBaseMap {
    /// All possible bases that `BaseChoice::Many` can refer to, in this function.
    many_choice_bases: FxIndexSet<Base>,
}

/// Non-empty set of `Base`s, carrying all dynamic information necessary to pick
/// one specific base when multiple are possible (i.e. `BaseChoice::Many`), and
/// also generic over the value type `V`, so that e.g. `BaseChoice<()>` can be
/// used to e.g. abstractly track which bases a pointer *may* be using.
//
// FIXME(eddyb) support `HandleArrayIndex` as well, somehow.
#[derive(Clone, PartialEq, Eq)]
enum BaseChoice<V> {
    // FIXME(eddyb) consider using a `V`-typed field here, instead of `CanonPtr`'s
    // `single_base_buffer_data` (but could also help w/ e.g. `GlobalBase::Null`).
    Single(Base),

    Many {
        /// Ad-hoc bitset, with each bit index corresponding to the [`Base`] at
        /// the same index in [`FuncLocalBaseMap`]'s `many_choice_bases` set.
        //
        // FIXME(eddyb) this may be a performance hazard above 64 bases.
        // FIXME(eddyb) consider `Rc` for the non-small case.
        base_index_bitset: SmallVec<[u64; 1]>,

        /// Dynamic integer value (of type `u32`) choosing a base index, from
        /// any in `base_index_bitset` (other values cause undefined behavior).
        base_index_selector: V,
    },
}

impl<V: Copy> BaseChoice<V> {
    fn map_selector_value<V2>(self, f: impl FnOnce(V) -> V2) -> BaseChoice<V2> {
        match self {
            BaseChoice::Single(base) => BaseChoice::Single(base),
            BaseChoice::Many { base_index_bitset, base_index_selector } => {
                BaseChoice::Many { base_index_bitset, base_index_selector: f(base_index_selector) }
            }
        }
    }

    fn count(&self) -> usize {
        match self {
            BaseChoice::Single(_) => 1,
            BaseChoice::Many { base_index_bitset, .. } => {
                base_index_bitset.iter().map(|chunk| chunk.count_ones() as usize).sum()
            }
        }
    }

    // HACK(eddyb) the outer `Option` and per-base `Option` are either both
    // `None` (`BaseChoice::Single`) or both `Some` (`BaseChoice::Many`).
    fn iter_with_selector_value<'a>(
        &'a self,
        func_base_map: &'a FuncLocalBaseMap,
    ) -> (Option<V>, impl Iterator<Item = (Option<u32>, Base)> + 'a) {
        match self {
            &BaseChoice::Single(base) => (None, Either::Left([(None, base)].into_iter())),
            BaseChoice::Many { base_index_bitset, base_index_selector } => (
                Some(*base_index_selector),
                Either::Right(
                    // FIXME(eddyb) move this into a `BaseChoice` method
                    // (or really it should be using a `SmallBitSet`).
                    base_index_bitset
                        .iter()
                        .enumerate()
                        .flat_map(|(chunk_idx, &chunk)| {
                            let mut i = chunk_idx * 64;
                            let mut chunk = chunk;
                            iter::from_fn(move || {
                                let skip = NonZeroU64::new(chunk)?.trailing_zeros() as usize;
                                i += skip;
                                chunk >>= skip;
                                let r = Some(i);

                                // HACK(eddyb) advancing past the `1` bit is
                                // done separately to avoid `chunk >>= 64`.
                                i += 1;
                                chunk >>= 1;

                                r
                            })
                        })
                        .map(|base_idx| {
                            (
                                Some(u32::try_from(base_idx).unwrap()),
                                func_base_map.many_choice_bases[base_idx],
                            )
                        }),
                ),
            ),
        }
    }
}

// FIXME(eddyb) generalize this.
impl BaseChoice<()> {
    fn iter<'a>(&'a self, func_base_map: &'a FuncLocalBaseMap) -> impl Iterator<Item = Base> + 'a {
        self.iter_with_selector_value(func_base_map).1.map(|(_, base)| base)
    }

    fn insert(&mut self, base: Base, mut base_to_base_idx: impl FnMut(Base) -> usize) {
        // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
        if let Base::Global(GlobalBase::Undef) = base {
            return;
        }

        let prev_base = match self {
            &mut BaseChoice::Single(prev_base) => {
                // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
                if let Base::Global(GlobalBase::Undef) = prev_base {
                    *self = BaseChoice::Single(base);
                    return;
                }

                if prev_base == base {
                    return;
                }
                *self = BaseChoice::Many {
                    base_index_bitset: SmallVec::new(),
                    base_index_selector: (),
                };
                Some(prev_base)
            }
            BaseChoice::Many { .. } => None,
        };
        let BaseChoice::Many { base_index_bitset: chunks, base_index_selector: () } = self else {
            unreachable!();
        };
        for base in prev_base.into_iter().chain([base]) {
            let i = base_to_base_idx(base);
            let (chunk_idx, bit_mask) = (i / 64, 1 << (i % 64));
            if chunk_idx >= chunks.len() {
                chunks.resize(chunk_idx + 1, 0);
            }
            chunks[chunk_idx] |= bit_mask;
        }
    }

    #[must_use]
    fn merge(self, other: Self, base_to_base_idx: impl FnMut(Base) -> usize) -> Self {
        match (self, other) {
            (BaseChoice::Single(base), set) | (set, BaseChoice::Single(base)) => {
                let mut merged = set;
                merged.insert(base, base_to_base_idx);
                merged
            }

            (
                BaseChoice::Many { base_index_bitset: a, base_index_selector: () },
                BaseChoice::Many { base_index_bitset: b, base_index_selector: () },
            ) => {
                let (mut dst, src) = if a.len() > b.len() { (a, b) } else { (b, a) };
                for (dst, src) in dst.iter_mut().zip(src) {
                    *dst |= src;
                }
                BaseChoice::Many { base_index_bitset: dst, base_index_selector: () }
            }
        }
    }
}

// HACK(eddyb) helper for `Offset::merge`.
#[derive(Copy, Clone)]
struct Scaled<V> {
    value: V,
    multiplier: NonZeroU32,
}

impl<V> Scaled<V> {
    fn map_value<V2>(self, f: impl FnOnce(V) -> V2) -> Scaled<V2> {
        let Scaled { value, multiplier } = self;
        Scaled { value: f(value), multiplier }
    }
}

impl<V> Offset<V> {
    fn map_index_value<V2>(self, f: impl FnOnce(V) -> V2) -> Offset<V2> {
        match self {
            Offset::Zero => Offset::Zero,
            Offset::Strided { stride, index } => Offset::Strided { stride, index: f(index) },
        }
    }

    /// Merge `Offset`s, resolving stride conflicts between two `Offset::Strided`s
    /// (with strides `a` vs `b`) by computing a "common stride" `c` such that
    /// `a` and `b` are multiples of `c`, which could be satisfied by the GCD
    /// (greatest common divisor) of `a` and `b`, but the approach taken here
    /// is to use the greatest common *power of two* divisor instead, which is
    /// both cheaper to compute, and more likely to end up being (related to)
    /// the smallest unit of access *anyway*.
    #[must_use]
    fn merge<V2, V3>(
        self,
        other: Offset<V2>,
        map_self: impl FnOnce(V) -> V3,
        map_other: impl FnOnce(V2) -> V3,
        merge_both: impl FnOnce(Scaled<V>, Scaled<V2>) -> V3,
    ) -> Offset<V3> {
        match (self, other) {
            (Offset::Zero, Offset::Zero) => Offset::Zero,
            (Offset::Strided { stride, index }, Offset::Zero) => {
                Offset::Strided { stride, index: map_self(index) }
            }
            (Offset::Zero, Offset::Strided { stride, index }) => {
                Offset::Strided { stride, index: map_other(index) }
            }
            (
                Offset::Strided { stride: a_stride, index: a_index },
                Offset::Strided { stride: b_stride, index: b_index },
            ) => {
                let stride = if a_stride == b_stride {
                    a_stride
                } else {
                    NonZeroU32::new(1 << a_stride.trailing_zeros().min(b_stride.trailing_zeros()))
                        .unwrap()
                };
                let multiplier_from = |orig_stride: NonZeroU32| {
                    assert_eq!(orig_stride.get() % stride.get(), 0);
                    NonZeroU32::new(orig_stride.get() / stride.get()).unwrap()
                };
                Offset::Strided {
                    stride,
                    index: merge_both(
                        Scaled { value: a_index, multiplier: multiplier_from(a_stride) },
                        Scaled { value: b_index, multiplier: multiplier_from(b_stride) },
                    ),
                }
            }
        }
    }
}

struct ScanBasesInFunc<'a> {
    legalizer: &'a LegalizePtrs<'a>,
    loop_map: &'a LoopMap,

    escaped_base_map: &'a mut EscapedBaseMap,
    func_base_map: FuncLocalBaseMap,

    // HACK(eddyb) tracks whether escaping pointers are ever interacted with,
    // as both directions require e.g. adjusting `FuncLocalBaseMap` later.
    ever_encodes_or_decodes_escaped_ptrs: bool,

    /// If any escaping pointer writes (e.g. `mem.store` with a pointer value)
    /// write pointers that are themselves based on previously escaped pointers
    /// read from memory (i.e. `AnyEscapedBase`), their combined effect is
    /// tracked here.
    write_back_escaped_ptr_offset_shape: Option<(AnyEscapedBase, Offset<Type>)>,

    // HACK(eddyb) `Base::FuncLocal`s could not be added to `escaped_base_map`
    // themselves (not to mention cross-function accesses may be entirely valid),
    // so instead they are collected separately, to become `Private` globals.
    escaping_func_locals: FxIndexMap<FuncLocal, Offset<Type>>,

    // FIXME(eddyb) how expensive is this cache? (esp. wrt `BaseChoice`)
    // HACK(eddyb) this technically replicates part of `cyclotron::bruteforce`,
    // but without the fixpoint saturation (going for a more any-way traversal).
    // FIXME(eddyb) replace this with a variant of `EntityOrientedDenseMap`,
    // with the `indexmap`-style insertion order (to allow sound iteration).
    summarized_ptrs: FxIndexMap<Var, PtrSummary>,
}

// HACK(eddyb) marker type used to indicate that a pointer's base could be *any*
// of `EscapedBaseMap`'s *final* `bases` (after all functions have been scanned).
// FIXME(eddyb) in theory, something like union-find (or a much more complicated
// directed graph of "set inclusion" style relations) could be used to track
// disjoint "groups" of escaped bases, based on where they were written etc.
#[derive(Copy, Clone)]
struct AnyEscapedBase;

#[derive(Clone)]
struct PtrSummary {
    def_law: QPtrDefLaw,

    bases: EitherOrBoth<BaseChoice<()>, UnknownBases>,
    offset_shape: Offset<Type>,
}

impl PtrSummary {
    fn unsupported(ptr: Value) -> Self {
        PtrSummary {
            def_law: QPtrDefLaw::Unsupported,
            bases: EitherOrBoth::Right(UnknownBases {
                any_escaped: None,
                loop_body_input_deps: [].into_iter().collect(),
                unsupported: Some(UnsupportedBases { directly_used: [ptr].into_iter().collect() }),
            }),
            offset_shape: Offset::Zero,
        }
    }
}

/// Summary of pointer bases that couldn't be resolved to a `Base` right away.
//
// FIXME(eddyb) this has the kind of problem `EitherOrBoth` solves, except for
// a set of 3 fields, which shouldn't be able to be `(None, [], None)`.
#[derive(Clone)]
struct UnknownBases {
    any_escaped: Option<AnyEscapedBase>,

    /// Only used while summarizing a loop body input, to detect cyclic recursion,
    /// and to aid in tracking all the potential pointers flowing into the loop
    /// (i.e. this needs to be accurate, not just signaling an error).
    ///
    /// To prevent excessive cloning, `PtrSummary`s are only cached when
    /// `loop_body_input_deps` contains at most a single `loop_body_input_var`
    /// entry, which `summarize_ptr` can use to determine a retry is necessary,
    /// based on `summarized_ptrs[&loop_body_input_var].bases` losing its own
    /// `loop_body_input_deps` (indicating it's completed).
    //
    // FIXME(eddyb) this really wants to be `FxIndexSet` beyond a certain size.
    loop_body_input_deps: SmallVec<[Var; 1]>,

    /// Unknown/unsupported pointer values encountered, for which errors
    /// need to be later applied (as summarizing cannot mutate the function).
    unsupported: Option<UnsupportedBases>,
}

// HACK(eddyb) only separate for the field/documentation.
#[derive(Clone)]
struct UnsupportedBases {
    /// To avoid excessive errors, `directly_used` values are preserved only for
    /// the erroring pointer (i.e. `summarize_ptr(ptr)` with `directly_used == [ptr]`),
    /// and its direct users, but the latter only keep `directly_used` intact in
    /// `summarized_ptrs`, `summarize_ptr` clearing it before returning to further users.
    directly_used: SmallVec<[Value; 1]>,
}

impl ScanBasesInFunc<'_> {
    // HACK(eddyb) mutable access to the function is only for attaching `Diag`s
    // and the (artificial) `scan_ptr` vs `summarize_ptr` distinction.
    fn scan_func(&mut self, func_def_body: &mut FuncDefBody) {
        let cx = &self.legalizer.cx;

        let is_qptr = |ty: Type| matches!(cx[ty].kind, TypeKind::QPtr);

        // HACK(eddyb) despite the usage of "encode/decode" terminology,
        // function calls do not *actually* encode/decode pointers, they only
        // use the same `AnyEscapedBase` tracking (for now).
        {
            let body = func_def_body.body;
            let mut func = func_def_body.at_mut(());

            let num_body_outputs = func.reborrow().at(body).def().outputs.len();
            for body_output_idx in 0..num_body_outputs {
                let body_output = func.reborrow().at(body).def().outputs[body_output_idx];
                if is_qptr(func.reborrow().freeze().at(body_output).type_of(cx)) {
                    self.scan_escaping_ptr_for_encode(func.reborrow().at(body_output));
                }
            }
        }

        {
            // FIXME(eddyb) adopt this style of queue-based visiting in more places.
            // TODO(eddyb) now that everything is more random-access, strongly
            // consider using this technique in `LegalizePtrsInFunc` too.
            let mut queue = VecDeque::new();
            queue.push_back(func_def_body.body);
            while let Some(region) = queue.pop_front() {
                let mut func = func_def_body.at_mut(());

                let num_inputs = func.reborrow().at(region).def().inputs.len();
                for input_idx in 0..num_inputs {
                    let input = func.reborrow().at(region).def().inputs[input_idx];
                    if is_qptr(func.vars[input].ty) {
                        self.scan_ptr(func.reborrow().at(Value::Var(input)));
                    }
                }

                let mut func_at_children = func_def_body.at_mut(region).at_children().into_iter();
                while let Some(mut func_at_node) = func_at_children.next() {
                    {
                        let node_def = func_at_node.reborrow().def();
                        queue.extend(&node_def.child_regions);

                        // HACK(eddyb) while supporting unstructured control-flow
                        // in `qptr::legalize` is a non-goal, this functionality
                        // may be repurposed for some kind of CPS-like transform.
                        if let NodeKind::ThunkBind(target) = node_def.kind {
                            match target {
                                cf::unstructured::ControlTarget::Region(body) => {
                                    queue.push_back(body);
                                }
                                cf::unstructured::ControlTarget::Return => {}
                            }
                        }
                    }
                    self.scan_node(func_at_node);
                }
            }
        }

        // HACK(eddyb) re-scan anything `scan_node` ignores, but which happens
        // to be cached in `summarized_ptr` due to loop body input summarization,
        // left in its `loop_body_input_deps`-carrying transient state.
        for summarized_ptr_idx in 0.. {
            let Some((&ptr, summary)) = self.summarized_ptrs.get_index(summarized_ptr_idx) else {
                break;
            };

            let incomplete =
                summary.bases.as_ref().right().is_some_and(|u| !u.loop_body_input_deps.is_empty());
            if incomplete {
                self.scan_ptr(func_def_body.at_mut(Value::Var(ptr)));
            }

            // HACK(eddyb) opportunistically generate diagnostics here as well.
            let summary = &self.summarized_ptrs[summarized_ptr_idx];
            if let Some(unsupported) =
                (summary.bases.as_ref().right().and_then(|u| u.unsupported.as_ref()))
                    .filter(|u| !u.directly_used.is_empty())
            {
                // FIXME(eddyb) consider attaching diagnostics to an
                // e.g. entire `node`, in some cases.
                let ptr_def_attrs = &mut func_def_body.vars[ptr].attrs;

                let mut generic_diag_added = false;
                for &used_ptr in &unsupported.directly_used {
                    // FIXME(eddyb) provide more information.
                    match used_ptr {
                        Value::Const(ct) => {
                            ptr_def_attrs.push_diag(
                                cx,
                                Diag::bug(["unsupported pointer `".into(), ct.into(), "`".into()]),
                            );
                        }
                        Value::Var(used_ptr) => {
                            if used_ptr == ptr && !generic_diag_added {
                                ptr_def_attrs
                                    .push_diag(cx, Diag::bug(["unsupported pointer".into()]));
                                generic_diag_added = true;
                            }
                        }
                    }
                }
            }
        }
    }

    fn scan_node(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        let cx = &self.legalizer.cx;

        let is_qptr = |ty: Type| matches!(cx[ty].kind, TypeKind::QPtr);

        let node = func_at_node.position;

        let node_def = func_at_node.reborrow().def();
        let sole_output = match node_def.outputs[..] {
            [o] if node_def.child_regions.is_empty() => Some(o),
            _ => None,
        };
        let sole_output_def_law = sole_output.and_then(|output| {
            let func = func_at_node.reborrow().freeze();
            self.legalizer
                .maybe_def_law(func.at(Value::Var(output)), || self.qptr_def_law(func.at(output)))
        });

        if let Some(DefLaw::EncodeEscapingQPtr) = sole_output_def_law {
            let ptr = func_at_node.reborrow().def().inputs[0];
            self.scan_escaping_ptr_for_encode(func_at_node.reborrow().at(ptr));
        }
        if let Some(input_idx) = self
            .legalizer
            .node_qptr_expansions(func_at_node.reborrow().freeze())
            .pre_encode_escaping_input
        {
            let ptr = func_at_node.reborrow().def().inputs[input_idx];
            self.scan_escaping_ptr_for_encode(func_at_node.reborrow().at(ptr));
        }
        {
            let node_def = func_at_node.reborrow().def();
            if let NodeKind::FuncCall(_) | NodeKind::ThunkBind(_) = node_def.kind {
                for input_idx in 0..node_def.inputs.len() {
                    let func = func_at_node.reborrow().freeze();
                    let input = func.at(node).def().inputs[input_idx];
                    if is_qptr(func.at(input).type_of(cx)) {
                        self.scan_escaping_ptr_for_encode(func_at_node.reborrow().at(input));
                    }
                }
            }
        }

        // TODO(eddyb) use `DefLaw` for more than `QPtrDefLaw::DecodeEscaped`
        // and `DefLaw::CmpQPtrEqOrNe`.
        match sole_output_def_law {
            // While `QPtrDefLaw::Offset` is intended to be summarized on-demand,
            // leaf offsets, even those derived from `BaseChoice::Many` pointers,
            // may never be demanded by anything, and yet they should be scanned.
            Some(DefLaw::QPtr(QPtrDefLaw::Offset(offset))) => {
                let func = func_at_node.reborrow().freeze();
                let input_ptr = func.nodes[node].inputs[0];

                // HACK(eddyb) the scanning order isn't necessarily def-before-use,
                // so the cache can't be relied upon to have all relevant entries,
                // but thankfully only `Offset`s themselves must remain on-demand.
                let input_bases = {
                    let mut ptr = input_ptr;
                    loop {
                        let Value::Var(ptr_var) = ptr else {
                            break;
                        };
                        if self.summarized_ptrs.contains_key(&ptr_var) {
                            break;
                        }
                        let Some(ptr_node_def) =
                            func.vars[ptr_var].def_parent.right().map(|node| &func.nodes[node])
                        else {
                            break;
                        };

                        if let NodeKind::QPtr(QPtrOp::Offset(_) | QPtrOp::DynOffset { .. }) =
                            ptr_node_def.kind
                        {
                            ptr = ptr_node_def.inputs[0];
                        } else {
                            break;
                        }
                    }

                    self.summarize_ptr(func.at(ptr)).bases
                };

                let could_be_multi_base = input_bases.has_right()
                    || input_bases
                        .left()
                        .is_some_and(|bases| matches!(bases, BaseChoice::Many { .. }));

                // HACK(eddyb) to avoid having to handle negative offsets in
                // `mem::analyze` or `qptr::lift`, they are also always legalized.
                let could_be_negative = match offset {
                    Offset::Zero => false,
                    Offset::Strided { stride: _, index } => match index {
                        IndexValue::Dyn(offset) => {
                            let is_unsigned = matches!(
                                func.at(offset).type_of(cx).as_scalar(cx),
                                Some(scalar::Type::UInt(_))
                            );
                            !is_unsigned
                        }
                        IndexValue::One => false,
                        IndexValue::MinusOne => true,
                    },
                };

                let needs_legalization = could_be_multi_base || could_be_negative;
                if needs_legalization {
                    self.scan_ptr(func_at_node.reborrow().at(Value::Var(sole_output.unwrap())));
                }
            }

            Some(DefLaw::QPtr(QPtrDefLaw::DecodeEscaped)) => {
                self.scan_ptr(func_at_node.reborrow().at(Value::Var(sole_output.unwrap())));
            }

            Some(DefLaw::CmpQPtrEqOrNe { .. }) => {
                for input_ptr in func_at_node.reborrow().def().inputs.clone() {
                    self.scan_ptr(func_at_node.reborrow().at(input_ptr));
                }
            }

            Some(
                DefLaw::QPtr(QPtrDefLaw::Base(_) | QPtrDefLaw::Dyn | QPtrDefLaw::Unsupported)
                | DefLaw::EncodeEscapingQPtr,
            )
            | None => {}
        }

        let node_def = func_at_node.reborrow().def();
        match &node_def.kind {
            // FIXME(eddyb) consider attaching diagnostics here (via `FuncAtMut`)?
            NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::FuncCall(_) => {
                let num_outputs = node_def.outputs.len();

                let mut func = func_at_node.at(());
                for output_idx in 0..num_outputs {
                    let output_var = func.reborrow().at(node).def().outputs[output_idx];
                    if is_qptr(func.vars[output_var].ty) {
                        self.scan_ptr(func.reborrow().at(Value::Var(output_var)));
                    }
                }
            }

            NodeKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(_))
            | NodeKind::Scalar(_)
            | NodeKind::Vector(_)
            | NodeKind::Mem(
                MemOp::FuncLocalVar(_)
                | MemOp::Load { .. }
                | MemOp::Store { .. }
                | MemOp::Copy { .. },
            )
            | NodeKind::QPtr(
                QPtrOp::HandleArrayIndex
                | QPtrOp::BufferData
                | QPtrOp::BufferDynLen { .. }
                | QPtrOp::Offset(_)
                | QPtrOp::DynOffset { .. },
            )
            | NodeKind::ThunkBind(_)
            | NodeKind::SpvInst(..)
            | NodeKind::SpvExtInst { .. } => {}
        }
    }

    // HACK(eddyb) in order to guarantee `summarize_ptr` can't call this,
    // it requires `FuncAtMut` instead of just `FuncAt`.
    fn scan_escaping_ptr_for_encode(&mut self, func_at_ptr: FuncAtMut<'_, Value>) {
        self.ever_encodes_or_decodes_escaped_ptrs = true;

        let PtrSummary { def_law: _, bases, offset_shape } =
            self.summarize_ptr(func_at_ptr.freeze());
        if let Some(unknowns) = bases.as_ref().right() {
            let UnknownBases { any_escaped, loop_body_input_deps, unsupported: _ } = unknowns;
            if let Some(AnyEscapedBase) = any_escaped {
                self.legalizer.in_place_merge_offset_shapes(
                    &mut self
                        .write_back_escaped_ptr_offset_shape
                        .get_or_insert((AnyEscapedBase, Offset::Zero))
                        .1,
                    offset_shape,
                );
            }
            assert!(loop_body_input_deps.is_empty());
        }
        let bases =
            bases.as_ref().left().into_iter().flat_map(|bases| bases.iter(&self.func_base_map));
        for base in bases {
            let merged_offset_shape = match base {
                // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
                Base::Global(GlobalBase::Undef) => continue,

                Base::Global(global_base) => {
                    self.escaped_base_map.bases.entry(global_base).or_insert(Offset::Zero)
                }
                Base::FuncLocal(local) => {
                    self.escaping_func_locals.entry(local).or_insert(Offset::Zero)
                }
            };
            self.legalizer.in_place_merge_offset_shapes(merged_offset_shape, offset_shape);
        }
    }

    // HACK(eddyb) in order to guarantee `summarize_ptr` can't call this,
    // it requires `FuncAtMut` instead of just `FuncAt`.
    fn scan_ptr(&mut self, func_at_ptr: FuncAtMut<'_, Value>) {
        // NOTE(eddyb) a somewhat unusual invariant is that top-level calls to
        // `summarize_ptr` cannot result in any `loop_body_input_deps`, because
        // any loop body inputs encountered will either be already summarized,
        // or never encountered before (and fully summarize themselves), with
        // `loop_body_input_deps` only showing up in the results of calls to
        // `summarize_ptr` *from* the loop body input summarization loop.
        let summary = self.summarize_ptr(func_at_ptr.freeze());
        let incomplete =
            summary.bases.as_ref().right().is_some_and(|u| !u.loop_body_input_deps.is_empty());
        assert!(!incomplete);

        // FIXME(eddyb) consider attaching diagnostics here (via `FuncAtMut`)?
    }

    #[must_use]
    fn summarize_ptr(&mut self, func_at_ptr: FuncAt<'_, Value>) -> PtrSummary {
        let ptr = func_at_ptr.position;

        // HACK(eddyb) only cache `Value::Var`s, for post-scanning iteration.
        let cache_key = match ptr {
            Value::Var(ptr) => Some(ptr),
            Value::Const(_) => None,
        };

        let propagate = |mut s: PtrSummary| {
            if let Some(unknowns) = s.bases.as_mut().right()
                && let Some(unsupported) = &mut unknowns.unsupported
            {
                // Keep the value for direct users of `ptr`, but clear it for
                // users of direct users of `ptr` (while keeping it in the cache).
                if unsupported.directly_used[..] != [ptr] {
                    unsupported.directly_used = [].into_iter().collect();
                }
            }
            s
        };

        let cached = cache_key.and_then(|key| self.summarized_ptrs.get(&key)).filter(|cached| {
            let retry = cached.bases.as_ref().right().is_some_and(|unknowns| {
                // Retry if the relevant loop body input has since been processed.
                match unknowns.loop_body_input_deps[..] {
                    [] => false,
                    [input] => {
                        Value::Var(input) != ptr
                            && self.summarized_ptrs.get(&input).is_some_and(|input_summary| {
                                let incomplete = (input_summary.bases.as_ref().right())
                                    .is_some_and(|u| !u.loop_body_input_deps.is_empty());
                                !incomplete
                            })
                    }
                    // HACK(eddyb) more than one `loop_body_input_deps` entry
                    // is never cached, to make the above check simpler.
                    _ => unreachable!(),
                }
            });
            !retry
        });
        if let Some(cached) = cached {
            return propagate(cached.clone());
        }

        let s = self.summarize_ptr_uncached(func_at_ptr);
        if let Some(key) = cache_key {
            // HACK(eddyb) make it easier to check for retries.
            let avoid_caching =
                (s.bases.as_ref().right()).is_some_and(|u| u.loop_body_input_deps.len() > 1);
            if !avoid_caching {
                self.summarized_ptrs.insert(key, s.clone());
            }
        }
        propagate(s)
    }

    fn summarize_ptr_uncached(&mut self, func_at_ptr: FuncAt<'_, Value>) -> PtrSummary {
        let ptr = func_at_ptr.position;
        let func = func_at_ptr.at(());

        let ptr = match ptr {
            Value::Const(ptr) => {
                return match self.legalizer.const_qptr_to_base_and_offset(ptr) {
                    Some((base, offset)) => PtrSummary {
                        // HACK(eddyb) this field's value won't ever be read out.
                        def_law: QPtrDefLaw::Unsupported,
                        bases: EitherOrBoth::Left(BaseChoice::Single(Base::Global(base))),
                        offset_shape: offset.map_index_value(|_| {
                            // FIXME(eddyb) this could lead to index conflicts down the line.
                            self.legalizer.u32_type()
                        }),
                    },
                    None => PtrSummary::unsupported(Value::Const(ptr)),
                };
            }
            Value::Var(ptr) => ptr,
        };

        let def_law = self.qptr_def_law(func.at(ptr));
        match def_law {
            QPtrDefLaw::Base(base) => PtrSummary {
                def_law,
                bases: EitherOrBoth::Left(BaseChoice::Single(base)),
                offset_shape: Offset::Zero,
            },
            QPtrDefLaw::Offset(new_offset) => {
                let node = func.at(ptr).decl().def_parent.right().unwrap();

                let PtrSummary { def_law: _, bases, offset_shape } =
                    self.summarize_ptr(func.at(func.at(node).def().inputs[0]));

                // FIXME(eddyb) deduplicate this logic with legalization.
                let new_offset_shape = new_offset.map_index_value(|new_index| match new_index {
                    IndexValue::Dyn(index) => Some(func.at(index).type_of(&self.legalizer.cx)),
                    IndexValue::One | IndexValue::MinusOne => None,
                });

                PtrSummary {
                    def_law,
                    bases,
                    offset_shape: offset_shape.merge(
                        new_offset_shape,
                        |old_index_ty| old_index_ty,
                        |new_index_ty| {
                            new_index_ty.unwrap_or_else(|| {
                                // FIXME(eddyb) this could lead to index conflicts down the line.
                                self.legalizer.u32_type()
                            })
                        },
                        |old, new| {
                            // HACK(eddyb) always try merging, just in case
                            // the multiplifiers happen to be relevant.
                            self.legalizer.merge_scaled_index_types(
                                old,
                                new.map_value(|new_index_ty| new_index_ty.unwrap_or(old.value)),
                            )
                        },
                    ),
                }
            }
            QPtrDefLaw::DecodeEscaped => {
                self.ever_encodes_or_decodes_escaped_ptrs = true;
                PtrSummary {
                    def_law,
                    bases: EitherOrBoth::Right(UnknownBases {
                        any_escaped: Some(AnyEscapedBase),
                        loop_body_input_deps: [].into_iter().collect(),
                        unsupported: None,
                    }),
                    offset_shape: Offset::Zero,
                }
            }
            QPtrDefLaw::Dyn => match func.at(ptr).decl().kind() {
                VarKind::NodeOutput { node, output_idx } => {
                    let node_def = func.at(node).def();
                    match node_def.kind {
                        // HACK(eddyb) this is the exact same handling as for
                        // `QPtrDefLaw::DecodeEscaped` above, except that calls
                        // do not *actually* encode/decode pointers, they only
                        // use the same `AnyEscapedBase` tracking (for now).
                        NodeKind::FuncCall(_) => {
                            self.ever_encodes_or_decodes_escaped_ptrs = true;
                            return PtrSummary {
                                def_law,
                                bases: EitherOrBoth::Right(UnknownBases {
                                    any_escaped: Some(AnyEscapedBase),
                                    loop_body_input_deps: [].into_iter().collect(),
                                    unsupported: None,
                                }),
                                offset_shape: Offset::Zero,
                            };
                        }

                        NodeKind::Select(_) | NodeKind::Loop { .. } => {}

                        _ => unreachable!(),
                    }

                    let per_child_region_output = node_def
                        .child_regions
                        .iter()
                        .map(|&case| func.at(case).def().outputs[output_idx as usize]);

                    // HACK(eddyb) as the loop output and body input/output have
                    // to be legalized in exactly the same way, the initial input
                    // needs to be included in this merge, in order to get the
                    // same summary for both (loop output vs body input) `Var`s.
                    let loop_initial_input = match node_def.kind {
                        NodeKind::Loop { .. } => Some(node_def.inputs[output_idx as usize]),
                        _ => None,
                    };
                    let mut merge_sources =
                        loop_initial_input.into_iter().chain(per_child_region_output);

                    // HACK(eddyb) can't use `map` + `reduce` due to `self` borrowing.
                    let mut s = {
                        let first = merge_sources.next().unwrap();
                        self.summarize_ptr(func.at(first))
                    };
                    for source in merge_sources {
                        let source_summary = self.summarize_ptr(func.at(source));
                        s = self.merge_summaries(s, source_summary);
                    }
                    s
                }

                // Loop body inputs are the only values with inherently cyclic sources
                // (i.e. they can, and often do, depend on the previous loop iteration),
                // and as such they require a more free-form "saturating" algorithm.
                VarKind::RegionInput { region, input_idx: _ } => {
                    let Some(&node) = self.loop_map.loop_body_region_to_loop_node.get(region)
                    else {
                        // FIXME(eddyb) actually confirm this is e.g. the function body.

                        // HACK(eddyb) this is the exact same handling as for
                        // `QPtrDefLaw::DecodeEscaped` above, except that calls
                        // do not *actually* encode/decode pointers, they only
                        // use the same `AnyEscapedBase` tracking (for now).
                        self.ever_encodes_or_decodes_escaped_ptrs = true;
                        return PtrSummary {
                            def_law,
                            bases: EitherOrBoth::Right(UnknownBases {
                                any_escaped: Some(AnyEscapedBase),
                                loop_body_input_deps: [].into_iter().collect(),
                                unsupported: None,
                            }),
                            offset_shape: Offset::Zero,
                        };
                    };
                    let node_def = func.at(node).def();
                    assert!(matches!(node_def.kind, NodeKind::Loop { .. }));

                    let mut s = PtrSummary {
                        def_law,
                        bases: EitherOrBoth::Right(UnknownBases {
                            any_escaped: None,
                            loop_body_input_deps: [ptr].into_iter().collect(),
                            unsupported: None,
                        }),
                        offset_shape: Offset::Zero,
                    };

                    // Avoid cyclic recursion by pre-filling the cache entry.
                    self.summarized_ptrs.insert(ptr, s.clone());

                    // HACK(eddyb) using `loop_body_input_deps` as a weird queue.
                    let mut seen = FxHashSet::default();
                    while let Some(candidate) =
                        s.bases.as_mut().right().and_then(|u| u.loop_body_input_deps.pop())
                    {
                        if !seen.insert(candidate) {
                            continue;
                        }
                        // NOTE(eddyb) less care needed here because we ensure all
                        // entries in `loop_body_input_deps` come from loops.
                        let candidate_source_ptrs = {
                            let candidate_decl = func.at(candidate).decl();
                            let candidate_loop_body = candidate_decl.def_parent.left().unwrap();
                            [
                                &func
                                    .at(self.loop_map.loop_body_region_to_loop_node
                                        [candidate_loop_body])
                                    .def()
                                    .inputs,
                                &func.at(candidate_loop_body).def().outputs,
                            ]
                            .map(|inputs_or_outputs| {
                                inputs_or_outputs[candidate_decl.def_idx as usize]
                            })
                        };
                        for candidate_source_ptr in candidate_source_ptrs {
                            let candidate_summary =
                                self.summarize_ptr(func.at(candidate_source_ptr));
                            s = self.merge_summaries(s, candidate_summary);
                        }
                    }

                    // HACK(eddyb) this should be the only place where `UnknownBases`
                    // can end up with `(None, [], None)` fields.
                    let any_unknowns = s.bases.as_ref().right().is_some_and(
                        |UnknownBases { any_escaped, loop_body_input_deps, unsupported }| {
                            assert!(loop_body_input_deps.is_empty());
                            any_escaped.is_some() || unsupported.is_some()
                        },
                    );
                    if !any_unknowns && s.bases.has_left() {
                        s.bases = EitherOrBoth::Left(s.bases.left().unwrap());
                    }

                    s
                }
            },
            QPtrDefLaw::Unsupported => PtrSummary::unsupported(Value::Var(ptr)),
        }
    }

    #[must_use]
    fn merge_summaries(&mut self, a: PtrSummary, b: PtrSummary) -> PtrSummary {
        let (a_bases, a_unknowns) = a.bases.map_any(Some, Some).or_default();
        let (b_bases, b_unknowns) = b.bases.map_any(Some, Some).or_default();

        // FIXME(eddyb) this should really be part of `EitherOrBoth`.
        fn maybe_either_or_both<L, R>(
            left: Option<L>,
            right: Option<R>,
        ) -> Option<EitherOrBoth<L, R>> {
            match (left, right) {
                (Some(l), None) => Some(EitherOrBoth::Left(l)),
                (None, Some(r)) => Some(EitherOrBoth::Right(r)),
                (Some(l), Some(r)) => Some(EitherOrBoth::Both(l, r)),
                (None, None) => None,
            }
        }

        let bases = maybe_either_or_both(a_bases, b_bases).map(|ab| {
            ab.reduce(|a, b| {
                a.merge(b, |base| self.func_base_map.many_choice_bases.insert_full(base).0)
            })
        });

        let unknowns = maybe_either_or_both(a_unknowns, b_unknowns).map(|ab| {
            // HACK(eddyb) this may be useful elsewhere too?
            fn merge_smallvecs<A: smallvec::Array>(
                mut a: SmallVec<A>,
                mut b: SmallVec<A>,
            ) -> SmallVec<A> {
                if a.len() >= b.len() {
                    a.append(&mut b);
                    a
                } else {
                    b.insert_many(0, a);
                    b
                }
            }
            fn dedup_smallvec<T: Copy + Eq + Hash, const N: usize>(
                mut xs: SmallVec<[T; N]>,
            ) -> SmallVec<[T; N]>
            where
                [T; N]: smallvec::Array<Item = T>,
            {
                // HACK(eddyb) `dedup` can help but only for adjacent elements.
                xs.dedup();
                if xs.len() > 2 {
                    let mut seen = FxHashSet::default();
                    xs.retain(|x| seen.insert(*x));
                }
                if xs.len() <= N {
                    xs.shrink_to_fit();
                }
                xs
            }

            ab.reduce(|a, b| UnknownBases {
                any_escaped: a.any_escaped.or(b.any_escaped),
                loop_body_input_deps: dedup_smallvec(merge_smallvecs(
                    a.loop_body_input_deps,
                    b.loop_body_input_deps,
                )),
                unsupported: maybe_either_or_both(a.unsupported, b.unsupported).map(|ab| {
                    ab.reduce(|a, b| UnsupportedBases {
                        directly_used: merge_smallvecs(a.directly_used, b.directly_used),
                    })
                }),
            })
        });

        PtrSummary {
            def_law: QPtrDefLaw::Dyn,
            bases: maybe_either_or_both(bases, unknowns).unwrap(),
            offset_shape: self.legalizer.merge_offset_shapes(a.offset_shape, b.offset_shape),
        }
    }

    /// Determine the `QPtrDefLaw` (see also its docs) for a `qptr` value definition,
    /// returning `QPtrDefLaw::Unsupported` for unrecognized `qptr` values.
    //
    // FIXME(eddyb) `QPtrDefLaw::Unsupported` should probably be `Err`?
    // FIXME(eddyb) try refactoring this further, now that recomputation is rarer.
    fn qptr_def_law(&self, func_at_ptr: FuncAt<'_, Var>) -> QPtrDefLaw {
        let cx = &self.legalizer.cx;
        let wk = self.legalizer.wk;

        let ptr = func_at_ptr.position;
        let func = func_at_ptr.at(());

        let (node, output_idx) = match func.at(ptr).decl().kind() {
            VarKind::NodeOutput { node, output_idx } => (node, output_idx),
            VarKind::RegionInput { .. } => return QPtrDefLaw::Dyn,
        };

        // HACK(eddyb) `post_decode_escaped_output` is applied by moving `ptr`
        // to a new node (which on its own would be `QPtrDefLaw::DecodeEscaped`).
        if self.legalizer.node_qptr_expansions(func.at(node)).post_decode_escaped_output
            == Some(output_idx as usize)
        {
            return QPtrDefLaw::DecodeEscaped;
        }

        let node_def = func.at(node).def();
        match &node_def.kind {
            NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::FuncCall(_) => QPtrDefLaw::Dyn,

            // FIXME(eddyb) should these generate `Diag::bug` instead?
            NodeKind::Scalar(_) | NodeKind::Vector(_) => unreachable!(),

            NodeKind::Mem(op) => {
                assert_eq!(output_idx, 0);
                match *op {
                    MemOp::FuncLocalVar(_) => QPtrDefLaw::Base(Base::FuncLocal(FuncLocal {
                        qptr_output: node_def.outputs[0],
                    })),

                    // FIXME(eddyb) should these generate `Diag::bug` instead?
                    MemOp::Load { .. } | MemOp::Store { .. } | MemOp::Copy { .. } => {
                        unreachable!()
                    }
                }
            }

            NodeKind::QPtr(op) => {
                assert_eq!(output_idx, 0);
                match *op {
                    // FIXME(eddyb) implement.
                    QPtrOp::HandleArrayIndex => QPtrDefLaw::Unsupported,
                    QPtrOp::BufferData => match node_def.inputs[..] {
                        [Value::Const(ct)]
                            if matches!(
                                cx[ct].kind,
                                ConstKind::PtrToGlobalVar { global_var: _, offset: None }
                            ) =>
                        {
                            QPtrDefLaw::Base(Base::Global(GlobalBase::BufferData {
                                ptr_to_buffer: ct,
                            }))
                        }
                        _ => QPtrDefLaw::Unsupported,
                    },

                    QPtrOp::Offset(offset) => QPtrDefLaw::Offset(
                        NonZeroU32::new(offset.unsigned_abs()).map_or(Offset::Zero, |stride| {
                            Offset::Strided {
                                stride,
                                index: if offset < 0 {
                                    IndexValue::MinusOne
                                } else {
                                    IndexValue::One
                                },
                            }
                        }),
                    ),
                    // FIXME(eddyb) ignoring `index_bounds` (and later setting it to
                    // `None`) is lossy and can frustrate e.g. type recovery.
                    QPtrOp::DynOffset { stride, .. } => QPtrDefLaw::Offset(Offset::Strided {
                        stride,
                        index: IndexValue::Dyn(node_def.inputs[1]),
                    }),

                    // FIXME(eddyb) should these generate `Diag::bug` instead?
                    QPtrOp::BufferDynLen { .. } => {
                        unreachable!()
                    }
                }
            }

            NodeKind::SpvInst(spv_inst, lowering)
                if [wk.OpConvertUToPtr, wk.OpBitcast].contains(&spv_inst.opcode)
                    && lowering.disaggregated_output.is_none()
                    && lowering.disaggregated_inputs.is_empty()
                    && node_def.inputs.len() == 1
                    && func.at(node_def.inputs[0]).type_of(cx).as_scalar(cx).is_some() =>
            {
                let Some(input_scalar_type) = func.at(node_def.inputs[0]).type_of(cx).as_scalar(cx)
                else {
                    return QPtrDefLaw::Unsupported;
                };

                // HACK(eddyb) `post_decode_escaped_output` check above should
                // catch the cases where the type isn't exactly the expected one.
                assert!(
                    input_scalar_type
                        == self.legalizer.qptr_sized_uint_type().as_scalar(cx).unwrap()
                );

                QPtrDefLaw::DecodeEscaped
            }

            NodeKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(_))
            | NodeKind::ThunkBind(_)
            | NodeKind::SpvInst(..)
            | NodeKind::SpvExtInst { .. } => QPtrDefLaw::Unsupported,
        }
    }
}

// FIXME(eddyb) reconsider this name/API, maybe centralize it?
type InsertionCursor<'a> = FuncAtMut<'a, InsertPosition>;

// FIXME(eddyb) reconsider this name/API, maybe centralize it?
#[derive(Copy, Clone)]
struct InsertPosition {
    parent_region: Region,
    anchor: InsertAnchor,
}

// FIXME(eddyb) reconsider this name/API, maybe centralize it?
#[derive(Copy, Clone)]
enum InsertAnchor {
    /// Inserts a `node` at the start of the region, then becomes `After(node)`
    /// (in order to preserve the relative order of multiple inserted nodes).
    First,

    /// Inserts each new `node` after the existing one, then becomes `After(node)`
    /// (in order to preserve the relative order of multiple inserted nodes).
    After(Node),

    /// Inserts each new node before the existing one, without any updates
    /// (the relative order of multiple inserted nodes is inherently preserved).
    Before(Node),

    /// Inserts each new node at the end of the region, without any updates
    /// (the relative order of multiple inserted nodes is inherently preserved).
    Last,
}

// FIXME(eddyb) move these into `func_at` and/or introduce a new API.
// FIXME(eddyb) consider moving `Context` into a field of `FuncAt`, or even
// keep an `Rc<Context>` in each `EntityDefs` map?
impl FuncAtMut<'_, InsertPosition> {
    fn decl_var(&mut self, cx: &Context, ty: Type) -> Var {
        self.vars.define(
            cx,
            VarDecl {
                attrs: Default::default(),
                ty,

                // HACK(eddyb) placeholder `def_{parent,idx}` values.
                // FIXME(eddyb) consider supporting "detached `Var`"s outright.
                def_parent: Either::Left(self.position.parent_region),
                def_idx: !0,
            },
        )
    }

    fn insert_node(&mut self, cx: &Context, def: NodeDef) {
        let InsertPosition { parent_region, anchor } = &mut self.position;
        let node = self.nodes.define(cx, def.into());
        let nodes = &mut self.regions[*parent_region].children;
        match *anchor {
            InsertAnchor::First => {
                nodes.insert_first(node, self.nodes);
                *anchor = InsertAnchor::After(node);
            }
            InsertAnchor::After(prev) => {
                nodes.insert_after(node, prev, self.nodes);
                *anchor = InsertAnchor::After(node);
            }
            InsertAnchor::Before(next) => nodes.insert_before(node, next, self.nodes),
            InsertAnchor::Last => nodes.insert_last(node, self.nodes),
        }

        // HACK(eddyb) attach any provided output vars to the new node.
        for (output_idx, &output_var) in self.nodes[node].outputs.iter().enumerate() {
            let output_var_decl = &mut self.vars[output_var];
            output_var_decl.def_parent = Either::Right(node);
            output_var_decl.def_idx = output_idx.try_into().unwrap();
        }
    }
}

struct LegalizePtrsInFunc<'a> {
    legalizer: &'a LegalizePtrs<'a>,

    module_global_vars: &'a EntityDefs<GlobalVar>,
    escaped_base_map: &'a EscapedBaseMap,
    // HACK(eddyb) only used here to streamline function call legalization.
    escaped_ptr_shape: (BaseChoice<()>, Offset<Type>),
    func_base_map: FuncLocalBaseMap,

    parent_region: Option<Region>,

    // FIXME(eddyb) consider using `EntityOrientedDenseMap`, or some maybe-sparse
    // version of it, just to reduce the cost of random-access.
    canon_ptrs: FxHashMap<Var, CanonPtr>,
}

/// Canonicalized pointer value, combining any number of offsetting steps, on
/// top of a `base` that (when `BaseChoice::Many`) respects `FuncLocalBaseMap`.
///
/// While eagerly combining offsets may be suboptimal for highly local dataflow
/// (e.g. `p2 = if c { p.add(1) } else { p };` can be `p2 = p.add(c as usize);`,
/// but `CanonPtr` will use `p2 = p_base.add(p_idx + (c as usize))` instead),
/// it has the benefit of allow cheap merging of `CanonPtr`s, without having to
/// e.g. dig for some kind of "closest common base".
//
// FIXME(eddyb) reduce duplication between `PtrSummary` and this.
// FIXME(eddyb) support `HandleArrayIndex`.
#[derive(Clone)]
struct CanonPtr<V = DefOrUse> {
    def_law: QPtrDefLaw,

    // HACK(eddyb) there is no need for `BaseChoice::Many`'s `base_index_selector`
    // field to ever be `Value::ConstConst` (as `BaseChoice::Single` can be used),
    // in fact `BaseChoice::Many` should perhaps always use `Var` instead of `Value`?
    // (this is complicated by `CanonPtr` being generic over `Value` vs `DefOrUse`)
    base: BaseChoice<V>,

    offset: Offset<V>,

    /// If (and only if) `base` is `BaseChoice::Single(GlobalBase::BufferData)`,
    /// this is always an existing `QPtrOp::BufferData` output for that buffer.
    single_base_buffer_data: Option<V>,
}

#[derive(Copy, Clone)]
enum DefOrUse {
    Def(Var),
    Use(Value),
}

impl DefOrUse {
    #[track_caller]
    fn expect_def(self) -> Var {
        match self {
            DefOrUse::Def(v) => v,
            DefOrUse::Use(_) => unreachable!(),
        }
    }
    fn to_use(self) -> Value {
        match self {
            DefOrUse::Def(v) => Value::Var(v),
            DefOrUse::Use(v) => v,
        }
    }
}

// FIXME(eddyb) pull this out into better abstractions?
struct TransformSmallVec<T: Copy, const N: usize>
where
    [T; N]: smallvec::Array<Item = T>,
{
    old: SmallVec<[T; N]>,
    used_old_range: RangeTo<usize>,
    new: SmallVec<[T; N]>,
}
impl<T: Copy, const N: usize> TransformSmallVec<T, N>
where
    [T; N]: smallvec::Array<Item = T>,
{
    fn new(old: SmallVec<[T; N]>) -> Self {
        TransformSmallVec { old, used_old_range: ..0, new: SmallVec::new() }
    }

    fn finish(mut self) -> SmallVec<[T; N]> {
        if self.used_old_range == ..0 {
            self.old
        } else {
            self.ensure_new_for_old_range(..self.old.len());
            self.new
        }
    }

    fn ensure_new_for_old_range(&mut self, old_range: RangeTo<usize>) {
        if (self.new.len(), self.used_old_range) == (0, ..0) {
            self.new.reserve(self.old.len());
        }
        self.new.extend(self.old[self.used_old_range.end..old_range.end].iter().copied());
        self.used_old_range = old_range;
    }

    // FIXME(eddyb) document and/or design this better to be clearer that,
    // besides the (largely amortized) cost of allocation, this is ~O(1).
    fn replace_old_with_new(&mut self, old_idx: usize, new: impl IntoIterator<Item = T>) {
        self.ensure_new_for_old_range(..old_idx);
        self.new.extend(new);
        self.used_old_range.end += 1;

        assert_eq!(self.used_old_range, ..(old_idx + 1));
    }
}

/// `Node` inputs or `Region` outputs.
//
// FIXME(eddyb) pull this out into better abstractions?
struct DynSink {
    // NOTE(eddyb) "upstream" as in "can define values later used in `values`".
    // HACK(eddyb) this also serves to identify where to reinject `values`
    // (see also `dyn_sink_values_for` below).
    upstream: InsertPosition,
    values: TransformSmallVec<Value, 2>,
}
fn dyn_sink_values_downstream_of(
    upstream_cursor: FuncAtMut<'_, InsertPosition>,
) -> &mut SmallVec<[Value; 2]> {
    let InsertPosition { parent_region, anchor } = upstream_cursor.position;
    let func = upstream_cursor.at(());
    match anchor {
        InsertAnchor::First | InsertAnchor::After(_) => unreachable!(),
        InsertAnchor::Before(node) => &mut func.nodes[node].inputs,
        InsertAnchor::Last => &mut func.regions[parent_region].outputs,
    }
}

impl<'a> LegalizePtrsInFunc<'a> {
    fn new(
        legalizer: &'a LegalizePtrs<'a>,
        module_global_vars: &'a EntityDefs<GlobalVar>,
        escaped_base_map: &'a EscapedBaseMap,
        // HACK(eddyb) only used here to streamline function call legalization.
        escaped_ptr_shape: (BaseChoice<()>, Offset<Type>),
        func_base_map: FuncLocalBaseMap,
        func_def_body: &mut FuncDefBody,
        ptrs: impl ExactSizeIterator<
            Item = (
                Var,
                QPtrDefLaw,
                Result<impl FnOnce(&FuncLocalBaseMap) -> BaseChoice<()>, UnsupportedBases>,
                Offset<Type>,
            ),
        >,
    ) -> Self {
        let cx = &legalizer.cx;

        let mut this = LegalizePtrsInFunc {
            legalizer,
            module_global_vars,
            escaped_base_map,
            escaped_ptr_shape,
            func_base_map,

            parent_region: None,

            canon_ptrs: FxHashMap::default(),
        };

        this.canon_ptrs.reserve(ptrs.len());

        // Allocate canonical `Var`s for base selectors and indices,
        // as indicated by `bases`/`offset_shape`.
        for (ptr, def_law, get_bases_or_unsupported, offset_shape) in ptrs {
            let Ok(get_bases) = get_bases_or_unsupported else {
                continue;
            };
            let bases = get_bases(&this.func_base_map);

            let ptr_def_parent = func_def_body.at(ptr).decl().def_parent;

            let mut override_base_selector = None;
            let mut override_index = None;
            let mut override_single_base_buffer_data = None;

            // HACK(eddyb) pointer offseting cannot affect the base, so in order
            // to avoid generating inefficient patterns like `base2 = base1 + 0`,
            // the input pointer's base gets reused instead.
            // FIXME(eddyb) consider special-casing "offset chains" during scanning,
            // i.e. reducing them to just their irreducible base pointer, and
            // their final stride (might also help with loop fixpoint scanning).
            match def_law {
                QPtrDefLaw::Offset(new_offset) => {
                    // HACK(eddyb) this requires several assumptions
                    // (so that the input pointer already has a `CanonPtr`):
                    // - scanning order ensures def-before-use
                    // - unsupported summaries are transitive

                    // FIXME(eddyb) this may cause unnecessary cloning.
                    let input_canon_ptr = this.use_canonical_ptr(
                        func_def_body.nodes[ptr_def_parent.right().unwrap()].inputs[0],
                    );

                    if let BaseChoice::Many { base_index_selector, .. } = input_canon_ptr.base {
                        override_base_selector = Some(base_index_selector);
                    }
                    match (input_canon_ptr.offset, new_offset) {
                        (Offset::Strided { stride, index }, Offset::Zero) => {
                            override_index = Some((stride, index));
                        }
                        (Offset::Zero, Offset::Strided { stride, index }) => {
                            override_index = Some((
                                stride,
                                match index {
                                    IndexValue::Dyn(index) => index,

                                    // FIXME(eddyb) this can lead to index conflicts down the line.
                                    IndexValue::One => {
                                        Value::Const(cx.intern(scalar::Const::from_u32(1)))
                                    }

                                    // Directly applying a negative offset onto a base is
                                    // *never* legal, as it would *always* leave the bounds.
                                    //
                                    // FIXME(eddyb) this may accidentally get optimized out.
                                    IndexValue::MinusOne => {
                                        let mut attrs = AttrSet::default();
                                        attrs.push_diag(
                                            cx,
                                            Diag::bug(["illegal negative offset".into()]),
                                        );
                                        Value::Const(cx.intern(ConstDef {
                                            attrs,
                                            ty: cx.intern(scalar::Type::U32),
                                            kind: scalar::Const::from_u32(!0).into(),
                                        }))
                                    }
                                },
                            ));
                        }
                        _ => {}
                    }
                    override_single_base_buffer_data = input_canon_ptr.single_base_buffer_data;
                }

                QPtrDefLaw::Base(base @ Base::Global(GlobalBase::BufferData { .. })) => {
                    assert!(bases == BaseChoice::Single(base));
                    override_single_base_buffer_data = Some(Value::Var(ptr));
                }

                _ => {}
            }

            if let Some((stride, index)) = override_index {
                assert!(
                    offset_shape
                        == Offset::Strided { stride, index: func_def_body.at(index).type_of(cx) }
                );
            }

            let mut decl_var = |ty| {
                func_def_body.vars.define(
                    cx,
                    VarDecl {
                        attrs: AttrSet::default(),
                        ty,

                        // HACK(eddyb) placeholder `def_{parent,idx}` values.
                        // FIXME(eddyb) consider supporting "detached `Var`"s outright.
                        def_parent: ptr_def_parent,
                        def_idx: !0,
                    },
                )
            };

            // HACK(eddyb) always provide `single_base_buffer_data` when applicable,
            // with `DefOrUse::Def` indicating a new `QPtrOp::BufferData` node
            // must also be injected at a suitable location.
            // FIXME(eddyb) avoid being wasteful with these and cache them near
            // the top of the function (or the "nearest common dominator"), or
            // completely revamp how buffers are even accessed in the first place.
            let single_base_buffer_data = match bases {
                BaseChoice::Single(Base::Global(GlobalBase::BufferData { .. })) => {
                    Some(override_single_base_buffer_data.map_or_else(
                        || DefOrUse::Def(decl_var(legalizer.qptr_type())),
                        DefOrUse::Use,
                    ))
                }
                BaseChoice::Single(
                    Base::FuncLocal(_)
                    | Base::Global(
                        GlobalBase::Undef | GlobalBase::Null | GlobalBase::GlobalVar { .. },
                    ),
                )
                | BaseChoice::Many { .. } => None,
            };

            this.canon_ptrs.insert(
                ptr,
                CanonPtr {
                    def_law,

                    // TODO(eddyb) figure out if it's best to reference the set
                    // in `BaseChoice` (and/or use `Cow`/`Rc` inside?), or move
                    // it to create a new one (like currently done here).
                    base: bases.map_selector_value(|()| {
                        override_base_selector.map_or_else(
                            || DefOrUse::Def(decl_var(legalizer.u32_type())),
                            DefOrUse::Use,
                        )
                    }),

                    offset: offset_shape.map_index_value(|ty| {
                        override_index.map_or_else(
                            || DefOrUse::Def(decl_var(ty)),
                            |(_, index)| DefOrUse::Use(index),
                        )
                    }),

                    single_base_buffer_data,
                },
            );
        }

        this
    }

    // FIXME(eddyb) avoid cloning by borrowing in the `BaseChoice::Many` case,
    // and/or using some kind of (`Rc`) sharing when there are any allocations.
    fn try_use_canonical_ptr(&self, ptr: Value) -> Result<CanonPtr<Value>, UnsupportedBases> {
        // HACK(eddyb) reusing `UnsupportedBases` without its original purpose.
        let unsupported = UnsupportedBases { directly_used: [].into_iter().collect() };

        let ptr = match ptr {
            Value::Const(ptr) => {
                let (base, offset) =
                    self.legalizer.const_qptr_to_base_and_offset(ptr).ok_or(unsupported)?;
                return Ok(CanonPtr {
                    // HACK(eddyb) this field's value won't ever be read out.
                    def_law: QPtrDefLaw::Unsupported,
                    base: BaseChoice::Single(Base::Global(base)),
                    offset: offset.map_index_value(|index| match index {
                        // FIXME(eddyb) maybe avoid using `IndexValue` since it's
                        // always going to be `IndexValue::One` for const offsets.
                        IndexValue::One => {
                            Value::Const(self.legalizer.cx.intern(scalar::Const::from_u32(1)))
                        }

                        IndexValue::Dyn(_) | IndexValue::MinusOne => unreachable!(),
                    }),
                    single_base_buffer_data: None,
                });
            }
            Value::Var(ptr) => ptr,
        };

        // FIXME(eddyb) get rid of all this cloning.
        let CanonPtr { def_law, base, offset, single_base_buffer_data } =
            self.canon_ptrs.get(&ptr).ok_or(unsupported)?.clone();

        Ok(CanonPtr {
            def_law,
            base: base.map_selector_value(|v| v.to_use()),
            offset: offset.map_index_value(|v| v.to_use()),
            single_base_buffer_data: single_base_buffer_data.map(|v| v.to_use()),
        })
    }

    // FIXME(eddyb) avoid cloning by borrowing in the `BaseChoice::Many` case,
    // and/or using some kind of (`Rc`) sharing when there are any allocations.
    #[track_caller]
    fn use_canonical_ptr(&self, ptr: Value) -> CanonPtr<Value> {
        self.try_use_canonical_ptr(ptr).ok().expect("`CanonPtr` requested for unsupported pointer")
    }

    fn apply_node_qptr_expansions(
        &mut self,
        func_at_node: FuncAtMut<'_, Node>,
        node_qptr_expansions: QPtrExpansions,
    ) {
        let cx = &self.legalizer.cx;
        let wk = self.legalizer.wk;

        let node = func_at_node.position;

        let parent_region = self.parent_region.unwrap();

        let mut func = func_at_node.at(());

        let QPtrExpansions {
            pre_encode_escaping_input,
            post_decode_escaped_output,
            instantiate_by_multi_base_qptr_inputs,
        } = node_qptr_expansions;

        // Handle `pre_encode_escaping_input`/`post_decode_escaped_output`
        // before `instantiate_by_multi_base_qptr_inputs`, as the former simply
        // each add one extra node before/after `node`, whereas the latter can
        // duplicate `node`'s definition (once per possible base choice)
        // so flipping the order would needlessly amplify the final node count.
        if let Some(input_idx) = pre_encode_escaping_input {
            let input_ptr = func.nodes[node].inputs[input_idx];

            let encode_node = func.nodes.define(
                cx,
                NodeDef {
                    // FIXME(eddyb) copy at least debuginfo attrs from `node`.
                    attrs: Default::default(),
                    // FIXME(eddyb) `qptr`-native `OpConvertPtrToU` replacement?
                    kind: NodeKind::SpvInst(wk.OpConvertPtrToU.into(), Default::default()),
                    inputs: [input_ptr].into_iter().collect(),
                    child_regions: [].into_iter().collect(),
                    outputs: [].into_iter().collect(),
                }
                .into(),
            );
            let encode_output_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: self.legalizer.qptr_sized_uint_type(),
                    def_parent: Either::Right(encode_node),
                    def_idx: 0,
                },
            );
            func.nodes[encode_node].outputs.push(encode_output_var);

            // FIXME(eddyb) use `InsertionCursor` here.
            func.regions[parent_region].children.insert_before(encode_node, node, func.nodes);

            self.in_place_transform_node_def(func.reborrow().at(encode_node));

            func.nodes[node].inputs[input_idx] = Value::Var(encode_output_var);
        }
        if let Some(output_idx) = post_decode_escaped_output {
            let undecoded_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: self.legalizer.qptr_sized_uint_type(),
                    def_parent: Either::Right(node),
                    def_idx: output_idx.try_into().unwrap(),
                },
            );

            let output_ptr = mem::replace(&mut func.nodes[node].outputs[output_idx], undecoded_var);

            let decode_node = func.nodes.define(
                cx,
                NodeDef {
                    // FIXME(eddyb) copy at least debuginfo attrs from `node`.
                    attrs: Default::default(),
                    // FIXME(eddyb) `qptr`-native `OpConvertUToPtr` replacement?
                    kind: NodeKind::SpvInst(wk.OpConvertUToPtr.into(), Default::default()),
                    inputs: [Value::Var(undecoded_var)].into_iter().collect(),
                    child_regions: [].into_iter().collect(),
                    outputs: [output_ptr].into_iter().collect(),
                }
                .into(),
            );
            let output_ptr_decl = &mut func.vars[output_ptr];
            output_ptr_decl.def_parent = Either::Right(decode_node);
            output_ptr_decl.def_idx = 0;

            // FIXME(eddyb) use `InsertionCursor` here.
            func.regions[parent_region].children.insert_after(decode_node, node, func.nodes);

            self.in_place_transform_node_def(func.reborrow().at(decode_node));
        }

        let instantiate_by_input = instantiate_by_multi_base_qptr_inputs
            .then(|| func.nodes[node].inputs.iter())
            .and_then(|inputs| {
                inputs.enumerate().find_map(|(i, &input)| match self.try_use_canonical_ptr(input) {
                    Ok(canon @ CanonPtr { base: BaseChoice::Many { .. }, .. }) => Some((i, canon)),
                    _ => None,
                })
            });
        if let Some((input_idx, CanonPtr { base, offset, .. })) = instantiate_by_input {
            let (Some(base_selector), base_choices) =
                base.iter_with_selector_value(&self.func_base_map)
            else {
                unreachable!();
            };

            let node_def_template = NodeDef::clone(&func.nodes[node]);

            // Avoid needing to clone whole child regions as well, which is easy
            // because `Select`/`Loop` never `instantiate_by_multi_base_qptr_inputs`.
            assert_eq!(node_def_template.child_regions.len(), 0);

            // Replace `node` with `switch base_selector { ...base_choices }`,
            // each case having a clone of the original `node` with this
            // specific input replaced with its respective choice of base
            // (while sharing the same `offset` across all cases).
            let mut base_count = base.count();
            let mut switch_case_consts = Vec::with_capacity(base_count);
            let mut switch_cases = SmallVec::with_capacity(base_count + 1);
            for choice_or_default in base_choices.map(Some).chain([None]) {
                // HACK(eddyb) skip adding cases implying undefined behavior,
                // both reducing IR duplication, and avoiding needing to support
                // e.g. `mem::analyze`+`qptr::lift` pointlessly making up typed nulls.
                let always_accesses_input_ptr = match node_def_template.kind {
                    NodeKind::Mem(MemOp::Load { offset } | MemOp::Store { offset })
                        if input_idx == 0 =>
                    {
                        Some(offset.map_or(0, |o| o.get()))
                    }
                    NodeKind::Mem(MemOp::Copy { .. }) => Some(0),
                    _ => None,
                };
                struct AlwaysUb;
                let access_validity = always_accesses_input_ptr.map_or(Ok(()), |access_offset| {
                    match choice_or_default {
                        Some((_, Base::Global(GlobalBase::Undef | GlobalBase::Null))) => {
                            return Err(AlwaysUb);
                        }
                        Some((_, Base::Global(GlobalBase::GlobalVar { ptr_to_binding }))) => {
                            let ConstKind::PtrToGlobalVar { global_var, offset: None } =
                                cx[ptr_to_binding].kind
                            else {
                                unreachable!();
                            };
                            let gv_shape = self.module_global_vars[global_var].shape.unwrap();

                            // HACK(eddyb) a negative offset is out of bounds.
                            let access_offset = u32::try_from(access_offset).map_err(|_e| {
                                assert!(access_offset < 0);
                                AlwaysUb
                            })?;

                            use crate::mem::shapes::GlobalVarShape;
                            match gv_shape {
                                GlobalVarShape::Handles { .. } => return Err(AlwaysUb),
                                GlobalVarShape::UntypedData(mem_layout) => {
                                    // HACK(eddyb) eliding out of bounds accesses
                                    // avoids misshapen `#[mem.accesses]` later.
                                    if access_offset >= mem_layout.size {
                                        return Err(AlwaysUb);
                                    }
                                }
                                GlobalVarShape::TypedInterface(_) => {}
                            }
                        }
                        _ => {}
                    }
                    Ok(())
                });
                if let Err(AlwaysUb) = access_validity {
                    base_count -= 1;
                    continue;
                }

                let case = func.regions.define(cx, RegionDef::default());
                switch_cases.push(case);

                let outputs = if let Some((choice_const, chosen_base)) = choice_or_default {
                    switch_case_consts.push(scalar::Const::from_u32(choice_const.unwrap()));

                    let instantiated_node = func.nodes.define(cx, node_def_template.clone().into());
                    // FIXME(eddyb) use `InsertionCursor` here.
                    func.regions[case].children.insert_first(instantiated_node, func.nodes);

                    // HACK(eddyb) replace the cloned `Var`s with new ones.
                    for output_var in &mut func.nodes[instantiated_node].outputs {
                        let mut output_var_decl = func.vars[*output_var].clone();
                        output_var_decl.def_parent = Either::Right(instantiated_node);
                        *output_var = func.vars.define(cx, output_var_decl);
                    }

                    // NOTE(eddyb) "upstream" as in "can define values later used
                    // by `instantiated_node`".
                    let mut upstream_cursor = func.reborrow().at(InsertPosition {
                        parent_region: case,
                        anchor: InsertAnchor::Before(instantiated_node),
                    });

                    // FIXME(eddyb) try to deduplicate/add new abstractions,
                    // to make stuff like this less of an open-coded mess.
                    let chosen_canon_ptr = CanonPtr {
                        // NOTE(eddyb) this actually happens to cleanly bypass
                        // further processing (e.g. `legalize_vars_defined_by`).
                        def_law: QPtrDefLaw::Unsupported,

                        base: BaseChoice::Single(chosen_base),
                        offset: offset.map_index_value(DefOrUse::Use),

                        // HACK(eddyb) always provide `single_base_buffer_data` when applicable,
                        // with `DefOrUse::Def` indicating a new `QPtrOp::BufferData` node
                        // must also be injected at a suitable location.
                        // FIXME(eddyb) avoid being wasteful with these and cache them near
                        // the top of the function (or the "nearest common dominator"), or
                        // completely revamp how buffers are even accessed in the first place.
                        single_base_buffer_data: match chosen_base {
                            Base::Global(GlobalBase::BufferData { .. }) => Some(DefOrUse::Def(
                                upstream_cursor.decl_var(cx, self.legalizer.qptr_type()),
                            )),
                            Base::FuncLocal(_)
                            | Base::Global(
                                GlobalBase::Undef | GlobalBase::Null | GlobalBase::GlobalVar { .. },
                            ) => None,
                        },
                    };

                    self.legalize_deps_of_base_if_single(&chosen_canon_ptr, &mut upstream_cursor);

                    let chosen_ptr = upstream_cursor.decl_var(cx, self.legalizer.qptr_type());
                    upstream_cursor.insert_node(cx, {
                        let (kind, inputs) =
                            self.single_base_canon_ptr_to_node_kind_and_inputs(&chosen_canon_ptr);
                        NodeDef {
                            // FIXME(eddyb) propagate at least debuginfo attrs from somewhere.
                            attrs: Default::default(),
                            kind,
                            inputs,
                            child_regions: [].into_iter().collect(),
                            outputs: [chosen_ptr].into_iter().collect(),
                        }
                    });

                    // HACK(eddyb) `chosen_ptr` needs to be in `self.canon_ptrs`,
                    // just in case it's an input to e.g. `DefLaw::EncodeEscapingQPtr`.
                    self.canon_ptrs.insert(chosen_ptr, chosen_canon_ptr);

                    let instantiated_node_def = &mut func.nodes[instantiated_node];
                    instantiated_node_def.inputs[input_idx] = Value::Var(chosen_ptr);
                    instantiated_node_def.outputs.iter().copied().map(Value::Var).collect()
                } else {
                    assert_eq!(switch_case_consts.len(), base_count);
                    assert_eq!(switch_cases.len(), base_count + 1);

                    // FIXME(eddyb) the `default` case uses `undef` outputs,
                    // instead of being explicitly `unreachable` - perhaps
                    // the `default` should be optional in a `switch`?
                    node_def_template
                        .outputs
                        .iter()
                        .map(|&output_var| {
                            Value::Const(cx.intern(ConstDef {
                                attrs: Default::default(),
                                ty: func.vars[output_var].ty,
                                kind: ConstKind::Undef,
                            }))
                        })
                        .collect()
                };
                func.regions[case].outputs = outputs;
            }
            let switch_def = func.reborrow().at(node).def();
            switch_def.kind =
                NodeKind::Select(SelectionKind::Switch { case_consts: switch_case_consts });
            switch_def.child_regions = switch_cases;
            switch_def.inputs = [base_selector].into_iter().collect();

            // HACK(eddyb) instantiation can continue within each case, but
            // the outer node is now a `Select`, with no pointer inputs.
            assert!(matches!(
                self.legalizer.node_qptr_expansions(func.reborrow().freeze().at(node)),
                QPtrExpansions {
                    pre_encode_escaping_input: None,
                    post_decode_escaped_output: None,
                    instantiate_by_multi_base_qptr_inputs: false,
                }
            ));
        }
    }

    fn legalize_vars_defined_by(
        &self,
        func_at_def_parent: FuncAtMut<'_, Either<(Region, Option<Node>), Node>>,
    ) {
        let cx = &self.legalizer.cx;

        let def_parent = func_at_def_parent.position;
        let mut func = func_at_def_parent.at(());

        let def_parent = def_parent.map_either(
            |(region, parent_loop_node)| {
                (region, parent_loop_node.map(|node| (node, self.parent_region.unwrap())))
            },
            |node| (node, self.parent_region.unwrap()),
        );

        fn vars_defined_by(
            func_at_def_parent: FuncAtMut<
                '_,
                Either<(Region, Option<(Node, Region)>), (Node, Region)>,
            >,
        ) -> &mut SmallVec<[Var; 2]> {
            let def_parent = func_at_def_parent.position;
            let func = func_at_def_parent.at(());
            def_parent.either(
                |(region, _)| &mut func.regions[region].inputs,
                |(node, _)| &mut func.nodes[node].outputs,
            )
        }
        let mut vars =
            TransformSmallVec::new(mem::take(vars_defined_by(func.reborrow().at(def_parent))));

        // HACK(eddyb) loop outputs have to be changed at the same time as the
        // loop body inputs/outputs, and keep matching count and types, to them.
        let def_parent_loop = def_parent.left().and_then(|(_, parent_loop)| parent_loop);
        let mut loop_output_vars = def_parent_loop.map(|def_parent_loop| {
            TransformSmallVec::new(mem::take(vars_defined_by(
                func.reborrow().at(Either::Right(def_parent_loop)),
            )))
        });

        let maybe_parent_node_and_its_parent_region =
            def_parent.either(|(_, parent_loop)| parent_loop, Some);
        let dyn_sink_upstreams = maybe_parent_node_and_its_parent_region
            .map(|(parent_node, parent_region)| {
                let parent_node_def = func.reborrow().at(parent_node).def();

                // HACK(eddyb) cloning avoids borrow conflicts without messier logic.
                let parent_child_regions = parent_node_def.child_regions.clone();

                let loop_initial_inputs = match parent_node_def.kind {
                    NodeKind::Loop { .. } => Some(InsertPosition {
                        parent_region,
                        anchor: InsertAnchor::Before(parent_node),
                    }),
                    NodeKind::Select(_) => None,
                    _ => {
                        assert_eq!(parent_child_regions.len(), 0);
                        None
                    }
                };
                loop_initial_inputs.into_iter().chain(parent_child_regions.into_iter().map(
                    |child_region| InsertPosition {
                        parent_region: child_region,
                        anchor: InsertAnchor::Last,
                    },
                ))
            })
            .into_iter()
            .flatten();
        let mut dyn_sinks: SmallVec<[DynSink; 2]> = dyn_sink_upstreams
            .map(|upstream| DynSink {
                upstream,
                values: TransformSmallVec::new(mem::take(dyn_sink_values_downstream_of(
                    func.reborrow().at(upstream),
                ))),
            })
            .collect();

        // NOTE(eddyb) "downstream" as in "can use values defined by `def_parent`",
        // but before any pre-existing uses (i.e. allowing "interposing" defs).
        let mut downstream = def_parent.either(
            |(parent_region, parent_loop_node)| {
                let last_func_local_var =
                    parent_loop_node.is_none().then_some(parent_region).and_then(|body| {
                        func.reborrow()
                            .freeze()
                            .at(body)
                            .at_children()
                            .into_iter()
                            .take_while(|func_at_node| {
                                matches!(
                                    func_at_node.def().kind,
                                    NodeKind::Mem(MemOp::FuncLocalVar(_))
                                )
                            })
                            .map(|func_at_node| func_at_node.position)
                            .last()
                    });

                InsertPosition {
                    parent_region,
                    anchor: last_func_local_var.map_or(InsertAnchor::First, InsertAnchor::After),
                }
            },
            |(node, parent_region)| InsertPosition {
                parent_region,
                anchor: InsertAnchor::After(node),
            },
        );
        let mut downstream_of_loop = def_parent_loop.map(|(node, parent_region)| InsertPosition {
            parent_region,
            anchor: InsertAnchor::After(node),
        });

        for original_idx in 0..vars.old.len() {
            let original_var = vars.old[original_idx];

            let original_loop_output_var =
                loop_output_vars.as_ref().map(|vars| vars.old[original_idx]);

            let mut maybe_canon_def = None;
            let Some(def_law) = self.legalizer.maybe_def_law(
                func.reborrow().freeze().at(Value::Var(original_var)),
                || {
                    if maybe_canon_def.is_none() {
                        maybe_canon_def = self.canon_ptrs.get(&original_var);
                    }
                    maybe_canon_def.map_or(QPtrDefLaw::Unsupported, |canon| canon.def_law)
                },
            ) else {
                continue;
            };

            // HACK(eddyb) sanity-checking the loop output `CanonPtr`, if it exists.
            let mut maybe_loop_output_canon_def = None;
            if let Some(original_loop_output_var) = original_loop_output_var {
                match def_law {
                    DefLaw::QPtr(QPtrDefLaw::Dyn) => {
                        maybe_loop_output_canon_def =
                            self.canon_ptrs.get(&original_loop_output_var);

                        let canon_def = maybe_canon_def.unwrap();
                        let loop_output_canon_def = maybe_loop_output_canon_def.unwrap();

                        assert!(matches!(loop_output_canon_def.def_law, QPtrDefLaw::Dyn));

                        match (&canon_def.base, &loop_output_canon_def.base) {
                            (&BaseChoice::Single(a), &BaseChoice::Single(b)) => assert!(a == b),
                            (
                                BaseChoice::Many { base_index_bitset: a, base_index_selector: _ },
                                BaseChoice::Many { base_index_bitset: b, base_index_selector: _ },
                            ) => assert_eq!(a, b),
                            _ => unreachable!(),
                        }
                        match (&canon_def.offset, &loop_output_canon_def.offset) {
                            (Offset::Zero, Offset::Zero) => {}
                            (
                                &Offset::Strided { stride: stride_a, index: index_a },
                                &Offset::Strided { stride: stride_b, index: index_b },
                            ) => {
                                assert_eq!(stride_a, stride_b);
                                assert!(
                                    func.vars[index_a.expect_def()].ty
                                        == func.vars[index_b.expect_def()].ty
                                );
                            }
                            _ => unreachable!(),
                        }
                    }
                    DefLaw::QPtr(QPtrDefLaw::Unsupported) => {}
                    _ => unreachable!(),
                }
            }

            match def_law {
                DefLaw::QPtr(QPtrDefLaw::Base(_)) => {}
                DefLaw::QPtr(QPtrDefLaw::Offset(new_offset)) => {
                    let canon_def = maybe_canon_def.unwrap();

                    let (node, parent_region) = def_parent.right().unwrap();

                    let old_offset = self.use_canonical_ptr(func.nodes[node].inputs[0]).offset;

                    // NOTE(eddyb) "upstream" as in "can define values later used by `node`".
                    let mut upstream_cursor = func
                        .reborrow()
                        .at(InsertPosition { parent_region, anchor: InsertAnchor::Before(node) });

                    self.legalize_deps_of_base_if_single(canon_def, &mut upstream_cursor);

                    self.legalize_offset_sum(
                        canon_def.offset,
                        old_offset,
                        new_offset,
                        &mut upstream_cursor,
                    );

                    let new_kind_and_inputs = match canon_def.base {
                        BaseChoice::Single(_) => {
                            self.single_base_canon_ptr_to_node_kind_and_inputs(canon_def)
                        }
                        BaseChoice::Many { .. } => {
                            func.regions[parent_region].children.remove(node, func.nodes);

                            // HACK(eddyb) no good "tombstone" for the original def.
                            vars.replace_old_with_new(original_idx, []);
                            (
                                NodeKind::SpvInst(
                                    self.legalizer.wk.OpNop.into(),
                                    spv::InstLowering::default(),
                                ),
                                [].into_iter().collect(),
                            )
                        }
                    };
                    let node_def = func.reborrow().at(node).def();
                    (node_def.kind, node_def.inputs) = new_kind_and_inputs;
                }
                DefLaw::QPtr(QPtrDefLaw::Dyn) => {
                    let mut legalize_def =
                        |original_var: Var,
                         canon_def: &CanonPtr,
                         vars: &mut TransformSmallVec<Var, _>,
                         downstream: &mut InsertPosition| {
                            let vars_for_base = match canon_def.base {
                                BaseChoice::Single(_) => None,
                                BaseChoice::Many { base_index_bitset: _, base_index_selector } => {
                                    Some(base_index_selector.expect_def())
                                }
                            };

                            let vars_for_offset = match canon_def.offset {
                                Offset::Zero => None,
                                Offset::Strided { stride: _, index } => Some(index.expect_def()),
                            };

                            vars.replace_old_with_new(
                                original_idx,
                                vars_for_base.into_iter().chain(vars_for_offset),
                            );

                            match canon_def.base {
                                BaseChoice::Single(_) => {
                                    let mut downstream_cursor = func.reborrow().at(*downstream);

                                    self.legalize_deps_of_base_if_single(
                                        canon_def,
                                        &mut downstream_cursor,
                                    );

                                    downstream_cursor.insert_node(cx, {
                                        let (kind, inputs) = self
                                            .single_base_canon_ptr_to_node_kind_and_inputs(
                                                canon_def,
                                            );
                                        NodeDef {
                                            // FIXME(eddyb) propagate at least debuginfo attrs from somewhere.
                                            attrs: Default::default(),
                                            kind,
                                            inputs,
                                            child_regions: [].into_iter().collect(),
                                            outputs: [original_var].into_iter().collect(),
                                        }
                                    });

                                    *downstream = downstream_cursor.position;
                                }
                                BaseChoice::Many { .. } => {}
                            }
                        };

                    let canon_def = maybe_canon_def.unwrap();
                    legalize_def(original_var, canon_def, &mut vars, &mut downstream);
                    if let Some(loop_output_canon_def) = maybe_loop_output_canon_def {
                        legalize_def(
                            original_loop_output_var.unwrap(),
                            loop_output_canon_def,
                            loop_output_vars.as_mut().unwrap(),
                            downstream_of_loop.as_mut().unwrap(),
                        );
                    }

                    let expected_offset_shape =
                        canon_def.offset.map_index_value(|index| func.vars[index.expect_def()].ty);
                    for dyn_sink in &mut dyn_sinks {
                        self.legalize_in_dyn_sink(
                            func.reborrow(),
                            dyn_sink,
                            original_idx,
                            &canon_def.base,
                            expected_offset_shape,
                        );
                    }
                }
                DefLaw::CmpQPtrEqOrNe { not_eq } => {
                    let (node, parent_region) = def_parent.right().unwrap();

                    let bool_ty = func.vars[original_var].ty;
                    let cmp_op = if not_eq { scalar::IntBinOp::Ne } else { scalar::IntBinOp::Eq };
                    let inputs = {
                        let inputs = &func.nodes[node].inputs;
                        [inputs[0], inputs[1]]
                    };

                    // HACK(eddyb) an unsupported pointer input doesn't stop
                    // `maybe_def_law` from returning `DefLaw::CmpQPtrEqOrNe`.
                    let [Ok(a), Ok(b)] = inputs.map(|ptr| self.try_use_canonical_ptr(ptr)) else {
                        continue;
                    };

                    // FIXME(eddyb) this is inefficient, but thankfully it's only
                    // non-trivially populated if an error would be emitted - at most,
                    // it might benefit from some kind of "small set" per-alias-group.
                    let ambiguous_bases = {
                        let mut aliasable_bases: FxIndexMap<AliasGroup, FxIndexSet<Base>> =
                            FxIndexMap::default();

                        for canon_ptr in [&a, &b] {
                            for (_, base) in
                                canon_ptr.base.iter_with_selector_value(&self.func_base_map).1
                            {
                                if let Some(alias_group) = self.legalizer.base_alias_group(base) {
                                    aliasable_bases.entry(alias_group).or_default().insert(base);
                                }
                            }
                        }

                        aliasable_bases.retain(|_, bases| bases.len() > 1);
                        aliasable_bases
                    };

                    // Only attach a diagnostic, if there are any alias groups
                    // that could introduce false negatives wrt pointer equality,
                    // and bail out early to avoid causing an unsound legalization.
                    if !ambiguous_bases.is_empty() {
                        let attrs = &mut func.nodes[node].attrs;
                        for (_alias_group, bases) in ambiguous_bases {
                            // FIXME(eddyb) include the `AliasGroup` into this, perhaps?
                            // (but `WorkgroupSingletonMemory` doesn't need to exist,
                            // so really there's only one plausible situation for now)
                            let mut err = Diag::err([
                                "ambiguous pointer comparison, these may overlap: ".into(),
                            ]);
                            for (i, base) in bases.into_iter().enumerate() {
                                // HACK(eddyb) mimics `base_alias_group`'s logic.
                                let ptr_to_binding_or_buffer = match base {
                                    Base::Global(GlobalBase::Undef | GlobalBase::Null)
                                    | Base::FuncLocal(_) => {
                                        unreachable!()
                                    }

                                    Base::Global(
                                        GlobalBase::GlobalVar { ptr_to_binding: ptr }
                                        | GlobalBase::BufferData { ptr_to_buffer: ptr },
                                    ) => ptr,
                                };

                                if i > 0 {
                                    err.message.push(", ".into());
                                }
                                err.message.extend([
                                    "`".into(),
                                    ptr_to_binding_or_buffer.into(),
                                    "`".into(),
                                ]);
                            }
                            attrs.push_diag(cx, err);
                        }
                        continue;
                    }

                    // NOTE(eddyb) "upstream" as in "can define values later used by `node`".
                    let mut upstream_cursor = func
                        .reborrow()
                        .at(InsertPosition { parent_region, anchor: InsertAnchor::Before(node) });

                    let base_cmp = match (&a.base, &b.base) {
                        (BaseChoice::Single(a), BaseChoice::Single(b)) => Either::Left(a == b),
                        _ => Either::Right(
                            [&a, &b].map(|canon| self.materialize_base_index_selector(&canon.base)),
                        ),
                    };

                    let offsets = {
                        let [a, b] = [a.offset, b.offset].map(|offset| {
                            offset.map_index_value(|v| {
                                (upstream_cursor.reborrow().freeze().at(v).type_of(cx), v)
                            })
                        });
                        a.merge(
                            b,
                            |(ty, a)| (ty, Some(a), None),
                            |(ty, b)| (ty, None, Some(b)),
                            |a, b| {
                                let [a_ty, b_ty] = [a, b].map(|v| v.map_value(|(ty, _)| ty));
                                let [a, b] = [a, b].map(|v| v.map_value(|(_, v)| v));
                                let merged_ty = self.legalizer.merge_scaled_index_types(a_ty, b_ty);

                                // TODO(eddyb) implement index widening.
                                assert!(a_ty.value == merged_ty);
                                assert!(b_ty.value == merged_ty);

                                (
                                    merged_ty,
                                    Some(self.materialize_scaled_index(a, &mut upstream_cursor)),
                                    Some(self.materialize_scaled_index(b, &mut upstream_cursor)),
                                )
                            },
                        )
                    };
                    let offset_cmp = match offsets {
                        Offset::Zero => Either::Left(true),
                        Offset::Strided { stride: _, index: (index_ty, a_index, b_index) } => {
                            Either::Right([a_index, b_index].map(|index| {
                                index.unwrap_or_else(|| {
                                    // FIXME(eddyb) deduplicate this with other instances.
                                    // HACK(eddyb) not using `cx.intern(v)` to preserve `index_ty`
                                    // (which may have e.g. error diagnostic attributes).
                                    Value::Const(
                                        cx.intern(ConstDef {
                                            attrs: Default::default(),
                                            ty: index_ty,
                                            kind: scalar::Const::from_bits(
                                                index_ty.as_scalar(cx).unwrap(),
                                                0,
                                            )
                                            .into(),
                                        }),
                                    )
                                })
                            }))
                        }
                    };

                    // FIXME(eddyb) consider declaring a custom `enum` for this.
                    let [base_cmp_result, offset_cmp_result] =
                        [base_cmp, offset_cmp].map(|cmp| match cmp {
                            Either::Left(is_eq) => {
                                Value::Const(cx.intern(scalar::Const::from_bool(is_eq ^ not_eq)))
                            }
                            Either::Right([a, b]) => {
                                let output_var = upstream_cursor.decl_var(cx, bool_ty);
                                upstream_cursor.insert_node(
                                    cx,
                                    NodeDef {
                                        // FIXME(eddyb) copy at least debuginfo attrs.
                                        attrs: Default::default(),
                                        kind: scalar::Op::IntBinary(cmp_op).into(),
                                        inputs: [a, b].into_iter().collect(),
                                        child_regions: [].into_iter().collect(),
                                        outputs: [output_var].into_iter().collect(),
                                    },
                                );
                                Value::Var(output_var)
                            }
                        });

                    let node_def = func.reborrow().at(node).def();
                    node_def.kind = scalar::Op::BoolBinary(if not_eq {
                        scalar::BoolBinOp::Or
                    } else {
                        scalar::BoolBinOp::And
                    })
                    .into();
                    node_def.inputs = [base_cmp_result, offset_cmp_result].into_iter().collect();
                }
                // TODO(eddyb) !!! UNSOUNDNESS HAZARD !!! eager decoding could
                // ever be sound *iff* all encoded pointers are *guaranteed*
                // to be in-bounds! if not (e.g. `p.wrapping_sub(i)`) encoding
                // still produces some address that is the right distance from
                // the base address of its intended base, but only arithmetic on
                // the whole address can be used to get it back in-bounds before
                // any non-UB access - i.e. `qptr.offset`'s split `(base, index)`
                // would corrupt it, and decoding should be done at accesses.
                //
                // Complete example:
                // ```rust
                // let mut x: u32 = 123;
                // *((&mut x as *mut u32).wrapping_sub(1) as usize as *mut u32).add(1) += 1;
                // ```
                // (the current implementation *will not* write to `x`, but either
                // introduce UB where none existed, or write somewhere else instead)
                DefLaw::QPtr(QPtrDefLaw::DecodeEscaped) => {
                    let canon_def = maybe_canon_def.unwrap();

                    let (node, parent_region) = def_parent.right().unwrap();

                    let encoded = func.nodes[node].inputs[0];
                    let encoded_ty = func.reborrow().freeze().at(encoded).type_of(cx);
                    let encoded_scalar_ty = encoded_ty.as_scalar(cx).unwrap();

                    // FIXME(eddyb) deduplicate the logic here, even if simple,
                    // maybe by creating a type for "encoding scheme".
                    let encoded_base_bit_width = self.escaped_base_map.bits_needed_for_base_idx();
                    let encoded_offset_bit_width = encoded_scalar_ty
                        .bit_width()
                        .checked_sub(encoded_base_bit_width)
                        .unwrap_or_else(|| {
                            // HACK(eddyb) this doesn't have to worry about any
                            // diagnostic, as encoding will have emitted them,
                            // and this `0` will be used in `encoded >> 0` for
                            // `base_index_selector`, then `encoded >> 0 << 0`
                            // will be subtracted from `encoded`, correctly
                            // producing an offset of `0` (as useless as that is).
                            // TODO(eddyb) wouldn't this failure mean that there
                            // are e.g. 4 billion bases in `escaped_base_map`???
                            if true {
                                unreachable!(
                                    "a choice among {} bases can't be \
                                     represented in a {}-bit integer",
                                    self.escaped_base_map.bases.len(),
                                    encoded_scalar_ty.bit_width()
                                )
                            }
                            0
                        });

                    // NOTE(eddyb) "upstream" as in "can define values later used by `node`".
                    let mut upstream_cursor = func
                        .reborrow()
                        .at(InsertPosition { parent_region, anchor: InsertAnchor::Before(node) });

                    let encoded_offset_bit_width =
                        Value::Const(cx.intern(scalar::Const::from_u32(encoded_offset_bit_width)));

                    // HACK(eddyb) using `(x >> N, (x - (x >> N << N)) / S)`,
                    // instead of bit masking, so that re-encoding has a chance
                    // to simplify `((x >> N) << N) + ((x - (x >> N << N)) / S) * S`
                    // to `(x >> N << N) + (x - (x >> N << N))`, i.e. just `x`
                    // (where `N` is the base shift and `S` is the common stride,
                    // though simplifying away `(o/S)*S` would require *knowing*
                    // `o % S == 0`, i.e. assuming that the input of decoding is
                    // always a multiple of the common escaped pointer stride `S`,
                    // and not supporting `p |> ptr2int |> f |> int2ptr` when
                    // `f(x) != x` without losing provenance - this is not unlike
                    // e.g. CHERI, and so should be semantically compatible with
                    // at least "strict provenance" Rust, but without any checks).

                    let byte_offset = match canon_def.base {
                        BaseChoice::Single(base) => {
                            assert_eq!(encoded_base_bit_width, 0);
                            assert!(
                                (self.escaped_base_map.bases.keys().copied().exactly_one().ok())
                                    .map(Base::Global)
                                    == Some(base)
                            );
                            encoded
                        }
                        BaseChoice::Many { base_index_bitset: _, base_index_selector } => {
                            assert_ne!(encoded_base_bit_width, 0);

                            let base_index_selector = base_index_selector.expect_def();

                            // TODO(eddyb) implement integer widening.
                            assert!(upstream_cursor.vars[base_index_selector].ty == encoded_ty);

                            upstream_cursor.insert_node(
                                cx,
                                NodeDef {
                                    // FIXME(eddyb) copy at least debuginfo attrs.
                                    attrs: Default::default(),
                                    kind: scalar::Op::IntBinary(scalar::IntBinOp::ShrU).into(),
                                    inputs: [encoded, encoded_offset_bit_width]
                                        .into_iter()
                                        .collect(),
                                    child_regions: [].into_iter().collect(),
                                    outputs: [base_index_selector].into_iter().collect(),
                                },
                            );

                            let base_encoded_start = upstream_cursor.decl_var(cx, encoded_ty);
                            upstream_cursor.insert_node(
                                cx,
                                NodeDef {
                                    // FIXME(eddyb) copy at least debuginfo attrs.
                                    attrs: Default::default(),
                                    kind: scalar::Op::IntBinary(scalar::IntBinOp::Shl).into(),
                                    inputs: [
                                        Value::Var(base_index_selector),
                                        encoded_offset_bit_width,
                                    ]
                                    .into_iter()
                                    .collect(),
                                    child_regions: [].into_iter().collect(),
                                    outputs: [base_encoded_start].into_iter().collect(),
                                },
                            );

                            let byte_offset = upstream_cursor.decl_var(cx, encoded_ty);
                            upstream_cursor.insert_node(
                                cx,
                                NodeDef {
                                    // FIXME(eddyb) copy at least debuginfo attrs.
                                    attrs: Default::default(),
                                    kind: scalar::Op::IntBinary(scalar::IntBinOp::Sub).into(),
                                    inputs: [encoded, Value::Var(base_encoded_start)]
                                        .into_iter()
                                        .collect(),
                                    child_regions: [].into_iter().collect(),
                                    outputs: [byte_offset].into_iter().collect(),
                                },
                            );
                            Value::Var(byte_offset)
                        }
                    };

                    let new_kind_and_inputs = match canon_def.offset {
                        Offset::Zero => {
                            func.regions[parent_region].children.remove(node, func.nodes);

                            // HACK(eddyb) no good "tombstone" for the original def.
                            vars.replace_old_with_new(original_idx, []);
                            (
                                NodeKind::SpvInst(
                                    self.legalizer.wk.OpNop.into(),
                                    spv::InstLowering::default(),
                                ),
                                [].into_iter().collect(),
                            )
                        }
                        Offset::Strided { stride, index } => {
                            let index_var = index.expect_def();

                            // TODO(eddyb) implement integer widening.
                            assert!(func.vars[index_var].ty == encoded_ty);

                            // FIXME(eddyb) confirm that scanning guarantees
                            // this would never panic.
                            let stride = Value::Const(
                                cx.intern(
                                    scalar::Const::int_try_from_i128(
                                        encoded_scalar_ty,
                                        stride.get().into(),
                                    )
                                    .unwrap(),
                                ),
                            );

                            vars.replace_old_with_new(original_idx, [index_var]);
                            (
                                scalar::Op::IntBinary(scalar::IntBinOp::DivU).into(),
                                [byte_offset, stride].into_iter().collect(),
                            )
                        }
                    };
                    let node_def = func.reborrow().at(node).def();
                    (node_def.kind, node_def.inputs) = new_kind_and_inputs;
                }
                DefLaw::EncodeEscapingQPtr => {
                    let (node, parent_region) = def_parent.right().unwrap();

                    // HACK(eddyb) an unsupported pointer input doesn't stop
                    // `maybe_def_law` from returning `DefLaw::EncodeEscapingQPtr`.
                    let Ok(canon_ptr) = self.try_use_canonical_ptr(func.nodes[node].inputs[0])
                    else {
                        continue;
                    };

                    let encoded_ty = func.vars[original_var].ty;
                    let encoded_scalar_ty = encoded_ty.as_scalar(cx).unwrap();

                    // First, validate that all input bases are representable.
                    //
                    // FIXME(eddyb) move this to happen on the set of escaping
                    // bases, and ideally generate runtime panics for buffers etc.
                    let mut any_errors = false;
                    let mut push_diag = |diag| {
                        func.nodes[node].attrs.push_diag(cx, diag);
                        any_errors = true;
                    };
                    // FIXME(eddyb) deduplicate the logic here, even if simple,
                    // maybe by creating a type for "encoding scheme".
                    let encoded_base_bit_width = self.escaped_base_map.bits_needed_for_base_idx();
                    let encoded_offset_bit_width = encoded_scalar_ty
                        .bit_width()
                        .checked_sub(encoded_base_bit_width)
                        .unwrap_or_else(|| {
                            push_diag(Diag::bug([format!(
                                "cannot encode a choice among {} bases using a {}-bit integer",
                                self.escaped_base_map.bases.len(),
                                encoded_scalar_ty.bit_width()
                            )
                            .into()]));
                            0
                        });
                    let max_supported_encoded_offset = (!0u32)
                        .checked_shr(u32::checked_sub(32, encoded_offset_bit_width).unwrap_or(0))
                        .unwrap_or(0);
                    for (base_idx, base) in
                        canon_ptr.base.iter_with_selector_value(&self.func_base_map).1
                    {
                        let base = match base {
                            // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
                            Base::Global(GlobalBase::Undef) => continue,

                            // TODO(eddyb) support by lifting the local to a new
                            // (`Private`) global post-scanning, and rewriting
                            // all the uses of the `FuncLocal`, which is actually
                            // super easy now that all escaping bases do that.
                            Base::FuncLocal(_) => {
                                push_diag(Diag::bug(["NYI: encoding pointer to local".into()]));
                                continue;
                            }
                            Base::Global(base) => base,
                        };

                        assert!(
                            base_idx
                                .map(|i| i as usize)
                                .or_else(|| {
                                    self.func_base_map
                                        .many_choice_bases
                                        .get_index_of(&Base::Global(base))
                                })
                                .and_then(|base_idx| {
                                    let (&k, _) =
                                        self.escaped_base_map.bases.get_index(base_idx)?;
                                    Some(k)
                                })
                                == Some(base)
                        );

                        let ptr_to_binding_or_buffer = match base {
                            GlobalBase::Undef | GlobalBase::Null => continue,

                            GlobalBase::GlobalVar { ptr_to_binding: ptr }
                            | GlobalBase::BufferData { ptr_to_buffer: ptr } => ptr,
                        };
                        let ConstKind::PtrToGlobalVar { global_var, offset: None } =
                            cx[ptr_to_binding_or_buffer].kind
                        else {
                            unreachable!()
                        };

                        let gv_shape = self.module_global_vars[global_var].shape.unwrap();

                        use crate::mem::shapes::{GlobalVarShape, Handle};
                        let base_max_byte_size = match (gv_shape, base) {
                            // HACK(eddyb) rule these out first for exhaustiveness.
                            (_, GlobalBase::Undef | GlobalBase::Null) => unreachable!(),

                            (
                                GlobalVarShape::UntypedData(mem_layout),
                                GlobalBase::GlobalVar { .. },
                            ) => Ok(Some(mem_layout.size)),

                            (GlobalVarShape::Handles { .. }, GlobalBase::GlobalVar { .. }) => {
                                Err("NYI: encoding pointer to handles".into())
                            }
                            // FIXME(eddyb) this should still track a layout!
                            (GlobalVarShape::TypedInterface(_), GlobalBase::GlobalVar { .. }) => {
                                Err("NYI: encoding pointer to typed binding".into())
                            }

                            (
                                GlobalVarShape::Handles {
                                    handle: Handle::Buffer(_, layout),
                                    fixed_count,
                                },
                                GlobalBase::BufferData { .. },
                            ) => {
                                if fixed_count == Some(NonZeroU32::new(1).unwrap()) {
                                    Ok(layout
                                        .dyn_unit_stride
                                        .is_none()
                                        .then_some(layout.fixed_base.size))
                                } else {
                                    // FIXME(eddyb) this doesn't actually apply
                                    // to handle indexing (not supported at all),
                                    // but `qptr.buffer_data` *directly*, which
                                    // might or might not be the same as doing
                                    // handle indexing with index `0`?
                                    Err("NYI: encoding pointer derived from a handle array".into())
                                }
                            }
                            (_, GlobalBase::BufferData { .. }) => {
                                Err("misuse of `qptr.buffer_data` on non-buffer".into())
                            }
                        };
                        let valid = base_max_byte_size.and_then(|base_max_byte_size| {
                            match base_max_byte_size {
                                // TODO(eddyb) implement by injecting runtime
                                // panics at entry-points to limit the size.
                                None if true => Ok(()),
                                None => Err("NYI: unbounded size buffer".into()),

                                // NOTE(eddyb) `>` instead of `>=`, to support
                                // one-past-the-end pointers, too, just in case.
                                Some(byte_size) if byte_size > max_supported_encoded_offset => {
                                    Err(format!(
                                        "cannot encode with offsets up to {byte_size} bytes \
                                         ({max_supported_encoded_offset} bytes is the maximum)"
                                    )
                                    .into())
                                }
                                Some(_) => Ok(()),
                            }
                        });
                        if let Err(cause) = valid {
                            push_diag(Diag::bug([
                                cause,
                                ": `".into(),
                                ptr_to_binding_or_buffer.into(),
                                "`".into(),
                            ]));
                        }
                    }
                    if any_errors {
                        continue;
                    }

                    // NOTE(eddyb) "upstream" as in "can define values later used by `node`".
                    let mut upstream_cursor = func
                        .reborrow()
                        .at(InsertPosition { parent_region, anchor: InsertAnchor::Before(node) });

                    let byte_offset_shape =
                        Offset::Strided { stride: NonZeroU32::new(1).unwrap(), index: encoded_ty };
                    // FIXME(eddyb) deduplicate this with other instances.
                    let byte_offset = byte_offset_shape.merge(
                        canon_ptr.offset,
                        |expected_index_ty| {
                            // HACK(eddyb) not using `cx.intern(v)` to preserve `index_ty`
                            // (which may have e.g. error diagnostic attributes).
                            Value::Const(
                                cx.intern(ConstDef {
                                    attrs: Default::default(),
                                    ty: expected_index_ty,
                                    kind: scalar::Const::from_bits(
                                        // TODO(eddyb) inefficient, left as-is for dedup.
                                        expected_index_ty.as_scalar(cx).unwrap(),
                                        0,
                                    )
                                    .into(),
                                }),
                            )
                        },
                        |_| unreachable!(),
                        |shape, index| {
                            assert_eq!(shape.multiplier.get(), 1);
                            self.materialize_scaled_index(index, &mut upstream_cursor)
                        },
                    );
                    let Offset::Strided { stride: _, index: byte_offset } = byte_offset else {
                        unreachable!();
                    };

                    // FIXME(eddyb) maybe bypass this if `encoded_base_bit_width == 0`?
                    let base_index_selector = self.materialize_base_index_selector(&canon_ptr.base);

                    // TODO(eddyb) implement integer widening.
                    {
                        let func = upstream_cursor.reborrow().freeze().at(());
                        assert!(func.at(base_index_selector).type_of(cx) == encoded_ty);
                        assert!(func.at(byte_offset).type_of(cx) == encoded_ty);
                    }

                    // HACK(eddyb) avoid overlong shifts.
                    let base_encoded_start = if encoded_base_bit_width == 0 {
                        match canon_ptr.base {
                            // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
                            BaseChoice::Single(Base::Global(GlobalBase::Undef)) => {}

                            _ => {
                                // HACK(eddyb) there's only one thing this could be.
                                let zero_u32 = Value::Const(cx.intern(scalar::Const::from_u32(0)));
                                assert!(base_index_selector == zero_u32);
                            }
                        }
                        base_index_selector
                    } else {
                        let base_encoded_start = upstream_cursor.decl_var(cx, encoded_ty);
                        upstream_cursor.insert_node(
                            cx,
                            NodeDef {
                                // FIXME(eddyb) copy at least debuginfo attrs.
                                attrs: Default::default(),
                                kind: scalar::Op::IntBinary(scalar::IntBinOp::Shl).into(),
                                inputs: [
                                    base_index_selector,
                                    Value::Const(
                                        cx.intern(scalar::Const::from_u32(
                                            encoded_offset_bit_width,
                                        )),
                                    ),
                                ]
                                .into_iter()
                                .collect(),
                                child_regions: [].into_iter().collect(),
                                outputs: [base_encoded_start].into_iter().collect(),
                            },
                        );
                        Value::Var(base_encoded_start)
                    };

                    // HACK(eddyb) this uses addition, not bitwise OR, to combine
                    // the high (base) and low (offset) bits (see decode for why).
                    let node_def = func.reborrow().at(node).def();
                    node_def.kind = scalar::Op::IntBinary(scalar::IntBinOp::Add).into();
                    node_def.inputs = [base_encoded_start, byte_offset].into_iter().collect();
                }
                DefLaw::QPtr(QPtrDefLaw::Unsupported) => {
                    // HACK(eddyb) if `dyn_sinks` isn't empty (i.e. `Select`/`Loop`),
                    // matching valid `BaseChoice::Many` pointers must be cleared
                    // (should be sound as long as unsupported ones remain).
                    for DynSink { upstream: _, values } in &mut dyn_sinks {
                        if let Ok(canon) = self.try_use_canonical_ptr(values.old[original_idx])
                            && let BaseChoice::Many { .. } = canon.base
                        {
                            values.replace_old_with_new(
                                original_idx,
                                [Value::Const(self.legalizer.undef_qptr_const())],
                            );
                        }
                    }
                }
            }
        }

        // HACK(eddyb) only bother updating `VarDecl`s if any `Var`s changed.
        let mut update_vars =
            |vars: TransformSmallVec<Var, _>,
             def_parent: Either<(Region, Option<(Node, Region)>), (Node, Region)>| {
                let vars_changed = vars.used_old_range != ..0;
                let new_vars = vars.finish();
                if vars_changed {
                    for (new_idx, &new_var) in new_vars.iter().enumerate() {
                        let var_decl = &mut func.vars[new_var];
                        var_decl.def_parent = def_parent.map_either(|(r, _)| r, |(n, _)| n);
                        var_decl.def_idx = new_idx.try_into().unwrap();
                    }
                }
                *vars_defined_by(func.reborrow().at(def_parent)) = new_vars;
            };
        update_vars(vars, def_parent);
        if let Some(loop_output_vars) = loop_output_vars {
            update_vars(loop_output_vars, Either::Right(def_parent_loop.unwrap()));
        }

        for DynSink { upstream, values } in dyn_sinks {
            *dyn_sink_values_downstream_of(func.reborrow().at(upstream)) = values.finish();
        }
    }

    // FIXME(eddyb) this should have a better API (right now it just helps reuse).
    fn legalize_in_dyn_sink<V>(
        &self,
        mut func: FuncAtMut<'_, ()>,
        sink: &mut DynSink,
        original_idx_in_sink: usize,
        expected_base: &BaseChoice<V>,
        expected_offset_shape: Offset<Type>,
    ) {
        let cx = &self.legalizer.cx;

        let mut upstream_cursor = func.reborrow().at(sink.upstream);

        let found = self.use_canonical_ptr(sink.values.old[original_idx_in_sink]);

        let values_for_found_base = match expected_base {
            BaseChoice::Single(_) => None,
            BaseChoice::Many { .. } => Some(self.materialize_base_index_selector(&found.base)),
        };

        // FIXME(eddyb) deduplicate this with other instances.
        let scaled_found_offset = expected_offset_shape.merge(
            found.offset,
            |expected_index_ty| {
                // HACK(eddyb) not using `cx.intern(v)` to preserve `index_ty`
                // (which may have e.g. error diagnostic attributes).
                Value::Const(
                    cx.intern(ConstDef {
                        attrs: Default::default(),
                        ty: expected_index_ty,
                        kind: scalar::Const::from_bits(expected_index_ty.as_scalar(cx).unwrap(), 0)
                            .into(),
                    }),
                )
            },
            |_| unreachable!(),
            |expected, found| {
                assert_eq!(expected.multiplier.get(), 1);
                self.materialize_scaled_index(found, &mut upstream_cursor)
            },
        );
        let values_for_scaled_found_offset = match scaled_found_offset {
            Offset::Zero => None,
            Offset::Strided { stride: _, index } => Some(index),
        };
        sink.values.replace_old_with_new(
            original_idx_in_sink,
            values_for_found_base.into_iter().chain(values_for_scaled_found_offset),
        );

        sink.upstream = upstream_cursor.position;
    }

    fn single_base_canon_ptr_to_node_kind_and_inputs(
        &self,
        canon_ptr: &CanonPtr,
    ) -> (NodeKind, SmallVec<[Value; 2]>) {
        let BaseChoice::Single(base) = canon_ptr.base else { unreachable!() };

        let base_ptr = match base {
            Base::FuncLocal(local) => Value::Var(local.qptr_output),
            Base::Global(global_base) => match global_base {
                GlobalBase::Undef => Value::Const(self.legalizer.undef_qptr_const()),
                GlobalBase::Null => Value::Const(self.legalizer.null_qptr_const()),
                GlobalBase::GlobalVar { ptr_to_binding } => Value::Const(ptr_to_binding),
                GlobalBase::BufferData { .. } => {
                    canon_ptr.single_base_buffer_data.unwrap().to_use()
                }
            },
        };

        let (kind, maybe_index) = match canon_ptr.offset {
            // FIXME(eddyb) this should be avoided, but may show up in rare cases,
            // and generating `QPtrOp::Offset(0)` shouldn't hurt anything else.
            Offset::Zero => (QPtrOp::Offset(0), None),

            Offset::Strided { stride, index } => {
                let index = index.to_use();

                // HACK(eddyb) maybe `canon_ptr.offset` should be able to encode
                // constants offsets more naturally than this.
                let const_offset = match index {
                    Value::Const(index) => index.as_scalar(&self.legalizer.cx).and_then(|index| {
                        i32::try_from(index.int_as_u32()?.checked_mul(stride.get())?).ok()
                    }),
                    Value::Var(_) => None,
                };

                match const_offset {
                    Some(offset) => (QPtrOp::Offset(offset), None),
                    None => (
                        QPtrOp::DynOffset {
                            stride,
                            // FIXME(eddyb) this is lossy and can frustrate type recovery.
                            index_bounds: None,
                        },
                        Some(index),
                    ),
                }
            }
        };
        (kind.into(), [base_ptr].into_iter().chain(maybe_index).collect())
    }

    fn materialize_base_index_selector(&self, base: &BaseChoice<Value>) -> Value {
        match *base {
            // HACK(eddyb) `GlobalBase::Undef` ignored (see its comment).
            BaseChoice::Single(Base::Global(GlobalBase::Undef)) => {
                // FIXME(eddyb) consider caching this `undef`.
                Value::Const(self.legalizer.cx.intern(ConstDef {
                    attrs: Default::default(),
                    ty: self.legalizer.u32_type(),
                    kind: ConstKind::Undef,
                }))
            }

            // FIXME(eddyb) deduplicate this with other instances.
            BaseChoice::Single(base) => Value::Const(
                self.legalizer.cx.intern(scalar::Const::from_u32(
                    self.func_base_map
                        .many_choice_bases
                        .get_index_of(&base)
                        .unwrap()
                        .try_into()
                        .unwrap(),
                )),
            ),
            BaseChoice::Many { base_index_bitset: _, base_index_selector } => base_index_selector,
        }
    }

    // FIXME(eddyb) consider switching `DefOrUse::{Def -> Use}` after legalization.
    // FIXME(eddyb) avoid being wasteful with `BufferData`s and cache them near
    // the top of the function (or the "nearest common dominator"), or
    // completely revamp how buffers are even accessed in the first place.
    fn legalize_deps_of_base_if_single(
        &self,
        canon_ptr: &CanonPtr,
        cursor: &mut InsertionCursor<'_>,
    ) {
        match canon_ptr.base {
            BaseChoice::Single(Base::Global(GlobalBase::BufferData { ptr_to_buffer })) => {
                match canon_ptr.single_base_buffer_data.unwrap() {
                    DefOrUse::Def(buffer_data_output_var) => {
                        cursor.insert_node(
                            &self.legalizer.cx,
                            NodeDef {
                                // FIXME(eddyb) copy at least debuginfo attrs from somewhere.
                                attrs: Default::default(),
                                kind: QPtrOp::BufferData.into(),
                                inputs: [Value::Const(ptr_to_buffer)].into_iter().collect(),
                                child_regions: [].into_iter().collect(),
                                outputs: [buffer_data_output_var].into_iter().collect(),
                            },
                        );
                    }
                    DefOrUse::Use(_) => {}
                }
            }
            BaseChoice::Single(
                Base::FuncLocal(_)
                | Base::Global(GlobalBase::Undef | GlobalBase::Null | GlobalBase::GlobalVar { .. }),
            )
            | BaseChoice::Many { .. } => {}
        }
    }

    // FIXME(eddyb) consider switching `DefOrUse::{Def -> Use}` after legalization.
    fn legalize_offset_sum(
        &self,
        sum_output: Offset<DefOrUse>,
        lhs: Offset<Value>,
        rhs: Offset<IndexValue>,
        cursor: &mut InsertionCursor<'_>,
    ) {
        let cx = &self.legalizer.cx;

        let sum_inputs_or_one_sided =
            lhs.merge(rhs, |lhs| Err(IndexValue::Dyn(lhs)), Err, |lhs, rhs| Ok((lhs, rhs)));

        // HACK(eddyb) sanity-check the stride and def/use against `sum_output`.
        let sum = sum_output.merge(
            sum_inputs_or_one_sided,
            |_| unreachable!(),
            |_| unreachable!(),
            |Scaled { value: output, multiplier: mul_out },
             Scaled { value: inputs, multiplier: mul_in }| {
                assert_eq!((mul_in.get(), mul_out.get()), (1, 1));
                match (output, inputs) {
                    (DefOrUse::Use(expected), Err(IndexValue::Dyn(found))) => {
                        assert!(expected == found);
                        None
                    }
                    (
                        DefOrUse::Use(Value::Const(_)),
                        Err(IndexValue::One | IndexValue::MinusOne),
                    ) => None,
                    (_, Ok((lhs, rhs))) => Some((output.expect_def(), lhs, rhs)),
                    _ => unreachable!(),
                }
            },
        );

        let (sum_output, lhs, rhs) = match sum {
            Offset::Zero | Offset::Strided { stride: _, index: None } => return,
            Offset::Strided { stride: _, index: Some(sum) } => sum,
        };

        let lhs_ty = cursor.reborrow().freeze().at(lhs.value).type_of(cx);
        let index_ty = match rhs.value {
            IndexValue::Dyn(index_value) => self.legalizer.merge_scaled_index_types(
                lhs.map_value(|_| lhs_ty),
                rhs.map_value(|_| cursor.reborrow().freeze().at(index_value).type_of(cx)),
            ),
            IndexValue::One | IndexValue::MinusOne => lhs_ty,
        };
        let const_one =
            scalar::Const::int_try_from_i128(index_ty.as_scalar(cx).unwrap(), 1).unwrap();

        let add_or_sub = match rhs.value {
            IndexValue::Dyn(_) | IndexValue::One => scalar::IntBinOp::Add,
            IndexValue::MinusOne => scalar::IntBinOp::Sub,
        };
        let rhs = rhs.map_value(|rhs| match rhs {
            IndexValue::Dyn(rhs) => rhs,
            // HACK(eddyb) not using `cx.intern(const_one)` to preserve `index_ty`
            // (which may have e.g. error diagnostic attributes).
            IndexValue::One | IndexValue::MinusOne => Value::Const(cx.intern(ConstDef {
                attrs: Default::default(),
                ty: index_ty,
                kind: const_one.into(),
            })),
        });

        // FIXME(eddyb) copy at least debuginfo attrs from the original node.
        let mut attrs = AttrSet::default();

        // TODO(eddyb) implement index widening.
        if lhs_ty != index_ty {
            let [index_width, lhs_width] =
                [index_ty, lhs_ty].map(|ty| ty.as_scalar(cx).map(|ty| ty.bit_width()));

            // HACK(eddyb) addition and subtraction allow mixed signedness,
            // so that can be allowed to be emitted and should pose no issues,
            // but combining that with mixed width could result in ambiguities
            // between sign-extension and zero-extension, which could be UNSOUND
            // (see also: "UNSOUNDNESS HAZARD" warning on `QPtrDefLaw::DecodeEscaped`'s
            // implementation in `legalize_vars_defined_by`).
            let same_width = index_width.is_some() && index_width == lhs_width;

            if !same_width {
                let already_has_bug_diag = cx[index_ty]
                    .attrs
                    .diags(cx)
                    .iter()
                    .any(|diag| matches!(diag.level, DiagLevel::Bug(_)));
                if !already_has_bug_diag {
                    attrs.push_diag(
                        cx,
                        Diag::bug([
                            "NYI: widening `".into(),
                            lhs_ty.into(),
                            "` to `".into(),
                            index_ty.into(),
                            "`".into(),
                        ]),
                    );
                }
            }
        }

        let lhs = self.materialize_scaled_index(lhs, cursor);
        let rhs = self.materialize_scaled_index(rhs, cursor);

        cursor.insert_node(
            cx,
            NodeDef {
                attrs,
                kind: scalar::Op::IntBinary(add_or_sub).into(),
                inputs: [lhs, rhs].into_iter().collect(),
                child_regions: [].into_iter().collect(),
                outputs: [sum_output].into_iter().collect(),
            },
        );
    }

    // FIXME(eddyb) consider memoizing this (esp. for reusing int mul nodes).
    // FIXME(eddyb) deal with a mismatched desired (wider) index type.
    fn materialize_scaled_index(
        &self,
        Scaled { value: index, multiplier }: Scaled<Value>,
        cursor: &mut InsertionCursor<'_>,
    ) -> Value {
        let cx = &self.legalizer.cx;

        if multiplier.get() == 1 {
            return index;
        }

        let index_ty = cursor.reborrow().freeze().at(index).type_of(cx);
        let index_scalar_ty = index_ty.as_scalar(cx).unwrap();

        let try_index_ty_const_from_i128 = |v: i128| {
            let v = scalar::Const::int_try_from_i128(index_scalar_ty, v)?;
            // HACK(eddyb) not using `cx.intern(v)` to preserve `index_ty`
            // (which may have e.g. error diagnostic attributes).
            Some(Value::Const(cx.intern(ConstDef {
                attrs: Default::default(),
                ty: index_ty,
                kind: v.into(),
            })))
        };

        // HACK(eddyb) constant-folding failures fall back to runtime multiplication.
        let const_index = match index {
            Value::Const(ct) => ct.as_scalar(cx).copied(),
            Value::Var(_) => None,
        };
        let const_folded_result = const_index.and_then(|index| {
            let index = index.int_as_u128()?;

            try_index_ty_const_from_i128(
                index.checked_mul(multiplier.get().into())?.try_into().ok()?,
            )
        });
        if let Some(r) = const_folded_result {
            return r;
        }

        // HACK(eddyb) this can't fail thanks to similar checks during scanning.
        let multiplier = try_index_ty_const_from_i128(multiplier.get().into()).unwrap();

        let output_var = cursor.decl_var(cx, index_ty);
        cursor.insert_node(
            cx,
            NodeDef {
                // FIXME(eddyb) copy at least debuginfo attrs.
                attrs: Default::default(),
                kind: scalar::Op::IntBinary(scalar::IntBinOp::Mul).into(),
                inputs: [index, multiplier].into_iter().collect(),
                child_regions: [].into_iter().collect(),
                outputs: [output_var].into_iter().collect(),
            },
        );
        Value::Var(output_var)
    }
}

// FIXME(eddyb) does it make sense to even use `Transformer` here?
impl Transformer for LegalizePtrsInFunc<'_> {
    fn in_place_transform_func_decl(&mut self, func_decl: &mut FuncDecl) {
        let cx = &self.legalizer.cx;

        let is_qptr = |ty: Type| matches!(cx[ty].kind, TypeKind::QPtr);

        if let DeclDef::Present(func_def_body) = &mut func_decl.def {
            let body = func_def_body.body;

            self.legalize_vars_defined_by(func_def_body.at_mut(Either::Left((body, None))));
        }

        func_decl.inner_in_place_transform_with(self);

        if let DeclDef::Present(func_def_body) = &mut func_decl.def {
            let body = func_def_body.body;

            let mut body_outputs_dyn_sink = {
                let upstream = InsertPosition { parent_region: body, anchor: InsertAnchor::Last };
                DynSink {
                    upstream,
                    values: TransformSmallVec::new(mem::take(dyn_sink_values_downstream_of(
                        func_def_body.at_mut(upstream),
                    ))),
                }
            };
            for original_idx in 0..body_outputs_dyn_sink.values.old.len() {
                if is_qptr(
                    func_def_body.at(body_outputs_dyn_sink.values.old[original_idx]).type_of(cx),
                ) {
                    self.legalize_in_dyn_sink(
                        func_def_body.at_mut(()),
                        &mut body_outputs_dyn_sink,
                        original_idx,
                        &self.escaped_ptr_shape.0,
                        self.escaped_ptr_shape.1,
                    );
                }
            }
            {
                let DynSink { upstream, values } = body_outputs_dyn_sink;
                *dyn_sink_values_downstream_of(func_def_body.at_mut(upstream)) = values.finish();
            }

            // HACK(eddyb) it's easier to regenerate `FuncDecl` fields from the body.
            let body_def = func_def_body.at_body().def();
            func_decl.params = body_def
                .inputs
                .iter()
                .map(|&input_var| {
                    let input_decl = func_def_body.at(input_var).decl();
                    FuncParam { attrs: input_decl.attrs, ty: input_decl.ty }
                })
                .collect();
            func_decl.ret_types = body_def
                .outputs
                .iter()
                .map(|&output| func_def_body.at(output).type_of(cx))
                .collect();
        }
    }

    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_in_place_transform_with(self);
        self.parent_region = outer_region;
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        // Apply `QPtrExpansions` first, which can cause duplication (of leaf nodes),
        // but leaves behind the same kind(s) of nodes (with some replacements).
        let node_qptr_expansions =
            self.legalizer.node_qptr_expansions(func_at_node.reborrow().freeze());
        self.apply_node_qptr_expansions(func_at_node.reborrow(), node_qptr_expansions);

        // HACK(eddyb) doing this in the middle, instead of after handling vars,
        // lets child regions avoid accidentally handling the same vars twice.
        func_at_node.reborrow().inner_in_place_transform_with(self);

        let node = func_at_node.position;
        let mut func = func_at_node.at(());

        // HACK(eddyb) a bit like the general-purpose expansions, but simpler.
        if let NodeKind::FuncCall(_) | NodeKind::ThunkBind(_) = func.nodes[node].kind {
            let cx = &self.legalizer.cx;

            let is_qptr = |ty: Type| matches!(cx[ty].kind, TypeKind::QPtr);

            let mut dyn_sink = {
                let upstream = InsertPosition {
                    parent_region: self.parent_region.unwrap(),
                    anchor: InsertAnchor::Before(node),
                };
                DynSink {
                    upstream,
                    values: TransformSmallVec::new(mem::take(dyn_sink_values_downstream_of(
                        func.reborrow().at(upstream),
                    ))),
                }
            };
            for original_idx in 0..dyn_sink.values.old.len() {
                if is_qptr(
                    func.reborrow().freeze().at(dyn_sink.values.old[original_idx]).type_of(cx),
                ) {
                    self.legalize_in_dyn_sink(
                        func.reborrow(),
                        &mut dyn_sink,
                        original_idx,
                        &self.escaped_ptr_shape.0,
                        self.escaped_ptr_shape.1,
                    );
                }
            }
            {
                let DynSink { upstream, values } = dyn_sink;
                *dyn_sink_values_downstream_of(func.reborrow().at(upstream)) = values.finish();
            }
        }

        // Handle outputs (and therefore the node semantics tied to them),
        // relying on scanning finding them first (and preparing new `Var`s etc.).
        //
        // FIXME(eddyb) loop nodes are handled differently due to them not (yet)
        // exposing "loop state variables" through outputs (like RVSDG does),
        // and therefore having to focus on the loop body region inputs, instead.
        // FIXME(eddyb) even if the above comment is no longer accurate, it was
        // still easier to handle loops through their loop body input vars, even
        // if it may be possible now to swap it to focus on the loop outputs.
        let vars_def_parent = {
            let node_def = &mut func.nodes[node];
            if let NodeKind::Loop { .. } = node_def.kind {
                Either::Left((node_def.child_regions[0], Some(node)))
            } else {
                Either::Right(node)
            }
        };
        self.legalize_vars_defined_by(func.reborrow().at(vars_def_parent));

        // TODO(eddyb) if any node inputs remain pointing to multi canon, instantiate
        // (this is basically a kind of admission of failure for e.g. pointer comparison,
        // where at least one of the sides failed to be legalized)

        // Ensure that no `BaseChoice::Many` pointer `Var`s remain used directly,
        // even in the case of error (as they are permanently detached).
        let is_multi_base_qptr = |v: Value| match v {
            Value::Const(_) => false,
            Value::Var(v) => self
                .canon_ptrs
                .get(&v)
                .is_some_and(|canon| matches!(canon.base, BaseChoice::Many { .. })),
        };
        let node_def = &func.nodes[node];
        // TODO(eddyb) this triggers on e.g. atomic ops that output a pointer.
        if false {
            for &child_region in &node_def.child_regions {
                assert!(!func.regions[child_region].outputs.iter().any(|&v| is_multi_base_qptr(v)));
            }
        }
        match node_def.inputs.iter().filter(|&&v| is_multi_base_qptr(v)).count() {
            0 => {}

            // HACK(eddyb) only needed specifically for `DefLaw::CmpQPtrEqOrNe`,
            // which can fail to legalize if one of its two inputs is unsupported
            // (leaving it using the other input untouched, and that one could
            // be a now-dead `Var` that got canonicalized via `BaseChoice::Many`).
            1 => {
                self.apply_node_qptr_expansions(
                    func.at(node),
                    QPtrExpansions {
                        pre_encode_escaping_input: None,
                        post_decode_escaped_output: None,
                        instantiate_by_multi_base_qptr_inputs: true,
                    },
                );
            }

            // HACK(eddyb) this is only unreachable for now because the only
            // TODO(eddyb) above comment was never finished.
            _ => unreachable!(),
        }
    }
}
