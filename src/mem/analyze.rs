//! Memory access analysis (for "type recovery", i.e. untyped -> typed memory).
//
// TODO(eddyb) consider renaming this to `mem::typed`.

use crate::func_at::FuncAt;
use crate::mem::{DataHapp, DataHappKind, MemAccesses, MemAttr, MemOp, shapes};
use crate::qptr::{QPtrAttr, QPtrOp};
use crate::visit::{InnerVisit, Visitor};
use crate::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstKind, Context, DataInstKind, DeclDef, Diag,
    ExportKey, Exportee, Func, FxIndexMap, GlobalVar, Module, Node, NodeKind, OrdAssertEq, Type,
    TypeKind, Value, Var, VarKind,
};
use itertools::{Either, Itertools as _};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::mem;
use std::num::NonZeroU32;
use std::ops::Bound;
use std::rc::Rc;

// HACK(eddyb) sharing layout code with other modules.
// FIXME(eddyb) can this just be a non-glob import?
use crate::mem::layout::*;

#[derive(Clone)]
struct AnalysisError(Diag);

struct AccessMerger<'a> {
    layout_cache: &'a LayoutCache<'a>,
}

/// Result type for `AccessMerger` methods - unlike `Result<T, AnalysisError>`,
/// this always keeps the `T` value, even in the case of an error.
struct MergeResult<T> {
    merged: T,
    error: Option<AnalysisError>,
}

impl<T> MergeResult<T> {
    fn ok(merged: T) -> Self {
        Self { merged, error: None }
    }

    fn into_result(self) -> Result<T, AnalysisError> {
        let Self { merged, error } = self;
        match error {
            None => Ok(merged),
            Some(e) => Err(e),
        }
    }

    fn map<U>(self, f: impl FnOnce(T) -> U) -> MergeResult<U> {
        let Self { merged, error } = self;
        let merged = f(merged);
        MergeResult { merged, error }
    }
}

impl AccessMerger<'_> {
    fn merge(&self, a: MemAccesses, b: MemAccesses) -> MergeResult<MemAccesses> {
        match (a, b) {
            (
                MemAccesses::Handles(shapes::Handle::Opaque(a)),
                MemAccesses::Handles(shapes::Handle::Opaque(b)),
            ) if a == b => MergeResult::ok(MemAccesses::Handles(shapes::Handle::Opaque(a))),

            (
                MemAccesses::Handles(shapes::Handle::Buffer(a_as, a)),
                MemAccesses::Handles(shapes::Handle::Buffer(b_as, b)),
            ) => {
                // HACK(eddyb) the `AddrSpace` field is entirely redundant.
                assert!(a_as == AddrSpace::Handles && b_as == AddrSpace::Handles);

                self.merge_data(a, b).map(|happ| {
                    MemAccesses::Handles(shapes::Handle::Buffer(AddrSpace::Handles, happ))
                })
            }

            (MemAccesses::Data(a), MemAccesses::Data(b)) => {
                self.merge_data(a, b).map(MemAccesses::Data)
            }

            (a, b) => {
                MergeResult {
                    // FIXME(eddyb) there may be a better choice here, but it
                    // generally doesn't matter, as this method only has one
                    // caller, and it just calls `.into_result()` right away.
                    merged: a.clone(),
                    error: Some(AnalysisError(Diag::bug([
                        "merge: ".into(),
                        a.into(),
                        " vs ".into(),
                        b.into(),
                    ]))),
                }
            }
        }
    }

    fn merge_data(&self, a: DataHapp, b: DataHapp) -> MergeResult<DataHapp> {
        // NOTE(eddyb) this is doable because it's currently impossible for
        // the merged HAPP to be outside the bounds of *both* `a` and `b`.
        let max_size = match (a.max_size, b.max_size) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (None, _) | (_, None) => None,
        };

        // Ensure that `a` is "larger" than `b`, or at least the same size
        // (when either they're identical, or one is a "newtype" of the other),
        // to make it easier to handle all the possible interactions below,
        // by skipping (or deprioritizing, if supported) the "wrong direction".
        let mut sorted = [a, b];
        sorted.sort_by_key(|happ| {
            #[derive(PartialEq, Eq, PartialOrd, Ord)]
            enum MaxSize<T> {
                Fixed(T),
                // FIXME(eddyb) this probably needs to track "min size"?
                Dynamic,
            }
            let max_size = happ.max_size.map_or(MaxSize::Dynamic, MaxSize::Fixed);

            // When sizes are equal, pick the more restrictive side.
            #[derive(PartialEq, Eq, PartialOrd, Ord)]
            enum TypeStrictness {
                Any,
                Array,
                Exact,
            }
            #[allow(clippy::match_same_arms)]
            let type_strictness = match happ.kind {
                DataHappKind::Dead | DataHappKind::Disjoint(_) => TypeStrictness::Any,

                DataHappKind::Repeated { .. } => TypeStrictness::Array,

                // FIXME(eddyb) this should be `Any`, even if in theory it
                // could contain arrays or structs that need decomposition
                // (note that, for typed reads/write, arrays do not need to be
                // *indexed* to work, i.e. they *do not* require `DynOffset`s,
                // `Offset`s suffice, and for them `Repeated` is at most
                // a "run-length"/deduplication optimization over `Disjoint`).
                // NOTE(eddyb) this should still prefer `OpTypeVector` over `Repeated`!
                DataHappKind::Direct(_) => TypeStrictness::Exact,

                DataHappKind::StrictlyTyped(_) => TypeStrictness::Exact,
            };

            (max_size, type_strictness)
        });
        let [b, a] = sorted;
        assert_eq!(max_size, a.max_size);

        self.merge_data_at(a, 0, b)
    }

    // FIXME(eddyb) make the name of this clarify the asymmetric effect, something
    // like "make `a` compatible with `offset => b`".
    fn merge_data_at(&self, a: DataHapp, b_offset_in_a: u32, b: DataHapp) -> MergeResult<DataHapp> {
        // NOTE(eddyb) this is doable because it's currently impossible for
        // the merged HAPP to be outside the bounds of *both* `a` and `b`.
        let max_size = match (a.max_size, b.max_size) {
            (Some(a), Some(b)) => Some(a.max(b.checked_add(b_offset_in_a).unwrap())),
            (None, _) | (_, None) => None,
        };

        // HACK(eddyb) we require biased `a` vs `b` (see `merge_data` method above).
        assert_eq!(max_size, a.max_size);

        // Decompose the "smaller" and/or "less strict" side (`b`) first.
        match b.kind {
            // `Dead`s are always ignored.
            DataHappKind::Dead => return MergeResult::ok(a),

            DataHappKind::Disjoint(b_entries)
                if {
                    // HACK(eddyb) this check was added later, after it turned out
                    // that *deep* flattening of arbitrary offsets in `b` would've
                    // required constant-folding of `qptr.offset` in `qptr::lift`,
                    // to not need all the type nesting levels for `OpAccessChain`.
                    b_offset_in_a == 0
                } =>
            {
                // FIXME(eddyb) this whole dance only needed due to `Rc`.
                let b_entries = Rc::try_unwrap(b_entries);
                let b_entries = match b_entries {
                    Ok(entries) => Either::Left(entries.into_iter()),
                    Err(ref entries) => Either::Right(entries.iter().map(|(&k, v)| (k, v.clone()))),
                };

                let mut ab = a;
                let mut all_errors = None;
                for (b_offset, b_sub_happ) in b_entries {
                    let MergeResult { merged, error: new_error } = self.merge_data_at(
                        ab,
                        b_offset.checked_add(b_offset_in_a).unwrap(),
                        b_sub_happ,
                    );
                    ab = merged;

                    // FIXME(eddyb) move some of this into `MergeResult`!
                    if let Some(AnalysisError(e)) = new_error {
                        let all_errors =
                            &mut all_errors.get_or_insert(AnalysisError(Diag::bug([]))).0.message;
                        // FIXME(eddyb) should this mean `MergeResult` should
                        // use `errors: Vec<AnalysisError>` instead of `Option`?
                        if !all_errors.is_empty() {
                            all_errors.push("\n".into());
                        }
                        // FIXME(eddyb) this is scuffed because the error might
                        // (or really *should*) already refer to the right offset!
                        all_errors.push(format!("+{b_offset} => ").into());
                        all_errors.extend(e.message);
                    }
                }
                return MergeResult {
                    merged: ab,
                    // FIXME(eddyb) should this mean `MergeResult` should
                    // use `errors: Vec<AnalysisError>` instead of `Option`?
                    error: all_errors.map(|AnalysisError(mut e)| {
                        e.message.insert(0, "merge_data: conflicts:\n".into());
                        AnalysisError(e)
                    }),
                };
            }

            _ => {}
        }

        let kind = match a.kind {
            // `Dead`s are always ignored.
            DataHappKind::Dead => MergeResult::ok(b.kind),

            // Typed leaves must support any possible accesses applied to them
            // (when they match, or overtake, that access, in size, like here),
            // with their inherent hierarchy (i.e. their array/struct nesting).
            DataHappKind::StrictlyTyped(a_type) | DataHappKind::Direct(a_type) => {
                let b_type_at_offset_0 = match b.kind {
                    DataHappKind::StrictlyTyped(b_type) | DataHappKind::Direct(b_type)
                        if b_offset_in_a == 0 =>
                    {
                        Some(b_type)
                    }
                    _ => None,
                };
                let ty = if Some(a_type) == b_type_at_offset_0 {
                    MergeResult::ok(a_type)
                } else {
                    // Returns `Some(MergeResult::ok(ty))` iff `happ` is valid
                    // for type `ty`, and `None` iff invalid w/o layout errors
                    // (see `mem_layout_supports_happ_at_offset` for more details).
                    let type_supporting_happ_at_offset = |ty, happ_offset, happ| {
                        let supports_happ = match self.layout_of(ty) {
                            // FIXME(eddyb) should this be `unreachable!()`? also, is
                            // it possible to end up with `ty` being an `OpTypeStruct`
                            // decorated with `Block`, showing up as a `Buffer` handle?
                            //
                            // NOTE(eddyb) `Block`-annotated buffer types are *not*
                            // usable anywhere inside buffer data, since they would
                            // conflict with our own `Block`-annotated wrapper.
                            Ok(TypeLayout::Handle(_) | TypeLayout::HandleArray(..)) => {
                                Err(AnalysisError(Diag::bug([
                                    "merge_data: impossible handle type for DataHapp".into(),
                                ])))
                            }
                            Ok(TypeLayout::Concrete(concrete)) => {
                                Ok(concrete.supports_happ_at_offset(happ_offset, happ))
                            }

                            Err(e) => Err(e),
                        };
                        match supports_happ {
                            Ok(false) => None,
                            Ok(true) | Err(_) => {
                                Some(MergeResult { merged: ty, error: supports_happ.err() })
                            }
                        }
                    };

                    type_supporting_happ_at_offset(a_type, b_offset_in_a, &b)
                        .or_else(|| {
                            b_type_at_offset_0.and_then(|b_type_at_offset_0| {
                                type_supporting_happ_at_offset(b_type_at_offset_0, 0, &a)
                            })
                        })
                        .unwrap_or_else(|| {
                            MergeResult {
                                merged: a_type,
                                // FIXME(eddyb) this should ideally embed the types in the
                                // error somehow.
                                error: Some(AnalysisError(Diag::bug([
                                    "merge_data: type subcomponents incompatible with accesses ("
                                        .into(),
                                    MemAccesses::Data(a.clone()).into(),
                                    " vs ".into(),
                                    MemAccesses::Data(b.clone()).into(),
                                    ")".into(),
                                ]))),
                            }
                        })
                };

                // FIXME(eddyb) if the chosen (maybe-larger) side isn't strict,
                // it should also be possible to expand it into its components,
                // with the other (maybe-smaller) side becoming a leaf.

                // FIXME(eddyb) this might not enough because the
                // strict leaf could be *nested* inside `b`!!!
                let is_strict = |kind| matches!(kind, &DataHappKind::StrictlyTyped(_));
                if is_strict(&a.kind) || is_strict(&b.kind) {
                    ty.map(DataHappKind::StrictlyTyped)
                } else {
                    ty.map(DataHappKind::Direct)
                }
            }

            DataHappKind::Repeated { element: mut a_element, stride: a_stride } => {
                let b_offset_in_a_element = b_offset_in_a % a_stride;

                // Array-like dynamic offsetting needs to always merge any HAPP that
                // fits inside the stride, with its "element" HAPP, no matter how
                // complex it may be (notably, this is needed for nested arrays).
                if b.max_size
                    .and_then(|b_max_size| b_max_size.checked_add(b_offset_in_a_element))
                    .is_some_and(|b_in_a_max_size| b_in_a_max_size <= a_stride.get())
                {
                    // FIXME(eddyb) this in-place merging dance only needed due to `Rc`.
                    ({
                        let a_element_mut = Rc::make_mut(&mut a_element);
                        let a_element = mem::replace(a_element_mut, DataHapp::DEAD);
                        // FIXME(eddyb) remove this silliness by making `merge_data_at` do symmetrical sorting.
                        if b_offset_in_a_element == 0 {
                            self.merge_data(a_element, b)
                        } else {
                            self.merge_data_at(a_element, b_offset_in_a_element, b)
                        }
                        .map(|merged| *a_element_mut = merged)
                    })
                    .map(|()| DataHappKind::Repeated { element: a_element, stride: a_stride })
                } else {
                    match b.kind {
                        DataHappKind::Repeated { element: b_element, stride: b_stride }
                            if b_offset_in_a_element == 0 && a_stride == b_stride =>
                        {
                            // FIXME(eddyb) this in-place merging dance only needed due to `Rc`.
                            ({
                                let a_element_mut = Rc::make_mut(&mut a_element);
                                let a_element = mem::replace(a_element_mut, DataHapp::DEAD);
                                let b_element =
                                    Rc::try_unwrap(b_element).unwrap_or_else(|e| (*e).clone());
                                self.merge_data(a_element, b_element)
                                    .map(|merged| *a_element_mut = merged)
                            })
                            .map(|()| DataHappKind::Repeated {
                                element: a_element,
                                stride: a_stride,
                            })
                        }
                        _ => {
                            // FIXME(eddyb) implement somehow (by adjusting stride?).
                            // NOTE(eddyb) with `b` as an `Repeated`/`Disjoint`, it could
                            // also be possible to superimpose its offset patterns onto `a`,
                            // though that's easier for `Disjoint` than `Repeated`.
                            // HACK(eddyb) needed due to `a` being moved out of.
                            let a = DataHapp {
                                max_size: a.max_size,
                                kind: DataHappKind::Repeated {
                                    element: a_element,
                                    stride: a_stride,
                                },
                            };
                            MergeResult {
                                merged: a.kind.clone(),
                                error: Some(AnalysisError(Diag::bug([
                                    format!(
                                        "merge_data: unimplemented \
                                         non-intra-element merging into stride={a_stride} ("
                                    )
                                    .into(),
                                    MemAccesses::Data(a).into(),
                                    " vs ".into(),
                                    MemAccesses::Data(b).into(),
                                    ")".into(),
                                ]))),
                            }
                        }
                    }
                }
            }

            DataHappKind::Disjoint(mut a_entries) => {
                let overlapping_entries = a_entries
                    .range((
                        Bound::Unbounded,
                        b.max_size.map_or(Bound::Unbounded, |b_max_size| {
                            Bound::Excluded(b_offset_in_a.checked_add(b_max_size).unwrap())
                        }),
                    ))
                    .rev()
                    .take_while(|(a_sub_offset, a_sub_happ)| {
                        a_sub_happ.max_size.is_none_or(|a_sub_max_size| {
                            a_sub_offset.checked_add(a_sub_max_size).unwrap() > b_offset_in_a
                        })
                    });

                // FIXME(eddyb) this is a bit inefficient but we don't have
                // cursors, so we have to buffer the `BTreeMap` keys here.
                let overlapping_offsets: SmallVec<[u32; 16]> =
                    overlapping_entries.map(|(&a_sub_offset, _)| a_sub_offset).collect();
                let a_entries_mut = Rc::make_mut(&mut a_entries);
                let mut all_errors = None;
                let (mut b_offset_in_a, mut b) = (b_offset_in_a, b);
                for a_sub_offset in overlapping_offsets {
                    let a_sub_happ = a_entries_mut.remove(&a_sub_offset).unwrap();

                    // HACK(eddyb) this replicates the condition in which
                    // `merge_data_at` would fail its similar assert, some of
                    // the cases denied here might be legal, but they're rare
                    // enough that we can do this for now.
                    let is_illegal = a_sub_offset != b_offset_in_a && {
                        let (a_sub_total_max_size, b_total_max_size) = (
                            a_sub_happ.max_size.map(|a| a.checked_add(a_sub_offset).unwrap()),
                            b.max_size.map(|b| b.checked_add(b_offset_in_a).unwrap()),
                        );
                        let total_max_size_merged = match (a_sub_total_max_size, b_total_max_size) {
                            (Some(a), Some(b)) => Some(a.max(b)),
                            (None, _) | (_, None) => None,
                        };
                        total_max_size_merged
                            != if a_sub_offset < b_offset_in_a {
                                a_sub_total_max_size
                            } else {
                                b_total_max_size
                            }
                    };
                    if is_illegal {
                        // HACK(eddyb) needed due to `a` being moved out of.
                        let a = DataHapp {
                            max_size: a.max_size,
                            kind: DataHappKind::Disjoint(a_entries.clone()),
                        };
                        return MergeResult {
                            merged: DataHapp { max_size, kind: DataHappKind::Disjoint(a_entries) },
                            error: Some(AnalysisError(Diag::bug([
                                format!(
                                    "merge_data: unsupported straddling overlap \
                                     at offsets {a_sub_offset} vs {b_offset_in_a} ("
                                )
                                .into(),
                                MemAccesses::Data(a).into(),
                                " vs ".into(),
                                MemAccesses::Data(b).into(),
                                ")".into(),
                            ]))),
                        };
                    }

                    let new_error;
                    (b_offset_in_a, MergeResult { merged: b, error: new_error }) =
                        if a_sub_offset < b_offset_in_a {
                            (
                                a_sub_offset,
                                self.merge_data_at(a_sub_happ, b_offset_in_a - a_sub_offset, b),
                            )
                        } else {
                            // FIXME(eddyb) remove this silliness by making `merge_data_at` do symmetrical sorting.
                            if a_sub_offset - b_offset_in_a == 0 {
                                (b_offset_in_a, self.merge_data(b, a_sub_happ))
                            } else {
                                (
                                    b_offset_in_a,
                                    self.merge_data_at(b, a_sub_offset - b_offset_in_a, a_sub_happ),
                                )
                            }
                        };

                    // FIXME(eddyb) move some of this into `MergeResult`!
                    if let Some(AnalysisError(e)) = new_error {
                        let all_errors =
                            &mut all_errors.get_or_insert(AnalysisError(Diag::bug([]))).0.message;
                        // FIXME(eddyb) should this mean `MergeResult` should
                        // use `errors: Vec<AnalysisError>` instead of `Option`?
                        if !all_errors.is_empty() {
                            all_errors.push("\n".into());
                        }
                        // FIXME(eddyb) this is scuffed because the error might
                        // (or really *should*) already refer to the right offset!
                        all_errors.push(format!("+{a_sub_offset} => ").into());
                        all_errors.extend(e.message);
                    }
                }
                a_entries_mut.insert(b_offset_in_a, b);
                MergeResult {
                    merged: DataHappKind::Disjoint(a_entries),
                    // FIXME(eddyb) should this mean `MergeResult` should
                    // use `errors: Vec<AnalysisError>` instead of `Option`?
                    error: all_errors.map(|AnalysisError(mut e)| {
                        e.message.insert(0, "merge_data: conflicts:\n".into());
                        AnalysisError(e)
                    }),
                }
            }
        };
        kind.map(|kind| DataHapp { max_size, kind })
    }

    /// Attempt to compute a `TypeLayout` for a given (SPIR-V) `Type`.
    fn layout_of(&self, ty: Type) -> Result<TypeLayout, AnalysisError> {
        self.layout_cache.layout_of(ty).map_err(|LayoutError(err)| AnalysisError(err))
    }
}

impl MemTypeLayout {
    /// Determine if this layout is compatible with `happ` at `happ_offset`.
    ///
    /// That is, all typed leaves of `happ` must be found inside `self`, at
    /// their respective offsets, and all [`DataHappKind::Repeated`]s must
    /// find a same-stride array inside `self` (to allow dynamic indexing).
    //
    // FIXME(eddyb) consider using `Result` to make it unambiguous.
    fn supports_happ_at_offset(&self, happ_offset: u32, happ: &DataHapp) -> bool {
        if let DataHappKind::Dead = happ.kind {
            return true;
        }

        // "Fast accept" based on type alone (expected as recursion base case).
        if let DataHappKind::StrictlyTyped(happ_type) | DataHappKind::Direct(happ_type) = happ.kind
            && happ_offset == 0
            && self.original_type == happ_type
        {
            return true;
        }

        {
            // FIXME(eddyb) should `DataHapp` track a `min_size` as well?
            // FIXME(eddyb) duplicated below.
            let min_happ_offset_range =
                happ_offset..happ_offset.saturating_add(happ.max_size.unwrap_or(0));

            // "Fast reject" based on size alone (expected w/ multiple attempts).
            if self.mem_layout.dyn_unit_stride.is_none()
                && (self.mem_layout.fixed_base.size < min_happ_offset_range.end
                    || happ.max_size.is_none())
            {
                return false;
            }
        }

        let any_component_supports = |happ_offset: u32, happ: &DataHapp| {
            // FIXME(eddyb) should `DataHapp` track a `min_size` as well?
            // FIXME(eddyb) duplicated above.
            let min_happ_offset_range =
                happ_offset..happ_offset.saturating_add(happ.max_size.unwrap_or(0));

            // FIXME(eddyb) `find_components_containing` is linear today but
            // could be made logarithmic (via binary search).
            self.components.find_components_containing(min_happ_offset_range).any(|idx| match &self
                .components
            {
                Components::Scalar => unreachable!(),
                Components::Elements { stride, elem, .. } => {
                    elem.supports_happ_at_offset(happ_offset % stride.get(), happ)
                }
                Components::Fields { offsets, layouts, .. } => {
                    layouts[idx].supports_happ_at_offset(happ_offset - offsets[idx], happ)
                }
            })
        };
        match &happ.kind {
            _ if any_component_supports(happ_offset, happ) => true,

            DataHappKind::Dead => unreachable!(),

            DataHappKind::StrictlyTyped(_) | DataHappKind::Direct(_) => false,

            DataHappKind::Disjoint(entries) => {
                entries.iter().all(|(&sub_offset, sub_happ)| {
                    // FIXME(eddyb) maybe this overflow should be propagated up,
                    // as a sign that `happ` is malformed?
                    happ_offset.checked_add(sub_offset).is_some_and(|combined_offset| {
                        // NOTE(eddyb) the reason this is only applicable to
                        // offset `0` is that *in all other cases*, every
                        // individual `Disjoint` requires its own type, to
                        // allow performing offsets *in steps* (even if the
                        // offsets could easily be constant-folded, they'd
                        // *have to* be constant-folded *before* analysis,
                        // to ensure there is no need for the intermediaries).
                        if combined_offset == 0 {
                            self.supports_happ_at_offset(0, sub_happ)
                        } else {
                            any_component_supports(combined_offset, sub_happ)
                        }
                    })
                })
            }

            // Finding an array entirely nested in a component was handled above,
            // so here `layout` can only be a matching array (same stride and length).
            DataHappKind::Repeated { element: happ_elem, stride: happ_stride } => {
                let happ_fixed_len = happ
                    .max_size
                    .map(|size| {
                        if !size.is_multiple_of(happ_stride.get()) {
                            // FIXME(eddyb) maybe this should be propagated up,
                            // as a sign that `happ` is malformed?
                            return Err(());
                        }
                        NonZeroU32::new(size / happ_stride.get()).ok_or(())
                    })
                    .transpose();

                match &self.components {
                    // Dynamic offsetting into non-arrays is not supported, and it'd
                    // only make sense for legalization (or small-length arrays where
                    // selecting elements based on the index may be a practical choice).
                    Components::Scalar | Components::Fields { .. } => false,

                    Components::Elements {
                        stride: layout_stride,
                        elem: layout_elem,
                        fixed_len: layout_fixed_len,
                    } => {
                        // HACK(eddyb) extend the max length implied by `happ`,
                        // such that the array can start at offset `0`.
                        let ext_happ_offset = happ_offset % happ_stride.get();
                        let ext_happ_fixed_len = happ_fixed_len.and_then(|happ_fixed_len| {
                            happ_fixed_len
                                .map(|happ_fixed_len| {
                                    NonZeroU32::new(
                                        // FIXME(eddyb) maybe this overflow should be propagated up,
                                        // as a sign that `happ` is malformed?
                                        (happ_offset / happ_stride.get())
                                            .checked_add(happ_fixed_len.get())
                                            .ok_or(())?,
                                    )
                                    .ok_or(())
                                })
                                .transpose()
                        });

                        // FIXME(eddyb) this could maybe be allowed if there is still
                        // some kind of divisibility relation between the strides.
                        if ext_happ_offset != 0 {
                            return false;
                        }

                        layout_stride == happ_stride
                            && Ok(*layout_fixed_len) == ext_happ_fixed_len
                            && layout_elem.supports_happ_at_offset(0, happ_elem)
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
enum AttrTarget {
    Var(Var),
    Node(Node),
}

struct FuncGatherAccessesResults {
    param_accesses: SmallVec<[Option<Result<MemAccesses, AnalysisError>>; 2]>,
    accesses_or_err_attrs_to_attach: Vec<(AttrTarget, Result<MemAccesses, AnalysisError>)>,
}

#[derive(Clone)]
enum FuncGatherAccessesState {
    InProgress,
    Complete(Rc<FuncGatherAccessesResults>),
}

pub struct GatherAccesses<'a> {
    cx: Rc<Context>,
    layout_cache: LayoutCache<'a>,

    global_var_accesses: FxIndexMap<GlobalVar, Option<Result<MemAccesses, AnalysisError>>>,
    func_states: FxIndexMap<Func, FuncGatherAccessesState>,
}

impl<'a> GatherAccesses<'a> {
    pub fn new(cx: Rc<Context>, layout_config: &'a LayoutConfig) -> Self {
        Self {
            cx: cx.clone(),
            layout_cache: LayoutCache::new(cx, layout_config),

            global_var_accesses: Default::default(),
            func_states: Default::default(),
        }
    }

    pub fn gather_accesses_in_module(mut self, module: &mut Module) {
        for (export_key, &exportee) in &module.exports {
            if let Exportee::Func(func) = exportee {
                self.gather_accesses_in_func(module, func);
            }

            // Ensure even unused interface variables get their `mem.accesses`.
            match export_key {
                ExportKey::LinkName(_) => {}
                ExportKey::SpvEntryPoint { imms: _, interface_global_vars } => {
                    for &gv in interface_global_vars {
                        self.global_var_accesses.entry(gv).or_insert_with(|| {
                            Some(Ok(match module.global_vars[gv].shape {
                                Some(shapes::GlobalVarShape::Handles { handle, .. }) => {
                                    MemAccesses::Handles(match handle {
                                        shapes::Handle::Opaque(ty) => shapes::Handle::Opaque(ty),
                                        shapes::Handle::Buffer(..) => shapes::Handle::Buffer(
                                            AddrSpace::Handles,
                                            DataHapp::DEAD,
                                        ),
                                    })
                                }
                                _ => MemAccesses::Data(DataHapp::DEAD),
                            }))
                        });
                    }
                }
            }
        }

        // Analysis over, write all attributes back to the module.
        for (gv, accesses) in self.global_var_accesses {
            if let Some(accesses) = accesses {
                let global_var_def = &mut module.global_vars[gv];
                match accesses {
                    Ok(accesses) => {
                        // FIXME(eddyb) deduplicate attribute manipulation.
                        global_var_def.attrs = self.cx.intern(AttrSetDef {
                            attrs: self.cx[global_var_def.attrs]
                                .attrs
                                .iter()
                                .cloned()
                                .chain([Attr::Mem(MemAttr::Accesses(OrdAssertEq(accesses)))])
                                .collect(),
                        });
                    }
                    Err(AnalysisError(e)) => {
                        global_var_def.attrs.push_diag(&self.cx, e);
                    }
                }
            }
        }
        for (func, state) in self.func_states {
            match state {
                FuncGatherAccessesState::InProgress => unreachable!(),
                FuncGatherAccessesState::Complete(func_results) => {
                    let FuncGatherAccessesResults {
                        param_accesses,
                        accesses_or_err_attrs_to_attach,
                    } = Rc::try_unwrap(func_results).ok().unwrap();

                    let func_decl = &mut module.funcs[func];
                    for (param_decl, accesses) in func_decl.params.iter_mut().zip(param_accesses) {
                        if let Some(accesses) = accesses {
                            match accesses {
                                Ok(accesses) => {
                                    // FIXME(eddyb) deduplicate attribute manipulation.
                                    param_decl.attrs = self.cx.intern(AttrSetDef {
                                        attrs: self.cx[param_decl.attrs]
                                            .attrs
                                            .iter()
                                            .cloned()
                                            .chain([Attr::Mem(MemAttr::Accesses(OrdAssertEq(
                                                accesses,
                                            )))])
                                            .collect(),
                                    });
                                }
                                Err(AnalysisError(e)) => {
                                    param_decl.attrs.push_diag(&self.cx, e);
                                }
                            }
                        }
                    }

                    let func_def_body = match &mut module.funcs[func].def {
                        DeclDef::Present(func_def_body) => func_def_body,
                        DeclDef::Imported(_) => continue,
                    };

                    for (v, accesses) in accesses_or_err_attrs_to_attach {
                        let attrs = match v {
                            AttrTarget::Var(v) => &mut func_def_body.at_mut(v).decl().attrs,
                            AttrTarget::Node(node) => {
                                assert!(accesses.is_err());

                                &mut func_def_body.at_mut(node).def().attrs
                            }
                        };
                        match accesses {
                            Ok(accesses) => {
                                // FIXME(eddyb) deduplicate attribute manipulation.
                                *attrs = self.cx.intern(AttrSetDef {
                                    attrs: self.cx[*attrs]
                                        .attrs
                                        .iter()
                                        .cloned()
                                        .chain([Attr::Mem(MemAttr::Accesses(OrdAssertEq(
                                            accesses,
                                        )))])
                                        .collect(),
                                });
                            }
                            Err(AnalysisError(e)) => {
                                attrs.push_diag(&self.cx, e);
                            }
                        }
                    }
                }
            }
        }
    }

    // HACK(eddyb) `FuncGatherAccessesState` also serves to indicate recursion errors.
    fn gather_accesses_in_func(&mut self, module: &Module, func: Func) -> FuncGatherAccessesState {
        if let Some(cached) = self.func_states.get(&func).cloned() {
            return cached;
        }

        self.func_states.insert(func, FuncGatherAccessesState::InProgress);

        let completed_state = FuncGatherAccessesState::Complete(Rc::new(
            self.gather_accesses_in_func_uncached(module, func),
        ));

        self.func_states.insert(func, completed_state.clone());
        completed_state
    }
    fn gather_accesses_in_func_uncached(
        &mut self,
        module: &Module,
        func: Func,
    ) -> FuncGatherAccessesResults {
        let cx = self.cx.clone();
        let is_qptr = |ty: Type| matches!(cx[ty].kind, TypeKind::QPtr);

        let func_decl = &module.funcs[func];
        let mut param_accesses: SmallVec<[_; 2]> =
            (0..func_decl.params.len()).map(|_| None).collect();
        let mut accesses_or_err_attrs_to_attach = vec![];

        let func_def_body = match &module.funcs[func].def {
            DeclDef::Present(func_def_body) => func_def_body,
            DeclDef::Imported(_) => {
                for (param, param_accesses) in func_decl.params.iter().zip(&mut param_accesses) {
                    if is_qptr(param.ty) {
                        *param_accesses = Some(Err(AnalysisError(Diag::bug([
                            "pointer param of imported func".into(),
                        ]))));
                    }
                }
                return FuncGatherAccessesResults {
                    param_accesses,
                    accesses_or_err_attrs_to_attach,
                };
            }
        };

        let mut node_to_per_output_accesses: FxHashMap<_, SmallVec<[Option<_>; 2]>> =
            FxHashMap::default();

        // HACK(eddyb) reversing a post-order traversal to get RPO, which for
        // structured control-flow means outside-in/top-down (just like pre-order),
        // while post-order and reverse pre-order are inside-out/bottom-up.
        let mut post_order_nodes = vec![];
        func_def_body.inner_visit_with(&mut VisitAllNodes {
            before: |_| {},
            after: |node| post_order_nodes.push(node),
        });
        for node in post_order_nodes.into_iter().rev() {
            let per_output_accesses = node_to_per_output_accesses.remove(&node).unwrap_or_default();

            let node_def = func_def_body.at(node).def();
            let mut generate_accesses = |this: &mut Self, ptr: Value, new_accesses| {
                let slot = match ptr {
                    Value::Const(ct) => match cx[ct].kind {
                        ConstKind::PtrToGlobalVar(gv) => {
                            this.global_var_accesses.entry(gv).or_default()
                        }
                        // FIXME(eddyb) may be relevant?
                        _ => unreachable!(),
                    },
                    Value::Var(ptr) => match func_def_body.at(ptr).decl().kind() {
                        VarKind::RegionInput { region, input_idx }
                            if region == func_def_body.body =>
                        {
                            &mut param_accesses[input_idx as usize]
                        }
                        VarKind::RegionInput { .. } => {
                            // FIXME(eddyb) don't throw away `new_accesses`.
                            accesses_or_err_attrs_to_attach.push((
                                AttrTarget::Var(ptr),
                                Err(AnalysisError(Diag::bug(["unsupported φ".into()]))),
                            ));
                            return;
                        }
                        VarKind::NodeOutput { node: ptr_node, output_idx } => {
                            let i = output_idx as usize;
                            let slots = node_to_per_output_accesses.entry(ptr_node).or_default();
                            if i >= slots.len() {
                                slots.extend((slots.len()..=i).map(|_| None));
                            }
                            &mut slots[i]
                        }
                    },
                };
                *slot = Some(match slot.take() {
                    Some(old) => old.and_then(|old| {
                        AccessMerger { layout_cache: &this.layout_cache }
                            .merge(old, new_accesses?)
                            .into_result()
                    }),
                    None => new_accesses,
                });
            };

            match &node_def.kind {
                NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::ExitInvocation { .. } => {
                    for (&output_var, accesses) in node_def.outputs.iter().zip(&per_output_accesses)
                    {
                        if let Some(_accesses) = accesses {
                            // FIXME(eddyb) don't throw away `accesses`.
                            accesses_or_err_attrs_to_attach.push((
                                AttrTarget::Var(output_var),
                                Err(AnalysisError(Diag::bug(["unsupported φ".into()]))),
                            ));
                        }
                    }

                    continue;
                }

                DataInstKind::FuncCall(_)
                | DataInstKind::Mem(_)
                | DataInstKind::QPtr(_)
                | DataInstKind::SpvInst(_)
                | DataInstKind::SpvExtInst { .. } => {}
            }

            // HACK(eddyb) multi-output instructions don't exist pre-disaggregate.
            assert!(per_output_accesses.len() <= 1);
            let output_accesses = per_output_accesses.into_iter().next().flatten();

            // FIXME(eddyb) merge with `match &node_def.kind` above.
            let data_inst_def = node_def;
            match &data_inst_def.kind {
                NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::ExitInvocation(_) => {
                    unreachable!()
                }

                &DataInstKind::FuncCall(callee) => {
                    match self.gather_accesses_in_func(module, callee) {
                        FuncGatherAccessesState::Complete(callee_results) => {
                            for (&arg, param_accesses) in
                                data_inst_def.inputs.iter().zip(&callee_results.param_accesses)
                            {
                                if let Some(param_accesses) = param_accesses {
                                    generate_accesses(self, arg, param_accesses.clone());
                                }
                            }
                        }
                        FuncGatherAccessesState::InProgress => {
                            accesses_or_err_attrs_to_attach.push((
                                AttrTarget::Node(node),
                                Err(AnalysisError(Diag::bug(
                                    ["unsupported recursive call".into()],
                                ))),
                            ));
                        }
                    };
                    // HACK(eddyb) multi-output instructions don't exist pre-disaggregate.
                    if (data_inst_def.outputs.iter().at_most_one().ok().unwrap())
                        .is_some_and(|&o| is_qptr(func_def_body.at(o).decl().ty))
                        && let Some(accesses) = output_accesses
                    {
                        accesses_or_err_attrs_to_attach
                            .push((AttrTarget::Var(data_inst_def.outputs[0]), accesses));
                    }
                }

                DataInstKind::Mem(MemOp::FuncLocalVar(_)) => {
                    if let Some(accesses) = output_accesses {
                        accesses_or_err_attrs_to_attach
                            .push((AttrTarget::Var(data_inst_def.outputs[0]), accesses));
                    }
                }
                DataInstKind::QPtr(QPtrOp::HandleArrayIndex) => {
                    generate_accesses(
                        self,
                        data_inst_def.inputs[0],
                        output_accesses
                            .unwrap_or_else(|| {
                                Err(AnalysisError(Diag::bug([
                                    "HandleArrayIndex: unknown element".into()
                                ])))
                            })
                            .and_then(|accesses| match accesses {
                                MemAccesses::Handles(handle) => Ok(MemAccesses::Handles(handle)),
                                MemAccesses::Data(_) => Err(AnalysisError(Diag::bug([
                                    "HandleArrayIndex: cannot be accessed as data".into(),
                                ]))),
                            }),
                    );
                }
                DataInstKind::QPtr(QPtrOp::BufferData) => {
                    generate_accesses(
                        self,
                        data_inst_def.inputs[0],
                        output_accesses.unwrap_or(Ok(MemAccesses::Data(DataHapp::DEAD))).and_then(
                            |accesses| {
                                let happ = match accesses {
                                    MemAccesses::Handles(_) => {
                                        return Err(AnalysisError(Diag::bug([
                                            "BufferData: cannot be accessed as handles".into(),
                                        ])));
                                    }
                                    MemAccesses::Data(happ) => happ,
                                };
                                Ok(MemAccesses::Handles(shapes::Handle::Buffer(
                                    AddrSpace::Handles,
                                    happ,
                                )))
                            },
                        ),
                    );
                }
                &DataInstKind::QPtr(QPtrOp::BufferDynLen { fixed_base_size, dyn_unit_stride }) => {
                    let array_happ = DataHapp {
                        max_size: None,
                        kind: DataHappKind::Repeated {
                            element: Rc::new(DataHapp::DEAD),
                            stride: dyn_unit_stride,
                        },
                    };
                    let buf_data_happ = if fixed_base_size == 0 {
                        array_happ
                    } else {
                        DataHapp {
                            max_size: None,
                            kind: DataHappKind::Disjoint(Rc::new(
                                [(fixed_base_size, array_happ)].into(),
                            )),
                        }
                    };
                    generate_accesses(
                        self,
                        data_inst_def.inputs[0],
                        Ok(MemAccesses::Handles(shapes::Handle::Buffer(
                            AddrSpace::Handles,
                            buf_data_happ,
                        ))),
                    );
                }
                &DataInstKind::QPtr(QPtrOp::Offset(offset)) => {
                    generate_accesses(
                        self,
                        data_inst_def.inputs[0],
                        output_accesses.unwrap_or(Ok(MemAccesses::Data(DataHapp::DEAD))).and_then(
                            |accesses| {
                                let happ = match accesses {
                                    MemAccesses::Handles(_) => {
                                        return Err(AnalysisError(Diag::bug([format!(
                                            "Offset({offset}): cannot offset in handle memory"
                                        )
                                        .into()])));
                                    }
                                    MemAccesses::Data(happ) => happ,
                                };
                                let offset = u32::try_from(offset).ok().ok_or_else(|| {
                                    AnalysisError(Diag::bug([format!(
                                        "Offset({offset}): negative offset"
                                    )
                                    .into()]))
                                })?;

                                // FIXME(eddyb) these should be normalized
                                // (e.g. constant-folded) out of existence,
                                // but while they exist, they should be noops.
                                if offset == 0 {
                                    return Ok(MemAccesses::Data(happ));
                                }

                                Ok(MemAccesses::Data(DataHapp {
                                    max_size: happ
                                        .max_size
                                        .map(|max_size| {
                                            offset.checked_add(max_size).ok_or_else(|| {
                                                AnalysisError(Diag::bug([format!(
                                                    "Offset({offset}): size overflow \
                                                     ({offset}+{max_size})"
                                                )
                                                .into()]))
                                            })
                                        })
                                        .transpose()?,
                                    // FIXME(eddyb) allocating `Rc<BTreeMap<_, _>>`
                                    // to represent the one-element case, seems
                                    // quite wasteful when it's likely consumed.
                                    kind: DataHappKind::Disjoint(Rc::new([(offset, happ)].into())),
                                }))
                            },
                        ),
                    );
                }
                DataInstKind::QPtr(QPtrOp::DynOffset { stride, index_bounds }) => {
                    generate_accesses(
                        self,
                        data_inst_def.inputs[0],
                        output_accesses.unwrap_or(Ok(MemAccesses::Data(DataHapp::DEAD))).and_then(
                            |accesses| {
                                let happ = match accesses {
                                    MemAccesses::Handles(_) => {
                                        return Err(AnalysisError(Diag::bug([
                                            "DynOffset: cannot offset in handle memory".into(),
                                        ])));
                                    }
                                    MemAccesses::Data(happ) => happ,
                                };
                                match happ.max_size {
                                    None => {
                                        return Err(AnalysisError(Diag::bug([
                                            "DynOffset: unsized element".into(),
                                        ])));
                                    }
                                    // FIXME(eddyb) support this by "folding"
                                    // the HAPP onto itself (i.e. applying
                                    // `%= stride` on all offsets inside).
                                    Some(max_size) if max_size > stride.get() => {
                                        return Err(AnalysisError(Diag::bug([
                                            "DynOffset: element max_size exceeds stride".into(),
                                        ])));
                                    }
                                    Some(_) => {}
                                }
                                Ok(MemAccesses::Data(DataHapp {
                                    // FIXME(eddyb) does the `None` case allow
                                    // for negative offsets?
                                    max_size: index_bounds
                                        .as_ref()
                                        .map(|index_bounds| {
                                            if index_bounds.start < 0 || index_bounds.end < 0 {
                                                return Err(AnalysisError(Diag::bug([
                                                    "DynOffset: potentially negative offset".into(),
                                                ])));
                                            }
                                            let index_bounds_end =
                                                u32::try_from(index_bounds.end).unwrap();
                                            index_bounds_end.checked_mul(stride.get()).ok_or_else(
                                                || {
                                                    AnalysisError(Diag::bug([format!(
                                                        "DynOffset: size overflow \
                                                         ({index_bounds_end}*{stride})"
                                                    )
                                                    .into()]))
                                                },
                                            )
                                        })
                                        .transpose()?,
                                    kind: DataHappKind::Repeated {
                                        element: Rc::new(happ),
                                        stride: *stride,
                                    },
                                }))
                            },
                        ),
                    );
                }
                DataInstKind::Mem(op @ (MemOp::Load | MemOp::Store)) => {
                    // HACK(eddyb) `_` will match multiple variants soon.
                    #[allow(clippy::match_wildcard_for_single_variants)]
                    let (op_name, access_type) = match op {
                        MemOp::Load => {
                            ("Load", func_def_body.at(data_inst_def.outputs[0]).decl().ty)
                        }
                        MemOp::Store => {
                            ("Store", func_def_body.at(data_inst_def.inputs[1]).type_of(&cx))
                        }
                        _ => unreachable!(),
                    };
                    generate_accesses(
                        self,
                        data_inst_def.inputs[0],
                        self.layout_cache
                            .layout_of(access_type)
                            .map_err(|LayoutError(e)| AnalysisError(e))
                            .and_then(|layout| match layout {
                                TypeLayout::Handle(shapes::Handle::Opaque(ty)) => {
                                    Ok(MemAccesses::Handles(shapes::Handle::Opaque(ty)))
                                }
                                TypeLayout::Handle(shapes::Handle::Buffer(..)) => {
                                    Err(AnalysisError(Diag::bug([format!(
                                        "{op_name}: cannot access whole Buffer"
                                    )
                                    .into()])))
                                }
                                TypeLayout::HandleArray(..) => {
                                    Err(AnalysisError(Diag::bug([format!(
                                        "{op_name}: cannot access whole HandleArray"
                                    )
                                    .into()])))
                                }
                                TypeLayout::Concrete(concrete)
                                    if concrete.mem_layout.dyn_unit_stride.is_some() =>
                                {
                                    Err(AnalysisError(Diag::bug([format!(
                                        "{op_name}: cannot access unsized type"
                                    )
                                    .into()])))
                                }
                                TypeLayout::Concrete(concrete) => Ok(MemAccesses::Data(DataHapp {
                                    max_size: Some(concrete.mem_layout.fixed_base.size),
                                    kind: DataHappKind::Direct(access_type),
                                })),
                            }),
                    );
                }

                DataInstKind::SpvInst(_) | DataInstKind::SpvExtInst { .. } => {
                    let mut has_from_spv_ptr_output_attr = false;
                    for attr in &cx[data_inst_def.attrs].attrs {
                        match *attr {
                            Attr::QPtr(QPtrAttr::ToSpvPtrInput { input_idx, pointee }) => {
                                let ty = pointee.0;
                                generate_accesses(
                                    self,
                                    data_inst_def.inputs[input_idx as usize],
                                    self.layout_cache
                                        .layout_of(ty)
                                        .map_err(|LayoutError(e)| AnalysisError(e))
                                        .and_then(|layout| match layout {
                                            TypeLayout::Handle(handle) => {
                                                let handle = match handle {
                                                    shapes::Handle::Opaque(ty) => {
                                                        shapes::Handle::Opaque(ty)
                                                    }
                                                    // NOTE(eddyb) this error is important,
                                                    // as the `Block` annotation on the
                                                    // buffer type means the type is *not*
                                                    // usable anywhere inside buffer data,
                                                    // since it would conflict with our
                                                    // own `Block`-annotated wrapper.
                                                    shapes::Handle::Buffer(..) => {
                                                        return Err(AnalysisError(Diag::bug([
                                                            "ToSpvPtrInput: \
                                                             whole Buffer ambiguous \
                                                             (handle vs buffer data)"
                                                                .into(),
                                                        ])));
                                                    }
                                                };
                                                Ok(MemAccesses::Handles(handle))
                                            }
                                            // NOTE(eddyb) because we can't represent
                                            // the original type, in the same way we
                                            // use `DataHappKind::StrictlyTyped`
                                            // for non-handles, we can't guarantee
                                            // a generated type that matches the
                                            // desired `pointee` type.
                                            TypeLayout::HandleArray(..) => {
                                                Err(AnalysisError(Diag::bug([
                                                    "ToSpvPtrInput: whole handle array \
                                                     unrepresentable"
                                                        .into(),
                                                ])))
                                            }
                                            TypeLayout::Concrete(concrete) => {
                                                Ok(MemAccesses::Data(DataHapp {
                                                    max_size: if concrete
                                                        .mem_layout
                                                        .dyn_unit_stride
                                                        .is_some()
                                                    {
                                                        None
                                                    } else {
                                                        Some(concrete.mem_layout.fixed_base.size)
                                                    },
                                                    kind: DataHappKind::StrictlyTyped(ty),
                                                }))
                                            }
                                        }),
                                );
                            }
                            Attr::QPtr(QPtrAttr::FromSpvPtrOutput {
                                addr_space: _,
                                pointee: _,
                            }) => {
                                has_from_spv_ptr_output_attr = true;
                            }
                            _ => {}
                        }
                    }

                    if has_from_spv_ptr_output_attr {
                        // FIXME(eddyb) merge with `FromSpvPtrOutput`'s `pointee`.
                        if let Some(accesses) = output_accesses {
                            accesses_or_err_attrs_to_attach
                                .push((AttrTarget::Var(data_inst_def.outputs[0]), accesses));
                        }
                    }
                }
            }
        }

        FuncGatherAccessesResults { param_accesses, accesses_or_err_attrs_to_attach }
    }
}

// HACK(eddyb) this is easier than implementing a proper reverse traversal.
#[derive(Default)]
struct VisitAllNodes<B: FnMut(Node), A: FnMut(Node)> {
    before: B,
    after: A,
}

impl<B: FnMut(Node), A: FnMut(Node)> Visitor<'_> for VisitAllNodes<B, A> {
    // FIXME(eddyb) this is excessive, maybe different kinds of
    // visitors should exist for module-level and func-level?
    fn visit_attr_set_use(&mut self, _: AttrSet) {}
    fn visit_type_use(&mut self, _: Type) {}
    fn visit_const_use(&mut self, _: Const) {}
    fn visit_global_var_use(&mut self, _: GlobalVar) {}
    fn visit_func_use(&mut self, _: Func) {}

    fn visit_node_def(&mut self, func_at_node: FuncAt<'_, Node>) {
        (self.before)(func_at_node.position);
        func_at_node.inner_visit_with(self);
        (self.after)(func_at_node.position);
    }
}
