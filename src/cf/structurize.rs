//! Control-flow structurization (unstructured CFG -> structured regions).
//
// FIXME(eddyb) consider moving docs to the module level?

use crate::cf::SelectionKind;
use crate::cf::unstructured::{
    ControlFlowGraph, ControlInst, ControlInstKind, IncomingEdgeCount, LoopFinder, TraversalState,
};
use crate::transform::{InnerInPlaceTransform as _, Transformer};
use crate::{
    AttrSet, Const, ConstDef, ConstKind, Context, DbgSrcLoc, EntityOrientedDenseMap, FuncDefBody,
    FxIndexMap, FxIndexSet, Node, NodeDef, NodeKind, NodeOutputDecl, Region, RegionDef, Type,
    TypeKind, Value, spv,
};
use itertools::{Either, Itertools};
use smallvec::SmallVec;
use std::mem;
use std::rc::Rc;

#[allow(rustdoc::private_intra_doc_links)]
/// Control-flow "structurizer", which attempts to convert as much of the CFG
/// as possible into structural control-flow (regions).
///
/// See [`StructurizeRegionState`]'s docs for more details on the algorithm.
//
// FIXME(eddyb) document this (instead of having it on `StructurizeRegionState`).
//
// NOTE(eddyb) CFG structurizer has these stages (per-region):
//   1. absorb any deferred exits that finally have 100% refcount
//   2. absorb a single backedge deferred exit to the same region
//
//   What we could add is a third step, to handle irreducible controlflow:
//   3. check for groups of exits that have fully satisfied refcounts iff the
//     rest of the exits in the group are all added together - if so, the group
//     is *irreducible* and a single "loop header" can be created, that gets
//     the group of deferred exits, and any other occurrence of the deferred
//     exits (in either the original region, or amongst themselves) can be
//     replaced with the "loop header" with appropriate selector inputs
//
//   Sadly 3. requires a bunch of tests that are hard to craft (can rustc MIR
//   even end up in the right shape?).
//   OpenCL has `goto` so maybe it can also be used for this worse-than-diamond
//   example: `entry -> a,b,d` `a,b -> c` `a,b,c -> d` `a,b,c,d <-> a,b,c,d`
//   (the goal is avoiding a "flat group", i.e. where there is only one step
//   between every exit in the group and another exit)
pub struct Structurizer<'a> {
    cx: &'a Context,

    /// Scrutinee type for [`SelectionKind::BoolCond`].
    type_bool: Type,

    /// Scrutinee value for [`SelectionKind::BoolCond`], for the "then" case.
    const_true: Const,

    /// Scrutinee value for [`SelectionKind::BoolCond`], for the "else" case.
    const_false: Const,

    func_def_body: &'a mut FuncDefBody,

    // FIXME(eddyb) this feels a bit inefficient (are many-exit loops rare?).
    loop_header_to_exit_targets: FxIndexMap<Region, FxIndexSet<Region>>,

    // HACK(eddyb) this also tracks all of `loop_header_to_exit_targets`, as
    // "false edges" from every loop header to each exit target of that loop,
    // which structurizing that loop consumes to "unlock" its own exits.
    incoming_edge_counts_including_loop_exits: EntityOrientedDenseMap<Region, IncomingEdgeCount>,

    /// `structurize_region_state[region]` tracks `.structurize_region(region)`
    /// progress/results (see also [`StructurizeRegionState`]'s docs).
    //
    // FIXME(eddyb) use `EntityOrientedDenseMap` (which lacks iteration by design).
    structurize_region_state: FxIndexMap<Region, StructurizeRegionState>,

    /// Accumulated rewrites (caused by e.g. `target_inputs`s, but not only),
    /// i.e.: `Value::RegionInput { region, input_idx }` must be
    /// rewritten based on `region_input_rewrites[region]`, as either
    /// the original `region` wasn't reused, or its inputs were renumbered.
    region_input_rewrites: EntityOrientedDenseMap<Region, RegionInputRewrites>,
}

/// How all `Value::RegionInput { region, input_idx }` for a `region`
/// must be rewritten (see also `region_input_rewrites` docs).
enum RegionInputRewrites {
    /// Complete replacement with another value (which can take any form), as
    /// `region` wasn't kept in its original form in the final structured IR.
    ///
    /// **Note**: such replacement can be chained, i.e. a replacement value can
    /// be `Value::RegionInput { region: other_region, .. }`, and then
    /// `other_region` itself may have its inputs written.
    ReplaceWith(SmallVec<[Value; 2]>),

    /// The value may remain an input of the same `region`, only changing its
    /// `input_idx` (e.g. if indices need compaction after removing some inputs),
    /// or get replaced anyway, depending on the `Result` for `input_idx`.
    ///
    /// **Note**: renumbering can only be the last rewrite step of a value,
    /// as `region` must've been chosen to be kept in the final structured IR,
    /// but the `Err` cases are transitive just like `ReplaceWith`.
    //
    // FIXME(eddyb) this is a bit silly, maybe try to rely more on hermeticity
    // to get rid of this?
    RenumberOrReplaceWith(SmallVec<[Result<u32, Value>; 2]>),
}

impl RegionInputRewrites {
    // HACK(eddyb) this is here because it depends on a field of `Structurizer`
    // and borrowing issues ensue if it's made a method of `Structurizer`.
    fn rewrite_all(
        rewrites: &EntityOrientedDenseMap<Region, Self>,
    ) -> impl crate::transform::Transformer + '_ {
        // FIXME(eddyb) maybe this should be provided by `transform`.
        use crate::transform::*;
        struct ReplaceValueWith<F>(F);
        impl<F: Fn(Value) -> Option<Value>> Transformer for ReplaceValueWith<F> {
            fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
                self.0(*v).map_or(Transformed::Unchanged, Transformed::Changed)
            }
        }

        ReplaceValueWith(move |v| {
            let mut new_v = v;
            while let Value::RegionInput { region, input_idx } = new_v {
                match rewrites.get(region) {
                    // NOTE(eddyb) this needs to be able to apply multiple replacements,
                    // due to the input potentially having redundantly chained `OpPhi`s.
                    //
                    // FIXME(eddyb) union-find-style "path compression" could record the
                    // final value inside `rewrites` while replacements are being made,
                    // to avoid going through a chain more than once (and some of these
                    // replacements could also be applied early).
                    Some(RegionInputRewrites::ReplaceWith(replacements)) => {
                        new_v = replacements[input_idx as usize];
                    }
                    Some(RegionInputRewrites::RenumberOrReplaceWith(
                        renumbering_and_replacements,
                    )) => match renumbering_and_replacements[input_idx as usize] {
                        Ok(new_idx) => {
                            new_v = Value::RegionInput { region, input_idx: new_idx };
                            break;
                        }
                        Err(replacement) => new_v = replacement,
                    },
                    None => break,
                }
            }
            (v != new_v).then_some(new_v)
        })
    }
}

/// The state of one `.structurize_region(region)` invocation, and its result.
///
/// There is a fourth (or 0th) implicit state, which is where nothing has yet
/// observed some region, and [`Structurizer`] isn't tracking it at all.
//
// FIXME(eddyb) make the 0th state explicit and move `incoming_edge_counts` to it.
enum StructurizeRegionState {
    /// Structurization is still running, and observing this is a cycle.
    InProgress,

    /// Structurization completed, and this region can now be claimed.
    Ready {
        /// Cached `region_deferred_edges[region].edge_bundle.accumulated_count`,
        /// i.e. the total count of backedges (if any exist) pointing to `region`
        /// from the CFG subgraph that `region` itself dominates.
        ///
        /// Claiming a region with backedges can combine them with the bundled
        /// edges coming into the CFG cycle from outside, and instead of failing
        /// due to the latter not being enough to claim the region on their own,
        /// actually perform loop structurization.
        accumulated_backedge_count: IncomingEdgeCount,

        // HACK(eddyb) the only part of a `ClaimedRegion` that is computed by
        // `structurize_region` (the rest comes from `try_claim_edge_bundle`).
        region_deferred_edges: DeferredEdgeBundleSet,
    },

    /// Region was claimed (by an [`IncomingEdgeBundle`], with the appropriate
    /// total [`IncomingEdgeCount`], minus `accumulated_backedge_count`), and
    /// must eventually be incorporated as part of some larger region.
    Claimed,
}

/// An "(incoming) edge bundle" is a subset of the edges into a single `target`.
///
/// When `accumulated_count` reaches the total [`IncomingEdgeCount`] for `target`,
/// that [`IncomingEdgeBundle`] is said to "effectively own" its `target` (akin to
/// the more commonly used CFG domination relation, but more "incremental").
///
/// **Note**: `target` has a generic type `T` to reduce redundancy when it's
/// already implied (e.g. by the key in [`DeferredEdgeBundleSet`]'s map).
struct IncomingEdgeBundle<T> {
    /// Attributes from the original [`ControlInst`]s (likely debuginfo), kept
    /// when merging only when exactly identical, which can naturally be the case
    /// for debuginfo (e.g. for branches from inside `if`-`else`/`switch` to a
    /// common merge point, just after the whole control-flow construct).
    //
    // FIXME(eddyb) semantically filter these, maybe focus on debuginfo?
    attrs: AttrSet,

    target: T,
    accumulated_count: IncomingEdgeCount,

    /// The [`Value`]s that `Value::RegionInput { region, .. }` will get
    /// on entry into `region`, through this "edge bundle".
    target_inputs: SmallVec<[Value; 2]>,
}

impl<T> IncomingEdgeBundle<T> {
    fn with_target<U>(self, target: U) -> IncomingEdgeBundle<U> {
        let IncomingEdgeBundle { attrs, target: _, accumulated_count, target_inputs } = self;
        IncomingEdgeBundle { attrs, target, accumulated_count, target_inputs }
    }
}

/// A "deferred (incoming) edge bundle" is an [`IncomingEdgeBundle`] that cannot
/// be structurized immediately, but instead waits for its `accumulated_count`
/// to reach the full count of its `target`, before it can grafted into some
/// structured control-flow region.
///
/// While in the "deferred" state, its can accumulate a non-trivial `condition`,
/// every time it's propagated to an "outer" region, e.g. for this pseudocode:
/// ```text
/// if a {
///     branch => label1
/// } else {
///     if b {
///         branch => label1
///     }
/// }
/// ```
/// the deferral of branches to `label1` will result in:
/// ```text
/// label1_condition = if a {
///     true
/// } else {
///     if b {
///         true
///     } else {
///         false
///     }
/// }
/// if label1_condition {
///     branch => label1
/// }
/// ```
/// which could theoretically be simplified (after the [`Structurizer`]) to:
/// ```text
/// label1_condition = a | b
/// if label1_condition {
///     branch => label1
/// }
/// ```
///
/// **Note**: `edge_bundle.target` has a generic type `T` to reduce redundancy
/// when it's already implied (e.g. by the key in [`DeferredEdgeBundleSet`]'s map).
struct DeferredEdgeBundle<T = DeferredTarget> {
    condition: LazyCond,
    edge_bundle: IncomingEdgeBundle<T>,
}

impl<T> DeferredEdgeBundle<T> {
    fn with_target<U>(self, target: U) -> DeferredEdgeBundle<U> {
        let DeferredEdgeBundle { condition, edge_bundle } = self;
        DeferredEdgeBundle { condition, edge_bundle: edge_bundle.with_target(target) }
    }
}

/// A recipe for computing a control-flow-sensitive (boolean) condition [`Value`],
/// potentially requiring merging through an arbitrary number of `Select`s
/// (via per-case outputs and [`Value::NodeOutput`], for each `Select`).
///
/// This should largely be equivalent to eagerly generating all region outputs
/// that might be needed, and then removing the unused ones, but this way we
/// never generate unused outputs, and can potentially even optimize away some
/// redundant dataflow (e.g. `if cond { true } else { false }` is just `cond`).
#[derive(Clone)]
enum LazyCond {
    // HACK(eddyb) `Undef` is used when the condition comes from e.g. a `Select`
    // case that diverges and/or represents `unreachable`.
    Undef,

    False,
    True,

    Merge(Rc<LazyCondMerge>),
}

enum LazyCondMerge {
    Select {
        node: Node,
        // FIXME(eddyb) the lowest level of `LazyCond` ends up containing only
        // `LazyCond::{Undef,False,True}`, and that could more efficiently be
        // expressed using e.g. bitsets, but the `Rc` in `LazyCond::Merge`
        // means that this is more compact than it would otherwise be.
        per_case_conds: SmallVec<[LazyCond; 4]>,
    },
}

/// A target for one of the edge bundles in a [`DeferredEdgeBundleSet`], mostly
/// separate from [`Region`] to allow expressing returns as well.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum DeferredTarget {
    Region(Region),

    /// Structured "return" out of the function (with `target_inputs` used for
    /// the function body `output`s, i.e. inputs of [`ControlInstKind::Return`]).
    Return,
}

/// Set of [`DeferredEdgeBundle`]s, uniquely keyed by their `target`s.
///
/// Semantically equivalent to an unordered series of conditional branches
/// to each possible `target`, which corresponds to an unenforced invariant
/// that exactly one [`DeferredEdgeBundle`] condition must be `true` at any
/// given time (the only non-trivial case, [`DeferredEdgeBundleSet::Choice`],
/// satisfies it because it's only used for merging `Select` cases, and so
/// all the conditions will end up using disjoint [`LazyCond::Merge`]s).
enum DeferredEdgeBundleSet {
    Unreachable,

    // NOTE(eddyb) this erases the condition (by not using `DeferredEdgeBundle`).
    Always {
        // HACK(eddyb) fields are split here to allow e.g. iteration.
        target: DeferredTarget,
        edge_bundle: IncomingEdgeBundle<()>,
    },

    Choice {
        target_to_deferred: FxIndexMap<DeferredTarget, DeferredEdgeBundle<()>>,
    },
}

impl FromIterator<DeferredEdgeBundle> for DeferredEdgeBundleSet {
    fn from_iter<T: IntoIterator<Item = DeferredEdgeBundle>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        match iter.next() {
            None => Self::Unreachable,
            Some(first) => match iter.next() {
                // NOTE(eddyb) this erases the condition (by not using `DeferredEdgeBundle`).
                None => Self::Always {
                    target: first.edge_bundle.target,
                    edge_bundle: first.edge_bundle.with_target(()),
                },
                Some(second) => Self::Choice {
                    target_to_deferred: ([first, second].into_iter().chain(iter))
                        .map(|d| (d.edge_bundle.target, d.with_target(())))
                        .collect(),
                },
            },
        }
    }
}

impl From<FxIndexMap<DeferredTarget, DeferredEdgeBundle<()>>> for DeferredEdgeBundleSet {
    fn from(target_to_deferred: FxIndexMap<DeferredTarget, DeferredEdgeBundle<()>>) -> Self {
        if target_to_deferred.len() <= 1 {
            target_to_deferred
                .into_iter()
                .map(|(target, deferred)| deferred.with_target(target))
                .collect()
        } else {
            Self::Choice { target_to_deferred }
        }
    }
}

// HACK(eddyb) this API is a mess, is there an uncompromising way to clean it up?
impl DeferredEdgeBundleSet {
    fn get_edge_bundle_by_target(
        &self,
        search_target: DeferredTarget,
    ) -> Option<&IncomingEdgeBundle<()>> {
        match self {
            DeferredEdgeBundleSet::Unreachable => None,
            DeferredEdgeBundleSet::Always { target, edge_bundle } => {
                (*target == search_target).then_some(edge_bundle)
            }
            DeferredEdgeBundleSet::Choice { target_to_deferred } => {
                Some(&target_to_deferred.get(&search_target)?.edge_bundle)
            }
        }
    }

    fn get_edge_bundle_mut_by_target(
        &mut self,
        search_target: DeferredTarget,
    ) -> Option<&mut IncomingEdgeBundle<()>> {
        match self {
            DeferredEdgeBundleSet::Unreachable => None,
            DeferredEdgeBundleSet::Always { target, edge_bundle } => {
                (*target == search_target).then_some(edge_bundle)
            }
            DeferredEdgeBundleSet::Choice { target_to_deferred } => {
                Some(&mut target_to_deferred.get_mut(&search_target)?.edge_bundle)
            }
        }
    }

    fn iter_targets_with_edge_bundle(
        &self,
    ) -> impl Iterator<Item = (DeferredTarget, &IncomingEdgeBundle<()>)> {
        match self {
            DeferredEdgeBundleSet::Unreachable => Either::Left(None.into_iter()),
            DeferredEdgeBundleSet::Always { target, edge_bundle } => {
                Either::Left(Some((*target, edge_bundle)).into_iter())
            }
            DeferredEdgeBundleSet::Choice { target_to_deferred } => Either::Right(
                target_to_deferred
                    .iter()
                    .map(|(&target, deferred)| (target, &deferred.edge_bundle)),
            ),
        }
    }

    fn iter_targets_with_edge_bundle_mut(
        &mut self,
    ) -> impl Iterator<Item = (DeferredTarget, &mut IncomingEdgeBundle<()>)> {
        match self {
            DeferredEdgeBundleSet::Unreachable => Either::Left(None.into_iter()),
            DeferredEdgeBundleSet::Always { target, edge_bundle } => {
                Either::Left(Some((*target, edge_bundle)).into_iter())
            }
            DeferredEdgeBundleSet::Choice { target_to_deferred } => Either::Right(
                target_to_deferred
                    .iter_mut()
                    .map(|(&target, deferred)| (target, &mut deferred.edge_bundle)),
            ),
        }
    }

    // HACK(eddyb) this only exists because of `DeferredEdgeBundleSet`'s lossy
    // representation wrt conditions, so removal from a `DeferredEdgeBundleSet`
    // cannot be used for e.g. `Select` iterating over per-case deferreds.
    fn steal_deferred_by_target_without_removal(
        &mut self,
        search_target: DeferredTarget,
    ) -> Option<DeferredEdgeBundle<()>> {
        let steal_edge_bundle = |edge_bundle: &mut IncomingEdgeBundle<()>| IncomingEdgeBundle {
            attrs: edge_bundle.attrs,
            target: (),
            accumulated_count: edge_bundle.accumulated_count,
            target_inputs: mem::take(&mut edge_bundle.target_inputs),
        };
        match self {
            DeferredEdgeBundleSet::Unreachable => None,
            DeferredEdgeBundleSet::Always { target, edge_bundle } => (*target == search_target)
                .then(|| DeferredEdgeBundle {
                    condition: LazyCond::True,
                    edge_bundle: steal_edge_bundle(edge_bundle),
                }),
            DeferredEdgeBundleSet::Choice { target_to_deferred } => {
                let DeferredEdgeBundle { condition, edge_bundle } =
                    target_to_deferred.get_mut(&search_target)?;
                Some(DeferredEdgeBundle {
                    condition: mem::replace(condition, LazyCond::False),
                    edge_bundle: steal_edge_bundle(edge_bundle),
                })
            }
        }
    }

    // NOTE(eddyb) the returned `DeferredEdgeBundleSet` exists under the assumption
    // that `split_target` is not reachable from it, so this method is not suitable
    // for e.g. uniformly draining `DeferredEdgeBundleSet` in a way that preserves
    // conditions (but rather it's almost a kind of control-flow "slicing").
    fn split_out_target(self, split_target: DeferredTarget) -> (Option<DeferredEdgeBundle>, Self) {
        match self {
            DeferredEdgeBundleSet::Unreachable => (None, DeferredEdgeBundleSet::Unreachable),
            DeferredEdgeBundleSet::Always { target, edge_bundle } => {
                if target == split_target {
                    (
                        Some(DeferredEdgeBundle {
                            condition: LazyCond::True,
                            edge_bundle: edge_bundle.with_target(target),
                        }),
                        DeferredEdgeBundleSet::Unreachable,
                    )
                } else {
                    (None, DeferredEdgeBundleSet::Always { target, edge_bundle })
                }
            }
            DeferredEdgeBundleSet::Choice { mut target_to_deferred } => {
                // FIXME(eddyb) should this use `shift_remove` and/or emulate
                // extra tombstones, to avoid impacting the order?
                (
                    target_to_deferred
                        .swap_remove(&split_target)
                        .map(|d| d.with_target(split_target)),
                    Self::from(target_to_deferred),
                )
            }
        }
    }

    // HACK(eddyb) the strange signature is overfitted to its own callsite.
    fn split_out_matching<T>(
        self,
        mut matches: impl FnMut(DeferredEdgeBundle) -> Result<T, DeferredEdgeBundle>,
    ) -> (Option<T>, Self) {
        match self {
            DeferredEdgeBundleSet::Unreachable => (None, DeferredEdgeBundleSet::Unreachable),
            DeferredEdgeBundleSet::Always { target, edge_bundle } => {
                match matches(DeferredEdgeBundle {
                    condition: LazyCond::True,
                    edge_bundle: edge_bundle.with_target(target),
                }) {
                    Ok(x) => (Some(x), DeferredEdgeBundleSet::Unreachable),
                    Err(new_deferred) => {
                        assert!(new_deferred.edge_bundle.target == target);
                        assert!(matches!(new_deferred.condition, LazyCond::True));
                        (
                            None,
                            DeferredEdgeBundleSet::Always {
                                target,
                                edge_bundle: new_deferred.edge_bundle.with_target(()),
                            },
                        )
                    }
                }
            }
            DeferredEdgeBundleSet::Choice { mut target_to_deferred } => {
                let mut result = None;
                for (i, (&target, deferred)) in target_to_deferred.iter_mut().enumerate() {
                    // HACK(eddyb) "take" `deferred` so it can be passed to
                    // `matches` (and put back if that returned `Err`).
                    let taken_deferred = mem::replace(
                        deferred,
                        DeferredEdgeBundle {
                            condition: LazyCond::False,
                            edge_bundle: IncomingEdgeBundle {
                                attrs: Default::default(),
                                target: Default::default(),
                                accumulated_count: Default::default(),
                                target_inputs: Default::default(),
                            },
                        },
                    );

                    match matches(taken_deferred.with_target(target)) {
                        Ok(x) => {
                            result = Some(x);
                            // FIXME(eddyb) should this use `swap_remove_index`?
                            target_to_deferred.shift_remove_index(i).unwrap();
                            break;
                        }

                        // Put back the `DeferredEdgeBundle` and keep looking.
                        Err(new_deferred) => {
                            assert!(new_deferred.edge_bundle.target == target);
                            *deferred = new_deferred.with_target(());
                        }
                    }
                }
                (result, Self::from(target_to_deferred))
            }
        }
    }
}

/// A successfully "claimed" (via `try_claim_edge_bundle`) partially structurized
/// CFG subgraph (i.e. set of [`Region`]s previously connected by CFG edges),
/// which is effectively owned by the "claimer" and **must** be used for:
/// - the whole function body (if `deferred_edges` only contains `Return`)
/// - one of the cases of a `Select` node
/// - merging into a larger region (i.e. its nearest dominator)
//
// FIXME(eddyb) consider never having to claim the function body itself,
// by wrapping the CFG in a `Node` instead.
struct ClaimedRegion {
    // FIXME(eddyb) find a way to clarify that this can differ from the target
    // of `try_claim_edge_bundle`, and also that `deferred_edges` are from the
    // perspective of being "inside" `structured_body` (wrt hermeticity).
    structured_body: Region,

    /// The [`Value`]s that `Value::RegionInput { region: structured_body, .. }`
    /// will get on entry into `structured_body`, when this region ends up
    /// merged into a larger region, or as a child of a new [`Node`].
    //
    // FIXME(eddyb) don't replace `Value::RegionInput { region: structured_body, .. }`
    // with `region_inputs` when `structured_body` ends up a `Node` child,
    // but instead make all `Region`s entirely hermetic wrt inputs.
    structured_body_inputs: SmallVec<[Value; 2]>,

    /// The transitive targets which couldn't be claimed into `structured_body`
    /// remain as deferred exits, and will block further structurization until
    /// all other edges to those same targets are gathered together.
    ///
    /// **Note**: this will only be empty if the region can never exit,
    /// i.e. it has divergent control-flow (such as an infinite loop), as any
    /// control-flow path that can (eventually) return from the function, will
    /// end up using a deferred target for that (see [`DeferredTarget::Return`]).
    deferred_edges: DeferredEdgeBundleSet,
}

impl<'a> Structurizer<'a> {
    pub fn new(cx: &'a Context, func_def_body: &'a mut FuncDefBody) -> Self {
        // FIXME(eddyb) SPIR-T should have native booleans itself.
        let wk = &spv::spec::Spec::get().well_known;
        let type_bool = cx.intern(TypeKind::SpvInst {
            spv_inst: wk.OpTypeBool.into(),
            type_and_const_inputs: [].into_iter().collect(),
        });
        let const_true = cx.intern(ConstDef {
            attrs: AttrSet::default(),
            ty: type_bool,
            kind: ConstKind::SpvInst {
                spv_inst_and_const_inputs: Rc::new((
                    wk.OpConstantTrue.into(),
                    [].into_iter().collect(),
                )),
            },
        });
        let const_false = cx.intern(ConstDef {
            attrs: AttrSet::default(),
            ty: type_bool,
            kind: ConstKind::SpvInst {
                spv_inst_and_const_inputs: Rc::new((
                    wk.OpConstantFalse.into(),
                    [].into_iter().collect(),
                )),
            },
        });

        let (loop_header_to_exit_targets, incoming_edge_counts_including_loop_exits) =
            func_def_body
                .unstructured_cfg
                .as_ref()
                .map(|cfg| {
                    let loop_header_to_exit_targets =
                        LoopFinder::new(cfg).find_all_loops_starting_at(func_def_body.body);

                    let mut state = TraversalState {
                        incoming_edge_counts: EntityOrientedDenseMap::new(),

                        pre_order_visit: |_| {},
                        post_order_visit: |_| {},
                        reverse_targets: false,
                    };
                    cfg.traverse_whole_func(func_def_body, &mut state);

                    // HACK(eddyb) treat loop exits as "false edges", that their
                    // respective loop header "owns", such that structurization
                    // naturally stops at those loop exits, instead of continuing
                    // greedily into the loop exterior (producing "maximal loops").
                    for loop_exit_targets in loop_header_to_exit_targets.values() {
                        for &exit_target in loop_exit_targets {
                            *state
                                .incoming_edge_counts
                                .entry(exit_target)
                                .get_or_insert(Default::default()) += IncomingEdgeCount::ONE;
                        }
                    }

                    (loop_header_to_exit_targets, state.incoming_edge_counts)
                })
                .unwrap_or_default();

        Self {
            cx,
            type_bool,
            const_true,
            const_false,

            func_def_body,

            loop_header_to_exit_targets,
            incoming_edge_counts_including_loop_exits,

            structurize_region_state: FxIndexMap::default(),
            region_input_rewrites: EntityOrientedDenseMap::new(),
        }
    }

    pub fn structurize_func(mut self) {
        // Don't even try to re-structurize functions.
        if self.func_def_body.unstructured_cfg.is_none() {
            return;
        }

        // FIXME(eddyb) it might work much better to have the unstructured CFG
        // wrapped in a `Node` inside the function body, instead.
        let func_body_deferred_edges = {
            let func_entry_pseudo_edge = {
                let target = self.func_def_body.body;
                move || IncomingEdgeBundle {
                    attrs: Default::default(),
                    target,
                    accumulated_count: IncomingEdgeCount::ONE,
                    target_inputs: [].into_iter().collect(),
                }
            };

            // HACK(eddyb) it's easier to assume the function never loops back
            // to its body, than fix up the broken CFG if that never happens.
            if self.incoming_edge_counts_including_loop_exits[func_entry_pseudo_edge().target]
                != func_entry_pseudo_edge().accumulated_count
            {
                // FIXME(eddyb) find a way to attach (diagnostic) attributes
                // to a `FuncDefBody`, would be useful to have that here.
                return;
            }

            let ClaimedRegion { structured_body, structured_body_inputs, deferred_edges } =
                self.try_claim_edge_bundle(func_entry_pseudo_edge()).ok().unwrap();
            assert!(structured_body == func_entry_pseudo_edge().target);
            assert!(structured_body_inputs == func_entry_pseudo_edge().target_inputs);
            deferred_edges
        };

        match func_body_deferred_edges {
            // FIXME(eddyb) also support structured return when the whole body
            // is divergent, by generating undef constants (needs access to the
            // whole `FuncDecl`, not just `FuncDefBody`, to get the right types).
            DeferredEdgeBundleSet::Unreachable => {
                // HACK(eddyb) replace the CFG with one that only contains an
                // `Unreachable` terminator for the body, comparable to what
                // `rebuild_cfg_from_unclaimed_region_deferred_edges` would do
                // in the general case (but special-cased because this is very
                // close to being structurizable, just needs a bit of plumbing).
                let mut control_inst_on_exit_from = EntityOrientedDenseMap::new();
                control_inst_on_exit_from.insert(
                    self.func_def_body.body,
                    ControlInst {
                        attrs: AttrSet::default(),
                        kind: ControlInstKind::Unreachable,
                        inputs: [].into_iter().collect(),
                        targets: [].into_iter().collect(),
                        target_inputs: FxIndexMap::default(),
                    },
                );
                self.func_def_body.unstructured_cfg = Some(ControlFlowGraph {
                    control_inst_on_exit_from,
                    loop_merge_to_loop_header: Default::default(),
                });
            }

            // Structured return, the function is fully structurized.
            DeferredEdgeBundleSet::Always { target: DeferredTarget::Return, edge_bundle } => {
                let body_def = self.func_def_body.at_mut_body().def();
                body_def.outputs = edge_bundle.target_inputs;
                self.func_def_body.unstructured_cfg = None;
            }

            _ => {
                // Repair all the regions that remain unclaimed, including the body.
                let structurize_region_state =
                    mem::take(&mut self.structurize_region_state).into_iter().chain([(
                        self.func_def_body.body,
                        StructurizeRegionState::Ready {
                            accumulated_backedge_count: IncomingEdgeCount::default(),

                            region_deferred_edges: func_body_deferred_edges,
                        },
                    )]);
                for (target, state) in structurize_region_state {
                    if let StructurizeRegionState::Ready { region_deferred_edges, .. } = state {
                        self.rebuild_cfg_from_unclaimed_region_deferred_edges(
                            target,
                            region_deferred_edges,
                        );
                    }
                }
            }
        }

        // The last step of structurization is applying rewrites accumulated
        // while structurizing (i.e. `region_input_rewrites`).
        //
        // FIXME(eddyb) obsolete this by fully taking advantage of hermeticity,
        // and only replacing `Value::RegionInput { region, .. }` within
        // `region`'s children, shallowly, whenever `region` gets claimed.
        self.func_def_body.inner_in_place_transform_with(&mut RegionInputRewrites::rewrite_all(
            &self.region_input_rewrites,
        ));
    }

    fn try_claim_edge_bundle(
        &mut self,
        edge_bundle: IncomingEdgeBundle<Region>,
    ) -> Result<ClaimedRegion, IncomingEdgeBundle<Region>> {
        let target = edge_bundle.target;

        // Always attempt structurization before checking the `IncomingEdgeCount`,
        // to be able to make use of backedges (if any were found).
        if self.structurize_region_state.get(&target).is_none() {
            self.structurize_region(target);
        }

        let backedge_count = match self.structurize_region_state[&target] {
            // This `try_claim_edge_bundle` call is itself a backedge, and it's
            // coherent to not let any of them claim the loop itself, and only
            // allow claiming the whole loop (if successfully structurized).
            StructurizeRegionState::InProgress => IncomingEdgeCount::default(),

            StructurizeRegionState::Ready { accumulated_backedge_count, .. } => {
                accumulated_backedge_count
            }

            StructurizeRegionState::Claimed => {
                unreachable!("cfg::Structurizer::try_claim_edge_bundle: already claimed");
            }
        };

        if self.incoming_edge_counts_including_loop_exits[target]
            != edge_bundle.accumulated_count + backedge_count
        {
            return Err(edge_bundle);
        }

        let state =
            self.structurize_region_state.insert(target, StructurizeRegionState::Claimed).unwrap();

        let mut deferred_edges = match state {
            StructurizeRegionState::InProgress => unreachable!(
                "cfg::Structurizer::try_claim_edge_bundle: cyclic calls \
                 should not get this far"
            ),

            StructurizeRegionState::Ready { region_deferred_edges, .. } => region_deferred_edges,

            StructurizeRegionState::Claimed => {
                // Handled above.
                unreachable!()
            }
        };

        let mut backedge = None;
        if backedge_count != IncomingEdgeCount::default() {
            (backedge, deferred_edges) =
                deferred_edges.split_out_target(DeferredTarget::Region(target));
        }

        // If the target contains any backedge to itself, that's a loop, with:
        // * entry: `edge_bundle` (unconditional, i.e. `do`-`while`-like)
        // * body: `target`
        // * repeat ("continue") edge: `backedge` (with its `condition`)
        // * exit ("break") edges: `deferred_edges`
        let structured_body = if let Some(backedge) = backedge {
            let DeferredEdgeBundle { condition: repeat_condition, edge_bundle: backedge } =
                backedge;
            let body = target;

            // HACK(eddyb) due to `Loop` `Node`s not being hermetic on
            // the output side yet (i.e. they still have SSA-like semantics),
            // it gets wrapped in a `Region`, which can be as hermetic as
            // the loop body itself was originally.
            // NOTE(eddyb) both input declarations and the child `Loop` node are
            // added later down below, after the `Loop` node is created.
            let wrapper_region = self.func_def_body.regions.define(self.cx, RegionDef::default());

            // Any loop body region inputs, which must receive values from both
            // the loop entry and the backedge, become explicit "loop state",
            // starting as `initial_inputs` and being replaced with body outputs
            // after every loop iteration.
            //
            // FIXME(eddyb) `Loop` `Node`s should be changed to be hermetic
            // and have the loop state be output from the whole node itself,
            // for any outside uses of values defined within the loop body.
            let body_def = self.func_def_body.at_mut(body).def();
            let original_input_decls = mem::take(&mut body_def.inputs);
            assert!(body_def.outputs.is_empty());

            // HACK(eddyb) some dataflow through the loop body is redundant,
            // and can be lifted out of it, but the worst part is that applying
            // the replacement requires leaving alone all the non-redundant
            // `body` region inputs at the same time, and it's not really
            // feasible to move `body`'s children into a new region without
            // wasting it completely (i.e. can't swap with `wrapper_region`).
            let mut initial_inputs = SmallVec::<[_; 2]>::new();
            let body_input_rewrites = RegionInputRewrites::RenumberOrReplaceWith(
                backedge
                    .target_inputs
                    .into_iter()
                    .enumerate()
                    .map(|(original_idx, mut backedge_value)| {
                        RegionInputRewrites::rewrite_all(&self.region_input_rewrites)
                            .transform_value_use(&backedge_value)
                            .apply_to(&mut backedge_value);

                        let original_idx = u32::try_from(original_idx).unwrap();
                        if backedge_value
                            == (Value::RegionInput { region: body, input_idx: original_idx })
                        {
                            // FIXME(eddyb) does this have to be general purpose,
                            // or could this be handled as `None` with a single
                            // `wrapper_region` per `RegionInputRewrites`?
                            Err(Value::RegionInput {
                                region: wrapper_region,
                                input_idx: original_idx,
                            })
                        } else {
                            let renumbered_idx = u32::try_from(body_def.inputs.len()).unwrap();
                            initial_inputs.push(Value::RegionInput {
                                region: wrapper_region,
                                input_idx: original_idx,
                            });
                            body_def.inputs.push(original_input_decls[original_idx as usize]);
                            body_def.outputs.push(backedge_value);
                            Ok(renumbered_idx)
                        }
                    })
                    .collect(),
            );
            self.region_input_rewrites.insert(body, body_input_rewrites);

            assert_eq!(initial_inputs.len(), body_def.inputs.len());
            assert_eq!(body_def.outputs.len(), body_def.inputs.len());

            let repeat_condition = self.materialize_lazy_cond(&repeat_condition);
            let loop_node = self.func_def_body.nodes.define(
                self.cx,
                NodeDef {
                    // FIXME(eddyb) could it be possible to synthesize attrs
                    // from `ControlInst`s' attrs and/or `OpLoopMerge`'s?
                    attrs: AttrSet::default(),
                    kind: NodeKind::Loop { repeat_condition },
                    inputs: initial_inputs,
                    child_regions: [body].into_iter().collect(),
                    outputs: [].into_iter().collect(),
                }
                .into(),
            );

            let wrapper_region_def = &mut self.func_def_body.regions[wrapper_region];
            wrapper_region_def.inputs = original_input_decls;
            wrapper_region_def.children.insert_last(loop_node, &mut self.func_def_body.nodes);

            // HACK(eddyb) we've treated loop exits as extra "false edges", so
            // here they have to be added to the loop (potentially unlocking
            // structurization to the outside of the loop, in the caller).
            if let Some(exit_targets) = self.loop_header_to_exit_targets.get(&target) {
                for &exit_target in exit_targets {
                    // FIXME(eddyb) what if this is `None`, is that impossible?
                    if let Some(exit_edge_bundle) = deferred_edges
                        .get_edge_bundle_mut_by_target(DeferredTarget::Region(exit_target))
                    {
                        exit_edge_bundle.accumulated_count += IncomingEdgeCount::ONE;
                    }
                }
            }

            wrapper_region
        } else {
            target
        };
        let IncomingEdgeBundle { attrs, target: _, accumulated_count: _, target_inputs } =
            edge_bundle;

        // FIXME(eddyb) this loses `attrs`.
        let _ = attrs;

        Ok(ClaimedRegion { structured_body, structured_body_inputs: target_inputs, deferred_edges })
    }

    /// Structurize `region` by absorbing into it the entire CFG subgraph which
    /// it dominates (and deferring any other edges to the rest of the CFG).
    ///
    /// The output of this process is stored in, and any other bookkeeping is
    /// done through, `self.structurize_region_state[region]`.
    ///
    /// See also [`StructurizeRegionState`]'s docs.
    fn structurize_region(&mut self, region: Region) {
        {
            let old_state =
                self.structurize_region_state.insert(region, StructurizeRegionState::InProgress);
            if let Some(old_state) = old_state {
                unreachable!(
                    "cfg::Structurizer::structurize_region: \
                     already {}, when attempting to start structurization",
                    match old_state {
                        StructurizeRegionState::InProgress => "in progress (cycle detected)",
                        StructurizeRegionState::Ready { .. } => "completed",
                        StructurizeRegionState::Claimed => "claimed",
                    }
                );
            }
        }

        let control_inst_on_exit = self
            .func_def_body
            .unstructured_cfg
            .as_mut()
            .unwrap()
            .control_inst_on_exit_from
            .remove(region)
            .expect(
                "cfg::Structurizer::structurize_region: missing \
                   `ControlInst` (CFG wasn't unstructured in the first place?)",
            );

        // Start with the concatenation of `region` and `control_inst_on_exit`,
        // always appending `Node`s (including the children of entire
        // `ClaimedRegion`s) to `region`'s definition itself.
        let mut deferred_edges = {
            let ControlInst { attrs, kind, inputs, targets, target_inputs } = control_inst_on_exit;

            let target_regions: SmallVec<[_; 8]> = targets
                .iter()
                .map(|&target| {
                    self.try_claim_edge_bundle(IncomingEdgeBundle {
                        attrs: if targets.len() == 1 { attrs } else { AttrSet::default() },
                        target,
                        accumulated_count: IncomingEdgeCount::ONE,
                        target_inputs: target_inputs.get(&target).cloned().unwrap_or_default(),
                    })
                    .map_err(|edge_bundle| {
                        // HACK(eddyb) special-case "shared `unreachable`" to
                        // always inline it and avoid awkward "merges".
                        // FIXME(eddyb) should this be in a separate CFG pass?
                        // (i.e. is there a risk of other logic needing this?)
                        let target_is_trivial_unreachable =
                            match self.structurize_region_state.get(&edge_bundle.target) {
                                Some(StructurizeRegionState::Ready {
                                    region_deferred_edges: DeferredEdgeBundleSet::Unreachable,
                                    ..
                                }) => {
                                    // FIXME(eddyb) DRY this "is empty region" check.
                                    self.func_def_body
                                        .at(edge_bundle.target)
                                        .at_children()
                                        .into_iter()
                                        .next()
                                        .is_none()
                                }
                                _ => false,
                            };
                        if target_is_trivial_unreachable {
                            DeferredEdgeBundleSet::Unreachable
                        } else {
                            DeferredEdgeBundleSet::Always {
                                target: DeferredTarget::Region(edge_bundle.target),
                                edge_bundle: edge_bundle.with_target(()),
                            }
                        }
                    })
                })
                .collect();

            match kind {
                ControlInstKind::Unreachable => {
                    // FIXME(eddyb) this loses `attrs`.
                    let _ = attrs;

                    assert_eq!((inputs.len(), target_regions.len()), (0, 0));

                    // FIXME(eddyb) this may result in lost optimizations over
                    // actually encoding it in `Node`/`Region`
                    // (e.g. a new `NodeKind`, or replacing region `outputs`),
                    // but it's simpler to handle it like this.
                    //
                    // NOTE(eddyb) actually, this encoding is lossless *during*
                    // structurization, and a divergent region can only end up as:
                    // - the function body, where it implies the function can
                    //   never actually return: not fully structurized currently
                    //   (but only for a silly reason, and is entirely fixable)
                    // - a `Select` case, where it implies that case never merges
                    //   back into the `Select` node, and potentially that the
                    //   case can never be taken: this is where a structured
                    //   encoding can be introduced, by pruning unreachable
                    //   cases, and potentially even introducing `assume`s
                    // - a `Loop` body is not actually possible when divergent
                    //   (as there can be no backedge to form a cyclic CFG)
                    DeferredEdgeBundleSet::Unreachable
                }

                ControlInstKind::ExitInvocation(kind) => {
                    assert_eq!(target_regions.len(), 0);

                    let node = self.func_def_body.nodes.define(
                        self.cx,
                        NodeDef {
                            attrs,
                            kind: NodeKind::ExitInvocation(kind),
                            inputs,
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        }
                        .into(),
                    );
                    self.func_def_body.regions[region]
                        .children
                        .insert_last(node, &mut self.func_def_body.nodes);

                    DeferredEdgeBundleSet::Unreachable
                }

                ControlInstKind::Return => {
                    assert_eq!(target_regions.len(), 0);

                    DeferredEdgeBundleSet::Always {
                        target: DeferredTarget::Return,
                        edge_bundle: IncomingEdgeBundle {
                            attrs,
                            accumulated_count: IncomingEdgeCount::default(),
                            target: (),
                            target_inputs: inputs,
                        },
                    }
                }

                ControlInstKind::Branch => {
                    assert_eq!(inputs.len(), 0);

                    self.append_maybe_claimed_region(
                        region,
                        target_regions.into_iter().exactly_one().ok().unwrap(),
                    )
                }

                ControlInstKind::SelectBranch(kind) => {
                    assert_eq!(inputs.len(), 1);

                    let scrutinee = inputs[0];

                    self.structurize_select_into(region, attrs, kind, Ok(scrutinee), target_regions)
                }
            }
        };

        // Try to resolve deferred edges that may have accumulated, and keep
        // going until there's no more deferred edges that can be claimed.
        loop {
            // FIXME(eddyb) this should try to take as many edges as possible,
            // and incorporate them all at once, potentially with a switch instead
            // of N individual branches with their own booleans etc.
            let (claimed, else_deferred_edges) = deferred_edges.split_out_matching(|deferred| {
                let deferred_target = deferred.edge_bundle.target;
                let DeferredEdgeBundle { condition, edge_bundle } = match deferred_target {
                    DeferredTarget::Region(target) => deferred.with_target(target),
                    DeferredTarget::Return => return Err(deferred),
                };

                let edge_bundle_attrs = edge_bundle.attrs;
                match self.try_claim_edge_bundle(edge_bundle) {
                    Ok(claimed_region) => Ok((edge_bundle_attrs, condition, claimed_region)),

                    Err(new_edge_bundle) => {
                        let new_target = DeferredTarget::Region(new_edge_bundle.target);
                        Err(DeferredEdgeBundle {
                            condition,
                            edge_bundle: new_edge_bundle.with_target(new_target),
                        })
                    }
                }
            });
            let Some((branch_attrs, condition, then_region)) = claimed else {
                deferred_edges = else_deferred_edges;
                break;
            };

            deferred_edges = self.structurize_select_into(
                region,
                branch_attrs,
                SelectionKind::BoolCond,
                Err(&condition),
                [Ok(then_region), Err(else_deferred_edges)].into_iter().collect(),
            );
        }

        // Cache the edge count for backedges (which later get turned into loops).
        let accumulated_backedge_count = deferred_edges
            .get_edge_bundle_by_target(DeferredTarget::Region(region))
            .map(|backedge| backedge.accumulated_count)
            .unwrap_or_default();

        let old_state = self.structurize_region_state.insert(
            region,
            StructurizeRegionState::Ready {
                accumulated_backedge_count,
                region_deferred_edges: deferred_edges,
            },
        );
        if !matches!(old_state, Some(StructurizeRegionState::InProgress)) {
            unreachable!(
                "cfg::Structurizer::structurize_region: \
                 already {}, when attempting to store structurization result",
                match old_state {
                    None => "reverted to missing (removed from the map?)",
                    Some(StructurizeRegionState::InProgress) => unreachable!(),
                    Some(StructurizeRegionState::Ready { .. }) => "completed",
                    Some(StructurizeRegionState::Claimed) => "claimed",
                }
            );
        }
    }

    /// Append to `parent_region` a new `Select` [`Node`] built from
    /// partially structured `cases`, merging all of their `deferred_edges`
    /// together into a combined `DeferredEdgeBundleSet` (which gets returned).
    //
    // FIXME(eddyb) handle `unreachable` cases losslessly.
    fn structurize_select_into(
        &mut self,
        parent_region: Region,
        // FIXME(eddyb) semantically filter these, maybe focus on debuginfo?
        attrs: AttrSet,
        kind: SelectionKind,
        scrutinee: Result<Value, &LazyCond>,
        mut cases: SmallVec<[Result<ClaimedRegion, DeferredEdgeBundleSet>; 8]>,
    ) -> DeferredEdgeBundleSet {
        // HACK(eddyb) don't nest a sole convergent case inside the `Select`,
        // and instead prefer early convergence (see also `EventualCfgExits`).
        // NOTE(eddyb) this also happens to handle the situation where `Select`
        // isn't even needed (i.e. the other cases don't even have side-effects),
        // via the `any_non_empty_case` check (after taking `convergent_case`).
        // FIXME(eddyb) consider introducing some kind of `assume` for `scrutinee`,
        // to preserve its known value (whenever `convergent_case` is reached).
        let convergent_cases = cases.iter_mut().filter(|case| match case {
            Ok(ClaimedRegion { deferred_edges, .. }) | Err(deferred_edges) => {
                !matches!(deferred_edges, DeferredEdgeBundleSet::Unreachable)
            }
        });
        if let Ok(convergent_case) = convergent_cases.exactly_one() {
            // HACK(eddyb) this relies on `structurize_select_into`'s behavior
            // for `unreachable` cases being largely equivalent to empty cases.
            let convergent_case =
                mem::replace(convergent_case, Err(DeferredEdgeBundleSet::Unreachable));

            // FIXME(eddyb) avoid needing recursion, by instead changing the
            // "`Select` node insertion cursor" (into `parent_region`), and
            // stashing `convergent_case`'s deferred edges to return later.
            let deferred_edges =
                self.structurize_select_into(parent_region, attrs, kind, scrutinee, cases);
            assert!(matches!(deferred_edges, DeferredEdgeBundleSet::Unreachable));

            // The sole convergent case goes in the `parent_region`, and its
            // relationship with the `Select` (if it was even necessary at all)
            // is only at most one of side-effect sequencing.
            return self.append_maybe_claimed_region(parent_region, convergent_case);
        }

        // Extends a debug location "forward", from `initial_loc` (typically
        // the location of the conditional branch/switch being structurized),
        // to end at a later location in the same file, before any merge targets,
        // but after all of the cases (i.e. returns `None` if the `Select` is
        // made up of disjoint source ranges).
        let extend_dbg_src_loc =
            |this: &Self,
             mut initial_loc: DbgSrcLoc,
             cases: &[Result<ClaimedRegion, DeferredEdgeBundleSet>]| {
                // HACK(eddyb) see comment on `if initial_start_line == start_line` below.
                let mut shrink_initial_start_col = initial_loc.start_line_col.1;

                let mut relevant_dbg_src_loc = |attrs: AttrSet| {
                    attrs
                        .dbg_src_loc(this.cx)
                        .filter(|dbg_src_loc| {
                            // FIXME(eddyb) walk up `inlined_callee_name_and_call_site`
                            // in case `initial_loc` is e.g. next to some callsite.
                            dbg_src_loc.file_path == initial_loc.file_path
                                && dbg_src_loc.inlined_callee_name_and_call_site
                                    == initial_loc.inlined_callee_name_and_call_site
                        })
                        .filter(|dbg_src_loc| {
                            let (initial_start_line, initial_start_col) =
                                initial_loc.start_line_col;
                            let (start_line, start_col) = dbg_src_loc.start_line_col;

                            if (initial_start_line, initial_start_col) <= (start_line, start_col) {
                                return true;
                            }

                            // HACK(eddyb) this only exists because the debuginfo
                            // emited by Rust-GPU for `if cond { ... } else { ... }`'s
                            // conditional branch points to the start of `cond`,
                            // instead of at the `if`, but the merges *do* point
                            // at the whole `if` (or rather its start).
                            if initial_start_line == start_line {
                                shrink_initial_start_col = shrink_initial_start_col.min(start_col);
                            }

                            false
                        })
                };

                let max_cases_line_col = cases
                    .iter()
                    .filter_map(|case| {
                        let &ClaimedRegion { structured_body, .. } = case.as_ref().ok()?;
                        // FIXME(eddyb) maybe there should be a `FuncAt<Region>`
                        // helper for "debug locations from all `Block` `DataInst`s
                        // and non-`Block` `Node`"? (i.e. only flattening `Block`s)
                        this.func_def_body
                            .at(structured_body)
                            .at_children()
                            .into_iter()
                            .flat_map(|func_at_child| {
                                let child_def = func_at_child.def();
                                if let NodeKind::Block { insts } = child_def.kind {
                                    Either::Left(
                                        func_at_child
                                            .at(insts)
                                            .into_iter()
                                            .map(|func_at_inst| func_at_inst.def().attrs),
                                    )
                                } else {
                                    Either::Right([child_def.attrs].into_iter())
                                }
                            })
                            .rev()
                            .find_map(&mut relevant_dbg_src_loc)
                            .map(|dbg_src_loc| dbg_src_loc.end_line_col)
                    })
                    .max();
                let min_merges_line_col = cases
                    .iter()
                    .flat_map(|case| {
                        let case_deferred_edges = match case {
                            Ok(ClaimedRegion { deferred_edges, .. }) | Err(deferred_edges) => {
                                deferred_edges
                            }
                        };
                        case_deferred_edges.iter_targets_with_edge_bundle().map(|(_, e)| e.attrs)
                    })
                    .filter_map(&mut relevant_dbg_src_loc)
                    .map(|dbg_src_loc| dbg_src_loc.start_line_col)
                    .min();

                // HACK(eddyb) see comment on `if initial_start_line == start_line` above.
                initial_loc.start_line_col.1 = shrink_initial_start_col;

                // HACK(eddyb) prefers merges because otherwise the end location
                // ends up pointing into one of the cases (e.g. at the end of
                // `expr` in `if ... { ... } else { ... expr }`).
                // FIXME(eddyb) this doesn't pan out because the merges point
                // at the *start* of the `if` in Rust-GPU-emitted debuginfo
                // currently (it's likely a range being shrunk to its start
                // point - Rust-GPU's custom debuginfo could probably fix that).
                let end_line_col =
                    min_merges_line_col.or(max_cases_line_col).unwrap_or(initial_loc.end_line_col);

                if let Some(max_cases_line_col) = max_cases_line_col {
                    // NOTE(eddyb) can only realistically be the case if
                    // some of the merges are inside a larger high-level
                    // control-flow construct that doesn't map to fully
                    // structured control-flow (e.g. `switch` fallthrough).
                    if max_cases_line_col > end_line_col {
                        return None;
                    }
                }

                Some(DbgSrcLoc { end_line_col, ..initial_loc })
            };

        // Support lazily defining the `Select` node, as soon as it's necessary
        // (i.e. to plumb per-case dataflow through `Value::NodeOutput`s),
        // but also if any of the cases actually have non-empty regions, which
        // is checked after the special-cases (which return w/o a `Select` at all).
        //
        // FIXME(eddyb) some cases may be `unreachable`, and that's erased here.
        let mut cached_select_node = None;
        let mut non_move_kind = Some(kind);
        let mut get_or_define_select_node = |this: &mut Self, cases: &[_]| {
            *cached_select_node.get_or_insert_with(|| {
                let mut attrs = attrs;
                if let Some(select_dbg_src_loc) = attrs.dbg_src_loc(this.cx)
                    && let Some(dbg_src_loc) = extend_dbg_src_loc(this, select_dbg_src_loc, cases)
                {
                    attrs.set_dbg_src_loc(this.cx, dbg_src_loc);
                }

                let kind = non_move_kind.take().unwrap();
                let cases = cases
                    .iter()
                    .map(|case| {
                        let case_region = match case {
                            &Ok(ClaimedRegion { structured_body, .. }) => structured_body,
                            Err(_) => {
                                this.func_def_body.regions.define(this.cx, RegionDef::default())
                            }
                        };

                        // FIXME(eddyb) should these be asserts that it's already empty?
                        let case_region_def = this.func_def_body.at_mut(case_region).def();
                        case_region_def.outputs.clear();
                        case_region
                    })
                    .collect();
                let scrutinee =
                    scrutinee.unwrap_or_else(|lazy_cond| this.materialize_lazy_cond(lazy_cond));
                let select_node = this.func_def_body.nodes.define(
                    this.cx,
                    NodeDef {
                        attrs,
                        kind: NodeKind::Select(kind),
                        inputs: [scrutinee].into_iter().collect(),
                        child_regions: cases,
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );
                this.func_def_body.regions[parent_region]
                    .children
                    .insert_last(select_node, &mut this.func_def_body.nodes);
                select_node
            })
        };

        // Ensure the `Select` exists if needed for any per-case side-effects.
        let any_non_empty_case = cases.iter().any(|case| {
            case.as_ref().is_ok_and(|&ClaimedRegion { structured_body, .. }| {
                self.func_def_body.at(structured_body).at_children().into_iter().next().is_some()
            })
        });
        if any_non_empty_case {
            get_or_define_select_node(self, &cases);
        }

        // Gather the full set of deferred edges (and returns).
        struct DeferredTargetSummary {
            input_count: usize,
            total_edge_count: IncomingEdgeCount,
        }
        let mut deferred_targets = FxIndexMap::default();
        for case in &cases {
            let case_deferred_edges = match case {
                Ok(ClaimedRegion { deferred_edges, .. }) | Err(deferred_edges) => deferred_edges,
            };
            for (target, edge_bundle) in case_deferred_edges.iter_targets_with_edge_bundle() {
                let input_count = edge_bundle.target_inputs.len();

                let summary = deferred_targets.entry(target).or_insert(DeferredTargetSummary {
                    input_count,
                    total_edge_count: IncomingEdgeCount::default(),
                });
                assert_eq!(summary.input_count, input_count);
                summary.total_edge_count += edge_bundle.accumulated_count;
            }
        }

        // FIXME(eddyb) `region_input_rewrites` mappings, generated
        // for every `ClaimedRegion` that has been merged into a larger region,
        // only get applied after structurization fully completes, but here it's
        // very useful to have the fully resolved values across all `cases`'
        // incoming/outgoing edges (note, however, that within outgoing edges,
        // i.e. `case_deferred_edges`' `target_inputs`, `Value::RegionInput`
        // are not resolved using the contents of `case_structured_body_inputs`,
        // which is kept hermetic until just before `structurize_select` returns).
        for case in &mut cases {
            let (case_structured_body_inputs, case_deferred_edges) = match case {
                Ok(ClaimedRegion { structured_body_inputs, deferred_edges, .. }) => {
                    (&mut structured_body_inputs[..], deferred_edges)
                }
                Err(deferred_edges) => (&mut [][..], deferred_edges),
            };
            let all_values = case_structured_body_inputs.iter_mut().chain(
                case_deferred_edges
                    .iter_targets_with_edge_bundle_mut()
                    .flat_map(|(_, edge_bundle)| &mut edge_bundle.target_inputs),
            );
            for v in all_values {
                RegionInputRewrites::rewrite_all(&self.region_input_rewrites)
                    .transform_value_use(v)
                    .apply_to(v);
            }
        }

        // Merge all `deferred_edges` by plumbing their per-case `target_input`s
        // (into per-case region outputs, and therefore the `Select` outputs)
        // out of all cases that can reach them, with undef constants used to
        // fill any gaps (i.e. for the targets not reached through each case),
        // while deferred conditions are collected separately (for `LazyCond`).
        let deferred_edges = deferred_targets.into_iter().map(|(target, target_summary)| {
            let DeferredTargetSummary { input_count, total_edge_count } = target_summary;

            // HACK(eddyb) `Err` wraps only `LazyCond::{Undef,False}`, which allows
            // distinguishing between "not taken" and "not even reachable".
            let per_case_deferred: SmallVec<[Result<DeferredEdgeBundle<()>, LazyCond>; 8]> = cases
                .iter_mut()
                .map(|case| match case {
                    Ok(ClaimedRegion { deferred_edges, .. }) | Err(deferred_edges) => {
                        if let DeferredEdgeBundleSet::Unreachable = deferred_edges {
                            Err(LazyCond::Undef)
                        } else {
                            deferred_edges
                                .steal_deferred_by_target_without_removal(target)
                                .ok_or(LazyCond::False)
                        }
                    }
                })
                .collect();

            let target_inputs = (0..input_count)
                .map(|target_input_idx| {
                    let per_case_target_input = per_case_deferred.iter().map(|per_case_deferred| {
                        per_case_deferred.as_ref().ok().map(
                            |DeferredEdgeBundle { edge_bundle, .. }| {
                                edge_bundle.target_inputs[target_input_idx]
                            },
                        )
                    });

                    // Avoid introducing dynamic dataflow when the same value is
                    // used across all cases (which can reach this `target`).
                    let unique_target_input_value = per_case_target_input
                        .clone()
                        .zip_eq(&cases)
                        .filter_map(|(v, case)| Some((v?, case)))
                        .map(|(v, case)| {
                            // If possible, resolve `v` to a `Value` valid in
                            // `parent_region` (i.e. the `Select` node parent).
                            match case {
                                // `case`'s `structured_body` effectively "wraps"
                                // its `deferred_edges` (where `v` came from),
                                // so values from `parent_region` can only be
                                // hermetically used via `structured_body` inputs.
                                Ok(ClaimedRegion {
                                    structured_body,
                                    structured_body_inputs,
                                    ..
                                }) => match v {
                                    Value::Const(_) => Ok(v),
                                    Value::RegionInput { region, input_idx }
                                        if region == *structured_body =>
                                    {
                                        Ok(structured_body_inputs[input_idx as usize])
                                    }
                                    _ => Err(()),
                                },

                                // `case` has no region of its own, so everything
                                // it carries is already from within `parent_region`.
                                Err(_) => Ok(v),
                            }
                        })
                        .dedup()
                        .exactly_one();
                    if let Ok(Ok(v)) = unique_target_input_value {
                        return v;
                    }

                    let ty = match target {
                        DeferredTarget::Region(target) => {
                            self.func_def_body.at(target).def().inputs[target_input_idx].ty
                        }
                        // HACK(eddyb) in the absence of `FuncDecl`, infer the
                        // type from each returned value (and require they match).
                        DeferredTarget::Return => per_case_target_input
                            .clone()
                            .flatten()
                            .map(|v| self.func_def_body.at(v).type_of(self.cx))
                            .dedup()
                            .exactly_one()
                            .ok()
                            .expect("mismatched `return`ed value types"),
                    };

                    let select_node = get_or_define_select_node(self, &cases);
                    let output_decls = &mut self.func_def_body.at_mut(select_node).def().outputs;
                    let output_idx = output_decls.len();
                    output_decls.push(NodeOutputDecl { attrs: AttrSet::default(), ty });
                    for (case_idx, v) in per_case_target_input.enumerate() {
                        let v = v.unwrap_or_else(|| Value::Const(self.const_undef(ty)));

                        let case_region =
                            self.func_def_body.at(select_node).def().child_regions[case_idx];
                        let outputs = &mut self.func_def_body.at_mut(case_region).def().outputs;
                        assert_eq!(outputs.len(), output_idx);
                        outputs.push(v);
                    }
                    Value::NodeOutput {
                        node: select_node,
                        output_idx: output_idx.try_into().unwrap(),
                    }
                })
                .collect();

            // Simplify `LazyCond`s eagerly, to reduce costs later on, or even
            // outright avoid defining the `Select` node in the first place.
            //
            // FIXME(eddyb) move all simplifications from `materialize_lazy_cond`
            // to here (allowing e.g. not defining the `Select` in more cases).
            let per_case_conds =
                per_case_deferred.iter().map(|per_case_deferred| match per_case_deferred {
                    Ok(DeferredEdgeBundle { condition, .. }) => condition,
                    Err(undef_or_false) => undef_or_false,
                });
            let condition = if per_case_conds
                .clone()
                .all(|cond| matches!(cond, LazyCond::Undef | LazyCond::True))
            {
                LazyCond::True
            } else {
                LazyCond::Merge(Rc::new(LazyCondMerge::Select {
                    node: get_or_define_select_node(self, &cases),
                    per_case_conds: per_case_conds.cloned().collect(),
                }))
            };

            DeferredEdgeBundle {
                condition,
                edge_bundle: IncomingEdgeBundle {
                    // FIXME(eddyb) merge debug locations when attributes differ.
                    attrs: per_case_deferred
                        .iter()
                        .filter_map(|d| d.as_ref().ok())
                        .map(|e| e.edge_bundle.attrs)
                        .unique()
                        .exactly_one()
                        .ok()
                        .unwrap_or_default(),
                    target,
                    accumulated_count: total_edge_count,
                    target_inputs,
                },
            }
        });
        let deferred_edges = deferred_edges.collect();

        // Only as the very last step, can per-case `region_inputs` be added to
        // `region_input_rewrites`.
        //
        // FIXME(eddyb) don't replace `Value::RegionInput { region, .. }`
        // with `region_inputs` when the `region` ends up a `Node` child,
        // but instead make all `Region`s entirely hermetic wrt inputs.
        #[allow(clippy::manual_flatten)]
        for case in cases {
            if let Ok(ClaimedRegion { structured_body, structured_body_inputs, .. }) = case
                && !structured_body_inputs.is_empty()
            {
                self.region_input_rewrites.insert(
                    structured_body,
                    RegionInputRewrites::ReplaceWith(structured_body_inputs),
                );
                self.func_def_body.at_mut(structured_body).def().inputs.clear();
            }
        }

        deferred_edges
    }

    // FIXME(eddyb) this should try to handle as many `LazyCond` as are available,
    // for incorporating them all at once, ideally with a switch instead
    // of N individual branches with their own booleans etc.
    fn materialize_lazy_cond(&mut self, cond: &LazyCond) -> Value {
        match cond {
            LazyCond::Undef => Value::Const(self.const_undef(self.type_bool)),
            LazyCond::False => Value::Const(self.const_false),
            LazyCond::True => Value::Const(self.const_true),

            // `LazyCond::Merge` was only created in the first place if a merge
            // was actually necessary, so there shouldn't be simplifications to
            // do here (i.e. the value provided is if `materialize_lazy_cond`
            // never gets called because the target has become unconditional).
            //
            // FIXME(eddyb) there is still an `if cond { true } else { false }`
            // special-case (repalcing with just `cond`), that cannot be expressed
            // currently in `LazyCond` itself (but maybe it should be).
            LazyCond::Merge(merge) => {
                let LazyCondMerge::Select { node, ref per_case_conds } = **merge;

                // HACK(eddyb) this won't actually allocate most of the time,
                // and avoids complications later below, when mutating the cases.
                let per_case_conds: SmallVec<[_; 8]> = per_case_conds
                    .into_iter()
                    .map(|cond| self.materialize_lazy_cond(cond))
                    .collect();

                let NodeDef { attrs: _, kind, inputs, child_regions, outputs: output_decls } =
                    &mut *self.func_def_body.nodes[node];
                let cases = match kind {
                    NodeKind::Select(kind) => {
                        assert_eq!(child_regions.len(), per_case_conds.len());

                        if let SelectionKind::BoolCond = kind {
                            let cond = inputs[0];

                            let [val_false, val_true] =
                                [self.const_false, self.const_true].map(Value::Const);
                            if per_case_conds[..] == [val_true, val_false] {
                                return cond;
                            } else if per_case_conds[..] == [val_false, val_true] {
                                // FIXME(eddyb) this could also be special-cased,
                                // at least when called from the topmost level,
                                // where which side is `false`/`true` doesn't
                                // matter (or we could even generate `!cond`?).
                                let _not_cond = cond;
                            }
                        }

                        child_regions
                    }
                    _ => unreachable!(),
                };

                let output_idx = u32::try_from(output_decls.len()).unwrap();
                output_decls.push(NodeOutputDecl { attrs: AttrSet::default(), ty: self.type_bool });

                for (&case, cond) in cases.iter().zip_eq(per_case_conds) {
                    let RegionDef { outputs, .. } = &mut self.func_def_body.regions[case];
                    outputs.push(cond);
                    assert_eq!(outputs.len(), output_decls.len());
                }

                Value::NodeOutput { node, output_idx }
            }
        }
    }

    /// Append to `parent_region` the children of `maybe_claimed_region` (if `Ok`),
    /// returning the `DeferredEdgeBundleSet` from `maybe_claimed_region`.
    //
    // FIXME(eddyb) the name isn't great, but e.g. "absorb into" would also be
    // weird (and on top of that, the append direction can be tricky to express).
    fn append_maybe_claimed_region(
        &mut self,
        parent_region: Region,
        maybe_claimed_region: Result<ClaimedRegion, DeferredEdgeBundleSet>,
    ) -> DeferredEdgeBundleSet {
        match maybe_claimed_region {
            Ok(ClaimedRegion { structured_body, structured_body_inputs, deferred_edges }) => {
                if !structured_body_inputs.is_empty() {
                    self.region_input_rewrites.insert(
                        structured_body,
                        RegionInputRewrites::ReplaceWith(structured_body_inputs),
                    );
                }
                let new_children =
                    mem::take(&mut self.func_def_body.at_mut(structured_body).def().children);
                self.func_def_body.regions[parent_region]
                    .children
                    .append(new_children, &mut self.func_def_body.nodes);
                deferred_edges
            }
            Err(deferred_edges) => deferred_edges,
        }
    }

    /// When structurization is only partial, and there remain unclaimed regions,
    /// they have to be reintegrated into the CFG, putting back [`ControlInst`]s
    /// where `structurize_region` has taken them from.
    ///
    /// This function handles one region at a time to make it more manageable,
    /// despite it having a single call site (in a loop in `structurize_func`).
    fn rebuild_cfg_from_unclaimed_region_deferred_edges(
        &mut self,
        region: Region,
        mut deferred_edges: DeferredEdgeBundleSet,
    ) {
        assert!(
            self.structurize_region_state.is_empty(),
            "cfg::Structurizer::rebuild_cfg_from_unclaimed_region_deferred_edges:
             must only be called from `structurize_func`, \
             after it takes `structurize_region_state`"
        );

        // Build a chain of conditional branches to apply deferred edges.
        let mut control_source = Some(region);
        loop {
            let taken_then;
            (taken_then, deferred_edges) =
                deferred_edges.split_out_matching(|deferred| match deferred.edge_bundle.target {
                    DeferredTarget::Region(target) => {
                        Ok((deferred.condition, (target, deferred.edge_bundle.target_inputs)))
                    }
                    DeferredTarget::Return => Err(deferred),
                });
            let Some((condition, then_target_and_inputs)) = taken_then else {
                break;
            };
            let branch_source = control_source.take().unwrap();
            let else_target_and_inputs = match deferred_edges {
                // At most one deferral left, so it can be used as the "else"
                // case, or the branch left unconditional in its absence.
                DeferredEdgeBundleSet::Unreachable => None,
                DeferredEdgeBundleSet::Always {
                    target: DeferredTarget::Region(else_target),
                    edge_bundle,
                } => {
                    deferred_edges = DeferredEdgeBundleSet::Unreachable;
                    Some((else_target, edge_bundle.target_inputs))
                }

                // Either more branches, or a deferred return, are needed, so
                // the "else" case must be a `Region` that itself can
                // have a `ControlInst` attached to it later on.
                _ => {
                    let new_empty_region =
                        self.func_def_body.regions.define(self.cx, RegionDef::default());
                    control_source = Some(new_empty_region);
                    Some((new_empty_region, [].into_iter().collect()))
                }
            };

            let condition = Some(condition)
                .filter(|_| else_target_and_inputs.is_some())
                .map(|cond| self.materialize_lazy_cond(&cond));
            let branch_control_inst = ControlInst {
                attrs: AttrSet::default(),
                kind: if condition.is_some() {
                    ControlInstKind::SelectBranch(SelectionKind::BoolCond)
                } else {
                    ControlInstKind::Branch
                },
                inputs: condition.into_iter().collect(),
                targets: [&then_target_and_inputs]
                    .into_iter()
                    .chain(&else_target_and_inputs)
                    .map(|&(target, _)| target)
                    .collect(),
                target_inputs: [then_target_and_inputs]
                    .into_iter()
                    .chain(else_target_and_inputs)
                    .filter(|(_, inputs)| !inputs.is_empty())
                    .collect(),
            };
            assert!(
                self.func_def_body
                    .unstructured_cfg
                    .as_mut()
                    .unwrap()
                    .control_inst_on_exit_from
                    .insert(branch_source, branch_control_inst)
                    .is_none()
            );
        }

        let deferred_return = match deferred_edges {
            DeferredEdgeBundleSet::Unreachable => None,
            DeferredEdgeBundleSet::Always { target: DeferredTarget::Return, edge_bundle } => {
                Some(edge_bundle.target_inputs)
            }
            _ => unreachable!(),
        };

        let final_source = match control_source {
            Some(region) => region,
            None => {
                // The loop above handled all the targets, nothing left to do.
                assert!(deferred_return.is_none());
                return;
            }
        };

        // Final deferral is either a `Return` (if needed), or an `Unreachable`
        // (only when truly divergent, i.e. no `deferred_edges`/`deferred_return`).
        let final_control_inst = {
            let (kind, inputs) = match deferred_return {
                Some(return_values) => (ControlInstKind::Return, return_values),
                None => (ControlInstKind::Unreachable, [].into_iter().collect()),
            };
            ControlInst {
                attrs: AttrSet::default(),
                kind,
                inputs,
                targets: [].into_iter().collect(),
                target_inputs: FxIndexMap::default(),
            }
        };
        assert!(
            self.func_def_body
                .unstructured_cfg
                .as_mut()
                .unwrap()
                .control_inst_on_exit_from
                .insert(final_source, final_control_inst)
                .is_none()
        );
    }

    /// Create an undefined constant (as a placeholder where a value needs to be
    /// present, but won't actually be used), of type `ty`.
    fn const_undef(&self, ty: Type) -> Const {
        // FIXME(eddyb) SPIR-T should have native undef itself.
        let wk = &spv::spec::Spec::get().well_known;
        self.cx.intern(ConstDef {
            attrs: AttrSet::default(),
            ty,
            kind: ConstKind::SpvInst {
                spv_inst_and_const_inputs: Rc::new((wk.OpUndef.into(), [].into_iter().collect())),
            },
        })
    }
}
