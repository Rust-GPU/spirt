//! Control-flow graph (CFG) abstractions and utilities.

use crate::{
    spv, AttrSet, Const, ConstDef, ConstKind, Context, ControlNode, ControlNodeDef,
    ControlNodeKind, ControlNodeOutputDecl, ControlRegion, ControlRegionDef,
    EntityOrientedDenseMap, FuncDefBody, FxIndexMap, FxIndexSet, SelectionKind, Type, TypeKind,
    Value,
};
use itertools::{Either, Itertools};
use smallvec::SmallVec;
use std::mem;
use std::rc::Rc;

/// The control-flow graph (CFG) of a function, as control-flow instructions
/// ([`ControlInst`]s) attached to [`ControlRegion`]s, as an "action on exit", i.e.
/// "terminator" (while intra-region control-flow is strictly structured).
#[derive(Clone, Default)]
pub struct ControlFlowGraph {
    pub control_inst_on_exit_from: EntityOrientedDenseMap<ControlRegion, ControlInst>,

    // HACK(eddyb) this currently only comes from `OpLoopMerge`, and cannot be
    // inferred (because implies too strong of an ownership/uniqueness notion).
    pub loop_merge_to_loop_header: FxIndexMap<ControlRegion, ControlRegion>,
}

#[derive(Clone)]
pub struct ControlInst {
    pub attrs: AttrSet,

    pub kind: ControlInstKind,

    pub inputs: SmallVec<[Value; 2]>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    pub targets: SmallVec<[ControlRegion; 4]>,

    /// `target_inputs[region][input_idx]` is the [`Value`] that
    /// `Value::ControlRegionInput { region, input_idx }` will get on entry,
    /// where `region` must be appear at least once in `targets` - this is a
    /// separate map instead of being part of `targets` because it reflects the
    /// limitations of φ ("phi") nodes, which (unlike "basic block arguments")
    /// cannot tell apart multiple edges with the same source and destination.
    pub target_inputs: FxIndexMap<ControlRegion, SmallVec<[Value; 2]>>,
}

#[derive(Clone)]
pub enum ControlInstKind {
    /// Reaching this point in the control-flow is undefined behavior, e.g.:
    /// * a `SelectBranch` case that's known to be impossible
    /// * after a function call, where the function never returns
    ///
    /// Optimizations can take advantage of this information, to assume that any
    /// necessary preconditions for reaching this point, are never met.
    Unreachable,

    /// Leave the current function, optionally returning a value.
    Return,

    /// Leave the current invocation, similar to returning from every function
    /// call in the stack (up to and including the entry-point), but potentially
    /// indicating a fatal error as well.
    ExitInvocation(ExitInvocationKind),

    /// Unconditional branch to a single target.
    Branch,

    /// Branch to one of several targets, chosen by a single value input.
    SelectBranch(SelectionKind),
}

#[derive(Clone)]
pub enum ExitInvocationKind {
    SpvInst(spv::Inst),
}

impl ControlFlowGraph {
    /// Iterate over all [`ControlRegion`]s making up `func_def_body`'s CFG, in
    /// reverse post-order (RPO).
    ///
    /// RPO iteration over a CFG provides certain guarantees, most importantly
    /// that dominators are visited before the entire subgraph they dominate.
    pub fn rev_post_order(
        &self,
        func_def_body: &FuncDefBody,
    ) -> impl DoubleEndedIterator<Item = ControlRegion> {
        let mut post_order = SmallVec::<[_; 8]>::new();
        self.traverse_whole_func(
            func_def_body,
            &mut TraversalState {
                incoming_edge_counts: EntityOrientedDenseMap::new(),

                pre_order_visit: |_| {},
                post_order_visit: |region| post_order.push(region),

                // NOTE(eddyb) this doesn't impact semantics, but combined with
                // the final reversal, it should keep targets in the original
                // order in the cases when they didn't get deduplicated.
                reverse_targets: true,
            },
        );
        post_order.into_iter().rev()
    }
}

// HACK(eddyb) this only serves to disallow accessing `private_count` field of
// `IncomingEdgeCount`.
mod sealed {
    /// Opaque newtype for the count of incoming edges (into a [`ControlRegion`](crate::ControlRegion)).
    ///
    /// The private field prevents direct mutation or construction, forcing the
    /// use of [`IncomingEdgeCount::ONE`] and addition operations to produce some
    /// specific count (which would require explicit workarounds for misuse).
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub(super) struct IncomingEdgeCount(usize);

    impl IncomingEdgeCount {
        pub(super) const ONE: Self = Self(1);
    }

    impl std::ops::Add for IncomingEdgeCount {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
    }

    impl std::ops::AddAssign for IncomingEdgeCount {
        fn add_assign(&mut self, other: Self) {
            *self = *self + other;
        }
    }
}
use sealed::IncomingEdgeCount;

struct TraversalState<PreVisit: FnMut(ControlRegion), PostVisit: FnMut(ControlRegion)> {
    incoming_edge_counts: EntityOrientedDenseMap<ControlRegion, IncomingEdgeCount>,
    pre_order_visit: PreVisit,
    post_order_visit: PostVisit,

    // FIXME(eddyb) should this be a generic parameter for "targets iterator"?
    reverse_targets: bool,
}

impl ControlFlowGraph {
    fn traverse_whole_func(
        &self,
        func_def_body: &FuncDefBody,
        state: &mut TraversalState<impl FnMut(ControlRegion), impl FnMut(ControlRegion)>,
    ) {
        let func_at_body = func_def_body.at_body();

        // Quick sanity check that this is the right CFG for `func_def_body`.
        assert!(std::ptr::eq(func_def_body.unstructured_cfg.as_ref().unwrap(), self));
        assert!(func_at_body.def().outputs.is_empty());

        self.traverse(func_def_body.body, state);
    }

    fn traverse(
        &self,
        region: ControlRegion,
        state: &mut TraversalState<impl FnMut(ControlRegion), impl FnMut(ControlRegion)>,
    ) {
        // FIXME(eddyb) `EntityOrientedDenseMap` should have an `entry` API.
        if let Some(existing_count) = state.incoming_edge_counts.get_mut(region) {
            *existing_count += IncomingEdgeCount::ONE;
            return;
        }
        state.incoming_edge_counts.insert(region, IncomingEdgeCount::ONE);

        (state.pre_order_visit)(region);

        let control_inst = self
            .control_inst_on_exit_from
            .get(region)
            .expect("cfg: missing `ControlInst`, despite having left structured control-flow");

        let targets = control_inst.targets.iter().copied();
        let targets = if state.reverse_targets {
            Either::Left(targets.rev())
        } else {
            Either::Right(targets)
        };
        for target in targets {
            self.traverse(target, state);
        }

        (state.post_order_visit)(region);
    }
}

/// Minimal loop analysis, based on Tarjan's SCC (strongly connected components)
/// algorithm, applied recursively (for every level of loop nesting).
///
/// Here "minimal" means that each loops is the smallest CFG subgraph possible
/// (excluding any control-flow paths that cannot reach a backedge and cycle),
/// i.e. each loop is a CFG SCC (strongly connected component).
///
/// These "minimal loops" contrast with the "maximal loops" that the greedy
/// architecture of the structurizer would naively produce, with the main impact
/// of the difference being where loop exits (`break`s) "merge" (or "reconverge"),
/// which SPIR-V encodes via `OpLoopMerge`, and is significant for almost anything
/// where shared memory and/or subgroup ops can allow observing when invocations
/// "wait for others in the subgroup to exit the loop" (or when they fail to wait).
///
/// This analysis was added to because of two observations wrt "reconvergence":
/// 1. syntactic loops (from some high-level language), when truly structured
///    (i.e. only using `while`/`do`-`while` exit conditions, not `break` etc.),
///    *always* map to "minimal loops" on a CFG, as the only loop exit edge is
///    built-in, and no part of the syntactic "loop body" can be its successor
/// 2. more pragmatically, compiling shader languages to SPIR-V seems to (almost?)
///    always *either* fully preserve syntactic loops (via SPIR-V `OpLoopMerge`),
///    *or* structurize CFGs in a way that produces "minimal loops", which can
///    be misleading with explicit `break`s (moving user code from just before
///    the `break` to after the loop), but is less impactful than "maximal loops"
struct LoopFinder<'a> {
    cfg: &'a ControlFlowGraph,

    // FIXME(eddyb) this feels a bit inefficient (are many-exit loops rare?).
    loop_header_to_exit_targets: FxIndexMap<ControlRegion, FxIndexSet<ControlRegion>>,

    /// SCC accumulation stack, where CFG nodes collect during the depth-first
    /// traversal, and are only popped when their "SCC root" (loop header) is
    /// (note that multiple SCCs on the stack does *not* indicate SCC nesting,
    /// but rather a path between two SCCs, i.e. a loop *following* another).
    scc_stack: Vec<ControlRegion>,
    /// Per-CFG-node traversal state (often just pointing to a `scc_stack` slot).
    scc_state: EntityOrientedDenseMap<ControlRegion, SccState>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SccStackIdx(u32);

#[derive(PartialEq, Eq)]
enum SccState {
    /// CFG node has been reached and ended up somewhere on the `scc_stack`,
    /// where it will remain until the SCC it's part of will be completed.
    Pending(SccStackIdx),

    /// CFG node had been reached once, but is no longer on the `scc_stack`, its
    /// parent SCC having been completed (or it wasn't in an SCC to begin with).
    Complete(EventualCfgExits),
}

/// Summary of all the ways in which a CFG node may eventually leave the CFG.
///
// HACK(eddyb) a loop can reach a CFG subgraph that happens to always "diverge"
// (e.g. ending in `unreachable`, `ExitInvocation`, or even infinite loops,
// though those have other issues) and strictly speaking that would always be
// an edge leaving the SCC of the loop (as it can't reach a backedge), but it
// still shouldn't be treated as an exit because it doesn't reconverge to the
// rest of the function, i.e. it can't reach any `return`s, which is what this
// tracks in order to later make a more accurate decision wrt loop exits.
//
// NOTE(eddyb) only in the case where a loop *also* has non-"diverging" exits,
// do the "diverging" ones not get treated as exits, as the presence of both
// disambiguates `break`s from naturally "diverging" sections of the loop body
// (at least for CFGs built from languages without labelled `break` or `goto`,
// but even then it would be pretty convoluted to set up `break` to diverge,
// while `break some_outer_label` to reconverge to the rest of the function).
#[derive(Copy, Clone, Default, PartialEq, Eq)]
struct EventualCfgExits {
    // FIXME(eddyb) do the other situations need their own flags here?
    may_return_from_func: bool,
}

impl std::ops::BitOr for EventualCfgExits {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self { may_return_from_func: self.may_return_from_func | other.may_return_from_func }
    }
}
impl std::ops::BitOrAssign for EventualCfgExits {
    fn bitor_assign(&mut self, other: Self) {
        *self = *self | other;
    }
}

impl<'a> LoopFinder<'a> {
    fn new(cfg: &'a ControlFlowGraph) -> Self {
        Self {
            cfg,
            loop_header_to_exit_targets: FxIndexMap::default(),
            scc_stack: vec![],
            scc_state: EntityOrientedDenseMap::new(),
        }
    }

    /// Tarjan's SCC algorithm works by computing the "earliest" reachable node,
    /// from every node (often using the name `lowlink`), which will be equal
    /// to the origin node itself iff that node is an "SCC root" (loop header),
    /// and always point to an "earlier" node if a cycle (via loop backedge) was
    /// found from somewhere else in the SCC (i.e. from inside the loop body).
    ///
    /// Here we track stack indices (as the stack order is the traversal order),
    /// and distinguish the acyclic case to avoid treating most nodes as self-loops.
    //
    // FIXME(eddyb) name of the function is a bit clunky wrt its return type.
    fn find_earliest_scc_root_of(
        &mut self,
        node: ControlRegion,
    ) -> (Option<SccStackIdx>, EventualCfgExits) {
        let state_entry = self.scc_state.entry(node);
        if let Some(state) = &state_entry {
            return match *state {
                SccState::Pending(scc_stack_idx) => {
                    // HACK(eddyb) this means that `EventualCfgExits`s will be
                    // inconsistently observed across the `Pending` nodes of a
                    // loop body, but that is sound as it cannot feed into any
                    // `Complete` state until the loop header itself is complete,
                    // and the monotonic nature of `EventualCfgExits` means that
                    // the loop header will still get to see the complete picture.
                    (Some(scc_stack_idx), EventualCfgExits::default())
                }
                SccState::Complete(eventual_cfg_exits) => (None, eventual_cfg_exits),
            };
        }
        let scc_stack_idx = SccStackIdx(self.scc_stack.len().try_into().unwrap());
        self.scc_stack.push(node);
        *state_entry = Some(SccState::Pending(scc_stack_idx));

        let control_inst = self
            .cfg
            .control_inst_on_exit_from
            .get(node)
            .expect("cfg: missing `ControlInst`, despite having left structured control-flow");

        let mut eventual_cfg_exits = EventualCfgExits::default();

        if let ControlInstKind::Return = control_inst.kind {
            eventual_cfg_exits.may_return_from_func = true;
        }

        let earliest_scc_root = control_inst
            .targets
            .iter()
            .flat_map(|&target| {
                let (earliest_scc_root_of_target, eventual_cfg_exits_of_target) =
                    self.find_earliest_scc_root_of(target);
                eventual_cfg_exits |= eventual_cfg_exits_of_target;

                // HACK(eddyb) if one of the edges is already known to be a loop exit
                // (from `OpLoopMerge` specifically), treat it almost like a backedge,
                // but with the additional requirement that the loop header is already
                // on the stack (i.e. this `node` is reachable from that loop header).
                let root_candidate_from_loop_merge =
                    self.cfg.loop_merge_to_loop_header.get(&target).and_then(|&loop_header| {
                        match self.scc_state.get(loop_header) {
                            Some(&SccState::Pending(scc_stack_idx)) => Some(scc_stack_idx),
                            _ => None,
                        }
                    });

                earliest_scc_root_of_target.into_iter().chain(root_candidate_from_loop_merge)
            })
            .min();

        // If this node has been chosen as the root of an SCC, complete that SCC.
        if earliest_scc_root == Some(scc_stack_idx) {
            let scc_start = scc_stack_idx.0 as usize;

            // It's now possible to find all the loop exits: they're all the
            // edges from nodes of this SCC (loop) to nodes not in the SCC.
            let target_is_exit = |target| {
                match self.scc_state[target] {
                    SccState::Pending(i) => {
                        assert!(i >= scc_stack_idx);
                        false
                    }
                    SccState::Complete(eventual_cfg_exits_of_target) => {
                        let EventualCfgExits { may_return_from_func: loop_may_reconverge } =
                            eventual_cfg_exits;
                        let EventualCfgExits { may_return_from_func: target_may_reconverge } =
                            eventual_cfg_exits_of_target;

                        // HACK(eddyb) see comment on `EventualCfgExits` for why
                        // edges leaving the SCC aren't treated as loop exits
                        // when they're "more divergent" than the loop itself,
                        // i.e. if any edges leaving the SCC can reconverge,
                        // (and therefore the loop as a whole can reconverge)
                        // only those edges are kept as loop exits.
                        target_may_reconverge == loop_may_reconverge
                    }
                }
            };
            self.loop_header_to_exit_targets.insert(
                node,
                self.scc_stack[scc_start..]
                    .iter()
                    .flat_map(|&scc_node| {
                        self.cfg.control_inst_on_exit_from[scc_node].targets.iter().copied()
                    })
                    .filter(|&target| target_is_exit(target))
                    .collect(),
            );

            // Find nested loops by marking *only* the loop header as complete,
            // clearing loop body nodes' state, and recursing on them: all the
            // nodes outside the loop (otherwise reachable from within), and the
            // loop header itself, are already marked as complete, meaning that
            // all exits and backedges will be ignored, and the recursion will
            // only find more SCCs within the loop body (i.e. nested loops).
            self.scc_state[node] = SccState::Complete(eventual_cfg_exits);
            let loop_body_range = scc_start + 1..self.scc_stack.len();
            for &scc_node in &self.scc_stack[loop_body_range.clone()] {
                self.scc_state.remove(scc_node);
            }
            for i in loop_body_range.clone() {
                self.find_earliest_scc_root_of(self.scc_stack[i]);
            }
            assert_eq!(self.scc_stack.len(), loop_body_range.end);

            // Remove the entire SCC from the accumulation stack all at once.
            self.scc_stack.truncate(scc_start);

            return (None, eventual_cfg_exits);
        }

        // Not actually in an SCC at all, just some node outside any CFG cycles.
        if earliest_scc_root.is_none() {
            assert!(self.scc_stack.pop() == Some(node));
            self.scc_state[node] = SccState::Complete(eventual_cfg_exits);
        }

        (earliest_scc_root, eventual_cfg_exits)
    }
}

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
    loop_header_to_exit_targets: FxIndexMap<ControlRegion, FxIndexSet<ControlRegion>>,

    // HACK(eddyb) this also tracks all of `loop_header_to_exit_targets`, as
    // "false edges" from every loop header to each exit target of that loop,
    // which structurizing that loop consumes to "unlock" its own exits.
    incoming_edge_counts_including_loop_exits:
        EntityOrientedDenseMap<ControlRegion, IncomingEdgeCount>,

    /// `structurize_region_state[region]` tracks `.structurize_region(region)`
    /// progress/results (see also [`StructurizeRegionState`]'s docs).
    //
    // FIXME(eddyb) use `EntityOrientedDenseMap` (which lacks iteration by design).
    structurize_region_state: FxIndexMap<ControlRegion, StructurizeRegionState>,

    /// Accumulated replacements (caused by `target_inputs`s), i.e.:
    /// `Value::ControlRegionInput { region, input_idx }` must be replaced
    /// with `control_region_input_replacements[region][input_idx]`, as
    /// the original `region` cannot have be directly reused.
    control_region_input_replacements: EntityOrientedDenseMap<ControlRegion, SmallVec<[Value; 2]>>,
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
    target: T,
    accumulated_count: IncomingEdgeCount,

    /// The [`Value`]s that `Value::ControlRegionInput { region, .. }` will get
    /// on entry into `region`, through this "edge bundle".
    target_inputs: SmallVec<[Value; 2]>,
}

impl<T> IncomingEdgeBundle<T> {
    fn with_target<U>(self, target: U) -> IncomingEdgeBundle<U> {
        let IncomingEdgeBundle { target: _, accumulated_count, target_inputs } = self;
        IncomingEdgeBundle { target, accumulated_count, target_inputs }
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
/// (via per-case outputs and [`Value::ControlNodeOutput`], for each `Select`).
///
/// This should largely be equivalent to eagerly generating all region outputs
/// that might be needed, and then removing the unused ones, but this way we
/// never generate unused outputs, and can potentially even optimize away some
/// redundant dataflow (e.g. `if cond { true } else { false }` is just `cond`).
enum LazyCond {
    // FIXME(eddyb) remove `False` in favor of `Option<LazyCond>`?
    False,
    True,
    MergeSelect {
        control_node: ControlNode,
        // FIXME(eddyb) the lowest level of this ends up with a `Vec` containing
        // only `LazyCond::{False,True}`, and that could more easily be expressed
        // as e.g. a bitset? (or even `SmallVec<[bool; 16]>`, tho that's silly)
        per_case_conds: Vec<LazyCond>,
    },
}

/// A target for one of the edge bundles in a [`DeferredEdgeBundleSet`], mostly
/// separate from [`ControlRegion`] to allow expressing returns as well.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum DeferredTarget {
    Region(ControlRegion),

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
/// all the conditions will end up using disjoint [`LazyCond::MergeSelect`]s).
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

    // HACK(eddyb) this only exists because of `DeferredEdgeBundleSet`'s lossy
    // representation wrt conditions, so removal from a `DeferredEdgeBundleSet`
    // cannot be used for e.g. `Select` iterating over per-case deferreds.
    fn steal_deferred_by_target_without_removal(
        &mut self,
        search_target: DeferredTarget,
    ) -> Option<DeferredEdgeBundle<()>> {
        let steal_edge_bundle = |edge_bundle: &mut IncomingEdgeBundle<()>| IncomingEdgeBundle {
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
/// CFG subgraph (i.e. set of [`ControlRegion`]s previously connected by CFG edges),
/// which is effectively owned by the "claimer" and **must** be used for:
/// - the whole function body (if `deferred_edges` only contains `Return`)
/// - one of the cases of a `Select` node
/// - merging into a larger region (i.e. its nearest dominator)
//
// FIXME(eddyb) consider never having to claim the function body itself,
// by wrapping the CFG in a `ControlNode` instead.
struct ClaimedRegion {
    // FIXME(eddyb) find a way to clarify that this can differ from the target
    // of `try_claim_edge_bundle`, and also that `deferred_edges` are from the
    // perspective of being "inside" `structured_body` (wrt hermeticity).
    structured_body: ControlRegion,

    /// The [`Value`]s that `Value::ControlRegionInput { region: structured_body, .. }`
    /// will get on entry into `structured_body`, when this region ends up
    /// merged into a larger region, or as a child of a new [`ControlNode`].
    //
    // FIXME(eddyb) don't replace `Value::ControlRegionInput { region: structured_body, .. }`
    // with `region_inputs` when `structured_body` ends up a `ControlNode` child,
    // but instead make all `ControlRegion`s entirely hermetic wrt inputs.
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
                    let loop_header_to_exit_targets = {
                        let mut loop_finder = LoopFinder::new(cfg);
                        loop_finder.find_earliest_scc_root_of(func_def_body.body);
                        loop_finder.loop_header_to_exit_targets
                    };

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
            control_region_input_replacements: EntityOrientedDenseMap::new(),
        }
    }

    pub fn structurize_func(mut self) {
        // Don't even try to re-structurize functions.
        if self.func_def_body.unstructured_cfg.is_none() {
            return;
        }

        // FIXME(eddyb) it might work much better to have the unstructured CFG
        // wrapped in a `ControlNode` inside the function body, instead.
        let func_body_deferred_edges = {
            let func_entry_pseudo_edge = {
                let target = self.func_def_body.body;
                move || IncomingEdgeBundle {
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

        self.apply_value_replacements();
    }

    /// The last step of structurization is processing bulk replacements
    /// collected while structurizing (like `control_region_input_replacements`).
    fn apply_value_replacements(self) {
        // FIXME(eddyb) maybe this should be provided by `transform`.
        use crate::transform::*;
        struct ReplaceValueWith<F>(F);
        impl<F: Fn(Value) -> Option<Value>> Transformer for ReplaceValueWith<F> {
            fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
                self.0(*v).map_or(Transformed::Unchanged, Transformed::Changed)
            }
        }

        self.func_def_body.inner_in_place_transform_with(&mut ReplaceValueWith(|v| {
            // NOTE(eddyb) this needs to be able to apply multiple replacements,
            // due to the input potentially having redundantly chained `OpPhi`s.
            //
            // FIXME(eddyb) union-find-style "path compression" could record the
            // final value inside `self.control_region_input_replacements` while
            // replacements are being made, to avoid going through a chain more
            // than once (and some of these replacements could be applied early).
            let mut new_v = v;
            while let Value::ControlRegionInput { region, input_idx } = new_v {
                if let Some(replacements) = self.control_region_input_replacements.get(region) {
                    new_v = replacements[input_idx as usize];
                } else {
                    break;
                }
            }
            (v != new_v).then_some(new_v)
        }));
    }

    fn try_claim_edge_bundle(
        &mut self,
        edge_bundle: IncomingEdgeBundle<ControlRegion>,
    ) -> Result<ClaimedRegion, IncomingEdgeBundle<ControlRegion>> {
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

            // HACK(eddyb) due to `Loop` `ControlNode`s not being hermetic on
            // the output side yet (i.e. they still have SSA-like semantics),
            // it gets wrapped in a `ControlRegion`, which can be as hermetic as
            // the loop body itself was originally.
            let wrapper_region = self.func_def_body.control_regions.define(
                self.cx,
                ControlRegionDef {
                    inputs: self.func_def_body.at(body).def().inputs.clone(),
                    ..Default::default()
                },
            );

            // Any loop body region inputs, which must receive values from both
            // the loop entry and the backedge, become explicit "loop state",
            // starting as `initial_inputs` and being replaced with body outputs
            // after every loop iteration.
            //
            // FIXME(eddyb) `Loop` `ControlNode`s should be changed to be hermetic
            // and have the loop state be output from the whole node itself,
            // for any outside uses of values defined within the loop body.
            let body_def = self.func_def_body.at_mut(body).def();
            let initial_inputs =
                (0..edge_bundle.target_inputs.len()).map(|input_idx| Value::ControlRegionInput {
                    region: wrapper_region,
                    input_idx: input_idx.try_into().unwrap(),
                });
            body_def.outputs = backedge.target_inputs;
            assert_eq!(initial_inputs.len(), body_def.inputs.len());
            assert_eq!(body_def.outputs.len(), body_def.inputs.len());

            let repeat_condition = self.materialize_lazy_cond(repeat_condition);
            let loop_node = self.func_def_body.control_nodes.define(
                self.cx,
                ControlNodeDef {
                    kind: ControlNodeKind::Loop {
                        initial_inputs: initial_inputs.collect(),
                        body,
                        repeat_condition,
                    },
                    outputs: [].into_iter().collect(),
                }
                .into(),
            );

            self.func_def_body.control_regions[wrapper_region]
                .children
                .insert_last(loop_node, &mut self.func_def_body.control_nodes);

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
        Ok(ClaimedRegion {
            structured_body,
            structured_body_inputs: edge_bundle.target_inputs,
            deferred_edges,
        })
    }

    /// Structurize `region` by absorbing into it the entire CFG subgraph which
    /// it dominates (and deferring any other edges to the rest of the CFG).
    ///
    /// The output of this process is stored in, and any other bookkeeping is
    /// done through, `self.structurize_region_state[region]`.
    ///
    /// See also [`StructurizeRegionState`]'s docs.
    fn structurize_region(&mut self, region: ControlRegion) {
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

        /// Marker error type for unhandled [`ControlInst`]s below.
        struct UnsupportedControlInst(ControlInst);

        // Start with the concatenation of `region` and `control_inst_on_exit`,
        // always appending `ControlNode`s (including the children of entire
        // `ClaimedRegion`s) to `region`'s definition itself.
        let deferred_edges_from_control_inst = {
            let ControlInst { attrs, kind, inputs, targets, target_inputs } = control_inst_on_exit;

            // FIXME(eddyb) this loses `attrs`.
            let _ = attrs;

            let target_regions: SmallVec<[_; 8]> = targets
                .iter()
                .map(|&target| {
                    self.try_claim_edge_bundle(IncomingEdgeBundle {
                        target,
                        accumulated_count: IncomingEdgeCount::ONE,
                        target_inputs: target_inputs.get(&target).cloned().unwrap_or_default(),
                    })
                    .map_err(|edge_bundle| DeferredEdgeBundleSet::Always {
                        target: DeferredTarget::Region(edge_bundle.target),
                        edge_bundle: edge_bundle.with_target(()),
                    })
                })
                .collect();

            match kind {
                ControlInstKind::Unreachable => {
                    assert_eq!((inputs.len(), target_regions.len()), (0, 0));

                    // FIXME(eddyb) this may result in lost optimizations over
                    // actually encoding it in `ControlNode`/`ControlRegion`
                    // (e.g. a new `ControlNodeKind`, or replacing region `outputs`),
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
                    Ok(DeferredEdgeBundleSet::Unreachable)
                }

                ControlInstKind::ExitInvocation(_) => {
                    assert_eq!(target_regions.len(), 0);

                    // FIXME(eddyb) introduce equivalent `ControlNodeKind` for these.
                    Err(UnsupportedControlInst(ControlInst {
                        attrs,
                        kind,
                        inputs,
                        targets,
                        target_inputs,
                    }))
                }

                ControlInstKind::Return => {
                    assert_eq!(target_regions.len(), 0);

                    Ok(DeferredEdgeBundleSet::Always {
                        target: DeferredTarget::Return,
                        edge_bundle: IncomingEdgeBundle {
                            accumulated_count: IncomingEdgeCount::default(),
                            target: (),
                            target_inputs: inputs,
                        },
                    })
                }

                ControlInstKind::Branch => {
                    assert_eq!((inputs.len(), target_regions.len()), (0, 1));

                    Ok(self.append_maybe_claimed_region(
                        region,
                        target_regions.into_iter().next().unwrap(),
                    ))
                }

                ControlInstKind::SelectBranch(kind) => {
                    assert_eq!(inputs.len(), 1);

                    let scrutinee = inputs[0];

                    Ok(self.structurize_select_into(region, kind, scrutinee, target_regions))
                }
            }
        };

        let mut deferred_edges = deferred_edges_from_control_inst.unwrap_or_else(
            |UnsupportedControlInst(control_inst)| {
                // HACK(eddyb) this only remains used for `ExitInvocation`.
                // FIXME(eddyb) implement this as first-class, it keeps causing
                // issues where it needlessly results in an unstructured "tail".
                assert!(control_inst.targets.is_empty());

                // HACK(eddyb) attach the unsupported `ControlInst` to a fresh
                // new "proxy" `ControlRegion`, that can then be the target of
                // a deferred edge, specially crafted to be unclaimable.
                let proxy =
                    self.func_def_body.control_regions.define(self.cx, ControlRegionDef::default());
                self.func_def_body
                    .unstructured_cfg
                    .as_mut()
                    .unwrap()
                    .control_inst_on_exit_from
                    .insert(proxy, control_inst);
                self.structurize_region_state.insert(proxy, StructurizeRegionState::InProgress);
                self.incoming_edge_counts_including_loop_exits
                    .insert(proxy, IncomingEdgeCount::ONE);

                DeferredEdgeBundleSet::Always {
                    target: DeferredTarget::Region(proxy),
                    edge_bundle: IncomingEdgeBundle {
                        target: (),
                        accumulated_count: IncomingEdgeCount::default(),
                        target_inputs: [].into_iter().collect(),
                    },
                }
            },
        );

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

                match self.try_claim_edge_bundle(edge_bundle) {
                    Ok(claimed_region) => Ok((condition, claimed_region)),

                    Err(new_edge_bundle) => {
                        let new_target = DeferredTarget::Region(new_edge_bundle.target);
                        Err(DeferredEdgeBundle {
                            condition,
                            edge_bundle: new_edge_bundle.with_target(new_target),
                        })
                    }
                }
            });
            let Some((condition, then_region)) = claimed else {
                deferred_edges = else_deferred_edges;
                break;
            };

            let else_is_unreachable =
                matches!(else_deferred_edges, DeferredEdgeBundleSet::Unreachable);

            // The "then" side is only taken if `condition` holds, except that
            // `condition` can be ignored when the "else" side is unreachable.
            //
            // FIXME(eddyb) move this into `structurize_select_into`, by letting
            // it also take `LazyCond`, not just `Value`, for `scrutinee`.
            deferred_edges = if else_is_unreachable {
                self.append_maybe_claimed_region(region, Ok(then_region))
            } else {
                let condition = self.materialize_lazy_cond(condition);
                self.structurize_select_into(
                    region,
                    SelectionKind::BoolCond,
                    condition,
                    [Ok(then_region), Err(else_deferred_edges)].into_iter().collect(),
                )
            };
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

    /// Append to `parent_region` a new `Select` [`ControlNode`] built from
    /// partially structured `cases`, merging all of their `deferred_edges`
    /// together into a combined `DeferredEdgeBundleSet` (which gets returned).
    fn structurize_select_into(
        &mut self,
        parent_region: ControlRegion,
        kind: SelectionKind,
        scrutinee: Value,
        cases: SmallVec<[Result<ClaimedRegion, DeferredEdgeBundleSet>; 8]>,
    ) -> DeferredEdgeBundleSet {
        // `Select` isn't actually needed unless there's at least two `cases`.
        if cases.len() <= 1 {
            return cases.into_iter().next().map_or(DeferredEdgeBundleSet::Unreachable, |case| {
                self.append_maybe_claimed_region(parent_region, case)
            });
        }

        // Gather the full set of deferred edges (and returns), along with the
        // necessary information for the `Select`'s `ControlNodeOutputDecl`s.
        let mut deferred_edges_to_input_count_and_total_edge_count = FxIndexMap::default();
        let mut deferred_return_types = None;
        for case in &cases {
            let case_deferred_edges = match case {
                Ok(ClaimedRegion { deferred_edges, .. }) | Err(deferred_edges) => deferred_edges,
            };
            for (target, edge_bundle) in case_deferred_edges.iter_targets_with_edge_bundle() {
                let input_count = edge_bundle.target_inputs.len();

                let (old_input_count, accumulated_edge_count) =
                    deferred_edges_to_input_count_and_total_edge_count
                        .entry(target)
                        .or_insert((input_count, IncomingEdgeCount::default()));
                assert_eq!(*old_input_count, input_count);
                *accumulated_edge_count += edge_bundle.accumulated_count;

                if target == DeferredTarget::Return && deferred_return_types.is_none() {
                    // HACK(eddyb) because there's no `FuncDecl` available, take the
                    // types from the returned values and hope they match.
                    deferred_return_types = Some(
                        edge_bundle
                            .target_inputs
                            .iter()
                            .map(|&v| self.func_def_body.at(v).type_of(self.cx)),
                    );
                }
            }
        }

        // The `Select` outputs are the concatenation of `target_inputs`, for
        // each unique `deferred_edges` target.
        //
        // FIXME(eddyb) this `struct` only really exists for readability.
        struct Deferred {
            target: DeferredTarget,
            target_input_count: usize,

            /// Sum of `accumulated_count` for this `target` across all `cases`.
            total_edge_count: IncomingEdgeCount,
        }
        let deferreds = || {
            deferred_edges_to_input_count_and_total_edge_count.iter().map(
                |(&target, &(target_input_count, total_edge_count))| Deferred {
                    target,
                    target_input_count,
                    total_edge_count,
                },
            )
        };
        let mut output_decls: SmallVec<[_; 2]> =
            SmallVec::with_capacity(deferreds().map(|deferred| deferred.target_input_count).sum());
        for deferred in deferreds() {
            let target_input_types = match deferred.target {
                DeferredTarget::Region(target) => {
                    Either::Left(self.func_def_body.at(target).def().inputs.iter().map(|i| i.ty))
                }
                DeferredTarget::Return => Either::Right(deferred_return_types.take().unwrap()),
            };
            assert_eq!(target_input_types.len(), deferred.target_input_count);

            output_decls.extend(
                target_input_types
                    .map(|ty| ControlNodeOutputDecl { attrs: AttrSet::default(), ty }),
            );
        }

        // Convert the cases into `ControlRegion`s, each outputting the full set
        // of values described by `outputs` (with undef filling in any gaps),
        // while deferred conditions are collected separately (for `LazyCond`).
        let mut deferred_per_case_conditions: SmallVec<[_; 8]> =
            deferreds().map(|_| Vec::with_capacity(cases.len())).collect();
        let cases = cases
            .into_iter()
            .enumerate()
            .map(|(case_idx, case)| {
                let (case_region, mut deferred_edges) = match case {
                    Ok(ClaimedRegion {
                        structured_body,
                        structured_body_inputs,
                        deferred_edges,
                    }) => {
                        if !structured_body_inputs.is_empty() {
                            self.control_region_input_replacements
                                .insert(structured_body, structured_body_inputs);
                        }
                        (structured_body, deferred_edges)
                    }
                    Err(deferred_edges) => (
                        self.func_def_body
                            .control_regions
                            .define(self.cx, ControlRegionDef::default()),
                        deferred_edges,
                    ),
                };

                let mut outputs = SmallVec::with_capacity(output_decls.len());
                for (deferred, per_case_conditions) in
                    deferreds().zip_eq(&mut deferred_per_case_conditions)
                {
                    let (edge_condition, values_or_count) = match deferred_edges
                        .steal_deferred_by_target_without_removal(deferred.target)
                    {
                        Some(DeferredEdgeBundle { condition, edge_bundle }) => {
                            (Some(condition), Ok(edge_bundle.target_inputs))
                        }

                        None => (Some(LazyCond::False), Err(deferred.target_input_count)),
                    };

                    if let Some(edge_condition) = edge_condition {
                        assert_eq!(per_case_conditions.len(), case_idx);
                        per_case_conditions.push(edge_condition);
                    }

                    match values_or_count {
                        Ok(values) => outputs.extend(values),
                        Err(missing_value_count) => {
                            let decls_for_missing_values =
                                &output_decls[outputs.len()..][..missing_value_count];
                            outputs.extend(
                                decls_for_missing_values
                                    .iter()
                                    .map(|output| Value::Const(self.const_undef(output.ty))),
                            );
                        }
                    }
                }

                // All deferrals must have been converted into outputs above.
                assert_eq!(outputs.len(), output_decls.len());

                self.func_def_body.at_mut(case_region).def().inputs.clear();
                self.func_def_body.at_mut(case_region).def().outputs = outputs;
                case_region
            })
            .collect();

        let kind = ControlNodeKind::Select { kind, scrutinee, cases };
        let select_node = self
            .func_def_body
            .control_nodes
            .define(self.cx, ControlNodeDef { kind, outputs: output_decls }.into());

        // Build `deferred_edges` for the whole `Select`, pointing to
        // the outputs of the `select_node` `ControlNode` for all `Value`s.
        let mut outputs = (0..)
            .map(|output_idx| Value::ControlNodeOutput { control_node: select_node, output_idx });
        let deferreds = deferreds().zip_eq(deferred_per_case_conditions).map(
            |(deferred, per_case_conditions)| {
                let target_inputs = outputs.by_ref().take(deferred.target_input_count).collect();

                // Simplify `LazyCond`s eagerly, to reduce costs later on.
                let condition =
                    if per_case_conditions.iter().all(|cond| matches!(cond, LazyCond::True)) {
                        LazyCond::True
                    } else {
                        LazyCond::MergeSelect {
                            control_node: select_node,
                            per_case_conds: per_case_conditions,
                        }
                    };

                DeferredEdgeBundle {
                    condition,
                    edge_bundle: IncomingEdgeBundle {
                        target: deferred.target,
                        accumulated_count: deferred.total_edge_count,
                        target_inputs,
                    },
                }
            },
        );

        self.func_def_body.control_regions[parent_region]
            .children
            .insert_last(select_node, &mut self.func_def_body.control_nodes);

        deferreds.collect()
    }

    // FIXME(eddyb) this should try to handle as many `LazyCond` as are available,
    // for incorporating them all at once, ideally with a switch instead
    // of N individual branches with their own booleans etc.
    fn materialize_lazy_cond(&mut self, cond: LazyCond) -> Value {
        match cond {
            LazyCond::False => Value::Const(self.const_false),
            LazyCond::True => Value::Const(self.const_true),
            LazyCond::MergeSelect { control_node, per_case_conds } => {
                // HACK(eddyb) this should not allocate most of the time, and
                // avoids complications later below, when mutating the cases.
                let per_case_conds: SmallVec<[_; 8]> = per_case_conds
                    .into_iter()
                    .map(|cond| self.materialize_lazy_cond(cond))
                    .collect();

                // FIXME(eddyb) this should handle an all-`true` `per_case_conds`
                // (but `structurize_select` currently takes care of those).

                let ControlNodeDef { kind, outputs: output_decls } =
                    &mut *self.func_def_body.control_nodes[control_node];
                let cases = match kind {
                    ControlNodeKind::Select { kind, scrutinee, cases } => {
                        assert_eq!(cases.len(), per_case_conds.len());

                        if let SelectionKind::BoolCond = kind {
                            let [val_false, val_true] =
                                [self.const_false, self.const_true].map(Value::Const);
                            if per_case_conds[..] == [val_true, val_false] {
                                return *scrutinee;
                            } else if per_case_conds[..] == [val_false, val_true] {
                                // FIXME(eddyb) this could also be special-cased,
                                // at least when called from the topmost level,
                                // where which side is `false`/`true` doesn't
                                // matter (or we could even generate `!cond`?).
                                let _not_cond = *scrutinee;
                            }
                        }

                        cases
                    }
                    _ => unreachable!(),
                };

                let output_idx = u32::try_from(output_decls.len()).unwrap();
                output_decls
                    .push(ControlNodeOutputDecl { attrs: AttrSet::default(), ty: self.type_bool });

                for (&case, cond) in cases.iter().zip_eq(per_case_conds) {
                    let ControlRegionDef { outputs, .. } =
                        &mut self.func_def_body.control_regions[case];
                    outputs.push(cond);
                    assert_eq!(outputs.len(), output_decls.len());
                }

                Value::ControlNodeOutput { control_node, output_idx }
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
        parent_region: ControlRegion,
        maybe_claimed_region: Result<ClaimedRegion, DeferredEdgeBundleSet>,
    ) -> DeferredEdgeBundleSet {
        match maybe_claimed_region {
            Ok(ClaimedRegion { structured_body, structured_body_inputs, deferred_edges }) => {
                if !structured_body_inputs.is_empty() {
                    self.control_region_input_replacements
                        .insert(structured_body, structured_body_inputs);
                }
                let new_children =
                    mem::take(&mut self.func_def_body.at_mut(structured_body).def().children);
                self.func_def_body.control_regions[parent_region]
                    .children
                    .append(new_children, &mut self.func_def_body.control_nodes);
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
        region: ControlRegion,
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
                // the "else" case must be a `ControlRegion` that itself can
                // have a `ControlInst` attached to it later on.
                _ => {
                    let new_empty_region = self
                        .func_def_body
                        .control_regions
                        .define(self.cx, ControlRegionDef::default());
                    control_source = Some(new_empty_region);
                    Some((new_empty_region, [].into_iter().collect()))
                }
            };

            let condition = Some(condition)
                .filter(|_| else_target_and_inputs.is_some())
                .map(|cond| self.materialize_lazy_cond(cond));
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
