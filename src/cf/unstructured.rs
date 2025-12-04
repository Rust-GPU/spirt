//! Unstructured control-flow graph (CFG) abstractions and utilities.

use crate::func_at::FuncAt;
use crate::{
    AttrSet, Const, ConstKind, Context, EntityOrientedDenseMap, FuncDefBody, FxIndexMap,
    FxIndexSet, NodeKind, Region, Value, VarKind,
};
use itertools::{Either, Itertools as _};
use smallvec::SmallVec;

/// The control-flow graph (CFG) of a function, as control-flow edges (`thunk`s)
/// attached to [`Region`]s, as an "action on exit" (similar to "terminator"s),
/// while intra-region control-flow is strictly structured.
//
// FIXME(eddyb) update docs above (or remove `ControlFlowGraph` entirely).
//
// FIXME(eddyb) treat "loop `break`s" (branches to the "loop merge" block)
// as Rust-like "`break` out of labeled block" for a block around the loop,
// e.g. `loop { ... break; ... }` -> `'label: { loop { ... break 'label; ... } }`,
// and encode the "labeled block" aspect using a node for "executing a CFG",
// with a `thunk` output type whenever the loop body has control-flow edges
// to the outside of the loop (and they're not to the "loop merge" block),
// such as `break`-ing out of an enclosing `switch`, or even `return`ing.
//
// FIXME(eddyb) the above approach could also be extended to encode other
// kinds of "merges" (`OpSelectionMerge`, or even the "continue" of a loop)
// as well (same "break out of labeled block" idea, really), and inside-out
// structurization should then preserve all possible SPIR-V merge hierarchies
// (even if structured SPIR-T lacks e.g. `switch` fallthru, it should always
// correctly reconverge, thanks to structurization never duplicating regions).
//
// FIXME(eddyb) in the end, it seems like there is a need for *both*:
// 1. `thunk` `switch`ing (a new `SelectionKind` variant?)
//    - structurization (and similar transforms) can be seen as replacing
//      `thunk` with boolean/integer target encodings, along the "target inputs",
//      as plain values region can output, and `if`-`else`/`switch` for dispatch
// 2. `thunk` `loop`ing (new node or some kind of modified `NodeKind::Loop`?)
//    - can be seen as taking the fixpoint of a `thunk -> thunk` region,
//      which, in the fully unstructured CFG case, would consist of a single node
//      (specifically a "`thunk` `switch`" dispatching all regions in the CFG,
//      which makes it feel wasteful, but there has to be a way to distinguish
//      between single-step and "until `break`-ing out")
//    - maybe this can be done using the `thunk` `switch` node, with extra modes
//      in e.g. its `SelectionKind` variant(s), such as "has a default case" and
//      "`loop`-`switch` until `break" (although there is an unique hazard with
//      reusing `NodeKind::Select` while having regions with different signatures,
//      in that passes might assume matching signatures, and also any kind of
//      looping control-flow has to account for the potential to execute any
//      of the child regions, and therefore any side-effect, more than once etc.)
#[derive(Clone, Default)]
pub struct ControlFlowGraph {
    pub target_thunk_on_exit_from: EntityOrientedDenseMap<Region, Value>,

    // HACK(eddyb) this currently only comes from `OpLoopMerge`, and cannot be
    // inferred (because implies too strong of an ownership/uniqueness notion).
    pub loop_merge_to_loop_header: FxIndexMap<Region, Region>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum ControlTarget {
    Region(Region),

    /// Leave the current function (returning `target_inputs`, if any).
    //
    // FIXME(eddyb) now that this is used through `NodeKind::ThunkBind`,
    // it should probably be more like `break` or some kind of "leave scope".
    // FIXME(eddyb) one way to encode this unambiguously, even when dealing
    // with e.g. multiple nested CFGs (such as the "execute a CFG" node idea),
    // is to pick a region as the function-scoped "exit" (or "postdominator"),
    // and in the worst-case it would only serve to output its own inputs
    // (alternatively, non-trivial exits from "execute a CFG" can use `thunk`
    // wrapping, so in the case of e.g. `return`ing from inside N nested loops,
    // one wrap N `break`-like thunks, with each of the N levels being unwrapped
    // by leaving the loops etc. - but that still technically has the ambiguity,
    // and would result in the nested encoding showing up in the final structured
    // result, so e.g. even if integers are used, N of them would be neeeded).
    Return,
}

// HACK(eddyb) this is only public so it can be shared with `spv::lift`.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ControlEdge {
    pub source_region: Region,

    // HACK(eddyb) this is the minimal amount of information needed to look up
    // the thunk, given access to the function (including `ControlFlowGraph`).
    pub thunk_binding_region: Region,
}

impl ControlFlowGraph {
    /// Iterate over all [`Region`]s making up `func_def_body`'s CFG, in
    /// reverse post-order (RPO).
    ///
    /// RPO iteration over a CFG provides certain guarantees, most importantly
    /// that dominators are visited before the entire subgraph they dominate.
    pub fn rev_post_order(
        &self,
        func_def_body: &FuncDefBody,
    ) -> impl DoubleEndedIterator<Item = Region> + use<> {
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

    pub fn edges_from<'a>(
        &'a self,
        func_at_edge_source: FuncAt<'a, Region>,
    ) -> impl DoubleEndedIterator<Item = ControlEdge> + ExactSizeIterator + Clone + 'a {
        self.edges_from_thunk_tailed_region(func_at_edge_source, func_at_edge_source.position)
    }

    // FIXME(eddyb) properly distinguish the different failure cases possible,
    // maybe even avoid panicking entirely?
    fn edges_from_thunk_tailed_region<'a>(
        &'a self,
        func_at_edge_source: FuncAt<'a, Region>,
        thunk_binding_region: Region,
    ) -> impl DoubleEndedIterator<Item = ControlEdge> + ExactSizeIterator + Clone + 'a {
        let func = func_at_edge_source.at(());
        let edge_source_region = func_at_edge_source.position;

        let thunk = if edge_source_region == thunk_binding_region {
            // FIXME(eddyb) make the thunk an actual region output instead.
            *self.target_thunk_on_exit_from.get(edge_source_region).expect(
                "cf::unstructured: missing `target_thunk_on_exit_from`, \
                 despite having left structured control-flow",
            )
        } else {
            func.at(thunk_binding_region).def().outputs.iter().copied().exactly_one().ok().unwrap()
        };

        let thunk_node = match thunk {
            Value::Var(thunk) => match func.at(thunk).decl().kind() {
                VarKind::NodeOutput { node, output_idx: 0 } => node,
                _ => unreachable!(),
            },
            Value::Const(_) => {
                return Either::Left(
                    [ControlEdge { source_region: edge_source_region, thunk_binding_region }]
                        .into_iter(),
                );
            }
        };

        let thunk_binding_region_children = func.at(thunk_binding_region).def().children.iter();
        assert!(thunk_binding_region_children.last == Some(thunk_node));
        if edge_source_region != thunk_binding_region {
            assert!(thunk_binding_region_children.first == thunk_binding_region_children.last);
        }

        let thunk_node_def = func.at(thunk_node).def();
        match thunk_node_def.kind {
            NodeKind::ThunkBind(_) => Either::Left(
                [ControlEdge { source_region: edge_source_region, thunk_binding_region }]
                    .into_iter(),
            ),
            NodeKind::Select(_) => {
                assert!(edge_source_region == thunk_binding_region);

                Either::Right(thunk_node_def.child_regions.iter().map(move |&case| {
                    self.edges_from_thunk_tailed_region(func_at_edge_source, case)
                        .exactly_one()
                        .ok()
                        .unwrap()
                }))
            }
            _ => unreachable!(),
        }
    }

    pub fn edge_attrs_target_and_inputs<'a>(
        &self,
        func_at_edge: FuncAt<'a, ControlEdge>,
    ) -> (AttrSet, Result<ControlTarget, Const>, &'a [Value]) {
        let func = func_at_edge.at(());
        let edge = func_at_edge.position;

        let thunk = if edge.source_region == edge.thunk_binding_region {
            // FIXME(eddyb) make the thunk an actual region output instead.
            *self.target_thunk_on_exit_from.get(edge.thunk_binding_region).expect(
                "cf::unstructured: missing `target_thunk_on_exit_from`, \
                     despite having left structured control-flow",
            )
        } else {
            func.at(edge.thunk_binding_region)
                .def()
                .outputs
                .iter()
                .copied()
                .exactly_one()
                .ok()
                .unwrap()
        };

        match thunk {
            Value::Var(thunk) => match func.at(thunk).decl().kind() {
                VarKind::NodeOutput { node, output_idx: 0 } => {
                    let thunk_node_def = func.at(node).def();
                    match thunk_node_def.kind {
                        NodeKind::ThunkBind(target) => {
                            (thunk_node_def.attrs, Ok(target), &thunk_node_def.inputs)
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            },
            Value::Const(ct) => (AttrSet::default(), Err(ct), &[]),
        }
    }
}

// HACK(eddyb) this only serves to disallow accessing `IncomingEdgeCount`'s private field.
mod sealed {
    /// Opaque newtype for the count of incoming edges (into a [`Region`](crate::Region)).
    ///
    /// The private field prevents direct mutation or construction, forcing the
    /// use of [`IncomingEdgeCount::ONE`] and addition operations to produce some
    /// specific count (which would require explicit workarounds for misuse).
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct IncomingEdgeCount(usize);

    impl IncomingEdgeCount {
        pub const ONE: Self = Self(1);
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
pub use sealed::IncomingEdgeCount;

pub struct TraversalState<PreVisit: FnMut(Region), PostVisit: FnMut(Region)> {
    pub incoming_edge_counts: EntityOrientedDenseMap<Region, IncomingEdgeCount>,
    pub pre_order_visit: PreVisit,
    pub post_order_visit: PostVisit,

    // FIXME(eddyb) should this be a generic parameter for "targets iterator"?
    pub reverse_targets: bool,
}

impl ControlFlowGraph {
    pub fn traverse_whole_func(
        &self,
        func_def_body: &FuncDefBody,
        state: &mut TraversalState<impl FnMut(Region), impl FnMut(Region)>,
    ) {
        let func_at_body = func_def_body.at_body();

        // Quick sanity check that this is the right CFG for `func_def_body`.
        assert!(std::ptr::eq(func_def_body.unstructured_cfg.as_ref().unwrap(), self));
        assert!(func_at_body.def().outputs.is_empty());

        self.traverse(func_at_body, state);
    }

    fn traverse(
        &self,
        func_at_region: FuncAt<'_, Region>,
        state: &mut TraversalState<impl FnMut(Region), impl FnMut(Region)>,
    ) {
        let func = func_at_region.at(());
        let region = func_at_region.position;

        // FIXME(eddyb) `EntityOrientedDenseMap` should have an `entry` API.
        if let Some(existing_count) = state.incoming_edge_counts.get_mut(region) {
            *existing_count += IncomingEdgeCount::ONE;
            return;
        }
        state.incoming_edge_counts.insert(region, IncomingEdgeCount::ONE);

        (state.pre_order_visit)(region);

        let targets = self.targets_of(func_at_region);

        let targets = targets.filter_map(|target| match target {
            Ok(ControlTarget::Region(target)) => Some(target),
            Ok(ControlTarget::Return) | Err(_) => None,
        });
        let targets = if state.reverse_targets {
            Either::Left(targets.rev())
        } else {
            Either::Right(targets)
        };
        for target in targets {
            self.traverse(func.at(target), state);
        }

        (state.post_order_visit)(region);
    }

    fn targets_of<'a>(
        &'a self,
        func_at_edge_source: FuncAt<'a, Region>,
    ) -> impl DoubleEndedIterator<Item = Result<ControlTarget, Const>> + Clone + 'a {
        let func = func_at_edge_source.at(());
        self.edges_from(func_at_edge_source)
            .map(move |edge| self.edge_attrs_target_and_inputs(func.at(edge)).1)
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
pub struct LoopFinder<'a> {
    cx: &'a Context,
    func: FuncAt<'a, ()>,
    cfg: &'a ControlFlowGraph,

    // FIXME(eddyb) this feels a bit inefficient (are many-exit loops rare?).
    // FIXME(eddyb) rename this and/or wrap the value type in a newtype.
    loop_header_to_exit_targets: FxIndexMap<Region, FxIndexSet<Region>>,

    /// SCC accumulation stack, where CFG nodes collect during the depth-first
    /// traversal, and are only popped when their "SCC root" (loop header) is
    /// (note that multiple SCCs on the stack does *not* indicate SCC nesting,
    /// but rather a path between two SCCs, i.e. a loop *following* another).
    scc_stack: Vec<Region>,
    /// Per-CFG-node traversal state (often just pointing to a `scc_stack` slot).
    scc_state: EntityOrientedDenseMap<Region, SccState>,
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

// HACK(eddyb) marker type for an `undef` thunk.
#[derive(Copy, Clone)]
struct Unreachable;

impl<'a> LoopFinder<'a> {
    pub fn new(cx: &'a Context, func: FuncAt<'a, ()>, cfg: &'a ControlFlowGraph) -> Self {
        Self {
            cx,
            func,
            cfg,
            loop_header_to_exit_targets: FxIndexMap::default(),
            scc_stack: vec![],
            scc_state: EntityOrientedDenseMap::new(),
        }
    }

    /// Returns a map from every loop header to its set of exit targets, in the
    /// control-flow (sub)graph starting at `entry`.
    //
    // FIXME(eddyb) reconsider this entire API (it used to be all private).
    pub fn find_all_loops_starting_at(
        mut self,
        entry: Region,
    ) -> FxIndexMap<Region, FxIndexSet<Region>> {
        self.find_earliest_scc_root_of(entry);
        self.loop_header_to_exit_targets
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
        node: Region,
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
                    (Some(scc_stack_idx), EventualCfgExits { may_return_from_func: false })
                }
                SccState::Complete(eventual_cfg_exits) => (None, eventual_cfg_exits),
            };
        }
        let scc_stack_idx = SccStackIdx(self.scc_stack.len().try_into().unwrap());
        self.scc_stack.push(node);
        *state_entry = Some(SccState::Pending(scc_stack_idx));

        let targets = self.targets_of(node);

        let mut eventual_cfg_exits = EventualCfgExits {
            may_return_from_func: targets
                .clone()
                .any(|target| matches!(target, Ok(ControlTarget::Return))),
        };

        let earliest_scc_root = targets
            .flat_map(|target| {
                let target = match target {
                    Ok(ControlTarget::Region(target)) => target,
                    Ok(ControlTarget::Return) | Err(Unreachable) => {
                        return None.into_iter().chain(None);
                    }
                };

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
                        self.targets_of(scc_node).filter_map(|target| match target {
                            Ok(ControlTarget::Region(target)) => Some(target),
                            Ok(ControlTarget::Return) | Err(Unreachable) => None,
                        })
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

    fn targets_of(
        &self,
        region: Region,
    ) -> impl Iterator<Item = Result<ControlTarget, Unreachable>> + Clone + 'a {
        let cx = self.cx;

        self.cfg.targets_of(self.func.at(region)).map(|edge| {
            edge.map_err(|ct| match cx[ct].kind {
                ConstKind::Undef => Unreachable,
                _ => unreachable!(),
            })
        })
    }
}
