//! [`QPtr`](crate::TypeKind::QPtr) simplification passes.

// TODO(eddyb) make note of this module using the new post-`Var` terminology,
// i.e. global/local "binding", and replace the names eveywhere else, too
// (there is a small conflict with SPIR-V "binding" but not a logical one,
// and all other choices have headaches of their own) - also, the "local" side
// (e.g. this module) prefers "local" and may avoid "binding" entirely, and/or
// use the verb "to bind" (for specifically `QPtrOp::FuncLocalVar`) etc.
// (unlike globals, it's almost always plain memory, which should help).

use crate::func_at::{FuncAt, FuncAtMut};
use crate::mem::{MemOp, shapes};
use crate::qptr::QPtrOp;
use crate::visit::{InnerVisit, Visitor};
use crate::{
    AttrSet, Const, ConstDef, ConstKind, Context, Func, FuncDefBody, FxIndexMap, FxIndexSet,
    GlobalVar, Node, NodeDef, NodeKind, Region, RegionDef, Type, TypeKind, Value, Var, VarDecl,
    vector,
};
use itertools::Either;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::num::{NonZeroI32, NonZeroU32};
use std::ops::{Bound, Range};
use std::rc::Rc;
use std::{mem, slice};

// HACK(eddyb) sharing layout code with other modules.
use crate::mem::layout::*;

// HACK(eddyb) common helper due to potential terminological confusion.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct LocalKey {
    qptr_output: Var,
}

impl LocalKey {
    fn maybe_from_qptr_value(v: Value) -> Option<Self> {
        match v {
            Value::Var(qptr_output) => Some(LocalKey { qptr_output }),
            Value::Const(_) => None,
        }
    }

    fn qptr_value(self) -> Value {
        Value::Var(self.qptr_output)
    }
}

/// Split all function-local bindings ("locals") in `func_def_body` into
/// as many (independently accessed) locals as possible.
//
// FIXME(eddyb) reduce the cost of creating all the per-partition locals by
// feeding partitions to `propagate_contents_of_locals_in_func` directly.
pub fn partition_locals_in_func(
    cx: Rc<Context>,
    config: &LayoutConfig,
    func_def_body: &mut FuncDefBody,
) {
    let locals = {
        let mut collector = CollectLocalPartitions {
            cx: cx.clone(),
            layout_cache: LayoutCache::new(cx.clone(), config),
            locals: FxIndexMap::default(),

            parent_region: None,
        };
        func_def_body.inner_visit_with(&mut collector);
        collector.locals
    };

    let qptr_type = cx.intern(TypeKind::QPtr);

    // Create new locals for all partitions, and replace their respective uses.
    for (original_local_key, local) in locals {
        let original_local_node =
            func_def_body.at(original_local_key.qptr_output).decl().def_parent.right().unwrap();

        // Also shrink the original local, if necessary.
        if local.zero_offset_partition_size < local.original_layout.size {
            func_def_body.at_mut(original_local_node).def().kind =
                MemOp::FuncLocalVar(shapes::MemLayout {
                    size: local.zero_offset_partition_size,
                    ..local.original_layout
                })
                .into();
        }

        let mut insert_after_node = original_local_node;

        for (partition_offset, partition) in local.non_zero_offset_to_partition {
            let align_for_offset = 1 << partition_offset.trailing_zeros();

            let partition_local_node = func_def_body.nodes.define(
                &cx,
                NodeDef {
                    // FIXME(eddyb) preserve at least debuginfo attrs.
                    attrs: Default::default(),
                    kind: MemOp::FuncLocalVar(shapes::MemLayout {
                        align: local.original_layout.align.min(align_for_offset),
                        legacy_align: local.original_layout.legacy_align.min(align_for_offset),
                        size: partition.size,
                    })
                    .into(),
                    inputs: [].into_iter().collect(),
                    child_regions: [].into_iter().collect(),
                    outputs: [].into_iter().collect(),
                }
                .into(),
            );
            let partition_local_qptr_output_var = func_def_body.vars.define(
                &cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: qptr_type,
                    def_parent: Either::Right(partition_local_node),
                    def_idx: 0,
                },
            );
            func_def_body.nodes[partition_local_node].outputs.push(partition_local_qptr_output_var);

            func_def_body.regions[local.parent_region].children.insert_after(
                partition_local_node,
                insert_after_node,
                &mut func_def_body.nodes,
            );
            insert_after_node = partition_local_node;

            let partition_local_qptr = Value::Var(partition_local_qptr_output_var);

            // FIMXE(eddyb) when `QPtrOp::Offset` ends up with a `0` offset,
            // some further simplifications are possible, but it's not that
            // relevant for now, as we're mainly interested in loads/stores.
            let partition_offset = i32::try_from(partition_offset.get()).unwrap();
            for use_node in partition.uses {
                let node_def = func_def_body.at_mut(use_node).def();

                assert!(
                    mem::replace(&mut node_def.inputs[0], partition_local_qptr)
                        == original_local_key.qptr_value()
                );

                match &mut node_def.kind {
                    NodeKind::Mem(MemOp::Load { offset } | MemOp::Store { offset }) => {
                        *offset = NonZeroI32::new(
                            offset.map_or(0, |o| o.get()).checked_sub(partition_offset).unwrap(),
                        );
                    }
                    NodeKind::QPtr(QPtrOp::Offset(offset)) => {
                        *offset = offset.checked_sub(partition_offset).unwrap();
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

struct CollectLocalPartitions<'a> {
    cx: Rc<Context>,
    layout_cache: LayoutCache<'a>,
    locals: FxIndexMap<LocalKey, LocalPartitions>,

    parent_region: Option<Region>,
}

struct LocalPartitions {
    parent_region: Region,
    original_layout: shapes::MemLayout,
    // HACK(eddyb) offset `0` reuses the original local and is tracked separately,
    // to reduce the cost for both the collection, and to make replacement a noop.
    zero_offset_partition_size: u32,
    non_zero_offset_to_partition: BTreeMap<NonZeroU32, Partition>,
}

#[derive(Default)]
struct Partition {
    size: u32,

    /// All `Node`s with a `QPtr` input and an immediate offset (`QPtrOp::Offset`
    /// and `MemOp::{Load,Store}`), which have to be updated after partitioning.
    uses: SmallVec<[Node; 4]>,
}

impl LocalPartitions {
    /// Remove all partitions and prevent any further ones from being added
    /// (typically only needed for locals used in unknown ways).
    fn forfeit_partitioning(&mut self) {
        self.zero_offset_partition_size = self.original_layout.size;
        mem::take(&mut self.non_zero_offset_to_partition);
    }

    /// Record a new partition range, and the `Node` it originated from,
    /// merging ranges and uses with existing ones, in case of overlaps.
    fn add_use(&mut self, range: Range<u32>, use_node: Node) {
        // FIXME(eddyb) the logic below is not amenable to ZSTs.
        if range.is_empty() {
            return self.forfeit_partitioning();
        }

        // The partition starting at `0` is special and does not track `uses`.
        if range.start == 0 || range.start < self.zero_offset_partition_size {
            self.zero_offset_partition_size = self.zero_offset_partition_size.max(range.end);

            // Absorb overlaps without keeping track of their `uses`.
            while let Some(entry) = self.non_zero_offset_to_partition.first_entry() {
                let partition_offset = entry.key().get();
                if range.end <= partition_offset {
                    break;
                }
                let partition = entry.remove();
                self.zero_offset_partition_size =
                    partition_offset.checked_add(partition.size).unwrap();
            }
            return;
        }

        let range = NonZeroU32::new(range.start).unwrap()..NonZeroU32::new(range.end).unwrap();
        let mut rev_overlapping_entries = self
            .non_zero_offset_to_partition
            .range_mut((Bound::Unbounded, Bound::Excluded(range.end)))
            .rev()
            .take_while(|(partition_offset, partition)| {
                partition_offset.checked_add(partition.size).unwrap() > range.start
            });

        // Fast path: `range` begins in an existing partition, and either already
        // ends within it, or at least ends before the next existing partition
        // (the second condition is guaranteed by this being the *last* overlap).
        let mut last_overlapping_entry = rev_overlapping_entries.next();
        if let Some((partition_offset, partition)) = &mut last_overlapping_entry
            && **partition_offset <= range.start
        {
            partition.size = partition.size.max(range.end.get() - partition_offset.get());
            partition.uses.push(use_node);
            return;
        }

        let rev_overlapping_entries =
            last_overlapping_entry.into_iter().chain(rev_overlapping_entries);

        // FIXME(eddyb) this is a bit inefficient but we don't have
        // cursors, so we have to buffer the `BTreeMap` keys here.
        let rev_overlapping_offsets: SmallVec<[_; 4]> =
            rev_overlapping_entries.map(|(&offset, _)| offset).collect();

        let merged_entry = rev_overlapping_offsets
            .into_iter()
            .rev()
            .map(|offset| (offset, self.non_zero_offset_to_partition.remove(&offset).unwrap()))
            .chain([(
                range.start,
                Partition {
                    size: range.end.get() - range.start.get(),
                    uses: [use_node].into_iter().collect(),
                },
            )])
            .reduce(|(a_start, a), (b_start, b)| {
                let (a_end, b_end) =
                    (a_start.checked_add(a.size).unwrap(), b_start.checked_add(b.size).unwrap());
                let start = a_start.min(b_start);
                let mut uses = a.uses;
                uses.extend(b.uses);
                (start, Partition { size: a_end.max(b_end).get() - start.get(), uses })
            })
            .unwrap();
        self.non_zero_offset_to_partition.extend([merged_entry]);
    }
}

impl Visitor<'_> for CollectLocalPartitions<'_> {
    // FIXME(eddyb) this is excessive, maybe different kinds of
    // visitors should exist for module-level and func-level?
    fn visit_attr_set_use(&mut self, _: AttrSet) {}
    fn visit_type_use(&mut self, _: Type) {}
    fn visit_const_use(&mut self, _: Const) {}
    fn visit_global_var_use(&mut self, _: GlobalVar) {}
    fn visit_func_use(&mut self, _: Func) {}

    // NOTE(eddyb) uses of locals that end up here disable partitioning of
    // that local, as they're not one of the special cases which are allowed.
    fn visit_value_use(&mut self, &v: &Value) {
        if let Some(local_key) = LocalKey::maybe_from_qptr_value(v)
            && let Some(local) = self.locals.get_mut(&local_key)
        {
            local.forfeit_partitioning();
        }
    }

    fn visit_region_def(&mut self, func_at_region: FuncAt<'_, Region>) {
        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_visit_with(self);
        self.parent_region = outer_region;
    }

    fn visit_node_def(&mut self, func_at_node: FuncAt<'_, Node>) {
        let node_def = func_at_node.def();

        let first_input_qptr_with_offset_and_access_type = match node_def.kind {
            NodeKind::Mem(MemOp::FuncLocalVar(layout)) => {
                // FIXME(eddyb) support optional initializers.
                if node_def.inputs.is_empty() {
                    self.locals.insert(
                        LocalKey { qptr_output: node_def.outputs[0] },
                        LocalPartitions {
                            parent_region: self.parent_region.unwrap(),
                            original_layout: layout,
                            zero_offset_partition_size: 0,
                            non_zero_offset_to_partition: BTreeMap::new(),
                        },
                    );
                }

                None
            }

            // FIXME(eddyb) support more uses of `qptr`s.
            NodeKind::QPtr(QPtrOp::Offset(offset)) => {
                // FIXME(eddyb) we could have a narrower range here,
                // if it was recorded during `qptr::lower`.
                Some((NonZeroI32::new(offset), None))
            }
            NodeKind::Mem(MemOp::Load { offset }) => {
                Some((offset, Some(func_at_node.at(node_def.outputs[0]).decl().ty)))
            }
            NodeKind::Mem(MemOp::Store { offset }) => {
                Some((offset, Some(func_at_node.at(node_def.inputs[1]).type_of(&self.cx))))
            }

            _ => None,
        };
        let first_input_local_with_offset_range = first_input_qptr_with_offset_and_access_type
            .and_then(|(offset, access_type)| {
                let local =
                    self.locals.get_mut(&LocalKey::maybe_from_qptr_value(node_def.inputs[0])?)?;

                let start = u32::try_from(offset.map_or(0, |o| o.get())).ok()?;

                let end = match access_type {
                    Some(ty) => match self.layout_cache.layout_of(ty).ok()? {
                        TypeLayout::Concrete(layout)
                            if layout.mem_layout.dyn_unit_stride.is_none() =>
                        {
                            start.checked_add(layout.mem_layout.fixed_base.size)?
                        }
                        _ => return None,
                    },
                    None => local.original_layout.size,
                };

                Some((local, start..end))
            });
        if let Some((local, offset_range)) = first_input_local_with_offset_range {
            local.add_use(offset_range, func_at_node.position);

            // Only visit the *other* inputs, not the `qptr` one.
            for v in &node_def.inputs[1..] {
                self.visit_value_use(v);
            }

            return;
        }

        func_at_node.inner_visit_with(self);
    }
}

#[must_use]
#[derive(Default)]
pub struct PropagateLocalContentsReport {
    /// Whether at least one of the locals that had its contents propagated,
    /// held a `qptr`, which may now allow further simplifications.
    pub any_qptrs_propagated: bool,
}

/// Propagate contents of function-local bindings ("locals") throughout all of
/// `func_def_body` (i.e. replacing loads with previously-stored values).
pub fn propagate_contents_of_locals_in_func(
    cx: Rc<Context>,
    config: &LayoutConfig,
    func_def_body: &mut FuncDefBody,
) -> PropagateLocalContentsReport {
    let mut report = PropagateLocalContentsReport::default();

    // Avoid having to support unstructured control-flow.
    if func_def_body.unstructured_cfg.is_some() {
        return report;
    }

    let (locals, propagated_loads) = {
        let mut propagator = PropagateLocalContents {
            cx: &cx,
            layout_cache: LayoutCache::new(cx.clone(), config),
            parent_region: None,
            locals: FxIndexMap::default(),
            mutation_log: vec![],
            propagated_loads: FxIndexMap::default(),
        };
        propagator.propagate_through_region(func_def_body.at_mut_body());
        (propagator.locals, propagator.propagated_loads)
    };

    // FIXME(eddyb) this is not the most efficient way to compute this, but it
    // should be straight-forwardly correct to do it here.
    report.any_qptrs_propagated = locals
        .values()
        .filter_map(|local| local.as_ref().ok()?.ty)
        .any(|ty| matches!(cx[ty].kind, TypeKind::QPtr));

    let node_from_output = |output_var| func_def_body.vars[output_var].def_parent.right().unwrap();
    let nodes_to_remove = propagated_loads
        .into_iter()
        .map(|(output_var, (_, parent_region))| (node_from_output(output_var), parent_region))
        .chain(locals.into_iter().flat_map(|(original_local_key, local_contents)| {
            local_contents.ok().into_iter().flat_map(move |local_contents| {
                [(node_from_output(original_local_key.qptr_output), local_contents.parent_region)]
                    .into_iter()
                    .chain(local_contents.stores_with_parent_region)
            })
        }));
    for (node, parent_region) in nodes_to_remove {
        func_def_body.regions[parent_region].children.remove(node, &mut func_def_body.nodes);
    }

    report
}

struct PropagateLocalContents<'a> {
    cx: &'a Context,
    layout_cache: LayoutCache<'a>,

    parent_region: Option<Region>,

    locals: FxIndexMap<LocalKey, Result<LocalContents, UnknowableLocal>>,

    // HACK(eddyb) this allows a flat representation, and handling `Select`
    // nodes at a cost proportional only to the number of locals
    // modified in any of the child regions (not the total number of locals).
    mutation_log: Vec<LocalMutation>,

    /// `MemOp::Load` outputs with known `Value`s, and also tracking their
    /// node's parent `Region` for later removal.
    //
    // FIXME(eddyb) it should be possible to remove the loads as they are seen.
    propagated_loads: FxIndexMap<Var, (Value, Region)>,
}

/// Error type for when some `LocalContents` cannot be tracked, either
/// due to escaping pointers (allowing indirect accesses), or the presence of
/// another issue preventing tracking (e.g. layout error, type mismatch, etc.).
struct UnknowableLocal;

struct LocalContents {
    parent_region: Region,
    size: u32,

    /// Deduced type (of `value`, but may be present even if `value` is missing),
    /// which cannot change once set (instead, `UnknowableLocal` is produced).
    ty: Option<Type>,

    value: Option<Value>,

    /// `MemOp::Store` nodes to remove, if the whole local is removed,
    /// and their parent `Region`.
    stores_with_parent_region: SmallVec<[(Node, Region); 4]>,
}

struct LocalMutation {
    /// Index of the local in the `locals` field of `PropagateLocalContents`.
    local_idx: usize,

    /// Previous value of the `value` field of `LocalContents`.
    prev_value: Option<Value>,
}

struct LocalAccess<'a> {
    /// Index of the local in the `locals` field of `PropagateLocalContents`.
    local_idx: usize,

    local: &'a mut LocalContents,

    /// If the local is an `OpTypeVector`, and this access is for one
    /// of its scalar elements, this will contain that element's index.
    vector_elem_idx: Option<u8>,
}

impl PropagateLocalContents<'_> {
    /// Validate an access into `local_qptr`, at `offset`, with type `access_type`,
    /// returning `Some` if, and only if, the access does not conflict with any
    /// previous ones, type-wise (with accesses smaller than the whole local
    /// being inferred as vector element accesses if a valid vector type fits).
    ///
    /// When `Some(access)` is returned, `access.local.ty` is guaranteed to be
    /// `Some`, and the type of `access.local.value` (if the latter is present),
    /// but can still differ from `access_type` even if they're the same size
    /// (in which case introducing bitcasts will likely be needed).
    fn lookup_local_for_access(
        &mut self,
        local_qptr: Value,
        offset: Option<NonZeroI32>,
        access_type: Type,
    ) -> Option<LocalAccess<'_>> {
        // HACK(eddyb) we steal the `LocalContents` to make the logic below
        // easier to write: if *anything* goes wrong, `Err(UnknowableLocal)`
        // will be left behind, and `Ok(local)` will be be restored if and only if
        // everything about this access is valid (and `Some` will be returned).
        let (local_idx, _, local) =
            self.locals.get_full_mut(&LocalKey::maybe_from_qptr_value(local_qptr)?)?;
        let mut local = mem::replace(local, Err(UnknowableLocal)).ok()?;

        let offset = u32::try_from(offset.map_or(0, |o| o.get())).ok()?;

        let layout = match self.layout_cache.layout_of(access_type).ok()? {
            TypeLayout::Concrete(layout) if layout.mem_layout.dyn_unit_stride.is_none() => layout,
            _ => return None,
        };
        let access_size = layout.mem_layout.fixed_base.size;

        let (inferred_local_type, vector_elem_idx) = if offset == 0 && access_size == local.size {
            (layout.original_type, None)
        } else {
            // HACK(eddyb) we only support vector types here, as
            // they're the most common cause of partial loads/stores.
            let inferred_vector_len = local.size / access_size;
            let elem_idx = offset / access_size;

            let scalar_access_type = access_type.as_scalar(self.cx)?;
            let legal_vector = local.size.is_multiple_of(access_size)
                && offset.is_multiple_of(access_size)
                && (2..=4).contains(&inferred_vector_len);
            if !legal_vector {
                return None;
            }
            (
                self.cx.intern(vector::Type {
                    elem: scalar_access_type,
                    elem_count: u8::try_from(inferred_vector_len).ok()?.try_into().ok()?,
                }),
                Some(u8::try_from(elem_idx).unwrap()),
            )
        };

        // HACK(eddyb) allow bitcasts in general by only requiring type equality
        // when a vector type was synthesized (for accessing a single element).
        if inferred_local_type != layout.original_type
            && local.ty.is_some_and(|ty| ty != inferred_local_type)
        {
            return None;
        }
        if local.ty.is_none() {
            local.ty = Some(inferred_local_type);
        }

        self.locals[local_idx] = Ok(local);
        let local = self.locals[local_idx].as_mut().ok().unwrap();

        // FIXME(eddyb) should the returned value not even contain a reference
        // into `self.locals`, given that it's entirely relying on indexing?
        Some(LocalAccess { local_idx, local, vector_elem_idx })
    }

    /// Apply active rewrites (i.e. `propagated_loads`) to all `values`.
    fn propagate_into_values(&mut self, values: &mut [Value]) {
        for v in values {
            if let Value::Var(var) = *v
                && let Some(&(replacement_value, _)) = self.propagated_loads.get(&var)
            {
                *v = replacement_value;
            }
        }
    }

    /// Record `values` as used - this is expected to be called only after
    /// `propagate_into_values` was applied, and not to include `qptr`s which
    /// were part of propagated loads/stores, as this'd mark them as unknowable.
    fn track_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Some(local_key) = LocalKey::maybe_from_qptr_value(v)
                && let Some(local) = self.locals.get_mut(&local_key)
            {
                *local = Err(UnknowableLocal);
            }
        }
    }

    fn propagate_through_region(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        let outer_region = self.parent_region.replace(func_at_region.position);

        let mut children = func_at_region.reborrow().at_children().into_iter();
        while let Some(func_at_node) = children.next() {
            self.propagate_through_node(func_at_node);
        }

        let RegionDef { inputs: _, children: _, outputs } = func_at_region.def();
        self.propagate_into_values(outputs);
        self.track_value_uses(outputs);

        self.parent_region = outer_region;
    }

    fn propagate_through_node(&mut self, func_at_node: FuncAtMut<'_, Node>) {
        let cx = self.cx;

        let const_undef = |ty| {
            Value::Const(cx.intern(ConstDef {
                attrs: AttrSet::default(),
                ty,
                kind: ConstKind::Undef,
            }))
        };

        let parent_region = self.parent_region.unwrap();

        let node = func_at_node.position;

        // FIXME(eddyb) is this a good convention?
        let mut func = func_at_node.at(());

        let NodeDef { attrs: _, kind, inputs, child_regions, outputs } = &mut *func.nodes[node];

        // FIXME(eddyb) it may be helpful to fold uses after propagation,
        // (e.g. `qptr.offset` into `mem.{load,store}`), to allow propagation
        // of locals who had their pointers stored in other locals - note
        // that multiple propagation passes would *still* be needed, because the
        // original store of a pointer to a local will make it unknowable.
        self.propagate_into_values(inputs);

        // HACK(eddyb) this uses "has child regions" as "has nested dataflow".
        if !child_regions.is_empty() {
            self.track_value_uses(inputs);
        } else {
            // NOTE(eddyb) inputs tracked after the `match` below.
        }

        match *kind {
            NodeKind::Select(_) => {
                let num_cases = child_regions.len();

                // HACK(eddyb) this is how we can both roll back changes to
                // locals' `value`s, and know which locals were changed
                // in the first place (to merge their changes values, together).
                let mutation_log_start = self.mutation_log.len();

                let mut local_idx_to_per_case_values =
                    FxIndexMap::<usize, SmallVec<[_; 2]>>::default();
                for case_idx in 0..num_cases {
                    let case = func.reborrow().at(node).def().child_regions[case_idx];
                    self.propagate_through_region(func.reborrow().at(case));

                    // NOTE(eddyb) we traverse the mutation log forwards, as we
                    // already have a way to determine whether we've seen any
                    // mutations for each local, and only the oldest mutation
                    // is needed to roll back the local to its original state.
                    for mutation in self.mutation_log.drain(mutation_log_start..) {
                        let original_local_value = mutation.prev_value;
                        if let Ok(local) = &mut self.locals[mutation.local_idx] {
                            let per_case_local_values = local_idx_to_per_case_values
                                .entry(mutation.local_idx)
                                .or_insert_with(|| {
                                    let mut per_case_local_values =
                                        SmallVec::with_capacity(num_cases);

                                    // This case may be the first to mutate this
                                    // local - thankfully we know the original
                                    // value (which will be common across all cases).
                                    per_case_local_values
                                        .extend((0..case_idx).map(|_| original_local_value));

                                    per_case_local_values
                                });

                            if per_case_local_values.len() <= case_idx {
                                let new_local_value =
                                    mem::replace(&mut local.value, original_local_value);
                                per_case_local_values.push(new_local_value);
                            }
                            assert_eq!(per_case_local_values.len() - 1, case_idx);
                        }
                    }

                    // Some locals may only have been mutated in previous cases.
                    for (&local_idx, per_case_local_values) in &mut local_idx_to_per_case_values {
                        if per_case_local_values.len() <= case_idx
                            && let Ok(local) = &self.locals[local_idx]
                        {
                            per_case_local_values.push(local.value);
                            assert_eq!(per_case_local_values.len() - 1, case_idx);
                        }
                    }
                }

                // Locals mutated in at least one case can now be merged,
                // by creating `Select` outputs for all of them.
                for (local_idx, per_case_local_values) in local_idx_to_per_case_values {
                    if let Ok(local) = &mut self.locals[local_idx] {
                        assert_eq!(per_case_local_values.len(), num_cases);

                        // HACK(eddyb) do not create outputs if all cases agree.
                        let v0 = per_case_local_values[0];
                        if per_case_local_values[1..].iter().all(|&v| v == v0) {
                            let prev_value = mem::replace(&mut local.value, v0);
                            self.mutation_log.push(LocalMutation { local_idx, prev_value });
                            continue;
                        }

                        let local_ty = local.ty.unwrap();

                        let select_outputs = &mut func.nodes[node].outputs;
                        let new_select_output_var = func.vars.define(
                            cx,
                            VarDecl {
                                attrs: Default::default(),
                                ty: local_ty,
                                def_parent: Either::Right(node),
                                def_idx: select_outputs.len().try_into().unwrap(),
                            },
                        );
                        select_outputs.push(new_select_output_var);

                        // FIXME(eddyb) avoid random access, perhaps by handling
                        // locals per-case, instead of cases per-local.
                        for (&case, per_case_local_value) in
                            func.nodes[node].child_regions.iter().zip(per_case_local_values)
                        {
                            func.regions[case].outputs.push(
                                per_case_local_value.unwrap_or_else(|| const_undef(local_ty)),
                            );
                        }

                        let prev_value = local.value.replace(Value::Var(new_select_output_var));
                        self.mutation_log.push(LocalMutation { local_idx, prev_value });
                    }
                }
            }
            NodeKind::Loop { repeat_condition: _ } => {
                let body = child_regions[0];

                // HACK(eddyb) as the body of the loop may execute multiple times,
                // the initial states of locals have to account for potential
                // mutations in previous iterations, which we detect with this
                // separate visitor, then plumb through the region inputs/outputs.
                let mut mutated_local_indices = {
                    let mut mutation_finder = FindMutatedLocals {
                        propagator: self,
                        mutated_local_indices: FxIndexSet::default(),
                    };
                    mutation_finder.visit_region_def(func.reborrow().freeze().at(body));
                    mutation_finder.mutated_local_indices
                };
                mutated_local_indices.retain(|&local_idx| match &mut self.locals[local_idx] {
                    Ok(local) => {
                        let local_ty = local.ty.unwrap();

                        let body_inputs = &mut func.regions[body].inputs;
                        let new_body_input_var = func.vars.define(
                            cx,
                            VarDecl {
                                attrs: Default::default(),
                                ty: local_ty,
                                def_parent: Either::Left(body),
                                def_idx: body_inputs.len().try_into().unwrap(),
                            },
                        );
                        body_inputs.push(new_body_input_var);

                        let prev_value = local.value.replace(Value::Var(new_body_input_var));

                        func.nodes[node]
                            .inputs
                            .push(prev_value.unwrap_or_else(|| const_undef(local_ty)));

                        // NOTE(eddyb) can't avoid this, because the original
                        // values of mutated locals would otherwise be lost.
                        self.mutation_log.push(LocalMutation { local_idx, prev_value });

                        true
                    }
                    Err(_) => false,
                });

                let body_mutation_log_start = self.mutation_log.len();
                self.propagate_through_region(func.reborrow().at(body));

                // Record the updated values of locals, for future iterations.
                let body_outputs = &mut func.regions[body].outputs;
                let loop_outputs = &mut func.nodes[node].outputs;
                for &local_idx in &mutated_local_indices {
                    // HACK(eddyb) we require `FindMutatedLocals` to perfectly
                    // model all the situations in which we may reach an error
                    // (i.e. `UnknowableLocal`), and in which locals get
                    // mutated, because we may have *already* replaced loads in
                    // `body` to refer to values stored *in previous iterations*,
                    // so we need those values to actually be always usable.
                    let local = self.locals[local_idx].as_mut().ok().unwrap();
                    let local_ty = local.ty.unwrap();

                    let new_loop_output_var = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: local_ty,
                            def_parent: Either::Right(node),
                            def_idx: loop_outputs.len().try_into().unwrap(),
                        },
                    );
                    loop_outputs.push(new_loop_output_var);

                    let value_after_body = local.value.replace(Value::Var(new_loop_output_var));
                    body_outputs.push(value_after_body.unwrap());
                }

                // HACK(eddyb) because we already recorded all the mutations
                // based on `mutated_local_indices` alone, we can discard all the
                // redundant log entries (this also doubles as a sanity check).
                // FIXME(eddyb) this requires two passes to avoid new allocations
                // for deduplicating the set mutated locals - perhaps it may
                // be possible for `mutation_log` to always deduplicate itself
                // "since the most recent snapshot" or something?
                for mutation in &self.mutation_log[body_mutation_log_start..] {
                    assert!(mutated_local_indices.contains(&mutation.local_idx));
                }
                for mutation in self.mutation_log.drain(body_mutation_log_start..) {
                    mutated_local_indices.swap_remove(&mutation.local_idx);
                }
                assert_eq!(mutated_local_indices.len(), 0);
            }

            NodeKind::Mem(MemOp::FuncLocalVar(layout)) => {
                assert!(inputs.len() <= 1);
                let init_value = inputs.first().copied();

                self.locals.insert(
                    LocalKey { qptr_output: outputs[0] },
                    Ok(LocalContents {
                        parent_region,
                        size: layout.size,
                        ty: init_value.map(|v| func.reborrow().freeze().at(v).type_of(cx)),
                        value: init_value,
                        stores_with_parent_region: Default::default(),
                    }),
                );
            }

            NodeKind::Mem(MemOp::Load { offset }) => {
                assert_eq!(inputs.len(), 1);
                let src_ptr = inputs[0];

                let access_type = func.vars[outputs[0]].ty;

                if let Some(access) = self.lookup_local_for_access(src_ptr, offset, access_type) {
                    let local_ty = access.local.ty.unwrap();

                    // HACK(eddyb) cache the `OpUndef` constant in-place.
                    let local_value =
                        *access.local.value.get_or_insert_with(|| const_undef(local_ty));

                    match access.vector_elem_idx {
                        // Loads of the wrong type (but right size) don't need to
                        // have their uses replaced, but rather become bitcasts.
                        None if local_ty != access_type => {
                            // FIXME(eddyb) SPIR-T-native bitcasts.
                            *kind = NodeKind::SpvInst(
                                crate::spv::spec::Spec::get().well_known.OpBitcast.into(),
                                Default::default(),
                            );
                            *inputs = [local_value].into_iter().collect();
                        }

                        None => {
                            self.propagated_loads.insert(outputs[0], (local_value, parent_region));
                            // FIXME(eddyb) maybe remove the node here and now?
                        }

                        // Element loads from vector locals don't need to
                        // have their uses replaced, but rather become extracts.
                        Some(elem_idx) => {
                            assert!(
                                local_ty.as_vector(cx).unwrap().elem
                                    == access_type.as_scalar(cx).unwrap(),
                            );

                            *kind = vector::Op::from(vector::WholeOp::Extract { elem_idx }).into();
                            *inputs = [local_value].into_iter().collect();
                        }
                    }

                    return;
                }
            }

            NodeKind::Mem(MemOp::Store { offset }) => {
                assert_eq!(inputs.len(), 2);
                let dst_ptr = inputs[0];
                let stored_value = inputs[1];

                let access_type = func.reborrow().freeze().at(stored_value).type_of(cx);

                if let Some(access) = self.lookup_local_for_access(dst_ptr, offset, access_type) {
                    let local_ty = access.local.ty.unwrap();

                    let new_local_value = match access.vector_elem_idx {
                        // Stores of the wrong type (but right size) need extra
                        // bitcasts, to ensure `local.value` never changes type.
                        //
                        // FIXME(eddyb) avoid this by only bitcasting on loads,
                        // but also on merges (which can't assume matching type).
                        None if local_ty != access_type => {
                            let bitcast_node = func.nodes.define(
                                cx,
                                NodeDef {
                                    // FIXME(eddyb) preserve at least debuginfo attrs.
                                    attrs: Default::default(),
                                    // FIXME(eddyb) SPIR-T-native bitcasts.
                                    kind: NodeKind::SpvInst(
                                        crate::spv::spec::Spec::get().well_known.OpBitcast.into(),
                                        Default::default(),
                                    ),
                                    inputs: [stored_value].into_iter().collect(),
                                    child_regions: [].into_iter().collect(),
                                    outputs: [].into_iter().collect(),
                                }
                                .into(),
                            );
                            let bitcast_output_var = func.vars.define(
                                cx,
                                VarDecl {
                                    attrs: Default::default(),
                                    ty: local_ty,
                                    def_parent: Either::Right(bitcast_node),
                                    def_idx: 0,
                                },
                            );
                            func.nodes[bitcast_node].outputs.push(bitcast_output_var);

                            // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                            // due to the need to borrow `regions` and `nodes`
                            // at the same time - perhaps some kind of `FuncAtMut` position
                            // types for "where a list is in a parent entity" could be used
                            // to make this more ergonomic, although the potential need for
                            // an actual list entity of its own, should be considered.
                            func.regions[parent_region].children.insert_before(
                                bitcast_node,
                                node,
                                func.nodes,
                            );

                            Value::Var(bitcast_output_var)
                        }

                        None => stored_value,

                        // Element stores into vector locals become inserts,
                        // but because we don't know yet if the store will be
                        // removed (as the local can still escape later, or
                        // change type, etc.), the insert needs to be separate.
                        Some(elem_idx) => {
                            assert!(
                                local_ty.as_vector(cx).unwrap().elem
                                    == access_type.as_scalar(cx).unwrap(),
                            );

                            // HACK(eddyb) cache the `OpUndef` constant in-place
                            // (this may seem unnecessary, but the `mutation_log`
                            // will record the `OpUndef` as the `prev_value`).
                            let local_value =
                                *access.local.value.get_or_insert_with(|| const_undef(local_ty));

                            let vector_insert_node = func.nodes.define(
                                cx,
                                NodeDef {
                                    // FIXME(eddyb) preserve at least debuginfo attrs.
                                    attrs: Default::default(),
                                    kind: vector::Op::from(vector::WholeOp::Insert { elem_idx })
                                        .into(),
                                    inputs: [stored_value, local_value].into_iter().collect(),
                                    child_regions: [].into_iter().collect(),
                                    outputs: [].into_iter().collect(),
                                }
                                .into(),
                            );
                            let vector_insert_output_var = func.vars.define(
                                cx,
                                VarDecl {
                                    attrs: Default::default(),
                                    ty: local_ty,
                                    def_parent: Either::Right(vector_insert_node),
                                    def_idx: 0,
                                },
                            );
                            func.nodes[vector_insert_node].outputs.push(vector_insert_output_var);

                            // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                            // due to the need to borrow `regions` and `nodes`
                            // at the same time - perhaps some kind of `FuncAtMut` position
                            // types for "where a list is in a parent entity" could be used
                            // to make this more ergonomic, although the potential need for
                            // an actual list entity of its own, should be considered.
                            func.regions[parent_region].children.insert_before(
                                vector_insert_node,
                                node,
                                func.nodes,
                            );

                            Value::Var(vector_insert_output_var)
                        }
                    };

                    let prev_value = access.local.value.replace(new_local_value);
                    access.local.stores_with_parent_region.push((node, parent_region));
                    let local_idx = access.local_idx;
                    self.mutation_log.push(LocalMutation { local_idx, prev_value });

                    // Only visit the value input, not the destination pointer.
                    self.track_value_uses(&[stored_value]);

                    return;
                }
            }

            NodeKind::ExitInvocation(crate::cf::ExitInvocationKind::SpvInst(_))
            | NodeKind::Scalar(_)
            | NodeKind::Vector(_)
            | NodeKind::FuncCall(_)
            | NodeKind::Mem(_)
            | NodeKind::QPtr(_)
            | NodeKind::ThunkBind(_)
            | NodeKind::SpvInst(..)
            | NodeKind::SpvExtInst { .. } => {}
        }

        let NodeDef { attrs: _, kind, inputs, child_regions, outputs: _ } = func.at(node).def();

        // HACK(eddyb) this uses "has child regions" as "has nested dataflow".
        if !child_regions.is_empty() {
            // NOTE(eddyb) inputs tracked before the `match` above.

            // HACK(eddyb) semantically, `repeat_condition` is a body region output.
            if let NodeKind::Loop { repeat_condition } = kind {
                self.propagate_into_values(slice::from_mut(repeat_condition));
                self.track_value_uses(&[*repeat_condition]);
            }
        } else {
            self.track_value_uses(inputs);
        }
    }
}

/// Helper `Visitor` used when propagating locals across a `Loop`, to
/// determine *ahead of time* which locals require `Region` inputs.
struct FindMutatedLocals<'a, 'b> {
    propagator: &'a mut PropagateLocalContents<'b>,

    /// Indices of mutated locals, in the `propagator.locals` `IndexMap`.
    // FIXME(eddyb) this could probably be a compact bitset.
    // FIXME(eddyb) a more accurate check would also consider whether values from
    // previous iterations (or before the loop) are needed, not just mutations.
    mutated_local_indices: FxIndexSet<usize>,
}

impl Visitor<'_> for FindMutatedLocals<'_, '_> {
    // FIXME(eddyb) this is excessive, maybe different kinds of
    // visitors should exist for module-level and func-level?
    fn visit_attr_set_use(&mut self, _: AttrSet) {}
    fn visit_type_use(&mut self, _: Type) {}
    fn visit_const_use(&mut self, _: Const) {}
    fn visit_global_var_use(&mut self, _: GlobalVar) {}
    fn visit_func_use(&mut self, _: Func) {}

    // NOTE(eddyb) uses of locals that end up here disable tracking of
    // that local's contents (see also `UnknowableLocal`).
    fn visit_value_use(&mut self, &v: &Value) {
        if let Some(local_key) = LocalKey::maybe_from_qptr_value(v)
            && let Some(local) = self.propagator.locals.get_mut(&local_key)
        {
            *local = Err(UnknowableLocal);
        }
    }

    fn visit_node_def(&mut self, func_at_node: FuncAt<'_, Node>) {
        let node_def = func_at_node.def();

        let first_input_qptr_with_offset_and_access_type = match node_def.kind {
            // HACK(eddyb) declaring locals in loops is unsupported.
            NodeKind::Mem(MemOp::FuncLocalVar(_)) => {
                self.propagator
                    .locals
                    .insert(LocalKey { qptr_output: node_def.outputs[0] }, Err(UnknowableLocal));

                None
            }

            // NOTE(eddyb) these need to match up exactly with
            // `propagate_through_node`, for correctness.
            NodeKind::Mem(MemOp::Load { offset }) => {
                Some((offset, func_at_node.at(node_def.outputs[0]).decl().ty))
            }
            NodeKind::Mem(MemOp::Store { offset }) => {
                Some((offset, func_at_node.at(node_def.inputs[1]).type_of(self.propagator.cx)))
            }

            _ => None,
        };
        if let Some((offset, access_type)) = first_input_qptr_with_offset_and_access_type
            && let Some(access) =
                self.propagator.lookup_local_for_access(node_def.inputs[0], offset, access_type)
        {
            // FIXME(eddyb) a more accurate check would also
            // consider whether values from previous iterations
            // (or before the loop) are needed, not just mutations.
            let _needs_previous_value = matches!(node_def.kind, NodeKind::Mem(MemOp::Load { .. }))
                || access.vector_elem_idx.is_some();

            if let NodeKind::Mem(MemOp::Store { .. }) = node_def.kind {
                self.mutated_local_indices.insert(access.local_idx);
            }

            // Only visit the *other* inputs, not the `qptr` one.
            for v in &node_def.inputs[1..] {
                self.visit_value_use(v);
            }

            return;
        }

        func_at_node.inner_visit_with(self);
    }
}
