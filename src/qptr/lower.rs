//! [`QPtr`](crate::TypeKind::QPtr) lowering (e.g. from SPIR-V).

use crate::func_at::FuncAtMut;
use crate::mem::{MemOp, shapes};
use crate::qptr::{QPtrAttr, QPtrOp};
use crate::transform::{InnerInPlaceTransform, Transformed, Transformer};
use crate::{
    AddrSpace, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst, DataInstDef,
    DataInstKind, Diag, FuncDecl, GlobalVarDecl, Node, NodeKind, NodeOutputDecl, OrdAssertEq,
    Region, Type, TypeKind, TypeOrConst, Value, spv,
};
use itertools::Itertools as _;
use smallvec::SmallVec;
use std::cell::Cell;
use std::num::NonZeroU32;
use std::rc::Rc;

// HACK(eddyb) sharing layout code with other modules.
use crate::mem::layout::*;

struct LowerError(Diag);

/// Context for lowering SPIR-V `OpTypePointer`s to `QPtr`s.
///
/// See also `passes::qptr::lower_from_spv_ptrs` (which drives this).
pub struct LowerFromSpvPtrs<'a> {
    cx: Rc<Context>,
    wk: &'static spv::spec::WellKnown,
    layout_cache: LayoutCache<'a>,

    cached_qptr_type: Cell<Option<Type>>,
}

impl<'a> LowerFromSpvPtrs<'a> {
    pub fn new(cx: Rc<Context>, layout_config: &'a LayoutConfig) -> Self {
        Self {
            cx: cx.clone(),
            wk: &spv::spec::Spec::get().well_known,
            layout_cache: LayoutCache::new(cx, layout_config),
            cached_qptr_type: Default::default(),
        }
    }

    pub fn lower_global_var(&self, global_var_decl: &mut GlobalVarDecl) {
        let wk = self.wk;

        let (_, pointee_type) = self.as_spv_ptr_type(global_var_decl.type_of_ptr_to).unwrap();
        let handle_layout_to_handle = |handle_layout: HandleLayout| match handle_layout {
            shapes::Handle::Opaque(ty) => shapes::Handle::Opaque(ty),
            shapes::Handle::Buffer(addr_space, buf) => {
                shapes::Handle::Buffer(addr_space, buf.mem_layout)
            }
        };
        let mut shape_result = self.layout_of(pointee_type).and_then(|layout| {
            Ok(match layout {
                TypeLayout::Handle(handle) => shapes::GlobalVarShape::Handles {
                    handle: handle_layout_to_handle(handle),
                    fixed_count: Some(NonZeroU32::new(1).unwrap()),
                },
                TypeLayout::HandleArray(handle, fixed_count) => shapes::GlobalVarShape::Handles {
                    handle: handle_layout_to_handle(handle),
                    fixed_count,
                },
                TypeLayout::Concrete(concrete) => {
                    if concrete.mem_layout.dyn_unit_stride.is_some() {
                        return Err(LowerError(Diag::err([
                            "global variable cannot have dynamically sized type `".into(),
                            pointee_type.into(),
                            "`".into(),
                        ])));
                    }
                    match global_var_decl.addr_space {
                        // These SPIR-V Storage Classes are defined to require
                        // exact types, either because they're `BuiltIn`s, or
                        // for "interface matching" between pipeline stages.
                        AddrSpace::SpvStorageClass(sc)
                            if [
                                wk.Input,
                                wk.Output,
                                wk.IncomingRayPayloadKHR,
                                wk.IncomingCallableDataKHR,
                                wk.HitAttributeKHR,
                                wk.RayPayloadKHR,
                                wk.CallableDataKHR,
                            ]
                            .contains(&sc) =>
                        {
                            shapes::GlobalVarShape::TypedInterface(pointee_type)
                        }

                        _ => shapes::GlobalVarShape::UntypedData(concrete.mem_layout.fixed_base),
                    }
                }
            })
        });
        if let Ok(shapes::GlobalVarShape::Handles { handle, .. }) = &mut shape_result {
            match handle {
                shapes::Handle::Opaque(_) => {
                    if global_var_decl.addr_space != AddrSpace::SpvStorageClass(wk.UniformConstant)
                    {
                        shape_result = Err(LowerError(Diag::bug([
                            "opaque Handles require UniformConstant".into(),
                        ])));
                    }
                }
                // FIXME(eddyb) not all "interface blocks" imply buffers, so this
                // may need to be ignored based on the SPIR-V storage class.
                //
                // OH GOD but the lowering of operations to the right thing.......
                // depends on whether it's a buffer or not...... outside of
                // Rust-GPU's abuse of `Generic` it should at least be possible
                // to determine it from the pointer type itself, at the lowering
                // op time, but with storage class inference.... THIS IS FUCKED
                // OTOH, Rust-GPU doesn't really use `Block` outside of buffers!
                // Long-term it should probably have different types per storage
                // class, or even represent buffers as pointers.
                shapes::Handle::Buffer(buf_addr_space, _) => {
                    // HACK(eddyb) it couldn't have been known in `layout_of`.
                    assert!(*buf_addr_space == AddrSpace::Handles);
                    *buf_addr_space = global_var_decl.addr_space;
                }
            }
            if shape_result.is_ok() {
                global_var_decl.addr_space = AddrSpace::Handles;
            }
        }
        match shape_result {
            Ok(shape) => {
                global_var_decl.shape = Some(shape);

                // HACK(eddyb) this should handle shallow `QPtr` in the initializer, but
                // typed initializers should be replaced with miri/linker-style ones.
                EraseSpvPtrs { lowerer: self }.in_place_transform_global_var_decl(global_var_decl);
            }
            Err(LowerError(e)) => {
                global_var_decl.attrs.push_diag(&self.cx, e);
            }
        }
    }

    pub fn lower_func(&self, func_decl: &mut FuncDecl) {
        // HACK(eddyb) two-step to avoid having to record the original types
        // separately - so `LowerFromSpvPtrInstsInFunc` will leave all value defs
        // (including replaced instructions!) with unchanged `OpTypePointer`
        // types, that only `EraseSpvPtrs`, later, replaces with `QPtr`.
        LowerFromSpvPtrInstsInFunc { lowerer: self, parent_region: None }
            .in_place_transform_func_decl(func_decl);
        EraseSpvPtrs { lowerer: self }.in_place_transform_func_decl(func_decl);
    }

    /// Returns `Some` iff `ty` is a SPIR-V `OpTypePointer`.
    //
    // FIXME(eddyb) deduplicate with `qptr::lift`.
    //
    // FIXME(eddyb) consider using the storage class to determine whether a
    // `Block`-annotated type is a buffer or just interface nonsense.
    // (!!! may cause bad interactions with storage class inference `Generic` abuse)
    fn as_spv_ptr_type(&self, ty: Type) -> Option<(AddrSpace, Type)> {
        match &self.cx[ty].kind {
            TypeKind::SpvInst { spv_inst, type_and_const_inputs }
                if spv_inst.opcode == self.wk.OpTypePointer =>
            {
                let sc = match spv_inst.imms[..] {
                    [spv::Imm::Short(_, sc)] => sc,
                    _ => unreachable!(),
                };

                // HACK(eddyb) keep function pointers separate, perhaps eventually
                // adding an `OpTypeUntypedPointerKHR CodeSectionINTEL` equivalent
                // to SPIR-T itself (after `SPV_KHR_untyped_pointers` support).
                if sc == self.wk.CodeSectionINTEL {
                    return None;
                }

                let pointee = match type_and_const_inputs[..] {
                    [TypeOrConst::Type(elem_type)] => elem_type,
                    _ => unreachable!(),
                };
                Some((AddrSpace::SpvStorageClass(sc), pointee))
            }
            _ => None,
        }
    }

    // FIXME(eddyb) properly distinguish between zero-extension and sign-extension.
    fn const_as_u32(&self, ct: Const) -> Option<u32> {
        if let ConstKind::SpvInst { spv_inst_and_const_inputs } = &self.cx[ct].kind {
            let (spv_inst, _const_inputs) = &**spv_inst_and_const_inputs;
            if spv_inst.opcode == self.wk.OpConstant && spv_inst.imms.len() == 1 {
                match spv_inst.imms[..] {
                    [spv::Imm::Short(_, x)] => return Some(x),
                    _ => unreachable!(),
                }
            }
        }
        None
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

    /// Attempt to compute a `TypeLayout` for a given (SPIR-V) `Type`.
    fn layout_of(&self, ty: Type) -> Result<TypeLayout, LowerError> {
        self.layout_cache.layout_of(ty).map_err(|LayoutError(err)| LowerError(err))
    }
}

struct EraseSpvPtrs<'a> {
    lowerer: &'a LowerFromSpvPtrs<'a>,
}

impl Transformer for EraseSpvPtrs<'_> {
    // FIXME(eddyb) this is intentionally *shallow* and will not handle pointers
    // "hidden" in composites (which should be handled in SPIR-T explicitly).
    fn transform_type_use(&mut self, ty: Type) -> Transformed<Type> {
        // FIXME(eddyb) maybe cache this remap (in `LowerFromSpvPtrs`, globally).
        if self.lowerer.as_spv_ptr_type(ty).is_some() {
            Transformed::Changed(self.lowerer.qptr_type())
        } else {
            Transformed::Unchanged
        }
    }

    // FIXME(eddyb) this is intentionally *shallow* and will not handle pointers
    // "hidden" in composites (which should be handled in SPIR-T explicitly).
    fn transform_const_use(&mut self, ct: Const) -> Transformed<Const> {
        // FIXME(eddyb) maybe cache this remap (in `LowerFromSpvPtrs`, globally).
        let ct_def = &self.lowerer.cx[ct];
        if let ConstKind::PtrToGlobalVar(_) = ct_def.kind {
            Transformed::Changed(self.lowerer.cx.intern(ConstDef {
                attrs: ct_def.attrs,
                ty: self.lowerer.qptr_type(),
                kind: ct_def.kind.clone(),
            }))
        } else {
            Transformed::Unchanged
        }
    }
}

struct LowerFromSpvPtrInstsInFunc<'a> {
    lowerer: &'a LowerFromSpvPtrs<'a>,

    parent_region: Option<Region>,
}

/// One `QPtr`->`QPtr` step used in the lowering of `Op*AccessChain`.
///
/// The `op` should take a `QPtr` as its first input and produce a `QPtr`.
struct QPtrChainStep {
    op: QPtrOp,

    /// For `QPtrOp::HandleArrayIndex` and `QPtrOp::DynOffset`, this is the
    /// second input (after the `QPtr` which is automatically handled).
    dyn_idx: Option<Value>,
}

impl QPtrChainStep {
    fn into_data_inst_kind_and_inputs(
        self,
        in_qptr: Value,
    ) -> (DataInstKind, SmallVec<[Value; 2]>) {
        let Self { op, dyn_idx } = self;
        (op.into(), [in_qptr].into_iter().chain(dyn_idx).collect())
    }
}

impl LowerFromSpvPtrInstsInFunc<'_> {
    fn try_lower_access_chain(
        &self,
        mut layout: TypeLayout,
        indices: &[Value],
    ) -> Result<SmallVec<[QPtrChainStep; 4]>, LowerError> {
        // FIXME(eddyb) pass in the `AddrSpace` to determine this correctly.
        let is_logical_addressing = true;

        let const_idx_as_i32 = |idx| match idx {
            // FIXME(eddyb) figure out the signedness semantics here.
            Value::Const(idx) => self.lowerer.const_as_u32(idx).map(|idx_u32| idx_u32 as i32),
            _ => None,
        };

        let mut steps: SmallVec<[QPtrChainStep; 4]> = SmallVec::new();
        let mut indices = indices.iter().copied();
        while indices.len() > 0 {
            let (mut op, component_layout) = match layout {
                TypeLayout::Handle(shapes::Handle::Opaque(_)) => {
                    return Err(LowerError(Diag::bug([
                        "opaque handles have no sub-components".into()
                    ])));
                }
                TypeLayout::Handle(shapes::Handle::Buffer(_, buffer_data_layout)) => {
                    (QPtrOp::BufferData, TypeLayout::Concrete(buffer_data_layout))
                }
                TypeLayout::HandleArray(handle, _) => {
                    (QPtrOp::HandleArrayIndex, TypeLayout::Handle(handle))
                }
                TypeLayout::Concrete(concrete) => match &concrete.components {
                    Components::Scalar => {
                        return Err(LowerError(Diag::bug([
                            "scalars have no sub-components".into()
                        ])));
                    }
                    // FIXME(eddyb) handle the weird `OpTypeMatrix` layout when `RowMajor`.
                    Components::Elements { stride, elem, fixed_len } => (
                        QPtrOp::DynOffset {
                            stride: *stride,
                            // FIXME(eddyb) even without a fixed length, logical
                            // addressing still implies the index is *positive*,
                            // that should be encoded here, to help analysis.
                            index_bounds: fixed_len
                                .filter(|_| is_logical_addressing)
                                .and_then(|len| Some(0..len.get().try_into().ok()?)),
                        },
                        TypeLayout::Concrete(elem.clone()),
                    ),
                    Components::Fields { offsets, layouts } => {
                        let field_idx =
                            const_idx_as_i32(indices.next().unwrap()).ok_or_else(|| {
                                LowerError(Diag::bug(["non-constant field index".into()]))
                            })?;
                        let (field_offset, field_layout) = usize::try_from(field_idx)
                            .ok()
                            .and_then(|field_idx| {
                                Some((*offsets.get(field_idx)?, layouts.get(field_idx)?.clone()))
                            })
                            .ok_or_else(|| {
                                LowerError(Diag::bug([format!(
                                    "field {field_idx} out of bounds (expected 0..{})",
                                    offsets.len()
                                )
                                .into()]))
                            })?;
                        (
                            QPtrOp::Offset(i32::try_from(field_offset).ok().ok_or_else(|| {
                                LowerError(Diag::bug([format!(
                                    "{field_offset} not representable as a positive s32"
                                )
                                .into()]))
                            })?),
                            TypeLayout::Concrete(field_layout),
                        )
                    }
                },
            };
            layout = component_layout;

            // Automatically grab the dynamic index, whenever necessary.
            let mut dyn_idx = match op {
                QPtrOp::HandleArrayIndex | QPtrOp::DynOffset { .. } => {
                    Some(indices.next().unwrap())
                }
                _ => None,
            };

            // Constant-fold dynamic indexing, whenever possible.
            if let QPtrOp::DynOffset { stride, index_bounds } = &op {
                let const_offset = const_idx_as_i32(dyn_idx.unwrap())
                    .filter(|const_idx| {
                        index_bounds.as_ref().is_none_or(|bounds| bounds.contains(const_idx))
                    })
                    .and_then(|const_idx| i32::try_from(stride.get()).ok()?.checked_mul(const_idx));
                if let Some(const_offset) = const_offset {
                    op = QPtrOp::Offset(const_offset);
                    dyn_idx = None;
                }
            }

            // Combine consecutive immediate offsets, whenever possible.
            match (steps.last_mut().map(|last_step| &mut last_step.op), &op) {
                // Complete ignore noop offsets.
                (_, QPtrOp::Offset(0)) => {}

                (Some(QPtrOp::Offset(last_offset)), &QPtrOp::Offset(new_offset)) => {
                    *last_offset = last_offset.checked_add(new_offset).ok_or_else(|| {
                        LowerError(Diag::bug([format!(
                            "offset overflow ({last_offset}+{new_offset})"
                        )
                        .into()]))
                    })?;
                }

                _ => steps.push(QPtrChainStep { op, dyn_idx }),
            }
        }
        Ok(steps)
    }

    fn try_lower_data_inst_def(
        &self,
        mut func_at_data_inst: FuncAtMut<'_, DataInst>,
    ) -> Result<Transformed<DataInstDef>, LowerError> {
        let cx = &self.lowerer.cx;
        let wk = self.lowerer.wk;

        let func_at_data_inst_frozen = func_at_data_inst.reborrow().freeze();
        let data_inst = func_at_data_inst_frozen.position;
        let data_inst_def = func_at_data_inst_frozen.def();

        // FIXME(eddyb) is this a good convention?
        let func = func_at_data_inst_frozen.at(());

        let mut attrs = data_inst_def.attrs;

        let spv_inst = match &data_inst_def.kind {
            DataInstKind::SpvInst(spv_inst) => spv_inst,
            _ => return Ok(Transformed::Unchanged),
        };

        // FIXME(eddyb) wasteful clone? (needed due to borrowing issues)
        let outputs = data_inst_def.outputs.clone();

        let replacement_kind_and_inputs = if spv_inst.opcode == wk.OpVariable {
            assert!(data_inst_def.inputs.len() <= 1);
            let (_, var_data_type) =
                self.lowerer.as_spv_ptr_type(outputs[0].ty).ok_or_else(|| {
                    LowerError(Diag::bug(["output type not an `OpTypePointer`".into()]))
                })?;
            match self.lowerer.layout_of(var_data_type)? {
                TypeLayout::Concrete(concrete) if concrete.mem_layout.dyn_unit_stride.is_none() => {
                    (
                        MemOp::FuncLocalVar(concrete.mem_layout.fixed_base).into(),
                        data_inst_def.inputs.clone(),
                    )
                }
                _ => return Ok(Transformed::Unchanged),
            }
        } else if spv_inst.opcode == wk.OpLoad {
            // FIXME(eddyb) support memory operands somehow.
            if !spv_inst.imms.is_empty() {
                return Ok(Transformed::Unchanged);
            }
            assert_eq!(data_inst_def.inputs.len(), 1);
            (MemOp::Load.into(), data_inst_def.inputs.clone())
        } else if spv_inst.opcode == wk.OpStore {
            // FIXME(eddyb) support memory operands somehow.
            if !spv_inst.imms.is_empty() {
                return Ok(Transformed::Unchanged);
            }
            assert_eq!(data_inst_def.inputs.len(), 2);
            (MemOp::Store.into(), data_inst_def.inputs.clone())
        } else if spv_inst.opcode == wk.OpArrayLength {
            let field_idx = match spv_inst.imms[..] {
                [spv::Imm::Short(_, field_idx)] => field_idx,
                _ => unreachable!(),
            };
            assert_eq!(data_inst_def.inputs.len(), 1);
            let ptr = data_inst_def.inputs[0];
            let (_, pointee_type) =
                self.lowerer.as_spv_ptr_type(func.at(ptr).type_of(cx)).ok_or_else(|| {
                    LowerError(Diag::bug(["pointer input not an `OpTypePointer`".into()]))
                })?;

            let buf_data_layout = match self.lowerer.layout_of(pointee_type)? {
                TypeLayout::Handle(shapes::Handle::Buffer(_, buf)) => buf,
                _ => return Err(LowerError(Diag::bug(["non-Buffer pointee".into()]))),
            };

            let (field_offset, field_layout) = match &buf_data_layout.components {
                Components::Fields { offsets, layouts } => usize::try_from(field_idx)
                    .ok()
                    .and_then(|field_idx| {
                        Some((*offsets.get(field_idx)?, layouts.get(field_idx)?.clone()))
                    })
                    .ok_or_else(|| {
                        LowerError(Diag::bug([format!(
                            "field {field_idx} out of bounds (expected 0..{})",
                            offsets.len()
                        )
                        .into()]))
                    })?,

                _ => {
                    return Err(LowerError(Diag::bug(
                        ["buffer data not an `OpTypeStruct`".into()],
                    )));
                }
            };
            let array_stride = match field_layout.components {
                Components::Elements { stride, fixed_len: None, .. } => stride,

                _ => {
                    return Err(LowerError(Diag::bug([format!(
                        "buffer data field #{field_idx} not an `OpTypeRuntimeArray`"
                    )
                    .into()])));
                }
            };

            // Sanity-check layout invariants (should always hold given above checks).
            assert_eq!(field_layout.mem_layout.fixed_base.size, 0);
            assert_eq!(field_layout.mem_layout.dyn_unit_stride, Some(array_stride));
            assert_eq!(buf_data_layout.mem_layout.fixed_base.size, field_offset);
            assert_eq!(buf_data_layout.mem_layout.dyn_unit_stride, Some(array_stride));

            (
                QPtrOp::BufferDynLen {
                    fixed_base_size: field_offset,
                    dyn_unit_stride: array_stride,
                }
                .into(),
                data_inst_def.inputs.clone(),
            )
        } else if [
            wk.OpAccessChain,
            wk.OpInBoundsAccessChain,
            wk.OpPtrAccessChain,
            wk.OpInBoundsPtrAccessChain,
        ]
        .contains(&spv_inst.opcode)
        {
            // FIXME(eddyb) avoid erasing the "inbounds" qualifier.
            let base_ptr = data_inst_def.inputs[0];
            let (_, base_pointee_type) =
                self.lowerer.as_spv_ptr_type(func.at(base_ptr).type_of(cx)).ok_or_else(|| {
                    LowerError(Diag::bug(["pointer input not an `OpTypePointer`".into()]))
                })?;

            // HACK(eddyb) for `OpPtrAccessChain`, this pretends to be indexing
            // a `OpTypeRuntimeArray`, with the original type as the element type.
            let access_chain_base_layout =
                if [wk.OpPtrAccessChain, wk.OpInBoundsPtrAccessChain].contains(&spv_inst.opcode) {
                    self.lowerer.layout_of(cx.intern(TypeKind::SpvInst {
                        spv_inst: wk.OpTypeRuntimeArray.into(),
                        type_and_const_inputs:
                            [TypeOrConst::Type(base_pointee_type)].into_iter().collect(),
                    }))?
                } else {
                    self.lowerer.layout_of(base_pointee_type)?
                };

            let mut steps =
                self.try_lower_access_chain(access_chain_base_layout, &data_inst_def.inputs[1..])?;
            // HACK(eddyb) noop cases should probably not use any `DataInst`s at all,
            // but that would require the ability to replace all uses of a `Value`.
            let final_step =
                steps.pop().unwrap_or(QPtrChainStep { op: QPtrOp::Offset(0), dyn_idx: None });

            let mut ptr = base_ptr;
            for step in steps {
                let (kind, inputs) = step.into_data_inst_kind_and_inputs(ptr);
                let step_data_inst = func_at_data_inst.reborrow().nodes.define(
                    cx,
                    DataInstDef {
                        attrs: Default::default(),
                        kind,
                        inputs,
                        child_regions: [].into_iter().collect(),
                        outputs: [NodeOutputDecl {
                            attrs: Default::default(),
                            ty: self.lowerer.qptr_type(),
                        }]
                        .into_iter()
                        .collect(),
                    }
                    .into(),
                );

                // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                // due to the need to borrow `regions` and `nodes`
                // at the same time - perhaps some kind of `FuncAtMut` position
                // types for "where a list is in a parent entity" could be used
                // to make this more ergonomic, although the potential need for
                // an actual list entity of its own, should be considered.
                let func = func_at_data_inst.reborrow().at(());
                func.regions[self.parent_region.unwrap()].children.insert_before(
                    step_data_inst,
                    data_inst,
                    func.nodes,
                );

                ptr = Value::NodeOutput { node: step_data_inst, output_idx: 0 };
            }
            final_step.into_data_inst_kind_and_inputs(ptr)
        } else if spv_inst.opcode == wk.OpBitcast {
            let input = data_inst_def.inputs[0];
            // Pointer-to-pointer casts are noops on `qptr`.
            if self.lowerer.as_spv_ptr_type(func.at(input).type_of(cx)).is_some()
                && self.lowerer.as_spv_ptr_type(outputs[0].ty).is_some()
            {
                // HACK(eddyb) noop cases should not use any `DataInst`s at all,
                // but that would require the ability to replace all uses of a `Value`.
                let noop_step = QPtrChainStep { op: QPtrOp::Offset(0), dyn_idx: None };

                // HACK(eddyb) since we're not removing the `DataInst` entirely,
                // at least get rid of its attributes to clearly mark it as synthetic.
                attrs = AttrSet::default();

                noop_step.into_data_inst_kind_and_inputs(input)
            } else {
                return Ok(Transformed::Unchanged);
            }
        } else {
            return Ok(Transformed::Unchanged);
        };
        // FIXME(eddyb) should the `if`-`else` chain above produce `DataInstDef`s?
        let (new_kind, new_inputs) = replacement_kind_and_inputs;
        Ok(Transformed::Changed(DataInstDef {
            attrs,
            kind: new_kind,
            inputs: new_inputs,
            child_regions: [].into_iter().collect(),
            outputs,
        }))
    }

    fn add_fallback_attrs_to_data_inst_def(
        &self,
        mut func_at_data_inst: FuncAtMut<'_, DataInst>,
        extra_error: Option<LowerError>,
    ) {
        let cx = &self.lowerer.cx;

        let func_at_data_inst_frozen = func_at_data_inst.reborrow().freeze();
        let data_inst_def = func_at_data_inst_frozen.def();

        // FIXME(eddyb) is this a good convention?
        let func = func_at_data_inst_frozen.at(());

        match data_inst_def.kind {
            // Known semantics, no need to preserve SPIR-V pointer information.
            NodeKind::Select(_)
            | NodeKind::Loop { .. }
            | NodeKind::ExitInvocation(_)
            | DataInstKind::FuncCall(_)
            | DataInstKind::Mem(_)
            | DataInstKind::QPtr(_) => return,

            DataInstKind::SpvInst(_) | DataInstKind::SpvExtInst { .. } => {}
        }

        let mut old_and_new_attrs = None;
        let get_old_attrs = || AttrSetDef { attrs: cx[data_inst_def.attrs].attrs.clone() };

        for (input_idx, &v) in data_inst_def.inputs.iter().enumerate() {
            if let Some((_, pointee)) = self.lowerer.as_spv_ptr_type(func.at(v).type_of(cx)) {
                old_and_new_attrs.get_or_insert_with(get_old_attrs).attrs.insert(
                    QPtrAttr::ToSpvPtrInput {
                        input_idx: input_idx.try_into().unwrap(),
                        pointee: OrdAssertEq(pointee),
                    }
                    .into(),
                );
            }
        }
        // HACK(eddyb) multi-output instructions don't exist pre-disaggregate.
        if let Some(output) = data_inst_def.outputs.iter().at_most_one().ok().unwrap()
            && let Some((addr_space, pointee)) = self.lowerer.as_spv_ptr_type(output.ty)
        {
            old_and_new_attrs.get_or_insert_with(get_old_attrs).attrs.insert(
                QPtrAttr::FromSpvPtrOutput {
                    addr_space: OrdAssertEq(addr_space),
                    pointee: OrdAssertEq(pointee),
                }
                .into(),
            );
        }

        if let Some(LowerError(e)) = extra_error {
            old_and_new_attrs.get_or_insert_with(get_old_attrs).push_diag(e);
        }

        if let Some(attrs) = old_and_new_attrs {
            func_at_data_inst.def().attrs = cx.intern(attrs);
        }
    }
}

impl Transformer for LowerFromSpvPtrInstsInFunc<'_> {
    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_in_place_transform_with(self);
        self.parent_region = outer_region;
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        func_at_node.reborrow().inner_in_place_transform_with(self);

        match self.try_lower_data_inst_def(func_at_node.reborrow()) {
            Ok(Transformed::Changed(new_def)) => {
                *func_at_node.def() = new_def;
            }
            result @ (Ok(Transformed::Unchanged) | Err(_)) => {
                self.add_fallback_attrs_to_data_inst_def(func_at_node, result.err());
            }
        }
    }
}
