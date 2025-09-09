//! [`QPtr`](crate::TypeKind::QPtr) lowering (e.g. from SPIR-V).

use crate::cf::SelectionKind;
use crate::func_at::FuncAtMut;
use crate::mem::{MemOp, const_data, shapes};
use crate::qptr::{QPtrAttr, QPtrOp};
use crate::transform::{InnerInPlaceTransform, Transformed, Transformer};
use crate::{
    AddrSpace, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst, DataInstDef,
    DataInstKind, DeclDef, Diag, EntityOrientedDenseMap, FuncDecl, GlobalVarDecl, GlobalVarInit,
    Node, NodeDef, NodeKind, OrdAssertEq, Region, RegionDef, Type, TypeKind, TypeOrConst, Value,
    Var, VarDecl, VarKind, scalar, spv,
};
use itertools::{Either, Itertools};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cell::Cell;
use std::mem;
use std::num::{NonZeroI32, NonZeroU32};
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
        let addr_space_requires_typed_interface = match global_var_decl.addr_space {
            // These SPIR-V Storage Classes are defined to require
            // exact types, either because they're `BuiltIn`s, or
            // for "interface matching" between pipeline stages.
            AddrSpace::SpvStorageClass(sc) => [
                wk.Input,
                wk.Output,
                wk.IncomingRayPayloadKHR,
                wk.IncomingCallableDataKHR,
                wk.HitAttributeKHR,
                wk.RayPayloadKHR,
                wk.CallableDataKHR,
            ]
            .contains(&sc),

            AddrSpace::Handles => false,
        };
        let layout_result = self.layout_of(pointee_type);
        let concrete_mem_layout = layout_result.as_ref().ok().and_then(|layout| match layout {
            TypeLayout::Handle(_) | TypeLayout::HandleArray(..) => None,
            TypeLayout::Concrete(concrete) => Some(concrete.mem_layout),
        });
        let mut shape_result = layout_result.and_then(|layout| {
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
                    if addr_space_requires_typed_interface {
                        shapes::GlobalVarShape::TypedInterface(pointee_type)
                    } else {
                        shapes::GlobalVarShape::UntypedData(concrete.mem_layout.fixed_base)
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

        // HACK(eddyb) the interactions with `shape_result` are a bit too ad-hoc,
        // but they help testing for now (until Rust-GPU is more accurate).
        if let DeclDef::Present(global_var_def_body) = &mut global_var_decl.def {
            let lowered_init = global_var_def_body.initializer.as_ref().and_then(|init| {
                self.try_lower_global_var_init(init)
                    .map_err(|LowerError(e)| {
                        if shape_result.is_ok() {
                            shape_result = Err(LowerError(e));
                        } else {
                            global_var_decl.attrs.push_diag(&self.cx, e);
                        }
                    })
                    .ok()
            });
            if let Some(init) = lowered_init {
                // HACK(eddyb) recover the shape from the initializer.
                match (&shape_result, concrete_mem_layout, &init) {
                    (Err(_), Some(mem_layout), GlobalVarInit::Data(data))
                        if !addr_space_requires_typed_interface
                            && mem_layout.dyn_unit_stride.is_some()
                            && mem_layout.fixed_base.size <= data.size() =>
                    {
                        let mut fixed_layout = mem_layout.fixed_base;
                        fixed_layout.size = data.size();
                        shape_result = Ok(shapes::GlobalVarShape::UntypedData(fixed_layout));
                    }
                    _ => {}
                }
                if shape_result.is_ok() {
                    global_var_def_body.initializer = Some(init);
                }
            }
        }

        // HACK(eddyb) in case anything goes wrong, we want to keep `OpTypePointer`.
        let original_type_of_ptr_to = global_var_decl.type_of_ptr_to;

        EraseSpvPtrs { lowerer: self }.in_place_transform_global_var_decl(global_var_decl);

        match shape_result {
            Ok(shape) => {
                global_var_decl.shape = Some(shape);
            }
            Err(LowerError(e)) => {
                global_var_decl.attrs.push_diag(&self.cx, e);

                // HACK(eddyb) effectively undoes `EraseSpvPtrs` for one field.
                global_var_decl.type_of_ptr_to = original_type_of_ptr_to;
            }
        }
    }
    fn try_lower_global_var_init(
        &self,
        global_var_init: &GlobalVarInit,
    ) -> Result<GlobalVarInit, LowerError> {
        let (aggregate_type, aggregate_leaves) = match global_var_init {
            &GlobalVarInit::Direct(ct) => return Ok(GlobalVarInit::Direct(ct)),

            GlobalVarInit::Data(_) => {
                return Err(LowerError(Diag::bug([
                    "unexpected `GlobalVarInit::Data` (already lowered?)".into(),
                ])));
            }

            GlobalVarInit::SpvAggregate { ty, leaves } => (*ty, leaves),
        };
        let aggregate_layout = match self.layout_of(aggregate_type)? {
            // FIXME(eddyb) consider bad interactions with "interface blocks"?
            TypeLayout::Handle(_) | TypeLayout::HandleArray(..) => {
                return Err(LowerError(Diag::bug(["handles are not aggregates".into()])));
            }
            TypeLayout::Concrete(layout) => layout,
        };

        let mut leaf_values = aggregate_leaves.iter().copied();
        let mut data = const_data::ConstData::new(aggregate_layout.mem_layout.fixed_base.size);
        let result = aggregate_layout.deeply_flatten_if(
            0,
            // Whether `candidate_layout` is an aggregate (to recurse into).
            &|candidate_layout| {
                matches!(
                    &self.cx[candidate_layout.original_type].kind,
                    TypeKind::SpvInst { value_lowering: spv::ValueLowering::Disaggregate(_), .. }
                )
            },
            &mut |leaf_offset, leaf| {
                let leaf_offset = u32::try_from(leaf_offset).ok().ok_or_else(|| {
                    LayoutError(Diag::bug([format!(
                        "negative initializer leaf offset {leaf_offset}"
                    )
                    .into()]))
                })?;

                let leaf_value = leaf_values.next().ok_or_else(|| {
                    LayoutError(Diag::bug(["fewer initializer leaves than layout".into()]))
                })?;
                let leaf_value_def = &self.cx[leaf_value];

                // FIXME(eddyb) should this compare only size/shape?
                let expected_ty = leaf.original_type;
                let found_ty = leaf_value_def.ty;
                if expected_ty != found_ty {
                    return Err(LayoutError(Diag::bug([
                        "initializer leaf type mismatch: expected `".into(),
                        expected_ty.into(),
                        "`, found `".into(),
                        found_ty.into(),
                        "` typed value `".into(),
                        leaf_value.into(),
                        "`".into(),
                    ])));
                }

                let leaf_size =
                    NonZeroU32::new(leaf.mem_layout.fixed_base.size).ok_or_else(|| {
                        LayoutError(Diag::bug([
                            format!(
                                "zero-sized initializer leaf at offset {leaf_offset}, with value `"
                            )
                            .into(),
                            leaf_value.into(),
                            "`".into(),
                        ]))
                    })?;

                self.try_write_to_const_data_at(&mut data, leaf_offset, leaf_size, leaf_value)
            },
        );
        result.map_err(|LayoutError(e)| LowerError(e))?;

        if leaf_values.next().is_some() {
            return Err(LowerError(Diag::bug(["more initializer leaves than layout".into()])));
        }

        Ok(GlobalVarInit::Data(data))
    }
    // FIXME(eddyb) move this to a more general `ConstData` helper.
    fn try_write_to_const_data_at(
        &self,
        data: &mut const_data::ConstData<Const>,
        offset: u32,
        size: NonZeroU32,
        ct: Const,
    ) -> Result<(), LayoutError> {
        // HACK(eddyb) strip bitcasts as long as the input and output size match.
        let (ct, ct_def) = {
            let (mut ct, mut ct_def) = (ct, &self.cx[ct]);
            while let ConstKind::SpvInst { spv_inst_and_const_inputs } = &ct_def.kind {
                let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;

                if let (&[input], &[spv::Imm::Short(_, op)]) =
                    (&const_inputs[..], &spv_inst.imms[..])
                    && spv_inst.opcode == self.wk.OpSpecConstantOp
                    && op == u32::from(self.wk.OpBitcast.as_u16())
                {
                    let input_def = &self.cx[input];
                    let input_size =
                        self.layout_cache.layout_of(input_def.ty).ok().and_then(|layout| {
                            match layout {
                                TypeLayout::Concrete(layout)
                                    if layout.mem_layout.dyn_unit_stride.is_none() =>
                                {
                                    NonZeroU32::new(layout.mem_layout.fixed_base.size)
                                }
                                _ => None,
                            }
                        });
                    if input_size == Some(size) {
                        (ct, ct_def) = (input, input_def);
                        continue;
                    }
                }
                break;
            }
            (ct, ct_def)
        };

        let err_to_diag = |err| {
            let const_data::PartialSymbolicOverlap { offsets } = err;
            LayoutError(Diag::bug([
                format!("initializer leaf at offset {offset}, with value `").into(),
                ct.into(),
                format!("`, overlaps with leaf at offsets {offsets:?} (invalid layout?)").into(),
            ]))
        };

        let mut total_written_range = offset..offset;

        // HACK(eddyb) helper shared by `Scalar` and `Vector`.
        let mut write_next_scalar = |leaf_scalar: scalar::Const| {
            // FIXME(eddyb) try harder to avoid panicking due to out-of-bounds
            // offsets caused by e.g. malformed layouts (and/or guarantee certain
            // invariants for types that didn't error during layout computation).
            let written_range = data
                .write_scalar(total_written_range.end, leaf_scalar, self.layout_cache.config)
                .map_err(err_to_diag)?;
            total_written_range.end = written_range.end;
            Ok(())
        };

        match &ct_def.kind {
            // HACK(eddyb) Rust-GPU still uses `undef`
            // w/ custom attributes for some error cases,
            // so care must be taken until that's deemed
            // incorrect (if at all).
            // FIXME(eddyb) handle this elsewhere, too.
            ConstKind::Undef if ct_def.attrs == AttrSet::default() => {
                return Ok(());
            }

            &ConstKind::Scalar(leaf_scalar) => {
                write_next_scalar(leaf_scalar)?;
            }

            ConstKind::Vector(leaf_vector) => {
                for elem in leaf_vector.elems() {
                    write_next_scalar(elem)?;
                }
            }

            // FIXME(eddyb) try harder to avoid panicking due to out-of-bounds
            // offsets caused by e.g. malformed layouts (and/or guarantee certain
            // invariants for types that didn't error during layout computation).
            _ => {
                data.write_symbolic(offset, size, ct).map_err(err_to_diag)?;
                total_written_range.end += size.get();
            }
        }

        assert_eq!(total_written_range, offset..(offset + size.get()));

        Ok(())
    }

    pub fn lower_func(&self, func_decl: &mut FuncDecl) {
        // HACK(eddyb) two-step to avoid having to record the original types
        // separately - so `LowerFromSpvPtrInstsInFunc` will leave all value defs
        // (including replaced instructions!) with unchanged `OpTypePointer`
        // types, that only `EraseSpvPtrs`, later, replaces with `QPtr`.
        LowerFromSpvPtrInstsInFunc {
            lowerer: self,
            parent_region: None,
            var_use_counts: Default::default(),
            remove_inst_if_dead_output_with_parent_region: Default::default(),
            noop_offsets_to_base_ptr: Default::default(),
        }
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
            TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
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

    fn const_as_u32(&self, ct: Const) -> Option<u32> {
        // HACK(eddyb) lossless roundtrip through `i32` is most conservative
        // option (only `0..=i32::MAX`, i.e. `0 <= x < 2**32, is allowed).
        u32::try_from(ct.as_scalar(&self.cx)?.int_as_i32()?).ok()
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
        let cx = &self.lowerer.cx;
        let wk = self.lowerer.wk;

        // FIXME(eddyb) maybe cache this remap (in `LowerFromSpvPtrs`, globally).
        let ct_def = &cx[ct];

        // HACK(eddyb) Rust-GPU relies on `undef` with attached diagnostics
        // (either on the value or the type) to create invalid values.
        if let ConstKind::Undef = ct_def.kind
            && (!ct_def.attrs.diags(cx).is_empty() || !cx[ct_def.ty].attrs.diags(cx).is_empty())
        {
            return Transformed::Unchanged;
        }

        let erased_ct_def = match self.transform_const_def(ct_def) {
            Transformed::Unchanged => return Transformed::Unchanged,
            Transformed::Changed(erased_ct_def) => erased_ct_def,
        };
        match &erased_ct_def.kind {
            ConstKind::Undef | ConstKind::PtrToGlobalVar { .. } => {
                Transformed::Changed(cx.intern(erased_ct_def))
            }

            ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;

                match (&const_inputs[..], &spv_inst.imms[..]) {
                    // FIXME(eddyb) maybe `qptr` should have its own null constant?
                    ([], []) if spv_inst.opcode == wk.OpConstantNull => {
                        Transformed::Changed(cx.intern(erased_ct_def))
                    }

                    (&[input], &[spv::Imm::Short(_, op)])
                        if spv_inst.opcode == wk.OpSpecConstantOp
                            && op == u32::from(wk.OpBitcast.as_u16()) =>
                    {
                        // HACK(eddyb) elide `qptr` -> `qptr` noop bitcasts,
                        // originating from necessary SPIR-V `*T` -> `*U` casts.
                        Transformed::Changed(if cx[input].ty == erased_ct_def.ty {
                            input
                        } else {
                            cx.intern(erased_ct_def)
                        })
                    }

                    _ => Transformed::Unchanged,
                }
            }

            _ => Transformed::Unchanged,
        }
    }
}

struct LowerFromSpvPtrInstsInFunc<'a> {
    lowerer: &'a LowerFromSpvPtrs<'a>,

    parent_region: Option<Region>,

    var_use_counts: EntityOrientedDenseMap<Var, NonZeroU32>,

    // HACK(eddyb) this acts as a "queue" for `qptr` outputs of instructions,
    // which may end up dead because they're unused (either unused originally,
    // in SPIR-V, or because of offset folding).
    remove_inst_if_dead_output_with_parent_region: Vec<(Var, Region)>,

    // FIXME(eddyb) this is redundant with a few other things and only here
    // because it needs to be available from `transform_value`, which doesn't
    // have access to a `FuncAt` to look up anything.
    noop_offsets_to_base_ptr: FxHashMap<Var, Value>,
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
            Value::Var(_) => None,
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
        &mut self,
        mut func_at_data_inst: FuncAtMut<'_, DataInst>,
    ) -> Result<Transformed<DataInstDef>, LowerError> {
        let cx = &self.lowerer.cx;
        let wk = self.lowerer.wk;

        let func_at_data_inst_frozen = func_at_data_inst.reborrow().freeze();
        let data_inst = func_at_data_inst_frozen.position;
        let data_inst_def = func_at_data_inst_frozen.def();

        // FIXME(eddyb) is this a good convention?
        let func = func_at_data_inst_frozen.at(());

        let attrs = data_inst_def.attrs;

        let (spv_inst, spv_inst_lowering) = match &data_inst_def.kind {
            DataInstKind::SpvInst(spv_inst, lowering) => (spv_inst, lowering),

            // FIXME(eddyb) these (e.g. `ptr2int(p) == ptr2int(q) -> p == q`)
            // transforms belong in a general-purpose rewrite rule system, not here.
            // FIXME(eddyb) pointer equality could, in theory, fail due to
            // provenance mismatch, even when integer equality would fail,
            // but there is no way to encode this in a sane way yet, and
            // might require a "pointer equality by address" operation.
            &DataInstKind::Scalar(scalar::Op::IntBinary(
                cmp_op @ (scalar::IntBinOp::Eq | scalar::IntBinOp::Ne),
            )) => {
                let ptr_addr_bit_width = self
                    .lowerer
                    .layout_cache
                    .config
                    .logical_ptr_size_align
                    .0
                    .checked_mul(8)
                    .unwrap();
                let Some(int_scalar_ty) = data_inst_def
                    .inputs
                    .iter()
                    .map(|&v| func.at(v).type_of(cx).as_scalar(cx))
                    .dedup()
                    .exactly_one()
                    .ok()
                    .flatten()
                    .filter(|ty| ty.bit_width() >= ptr_addr_bit_width)
                else {
                    return Ok(Transformed::Unchanged);
                };
                let Some(inputs) = data_inst_def
                    .inputs
                    .iter()
                    .map(|&v| {
                        match v {
                            Value::Var(v) => match func.at(v).decl().kind() {
                                VarKind::NodeOutput { node, output_idx: 0 } => {
                                    let def = func.at(node).def();
                                    match &def.kind {
                                        DataInstKind::SpvInst(spv_inst, lowering)
                                            if spv_inst.opcode == wk.OpConvertPtrToU
                                                && lowering.disaggregated_output.is_none()
                                                && lowering.disaggregated_inputs.is_empty() =>
                                        {
                                            assert_eq!(def.inputs.len(), 1);
                                            Some(def.inputs[0])
                                        }
                                        _ => None,
                                    }
                                }
                                _ => None,
                            },
                            Value::Const(ct) => {
                                let ct = ct.as_scalar(cx)?;
                                assert!(ct.ty() == int_scalar_ty);

                                // FIXME(eddyb) support arbitrary constant address ptrs.
                                match ct.int_as_u32()? {
                                    0 if self
                                        .lowerer
                                        .layout_cache
                                        .config
                                        .logical_ptr_null_is_zero =>
                                    {
                                        Some(Value::Const(cx.intern(ConstDef {
                                            attrs: Default::default(),
                                            ty: self.lowerer.qptr_type(),
                                            // FIXME(eddyb) maybe `qptr` should
                                            // have its own null constant?
                                            kind: ConstKind::SpvInst {
                                                spv_inst_and_const_inputs: Rc::new((
                                                    wk.OpConstantNull.into(),
                                                    [].into_iter().collect(),
                                                )),
                                            },
                                        })))
                                    }
                                    _ => None,
                                }
                            }
                        }
                    })
                    .collect()
                else {
                    return Ok(Transformed::Unchanged);
                };

                // FIXME(eddyb) these are only supported since
                // SPIR-V 1.4, and also `qptr` should maybe have
                // its own comparison ops (esp. if they might
                // require emulation).
                let spv_ptr_cmp_opcode = match cmp_op {
                    scalar::IntBinOp::Eq => wk.OpPtrEqual,
                    scalar::IntBinOp::Ne => wk.OpPtrNotEqual,
                    _ => unreachable!(),
                };

                return Ok(Transformed::Changed(DataInstDef {
                    attrs,
                    kind: DataInstKind::SpvInst(
                        spv_ptr_cmp_opcode.into(),
                        spv::InstLowering::default(),
                    ),
                    inputs,
                    child_regions: [].into_iter().collect(),
                    outputs: data_inst_def.outputs.clone(),
                }));
            }

            _ => return Ok(Transformed::Unchanged),
        };

        // FIXME(eddyb) wasteful clone? (needed due to borrowing issues)
        let outputs = data_inst_def.outputs.clone();

        // HACK(eddyb) this is for easy bailing/asserting.
        let disaggregated_output_or_inputs_during_lowering =
            spv_inst_lowering.disaggregated_output.is_some()
                || !spv_inst_lowering.disaggregated_inputs.is_empty();

        // Flatten `QPtrOp::Offset`s behind `ptr` into a base pointer and offset.
        let flatten_offsets = |mut ptr| {
            let mut offset = 0;
            loop {
                (ptr, offset) = if let Value::Var(ptr) = ptr
                    && let VarKind::NodeOutput { node: ptr_inst, output_idx: 0 } =
                        func.at(ptr).decl().kind()
                    && let NodeDef {
                        kind: DataInstKind::QPtr(QPtrOp::Offset(ptr_offset)),
                        inputs,
                        ..
                    } = func.at(ptr_inst).def()
                    && let Some(new_offset) = ptr_offset.checked_add(offset)
                {
                    (inputs[0], new_offset)
                } else {
                    break;
                };
            }
            (ptr, offset)
        };

        // NOTE(eddyb) the ordering of some checks below is not purely aesthetic,
        // if the types are invalid there could e.g. be disaggregation where it
        // should never otherwise appear, so type checks should precede them.

        let replacement_kind_and_inputs = if spv_inst.opcode == wk.OpVariable {
            // HACK(eddyb) only needed because of potentially invalid SPIR-V.
            let output_type = spv_inst_lowering
                .disaggregated_output
                .unwrap_or_else(|| func.at(outputs[0]).decl().ty);
            let (_, var_data_type) =
                self.lowerer.as_spv_ptr_type(output_type).ok_or_else(|| {
                    LowerError(Diag::bug(["output type not an `OpTypePointer`".into()]))
                })?;

            assert!(spv_inst_lowering.disaggregated_output.is_none());

            // FIXME(eddyb) this can be happen due to the optional initializer.
            // FIXME(eddyb) lower the initializer to store(s) just after variables.
            if !spv_inst_lowering.disaggregated_inputs.is_empty() {
                return Ok(Transformed::Unchanged);
            }

            assert_eq!(outputs.len(), 1);
            assert!(data_inst_def.inputs.len() <= 1);

            match self.lowerer.layout_of(var_data_type)? {
                TypeLayout::Concrete(concrete) if concrete.mem_layout.dyn_unit_stride.is_none() => {
                    (
                        MemOp::FuncLocalVar(concrete.mem_layout.fixed_base).into(),
                        data_inst_def.inputs.clone(),
                    )
                }
                _ => return Ok(Transformed::Unchanged),
            }
        } else if spv_inst.opcode == wk.OpArrayLength {
            if disaggregated_output_or_inputs_during_lowering {
                return Err(LowerError(Diag::bug([format!(
                    "unexpected aggregate types in `{}`",
                    spv_inst.opcode.name()
                )
                .into()])));
            }

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
            if disaggregated_output_or_inputs_during_lowering {
                return Err(LowerError(Diag::bug([format!(
                    "unexpected aggregate types in `{}`",
                    spv_inst.opcode.name()
                )
                .into()])));
            }

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
                    self.lowerer.layout_of(cx.intern(
                        spv::Inst::from(wk.OpTypeRuntimeArray).into_canonical_type_with(
                            cx,
                            [TypeOrConst::Type(base_pointee_type)].into_iter().collect(),
                        ),
                    ))?
                } else {
                    self.lowerer.layout_of(base_pointee_type)?
                };

            let mut ptr = base_ptr;
            let mut steps =
                self.try_lower_access_chain(access_chain_base_layout, &data_inst_def.inputs[1..])?;

            // Fold a previous `Offset` into an initial offset step, where possible.
            if let Some(QPtrChainStep { op: QPtrOp::Offset(first_offset), dyn_idx: None }) =
                steps.first_mut()
            {
                let (ptr_base_ptr, ptr_offset) = flatten_offsets(ptr);
                if let Some(new_first_offset) = first_offset.checked_add(ptr_offset) {
                    ptr = ptr_base_ptr;
                    *first_offset = new_first_offset;
                }
            }

            // HACK(eddyb) noop cases should probably not use any `DataInst`s at all,
            // but that would require the ability to replace all uses of a `Value`.
            let final_step =
                steps.pop().unwrap_or(QPtrChainStep { op: QPtrOp::Offset(0), dyn_idx: None });

            for step in steps {
                let func = func_at_data_inst.reborrow().at(());

                let (kind, inputs) = step.into_data_inst_kind_and_inputs(ptr);
                let step_data_inst = func.nodes.define(
                    cx,
                    DataInstDef {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        attrs,
                        kind,
                        inputs,
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );
                let step_output_var = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: self.lowerer.qptr_type(),
                        def_parent: Either::Right(step_data_inst),
                        def_idx: 0,
                    },
                );
                func.nodes[step_data_inst].outputs.push(step_output_var);

                // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                // due to the need to borrow `regions` and `nodes`
                // at the same time - perhaps some kind of `FuncAtMut` position
                // types for "where a list is in a parent entity" could be used
                // to make this more ergonomic, although the potential need for
                // an actual list entity of its own, should be considered.
                func.regions[self.parent_region.unwrap()].children.insert_before(
                    step_data_inst,
                    data_inst,
                    func.nodes,
                );

                // HACK(eddyb) account for traversal never seeing this,
                // while still needing value replacement and/or use tracking.
                func.at(step_data_inst).inner_in_place_transform_with(self);

                // HACK(eddyb) this tracking is kind of ad-hoc but should
                // easily cover everything we care about for now.
                self.remove_inst_if_dead_output_with_parent_region
                    .push((step_output_var, self.parent_region.unwrap()));

                ptr = Value::Var(step_output_var);
            }
            final_step.into_data_inst_kind_and_inputs(ptr)
        } else if [wk.OpLoad, wk.OpStore].contains(&spv_inst.opcode) {
            let ptr = data_inst_def.inputs[0];

            // HACK(eddyb) only needed because of potentially invalid SPIR-V.
            let type_of_ptr = match &spv_inst_lowering.disaggregated_inputs[..] {
                [(range, _), ..] if range.start == 0 => None,
                _ => Some(func.at(ptr).type_of(cx)),
            };
            let (_, pointee_type) = type_of_ptr
                .and_then(|type_of_ptr| self.lowerer.as_spv_ptr_type(type_of_ptr))
                .ok_or_else(|| {
                    LowerError(Diag::bug(["pointer input not an `OpTypePointer`".into()]))
                })?;

            #[derive(Copy, Clone)]
            enum Access {
                Load { output: Var },
                Store(Value),
            }

            impl Access {
                fn to_data_inst_def(self, attrs: AttrSet, ptr: Value, offset: i32) -> DataInstDef {
                    let offset = NonZeroI32::new(offset);
                    match self {
                        Access::Load { output } => DataInstDef {
                            attrs,
                            kind: MemOp::Load { offset }.into(),
                            inputs: [ptr].into_iter().collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [output].into_iter().collect(),
                        },
                        Access::Store(value) => DataInstDef {
                            attrs,
                            kind: MemOp::Store { offset }.into(),
                            inputs: [ptr, value].into_iter().collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        },
                    }
                }
            }

            enum Accesses<LLA: Iterator<Item = Access>> {
                Single(Access),
                AggregateLeaves { aggregate_type: Type, leaf_accesses: LLA },
            }

            let accesses = if spv_inst.opcode == wk.OpLoad {
                assert!(spv_inst_lowering.disaggregated_inputs.is_empty());
                assert_eq!(data_inst_def.inputs.len(), 1);

                match spv_inst_lowering.disaggregated_output {
                    None => Accesses::Single(Access::Load { output: outputs[0] }),
                    Some(aggregate_type) => Accesses::AggregateLeaves {
                        aggregate_type,
                        leaf_accesses: Either::Left(
                            outputs.iter().map(|&output| Access::Load { output }),
                        ),
                    },
                }
            } else {
                assert!(spv_inst_lowering.disaggregated_output.is_none());

                match spv_inst_lowering.disaggregated_inputs[..] {
                    [] => {
                        assert_eq!(data_inst_def.inputs.len(), 2);

                        Accesses::Single(Access::Store(data_inst_def.inputs[1]))
                    }
                    [(ref range, aggregate_type)] => {
                        assert_eq!(*range, 1..u32::try_from(data_inst_def.inputs.len()).unwrap());

                        Accesses::AggregateLeaves {
                            aggregate_type,
                            leaf_accesses: Either::Right(
                                data_inst_def.inputs[1..].iter().map(|&v| Access::Store(v)),
                            ),
                        }
                    }
                    _ => unreachable!(),
                }
            };

            let type_of_access = |access| match access {
                Access::Load { output } => func.at(output).decl().ty,
                Access::Store(value) => func.at(value).type_of(cx),
            };

            let original_access_type = match accesses {
                Accesses::Single(access) => type_of_access(access),
                Accesses::AggregateLeaves { aggregate_type, .. } => aggregate_type,
            };

            if pointee_type != original_access_type {
                return Err(LowerError(Diag::bug([
                    "access type different from pointee type".into()
                ])));
            }

            let (ptr, base_offset) = flatten_offsets(ptr);

            // FIXME(eddyb) support memory operands somehow.
            if !spv_inst.imms.is_empty() {
                return Ok(Transformed::Unchanged);
            }

            // FIXME(eddyb) consider skipping `undef` leaf stores (and even
            // treating a non-aggregate `undef` store as a single leaf),
            // but that might be too much of an "implicit optimization" here.
            match accesses {
                Accesses::Single(access) => {
                    return Ok(Transformed::Changed(access.to_data_inst_def(
                        attrs,
                        ptr,
                        base_offset,
                    )));
                }

                // If this is an aggregate `OpLoad`/`OpStore`, we should generate
                // one instruction per leaf, instead.
                Accesses::AggregateLeaves { aggregate_type: _, mut leaf_accesses } => {
                    // FIXME(eddyb) this may need to automatically generate an
                    // intermediary `QPtrOp::BufferData` when accessing buffers.
                    let mem_data_layout = match self.lowerer.layout_of(pointee_type)? {
                        TypeLayout::Concrete(mem) => mem,
                        _ => {
                            return Err(LowerError(Diag::bug([
                                "by-value aggregate type without memory layout: ".into(),
                                pointee_type.into(),
                            ])));
                        }
                    };

                    // HACK(eddyb) we have to buffer the details of the new
                    // instructions because we're iterating over the original
                    // one, and can't allocate the new `DataInst`s as we go.
                    let mut leaf_accesses_with_offsets = SmallVec::<[_; 4]>::new();
                    mem_data_layout
                        .deeply_flatten_if(
                            base_offset,
                            // Whether `candidate_layout` is an aggregate (to recurse into).
                            &|candidate_layout| {
                                matches!(
                                    &cx[candidate_layout.original_type].kind,
                                    TypeKind::SpvInst {
                                        value_lowering: spv::ValueLowering::Disaggregate(_),
                                        ..
                                    }
                                )
                            },
                            &mut |leaf_offset, leaf| {
                                let leaf_access = leaf_accesses.next().ok_or_else(|| {
                                    LayoutError(Diag::bug([
                                        "`spv::lower` and `mem::layout` disagree \
                                         on aggregate leaves of "
                                            .into(),
                                        pointee_type.into(),
                                    ]))
                                })?;
                                let leaf_type = type_of_access(leaf_access);
                                if leaf_type != leaf.original_type {
                                    return Err(LayoutError(Diag::bug([
                                        "aggregate leaf mismatch: `".into(),
                                        leaf_type.into(),
                                        "` vs `".into(),
                                        leaf.original_type.into(),
                                        "`".into(),
                                    ])));
                                }
                                leaf_accesses_with_offsets.push((leaf_access, leaf_offset));
                                Ok(())
                            },
                        )
                        .map_err(|LayoutError(err)| LowerError(err))?;

                    if leaf_accesses.next().is_some() {
                        return Err(LowerError(Diag::bug([
                            "`spv::lower` and `mem::layout` disagree on aggregate leaves of "
                                .into(),
                            pointee_type.into(),
                        ])));
                    }

                    let mut func = func_at_data_inst.reborrow().at(());

                    // This is the point of no return: we're inserting several
                    // new instructions, and removing the original one entirely.
                    for (leaf_access, leaf_offset) in leaf_accesses_with_offsets {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        let leaf_attrs = attrs;

                        let leaf_data_inst = func.nodes.define(
                            cx,
                            leaf_access.to_data_inst_def(leaf_attrs, ptr, leaf_offset).into(),
                        );

                        // HACK(eddyb) attach any output vars to the new node.
                        for (output_idx, &output_var) in
                            func.nodes[leaf_data_inst].outputs.iter().enumerate()
                        {
                            let output_var_decl = &mut func.vars[output_var];
                            output_var_decl.def_parent = Either::Right(leaf_data_inst);
                            output_var_decl.def_idx = output_idx.try_into().unwrap();
                        }

                        // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                        // due to the need to borrow `regions` and `nodes`
                        // at the same time - perhaps some kind of `FuncAtMut` position
                        // types for "where a list is in a parent entity" could be used
                        // to make this more ergonomic, although the potential need for
                        // an actual list entity of its own, should be considered.
                        func.regions[self.parent_region.unwrap()].children.insert_before(
                            leaf_data_inst,
                            data_inst,
                            func.nodes,
                        );

                        // HACK(eddyb) account for traversal never seeing this,
                        // while still needing value replacement and/or use tracking.
                        func.reborrow().at(leaf_data_inst).inner_in_place_transform_with(self);
                    }

                    func.regions[self.parent_region.unwrap()]
                        .children
                        .remove(data_inst, func.nodes);

                    // HACK(eddyb) no good "tombstone" for the original def.
                    return Ok(Transformed::Changed(DataInstDef {
                        attrs: AttrSet::default(),
                        kind: DataInstKind::SpvInst(wk.OpNop.into(), spv::InstLowering::default()),
                        inputs: [].into_iter().collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }));
                }
            }
        } else if spv_inst.opcode == wk.OpCopyMemorySized {
            if disaggregated_output_or_inputs_during_lowering {
                return Err(LowerError(Diag::bug([format!(
                    "unexpected aggregate types in `{}`",
                    spv_inst.opcode.name()
                )
                .into()])));
            }

            assert_eq!(data_inst_def.inputs.len(), 3);

            let dst_ptr = data_inst_def.inputs[0];
            let src_ptr = data_inst_def.inputs[1];
            let size = data_inst_def.inputs[2];

            // HACK(eddyb) this isn't just a simple `match` to avoid indentation.
            if let Value::Var(size) = size {
                let mut func = func_at_data_inst.reborrow().at(());

                let (count, stride) = func.vars[size]
                    .def_parent
                    .right()
                    .and_then(|size_node| {
                        let size_node_def = &func.nodes[size_node];
                        match (&size_node_def.kind, &size_node_def.inputs[..]) {
                            (
                                NodeKind::Scalar(scalar::Op::IntBinary(scalar::IntBinOp::Mul)),
                                &[Value::Const(stride), count] | &[count, Value::Const(stride)],
                            ) => {
                                Some((count, NonZeroU32::new(stride.as_scalar(cx)?.int_as_u32()?)?))
                            }
                            _ => None,
                        }
                    })
                    .unwrap_or((Value::Var(size), NonZeroU32::new(1).unwrap()));
                let count_ty = func.reborrow().freeze().at(count).type_of(cx);

                let count_scalar_ty = count_ty
                    .as_scalar(cx)
                    .filter(|ty| matches!(ty, scalar::Type::UInt(_)))
                    .ok_or_else(|| {
                        LowerError(Diag::bug([
                            "`OpCopyMemorySized` with non-uint size input type: ".into(),
                            count_ty.into(),
                        ]))
                    })?;

                // Generate a loop like this Rust code (`let`s becoming `Var`s):
                //
                // let mut i = 0;
                // loop {
                //     let i_next = i + 1;
                //     let any_left = i < count;
                //     if any_left {
                //         let dst_elem = (dst as *mut [u8; STRIDE]).add(i);
                //         let src_elem = (src as *const [u8; STRIDE]).add(i);
                //         dst_elem.copy_from(src_elem, 1);
                //     }
                //
                //     if !any_left { break; } // aka do {...} while(any_left)
                //     i = i_next;
                // }

                let loop_initial_inputs =
                    [Value::Const(cx.intern(scalar::Const::from_bits(count_scalar_ty, 0)))];
                let loop_output_vars = [func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: count_ty,
                        def_parent: Either::Right(data_inst),
                        def_idx: 0,
                    },
                )];

                // HACK(eddyb) doing this early to interfere less w/ mut borrows.
                let if_any_left_child_regions = [
                    func.regions.define(cx, RegionDef::default()),
                    func.regions.define(cx, RegionDef::default()),
                ];

                let loop_body = func.regions.define(cx, RegionDef::default());
                let loop_body_def = &mut func.regions[loop_body];

                let index = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: count_ty,
                        def_parent: Either::Left(loop_body),
                        def_idx: loop_body_def.inputs.len().try_into().unwrap(),
                    },
                );
                loop_body_def.inputs.push(index);

                let index_plus_one_inst = func.nodes.define(
                    cx,
                    DataInstDef {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        attrs,
                        kind: scalar::Op::IntBinary(scalar::IntBinOp::Add).into(),
                        inputs: [
                            Value::Var(index),
                            Value::Const(cx.intern(
                                scalar::Const::int_try_from_i128(count_scalar_ty, 1).unwrap(),
                            )),
                        ]
                        .into_iter()
                        .collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );
                let index_plus_one = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: count_ty,
                        def_parent: Either::Right(index_plus_one_inst),
                        def_idx: 0,
                    },
                );
                func.nodes[index_plus_one_inst].outputs.push(index_plus_one);
                loop_body_def.children.insert_last(index_plus_one_inst, func.nodes);

                loop_body_def.outputs.push(Value::Var(index_plus_one));

                let any_left_inst = func.nodes.define(
                    cx,
                    DataInstDef {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        attrs,
                        kind: scalar::Op::IntBinary(scalar::IntBinOp::LtU).into(),
                        inputs: [Value::Var(index), count].into_iter().collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );
                let any_left = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: cx.intern(scalar::Type::Bool),
                        def_parent: Either::Right(any_left_inst),
                        def_idx: 0,
                    },
                );
                func.nodes[any_left_inst].outputs.push(any_left);
                loop_body_def.children.insert_last(any_left_inst, func.nodes);

                let if_any_left_node = func.nodes.define(
                    cx,
                    NodeDef {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        attrs,
                        kind: NodeKind::Select(SelectionKind::BoolCond),
                        inputs: [Value::Var(any_left)].into_iter().collect(),
                        child_regions: if_any_left_child_regions.into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );
                loop_body_def.children.insert_last(if_any_left_node, func.nodes);

                let if_any_left_then_region_def = &mut func.regions[if_any_left_child_regions[0]];

                let [dst_elem_ptr, src_elem_ptr] = [dst_ptr, src_ptr].map(|ptr| {
                    let inst = func.nodes.define(
                        cx,
                        DataInstDef {
                            // FIXME(eddyb) filter attributes into debuginfo and
                            // semantic, and understand the semantic ones.
                            attrs,
                            kind: QPtrOp::DynOffset { stride, index_bounds: None }.into(),
                            inputs: [ptr, Value::Var(index)].into_iter().collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        }
                        .into(),
                    );
                    let output_var = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: self.lowerer.qptr_type(),
                            def_parent: Either::Right(inst),
                            def_idx: 0,
                        },
                    );
                    func.nodes[inst].outputs.push(output_var);
                    if_any_left_then_region_def.children.insert_last(inst, func.nodes);
                    Value::Var(output_var)
                });

                let copy_inst = func.nodes.define(
                    cx,
                    DataInstDef {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        attrs,
                        kind: MemOp::Copy { size: stride }.into(),
                        inputs: [dst_elem_ptr, src_elem_ptr].into_iter().collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );
                if_any_left_then_region_def.children.insert_last(copy_inst, func.nodes);

                return Ok(Transformed::Changed(NodeDef {
                    attrs,
                    kind: NodeKind::Loop { repeat_condition: Value::Var(any_left) },
                    inputs: loop_initial_inputs.into_iter().collect(),
                    child_regions: [loop_body].into_iter().collect(),
                    outputs: loop_output_vars.into_iter().collect(),
                }));
            }

            let Value::Const(size) = size else { unreachable!() };
            let size = size.as_scalar(cx).and_then(|size| size.int_as_u32()).ok_or_else(|| {
                LowerError(Diag::bug([
                    "`OpCopyMemorySized` with constant, but non-`u32` size: ".into(),
                    size.into(),
                ]))
            })?;

            // FIXME(eddyb) remove instruction if `size == 0`?
            let size = NonZeroU32::new(size)
                .ok_or_else(|| LowerError(Diag::bug(["`OpCopyMemorySized` of 0 bytes".into()])))?;

            // FIXME(eddyb) do something similar for `OpCopyMemory` as well?
            (MemOp::Copy { size }.into(), [dst_ptr, src_ptr].into_iter().collect())
        } else if spv_inst.opcode == wk.OpCopyMemory {
            if disaggregated_output_or_inputs_during_lowering {
                return Err(LowerError(Diag::bug([format!(
                    "unexpected aggregate types in `{}`",
                    spv_inst.opcode.name()
                )
                .into()])));
            }

            assert_eq!(data_inst_def.inputs.len(), 2);

            let dst_ptr = data_inst_def.inputs[0];
            let src_ptr = data_inst_def.inputs[1];

            let (_, dst_pointee_type) =
                self.lowerer.as_spv_ptr_type(func.at(dst_ptr).type_of(cx)).ok_or_else(|| {
                    LowerError(Diag::bug([
                        "destination pointer input not an `OpTypePointer`".into()
                    ]))
                })?;
            let (_, src_pointee_type) =
                self.lowerer.as_spv_ptr_type(func.at(src_ptr).type_of(cx)).ok_or_else(|| {
                    LowerError(Diag::bug(["source pointer input not an `OpTypePointer`".into()]))
                })?;

            if dst_pointee_type != src_pointee_type {
                return Err(LowerError(Diag::bug([
                    "copy destination pointee type different from source pointee type".into(),
                ])));
            }

            // FIXME(eddyb) this may need to automatically generate an
            // intermediary `QPtrOp::BufferData` when accessing buffers.
            let mem_data_layout_or_opaque_handle_type =
                match self.lowerer.layout_of(src_pointee_type)? {
                    TypeLayout::Concrete(mem) => Ok(mem),
                    // HACK(eddyb) Rust-GPU generates `OpCopyMemory`s of handles.
                    TypeLayout::Handle(shapes::Handle::Opaque(ty)) => Err(ty),
                    _ => {
                        return Err(LowerError(Diag::bug([
                            "`OpCopyMemory` of data with non-memory type: ".into(),
                            src_pointee_type.into(),
                        ])));
                    }
                };

            let (dst_ptr, dst_base_offset) = flatten_offsets(dst_ptr);
            let (src_ptr, src_base_offset) = flatten_offsets(src_ptr);

            // FIXME(eddyb) support memory operands somehow.
            if !spv_inst.imms.is_empty() {
                return Ok(Transformed::Unchanged);
            }

            // HACK(eddyb) this is speculative, so we just give up if we hit
            // some situation we don't currently support - ideally, there would
            // be an *untyped* `mem.copy`, but that is harder to support overall.
            // HACK(eddyb) this is a `try {...}`-like use of a closure.
            let try_gather_leaf_offsets_and_types = || {
                struct UnsupportedLargeArray;
                let recurse_into_layout = |layout: &MemTypeLayout| {
                    let aggregate_shape = match &cx[layout.original_type].kind {
                        TypeKind::SpvInst {
                            value_lowering: spv::ValueLowering::Disaggregate(aggregate_shape),
                            ..
                        } => aggregate_shape,
                        _ => return Ok(false),
                    };
                    match *aggregate_shape {
                        spv::AggregateShape::Struct { .. } => Ok(true),

                        // HACK(eddyb) 16 leaves allows for a 4x4 matrix, even
                        // when represented as e.g. `[f32; 16]` or `[[f32; 4]; 4]`
                        // (this comparison gets more complex when accounting
                        // for vectors, e.g. `[f32x4; 4]`, which is only 4 leaves),
                        // but ideally most types accepted here will be even
                        // smaller arrays (which could've e.g. been structs).
                        // FIXME(eddyb) larger arrays should lower to loops that
                        // copy a small number of leaves per iteration, or even
                        // some general-purpose `mem.copy`, to avoid generating
                        // amounts of IR that scale with the array length, which
                        // (unlike struct fields) can be arbitrarily large.
                        spv::AggregateShape::Array { total_leaf_count, .. } => {
                            if total_leaf_count <= 16 {
                                Ok(true)
                            } else {
                                Err(UnsupportedLargeArray)
                            }
                        }
                    }
                };

                // HACK(eddyb) buffering the details of the instructions we'll
                // be generating, because we don't know ahead of time whether we
                // even want to expand the `OpCopyMemory`, at all.
                let mut leaf_offsets_and_types = SmallVec::<[_; 8]>::new();
                let mem_data_layout = match mem_data_layout_or_opaque_handle_type {
                    Ok(mem_data_layout) => mem_data_layout,
                    Err(opaque_handle_type) => {
                        leaf_offsets_and_types.push((0, opaque_handle_type));
                        return Some(leaf_offsets_and_types);
                    }
                };
                mem_data_layout
                    .deeply_flatten_if(
                        0,
                        &|candidate_layout| recurse_into_layout(candidate_layout).unwrap_or(false),
                        &mut |leaf_offset, leaf| {
                            // FIMXE(eddyb) ideally this would not be computed twice.
                            recurse_into_layout(leaf).map_err(|UnsupportedLargeArray| {
                                // HACK(eddyb) not an error, just stopping traversal.
                                LayoutError(Diag::bug([]))
                            })?;

                            // HACK(eddyb) `deeply_flatten_if` takes a base offset,
                            // but we have two, so we need our own overflow checks.
                            if dst_base_offset.checked_add(leaf_offset).is_none()
                                || src_base_offset.checked_add(leaf_offset).is_none()
                            {
                                // HACK(eddyb) not an error, just stopping traversal.
                                return Err(LayoutError(Diag::bug([])));
                            }

                            leaf_offsets_and_types.push((leaf_offset, leaf.original_type));

                            Ok(())
                        },
                    )
                    .ok()?;
                Some(leaf_offsets_and_types)
            };
            let leaf_offsets_and_types = match try_gather_leaf_offsets_and_types() {
                Some(leaf_offsets_and_types) => leaf_offsets_and_types,
                None => return Ok(Transformed::Unchanged),
            };

            let mut func = func_at_data_inst.reborrow().at(());

            // This is the point of no return: we're inserting several
            // new instructions, and removing the original one entirely.
            for (leaf_offset, leaf_type) in leaf_offsets_and_types {
                let leaf_load_data_inst = func.nodes.define(
                    cx,
                    DataInstDef {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        attrs,
                        kind: MemOp::Load {
                            offset: NonZeroI32::new(
                                src_base_offset.checked_add(leaf_offset).unwrap(),
                            ),
                        }
                        .into(),
                        inputs: [src_ptr].into_iter().collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );
                let leaf_load_output_var = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: leaf_type,
                        def_parent: Either::Right(leaf_load_data_inst),
                        def_idx: 0,
                    },
                );
                func.nodes[leaf_load_data_inst].outputs.push(leaf_load_output_var);

                let leaf_store_data_inst = func.nodes.define(
                    cx,
                    DataInstDef {
                        // FIXME(eddyb) filter attributes into debuginfo and
                        // semantic, and understand the semantic ones.
                        attrs,
                        kind: MemOp::Store {
                            offset: NonZeroI32::new(
                                dst_base_offset.checked_add(leaf_offset).unwrap(),
                            ),
                        }
                        .into(),
                        inputs: [dst_ptr, Value::Var(leaf_load_output_var)].into_iter().collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );

                // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                // due to the need to borrow `regions` and `nodes`
                // at the same time - perhaps some kind of `FuncAtMut` position
                // types for "where a list is in a parent entity" could be used
                // to make this more ergonomic, although the potential need for
                // an actual list entity of its own, should be considered.
                let parent_region_children =
                    &mut func.regions[self.parent_region.unwrap()].children;
                parent_region_children.insert_before(leaf_load_data_inst, data_inst, func.nodes);
                parent_region_children.insert_before(leaf_store_data_inst, data_inst, func.nodes);

                // HACK(eddyb) account for traversal never seeing these,
                // while still needing value replacement and/or use tracking.
                func.reborrow().at(leaf_load_data_inst).inner_in_place_transform_with(self);
                func.reborrow().at(leaf_store_data_inst).inner_in_place_transform_with(self);
            }

            func.regions[self.parent_region.unwrap()].children.remove(data_inst, func.nodes);

            // HACK(eddyb) no good "tombstone" for the original def.
            return Ok(Transformed::Changed(DataInstDef {
                attrs: AttrSet::default(),
                kind: DataInstKind::SpvInst(wk.OpNop.into(), spv::InstLowering::default()),
                inputs: [].into_iter().collect(),
                child_regions: [].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }));
        } else if spv_inst.opcode == wk.OpBitcast {
            if disaggregated_output_or_inputs_during_lowering {
                return Err(LowerError(Diag::bug([format!(
                    "unexpected aggregate types in `{}`",
                    spv_inst.opcode.name()
                )
                .into()])));
            }

            assert_eq!(outputs.len(), 1);
            assert_eq!(data_inst_def.inputs.len(), 1);

            let input = data_inst_def.inputs[0];
            // Pointer-to-pointer casts are noops on `qptr`.
            if self.lowerer.as_spv_ptr_type(func.at(input).type_of(cx)).is_some()
                && self.lowerer.as_spv_ptr_type(func.at(outputs[0]).decl().ty).is_some()
            {
                // HACK(eddyb) this will end added to `noop_offsets_to_base_ptr`,
                // which should replace all uses of this bitcast with its input.
                (QPtrOp::Offset(0).into(), data_inst_def.inputs.clone())
            } else {
                return Ok(Transformed::Unchanged);
            }
        } else if spv_inst.opcode == wk.OpConvertPtrToU {
            // HACK(eddyb) this is only here because of potential simplifications
            // (when compared against the integer constant `0`) which aren't
            // currently handled in a more generalized manner anywhere else.
            self.remove_inst_if_dead_output_with_parent_region
                .push((outputs[0], self.parent_region.unwrap()));
            return Ok(Transformed::Unchanged);
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
        let wk = self.lowerer.wk;
        let cx = &self.lowerer.cx;

        let func_at_data_inst_frozen = func_at_data_inst.reborrow().freeze();
        let data_inst_def = func_at_data_inst_frozen.def();

        // FIXME(eddyb) is this a good convention?
        let func = func_at_data_inst_frozen.at(());

        let spv_inst_lowering = match &data_inst_def.kind {
            // Known semantics, no need to preserve SPIR-V pointer information.
            NodeKind::Select(_)
            | NodeKind::Loop { .. }
            | NodeKind::ExitInvocation(_)
            | DataInstKind::Scalar(_)
            | DataInstKind::Vector(_)
            | DataInstKind::FuncCall(_)
            | DataInstKind::Mem(_)
            | DataInstKind::QPtr(_)
            | DataInstKind::ThunkBind(_) => return,

            DataInstKind::SpvInst(_, lowering) | DataInstKind::SpvExtInst { lowering, .. } => {
                lowering
            }
        };

        let mut old_and_new_attrs = None;
        let get_old_attrs = || AttrSetDef { attrs: cx[data_inst_def.attrs].attrs.clone() };

        if let Some(LowerError(e)) = extra_error {
            old_and_new_attrs.get_or_insert_with(get_old_attrs).push_diag(e);
        }

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
        for (output_idx, &output_var) in data_inst_def.outputs.iter().enumerate() {
            if let Some((addr_space, pointee)) =
                self.lowerer.as_spv_ptr_type(func.at(output_var).decl().ty)
            {
                // FIXME(eddyb) make this impossible by lowering all instructions
                // that may produce aggregates with pointer leaves.
                if output_idx != 0 || spv_inst_lowering.disaggregated_output.is_some() {
                    old_and_new_attrs.get_or_insert_with(get_old_attrs).push_diag(Diag::bug([
                        format!("unsupported pointer as aggregate leaf (output #{output_idx})")
                            .into(),
                    ]));
                    continue;
                }

                // HACK(eddyb) avoid otherwise-unsupported instructions ending up
                // with invalid address spaces (such cases should be errors,
                // but Rust-GPU still emits `Generic` everywhere, and having
                // def-vs-use type mismatches, instead, would also cause issues).
                let addr_space = match &data_inst_def.kind {
                    DataInstKind::SpvInst(spv_inst, _) => {
                        if spv_inst.opcode == wk.OpVariable {
                            AddrSpace::SpvStorageClass(wk.Function)
                        } else if spv_inst.opcode == wk.OpImageTexelPointer {
                            AddrSpace::SpvStorageClass(wk.Image)
                        } else {
                            addr_space
                        }
                    }
                    _ => addr_space,
                };

                old_and_new_attrs.get_or_insert_with(get_old_attrs).attrs.insert(
                    QPtrAttr::FromSpvPtrOutput {
                        addr_space: OrdAssertEq(addr_space),
                        pointee: OrdAssertEq(pointee),
                    }
                    .into(),
                );
            }
        }

        if let Some(attrs) = old_and_new_attrs {
            func_at_data_inst.def().attrs = cx.intern(attrs);
        }
    }

    // FIXME(eddyb) these are only this whacky because an `u32` is being
    // encoded as `Option<NonZeroU32>` for (dense) map entry reasons.
    fn add_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Value::Var(v) = v {
                let count = self.var_use_counts.entry(v);
                *count = Some(
                    NonZeroU32::new(count.map_or(0, |c| c.get()).checked_add(1).unwrap()).unwrap(),
                );
            }
        }
    }
    fn remove_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Value::Var(v) = v {
                let count = self.var_use_counts.entry(v);
                *count = NonZeroU32::new(count.unwrap().get() - 1);
            }
        }
    }

    // HACK(eddyb) this is a helper *only* for `transform_value_use` and
    // `in_place_transform_node_def`, and should not be used elsewhere.
    fn apply_value_replacements(&self, mut value: Value) -> Value {
        while let Value::Var(var) = value {
            value = if let Some(&base_ptr) = self.noop_offsets_to_base_ptr.get(&var) {
                base_ptr
            } else {
                break;
            };
        }
        value
    }
}

impl Transformer for LowerFromSpvPtrInstsInFunc<'_> {
    // NOTE(eddyb) it's important that this only gets invoked on already lowered
    // `Value`s, so we can rely on e.g. `noop_offsets_to_base_ptr` being filled.
    fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
        let new_v = self.apply_value_replacements(*v);

        self.add_value_uses(&[new_v]);

        if *v == new_v { Transformed::Unchanged } else { Transformed::Changed(new_v) }
    }

    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_in_place_transform_with(self);
        self.parent_region = outer_region;
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        match self.try_lower_data_inst_def(func_at_node.reborrow()) {
            Ok(Transformed::Changed(new_def)) => {
                // HACK(eddyb) this tracking is kind of ad-hoc but should
                // easily cover everything we care about for now.
                if let DataInstKind::QPtr(
                    op @ (QPtrOp::HandleArrayIndex
                    | QPtrOp::BufferData
                    | QPtrOp::BufferDynLen { .. }
                    | QPtrOp::Offset(_)
                    | QPtrOp::DynOffset { .. }),
                ) = &new_def.kind
                {
                    self.remove_inst_if_dead_output_with_parent_region.push((
                        func_at_node.reborrow().def().outputs[0],
                        self.parent_region.unwrap(),
                    ));

                    if let QPtrOp::Offset(0) = op {
                        let base_ptr = self.apply_value_replacements(new_def.inputs[0]);
                        self.noop_offsets_to_base_ptr
                            .insert(func_at_node.reborrow().def().outputs[0], base_ptr);
                    }
                }

                *func_at_node.reborrow().def() = new_def;
            }
            result @ (Ok(Transformed::Unchanged) | Err(_)) => {
                self.add_fallback_attrs_to_data_inst_def(func_at_node.reborrow(), result.err());
            }
        }

        // NOTE(eddyb) this is done last so that `transform_value_use` only sees
        // the lowered `Value`s, not the original ones.
        func_at_node.inner_in_place_transform_with(self);
    }

    fn in_place_transform_func_decl(&mut self, func_decl: &mut FuncDecl) {
        // HACK(eddyb) separately pre-process all `OpVariable`s with initializers,
        // as the `OpStore`s needed to initialize them, have to be injected
        // *after* the last `OpVariable`
        if let DeclDef::Present(func_def_body) = &mut func_decl.def {
            let last_func_local_var = func_def_body
                .at_body()
                .at_children()
                .into_iter()
                .take_while(|func_at_node| match &func_at_node.def().kind {
                    DataInstKind::SpvInst(spv_inst, _) => {
                        spv_inst.opcode == self.lowerer.wk.OpVariable
                    }
                    _ => false,
                })
                .map(|func_at_node| func_at_node.position)
                .last();

            // FIXME(eddyb) a cursor abstraction would be clearer.
            let mut insert_after = last_func_local_var;

            let body = func_def_body.body;
            let mut func_at_body_children = func_def_body.at_mut_body().at_children().into_iter();
            while let Some(func_at_node) = func_at_body_children.next() {
                let node = func_at_node.position;
                let func = func_at_node.at(());

                let node_def = &mut *func.nodes[node];
                let spv_inst_lowering = match &mut node_def.kind {
                    DataInstKind::SpvInst(spv_inst, lowering)
                        if spv_inst.opcode == self.lowerer.wk.OpVariable =>
                    {
                        lowering
                    }
                    _ => break,
                };

                let Some(local_var_ptr) = (spv_inst_lowering.disaggregated_output.is_none())
                    .then(|| {
                        assert!(node_def.outputs.len() == 1);
                        node_def.outputs[0]
                    })
                    .filter(|&output_var| {
                        self.lowerer.as_spv_ptr_type(func.vars[output_var].ty).is_some()
                    })
                    .map(Value::Var)
                else {
                    continue;
                };

                // FIXME(eddyb) filter attributes into debuginfo and
                // semantic, and understand the semantic ones.
                let init_attrs = node_def.attrs;
                let mut init_inputs = mem::take(&mut node_def.inputs);
                let mut init_input_lowering =
                    mem::take(&mut spv_inst_lowering.disaggregated_inputs);

                init_inputs.insert(0, local_var_ptr);
                match &mut init_input_lowering[..] {
                    [] => {
                        if init_inputs.len() == 1 {
                            continue;
                        }
                    }
                    [(old_range, _)] => {
                        let new_range = 1..u32::try_from(init_inputs.len()).unwrap();
                        assert_eq!(*old_range, 0..(new_range.end - 1));
                        *old_range = new_range;
                    }
                    _ => unreachable!(),
                }

                let store_inst = func.nodes.define(
                    &self.lowerer.cx,
                    DataInstDef {
                        attrs: init_attrs,
                        kind: DataInstKind::SpvInst(
                            self.lowerer.wk.OpStore.into(),
                            spv::InstLowering {
                                disaggregated_output: None,
                                disaggregated_inputs: init_input_lowering,
                            },
                        ),
                        inputs: init_inputs,
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );

                // FIXME(eddyb) a cursor abstraction would be clearer.
                let insert_after = insert_after.as_mut().unwrap();
                func.regions[body].children.insert_after(store_inst, *insert_after, func.nodes);
                *insert_after = store_inst;
            }
        }

        func_decl.inner_in_place_transform_with(self);

        // Apply all `remove_inst_if_dead_output_with_parent_region` removals, that are truly unused.
        if let DeclDef::Present(func_def_body) = &mut func_decl.def {
            let remove_inst_if_dead_output_with_parent_region =
                mem::take(&mut self.remove_inst_if_dead_output_with_parent_region);
            // NOTE(eddyb) reverse order is important, as each removal can reduce
            // use counts of an earlier definition, allowing further removal.
            for (output_var, parent_region) in
                remove_inst_if_dead_output_with_parent_region.into_iter().rev()
            {
                let is_used = self.var_use_counts.get(output_var).is_some();
                if !is_used {
                    let inst = func_def_body.at(output_var).decl().def_parent.right().unwrap();

                    // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                    // due to the need to borrow `regions` and `nodes`
                    // at the same time - perhaps some kind of `FuncAtMut` position
                    // types for "where a list is in a parent entity" could be used
                    // to make this more ergonomic, although the potential need for
                    // an actual list entity of its own, should be considered.
                    func_def_body.regions[parent_region]
                        .children
                        .remove(inst, &mut func_def_body.nodes);

                    self.remove_value_uses(&func_def_body.at(inst).def().inputs);
                }
            }
        }
    }
}
