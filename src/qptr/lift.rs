//! [`QPtr`](crate::TypeKind::QPtr) lifting (e.g. to SPIR-V).

use crate::func_at::FuncAtMut;
use crate::mem::{DataHapp, DataHappKind, MemAccesses, MemAttr, MemOp, shapes};
use crate::qptr::{QPtrAttr, QPtrOp};
use crate::transform::{InnerInPlaceTransform, InnerTransform, Transformed, Transformer};
use crate::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstKind, DeclDef, Diag, DiagLevel, EntityDefs, EntityOrientedDenseMap, Func,
    FuncDecl, FxIndexMap, GlobalVar, GlobalVarDecl, Module, Node, NodeKind, NodeOutputDecl, Region,
    Type, TypeDef, TypeKind, TypeOrConst, Value, VarKind, spv,
};
use smallvec::SmallVec;
use std::cell::Cell;
use std::mem;
use std::num::NonZeroU32;
use std::rc::Rc;

// HACK(eddyb) sharing layout code with other modules.
// FIXME(eddyb) can this just be a non-glob import?
use crate::mem::layout::*;

struct LiftError(Diag);

/// Context for lifting `QPtr`s to SPIR-V `OpTypePointer`s.
///
/// See also `passes::qptr::lift_to_spv_ptrs` (which drives this).
pub struct LiftToSpvPtrs<'a> {
    cx: Rc<Context>,
    wk: &'static spv::spec::WellKnown,
    layout_cache: LayoutCache<'a>,

    cached_u32_type: Cell<Option<Type>>,
}

impl<'a> LiftToSpvPtrs<'a> {
    pub fn new(cx: Rc<Context>, layout_config: &'a LayoutConfig) -> Self {
        Self {
            cx: cx.clone(),
            wk: &spv::spec::Spec::get().well_known,
            layout_cache: LayoutCache::new(cx, layout_config),
            cached_u32_type: Default::default(),
        }
    }

    pub fn lift_global_var(&self, global_var_decl: &mut GlobalVarDecl) {
        match self.spv_ptr_type_and_addr_space_for_global_var(global_var_decl) {
            Ok((spv_ptr_type, addr_space)) => {
                global_var_decl.attrs = self.strip_mem_accesses_attr(global_var_decl.attrs);
                global_var_decl.type_of_ptr_to = spv_ptr_type;
                global_var_decl.addr_space = addr_space;
                global_var_decl.shape = None;
            }
            Err(LiftError(e)) => {
                global_var_decl.attrs.push_diag(&self.cx, e);
            }
        }
        // FIXME(eddyb) if globals have initializers pointing at other globals,
        // here is where they might get fixed up, but that usage is illegal so
        // likely needs to get legalized on `qptr`s, before here.
    }

    pub fn lift_all_funcs(&self, module: &mut Module, funcs: impl IntoIterator<Item = Func>) {
        for func in funcs {
            LiftToSpvPtrInstsInFunc {
                lifter: self,
                global_vars: &module.global_vars,

                parent_region: None,

                deferred_ptr_noops: Default::default(),
                data_inst_use_counts: Default::default(),

                func_has_mem_analysis_bug_diags: false,
            }
            .in_place_transform_func_decl(&mut module.funcs[func]);
        }
    }

    fn find_mem_accesses_attr(&self, attrs: AttrSet) -> Result<&MemAccesses, LiftError> {
        self.cx[attrs]
            .attrs
            .iter()
            .find_map(|attr| match attr {
                Attr::Mem(MemAttr::Accesses(accesses)) => Some(&accesses.0),
                _ => None,
            })
            .ok_or_else(|| LiftError(Diag::bug(["missing `mem.accesses` attribute".into()])))
    }

    fn strip_mem_accesses_attr(&self, attrs: AttrSet) -> AttrSet {
        self.cx.intern(AttrSetDef {
            attrs: self.cx[attrs]
                .attrs
                .iter()
                .filter(|attr| !matches!(attr, Attr::Mem(MemAttr::Accesses(_))))
                .cloned()
                .collect(),
        })
    }

    fn spv_ptr_type_and_addr_space_for_global_var(
        &self,
        global_var_decl: &GlobalVarDecl,
    ) -> Result<(Type, AddrSpace), LiftError> {
        let wk = self.wk;

        let mem_accesses = self.find_mem_accesses_attr(global_var_decl.attrs)?;

        let shape =
            global_var_decl.shape.ok_or_else(|| LiftError(Diag::bug(["missing shape".into()])))?;
        let (storage_class, pointee_type) = match (global_var_decl.addr_space, shape) {
            (AddrSpace::Handles, shapes::GlobalVarShape::Handles { handle, fixed_count }) => {
                let (storage_class, handle_type) = match handle {
                    shapes::Handle::Opaque(ty) => {
                        if self.pointee_type_for_accesses(mem_accesses)? != ty {
                            return Err(LiftError(Diag::bug([
                                "mismatched opaque handle types in `mem.accesses` vs `shape`"
                                    .into(),
                            ])));
                        }
                        (wk.UniformConstant, ty)
                    }
                    // FIXME(eddyb) validate accesses against `buf` and/or expand
                    // the type to make sure it has the right size.
                    shapes::Handle::Buffer(AddrSpace::SpvStorageClass(storage_class), _buf) => {
                        (storage_class, self.pointee_type_for_accesses(mem_accesses)?)
                    }
                    shapes::Handle::Buffer(AddrSpace::Handles, _) => {
                        return Err(LiftError(Diag::bug([
                            "invalid `AddrSpace::Handles` in `Handle::Buffer`".into(),
                        ])));
                    }
                };
                (
                    storage_class,
                    if fixed_count == Some(NonZeroU32::new(1).unwrap()) {
                        handle_type
                    } else {
                        self.spv_op_type_array(handle_type, fixed_count.map(|c| c.get()), None)?
                    },
                )
            }
            // FIXME(eddyb) validate accesses against `layout` and/or expand
            // the type to make sure it has the right size.
            (
                AddrSpace::SpvStorageClass(storage_class),
                shapes::GlobalVarShape::UntypedData(_layout),
            ) => (storage_class, self.pointee_type_for_accesses(mem_accesses)?),
            (
                AddrSpace::SpvStorageClass(storage_class),
                shapes::GlobalVarShape::TypedInterface(ty),
            ) => (storage_class, ty),

            (
                AddrSpace::Handles,
                shapes::GlobalVarShape::UntypedData(_) | shapes::GlobalVarShape::TypedInterface(_),
            )
            | (AddrSpace::SpvStorageClass(_), shapes::GlobalVarShape::Handles { .. }) => {
                return Err(LiftError(Diag::bug(["mismatched `addr_space` and `shape`".into()])));
            }
        };
        let addr_space = AddrSpace::SpvStorageClass(storage_class);
        Ok((self.spv_ptr_type(addr_space, pointee_type), addr_space))
    }

    /// Returns `Some` iff `ty` is a SPIR-V `OpTypePointer`.
    //
    // FIXME(eddyb) deduplicate with `qptr::lower`.
    fn as_spv_ptr_type(&self, ty: Type) -> Option<(AddrSpace, Type)> {
        match &self.cx[ty].kind {
            TypeKind::SpvInst { spv_inst, type_and_const_inputs }
                if spv_inst.opcode == self.wk.OpTypePointer =>
            {
                let sc = match spv_inst.imms[..] {
                    [spv::Imm::Short(_, sc)] => sc,
                    _ => unreachable!(),
                };
                let pointee = match type_and_const_inputs[..] {
                    [TypeOrConst::Type(elem_type)] => elem_type,
                    _ => unreachable!(),
                };
                Some((AddrSpace::SpvStorageClass(sc), pointee))
            }
            _ => None,
        }
    }

    fn spv_ptr_type(&self, addr_space: AddrSpace, pointee_type: Type) -> Type {
        let wk = self.wk;

        let storage_class = match addr_space {
            AddrSpace::Handles => unreachable!(),
            AddrSpace::SpvStorageClass(storage_class) => storage_class,
        };
        self.cx.intern(TypeKind::SpvInst {
            spv_inst: spv::Inst {
                opcode: wk.OpTypePointer,
                imms: [spv::Imm::Short(wk.StorageClass, storage_class)].into_iter().collect(),
            },
            type_and_const_inputs: [TypeOrConst::Type(pointee_type)].into_iter().collect(),
        })
    }

    fn pointee_type_for_accesses(&self, accesses: &MemAccesses) -> Result<Type, LiftError> {
        let wk = self.wk;

        match accesses {
            &MemAccesses::Handles(shapes::Handle::Opaque(ty)) => Ok(ty),
            MemAccesses::Handles(shapes::Handle::Buffer(_, data_happ)) => {
                let attr_spv_decorate_block = Attr::SpvAnnotation(spv::Inst {
                    opcode: wk.OpDecorate,
                    imms: [spv::Imm::Short(wk.Decoration, wk.Block)].into_iter().collect(),
                });
                match &data_happ.kind {
                    DataHappKind::Dead => self.spv_op_type_struct([], [attr_spv_decorate_block]),
                    DataHappKind::Disjoint(fields) => self.spv_op_type_struct(
                        fields.iter().map(|(&field_offset, field_happ)| {
                            Ok((field_offset, self.pointee_type_for_data_happ(field_happ)?))
                        }),
                        [attr_spv_decorate_block],
                    ),
                    DataHappKind::StrictlyTyped(_)
                    | DataHappKind::Direct(_)
                    | DataHappKind::Repeated { .. } => self.spv_op_type_struct(
                        [Ok((0, self.pointee_type_for_data_happ(data_happ)?))],
                        [attr_spv_decorate_block],
                    ),
                }
            }
            MemAccesses::Data(happ) => self.pointee_type_for_data_happ(happ),
        }
    }

    fn pointee_type_for_data_happ(&self, happ: &DataHapp) -> Result<Type, LiftError> {
        match &happ.kind {
            DataHappKind::Dead => self.spv_op_type_struct([], []),
            &DataHappKind::StrictlyTyped(ty) | &DataHappKind::Direct(ty) => Ok(ty),
            DataHappKind::Disjoint(fields) => self.spv_op_type_struct(
                fields.iter().map(|(&field_offset, field_happ)| {
                    Ok((field_offset, self.pointee_type_for_data_happ(field_happ)?))
                }),
                [],
            ),
            DataHappKind::Repeated { element, stride } => {
                let element_type = self.pointee_type_for_data_happ(element)?;

                let fixed_len = happ
                    .max_size
                    .map(|size| {
                        if !size.is_multiple_of(stride.get()) {
                            return Err(LiftError(Diag::bug([format!(
                                "Repeated: size ({size}) not a multiple of stride ({stride})"
                            )
                            .into()])));
                        }
                        Ok(size / stride.get())
                    })
                    .transpose()?;

                self.spv_op_type_array(element_type, fixed_len, Some(*stride))
            }
        }
    }

    fn spv_op_type_array(
        &self,
        element_type: Type,
        fixed_len: Option<u32>,
        stride: Option<NonZeroU32>,
    ) -> Result<Type, LiftError> {
        let wk = self.wk;

        let stride_attrs = stride.map(|stride| {
            self.cx.intern(AttrSetDef {
                attrs: [Attr::SpvAnnotation(spv::Inst {
                    opcode: wk.OpDecorate,
                    imms: [
                        spv::Imm::Short(wk.Decoration, wk.ArrayStride),
                        spv::Imm::Short(wk.LiteralInteger, stride.get()),
                    ]
                    .into_iter()
                    .collect(),
                })]
                .into(),
            })
        });

        let spv_opcode = if fixed_len.is_some() { wk.OpTypeArray } else { wk.OpTypeRuntimeArray };

        Ok(self.cx.intern(TypeDef {
            attrs: stride_attrs.unwrap_or_default(),
            kind: TypeKind::SpvInst {
                spv_inst: spv_opcode.into(),
                type_and_const_inputs: [TypeOrConst::Type(element_type)]
                    .into_iter()
                    .chain(fixed_len.map(|len| TypeOrConst::Const(self.const_u32(len))))
                    .collect(),
            },
        }))
    }

    fn spv_op_type_struct(
        &self,
        field_offsets_and_types: impl IntoIterator<Item = Result<(u32, Type), LiftError>>,
        extra_attrs: impl IntoIterator<Item = Attr>,
    ) -> Result<Type, LiftError> {
        let wk = self.wk;

        let field_offsets_and_types = field_offsets_and_types.into_iter();
        let mut attrs = AttrSetDef::default();
        let mut type_and_const_inputs =
            SmallVec::with_capacity(field_offsets_and_types.size_hint().0);
        for (i, field_offset_and_type) in field_offsets_and_types.enumerate() {
            let (offset, field_type) = field_offset_and_type?;
            attrs.attrs.insert(Attr::SpvAnnotation(spv::Inst {
                opcode: wk.OpMemberDecorate,
                imms: [
                    spv::Imm::Short(wk.LiteralInteger, i.try_into().unwrap()),
                    spv::Imm::Short(wk.Decoration, wk.Offset),
                    spv::Imm::Short(wk.LiteralInteger, offset),
                ]
                .into_iter()
                .collect(),
            }));
            type_and_const_inputs.push(TypeOrConst::Type(field_type));
        }
        attrs.attrs.extend(extra_attrs);
        Ok(self.cx.intern(TypeDef {
            attrs: self.cx.intern(attrs),
            kind: TypeKind::SpvInst { spv_inst: wk.OpTypeStruct.into(), type_and_const_inputs },
        }))
    }

    /// Get the (likely cached) `u32` type.
    fn u32_type(&self) -> Type {
        if let Some(cached) = self.cached_u32_type.get() {
            return cached;
        }
        let wk = self.wk;
        let ty = self.cx.intern(TypeKind::SpvInst {
            spv_inst: spv::Inst {
                opcode: wk.OpTypeInt,
                imms: [
                    spv::Imm::Short(wk.LiteralInteger, 32),
                    spv::Imm::Short(wk.LiteralInteger, 0),
                ]
                .into_iter()
                .collect(),
            },
            type_and_const_inputs: [].into_iter().collect(),
        });
        self.cached_u32_type.set(Some(ty));
        ty
    }

    fn const_u32(&self, x: u32) -> Const {
        let wk = self.wk;

        self.cx.intern(ConstDef {
            attrs: AttrSet::default(),
            ty: self.u32_type(),
            kind: ConstKind::SpvInst {
                spv_inst_and_const_inputs: Rc::new((
                    spv::Inst {
                        opcode: wk.OpConstant,
                        imms: [spv::Imm::Short(wk.LiteralContextDependentNumber, x)]
                            .into_iter()
                            .collect(),
                    },
                    [].into_iter().collect(),
                )),
            },
        })
    }

    /// Attempt to compute a `TypeLayout` for a given (SPIR-V) `Type`.
    fn layout_of(&self, ty: Type) -> Result<TypeLayout, LiftError> {
        self.layout_cache.layout_of(ty).map_err(|LayoutError(err)| LiftError(err))
    }
}

struct LiftToSpvPtrInstsInFunc<'a> {
    lifter: &'a LiftToSpvPtrs<'a>,
    global_vars: &'a EntityDefs<GlobalVar>,

    parent_region: Option<Region>,

    /// Some `QPtr`->`QPtr` `QPtrOp`s must be noops in SPIR-V, but because some
    /// of them have meaningful semantic differences in SPIR-T, replacement of
    /// their uses must be deferred until after `try_lift_data_inst_def` has had
    /// a chance to observe the distinction.
    ///
    /// E.g. `QPtrOp::BufferData`s cannot adjust the SPIR-V pointer type, due to
    /// interactions between the `Block` annotation and any potential trailing
    /// `OpTypeRuntimeArray`s (which cannot be nested in non-`Block` structs).
    ///
    /// The `QPtrOp` itself is only removed after the entire function is lifted,
    /// (using `data_inst_use_counts` to determine whether they're truly unused).
    deferred_ptr_noops: FxIndexMap<DataInst, DeferredPtrNoop>,

    // FIXME(eddyb) consider removing this and just do a full second traversal.
    data_inst_use_counts: EntityOrientedDenseMap<DataInst, NonZeroU32>,

    // HACK(eddyb) this is used to avoid noise when `mem::analyze` failed.
    func_has_mem_analysis_bug_diags: bool,
}

struct DeferredPtrNoop {
    output_pointer: Value,

    output_pointer_addr_space: AddrSpace,

    /// Should be equivalent to `layout_of` on `output_pointer`'s pointee type,
    /// except in the case of `QPtrOp::BufferData`.
    output_pointee_layout: TypeLayout,

    parent_region: Region,
}

impl LiftToSpvPtrInstsInFunc<'_> {
    fn try_lift_data_inst_def(
        &mut self,
        mut func_at_data_inst: FuncAtMut<'_, DataInst>,
    ) -> Result<Transformed<DataInstDef>, LiftError> {
        let wk = self.lifter.wk;
        let cx = &self.lifter.cx;

        let func_at_data_inst_frozen = func_at_data_inst.reborrow().freeze();
        let data_inst = func_at_data_inst_frozen.position;
        let data_inst_def = func_at_data_inst_frozen.def();
        let func = func_at_data_inst_frozen.at(());
        let type_of_val = |v: Value| func.at(v).type_of(cx);
        // FIXME(eddyb) maybe all this data should be packaged up together in a
        // type with fields like those of `DeferredPtrNoop` (or even more).
        let type_of_val_as_spv_ptr_with_layout = |v: Value| {
            if let Value::Var(VarKind::NodeOutput { node: v_data_inst, output_idx: 0 }) = v
                && let Some(ptr_noop) = self.deferred_ptr_noops.get(&v_data_inst)
            {
                return Ok((
                    ptr_noop.output_pointer_addr_space,
                    ptr_noop.output_pointee_layout.clone(),
                ));
            }

            let (addr_space, pointee_type) =
                self.lifter.as_spv_ptr_type(type_of_val(v)).ok_or_else(|| {
                    LiftError(Diag::bug(["pointer input not an `OpTypePointer`".into()]))
                })?;

            Ok((addr_space, self.lifter.layout_of(pointee_type)?))
        };
        let replacement_data_inst_def = match &data_inst_def.kind {
            NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::ExitInvocation(_) => {
                return Ok(Transformed::Unchanged);
            }

            &DataInstKind::FuncCall(_callee) => {
                for &v in &data_inst_def.inputs {
                    if self.lifter.as_spv_ptr_type(type_of_val(v)).is_some() {
                        return Err(LiftError(Diag::bug([
                            "unimplemented calls with pointer args".into(),
                        ])));
                    }
                }
                return Ok(Transformed::Unchanged);
            }

            DataInstKind::Mem(MemOp::FuncLocalVar(_mem_layout)) => {
                let mem_accesses =
                    self.lifter.find_mem_accesses_attr(data_inst_def.outputs[0].attrs)?;

                // FIXME(eddyb) validate against `mem_layout`!
                let pointee_type = self.lifter.pointee_type_for_accesses(mem_accesses)?;

                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.kind = DataInstKind::SpvInst(spv::Inst {
                    opcode: wk.OpVariable,
                    imms: [spv::Imm::Short(wk.StorageClass, wk.Function)].into_iter().collect(),
                });
                data_inst_def.outputs[0].attrs =
                    self.lifter.strip_mem_accesses_attr(data_inst_def.outputs[0].attrs);
                data_inst_def.outputs[0].ty =
                    self.lifter.spv_ptr_type(AddrSpace::SpvStorageClass(wk.Function), pointee_type);
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::HandleArrayIndex) => {
                let (addr_space, layout) =
                    type_of_val_as_spv_ptr_with_layout(data_inst_def.inputs[0])?;
                let handle = match layout {
                    // FIXME(eddyb) standardize variant order in enum/match.
                    TypeLayout::HandleArray(handle, _) => handle,
                    TypeLayout::Handle(_) => {
                        return Err(LiftError(Diag::bug(["cannot index single Handle".into()])));
                    }
                    TypeLayout::Concrete(_) => {
                        return Err(LiftError(Diag::bug(
                            ["cannot index memory as handles".into()],
                        )));
                    }
                };
                let handle_type = match handle {
                    shapes::Handle::Opaque(ty) => ty,
                    shapes::Handle::Buffer(_, buf) => buf.original_type,
                };

                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.kind = DataInstKind::SpvInst(wk.OpAccessChain.into());
                data_inst_def.outputs[0].ty = self.lifter.spv_ptr_type(addr_space, handle_type);
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::BufferData) => {
                let buf_ptr = data_inst_def.inputs[0];
                let (addr_space, buf_layout) = type_of_val_as_spv_ptr_with_layout(buf_ptr)?;

                let buf_data_layout = match buf_layout {
                    TypeLayout::Handle(shapes::Handle::Buffer(_, buf)) => TypeLayout::Concrete(buf),
                    _ => return Err(LiftError(Diag::bug(["non-Buffer pointee".into()]))),
                };

                self.deferred_ptr_noops.insert(
                    data_inst,
                    DeferredPtrNoop {
                        output_pointer: buf_ptr,
                        output_pointer_addr_space: addr_space,
                        output_pointee_layout: buf_data_layout,
                        parent_region: self.parent_region.unwrap(),
                    },
                );

                // FIXME(eddyb) avoid the repeated call to `type_of_val`,
                // maybe don't even replace the `QPtrOp::BufferData` instruction?
                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.outputs[0].ty = type_of_val(buf_ptr);
                data_inst_def
            }
            &DataInstKind::QPtr(QPtrOp::BufferDynLen { fixed_base_size, dyn_unit_stride }) => {
                let buf_ptr = data_inst_def.inputs[0];
                let (_, buf_layout) = type_of_val_as_spv_ptr_with_layout(buf_ptr)?;

                let buf_data_layout = match buf_layout {
                    TypeLayout::Handle(shapes::Handle::Buffer(_, buf)) => buf,
                    _ => return Err(LiftError(Diag::bug(["non-Buffer pointee".into()]))),
                };

                let field_idx = match &buf_data_layout.components {
                    Components::Fields { offsets, layouts }
                        if offsets.last() == Some(&fixed_base_size)
                            && layouts.last().is_some_and(|last_field| {
                                last_field.mem_layout.fixed_base.size == 0
                                    && last_field.mem_layout.dyn_unit_stride
                                        == Some(dyn_unit_stride)
                                    && matches!(
                                        last_field.components,
                                        Components::Elements { fixed_len: None, .. }
                                    )
                            }) =>
                    {
                        u32::try_from(offsets.len() - 1).unwrap()
                    }
                    // FIXME(eddyb) support/diagnose more cases.
                    _ => {
                        return Err(LiftError(Diag::bug([
                            "buffer data type shape mismatch".into()
                        ])));
                    }
                };

                DataInstDef {
                    kind: DataInstKind::SpvInst(spv::Inst {
                        opcode: wk.OpArrayLength,
                        imms: [spv::Imm::Short(wk.LiteralInteger, field_idx)].into_iter().collect(),
                    }),
                    ..data_inst_def.clone()
                }
            }
            &DataInstKind::QPtr(QPtrOp::Offset(offset)) => {
                let base_ptr = data_inst_def.inputs[0];
                let (addr_space, layout) = type_of_val_as_spv_ptr_with_layout(base_ptr)?;
                let mut layout = match layout {
                    TypeLayout::Handle(_) | TypeLayout::HandleArray(..) => {
                        return Err(LiftError(Diag::bug(["cannot offset Handles".into()])));
                    }
                    TypeLayout::Concrete(mem_layout) => mem_layout,
                };
                let mut offset = u32::try_from(offset)
                    .ok()
                    .ok_or_else(|| LiftError(Diag::bug(["negative offset".into()])))?;

                let mut access_chain_inputs: SmallVec<_> = [base_ptr].into_iter().collect();
                // FIXME(eddyb) deduplicate with access chain loop for Load/Store.
                while offset > 0 {
                    let idx = {
                        // HACK(eddyb) supporting ZSTs would be a pain because
                        // they can "fit" in weird ways, e.g. given 3 offsets
                        // A, B, C (before/between/after a pair of fields),
                        // `B..B` is included in both `A..B` and `B..C`.
                        let allow_zst = false;
                        let offset_range = if allow_zst {
                            offset..offset
                        } else {
                            offset..offset.saturating_add(1)
                        };
                        let mut component_indices =
                            layout.components.find_components_containing(offset_range);
                        match (component_indices.next(), component_indices.next()) {
                            (None, _) => {
                                // FIXME(eddyb) this could include the chosen indices,
                                // and maybe the current type and/or layout.
                                return Err(LiftError(Diag::bug([format!(
                                    "offset {offset} not found in type layout, \
                                     after {} access chain indices",
                                    access_chain_inputs.len() - 1
                                )
                                .into()])));
                            }
                            (Some(idx), Some(_)) => {
                                // FIXME(eddyb) !!! this can also be illegal overlap
                                if allow_zst {
                                    return Err(LiftError(Diag::bug([
                                        "ambiguity due to ZSTs in type layout".into(),
                                    ])));
                                }
                                // HACK(eddyb) letting illegal overlap through
                                idx
                            }
                            (Some(idx), None) => idx,
                        }
                    };

                    let idx_as_i32 = i32::try_from(idx).ok().ok_or_else(|| {
                        LiftError(Diag::bug([
                            format!("{idx} not representable as a positive s32").into()
                        ]))
                    })?;
                    access_chain_inputs
                        .push(Value::Const(self.lifter.const_u32(idx_as_i32 as u32)));

                    match &layout.components {
                        Components::Scalar => unreachable!(),
                        Components::Elements { stride, elem, .. } => {
                            offset %= stride.get();
                            layout = elem.clone();
                        }
                        Components::Fields { offsets, layouts } => {
                            offset -= offsets[idx];
                            layout = layouts[idx].clone();
                        }
                    }
                }

                let mut data_inst_def = data_inst_def.clone();
                if access_chain_inputs.len() == 1 {
                    self.deferred_ptr_noops.insert(
                        data_inst,
                        DeferredPtrNoop {
                            output_pointer: base_ptr,
                            output_pointer_addr_space: addr_space,
                            output_pointee_layout: TypeLayout::Concrete(layout),
                            parent_region: self.parent_region.unwrap(),
                        },
                    );

                    // FIXME(eddyb) avoid the repeated call to `type_of_val`,
                    // maybe don't even replace the `QPtrOp::Offset` instruction?
                    data_inst_def.kind = QPtrOp::Offset(0).into();
                    data_inst_def.outputs[0].ty = type_of_val(base_ptr);
                } else {
                    data_inst_def.kind = DataInstKind::SpvInst(wk.OpAccessChain.into());
                    data_inst_def.inputs = access_chain_inputs;
                    data_inst_def.outputs[0].ty =
                        self.lifter.spv_ptr_type(addr_space, layout.original_type);
                }
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::DynOffset { stride, index_bounds }) => {
                let base_ptr = data_inst_def.inputs[0];
                let (addr_space, layout) = type_of_val_as_spv_ptr_with_layout(base_ptr)?;
                let mut layout = match layout {
                    TypeLayout::Handle(_) | TypeLayout::HandleArray(..) => {
                        return Err(LiftError(Diag::bug(["cannot offset Handles".into()])));
                    }
                    TypeLayout::Concrete(mem_layout) => mem_layout,
                };

                let mut access_chain_inputs: SmallVec<_> = [base_ptr].into_iter().collect();
                loop {
                    if let Components::Elements { stride: layout_stride, elem, fixed_len } =
                        &layout.components
                        && layout_stride == stride
                        && Ok(index_bounds.clone())
                            == fixed_len
                                .map(|len| i32::try_from(len.get()).map(|len| 0..len))
                                .transpose()
                    {
                        access_chain_inputs.push(data_inst_def.inputs[1]);
                        layout = elem.clone();
                        break;
                    }

                    // FIXME(eddyb) deduplicate with `maybe_adjust_pointer_for_access`.
                    let idx = {
                        // FIXME(eddyb) there might be a better way to
                        // estimate a relevant offset range for the array,
                        // maybe assume length >= 1 so the minimum range
                        // is always `0..stride`?
                        let min_expected_len = index_bounds
                            .clone()
                            .and_then(|index_bounds| u32::try_from(index_bounds.end).ok())
                            .unwrap_or(0);
                        let offset_range =
                            0..min_expected_len.checked_add(stride.get()).unwrap_or(0);
                        let mut component_indices =
                            layout.components.find_components_containing(offset_range);
                        match (component_indices.next(), component_indices.next()) {
                            (None, _) => {
                                return Err(LiftError(Diag::bug([
                                    "matching array not found in pointee type layout".into(),
                                ])));
                            }
                            // FIXME(eddyb) obsolete this case entirely,
                            // by removing stores of ZSTs, and replacing
                            // loads of ZSTs with `OpUndef` constants.
                            (Some(_), Some(_)) => {
                                return Err(LiftError(Diag::bug([
                                    "ambiguity due to ZSTs in pointee type layout".into(),
                                ])));
                            }
                            (Some(idx), None) => idx,
                        }
                    };

                    let idx_as_i32 = i32::try_from(idx).ok().ok_or_else(|| {
                        LiftError(Diag::bug([
                            format!("{idx} not representable as a positive s32").into()
                        ]))
                    })?;
                    access_chain_inputs
                        .push(Value::Const(self.lifter.const_u32(idx_as_i32 as u32)));

                    layout = match &layout.components {
                        Components::Scalar => unreachable!(),
                        Components::Elements { elem, .. } => elem.clone(),
                        Components::Fields { layouts, .. } => layouts[idx].clone(),
                    };
                }
                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.kind = DataInstKind::SpvInst(wk.OpAccessChain.into());
                data_inst_def.inputs = access_chain_inputs;
                data_inst_def.outputs[0].ty =
                    self.lifter.spv_ptr_type(addr_space, layout.original_type);
                data_inst_def
            }
            DataInstKind::Mem(op @ (MemOp::Load | MemOp::Store)) => {
                // HACK(eddyb) `_` will match multiple variants soon.
                #[allow(clippy::match_wildcard_for_single_variants)]
                let (spv_opcode, access_type) = match op {
                    MemOp::Load => (wk.OpLoad, data_inst_def.outputs[0].ty),
                    MemOp::Store => (wk.OpStore, type_of_val(data_inst_def.inputs[1])),
                    _ => unreachable!(),
                };

                // FIXME(eddyb) written in a more general style for future deduplication.
                let maybe_ajustment = {
                    let input_idx = 0;
                    let ptr = data_inst_def.inputs[input_idx];
                    let (addr_space, pointee_layout) = type_of_val_as_spv_ptr_with_layout(ptr)?;
                    self.maybe_adjust_pointer_for_access(
                        ptr,
                        addr_space,
                        pointee_layout,
                        access_type,
                    )?
                    .map(|access_chain_data_inst_def| (input_idx, access_chain_data_inst_def))
                    .into_iter()
                };

                let mut new_data_inst_def = DataInstDef {
                    kind: DataInstKind::SpvInst(spv_opcode.into()),
                    ..data_inst_def.clone()
                };

                // FIXME(eddyb) written in a more general style for future deduplication.
                for (input_idx, mut access_chain_data_inst_def) in maybe_ajustment {
                    // HACK(eddyb) account for `deferred_ptr_noops` interactions.
                    self.resolve_deferred_ptr_noop_uses(&mut access_chain_data_inst_def.inputs);
                    self.add_value_uses(&access_chain_data_inst_def.inputs);

                    let access_chain_data_inst = func_at_data_inst
                        .reborrow()
                        .nodes
                        .define(cx, access_chain_data_inst_def.into());

                    // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                    // due to the need to borrow `regions` and `nodes`
                    // at the same time - perhaps some kind of `FuncAtMut` position
                    // types for "where a list is in a parent entity" could be used
                    // to make this more ergonomic, although the potential need for
                    // an actual list entity of its own, should be considered.
                    let data_inst = func_at_data_inst.position;
                    let func = func_at_data_inst.reborrow().at(());
                    func.regions[self.parent_region.unwrap()].children.insert_before(
                        access_chain_data_inst,
                        data_inst,
                        func.nodes,
                    );

                    new_data_inst_def.inputs[input_idx] = Value::Var(VarKind::NodeOutput {
                        node: access_chain_data_inst,
                        output_idx: 0,
                    });
                }

                new_data_inst_def
            }

            DataInstKind::SpvInst(_) | DataInstKind::SpvExtInst { .. } => {
                let mut to_spv_ptr_input_adjustments = vec![];
                let mut from_spv_ptr_output = None;
                for attr in &cx[data_inst_def.attrs].attrs {
                    match *attr {
                        Attr::QPtr(QPtrAttr::ToSpvPtrInput {
                            input_idx,
                            pointee: expected_pointee_type,
                        }) => {
                            let input_idx = usize::try_from(input_idx).unwrap();
                            let expected_pointee_type = expected_pointee_type.0;

                            let input_ptr = data_inst_def.inputs[input_idx];
                            let (input_ptr_addr_space, input_pointee_layout) =
                                type_of_val_as_spv_ptr_with_layout(input_ptr)?;

                            if let Some(access_chain_data_inst_def) = self
                                .maybe_adjust_pointer_for_access(
                                    input_ptr,
                                    input_ptr_addr_space,
                                    input_pointee_layout,
                                    expected_pointee_type,
                                )?
                            {
                                to_spv_ptr_input_adjustments
                                    .push((input_idx, access_chain_data_inst_def));
                            }
                        }
                        Attr::QPtr(QPtrAttr::FromSpvPtrOutput { addr_space, pointee }) => {
                            assert!(from_spv_ptr_output.is_none());
                            from_spv_ptr_output = Some((addr_space.0, pointee.0));
                        }
                        _ => {}
                    }
                }

                if to_spv_ptr_input_adjustments.is_empty() && from_spv_ptr_output.is_none() {
                    return Ok(Transformed::Unchanged);
                }

                let mut new_data_inst_def = data_inst_def.clone();

                // FIXME(eddyb) deduplicate with `Load`/`Store`.
                for (input_idx, mut access_chain_data_inst_def) in to_spv_ptr_input_adjustments {
                    // HACK(eddyb) account for `deferred_ptr_noops` interactions.
                    self.resolve_deferred_ptr_noop_uses(&mut access_chain_data_inst_def.inputs);
                    self.add_value_uses(&access_chain_data_inst_def.inputs);

                    let access_chain_data_inst = func_at_data_inst
                        .reborrow()
                        .nodes
                        .define(cx, access_chain_data_inst_def.into());

                    // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                    // due to the need to borrow `regions` and `nodes`
                    // at the same time - perhaps some kind of `FuncAtMut` position
                    // types for "where a list is in a parent entity" could be used
                    // to make this more ergonomic, although the potential need for
                    // an actual list entity of its own, should be considered.
                    let data_inst = func_at_data_inst.position;
                    let func = func_at_data_inst.reborrow().at(());
                    func.regions[self.parent_region.unwrap()].children.insert_before(
                        access_chain_data_inst,
                        data_inst,
                        func.nodes,
                    );

                    new_data_inst_def.inputs[input_idx] = Value::Var(VarKind::NodeOutput {
                        node: access_chain_data_inst,
                        output_idx: 0,
                    });
                }

                if let Some((addr_space, pointee_type)) = from_spv_ptr_output {
                    new_data_inst_def.outputs[0].ty =
                        self.lifter.spv_ptr_type(addr_space, pointee_type);
                }

                new_data_inst_def
            }
        };
        Ok(Transformed::Changed(replacement_data_inst_def))
    }

    /// If necessary, construct an `OpAccessChain` instruction to turn `ptr`
    /// (pointing to a type with `pointee_layout`) into a pointer to `access_type`
    /// (which can then be used with e.g. `OpLoad`/`OpStore`).
    //
    // FIXME(eddyb) customize errors, to tell apart Load/Store/ToSpvPtrInput.
    fn maybe_adjust_pointer_for_access(
        &self,
        ptr: Value,
        addr_space: AddrSpace,
        mut pointee_layout: TypeLayout,
        access_type: Type,
    ) -> Result<Option<DataInstDef>, LiftError> {
        let wk = self.lifter.wk;

        let access_layout = self.lifter.layout_of(access_type)?;

        // The access type might be merely a prefix of the pointee type,
        // requiring injecting an extra `OpAccessChain` to "dig in".
        let mut access_chain_inputs: SmallVec<_> = [ptr].into_iter().collect();

        if let TypeLayout::HandleArray(handle, _) = pointee_layout {
            access_chain_inputs.push(Value::Const(self.lifter.const_u32(0)));
            pointee_layout = TypeLayout::Handle(handle);
        }
        match (pointee_layout, access_layout) {
            (TypeLayout::HandleArray(..), _) => unreachable!(),

            // All the illegal cases are here to keep the rest tidier.
            (_, TypeLayout::Handle(shapes::Handle::Buffer(..))) => {
                return Err(LiftError(Diag::bug(["cannot access whole Buffer".into()])));
            }
            (_, TypeLayout::HandleArray(..)) => {
                return Err(LiftError(Diag::bug(["cannot access whole HandleArray".into()])));
            }
            (_, TypeLayout::Concrete(access_layout))
                if access_layout.mem_layout.dyn_unit_stride.is_some() =>
            {
                return Err(LiftError(Diag::bug(["cannot access unsized type".into()])));
            }
            (TypeLayout::Handle(shapes::Handle::Buffer(..)), _) => {
                return Err(LiftError(Diag::bug(["cannot access into Buffer".into()])));
            }
            (TypeLayout::Handle(_), TypeLayout::Concrete(_)) => {
                return Err(LiftError(Diag::bug(["cannot access Handle as memory".into()])));
            }
            (TypeLayout::Concrete(_), TypeLayout::Handle(_)) => {
                return Err(LiftError(Diag::bug(["cannot access memory as Handle".into()])));
            }

            (
                TypeLayout::Handle(shapes::Handle::Opaque(pointee_handle_type)),
                TypeLayout::Handle(shapes::Handle::Opaque(access_handle_type)),
            ) => {
                if pointee_handle_type != access_handle_type {
                    return Err(LiftError(Diag::bug([
                        "(opaque handle) pointer vs access type mismatch".into(),
                    ])));
                }
            }

            (TypeLayout::Concrete(mut pointee_layout), TypeLayout::Concrete(access_layout)) => {
                // FIXME(eddyb) deduplicate with access chain loop for Offset.
                while pointee_layout.original_type != access_layout.original_type {
                    let idx = {
                        let offset_range = 0..access_layout.mem_layout.fixed_base.size;
                        let mut component_indices =
                            pointee_layout.components.find_components_containing(offset_range);
                        match (component_indices.next(), component_indices.next()) {
                            (None, _) => {
                                return Err(LiftError(Diag::bug([
                                    "accessed type not found in pointee type layout".into(),
                                ])));
                            }
                            // FIXME(eddyb) obsolete this case entirely,
                            // by removing stores of ZSTs, and replacing
                            // loads of ZSTs with `OpUndef` constants.
                            (Some(_), Some(_)) => {
                                return Err(LiftError(Diag::bug([
                                    "ambiguity due to ZSTs in pointee type layout".into(),
                                ])));
                            }
                            (Some(idx), None) => idx,
                        }
                    };

                    let idx_as_i32 = i32::try_from(idx).ok().ok_or_else(|| {
                        LiftError(Diag::bug([
                            format!("{idx} not representable as a positive s32").into()
                        ]))
                    })?;
                    access_chain_inputs
                        .push(Value::Const(self.lifter.const_u32(idx_as_i32 as u32)));

                    pointee_layout = match &pointee_layout.components {
                        Components::Scalar => unreachable!(),
                        Components::Elements { elem, .. } => elem.clone(),
                        Components::Fields { layouts, .. } => layouts[idx].clone(),
                    };
                }
            }
        }

        Ok(if access_chain_inputs.len() > 1 {
            Some(DataInstDef {
                attrs: Default::default(),
                kind: DataInstKind::SpvInst(wk.OpAccessChain.into()),
                inputs: access_chain_inputs,
                child_regions: [].into_iter().collect(),
                outputs: [NodeOutputDecl {
                    attrs: Default::default(),
                    ty: self.lifter.spv_ptr_type(addr_space, access_type),
                }]
                .into_iter()
                .collect(),
            })
        } else {
            None
        })
    }

    /// Apply rewrites implied by `deferred_ptr_noops` to `values`.
    ///
    /// This **does not** update `data_inst_use_counts` - in order to do that,
    /// you must call `self.remove_value_uses(values)` beforehand, and then also
    /// call `self.after_value_uses(values)` afterwards.
    fn resolve_deferred_ptr_noop_uses(&self, values: &mut [Value]) {
        for v in values {
            // FIXME(eddyb) the loop could theoretically be avoided, but that'd
            // make tracking use counts harder.
            while let Value::Var(VarKind::NodeOutput { node: inst, output_idx: 0 }) = *v {
                match self.deferred_ptr_noops.get(&inst) {
                    Some(ptr_noop) => {
                        *v = ptr_noop.output_pointer;
                    }
                    None => break,
                }
            }
        }
    }

    // FIXME(eddyb) these are only this whacky because an `u32` is being
    // encoded as `Option<NonZeroU32>` for (dense) map entry reasons.
    fn add_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Value::Var(VarKind::NodeOutput { node: inst, .. }) = v {
                let count = self.data_inst_use_counts.entry(inst);
                *count = Some(
                    NonZeroU32::new(count.map_or(0, |c| c.get()).checked_add(1).unwrap()).unwrap(),
                );
            }
        }
    }
    fn remove_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Value::Var(VarKind::NodeOutput { node: inst, .. }) = v {
                let count = self.data_inst_use_counts.entry(inst);
                *count = NonZeroU32::new(count.unwrap().get() - 1);
            }
        }
    }
}

impl Transformer for LiftToSpvPtrInstsInFunc<'_> {
    // FIXME(eddyb) this is intentionally *shallow* and will not handle pointers
    // "hidden" in composites (which should be handled in SPIR-T explicitly).
    fn transform_const_use(&mut self, ct: Const) -> Transformed<Const> {
        // FIXME(eddyb) maybe cache this remap (in `LiftToSpvPtrs`, globally).
        let ct_def = &self.lifter.cx[ct];
        if let ConstKind::PtrToGlobalVar(gv) = ct_def.kind {
            Transformed::Changed(self.lifter.cx.intern(ConstDef {
                attrs: ct_def.attrs,
                ty: self.global_vars[gv].type_of_ptr_to,
                kind: ct_def.kind.clone(),
            }))
        } else {
            Transformed::Unchanged
        }
    }

    fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
        self.add_value_uses(&[*v]);

        v.inner_transform_with(self)
    }

    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_in_place_transform_with(self);
        self.parent_region = outer_region;
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        func_at_node.reborrow().inner_in_place_transform_with(self);

        let mut lifted = self.try_lift_data_inst_def(func_at_node.reborrow());
        if let Ok(Transformed::Unchanged) = lifted {
            let data_inst_def = func_at_node.reborrow().def();
            if let DataInstKind::QPtr(_) = data_inst_def.kind {
                lifted = Err(LiftError(Diag::bug(["unimplemented qptr instruction".into()])));
            } else {
                for output in &data_inst_def.outputs {
                    if matches!(self.lifter.cx[output.ty].kind, TypeKind::QPtr) {
                        lifted = Err(LiftError(Diag::bug([
                            "unimplemented qptr-producing instruction".into(),
                        ])));
                        break;
                    }
                }
            }
        }
        match lifted {
            Ok(Transformed::Unchanged) => {}
            Ok(Transformed::Changed(new_def)) => {
                // HACK(eddyb) this whole dance ensures that use counts
                // remain accurate, no matter what rewrites occur.
                let data_inst_def = func_at_node.def();
                self.remove_value_uses(&data_inst_def.inputs);
                *data_inst_def = new_def;
                self.resolve_deferred_ptr_noop_uses(&mut data_inst_def.inputs);
                self.add_value_uses(&data_inst_def.inputs);
            }
            Err(LiftError(e)) => {
                let data_inst_def = func_at_node.def();

                // HACK(eddyb) do not add redundant errors to `mem::analyze` bugs.
                self.func_has_mem_analysis_bug_diags = self.func_has_mem_analysis_bug_diags
                    || self.lifter.cx[data_inst_def.attrs].attrs.iter().any(|attr| match attr {
                        Attr::Diagnostics(diags) => diags.0.iter().any(|diag| match diag.level {
                            DiagLevel::Bug(loc) => {
                                loc.file().ends_with("mem/analyze.rs")
                                    || loc.file().ends_with("mem\\analyze.rs")
                            }
                            _ => false,
                        }),
                        _ => false,
                    });

                if !self.func_has_mem_analysis_bug_diags {
                    data_inst_def.attrs.push_diag(&self.lifter.cx, e);
                }
            }
        }
    }

    fn in_place_transform_func_decl(&mut self, func_decl: &mut FuncDecl) {
        func_decl.inner_in_place_transform_with(self);

        // Remove all `deferred_ptr_noops` instructions that are truly unused.
        if let DeclDef::Present(func_def_body) = &mut func_decl.def {
            let deferred_ptr_noops = mem::take(&mut self.deferred_ptr_noops);
            // NOTE(eddyb) reverse order is important, as each removal can reduce
            // use counts of an earlier definition, allowing further removal.
            for (inst, ptr_noop) in deferred_ptr_noops.into_iter().rev() {
                if self.data_inst_use_counts.get(inst).is_none() {
                    // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                    // due to the need to borrow `regions` and `nodes`
                    // at the same time - perhaps some kind of `FuncAtMut` position
                    // types for "where a list is in a parent entity" could be used
                    // to make this more ergonomic, although the potential need for
                    // an actual list entity of its own, should be considered.
                    func_def_body.regions[ptr_noop.parent_region]
                        .children
                        .remove(inst, &mut func_def_body.nodes);

                    self.remove_value_uses(&func_def_body.at(inst).def().inputs);
                }
            }
        }
    }
}
