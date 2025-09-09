//! Mutable IR traversal.

use crate::cf::{self, SelectionKind};
use crate::func_at::FuncAtMut;
use crate::mem::{DataHapp, DataHappKind, MemAccesses, MemAttr, MemOp};
use crate::qptr::{QPtrAttr, QPtrOp};
use crate::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, DataInst, DataInstDef,
    DataInstKind, DbgSrcLoc, DeclDef, EntityListIter, ExportKey, Exportee, Func, FuncDecl,
    FuncDefBody, FuncParam, GlobalVar, GlobalVarDecl, GlobalVarDefBody, Import, Module,
    ModuleDebugInfo, ModuleDialect, Node, NodeDef, NodeKind, NodeOutputDecl, OrdAssertEq, Region,
    RegionDef, RegionInputDecl, Type, TypeDef, TypeKind, TypeOrConst, Value, spv,
};
use std::cmp::Ordering;
use std::rc::Rc;
use std::slice;

/// The result of a transformation (which is not in-place).
#[must_use]
#[derive(Copy, Clone)]
pub enum Transformed<T> {
    /// The original `T` value remains as it was, at no cost.
    Unchanged,

    /// Some part of the original `T` value was transformed, and a new `T` value
    /// had to be constructed. This change will propagate in any "outer" value.
    Changed(T),
}

impl<T> Transformed<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Transformed<U> {
        match self {
            Transformed::Unchanged => Transformed::Unchanged,
            Transformed::Changed(new) => Transformed::Changed(f(new)),
        }
    }

    pub fn apply_to(self, dest: &mut T) {
        match self {
            Transformed::Unchanged => {}
            Transformed::Changed(new) => *dest = new,
        }
    }
}

// HACK(eddyb) the `Self` type here doesn't matter, it only exists so that we
// can call e.g. `Transformed::foo(...)` methods.
impl Transformed<()> {
    /// Map every element of an iterator through `transform_elem` and return
    /// [`Transformed::Changed(new_iter)`] if any `transform_elem` call returned
    /// [`Transformed::Changed`], with `new_iter` containing a combination of the
    /// changed elements, and clones of the unchanged elements.
    pub fn map_iter<'a, T: 'a + Clone>(
        iter: impl Iterator<Item = &'a T> + Clone + 'a,
        mut transform_elem: impl FnMut(&'a T) -> Transformed<T> + 'a,
    ) -> Transformed<impl Iterator<Item = T> + 'a> {
        for (i, elem_i) in iter.clone().enumerate() {
            if let Transformed::Changed(new_elem_i) = transform_elem(elem_i) {
                let mut new_elem_i = Some(new_elem_i);
                return Transformed::Changed(iter.enumerate().map(move |(j, elem_j)| {
                    match j.cmp(&i) {
                        // Earlier elements, for which `transform_elem` was called
                        // already, and had returned `Unchanged`.
                        Ordering::Less => elem_j.clone(),

                        // The first element for which `transform_elem` returned
                        // `Changed`, resulting in the `Changed` iterator.
                        Ordering::Equal => new_elem_i.take().unwrap(),

                        // Later elements, for which only now `transform_elem`
                        // gets called, and may be `Unchanged` or `Changed`.
                        Ordering::Greater => match transform_elem(elem_j) {
                            Transformed::Unchanged => elem_j.clone(),
                            Transformed::Changed(new_elem_j) => new_elem_j,
                        },
                    }
                }));
            }
        }
        Transformed::Unchanged
    }
}

/// Helper type for [`transform!`] - not public as it's easy to misuse.
enum TransformedWithOriginal<'a, T> {
    Original(&'a T),
    Changed(T),
}

impl<T> Transformed<T> {
    fn with_original(self, original: &T) -> TransformedWithOriginal<'_, T> {
        match self {
            Transformed::Unchanged => TransformedWithOriginal::Original(original),
            Transformed::Changed(new) => TransformedWithOriginal::Changed(new),
        }
    }
}

impl<T: Clone> TransformedWithOriginal<'_, T> {
    fn is_changed(&self) -> bool {
        matches!(self, TransformedWithOriginal::Changed(_))
    }
    fn changed_or_original_cloned(self) -> T {
        match self {
            TransformedWithOriginal::Original(original) => original.clone(),
            TransformedWithOriginal::Changed(new) => new,
        }
    }
}

// HACK(eddyb) `transform!` needs auto-ref-like behavior for inputs.
trait AutoRef {
    fn auto_ref(&self) -> &Self {
        self
    }
}

impl<T> AutoRef for T {}

/// Helper macro to create a combined [`Transformed`] out of several variables,
/// each with their own transformation, where any [`Transformed::Changed`] input
/// will result in a [`Transformed::Changed`] output, using a combination of the
/// changed inputs, and clones of the unchanged inputs.
macro_rules! transform {
    ({ $($input:ident -> $input_transformed:expr),+ $(,)? } => $output:expr) => {{
        let ($($input,)+) = ($($input_transformed.with_original($input.auto_ref()),)+);
        if $($input.is_changed())||+ {
            let ($($input,)*) = ($($input.changed_or_original_cloned(),)+);
            Transformed::Changed($output)
        } else {
            Transformed::Unchanged
        }
    }};
}

// FIXME(eddyb) `Sized` bound shouldn't be needed but removing it requires
// writing `impl Transformer + ?Sized` in `fn inner_transform_with` signatures.
pub trait Transformer: Sized {
    // Context-interned leaves (noop default behavior).
    fn transform_attr_set_use(&mut self, _attrs: AttrSet) -> Transformed<AttrSet> {
        Transformed::Unchanged
    }
    fn transform_type_use(&mut self, _ty: Type) -> Transformed<Type> {
        Transformed::Unchanged
    }
    fn transform_const_use(&mut self, _ct: Const) -> Transformed<Const> {
        Transformed::Unchanged
    }

    // Module-stored entity leaves (noop default behavior).
    fn transform_global_var_use(&mut self, _gv: GlobalVar) -> Transformed<GlobalVar> {
        Transformed::Unchanged
    }
    fn transform_func_use(&mut self, _func: Func) -> Transformed<Func> {
        Transformed::Unchanged
    }

    // Leaves transformed in-place (noop default behavior).
    fn in_place_transform_spv_dialect(&mut self, _dialect: &mut spv::Dialect) {}
    fn in_place_transform_spv_module_debug_info(&mut self, _debug_info: &mut spv::ModuleDebugInfo) {
    }

    // Non-leaves (defaulting to calling `.inner_transform_with(self)`).
    fn transform_attr_set_def(&mut self, attrs_def: &AttrSetDef) -> Transformed<AttrSetDef> {
        attrs_def.inner_transform_with(self)
    }
    fn transform_attr(&mut self, attr: &Attr) -> Transformed<Attr> {
        attr.inner_transform_with(self)
    }
    fn transform_type_def(&mut self, ty_def: &TypeDef) -> Transformed<TypeDef> {
        ty_def.inner_transform_with(self)
    }
    fn transform_const_def(&mut self, ct_def: &ConstDef) -> Transformed<ConstDef> {
        ct_def.inner_transform_with(self)
    }
    fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
        v.inner_transform_with(self)
    }

    // Non-leaves transformed in-place (defaulting to calling
    // `.inner_in_place_transform_with(self)`).
    fn in_place_transform_module(&mut self, module: &mut Module) {
        module.inner_in_place_transform_with(self);
    }
    fn in_place_transform_module_dialect(&mut self, dialect: &mut ModuleDialect) {
        dialect.inner_in_place_transform_with(self);
    }
    fn in_place_transform_module_debug_info(&mut self, debug_info: &mut ModuleDebugInfo) {
        debug_info.inner_in_place_transform_with(self);
    }
    fn in_place_transform_global_var_decl(&mut self, gv_decl: &mut GlobalVarDecl) {
        gv_decl.inner_in_place_transform_with(self);
    }
    fn in_place_transform_func_decl(&mut self, func_decl: &mut FuncDecl) {
        func_decl.inner_in_place_transform_with(self);
    }
    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        func_at_region.inner_in_place_transform_with(self);
    }
    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        func_at_node.inner_in_place_transform_with(self);
    }
    fn in_place_transform_data_inst_def(&mut self, mut func_at_data_inst: FuncAtMut<'_, DataInst>) {
        func_at_data_inst.inner_in_place_transform_with(self);
    }
}

/// Trait implemented on "transformable" types, to further "elaborate" a type by
/// transforming its "interior" (i.e. variants and/or fields).
///
/// That is, an `impl InnerTransform for X` will call the relevant [`Transformer`]
/// method for each `X` field, effectively performing a single level of a deep
/// transform.
/// Also, if `Transformer::transform_X` exists for a given `X`, its default should
/// be to call `X::inner_transform_with` (i.e. so that transforming is mostly-deep
/// by default).
pub trait InnerTransform: Sized {
    // FIXME(eddyb) the naming here isn't great, can it be improved?
    // FIXME(eddyb) should this be `self -> Self` instead of `&mut self -> ()`?
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self>;
}

/// Like [`InnerTransform`], but only for the `in_place_transform_X` cases.
pub trait InnerInPlaceTransform {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer);
}

// FIXME(eddyb) should the impls be here, or next to definitions? (maybe derived?)
impl InnerInPlaceTransform for Module {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        // FIXME(eddyb) this can't be exhaustive because of the private `cx` field.
        let Self { dialect, debug_info, global_vars: _, funcs: _, exports, .. } = self;

        transformer.in_place_transform_module_dialect(dialect);
        transformer.in_place_transform_module_debug_info(debug_info);

        // HACK(eddyb) this takes two passes, once for values and once for keys,
        // to be able to use in-place mutable iteration for the former, and
        // `Transformed::map_iter` (i.e. immutable iteration) for the latter.
        for exportee in exports.values_mut() {
            exportee.inner_transform_with(transformer).apply_to(exportee);
        }
        Transformed::map_iter(exports.keys(), |export_key| {
            export_key.inner_transform_with(transformer)
        })
        .map(|new_keys_iter| {
            // Recombine the new keys with the existing values.
            new_keys_iter.zip(exports.values().cloned()).collect()
        })
        .apply_to(exports);
    }
}

impl InnerInPlaceTransform for ModuleDialect {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        match self {
            Self::Spv(dialect) => transformer.in_place_transform_spv_dialect(dialect),
        }
    }
}

impl InnerInPlaceTransform for ModuleDebugInfo {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        match self {
            Self::Spv(debug_info) => {
                transformer.in_place_transform_spv_module_debug_info(debug_info);
            }
        }
    }
}

impl InnerTransform for ExportKey {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        match self {
            Self::LinkName(_) => Transformed::Unchanged,

            Self::SpvEntryPoint { imms, interface_global_vars } => transform!({
                imms -> Transformed::Unchanged,
                interface_global_vars -> Transformed::map_iter(
                    interface_global_vars.iter(),
                    |&gv| transformer.transform_global_var_use(gv),
                ).map(|new_iter| new_iter.collect()),
            } => Self::SpvEntryPoint {
                imms,
                interface_global_vars,
            }),
        }
    }
}

impl InnerTransform for Exportee {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        match *self {
            Self::GlobalVar(gv) => transform!({
                gv -> transformer.transform_global_var_use(gv),
            } => Self::GlobalVar(gv)),

            Self::Func(func) => transform!({
                func -> transformer.transform_func_use(func),
            } => Self::Func(func)),
        }
    }
}

impl InnerTransform for AttrSetDef {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        let Self { attrs } = self;

        transform!({
            attrs -> Transformed::map_iter(
                attrs.iter(),
                |attr| transformer.transform_attr(attr),
            ).map(|new_iter| new_iter.collect()),
        } => Self {
            attrs,
        })
    }
}

impl InnerTransform for Attr {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        match self {
            Attr::Diagnostics(_) | Attr::SpvAnnotation(_) | Attr::SpvBitflagsOperand(_) => {
                Transformed::Unchanged
            }

            &Attr::DbgSrcLoc(OrdAssertEq(DbgSrcLoc {
                file_path,
                start_line_col,
                end_line_col,
                inlined_callee_name_and_call_site,
            })) => {
                // FIXME(eddyb) this should be replaced with an impl of `InnerTransform`
                // for `Option<T>` or some other helper, to avoid "manual transpose".
                transform!({
                    inlined_callee_name_and_call_site -> inlined_callee_name_and_call_site
                        .map(|(callee_name, call_site_attrs)| {
                            transformer.transform_attr_set_use(call_site_attrs)
                                .map(|call_site_attrs| (callee_name, call_site_attrs))
                        })
                        .map_or(Transformed::Unchanged, |t| t.map(Some))
                } => Attr::DbgSrcLoc(OrdAssertEq(DbgSrcLoc {
                    file_path,
                    start_line_col,
                    end_line_col,
                    inlined_callee_name_and_call_site,
                })))
            }

            Attr::Mem(attr) => transform!({
                attr -> match attr {
                    MemAttr::Accesses(OrdAssertEq(accesses)) => transform!({
                        accesses -> match accesses {
                            &MemAccesses::Handles(crate::mem::shapes::Handle::Opaque(ty)) => transform!({
                                ty -> transformer.transform_type_use(ty),
                            } => MemAccesses::Handles(crate::mem::shapes::Handle::Opaque(ty))),
                            MemAccesses::Handles(crate::mem::shapes::Handle::Buffer(addr_space, data_happ)) => transform!({
                                data_happ -> data_happ.inner_transform_with(transformer),
                            } => MemAccesses::Handles(crate::mem::shapes::Handle::Buffer(*addr_space, data_happ))),
                            MemAccesses::Data(happ) => transform!({
                                happ -> happ.inner_transform_with(transformer),
                            } => MemAccesses::Data(happ)),
                        }
                    } => MemAttr::Accesses(OrdAssertEq(accesses))),
                }
            } => Attr::Mem(attr)),

            Attr::QPtr(attr) => transform!({
                attr -> match *attr {
                    QPtrAttr::ToSpvPtrInput { input_idx, pointee } => transform!({
                        pointee -> transformer.transform_type_use(pointee.0).map(OrdAssertEq),
                    } => QPtrAttr::ToSpvPtrInput { input_idx, pointee }),

                    QPtrAttr::FromSpvPtrOutput {
                        addr_space,
                        pointee,
                    } => transform!({
                        pointee -> transformer.transform_type_use(pointee.0).map(OrdAssertEq),
                    } => QPtrAttr::FromSpvPtrOutput {
                        addr_space,
                        pointee,
                    }),
                }
            } => Attr::QPtr(attr)),
        }
    }
}

// FIXME(eddyb) this should maybe be in a more general spot.
impl<T: InnerTransform> InnerTransform for Rc<T> {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        (**self).inner_transform_with(transformer).map(Rc::new)
    }
}

impl InnerTransform for DataHapp {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        let Self { max_size, kind } = self;

        transform!({
            kind -> kind.inner_transform_with(transformer)
        } => Self {
            max_size: *max_size,
            kind,
        })
    }
}

impl InnerTransform for DataHappKind {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        match self {
            Self::Dead => Transformed::Unchanged,
            &Self::StrictlyTyped(ty) => transform!({
                ty -> transformer.transform_type_use(ty),
            } => Self::StrictlyTyped(ty)),
            &Self::Direct(ty) => transform!({
                ty -> transformer.transform_type_use(ty),
            } => Self::Direct(ty)),
            Self::Disjoint(entries) => transform!({
                entries -> Transformed::map_iter(entries.values(), |sub_happ| {
                    sub_happ.inner_transform_with(transformer)
                }).map(|new_iter| {
                    // HACK(eddyb) this is a bit inefficient but `Transformed::map_iter`
                    // limits us here in how it handles the whole `Clone` thing.
                    entries.keys().copied().zip(new_iter).collect()
                }).map(Rc::new)
            } => Self::Disjoint(entries)),
            Self::Repeated { element, stride } => transform!({
                element -> element.inner_transform_with(transformer),
            } => Self::Repeated { element, stride: *stride }),
        }
    }
}

impl InnerTransform for TypeDef {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        let Self { attrs, kind } = self;

        transform!({
            attrs -> transformer.transform_attr_set_use(*attrs),
            kind -> match kind {
                TypeKind::QPtr | TypeKind::SpvStringLiteralForExtInst => Transformed::Unchanged,

                TypeKind::SpvInst { spv_inst, type_and_const_inputs } => Transformed::map_iter(
                    type_and_const_inputs.iter(),
                    |ty_or_ct| match *ty_or_ct {
                        TypeOrConst::Type(ty) => transform!({
                            ty -> transformer.transform_type_use(ty),
                        } => TypeOrConst::Type(ty)),

                        TypeOrConst::Const(ct) => transform!({
                            ct -> transformer.transform_const_use(ct),
                        } => TypeOrConst::Const(ct)),
                    },
                ).map(|new_iter| TypeKind::SpvInst {
                    spv_inst: spv_inst.clone(),
                    type_and_const_inputs: new_iter.collect(),
                }),
            },
        } => Self {
            attrs,
            kind,
        })
    }
}

impl InnerTransform for ConstDef {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        let Self { attrs, ty, kind } = self;

        transform!({
            attrs -> transformer.transform_attr_set_use(*attrs),
            ty -> transformer.transform_type_use(*ty),
            kind -> match kind {
                ConstKind::PtrToGlobalVar(gv) => transform!({
                    gv -> transformer.transform_global_var_use(*gv),
                } => ConstKind::PtrToGlobalVar(gv)),

                ConstKind::PtrToFunc(func) => transform!({
                    func -> transformer.transform_func_use(*func),
                } => ConstKind::PtrToFunc(func)),

                ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                    let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                    Transformed::map_iter(
                        const_inputs.iter(),
                        |&ct| transformer.transform_const_use(ct),
                    ).map(|new_iter| ConstKind::SpvInst {
                        spv_inst_and_const_inputs: Rc::new((spv_inst.clone(), new_iter.collect())),
                    })
                }
                ConstKind::SpvStringLiteralForExtInst(_) => Transformed::Unchanged
            },
        } => Self {
            attrs,
            ty,
            kind,
        })
    }
}

impl<D: InnerInPlaceTransform> InnerInPlaceTransform for DeclDef<D> {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        match self {
            Self::Imported(import) => match import {
                Import::LinkName(_) => {}
            },
            Self::Present(def) => def.inner_in_place_transform_with(transformer),
        }
    }
}

impl InnerInPlaceTransform for GlobalVarDecl {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        let Self { attrs, type_of_ptr_to, shape, addr_space, def } = self;

        transformer.transform_attr_set_use(*attrs).apply_to(attrs);
        transformer.transform_type_use(*type_of_ptr_to).apply_to(type_of_ptr_to);
        if let Some(shape) = shape {
            match shape {
                crate::mem::shapes::GlobalVarShape::TypedInterface(ty) => {
                    transformer.transform_type_use(*ty).apply_to(ty);
                }
                crate::mem::shapes::GlobalVarShape::Handles { .. }
                | crate::mem::shapes::GlobalVarShape::UntypedData(_) => {}
            }
        }
        match addr_space {
            AddrSpace::Handles | AddrSpace::SpvStorageClass(_) => {}
        }
        def.inner_in_place_transform_with(transformer);
    }
}

impl InnerInPlaceTransform for GlobalVarDefBody {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        let Self { initializer } = self;

        if let Some(initializer) = initializer {
            transformer.transform_const_use(*initializer).apply_to(initializer);
        }
    }
}

impl InnerInPlaceTransform for FuncDecl {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        let Self { attrs, ret_type, params, def } = self;

        transformer.transform_attr_set_use(*attrs).apply_to(attrs);
        transformer.transform_type_use(*ret_type).apply_to(ret_type);
        for param in params {
            param.inner_transform_with(transformer).apply_to(param);
        }
        def.inner_in_place_transform_with(transformer);
    }
}

impl InnerTransform for FuncParam {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        let Self { attrs, ty } = self;

        transform!({
            attrs -> transformer.transform_attr_set_use(*attrs),
            ty -> transformer.transform_type_use(*ty),
        } => Self {
            attrs,
            ty,
        })
    }
}

impl InnerInPlaceTransform for FuncDefBody {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        match &self.unstructured_cfg {
            None => transformer.in_place_transform_region_def(self.at_mut_body()),
            Some(cfg) => {
                // HACK(eddyb) have to compute this before borrowing any `self` fields.
                let rpo = cfg.rev_post_order(self);

                for region in rpo {
                    transformer.in_place_transform_region_def(self.at_mut(region));

                    let cfg = self.unstructured_cfg.as_mut().unwrap();
                    if let Some(control_inst) = cfg.control_inst_on_exit_from.get_mut(region) {
                        control_inst.inner_in_place_transform_with(transformer);
                    }
                }
            }
        }
    }
}

impl InnerInPlaceTransform for FuncAtMut<'_, Region> {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        // HACK(eddyb) handle the fields of `Region` separately, to
        // allow reborrowing `FuncAtMut` (for recursing into `Node`s).
        let RegionDef { inputs, children: _, outputs: _ } = self.reborrow().def();
        for input in inputs {
            input.inner_transform_with(transformer).apply_to(input);
        }

        self.reborrow().at_children().into_iter().inner_in_place_transform_with(transformer);

        let RegionDef { inputs: _, children: _, outputs } = self.reborrow().def();
        for v in outputs {
            transformer.transform_value_use(v).apply_to(v);
        }
    }
}

impl InnerTransform for RegionInputDecl {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        let Self { attrs, ty } = self;

        transform!({
            attrs -> transformer.transform_attr_set_use(*attrs),
            ty -> transformer.transform_type_use(*ty),
        } => Self {
            attrs,
            ty,
        })
    }
}

impl InnerInPlaceTransform for FuncAtMut<'_, EntityListIter<Node>> {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        let mut iter = self.reborrow();
        while let Some(func_at_node) = iter.next() {
            transformer.in_place_transform_node_def(func_at_node);
        }
    }
}

impl FuncAtMut<'_, Node> {
    fn child_regions(&mut self) -> &mut [Region] {
        match &mut self.reborrow().def().kind {
            NodeKind::Block { .. } | NodeKind::ExitInvocation { .. } => &mut [][..],

            NodeKind::Select { cases, .. } => cases,
            NodeKind::Loop { body, .. } => slice::from_mut(body),
        }
    }
}

impl InnerInPlaceTransform for FuncAtMut<'_, Node> {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        // HACK(eddyb) handle all pre-child-regions fields separately to
        // allow reborrowing `FuncAtMut` (for the child region recursion).
        let NodeDef { attrs, inputs, kind, outputs: _ } = self.reborrow().def();
        transformer.transform_attr_set_use(*attrs).apply_to(attrs);
        for v in inputs {
            transformer.transform_value_use(v).apply_to(v);
        }
        match kind {
            &mut NodeKind::Block { insts } => {
                let mut func_at_inst_iter = self.reborrow().at(insts).into_iter();
                while let Some(func_at_inst) = func_at_inst_iter.next() {
                    transformer.in_place_transform_data_inst_def(func_at_inst);
                }
            }
            NodeKind::Select {
                kind: SelectionKind::BoolCond | SelectionKind::SpvInst(_),
                cases: _,
            }
            | NodeKind::Loop { body: _, repeat_condition: _ }
            | NodeKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(_)) => {}
        }

        // FIXME(eddyb) represent the list of child regions without having them
        // in a `Vec` (or `SmallVec`), which requires workarounds like this.
        for child_region_idx in 0..self.child_regions().len() {
            let child_region = self.child_regions()[child_region_idx];
            transformer.in_place_transform_region_def(self.reborrow().at(child_region));
        }

        let NodeDef { attrs: _, inputs: _, kind, outputs } = self.reborrow().def();

        match kind {
            // Fully handled above, before recursing into any child regions.
            NodeKind::Block { insts: _ }
            | NodeKind::Select { kind: _, cases: _ }
            | NodeKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(_)) => {}

            NodeKind::Loop { body: _, repeat_condition } => {
                transformer.transform_value_use(repeat_condition).apply_to(repeat_condition);
            }
        };

        for output in outputs {
            output.inner_transform_with(transformer).apply_to(output);
        }
    }
}

impl InnerTransform for NodeOutputDecl {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        let Self { attrs, ty } = self;

        transform!({
            attrs -> transformer.transform_attr_set_use(*attrs),
            ty -> transformer.transform_type_use(*ty),
        } => Self {
            attrs,
            ty,
        })
    }
}

impl InnerInPlaceTransform for FuncAtMut<'_, DataInst> {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        let DataInstDef { attrs, kind, inputs, output_type } = self.reborrow().def();

        transformer.transform_attr_set_use(*attrs).apply_to(attrs);
        kind.inner_in_place_transform_with(transformer);
        for v in inputs {
            transformer.transform_value_use(v).apply_to(v);
        }
        if let Some(output_type) = output_type {
            transformer.transform_type_use(*output_type).apply_to(output_type);
        }
    }
}

impl InnerInPlaceTransform for DataInstKind {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        match self {
            DataInstKind::FuncCall(func) => transformer.transform_func_use(*func).apply_to(func),
            DataInstKind::Mem(op) => match op {
                MemOp::FuncLocalVar(_) | MemOp::Load | MemOp::Store => {}
            },
            DataInstKind::QPtr(op) => match op {
                QPtrOp::HandleArrayIndex
                | QPtrOp::BufferData
                | QPtrOp::BufferDynLen { .. }
                | QPtrOp::Offset(_)
                | QPtrOp::DynOffset { .. } => {}
            },
            DataInstKind::SpvInst(_) | DataInstKind::SpvExtInst { .. } => {}
        }
    }
}

impl InnerInPlaceTransform for cf::unstructured::ControlInst {
    fn inner_in_place_transform_with(&mut self, transformer: &mut impl Transformer) {
        let Self { attrs, kind, inputs, targets: _, target_inputs } = self;

        transformer.transform_attr_set_use(*attrs).apply_to(attrs);
        match kind {
            cf::unstructured::ControlInstKind::Unreachable
            | cf::unstructured::ControlInstKind::Return
            | cf::unstructured::ControlInstKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(
                _,
            ))
            | cf::unstructured::ControlInstKind::Branch
            | cf::unstructured::ControlInstKind::SelectBranch(
                SelectionKind::BoolCond | SelectionKind::SpvInst(_),
            ) => {}
        }
        for v in inputs {
            transformer.transform_value_use(v).apply_to(v);
        }
        for inputs in target_inputs.values_mut() {
            for v in inputs {
                transformer.transform_value_use(v).apply_to(v);
            }
        }
    }
}

impl InnerTransform for Value {
    fn inner_transform_with(&self, transformer: &mut impl Transformer) -> Transformed<Self> {
        match self {
            Self::Const(ct) => transform!({
                ct -> transformer.transform_const_use(*ct),
            } => Self::Const(ct)),

            Self::RegionInput { region: _, input_idx: _ }
            | Self::NodeOutput { node: _, output_idx: _ }
            | Self::DataInstOutput(_) => Transformed::Unchanged,
        }
    }
}
