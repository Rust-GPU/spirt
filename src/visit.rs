//! Immutable IR traversal.

use crate::cf::{self, SelectionKind};
use crate::func_at::FuncAt;
use crate::mem::{DataHapp, DataHappKind, MemAccesses, MemAttr, MemOp};
use crate::qptr::{QPtrAttr, QPtrOp};
use crate::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, DataInstKind, DbgSrcLoc,
    DeclDef, DiagMsgPart, EntityListIter, ExportKey, Exportee, Func, FuncDecl, FuncDefBody,
    FuncParam, GlobalVar, GlobalVarDecl, GlobalVarDefBody, Import, Module, ModuleDebugInfo,
    ModuleDialect, Node, NodeDef, NodeKind, NodeOutputDecl, OrdAssertEq, Region, RegionDef,
    RegionInputDecl, Type, TypeDef, TypeKind, TypeOrConst, Value, VarKind, spv,
};

// FIXME(eddyb) `Sized` bound shouldn't be needed but removing it requires
// writing `impl Visitor<'a> + ?Sized` in `fn inner_visit_with` signatures.
pub trait Visitor<'a>: Sized {
    // Context-interned leaves (no default provided).
    // FIXME(eddyb) treat these separately somehow and allow e.g. automatic deep
    // visiting (with a set to avoid repeat visits) if a `Rc<Context>` is provided.
    fn visit_attr_set_use(&mut self, attrs: AttrSet);
    fn visit_type_use(&mut self, ty: Type);
    fn visit_const_use(&mut self, ct: Const);

    // Module-stored entity leaves (no default provided).
    fn visit_global_var_use(&mut self, gv: GlobalVar);
    fn visit_func_use(&mut self, func: Func);

    // Leaves (noop default behavior).
    fn visit_spv_dialect(&mut self, _dialect: &spv::Dialect) {}
    fn visit_spv_module_debug_info(&mut self, _debug_info: &spv::ModuleDebugInfo) {}
    fn visit_import(&mut self, _import: &Import) {}

    // Non-leaves (defaulting to calling `.inner_visit_with(self)`).
    fn visit_module(&mut self, module: &'a Module) {
        module.inner_visit_with(self);
    }
    fn visit_module_dialect(&mut self, dialect: &'a ModuleDialect) {
        dialect.inner_visit_with(self);
    }
    fn visit_module_debug_info(&mut self, debug_info: &'a ModuleDebugInfo) {
        debug_info.inner_visit_with(self);
    }
    fn visit_attr_set_def(&mut self, attrs_def: &'a AttrSetDef) {
        attrs_def.inner_visit_with(self);
    }
    fn visit_attr(&mut self, attr: &'a Attr) {
        attr.inner_visit_with(self);
    }
    fn visit_type_def(&mut self, ty_def: &'a TypeDef) {
        ty_def.inner_visit_with(self);
    }
    fn visit_const_def(&mut self, ct_def: &'a ConstDef) {
        ct_def.inner_visit_with(self);
    }
    fn visit_global_var_decl(&mut self, gv_decl: &'a GlobalVarDecl) {
        gv_decl.inner_visit_with(self);
    }
    fn visit_func_decl(&mut self, func_decl: &'a FuncDecl) {
        func_decl.inner_visit_with(self);
    }
    fn visit_region_def(&mut self, func_at_region: FuncAt<'a, Region>) {
        func_at_region.inner_visit_with(self);
    }
    fn visit_node_def(&mut self, func_at_node: FuncAt<'a, Node>) {
        func_at_node.inner_visit_with(self);
    }
    fn visit_value_use(&mut self, v: &'a Value) {
        v.inner_visit_with(self);
    }
}

/// Trait implemented on "visitable" types (shallowly visitable, at least).
///
/// That is, an `impl Visit for X` will call the relevant [`Visitor`] method for
/// `X`, typically named `Visitor::visit_X` or `Visitor::visit_X_use`.
//
// FIXME(eddyb) use this more (e.g. in implementing `InnerVisit`).
pub trait Visit {
    fn visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>);
}

macro_rules! impl_visit {
    (
        by_val { $($by_val_method:ident($by_val_ty:ty)),* $(,)? }
        by_ref { $($by_ref_method:ident($by_ref_ty:ty)),* $(,)? }
        forward_to_inner_visit { $($forward_to_inner_visit_ty:ty),* $(,)? }
    ) => {
        $(impl Visit for $by_val_ty {
            fn visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
                visitor.$by_val_method(*self);
            }
        })*
        $(impl Visit for $by_ref_ty {
            fn visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
                visitor.$by_ref_method(self);
            }
        })*
        $(impl Visit for $forward_to_inner_visit_ty {
            fn visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
                self.inner_visit_with(visitor);
            }
        })*
    };
}

impl_visit! {
    by_val {
        visit_attr_set_use(AttrSet),
        visit_type_use(Type),
        visit_const_use(Const),
        visit_global_var_use(GlobalVar),
        visit_func_use(Func),
    }
    by_ref {
        visit_spv_dialect(spv::Dialect),
        visit_spv_module_debug_info(spv::ModuleDebugInfo),
        visit_import(Import),
        visit_module(Module),
        visit_module_dialect(ModuleDialect),
        visit_module_debug_info(ModuleDebugInfo),
        visit_attr_set_def(AttrSetDef),
        visit_attr(Attr),
        visit_type_def(TypeDef),
        visit_const_def(ConstDef),
        visit_global_var_decl(GlobalVarDecl),
        visit_func_decl(FuncDecl),
        visit_value_use(Value),
    }
    forward_to_inner_visit {
        // NOTE(eddyb) the interpolated parts of `Attr::Diagnostics` aren't visited
        // by default (as they're "inert data"), this is only for `print`'s usage.
        Vec<DiagMsgPart>,
    }
}

/// Trait implemented on "deeply visitable" types, to further "explore" a type
/// by visiting its "interior" (i.e. variants and/or fields).
///
/// That is, an `impl InnerVisit for X` will call the relevant [`Visitor`] method
/// for each `X` field, effectively performing a single level of a deep visit.
/// Also, if `Visitor::visit_X` exists for a given `X`, its default should be to
/// call `X::inner_visit_with` (i.e. so that visiting is mostly-deep by default).
pub trait InnerVisit {
    // FIXME(eddyb) the naming here isn't great, can it be improved?
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>);
}

/// Dynamic dispatch version of [`InnerVisit`].
///
/// `dyn DynInnerVisit<'a, V>` is possible, unlike `dyn InnerVisit`, because of
/// the `trait`-level type parameter `V`, which replaces the method parameter.
pub trait DynInnerVisit<'a, V> {
    fn dyn_inner_visit_with(&'a self, visitor: &mut V);
}

impl<'a, T: InnerVisit, V: Visitor<'a>> DynInnerVisit<'a, V> for T {
    fn dyn_inner_visit_with(&'a self, visitor: &mut V) {
        self.inner_visit_with(visitor);
    }
}

// FIXME(eddyb) should the impls be here, or next to definitions? (maybe derived?)
impl InnerVisit for Module {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        // FIXME(eddyb) this can't be exhaustive because of the private `cx` field.
        let Self { dialect, debug_info, global_vars: _, funcs: _, exports, .. } = self;

        visitor.visit_module_dialect(dialect);
        visitor.visit_module_debug_info(debug_info);
        for (export_key, exportee) in exports {
            export_key.inner_visit_with(visitor);
            exportee.inner_visit_with(visitor);
        }
    }
}

impl InnerVisit for ModuleDialect {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::Spv(dialect) => visitor.visit_spv_dialect(dialect),
        }
    }
}

impl InnerVisit for ModuleDebugInfo {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::Spv(debug_info) => {
                visitor.visit_spv_module_debug_info(debug_info);
            }
        }
    }
}

impl InnerVisit for ExportKey {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::LinkName(_) => {}

            Self::SpvEntryPoint { imms: _, interface_global_vars } => {
                for &gv in interface_global_vars {
                    visitor.visit_global_var_use(gv);
                }
            }
        }
    }
}

impl InnerVisit for Exportee {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match *self {
            Self::GlobalVar(gv) => visitor.visit_global_var_use(gv),
            Self::Func(func) => visitor.visit_func_use(func),
        }
    }
}

impl InnerVisit for AttrSetDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs } = self;

        for attr in attrs {
            visitor.visit_attr(attr);
        }
    }
}

impl InnerVisit for Attr {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Attr::Diagnostics(_) | Attr::SpvAnnotation(_) | Attr::SpvBitflagsOperand(_) => {}

            &Attr::DbgSrcLoc(OrdAssertEq(DbgSrcLoc {
                file_path: _,
                start_line_col: _,
                end_line_col: _,
                inlined_callee_name_and_call_site,
            })) => {
                if let Some((_callee_name, call_site_attrs)) = inlined_callee_name_and_call_site {
                    visitor.visit_attr_set_use(call_site_attrs);
                }
            }

            Attr::Mem(attr) => match attr {
                MemAttr::Accesses(accesses) => accesses.0.inner_visit_with(visitor),
            },

            Attr::QPtr(attr) => match attr {
                QPtrAttr::ToSpvPtrInput { input_idx: _, pointee }
                | QPtrAttr::FromSpvPtrOutput { addr_space: _, pointee } => {
                    visitor.visit_type_use(pointee.0);
                }
            },
        }
    }
}

// NOTE(eddyb) the interpolated parts of `Attr::Diagnostics` aren't visited
// by default (as they're "inert data"), this is only for `print`'s usage.
impl InnerVisit for Vec<DiagMsgPart> {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        for part in self {
            match part {
                DiagMsgPart::Plain(_) => {}
                &DiagMsgPart::Attrs(attrs) => visitor.visit_attr_set_use(attrs),
                &DiagMsgPart::Type(ty) => visitor.visit_type_use(ty),
                &DiagMsgPart::Const(ct) => visitor.visit_const_use(ct),
                DiagMsgPart::MemAccesses(accesses) => accesses.inner_visit_with(visitor),
            }
        }
    }
}

impl InnerVisit for MemAccesses {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            &MemAccesses::Handles(crate::mem::shapes::Handle::Opaque(ty)) => {
                visitor.visit_type_use(ty);
            }
            MemAccesses::Handles(crate::mem::shapes::Handle::Buffer(_, data_happ)) => {
                data_happ.inner_visit_with(visitor);
            }
            MemAccesses::Data(happ) => happ.inner_visit_with(visitor),
        }
    }
}

impl InnerVisit for DataHapp {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { max_size: _, kind } = self;
        kind.inner_visit_with(visitor);
    }
}

impl InnerVisit for DataHappKind {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::Dead => {}
            &Self::StrictlyTyped(ty) | &Self::Direct(ty) => {
                visitor.visit_type_use(ty);
            }
            Self::Disjoint(entries) => {
                for sub_happ in entries.values() {
                    sub_happ.inner_visit_with(visitor);
                }
            }
            Self::Repeated { element, stride: _ } => {
                element.inner_visit_with(visitor);
            }
        }
    }
}

impl InnerVisit for TypeDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, kind } = self;

        visitor.visit_attr_set_use(*attrs);
        match kind {
            TypeKind::QPtr | TypeKind::SpvStringLiteralForExtInst => {}

            TypeKind::SpvInst { spv_inst: _, type_and_const_inputs } => {
                for &ty_or_ct in type_and_const_inputs {
                    match ty_or_ct {
                        TypeOrConst::Type(ty) => visitor.visit_type_use(ty),
                        TypeOrConst::Const(ct) => visitor.visit_const_use(ct),
                    }
                }
            }
        }
    }
}

impl InnerVisit for ConstDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, ty, kind } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*ty);
        match kind {
            &ConstKind::PtrToGlobalVar(gv) => visitor.visit_global_var_use(gv),
            &ConstKind::PtrToFunc(func) => visitor.visit_func_use(func),
            ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                let (_spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                for &ct in const_inputs {
                    visitor.visit_const_use(ct);
                }
            }
            ConstKind::SpvStringLiteralForExtInst(_) => {}
        }
    }
}

impl<D: InnerVisit> InnerVisit for DeclDef<D> {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::Imported(import) => visitor.visit_import(import),
            Self::Present(def) => def.inner_visit_with(visitor),
        }
    }
}

impl InnerVisit for GlobalVarDecl {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, type_of_ptr_to, shape, addr_space, def } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*type_of_ptr_to);
        if let Some(shape) = shape {
            match shape {
                crate::mem::shapes::GlobalVarShape::TypedInterface(ty) => {
                    visitor.visit_type_use(*ty);
                }
                crate::mem::shapes::GlobalVarShape::Handles { .. }
                | crate::mem::shapes::GlobalVarShape::UntypedData(_) => {}
            }
        }
        match addr_space {
            AddrSpace::Handles | AddrSpace::SpvStorageClass(_) => {}
        }
        def.inner_visit_with(visitor);
    }
}

impl InnerVisit for GlobalVarDefBody {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { initializer } = self;

        if let Some(initializer) = *initializer {
            visitor.visit_const_use(initializer);
        }
    }
}

impl InnerVisit for FuncDecl {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, ret_type, params, def } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*ret_type);
        for param in params {
            param.inner_visit_with(visitor);
        }
        def.inner_visit_with(visitor);
    }
}

impl InnerVisit for FuncParam {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, ty } = *self;

        visitor.visit_attr_set_use(attrs);
        visitor.visit_type_use(ty);
    }
}

impl InnerVisit for FuncDefBody {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match &self.unstructured_cfg {
            None => visitor.visit_region_def(self.at_body()),
            Some(cfg) => {
                for region in cfg.rev_post_order(self) {
                    visitor.visit_region_def(self.at(region));

                    if let Some(control_inst) = cfg.control_inst_on_exit_from.get(region) {
                        control_inst.inner_visit_with(visitor);
                    }
                }
            }
        }
    }
}

// FIXME(eddyb) this can't implement `InnerVisit` because of the `&'a self`
// requirement, whereas this has `'a` in `self: FuncAt<'a, Region>`.
impl<'a> FuncAt<'a, Region> {
    pub fn inner_visit_with(self, visitor: &mut impl Visitor<'a>) {
        let RegionDef { inputs, children, outputs } = self.def();

        for input in inputs {
            input.inner_visit_with(visitor);
        }
        self.at(*children).into_iter().inner_visit_with(visitor);
        for v in outputs {
            visitor.visit_value_use(v);
        }
    }
}

impl InnerVisit for RegionInputDecl {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, ty } = *self;

        visitor.visit_attr_set_use(attrs);
        visitor.visit_type_use(ty);
    }
}

// FIXME(eddyb) this can't implement `InnerVisit` because of the `&'a self`
// requirement, whereas this has `'a` in `self: FuncAt<'a, ...>`.
impl<'a> FuncAt<'a, EntityListIter<Node>> {
    pub fn inner_visit_with(self, visitor: &mut impl Visitor<'a>) {
        for func_at_node in self {
            visitor.visit_node_def(func_at_node);
        }
    }
}

// FIXME(eddyb) this can't implement `InnerVisit` because of the `&'a self`
// requirement, whereas this has `'a` in `self: FuncAt<'a, Node>`.
impl<'a> FuncAt<'a, Node> {
    pub fn inner_visit_with(self, visitor: &mut impl Visitor<'a>) {
        let NodeDef { attrs, kind, inputs, child_regions, outputs } = self.def();

        visitor.visit_attr_set_use(*attrs);
        match kind {
            &DataInstKind::FuncCall(func) => visitor.visit_func_use(func),

            NodeKind::Select(SelectionKind::BoolCond | SelectionKind::SpvInst(_))
            | NodeKind::Loop { repeat_condition: _ }
            | NodeKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(_))
            | DataInstKind::Mem(MemOp::FuncLocalVar(_) | MemOp::Load | MemOp::Store)
            | DataInstKind::QPtr(
                QPtrOp::HandleArrayIndex
                | QPtrOp::BufferData
                | QPtrOp::BufferDynLen { .. }
                | QPtrOp::Offset(_)
                | QPtrOp::DynOffset { .. },
            )
            | DataInstKind::SpvInst(_)
            | DataInstKind::SpvExtInst { .. } => {}
        }
        for v in inputs {
            visitor.visit_value_use(v);
        }
        for &region in child_regions {
            visitor.visit_region_def(self.at(region));
        }

        // HACK(eddyb) semantically, `repeat_condition` is a body region output.
        if let NodeKind::Loop { repeat_condition } = kind {
            visitor.visit_value_use(repeat_condition);
        }

        for output in outputs {
            output.inner_visit_with(visitor);
        }
    }
}

impl InnerVisit for NodeOutputDecl {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, ty } = *self;

        visitor.visit_attr_set_use(attrs);
        visitor.visit_type_use(ty);
    }
}

impl InnerVisit for cf::unstructured::ControlInst {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, kind, inputs, targets: _, target_inputs } = self;

        visitor.visit_attr_set_use(*attrs);
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
            visitor.visit_value_use(v);
        }
        for inputs in target_inputs.values() {
            for v in inputs {
                visitor.visit_value_use(v);
            }
        }
    }
}

impl InnerVisit for Value {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            &Self::Const(ct) => visitor.visit_const_use(ct),
            Self::Var(var) => var.inner_visit_with(visitor),
        }
    }
}

impl InnerVisit for VarKind {
    fn inner_visit_with<'a>(&'a self, _visitor: &mut impl Visitor<'a>) {
        match *self {
            Self::RegionInput { region: _, input_idx: _ }
            | Self::NodeOutput { node: _, output_idx: _ } => {}
        }
    }
}
