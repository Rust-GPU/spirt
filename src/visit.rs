use crate::{
    spv, AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstCtor, ConstCtorArg, ConstDef, Func,
    GlobalVar, GlobalVarDef, Misc, MiscInput, MiscKind, MiscOutput, Module, ModuleDebugInfo,
    ModuleDialect, Type, TypeCtor, TypeCtorArg, TypeDef,
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

    // Module-stored (but context-allocated indices) leaves (no default provided).
    fn visit_global_var_use(&mut self, gv: GlobalVar);

    // Leaves (noop default behavior).
    fn visit_spv_dialect(&mut self, _dialect: &spv::Dialect) {}
    fn visit_spv_module_debug_info(&mut self, _debug_info: &spv::ModuleDebugInfo) {}

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
    fn visit_global_var_def(&mut self, gv_def: &'a GlobalVarDef) {
        gv_def.inner_visit_with(self);
    }
    fn visit_func(&mut self, func: &'a Func) {
        func.inner_visit_with(self);
    }
    fn visit_misc(&mut self, misc: &'a Misc) {
        misc.inner_visit_with(self);
    }
    fn visit_misc_output(&mut self, output: &'a MiscOutput) {
        output.inner_visit_with(self);
    }
    fn visit_misc_input(&mut self, input: &'a MiscInput) {
        input.inner_visit_with(self);
    }
}

/// Trait implemented on "visitable" types, to further "explore" a type by
/// visiting its "interior" (i.e. variants and/or fields).
///
/// That is, an `impl InnerVisit for X` will call the relevant `Visitor` method
/// for each `X` field, effectively performing a single level of a deep visit.
/// Also, if `Visitor::visit_X` exists for a given `X`, its default should be to
/// call `X::inner_visit_with` (i.e. so that visiting is mostly-deep by default).
pub trait InnerVisit {
    // FIXME(eddyb) the naming here isn't great, can it be improved?
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>);
}

/// Dynamic dispatch version of `InnerVisit`.
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
        let Self {
            dialect,
            debug_info,
            global_vars: _,
            funcs,
            ..
        } = self;

        visitor.visit_module_dialect(dialect);
        visitor.visit_module_debug_info(debug_info);
        for func in funcs {
            visitor.visit_func(func);
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
            Self::SpvEntryPoint {
                params: _,
                interface_global_vars,
            } => {
                for &gv in &interface_global_vars.0 {
                    visitor.visit_global_var_use(gv);
                }
            }
            Self::SpvAnnotation { .. } | Self::SpvDebugLine { .. } => {}
        }
    }
}

impl InnerVisit for TypeDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            ctor,
            ctor_args,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        match ctor {
            TypeCtor::SpvInst(_) => {}
        }
        for &arg in ctor_args {
            match arg {
                TypeCtorArg::Type(ty) => visitor.visit_type_use(ty),
                TypeCtorArg::Const(ct) => visitor.visit_const_use(ct),

                TypeCtorArg::SpvImm(_) => {}
            }
        }
    }
}

impl InnerVisit for ConstDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            ty,
            ctor,
            ctor_args,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*ty);
        match *ctor {
            ConstCtor::SpvInst(_) => {}
            ConstCtor::PtrToGlobalVar(gv) => visitor.visit_global_var_use(gv),
        }
        for &arg in ctor_args {
            match arg {
                ConstCtorArg::Const(ct) => visitor.visit_const_use(ct),

                ConstCtorArg::SpvImm(_) => {}
            }
        }
    }
}

impl InnerVisit for GlobalVarDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            type_of_ptr_to,
            addr_space,
            initializer,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*type_of_ptr_to);
        match addr_space {
            AddrSpace::SpvStorageClass(_) => {}
        }
        if let Some(initializer) = *initializer {
            visitor.visit_const_use(initializer);
        }
    }
}

impl InnerVisit for Func {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { insts } = self;

        for inst in insts {
            visitor.visit_misc(inst);
        }
    }
}

impl InnerVisit for Misc {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            kind,
            output,
            inputs,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        match kind {
            MiscKind::SpvInst(_) => {}
        }
        if let Some(output) = output {
            visitor.visit_misc_output(output);
        }
        for input in inputs {
            visitor.visit_misc_input(input);
        }
    }
}

impl InnerVisit for MiscOutput {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match *self {
            Self::SpvResult {
                result_type,
                result_id: _,
            } => {
                if let Some(ty) = result_type {
                    visitor.visit_type_use(ty);
                }
            }
        }
    }
}

impl InnerVisit for MiscInput {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match *self {
            Self::Type(ty) => visitor.visit_type_use(ty),
            Self::Const(ct) => visitor.visit_const_use(ct),

            Self::SpvImm(_) | Self::SpvUntrackedId(_) | Self::SpvExtInstImport(_) => {}
        }
    }
}
