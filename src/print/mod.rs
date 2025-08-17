//! Pretty-printing anything in the IR, from whole [`Module`]s to their leaves.
//!
//! # Usage
//!
//! To start, create a [`Plan`] (through e.g. [`Plan::for_root`] or [`Plan::for_module`]),
//! which will track the entire (transitive) set of (interned/entity) dependencies
//! required to produce complete pretty-printing outputs.
//!
//! On a [`Plan`], use [`.pretty_print()`](Plan::pretty_print) to print everything,
//! and get a "pretty document", with layout (inline-vs-multi-line decisions,
//! auto-indentation, etc.) already performed, and which supports outputting:
//! * plain text: `fmt::Display` (`{}` formatting) or `.to_string()`
//! * HTML (styled and hyperlinked): [`.render_to_html()`](Versions::render_to_html)
#![allow(rustdoc::private_intra_doc_links)]
//!   (returning a [`pretty::HtmlSnippet`])

// FIXME(eddyb) stop using `itertools` for methods like `intersperse` when they
// get stabilized on `Iterator` instead.
#![allow(unstable_name_collisions)]
use itertools::Itertools as _;

use crate::func_at::FuncAt;
use crate::print::multiversion::Versions;
use crate::qptr::{self, QPtrAttr, QPtrMemUsage, QPtrMemUsageKind, QPtrOp, QPtrUsage};
use crate::visit::{InnerVisit, Visit, Visitor};
use crate::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstKind, DbgSrcLoc, DeclDef, Diag, DiagLevel, DiagMsgPart,
    EntityOrientedDenseMap, ExportKey, Exportee, Func, FuncDecl, FuncDefBody, FuncParam,
    FxIndexMap, FxIndexSet, GlobalVar, GlobalVarDecl, GlobalVarDefBody, Import, InternedStr,
    Module, ModuleDebugInfo, ModuleDialect, Node, NodeDef, NodeKind, NodeOutputDecl, OrdAssertEq,
    Region, RegionDef, RegionInputDecl, SelectionKind, Type, TypeDef, TypeKind, TypeOrConst, Value,
    cfg, spv,
};
use arrayvec::ArrayVec;
use itertools::Either;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::fmt::{self, Write as _};
use std::hash::Hash;
use std::{iter, mem};

mod multiversion;
mod pretty;

/// "Definitions-before-uses" / "topo-sorted" printing plan.
///
/// In order to represent parts of a DAG textually, it first needs to have its
/// nodes "flattened" into an order (also known as "topo(logical) sorting"),
/// which [`Plan`] records (as [`PlanItem`]s), before any printing can commence.
///
/// Additionally, nodes without a significant identity (i.e. interned ones) may
/// have their separate definition omitted in some cases where printing them
/// inline at their use site(s) is preferred (e.g. when they have a single use).
///
/// Once a [`Plan`] contains everything that needs to be printed, calling the
/// [`.pretty_print()`](Plan::pretty_print) method will print all of the [`PlanItem`]s
/// in the [`Plan`], and its return value can be e.g. formatted with [`fmt::Display`].
pub struct Plan<'a> {
    cx: &'a Context,

    /// When visiting module-stored [`PlanItem`]s, the [`Module`] is needed to map
    /// the [`PlanItem`] to the (per-version) definition, which is then stored in
    /// the (per-version) `item_defs` map.
    current_module: Option<&'a Module>,

    /// Versions allow comparing multiple copies of the same e.g. [`Module`],
    /// with definitions sharing a [`PlanItem`] key being shown together.
    ///
    /// Specific [`PlanItem`]s may be present in only a subset of versions, and such
    /// a distinction will be reflected in the output.
    ///
    /// For [`PlanItem`] collection, `versions.last()` constitutes the "active" one.
    versions: Vec<PlanVersion<'a>>,

    /// Merged per-[`Use`] counts across all versions.
    ///
    /// That is, each [`Use`] maps to the largest count of that [`Use`] in any version,
    /// as opposed to their sum. This approach avoids pessimizing e.g. inline
    /// printing of interned definitions, which may need the use count to be `1`.
    use_counts: FxIndexMap<Use, usize>,

    /// Merged per-[`AttrSet`] unique SPIR-V `OpName`s across all versions.
    ///
    /// That is, each [`AttrSet`] maps to one of the SPIR-V `OpName`s contained
    /// within the [`AttrSet`], as long as these three conditions are met:
    /// * in each version using an [`AttrSet`], there is only one use of it
    ///   (e.g. different [`Type`]s/[`Const`]s/etc. can't use the same [`AttrSet`])
    /// * in each version using an [`AttrSet`], no other [`AttrSet`]s are used
    ///   that "claim" the same SPIR-V `OpName`
    /// * an [`AttrSet`] "claims" the same SPIR-V `OpName` across all versions
    ///   using it (with per-version "claiming", only merged after the fact)
    ///
    /// Note that these conditions still allow the same SPIR-V `OpName` being
    /// "claimed" by different [`AttrSet`]s, *as long as* they only show up in
    /// *disjoint* versions (e.g. [`GlobalVarDecl`] attributes being modified
    /// between versions, but keeping the same `OpName` attribute unchanged).
    attrs_to_unique_spv_name: FxHashMap<AttrSet, Result<&'a spv::Inst, AmbiguousName>>,

    /// Reverse map of `attrs_to_unique_spv_name_imms`, only used during visiting
    /// (i.e. intra-version), to track which SPIR-V `OpName`s have been "claimed"
    /// by some [`AttrSet`] and detect conflicts (which are resolved by marking
    /// *both* overlapping [`AttrSet`], *and* the `OpName` itself, as ambiguous).
    claimed_spv_names: FxHashMap<&'a spv::Inst, Result<AttrSet, AmbiguousName>>,
}

/// One version of a multi-version [`Plan`] (see also its `versions` field),
/// or the sole one (in the single-version mode).
struct PlanVersion<'a> {
    /// Descriptive name for this version (e.g. the name of a pass that produced it),
    /// or left empty (in the single-version mode).
    name: String,

    /// Definitions for all the [`PlanItem`]s which may need to be printed later
    /// (with the exception of [`PlanItem::AllCxInterned`], which is special-cased).
    item_defs: FxHashMap<PlanItem, PlanItemDef<'a>>,

    /// Either a whole [`Module`], or some other printable type passed to
    /// [`Plan::for_root`]/[`Plan::for_versions`], which gets printed last,
    /// after all of its dependencies (making up the rest of the [`Plan`]).
    root: &'a dyn Print<Output = pretty::Fragment>,
}

/// Error type used when tracking `OpName` uniqueness.
#[derive(Copy, Clone)]
struct AmbiguousName;

/// Print [`Plan`] top-level entry, an effective reification of SPIR-T's implicit DAG.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum PlanItem {
    /// Definitions for all [`CxInterned`] that need them, grouped together.
    AllCxInterned,

    // FIXME(eddyb) these do not support multiple `Module`s as they don't have
    // any way to distinguish between instances of them from different `Module`s.
    ModuleDialect,
    ModuleDebugInfo,

    GlobalVar(GlobalVar),
    Func(Func),
}

impl PlanItem {
    fn keyword_and_name_prefix(self) -> Result<(&'static str, &'static str), &'static str> {
        match self {
            Self::AllCxInterned => Err("PlanItem::AllCxInterned"),

            // FIXME(eddyb) these don't have the same kind of `{name_prefix}{idx}`
            // formatting, so maybe they don't belong in here to begin with?
            Self::ModuleDialect => Ok(("module.dialect", "")),
            Self::ModuleDebugInfo => Ok(("module.debug_info", "")),

            Self::GlobalVar(_) => Ok(("global_var", "GV")),
            Self::Func(_) => Ok(("func", "F")),
        }
    }
}

/// Definition of a [`PlanItem`] (i.e. a reference pointing into a [`Module`]).
///
/// Note: [`PlanItem::AllCxInterned`] does *not* have its own `PlanItemDef` variant,
/// as it *must* be specially handled instead.
#[derive(Copy, Clone, derive_more::From)]
enum PlanItemDef<'a> {
    ModuleDialect(&'a ModuleDialect),
    ModuleDebugInfo(&'a ModuleDebugInfo),
    GlobalVar(&'a GlobalVarDecl),
    Func(&'a FuncDecl),
}

/// Anything interned in [`Context`], that might need to be printed once
/// (as part of [`PlanItem::AllCxInterned`]) and referenced multiple times.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CxInterned {
    Type(Type),
    Const(Const),
}

impl CxInterned {
    fn keyword_and_name_prefix(self) -> (&'static str, &'static str) {
        match self {
            Self::Type(_) => ("type", "T"),
            Self::Const(_) => ("const", "C"),
        }
    }
}

// FIXME(eddyb) should `DbgSrcLoc` have a "parent scope" field like this?
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum DbgScope {
    // HACK(eddyb) `DbgSrcLoc`'s `inlined_callee_name_and_call_site`'s field.
    InlinedCalleeBody { callee_name: InternedStr, call_site: AttrSet },
}

impl DbgScope {
    fn try_from_attrs(cx: &Context, attrs: AttrSet) -> Option<Self> {
        DbgScope::try_from_dbg_src_loc(attrs.dbg_src_loc(cx)?)
    }

    fn try_from_dbg_src_loc(dbg_src_loc: DbgSrcLoc) -> Option<Self> {
        let DbgSrcLoc {
            file_path: _,
            start_line_col: _,
            end_line_col: _,
            inlined_callee_name_and_call_site,
        } = dbg_src_loc;

        let (callee_name, call_site) = inlined_callee_name_and_call_site?;

        Some(DbgScope::InlinedCalleeBody { callee_name, call_site })
    }

    fn parent(&self, cx: &Context) -> Option<Self> {
        match *self {
            DbgScope::InlinedCalleeBody { callee_name: _, call_site } => {
                DbgScope::try_from_attrs(cx, call_site)
            }
        }
    }

    fn parents<'a>(&self, cx: &'a Context) -> impl Iterator<Item = Self> + 'a {
        let mut scope = Some(*self);
        iter::from_fn(move || {
            scope = scope?.parent(cx);
            scope
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Use {
    PlanItem(PlanItem),

    CxInterned(CxInterned),

    DbgScope {
        scope: DbgScope,

        // HACK(eddyb) only needed because `DbgScope` itself can appear in many
        // functions with identical values (e.g. due to monomorphization), and
        // it lacks the kind of "globally unique" identity of `Region`/`Node`.
        parent_func: Func,
    },

    RegionLabel(Region),

    // FIXME(eddyb) these are `Value`'s variants except `Const`, maybe `Use`
    // should just use `Value` and assert it's never `Const`?
    RegionInput {
        region: Region,
        input_idx: u32,
    },
    NodeOutput {
        node: Node,
        output_idx: u32,
    },
    DataInstOutput(DataInst),

    // NOTE(eddyb) these overlap somewhat with other cases, but they're always
    // generated, even when there is no "use", for `multiversion` alignment.
    AlignmentAnchorForRegion(Region),
    AlignmentAnchorForNode(Node),
    AlignmentAnchorForDataInst(DataInst),
}

impl From<Value> for Use {
    fn from(value: Value) -> Self {
        match value {
            Value::Const(ct) => Use::CxInterned(CxInterned::Const(ct)),
            Value::RegionInput { region, input_idx } => Use::RegionInput { region, input_idx },
            Value::NodeOutput { node, output_idx } => Use::NodeOutput { node, output_idx },
            Value::DataInstOutput(inst) => Use::DataInstOutput(inst),
        }
    }
}

impl Use {
    // HACK(eddyb) this is used in `AttrsAndDef::insert_name_before_def` to
    // detect alignment anchors specifically, so it needs to not overlap with
    // any other name (including those containing escaped `OpName` strings).
    const ANCHOR_ALIGNMENT_NAME_PREFIX: &'static str = "AA.";

    fn keyword_and_name_prefix(self) -> (&'static str, &'static str) {
        match self {
            Self::PlanItem(item) => item.keyword_and_name_prefix().unwrap(),
            Self::CxInterned(interned) => interned.keyword_and_name_prefix(),

            Self::DbgScope { .. } => ("", "d"),
            Self::RegionLabel(_) => ("label", "L"),

            Self::RegionInput { .. } | Self::NodeOutput { .. } | Self::DataInstOutput(_) => {
                ("", "v")
            }

            Self::AlignmentAnchorForRegion(_)
            | Self::AlignmentAnchorForNode(_)
            | Self::AlignmentAnchorForDataInst(_) => ("", Self::ANCHOR_ALIGNMENT_NAME_PREFIX),
        }
    }
}

impl<'a> Plan<'a> {
    /// Create a [`Plan`] with all of `root`'s dependencies, followed by `root` itself.
    //
    // FIXME(eddyb) consider renaming this and removing the `for_module` shorthand.
    pub fn for_root(
        cx: &'a Context,
        root: &'a (impl Visit + Print<Output = pretty::Fragment>),
    ) -> Self {
        Self::for_versions(cx, [("", root)])
    }

    /// Create a [`Plan`] with all of `module`'s contents.
    ///
    /// Shorthand for `Plan::for_root(module.cx_ref(), module)`.
    pub fn for_module(module: &'a Module) -> Self {
        Self::for_root(module.cx_ref(), module)
    }

    /// Create a [`Plan`] that combines [`Plan::for_root`] from each version.
    ///
    /// Each version also has a string, which should contain a descriptive name
    /// (e.g. the name of a pass that produced that version).
    ///
    /// While the roots (and their dependencies) can be entirely unrelated, the
    /// output won't be very useful in that case. For ideal results, most of the
    /// same entities (e.g. [`GlobalVar`] or [`Func`]) should be in most versions,
    /// with most of the changes being limited to within their definitions.
    pub fn for_versions(
        cx: &'a Context,
        versions: impl IntoIterator<
            Item = (impl Into<String>, &'a (impl Visit + Print<Output = pretty::Fragment> + 'a)),
        >,
    ) -> Self {
        let mut plan = Self {
            cx,
            current_module: None,
            versions: vec![],
            use_counts: FxIndexMap::default(),
            attrs_to_unique_spv_name: FxHashMap::default(),
            claimed_spv_names: FxHashMap::default(),
        };
        for (version_name, version_root) in versions {
            let mut combined_use_counts = mem::take(&mut plan.use_counts);
            let mut combined_attrs_to_unique_spv_name =
                mem::take(&mut plan.attrs_to_unique_spv_name);
            plan.claimed_spv_names.clear();

            plan.versions.push(PlanVersion {
                name: version_name.into(),
                item_defs: FxHashMap::default(),
                root: version_root,
            });

            version_root.visit_with(&mut plan);

            // Merge use counts (from second version onward).
            if !combined_use_counts.is_empty() {
                for (use_kind, new_count) in plan.use_counts.drain(..) {
                    let count = combined_use_counts.entry(use_kind).or_default();
                    *count = new_count.max(*count);
                }
                plan.use_counts = combined_use_counts;
            }

            // Merge per-`AttrSet` unique `OpName`s (from second version onward).
            if !combined_attrs_to_unique_spv_name.is_empty() {
                for (attrs, unique_spv_name) in plan.attrs_to_unique_spv_name.drain() {
                    let combined =
                        combined_attrs_to_unique_spv_name.entry(attrs).or_insert(unique_spv_name);

                    *combined = match (*combined, unique_spv_name) {
                        // HACK(eddyb) can use pointer comparison because both
                        // `OpName`s are in the same `AttrSetDef`'s `BTreeSet`,
                        // so they can only be equal in the same set slot.
                        (Ok(combined), Ok(new)) if std::ptr::eq(combined, new) => Ok(combined),

                        _ => Err(AmbiguousName),
                    };
                }
                plan.attrs_to_unique_spv_name = combined_attrs_to_unique_spv_name;
            }
        }

        // HACK(eddyb) release memory used by this map (and avoid later misuse).
        mem::take(&mut plan.claimed_spv_names);

        plan
    }

    /// Add `interned` to the plan, after all of its dependencies.
    ///
    /// Only the first call recurses into the definition, subsequent calls only
    /// update its (internally tracked) "use count".
    fn use_interned(&mut self, interned: CxInterned) {
        let use_kind = Use::CxInterned(interned);
        if let Some(use_count) = self.use_counts.get_mut(&use_kind) {
            *use_count += 1;
            return;
        }

        match interned {
            CxInterned::Type(ty) => {
                self.visit_type_def(&self.cx[ty]);
            }
            CxInterned::Const(ct) => {
                self.visit_const_def(&self.cx[ct]);
            }
        }

        // Group all `CxInterned`s in a single top-level `PlanItem`.
        *self.use_counts.entry(Use::PlanItem(PlanItem::AllCxInterned)).or_default() += 1;

        *self.use_counts.entry(use_kind).or_default() += 1;
    }

    /// Add `item` to the plan, after all of its dependencies.
    ///
    /// Only the first call recurses into the definition, subsequent calls only
    /// update its (internally tracked) "use count".
    fn use_item<D: Visit>(&mut self, item: PlanItem, item_def: &'a D)
    where
        PlanItemDef<'a>: From<&'a D>,
    {
        if let Some(use_count) = self.use_counts.get_mut(&Use::PlanItem(item)) {
            *use_count += 1;
            return;
        }

        let current_version = self.versions.last_mut().unwrap();
        match current_version.item_defs.entry(item) {
            Entry::Occupied(entry) => {
                let old_ptr_eq_new = match (*entry.get(), PlanItemDef::from(item_def)) {
                    (PlanItemDef::ModuleDialect(old), PlanItemDef::ModuleDialect(new)) => {
                        std::ptr::eq(old, new)
                    }
                    (PlanItemDef::ModuleDebugInfo(old), PlanItemDef::ModuleDebugInfo(new)) => {
                        std::ptr::eq(old, new)
                    }
                    (PlanItemDef::GlobalVar(old), PlanItemDef::GlobalVar(new)) => {
                        std::ptr::eq(old, new)
                    }
                    (PlanItemDef::Func(old), PlanItemDef::Func(new)) => std::ptr::eq(old, new),
                    _ => false,
                };

                // HACK(eddyb) this avoids infinite recursion - we can't insert
                // into `use_counts` before `item_def.visit_with(self)` because
                // we want dependencies to come before dependends, so recursion
                // from the visitor (recursive `Func`s, or `visit_foo` calling
                // `use_item` which calls the same `visit_foo` method again)
                // ends up here, and we have to both allow it and early-return.
                assert!(
                    old_ptr_eq_new,
                    "print: same `{}` item has multiple distinct definitions in `Plan`",
                    item.keyword_and_name_prefix().map_or_else(|s| s, |(_, s)| s)
                );
                return;
            }
            Entry::Vacant(entry) => {
                entry.insert(PlanItemDef::from(item_def));
            }
        }

        item_def.visit_with(self);

        *self.use_counts.entry(Use::PlanItem(item)).or_default() += 1;
    }
}

impl<'a> Visitor<'a> for Plan<'a> {
    fn visit_attr_set_use(&mut self, attrs: AttrSet) {
        let wk = &spv::spec::Spec::get().well_known;

        let attrs_def = &self.cx[attrs];
        self.visit_attr_set_def(attrs_def);

        // Try to claim a SPIR-V `OpName`, if any are present in `attrs`.
        let mut spv_names = attrs_def
            .attrs
            .iter()
            .filter_map(|attr| match attr {
                Attr::SpvAnnotation(spv_inst) if spv_inst.opcode == wk.OpName => Some(spv_inst),
                _ => None,
            })
            .peekable();
        if let Some(existing_entry) = self.attrs_to_unique_spv_name.get_mut(&attrs) {
            // Same `attrs` seen more than once (from different definitions).
            *existing_entry = Err(AmbiguousName);
        } else if let Some(&first_spv_name) = spv_names.peek() {
            let mut result = Ok(first_spv_name);

            // HACK(eddyb) claim all SPIR-V `OpName`s in `attrs`, even if we'll
            // use only one - this guarantees any overlaps will be detected, and
            // while that may be overly strict, it's also the only easy way to
            // have a completely order-indepdendent name choices.
            for spv_name in spv_names {
                let claim = self.claimed_spv_names.entry(spv_name).or_insert(Ok(attrs));

                if let Ok(claimant) = *claim {
                    if claimant == attrs {
                        // Only possible way to succeed is if we just made the claim.
                        continue;
                    }

                    // Invalidate the old user of this SPIR-V `OpName`.
                    self.attrs_to_unique_spv_name.insert(claimant, Err(AmbiguousName));
                }

                // Either we just found a conflict, or one already existed.
                *claim = Err(AmbiguousName);
                result = Err(AmbiguousName);
            }

            self.attrs_to_unique_spv_name.insert(attrs, result);
        }
    }
    fn visit_type_use(&mut self, ty: Type) {
        self.use_interned(CxInterned::Type(ty));
    }
    fn visit_const_use(&mut self, ct: Const) {
        self.use_interned(CxInterned::Const(ct));
    }

    fn visit_global_var_use(&mut self, gv: GlobalVar) {
        if let Some(module) = self.current_module {
            self.use_item(PlanItem::GlobalVar(gv), &module.global_vars[gv]);
        } else {
            // FIXME(eddyb) should this be a hard error?
        }
    }

    fn visit_func_use(&mut self, func: Func) {
        if let Some(module) = self.current_module {
            self.use_item(PlanItem::Func(func), &module.funcs[func]);
        } else {
            // FIXME(eddyb) should this be a hard error?
        }
    }

    fn visit_module(&mut self, module: &'a Module) {
        assert!(
            std::ptr::eq(self.cx, &**module.cx_ref()),
            "print: `Plan::visit_module` does not support `Module`s from a \
             different `Context` than the one it was initially created with",
        );

        let old_module = self.current_module.replace(module);
        module.inner_visit_with(self);
        self.current_module = old_module;
    }
    fn visit_module_dialect(&mut self, dialect: &'a ModuleDialect) {
        self.use_item(PlanItem::ModuleDialect, dialect);
    }
    fn visit_module_debug_info(&mut self, debug_info: &'a ModuleDebugInfo) {
        self.use_item(PlanItem::ModuleDebugInfo, debug_info);
    }

    fn visit_attr(&mut self, attr: &'a Attr) {
        attr.inner_visit_with(self);

        // HACK(eddyb) the interpolated parts aren't visited by default
        // (as they're "inert data").
        if let Attr::Diagnostics(OrdAssertEq(diags)) = attr {
            for diag in diags {
                let Diag { level, message } = diag;
                match level {
                    DiagLevel::Bug(_) | DiagLevel::Error | DiagLevel::Warning => {}
                }
                message.inner_visit_with(self);
            }
        }
    }

    fn visit_const_def(&mut self, ct_def: &'a ConstDef) {
        // HACK(eddyb) the type of a `PtrToGlobalVar` is never printed, skip it.
        if let ConstKind::PtrToGlobalVar(gv) = ct_def.kind {
            self.visit_attr_set_use(ct_def.attrs);
            self.visit_global_var_use(gv);
        } else {
            ct_def.inner_visit_with(self);
        }
    }

    fn visit_global_var_decl(&mut self, gv_decl: &'a GlobalVarDecl) {
        // HACK(eddyb) get the pointee type from SPIR-V `OpTypePointer`, but
        // ideally the `GlobalVarDecl` would hold that type itself.
        let pointee_type = {
            let wk = &spv::spec::Spec::get().well_known;

            match &self.cx[gv_decl.type_of_ptr_to].kind {
                TypeKind::SpvInst { spv_inst, type_and_const_inputs }
                    if spv_inst.opcode == wk.OpTypePointer =>
                {
                    match type_and_const_inputs[..] {
                        [TypeOrConst::Type(ty)] => Some(ty),
                        _ => unreachable!(),
                    }
                }
                _ => None,
            }
        };

        // HACK(eddyb) if we can get away without visiting the `OpTypePointer`
        // `type_of_ptr_to`, but only its pointee type, do so to avoid spurious
        // `OpTypePointer` types showing up in the pretty-printed output.
        match (gv_decl, pointee_type) {
            (
                GlobalVarDecl {
                    attrs,
                    type_of_ptr_to: _,
                    shape: None,
                    addr_space: AddrSpace::SpvStorageClass(_),
                    def,
                },
                Some(pointee_type),
            ) => {
                self.visit_attr_set_use(*attrs);
                self.visit_type_use(pointee_type);
                def.inner_visit_with(self);
            }

            _ => {
                gv_decl.inner_visit_with(self);
            }
        }
    }

    fn visit_func_decl(&mut self, func_decl: &'a FuncDecl) {
        if let DeclDef::Present(func_def_body) = &func_decl.def
            && let Some(cfg) = &func_def_body.unstructured_cfg
        {
            for region in cfg.rev_post_order(func_def_body) {
                if let Some(control_inst) = cfg.control_inst_on_exit_from.get(region) {
                    for &target in &control_inst.targets {
                        *self.use_counts.entry(Use::RegionLabel(target)).or_default() += 1;
                    }
                }
            }
        }

        func_decl.inner_visit_with(self);
    }

    fn visit_value_use(&mut self, v: &'a Value) {
        match *v {
            Value::Const(_) => {}
            _ => *self.use_counts.entry(Use::from(*v)).or_default() += 1,
        }
        v.inner_visit_with(self);
    }
}

// FIXME(eddyb) make max line width configurable.
const MAX_LINE_WIDTH: usize = 120;

impl Plan<'_> {
    #[allow(rustdoc::private_intra_doc_links)]
    /// Print the whole [`Plan`] to a [`Versions<pretty::Fragment>`] and perform
    /// layout on its [`pretty::Fragment`]s.
    ///
    /// The resulting [`Versions<pretty::FragmentPostLayout>`] value supports
    /// [`fmt::Display`] for convenience, but also more specific methods
    /// (e.g. HTML output).
    pub fn pretty_print(&self) -> Versions<pretty::FragmentPostLayout> {
        self.print(&Printer::new(self))
            .map_pretty_fragments(|fragment| fragment.layout_with_max_line_width(MAX_LINE_WIDTH))
    }

    /// Like `pretty_print`, but separately pretty-printing "root dependencies"
    /// and the "root" itself (useful for nesting pretty-printed SPIR-T elsewhere).
    pub fn pretty_print_deps_and_root_separately(
        &self,
    ) -> (Versions<pretty::FragmentPostLayout>, Versions<pretty::FragmentPostLayout>) {
        let printer = Printer::new(self);
        (
            self.print_all_items_and_or_root(&printer, true, false).map_pretty_fragments(
                |fragment| fragment.layout_with_max_line_width(MAX_LINE_WIDTH),
            ),
            self.print_all_items_and_or_root(&printer, false, true).map_pretty_fragments(
                |fragment| fragment.layout_with_max_line_width(MAX_LINE_WIDTH),
            ),
        )
    }
}

/// Inside-out reverse traversal, placing debug scopes' definitions,
/// such that each debug scope's definition is always found:
/// - in the innermost `Region` which contains all uses of it
/// - just before the first `Node` which contains any uses of it
//
// FIXME(eddyb) try to turn this into debug scopes structurally "wrapping" the
// ranges of nodes "inside" it, but that's an a harder "aesthetic optimization",
// and it would still need an approach like this, as a general fallback, anyway.
// FIXME(eddyb) consider interning the `DbgScope`s, even if locally.
struct DbgScopeDefPlacer<'a> {
    cx: &'a Context,
}

#[derive(Copy, Clone)]
struct DbgScopeDefPlace {
    parent_region: Region,

    intra_region: DbgScopeDefPlaceInRegion,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct DbgScopeDefPlaceInRegion {
    // HACK(eddyb) `Option` only needed due to potentially empty regions.
    before_node: Option<Node>,
}

impl DbgScopeDefPlace {
    fn top_of(func_at_region: FuncAt<'_, Region>) -> Self {
        DbgScopeDefPlace {
            parent_region: func_at_region.position,
            intra_region: DbgScopeDefPlaceInRegion {
                before_node: func_at_region
                    .at_children()
                    .into_iter()
                    .next()
                    .map(|func_at_node| func_at_node.position),
            },
        }
    }
}

struct DbgScopeDefMap {
    // HACK(eddyb) a place that "dominates" (is earlier than, and
    // not nested any further than) all of the ones used in `scopes`.
    top: DbgScopeDefPlace,

    // HACK(eddyb) to simplify both collection and usage, the insertion order
    // (that `FxIndexMap` preserves) is reversed from printing order.
    rev_scopes: FxIndexMap<DbgScope, DbgScopeDefPlace>,
}

// FIXME(eddyb) do these methods need better names?
impl DbgScopeDefMap {
    fn merge_ordered(
        top: DbgScopeDefPlace,
        scope_maps: impl DoubleEndedIterator<Item = Self>,
    ) -> Self {
        let rev_scopes = scope_maps
            .rev()
            .reduce(|old_tail, new_head| {
                if old_tail.rev_scopes.is_empty() {
                    return new_head;
                }

                let mut merged =
                    DbgScopeDefMap { top: new_head.top, rev_scopes: old_tail.rev_scopes };
                for (new_scope, new_place) in new_head.rev_scopes {
                    merged
                        .rev_scopes
                        .entry(new_scope)
                        .and_modify(|merged_place| {
                            *merged_place = merged.top;
                        })
                        .or_insert(new_place);
                }
                merged
            })
            .map(|merged| merged.rev_scopes)
            .unwrap_or_default();
        DbgScopeDefMap { top, rev_scopes }
    }

    fn merge_unordered(
        common_top: DbgScopeDefPlace,
        scope_maps: impl DoubleEndedIterator<Item = Self>,
    ) -> Self {
        DbgScopeDefMap::merge_ordered(
            common_top,
            scope_maps.map(|mut scope_map| {
                scope_map.top = common_top;
                scope_map
            }),
        )
    }

    fn prepend_attrs(&mut self, cx: &Context, attrs: AttrSet) {
        if let Some(scope) = DbgScope::try_from_attrs(cx, attrs) {
            for scope in [scope].into_iter().chain(scope.parents(cx)) {
                use indexmap::map::Entry;

                match self.rev_scopes.entry(scope) {
                    Entry::Occupied(entry) => {
                        *entry.into_mut() = self.top;
                        break;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(self.top);
                    }
                }
            }
        }
    }
}

impl DbgScopeDefPlacer<'_> {
    fn scopes_used_in_whole_func(&mut self, func_def_body: &FuncDefBody) -> DbgScopeDefMap {
        match &func_def_body.unstructured_cfg {
            None => self.scopes_used_in_region(func_def_body.at_body()),
            Some(cfg) => DbgScopeDefMap::merge_unordered(
                DbgScopeDefPlace::top_of(func_def_body.at_body()),
                cfg.rev_post_order(func_def_body).map(|region| {
                    let mut scopes = self.scopes_used_in_region(func_def_body.at(region));
                    if let Some(control_inst) = cfg.control_inst_on_exit_from.get(region) {
                        scopes.prepend_attrs(self.cx, control_inst.attrs);
                    }
                    scopes
                }),
            ),
        }
    }

    // FIXME(eddyb) use some in-place tricks (a la `SnapshotVec`) to make
    // this cheaper than allocating and returning `DbgScopeDefMap`s.
    fn scopes_used_in_region(&mut self, func_at_region: FuncAt<'_, Region>) -> DbgScopeDefMap {
        // HACK(eddyb) helper used to avoid hardcoding `Node` details (see below).
        struct ShallowVisitAttrsAndRegions<VA, VR> {
            visit_attrs: VA,
            visit_region: VR,
        }
        impl<'a, VA: FnMut(AttrSet), VR: FnMut(Region)> Visitor<'a>
            for ShallowVisitAttrsAndRegions<VA, VR>
        {
            // FIXME(eddyb) this is excessive, maybe different kinds of
            // visitors should exist for module-level and func-level?
            fn visit_type_use(&mut self, _: Type) {}
            fn visit_const_use(&mut self, _: Const) {}
            fn visit_global_var_use(&mut self, _: GlobalVar) {}
            fn visit_func_use(&mut self, _: Func) {}

            fn visit_attr_set_use(&mut self, attrs: AttrSet) {
                (self.visit_attrs)(attrs);
            }
            fn visit_region_def(&mut self, func_at_region: FuncAt<'a, Region>) {
                (self.visit_region)(func_at_region.position);
            }
        }

        DbgScopeDefMap::merge_ordered(
            DbgScopeDefPlace::top_of(func_at_region),
            func_at_region.at_children().into_iter().map(|func_at_node| {
                // HACK(eddyb) working around pre-unification non-uniform `Node`
                // details by collecting the relevant fields with a `Visitor`.
                let mut child_regions = SmallVec::<[_; 4]>::new();
                ShallowVisitAttrsAndRegions {
                    visit_attrs: |_| {},
                    visit_region: |region| child_regions.push(region),
                }
                .visit_node_def(func_at_node);

                let mut map = DbgScopeDefMap::merge_unordered(
                    DbgScopeDefPlace {
                        parent_region: func_at_region.position,
                        intra_region: DbgScopeDefPlaceInRegion {
                            before_node: Some(func_at_node.position),
                        },
                    },
                    child_regions.into_iter().map(|child_region| {
                        self.scopes_used_in_region(func_at_node.at(child_region))
                    }),
                );

                ShallowVisitAttrsAndRegions {
                    visit_attrs: |attrs| {
                        map.prepend_attrs(self.cx, attrs);
                    },
                    visit_region: |_| {},
                }
                .visit_node_def(func_at_node);

                map
            }),
        )
    }
}

pub struct Printer<'a> {
    cx: &'a Context,
    use_styles: FxIndexMap<Use, UseStyle>,

    /// Subset of the `Plan`'s original `attrs_to_unique_spv_name` map, only
    /// containing those entries which are actively used for `UseStyle::Named`
    /// values in `use_styles`, and therefore need to be hidden from attributes.
    attrs_with_spv_name_in_use: FxHashMap<AttrSet, &'a spv::Inst>,

    /// Map from each `Region` to the `DbgScope`s placed in it in any version,
    /// indexed by `DbgScopeDefPlaceInRegion` (e.g. "just before some `Node`").
    ///
    /// **Note**: due to different versions potentially placing one `DbgScope`
    /// before two (or more) different `Node`s, which might not even have any
    /// theoretical global ordering (i.e. due to removed/reordered `Node`s),
    /// each `Region` must explicitly ensure each `DbgScope` is printed once.
    //
    // FIXME(eddyb) is `Use::DbgScope` even needed at all, with this?
    per_region_dbg_scope_def_placements: EntityOrientedDenseMap<
        Region,
        FxHashMap<DbgScopeDefPlaceInRegion, SmallIndexSet<DbgScope, 2>>,
    >,

    // HACK(eddyb) only used for `DbgScope` printing, but could/should be used
    // for more situations (and maybe include the version index, as well).
    current_plan_item: Cell<Option<PlanItem>>,
}

/// How an [`Use`] of a definition should be printed.
enum UseStyle {
    /// Refer to the definition by its name prefix and an `idx` (e.g. "T123").
    Anon {
        /// For intra-function [`Use`]s (i.e. [`Use::RegionLabel`] and values),
        /// this disambiguates the parent function (for e.g. anchors).
        parent_func: Option<Func>,

        idx: usize,
    },

    /// Refer to the definition by its name prefix and a `name` (e.g. "T`Foo`").
    Named {
        /// For intra-function [`Use`]s (i.e. [`Use::RegionLabel`] and values),
        /// this disambiguates the parent function (for e.g. anchors).
        parent_func: Option<Func>,

        name: String,
    },

    /// Print the definition inline at the use site.
    Inline,
}

// HACK(eddyb) move this elsewhere.
enum SmallIndexSet<T, const N: usize> {
    Linear(ArrayVec<T, N>),
    Hashed(Box<FxIndexSet<T>>),
}

type SmallSetIter<'a, T> = Either<std::slice::Iter<'a, T>, indexmap::set::Iter<'a, T>>;

impl<T, const N: usize> Default for SmallIndexSet<T, N> {
    fn default() -> Self {
        Self::Linear(ArrayVec::new())
    }
}

impl<T: Eq + Hash, const N: usize> SmallIndexSet<T, N> {
    fn insert(&mut self, x: T) -> bool {
        match self {
            Self::Linear(xs) => {
                // HACK(eddyb) this optimizes for values repeating, i.e.
                // `xs.last() == Some(&x)` being the most common case.
                if xs.iter().rev().any(|old| *old == x) {
                    return false;
                }
                if let Err(err) = xs.try_push(x) {
                    *self = Self::Hashed(Box::new(xs.drain(..).chain([err.element()]).collect()));
                }
                true
            }
            Self::Hashed(xs) => xs.insert(x),
        }
    }

    fn iter(&self) -> SmallSetIter<'_, T> {
        match self {
            Self::Linear(xs) => Either::Left(xs.iter()),
            Self::Hashed(xs) => Either::Right(xs.iter()),
        }
    }
}

impl<'a, T: Eq + Hash, const N: usize> IntoIterator for &'a SmallIndexSet<T, N> {
    type IntoIter = SmallSetIter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// FIXME(eddyb) maybe this should be provided by `visit`.
struct VisitAllRegionsAndNodes<VRN> {
    visit_region_or_node: VRN,
    parent_region: Option<Region>,
}
impl<'a, VRN: FnMut(FuncAt<'a, Either<Region, (Node, Region)>>)> Visitor<'a>
    for VisitAllRegionsAndNodes<VRN>
{
    // FIXME(eddyb) this is excessive, maybe different kinds of
    // visitors should exist for module-level and func-level?
    fn visit_attr_set_use(&mut self, _: AttrSet) {}
    fn visit_type_use(&mut self, _: Type) {}
    fn visit_const_use(&mut self, _: Const) {}
    fn visit_global_var_use(&mut self, _: GlobalVar) {}
    fn visit_func_use(&mut self, _: Func) {}

    fn visit_region_def(&mut self, func_at_region: FuncAt<'a, Region>) {
        (self.visit_region_or_node)(func_at_region.at(Either::Left(func_at_region.position)));

        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_visit_with(self);
        self.parent_region = outer_region;
    }
    fn visit_node_def(&mut self, func_at_node: FuncAt<'a, Node>) {
        (self.visit_region_or_node)(
            func_at_node.at(Either::Right((func_at_node.position, self.parent_region.unwrap()))),
        );
        func_at_node.inner_visit_with(self);
    }
}

impl<'a> Printer<'a> {
    fn new(plan: &Plan<'a>) -> Self {
        let cx = plan.cx;
        let wk = &spv::spec::Spec::get().well_known;

        let mut attrs_with_spv_name_in_use = FxHashMap::default();
        let mut per_region_dbg_scope_def_placements: EntityOrientedDenseMap<
            Region,
            FxHashMap<DbgScopeDefPlaceInRegion, SmallIndexSet<DbgScope, 2>>,
        > = EntityOrientedDenseMap::new();

        // NOTE(eddyb) `SmallSet` is an important optimization, as most attributes
        // *do not* change across versions, so we avoid a lot of repeated work.
        let mut try_claim_name_from_attrs_across_versions =
            |deduped_attrs_across_versions: SmallSetIter<'_, AttrSet>| {
                deduped_attrs_across_versions
                    .copied()
                    .map(|attrs| Some((attrs, plan.attrs_to_unique_spv_name.get(&attrs)?.ok()?)))
                    .collect::<Option<SmallVec<[_; 4]>>>()
                    .filter(|all_names| all_names.iter().map(|(_, spv_name)| spv_name).all_equal())
                    .and_then(|all_names| {
                        let &(_, spv_name) = all_names.first()?;
                        let name = spv::extract_literal_string(&spv_name.imms).ok()?;

                        // This is the point of no return: these `insert`s will
                        // cause `OpName`s to be hidden from their `AttrSet`s.
                        for (attrs, spv_name) in all_names {
                            attrs_with_spv_name_in_use.insert(attrs, spv_name);
                        }

                        Some(name)
                    })
            };

        #[derive(Default)]
        struct AnonCounters {
            types: usize,
            consts: usize,

            global_vars: usize,
            funcs: usize,
        }
        let mut anon_counters = AnonCounters::default();

        let mut use_styles: FxIndexMap<_, _> = plan
            .use_counts
            .iter()
            .map(|(&use_kind, &use_count)| {
                // HACK(eddyb) these are assigned later.
                if let Use::RegionLabel(_)
                | Use::RegionInput { .. }
                | Use::NodeOutput { .. }
                | Use::DataInstOutput(_) = use_kind
                {
                    return (use_kind, UseStyle::Inline);
                }

                // HACK(eddyb) these are "global" to the whole print `Plan`.
                if let Use::PlanItem(
                    PlanItem::AllCxInterned | PlanItem::ModuleDialect | PlanItem::ModuleDebugInfo,
                ) = use_kind
                {
                    return (use_kind, UseStyle::Anon { parent_func: None, idx: 0 });
                }

                let mut deduped_attrs_across_versions = SmallIndexSet::<_, 8>::default();
                match use_kind {
                    Use::CxInterned(interned) => {
                        deduped_attrs_across_versions.insert(match interned {
                            CxInterned::Type(ty) => cx[ty].attrs,
                            CxInterned::Const(ct) => cx[ct].attrs,
                        });
                    }
                    Use::PlanItem(item) => {
                        for version in &plan.versions {
                            let attrs = match version.item_defs.get(&item) {
                                Some(PlanItemDef::GlobalVar(gv_decl)) => gv_decl.attrs,
                                Some(PlanItemDef::Func(func_decl)) => func_decl.attrs,
                                _ => continue,
                            };
                            deduped_attrs_across_versions.insert(attrs);
                        }
                    }
                    Use::DbgScope { .. }
                    | Use::RegionLabel(_)
                    | Use::RegionInput { .. }
                    | Use::NodeOutput { .. }
                    | Use::DataInstOutput(_)
                    | Use::AlignmentAnchorForRegion(_)
                    | Use::AlignmentAnchorForNode(_)
                    | Use::AlignmentAnchorForDataInst(_) => unreachable!(),
                }

                if let Some(name) =
                    try_claim_name_from_attrs_across_versions(deduped_attrs_across_versions.iter())
                {
                    return (use_kind, UseStyle::Named { parent_func: None, name });
                }

                let inline = match use_kind {
                    Use::CxInterned(interned) => {
                        use_count == 1
                            || match interned {
                                CxInterned::Type(ty) => {
                                    let ty_def = &cx[ty];

                                    // FIXME(eddyb) remove the duplication between
                                    // here and `TypeDef`'s `Print` impl.
                                    let has_compact_print_or_is_leaf = match &ty_def.kind {
                                        TypeKind::SpvInst { spv_inst, type_and_const_inputs } => {
                                            [
                                                wk.OpTypeBool,
                                                wk.OpTypeInt,
                                                wk.OpTypeFloat,
                                                wk.OpTypeVector,
                                            ]
                                            .contains(&spv_inst.opcode)
                                                || type_and_const_inputs.is_empty()
                                        }

                                        TypeKind::QPtr | TypeKind::SpvStringLiteralForExtInst => {
                                            true
                                        }
                                    };

                                    ty_def.attrs == AttrSet::default()
                                        && has_compact_print_or_is_leaf
                                }
                                CxInterned::Const(ct) => {
                                    let ct_def = &cx[ct];

                                    // FIXME(eddyb) remove the duplication between
                                    // here and `ConstDef`'s `Print` impl.
                                    let (has_compact_print, has_nested_consts) = match &ct_def.kind
                                    {
                                        ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                                            let (spv_inst, const_inputs) =
                                                &**spv_inst_and_const_inputs;
                                            (
                                                [
                                                    wk.OpConstantFalse,
                                                    wk.OpConstantTrue,
                                                    wk.OpConstant,
                                                ]
                                                .contains(&spv_inst.opcode),
                                                !const_inputs.is_empty(),
                                            )
                                        }
                                        _ => (false, false),
                                    };

                                    ct_def.attrs == AttrSet::default()
                                        && (has_compact_print || !has_nested_consts)
                                }
                            }
                    }
                    Use::PlanItem(_) => false,

                    Use::DbgScope { .. }
                    | Use::RegionLabel(_)
                    | Use::RegionInput { .. }
                    | Use::NodeOutput { .. }
                    | Use::DataInstOutput(_)
                    | Use::AlignmentAnchorForRegion(_)
                    | Use::AlignmentAnchorForNode(_)
                    | Use::AlignmentAnchorForDataInst(_) => {
                        unreachable!()
                    }
                };
                let use_style = if inline {
                    UseStyle::Inline
                } else {
                    let ac = &mut anon_counters;
                    let counter = match use_kind {
                        Use::CxInterned(CxInterned::Type(_)) => &mut ac.types,
                        Use::CxInterned(CxInterned::Const(_)) => &mut ac.consts,
                        Use::PlanItem(PlanItem::GlobalVar(_)) => &mut ac.global_vars,
                        Use::PlanItem(PlanItem::Func(_)) => &mut ac.funcs,

                        Use::PlanItem(
                            PlanItem::AllCxInterned
                            | PlanItem::ModuleDialect
                            | PlanItem::ModuleDebugInfo,
                        )
                        | Use::DbgScope { .. }
                        | Use::RegionLabel(_)
                        | Use::RegionInput { .. }
                        | Use::NodeOutput { .. }
                        | Use::DataInstOutput(_)
                        | Use::AlignmentAnchorForRegion(_)
                        | Use::AlignmentAnchorForNode(_)
                        | Use::AlignmentAnchorForDataInst(_) => {
                            unreachable!()
                        }
                    };
                    let idx = *counter;
                    *counter += 1;
                    UseStyle::Anon { parent_func: None, idx }
                };
                (use_kind, use_style)
            })
            .collect();

        let all_funcs = plan.use_counts.keys().filter_map(|&use_kind| match use_kind {
            Use::PlanItem(PlanItem::Func(func)) => Some(func),
            _ => None,
        });
        for func in all_funcs {
            assert!(matches!(
                use_styles.get(&Use::PlanItem(PlanItem::Func(func))),
                Some(UseStyle::Anon { .. } | UseStyle::Named { .. })
            ));

            // HACK(eddyb) in order to claim `OpName`s unambiguously for any
            // intra-function `Use`, we need its attrs from *all* versions, at
            // the same time, but we visit each version's `FuncDefBody` one at
            // a time, and `EntityDefs` (intentionally) bans even checking if
            // some entity is defined at all, so we can't rely on random-access,
            // and we have to first buffer all the intra-function definitions.
            #[derive(Default)]
            struct IntraFuncDefAcrossVersions {
                deduped_attrs_across_versions: SmallIndexSet<AttrSet, 4>,
            }
            let mut intra_func_defs_across_versions: FxIndexMap<Use, IntraFuncDefAcrossVersions> =
                FxIndexMap::default();

            let func_def_bodies_across_versions = plan.versions.iter().filter_map(|version| {
                match version.item_defs.get(&PlanItem::Func(func))? {
                    PlanItemDef::Func(FuncDecl {
                        def: DeclDef::Present(func_def_body), ..
                    }) => Some(func_def_body),

                    _ => None,
                }
            });

            // HACK(eddyb) this approach could get expensive in some cases, and
            // while it doesn't make a great effort to unify across versions,
            // it still needs several passes and data structures to achieve
            // the goal of printing each `DbgScope` definition exactly once
            // (per version), while not knowing which version is being printed.
            let dbg_scopes_across_versions: Vec<_> = func_def_bodies_across_versions
                .clone()
                .map(|func_def_body| {
                    DbgScopeDefPlacer { cx }.scopes_used_in_whole_func(func_def_body)
                })
                .collect();
            let mut all_possible_dbg_scope_placements: EntityOrientedDenseMap<
                Region,
                FxIndexMap<DbgScopeDefPlaceInRegion, SmallIndexSet<DbgScope, 4>>,
            > = EntityOrientedDenseMap::new();
            for dbg_scopes in &dbg_scopes_across_versions {
                for (&scope, &place) in dbg_scopes.rev_scopes.iter().rev() {
                    let DbgScopeDefPlace { parent_region, intra_region } = place;
                    all_possible_dbg_scope_placements
                        .entry(parent_region)
                        .get_or_insert_default()
                        .entry(intra_region)
                        .or_default()
                        .insert(scope);
                }
            }

            // HACK(eddyb) outside the loop for buffer reuse reasons.
            let mut reusable_claimed_dbg_scopes_in_this_version = FxIndexSet::default();
            let mut reusable_dbg_scope_stack = vec![];

            for (func_def_body, dbg_scopes_in_this_version) in
                func_def_bodies_across_versions.clone().zip_eq(dbg_scopes_across_versions)
            {
                // HACK(eddyb) to avoid ending up with some versions having more
                // than one `Region` printing the same `DbgScope`, we filter the
                // "upper bound" (`all_possible_dbg_scope_placements`) entries,
                // by only keeping, within each version, the first placement of
                // each `DbgScope` - this effectively hoists the `DbgScope`s to
                // some common `Region`, if necessary to "synchronize" versions.
                //
                // FIXME(eddyb) this isn't even enough across all possible cases,
                // because a `Region` can get removed in a later version, and so
                // for now, the version with multiple applicable `Region`s will
                // get duplicated printing (to avoid further complicating this).
                reusable_claimed_dbg_scopes_in_this_version.clear();
                let mut claim_dbg_scopes_at = |place: DbgScopeDefPlace| {
                    let DbgScopeDefPlace { parent_region, intra_region } = place;
                    let Some(region_placements) =
                        all_possible_dbg_scope_placements.get(parent_region)
                    else {
                        return;
                    };

                    // HACK(eddyb) because `all_possible_dbg_scope_placements`
                    // is cross-version, further filtering is needed to ignore
                    // `DbgScope`s completely missing from certain versions.
                    let used_in_this_version = |scope: &DbgScope| {
                        dbg_scopes_in_this_version.rev_scopes.contains_key(scope)
                    };

                    // HACK(eddyb) take advantage of `IndexSet` ordering,
                    // in order to also use `claimed` as a "staging area".
                    let claimed = &mut reusable_claimed_dbg_scopes_in_this_version;
                    let newly_claimed_start = claimed.len();

                    // HACK(eddyb) here this isn't just used for empty regions,
                    // but indicates the start of `parent_region`, where anything
                    // placed in `parent_region` *across any version*, should be
                    // hoisted to (unless claimed by any child of `parent_region`).
                    if intra_region.before_node.is_none() {
                        // HACK(eddyb) the child nodes haven't been claimed yet,
                        // so to know which `DbgScope`s they *will have* claimed
                        // in the future, we simulate that entire process, with
                        // `claimed` potentially gaining a range of new entries,
                        // that will effectively filter a *second* such range,
                        // before removing the first range (of future claims).
                        claimed.extend(
                            func_def_body
                                .at(parent_region)
                                .at_children()
                                .into_iter()
                                .map(|func_at_node| Some(func_at_node.position))
                                .filter_map(|before_node| {
                                    region_placements.get(&DbgScopeDefPlaceInRegion { before_node })
                                })
                                .flatten()
                                .copied()
                                .filter(used_in_this_version),
                        );
                        let simulated_future_claims_range = newly_claimed_start..claimed.len();

                        // NOTE(eddyb) implicit filtering: only new `DbgScope`s
                        // get inserted into `claimed`, growing `claimed.len()`.
                        claimed.extend(
                            region_placements
                                .values()
                                .flatten()
                                .copied()
                                .filter(used_in_this_version),
                        );

                        claimed.splice(simulated_future_claims_range, []);
                    } else {
                        // NOTE(eddyb) implicit filtering: only new `DbgScope`s
                        // get inserted into `claimed`, growing `claimed.len()`.
                        claimed.extend(
                            region_placements
                                .get(&intra_region)
                                .into_iter()
                                .flatten()
                                .copied()
                                .filter(used_in_this_version),
                        );
                    }

                    let newly_claimed_range = newly_claimed_start..claimed.len();

                    if newly_claimed_range.is_empty() {
                        return;
                    }

                    let placements = per_region_dbg_scope_def_placements
                        .entry(parent_region)
                        .get_or_insert_default()
                        .entry(intra_region)
                        .or_default();
                    for claim_idx in newly_claimed_range {
                        let scope = claimed[claim_idx];

                        // HACK(eddyb) also claim all yet-unclaimed parent scopes.
                        let stack = &mut reusable_dbg_scope_stack;
                        stack.clear();
                        stack.extend(scope.parents(cx).take_while(|parent| {
                            claimed.get_index_of(parent).is_none_or(|parent_claim_idx| {
                                // HACK(eddyb) scopes later in `newly_claimed_range`
                                // get treated as unclaimed, in order to claim them
                                // even earlier (just before they're first needed).
                                parent_claim_idx > claim_idx
                            })
                        }));
                        for parent in stack.drain(..).rev() {
                            placements.insert(parent);
                        }

                        placements.insert(scope);
                    }
                };

                // HACK(eddyb) a full separate visit is done here, due to the
                // cross-version nature of `per_region_dbg_scope_def_placements`,
                // meaning later versions can hoist `DbgScope`s to earlier in a
                // `Region`, and therefore change the print order of `DbgScope`s,
                // so numbering can only be done *after* the order is settled.
                // FIXME(eddyb) maybe run `DbgScopeDefPlacer` twice, instead,
                // or find some other way to hoist `DbgScope`s to placements
                // that work well across all versions.
                let visit_region_or_node = |farn: FuncAt<'_, Either<Region, (Node, Region)>>| {
                    let region_or_node = farn.position;
                    if let Either::Left(region) = region_or_node {
                        // HACK(eddyb) this will claim all `DbgScope`s which both:
                        // - happen to be placed anywhere inside `region` in any version
                        // - won't be later claimed by a `region` child `Node`
                        // (see also comments above/inside `claim_dbg_scopes_at`)
                        claim_dbg_scopes_at(DbgScopeDefPlace {
                            parent_region: region,
                            intra_region: DbgScopeDefPlaceInRegion { before_node: None },
                        });
                    } else {
                        let (node, parent_region) = region_or_node.right().unwrap();

                        claim_dbg_scopes_at(DbgScopeDefPlace {
                            parent_region,
                            intra_region: DbgScopeDefPlaceInRegion { before_node: Some(node) },
                        });
                    }
                };

                func_def_body.inner_visit_with(&mut VisitAllRegionsAndNodes {
                    visit_region_or_node,
                    parent_region: None,
                });
            }
            drop(reusable_claimed_dbg_scopes_in_this_version);
            drop(reusable_dbg_scope_stack);

            let mut dbg_scope_counter = 0;

            for func_def_body in func_def_bodies_across_versions {
                let mut define_dbg_scopes_at = |place: DbgScopeDefPlace| {
                    let DbgScopeDefPlace { parent_region, intra_region } = place;
                    let Some(dbg_scopes) = per_region_dbg_scope_def_placements
                        .get(parent_region)
                        .and_then(|placements| placements.get(&intra_region))
                    else {
                        return;
                    };

                    let counter = &mut dbg_scope_counter;
                    for &scope in dbg_scopes {
                        use_styles
                            .entry(Use::DbgScope { scope, parent_func: func })
                            .or_insert_with(|| {
                                let idx = *counter;
                                *counter += 1;
                                UseStyle::Anon { parent_func: Some(func), idx }
                            });
                    }
                };
                let mut define = |use_kind, attrs| {
                    let def = intra_func_defs_across_versions.entry(use_kind).or_default();
                    if let Some(attrs) = attrs {
                        def.deduped_attrs_across_versions.insert(attrs);
                    }
                };
                // HACK(eddyb) this is as bad as it is due to the combination of:
                // - borrowing constraints on `define` (mutable access to maps)
                // - needing to minimize the changes to allow rebasing further
                //   refactors (after which it may be easier to clean up anyway)
                let visit_region_or_node = |farn: FuncAt<'_, Either<Region, (Node, Region)>>| {
                    let region_or_node = farn.position;
                    if let Either::Left(region) = region_or_node {
                        define_dbg_scopes_at(DbgScopeDefPlace {
                            parent_region: region,
                            intra_region: DbgScopeDefPlaceInRegion { before_node: None },
                        });

                        define(Use::AlignmentAnchorForRegion(region), None);
                        // FIXME(eddyb) should labels have names?
                        define(Use::RegionLabel(region), None);

                        let RegionDef { inputs, children: _, outputs: _ } =
                            func_def_body.at(region).def();

                        for (i, input_decl) in inputs.iter().enumerate() {
                            define(
                                Use::RegionInput { region, input_idx: i.try_into().unwrap() },
                                Some(input_decl.attrs),
                            );
                        }
                    } else {
                        let (node, parent_region) = region_or_node.right().unwrap();
                        let func_at_node = farn.at(node);

                        define_dbg_scopes_at(DbgScopeDefPlace {
                            parent_region,
                            intra_region: DbgScopeDefPlaceInRegion { before_node: Some(node) },
                        });

                        define(Use::AlignmentAnchorForNode(node), None);

                        let NodeDef { kind, outputs } = func_at_node.def();

                        if let NodeKind::Block { insts } = *kind {
                            for func_at_inst in func_def_body.at(insts) {
                                define(
                                    Use::AlignmentAnchorForDataInst(func_at_inst.position),
                                    None,
                                );
                                let inst_def = func_at_inst.def();
                                if inst_def.output_type.is_some() {
                                    define(
                                        Use::DataInstOutput(func_at_inst.position),
                                        Some(inst_def.attrs),
                                    );
                                }
                            }
                        }

                        for (i, output_decl) in outputs.iter().enumerate() {
                            define(
                                Use::NodeOutput { node, output_idx: i.try_into().unwrap() },
                                Some(output_decl.attrs),
                            );
                        }
                    }
                };

                func_def_body.inner_visit_with(&mut VisitAllRegionsAndNodes {
                    visit_region_or_node,
                    parent_region: None,
                });
            }

            let mut region_label_counter = 0;
            let mut value_counter = 0;
            let mut alignment_anchor_counter = 0;

            // Assign an unique label/value/alignment-anchor name/index to each
            // intra-function definition which appears in at least one version,
            // but only if it's actually used (or is an alignment anchor).
            for (use_kind, def) in intra_func_defs_across_versions {
                let (counter, use_style_slot) = match use_kind {
                    Use::RegionLabel(_) => {
                        (&mut region_label_counter, use_styles.get_mut(&use_kind))
                    }

                    Use::RegionInput { .. } | Use::NodeOutput { .. } | Use::DataInstOutput(_) => {
                        (&mut value_counter, use_styles.get_mut(&use_kind))
                    }

                    Use::AlignmentAnchorForRegion(_)
                    | Use::AlignmentAnchorForNode(_)
                    | Use::AlignmentAnchorForDataInst(_) => (
                        &mut alignment_anchor_counter,
                        Some(use_styles.entry(use_kind).or_insert(UseStyle::Inline)),
                    ),

                    _ => unreachable!(),
                };
                if let Some(use_style) = use_style_slot {
                    assert!(matches!(use_style, UseStyle::Inline));

                    let parent_func = Some(func);
                    let named_style = try_claim_name_from_attrs_across_versions(
                        def.deduped_attrs_across_versions.iter(),
                    )
                    .map(|name| UseStyle::Named { parent_func, name });

                    *use_style = named_style.unwrap_or_else(|| {
                        let idx = *counter;
                        *counter += 1;
                        UseStyle::Anon { parent_func, idx }
                    });
                }
            }
        }

        Self {
            cx,
            use_styles,
            attrs_with_spv_name_in_use,
            per_region_dbg_scope_def_placements,
            current_plan_item: Cell::new(None),
        }
    }

    pub fn cx(&self) -> &'a Context {
        self.cx
    }
}

// Styles for a variety of syntactic categories.
// FIXME(eddyb) this is a somewhat inefficient way of declaring these.
//
// NOTE(eddyb) these methods take `self` so they can become configurable in the future.
#[allow(clippy::unused_self)]
impl Printer<'_> {
    fn error_style(&self) -> pretty::Styles {
        pretty::Styles::color(pretty::palettes::simple::MAGENTA)
    }
    fn comment_style(&self) -> pretty::Styles {
        pretty::Styles {
            color_opacity: Some(0.3),
            size: Some(-4),
            // FIXME(eddyb) this looks wrong for some reason?
            // subscript: true,
            ..pretty::Styles::color(pretty::palettes::simple::DARK_GRAY)
        }
    }
    fn named_argument_label_style(&self) -> pretty::Styles {
        pretty::Styles {
            size: Some(-5),
            ..pretty::Styles::color(pretty::palettes::simple::DARK_GRAY)
        }
    }
    fn numeric_literal_style(&self) -> pretty::Styles {
        pretty::Styles::color(pretty::palettes::simple::YELLOW)
    }
    fn string_literal_style(&self) -> pretty::Styles {
        pretty::Styles::color(pretty::palettes::simple::RED)
    }
    fn string_literal_escape_style(&self) -> pretty::Styles {
        pretty::Styles::color(pretty::palettes::simple::ORANGE)
    }
    fn declarative_keyword_style(&self) -> pretty::Styles {
        pretty::Styles::color(pretty::palettes::simple::BLUE)
    }
    fn imperative_keyword_style(&self) -> pretty::Styles {
        pretty::Styles {
            thickness: Some(2),
            ..pretty::Styles::color(pretty::palettes::simple::MAGENTA)
        }
    }
    fn spv_base_style(&self) -> pretty::Styles {
        pretty::Styles::color(pretty::palettes::simple::ORANGE)
    }
    fn spv_op_style(&self) -> pretty::Styles {
        pretty::Styles { thickness: Some(3), ..self.spv_base_style() }
    }
    fn spv_enumerand_name_style(&self) -> pretty::Styles {
        pretty::Styles::color(pretty::palettes::simple::CYAN)
    }
    fn attr_style(&self) -> pretty::Styles {
        pretty::Styles {
            color_opacity: Some(0.6),
            thickness: Some(-2),
            ..pretty::Styles::color(pretty::palettes::simple::GREEN)
        }
    }

    /// Compute a suitable style for an unintrusive `foo.` "namespace prefix",
    /// from a more typical style (by shrinking and/or reducing visibility).
    fn demote_style_for_namespace_prefix(&self, mut style: pretty::Styles) -> pretty::Styles {
        // NOTE(eddyb) this was `opacity: Some(0.4)` + `thickness: Some(-3)`,
        // but thinner text ended up being more annoying to read while still
        // using up too much real-estate (compared to decreasing the size).
        style.color_opacity = Some(style.color_opacity.unwrap_or(1.0) * 0.6);
        // FIXME(eddyb) maybe this could be more uniform with a different unit.
        style.size = Some(style.size.map_or(-4, |size| size - 1));
        style
    }
}

impl Printer<'_> {
    /// Pretty-print a numeric (integer) literal, in either base 10 or 16.
    ///
    /// Heuristics (e.g. statistics of the digits/nibbles) are used to pick
    /// e.g. `0xff00` over `65280`, and `1000` over `0x3e8`.
    //
    // FIXME(eddyb) handle signedness, maybe tune heuristics?
    // FIXME(eddyb) add methods like this for other kinds of numeric literals.
    fn pretty_numeric_literal_as_dec_or_hex(&self, n: u128) -> pretty::Fragment {
        let style = self.numeric_literal_style();

        // FIXME(eddyb) it should be possible to avoid allocations or floats,
        // but this initial implementation focuses on simplicity above all else.

        let dec = format!("{n}");

        // HACK(eddyb) all 2-digit decimal numbers always have 1-2 unique nibbles,
        // making it effectively impossible to tell apart with a heuristic, and
        // on top of that, even numbers that are "simpler" in hexadecimal, may
        // still be more recognizable as decimal (e.g. `64` over `0x40`).
        if dec.len() <= 2 {
            return style.apply(dec).into();
        }

        let hex = format!("0x{n:x}");

        fn score<const BASE: usize>(s: &str) -> f64 {
            let probability_per_digit = 1.0 / (s.len() as f64);
            let mut digit_probabilities = [0.0; BASE];

            let mut total_probability = 0.0;

            for d in s.chars() {
                digit_probabilities[d.to_digit(BASE as u32).unwrap() as usize] +=
                    probability_per_digit;
                total_probability += probability_per_digit;
            }

            // HACK(eddyb) this will end up being `1.0 / N * N / BASE`, which
            // in theory should always result in `1.0 / BASE` (i.e. a constant),
            // except for float rounding, but maybe this shouldn't care?
            let avg_probability = total_probability / (BASE as f64);

            // HACK(eddyb) compute MSE (mean squared error), hoping that will
            // be inversely correlated with how "random" the digit string looks.
            digit_probabilities.iter().map(|&p| (p - avg_probability).powi(2)).sum::<f64>()
                / (BASE as f64)
        }

        let hex_over_dec = score::<16>(&hex[2..]) - score::<10>(&dec);

        // HACK(eddyb) arbitrary "epsilon" based on observed values.
        let hex_over_dec =
            if hex_over_dec.abs() < 1e-3 { hex.len() <= dec.len() } else { hex_over_dec > 0.0 };

        style.apply(if hex_over_dec { hex } else { dec }).into()
    }

    /// Pretty-print a string literal with escaping and styling.
    fn pretty_string_literal(&self, s: &str) -> pretty::Fragment {
        // HACK(eddyb) avoid using multiline formatting for trailing `\n`, which
        // may be common in e.g. `debugPrintf("foo\n")`-style messages.
        let use_multiline_format = s.trim_end_matches('\n').contains('\n');

        // HACK(eddyb) this is somewhat inefficient, but we need to allocate a
        // `String` for every piece anyway, so might as well make it convenient.
        pretty::Fragment::new(
            // HACK(eddyb) this allows aligning the actual string contents,
            // (see `c == '\n'` special-casing below for when this applies).
            (use_multiline_format.then_some(Either::Left(' ')).into_iter())
                .chain([Either::Left('"')])
                .chain(s.chars().flat_map(|c| {
                    let escaped = c.escape_debug();
                    let maybe_escaped = if c == '\'' {
                        // Unescape single quotes, we're in a double-quoted string.
                        assert_eq!(escaped.collect_tuple(), Some(('\\', c)));
                        Either::Left(c)
                    } else if let Some((single,)) = escaped.clone().collect_tuple() {
                        assert_eq!(single, c);
                        Either::Left(c)
                    } else {
                        assert_eq!(escaped.clone().next(), Some('\\'));
                        Either::Right(escaped)
                    };

                    // HACK(eddyb) move escaped `\n` to the start of a new line,
                    // using Rust's trailing `\` on the previous line, which eats
                    // all following whitespace (and only stops at the escape).
                    let extra_prefix_unescaped =
                        if c == '\n' && use_multiline_format { "\\\n" } else { "" };

                    (extra_prefix_unescaped.chars().map(Either::Left)).chain([maybe_escaped])
                }))
                .chain([Either::Left('"')])
                .group_by(|maybe_escaped| maybe_escaped.is_right())
                .into_iter()
                .map(|(escaped, group)| {
                    if escaped {
                        self.string_literal_escape_style()
                            .apply(group.flat_map(Either::unwrap_right).collect::<String>())
                    } else {
                        self.string_literal_style()
                            .apply(group.map(Either::unwrap_left).collect::<String>())
                    }
                }),
        )
    }

    /// Pretty-print a `name: ` style "named argument" prefix.
    fn pretty_named_argument_prefix<'b>(&self, name: impl Into<Cow<'b, str>>) -> pretty::Fragment {
        // FIXME(eddyb) avoid the cost of allocating here.
        self.named_argument_label_style().apply(format!("{}: ", name.into())).into()
    }

    /// Pretty-print a `: T` style "type ascription" suffix.
    ///
    /// This should be used everywhere some type ascription notation is needed,
    /// to ensure consistency across all such situations.
    fn pretty_type_ascription_suffix(&self, ty: Type) -> pretty::Fragment {
        pretty::join_space(":", [ty.print(self)])
    }

    /// Pretty-print a SPIR-V `opcode`'s name, prefixed by `"spv."`.
    fn pretty_spv_opcode(
        &self,
        opcode_name_style: pretty::Styles,
        opcode: spv::spec::Opcode,
    ) -> pretty::Fragment {
        pretty::Fragment::new([
            self.demote_style_for_namespace_prefix(self.spv_base_style()).apply("spv."),
            opcode_name_style.apply(opcode.name()),
        ])
    }

    /// Clean up a `spv::print::TokensForOperand` string (common helper used below).
    #[allow(clippy::unused_self)]
    fn sanitize_spv_operand_name<'b>(&self, name: &'b str) -> Option<Cow<'b, str>> {
        Some(name).and_then(|name| {
            // HACK(eddyb) some operand names are useless.
            if name == "Type"
                || name
                    .strip_prefix("Operand ")
                    .is_some_and(|s| s.chars().all(|c| c.is_ascii_digit()))
            {
                return None;
            }

            // Turn `Foo Bar` and `Foo bar` into `FooBar`.
            // FIXME(eddyb) use `&[AsciiChar]` when that stabilizes.
            let name = name
                .split_ascii_whitespace()
                .map(|word| {
                    if word.starts_with(|c: char| c.is_ascii_lowercase()) {
                        let mut word = word.to_string();
                        word[..1].make_ascii_uppercase();
                        Cow::Owned(word)
                    } else {
                        word.into()
                    }
                })
                .reduce(|out, extra| (out.into_owned() + &extra).into())
                .unwrap_or_default();

            Some(name)
        })
    }

    /// Pretty-print a `spv::print::TokensForOperand` (common helper used below).
    fn pretty_spv_print_tokens_for_operand(
        &self,
        operand: spv::print::TokensForOperand<Option<pretty::Fragment>>,
    ) -> pretty::Fragment {
        pretty::Fragment::new(operand.tokens.into_iter().map(|token| {
            match token {
                spv::print::Token::Error(s) => self.error_style().apply(s).into(),
                spv::print::Token::OperandName(s) => self
                    .sanitize_spv_operand_name(s)
                    .map(|name| self.pretty_named_argument_prefix(name))
                    .unwrap_or_default(),
                spv::print::Token::Punctuation(s) => s.into(),
                spv::print::Token::OperandKindNamespacePrefix(s) => {
                    pretty::Fragment::new([
                        // HACK(eddyb) double-demote to end up with `spv.A.B`,
                        // with increasing size from `spv.` to `A.` to `B`.
                        self.demote_style_for_namespace_prefix(
                            self.demote_style_for_namespace_prefix(self.spv_base_style()),
                        )
                        .apply("spv."),
                        // FIXME(eddyb) avoid the cost of allocating here.
                        self.demote_style_for_namespace_prefix(self.declarative_keyword_style())
                            .apply(format!("{s}.")),
                    ])
                }
                spv::print::Token::EnumerandName(s) => {
                    self.spv_enumerand_name_style().apply(s).into()
                }
                spv::print::Token::NumericLiteral(s) => {
                    self.numeric_literal_style().apply(s).into()
                }
                spv::print::Token::StringLiteral(s) => self.string_literal_style().apply(s).into(),
                spv::print::Token::Id(id) => {
                    id.unwrap_or_else(|| self.comment_style().apply("/* implicit ID */").into())
                }
            }
        }))
    }

    /// Pretty-print a single SPIR-V operand from only immediates, potentially
    /// composed of an enumerand with parameters (which consumes more immediates).
    fn pretty_spv_operand_from_imms(
        &self,
        imms: impl IntoIterator<Item = spv::Imm>,
    ) -> pretty::Fragment {
        self.pretty_spv_print_tokens_for_operand(spv::print::operand_from_imms(imms))
    }

    /// Pretty-print a single SPIR-V (short) immediate (e.g. an enumerand).
    fn pretty_spv_imm(&self, kind: spv::spec::OperandKind, word: u32) -> pretty::Fragment {
        self.pretty_spv_operand_from_imms([spv::Imm::Short(kind, word)])
    }

    /// Pretty-print an arbitrary SPIR-V `opcode` with its SPIR-V operands being
    /// given by `imms` (non-IDs) and `printed_ids` (IDs, printed by the caller).
    ///
    /// `printed_ids` elements can be `None` to indicate an ID operand is implicit
    /// in SPIR-T, and should not be printed (e.g. decorations' target IDs).
    /// But if `printed_ids` doesn't need to have `None` elements, it can skip
    /// the `Option` entirely (i.e. have `pretty::Fragment` elements directly).
    ///
    /// Immediate and `ID` operands are interleaved (in the order mandated by
    /// the SPIR-V standard) and together wrapped in parentheses, e.g.:
    /// `spv.OpFoo(spv.FooEnum.Bar, v1, 123, v2, "baz")`.
    ///
    /// This should be used everywhere a SPIR-V instruction needs to be printed,
    /// to ensure consistency across all such situations.
    fn pretty_spv_inst<OPF: Into<Option<pretty::Fragment>>>(
        &self,
        spv_inst_name_style: pretty::Styles,
        opcode: spv::spec::Opcode,
        imms: &[spv::Imm],
        printed_ids: impl IntoIterator<Item = OPF>,
    ) -> pretty::Fragment {
        let mut operands = spv::print::inst_operands(
            opcode,
            imms.iter().copied(),
            printed_ids.into_iter().map(|printed_id| printed_id.into()),
        )
        .filter_map(|operand| match operand.tokens[..] {
            [spv::print::Token::Id(None)]
            | [spv::print::Token::OperandName(_), spv::print::Token::Id(None)] => None,

            _ => Some(self.pretty_spv_print_tokens_for_operand(operand)),
        })
        .peekable();

        let mut out = self.pretty_spv_opcode(spv_inst_name_style, opcode);

        if operands.peek().is_some() {
            out = pretty::Fragment::new([out, pretty::join_comma_sep("(", operands, ")")]);
        }

        out
    }
}

/// A [`Print`] `Output` type that splits the attributes from the main body of the
/// definition, allowing additional processing before they get concatenated.
#[derive(Default)]
pub struct AttrsAndDef {
    pub attrs: pretty::Fragment,

    /// Definition that typically looks like one of these cases:
    /// * ` = ...` for `name = ...`
    /// * `(...) {...}` for `name(...) {...}` (i.e. functions)
    ///
    /// Where `name` is added later (i.e. between `attrs` and `def_without_name`).
    pub def_without_name: pretty::Fragment,
}

impl AttrsAndDef {
    /// Concat `attrs`, `name` and `def_without_name` into a [`pretty::Fragment`],
    /// effectively "filling in" the `name` missing from `def_without_name`.
    ///
    /// If `name` starts with an anchor definition, the definition of that anchor
    /// gets hoised to before (some non-empty) `attrs`, so that navigating to that
    /// anchor doesn't "hide" those attributes (requiring scrolling to see them).
    fn insert_name_before_def(self, name: impl Into<pretty::Fragment>) -> pretty::Fragment {
        let Self { attrs, def_without_name } = self;

        let mut maybe_hoisted_anchor = pretty::Fragment::default();
        let mut maybe_def_start_anchor = pretty::Fragment::default();
        let mut maybe_def_end_anchor = pretty::Fragment::default();
        let mut name = name.into();
        if let [
            pretty::Node::Anchor { is_def: original_anchor_is_def @ true, anchor, text: _ },
            ..,
        ] = &mut name.nodes[..]
        {
            if !attrs.nodes.is_empty() {
                *original_anchor_is_def = false;
                maybe_hoisted_anchor = pretty::Node::Anchor {
                    is_def: true,
                    anchor: anchor.clone(),
                    text: vec![].into(),
                }
                .into();
            }

            // HACK(eddyb) add a pair of anchors "bracketing" the definition
            // (though see below for why only the "start" side is currently
            // in use), to help with `multiversion` alignment, as long as
            // there's no alignment anchor already starting the definition.
            let has_alignment_anchor = match &def_without_name.nodes[..] {
                [pretty::Node::Anchor { is_def: true, anchor, text }, ..] => {
                    anchor.contains(Use::ANCHOR_ALIGNMENT_NAME_PREFIX) && text.is_empty()
                }

                _ => false,
            };
            let mk_anchor_def = |suffix| {
                pretty::Node::Anchor {
                    is_def: true,
                    anchor: format!("{anchor}.{suffix}").into(),
                    text: vec![].into(),
                }
                .into()
            };
            if !has_alignment_anchor {
                maybe_def_start_anchor = mk_anchor_def("start");
                // FIXME(eddyb) having end alignment may be useful, but the
                // current logic in `multiversion` would prefer aligning
                // the ends, to the detriment of the rest (causing huge gaps).
                if false {
                    maybe_def_end_anchor = mk_anchor_def("end");
                }
            }
        }
        pretty::Fragment::new([
            maybe_hoisted_anchor,
            attrs,
            name,
            maybe_def_start_anchor,
            def_without_name,
            maybe_def_end_anchor,
        ])
    }
}

pub trait Print {
    // FIXME(eddyb) maybe remove `type Output;` flexibility by having two traits
    // instead of one? (a method that returns `self.attrs` would allow for some
    // automation, and remove a lot of the noise that `AttrsAndDef` adds).
    type Output;
    fn print(&self, printer: &Printer<'_>) -> Self::Output;
}

impl Use {
    /// Common implementation for [`Use::print`] and [`Use::print_as_def`].
    fn print_as_ref_or_def(&self, printer: &Printer<'_>, is_def: bool) -> pretty::Fragment {
        // FIXME(eddyb) name rename `UseStyle` so it doesn't clash with `pretty::Styles`
        let use_style = printer.use_styles.get(self).unwrap_or(&UseStyle::Inline);
        match use_style {
            &UseStyle::Anon { parent_func, idx: _ } | &UseStyle::Named { parent_func, name: _ } => {
                // FIXME(eddyb) should this be used as part of `UseStyle`'s definition?
                #[derive(Debug, PartialEq, Eq)]
                enum Suffix<'a> {
                    Num(usize),
                    Name(&'a str),
                }

                impl Suffix<'_> {
                    /// Format `self` into `w`, minimally escaping (`Sufix::Name`)
                    /// `char`s as `&#...;` HTML entities, to limit the charset
                    /// to `[A-Za-z0-9_]` (plus `[&#;]`, for escapes alone).
                    fn write_escaped_to(&self, w: &mut impl fmt::Write) -> fmt::Result {
                        match *self {
                            Suffix::Num(idx) => write!(w, "{idx}"),
                            Suffix::Name(mut name) => {
                                // HACK(eddyb) clearly separate from whatever is
                                // before (e.g. a category name), and disambiguate
                                // between e.g. `Num(123)` and `Name("123")`.
                                w.write_str("_")?;

                                while !name.is_empty() {
                                    // HACK(eddyb) this is convenient way to
                                    // grab the longest prefix that is all valid.
                                    let is_valid = |c: char| c.is_ascii_alphanumeric() || c == '_';
                                    let name_after_valid = name.trim_start_matches(is_valid);
                                    let valid_prefix = &name[..name.len() - name_after_valid.len()];
                                    name = name_after_valid;

                                    if !valid_prefix.is_empty() {
                                        w.write_str(valid_prefix)?;
                                    }

                                    // `name` is either empty now, or starts with
                                    // an invalid `char` (that we need to escape).
                                    let mut chars = name.chars();
                                    if let Some(c) = chars.next() {
                                        assert!(!is_valid(c));
                                        write!(w, "&#{};", c as u32)?;
                                    }
                                    name = chars.as_str();
                                }
                                Ok(())
                            }
                        }
                    }
                }

                let suffix_of = |use_style| match use_style {
                    &UseStyle::Anon { idx, .. } => Suffix::Num(idx),
                    UseStyle::Named { name, .. } => Suffix::Name(name),
                    UseStyle::Inline => unreachable!(),
                };

                let (keyword, name_prefix) = self.keyword_and_name_prefix();
                let suffix = suffix_of(use_style);

                // FIXME(eddyb) could the `anchor: Rc<str>` be cached?
                let mk_anchor = |anchor: String, text: Vec<_>| pretty::Node::Anchor {
                    is_def,
                    anchor: anchor.into(),
                    text: text.into(),
                };

                // HACK(eddyb) these are "global" to the whole print `Plan`.
                if let Use::PlanItem(PlanItem::ModuleDialect | PlanItem::ModuleDebugInfo) = self {
                    assert_eq!((is_def, name_prefix, suffix), (true, "", Suffix::Num(0)));
                    return mk_anchor(keyword.into(), vec![(None, keyword.into())]).into();
                }

                let mut anchor = String::new();
                if let Some(func) = parent_func {
                    // Disambiguate intra-function anchors (labels/values) by
                    // prepending a prefix of the form `F123.`.
                    let func = Use::PlanItem(PlanItem::Func(func));
                    write!(anchor, "{}", func.keyword_and_name_prefix().1).unwrap();
                    suffix_of(&printer.use_styles[&func]).write_escaped_to(&mut anchor).unwrap();
                    anchor += ".";
                }
                anchor += name_prefix;
                suffix.write_escaped_to(&mut anchor).unwrap();

                let name = if let Self::AlignmentAnchorForRegion(_)
                | Self::AlignmentAnchorForNode(_)
                | Self::AlignmentAnchorForDataInst(_) = self
                {
                    vec![]
                } else {
                    // HACK(eddyb) `DbgScope`s only appear in "debuginfo comments",
                    // and that's easier to handle here, than post-process later.
                    // FIXME(eddyb) consider `thickness: Some(0)` or something
                    // intermediate, instead of the `4` that anchors default to.
                    let base_style = match self {
                        Self::DbgScope { .. } => Some(printer.comment_style()),
                        _ => None,
                    };

                    let suffix_style = {
                        // HACK(eddyb) make the suffix larger for e.g. `T` than `v`.
                        let suffix_size = if name_prefix.ends_with(|c: char| c.is_ascii_uppercase())
                        {
                            -1
                        } else {
                            -2
                        };

                        let base_or_default = base_style.unwrap_or_default();
                        pretty::Styles {
                            size: Some(base_or_default.size.unwrap_or_default() + suffix_size),
                            ..base_or_default
                        }
                    };
                    let suffix = match suffix {
                        Suffix::Num(idx) => (
                            Some(pretty::Styles { subscript: true, ..suffix_style }),
                            format!("{idx}").into(),
                        ),
                        Suffix::Name(name) => (
                            Some(pretty::Styles {
                                thickness: Some(0),
                                size: Some(suffix_style.size.unwrap_or_default() - 1),
                                color: Some(pretty::palettes::simple::LIGHT_GRAY),
                                ..suffix_style
                            }),
                            format!("`{name}`").into(),
                        ),
                    };
                    Some(keyword)
                        .filter(|kw| is_def && !kw.is_empty())
                        .into_iter()
                        .flat_map(|kw| [(base_style, kw.into()), (base_style, " ".into())])
                        .chain([(base_style, name_prefix.into()), suffix])
                        .collect()
                };
                mk_anchor(anchor, name).into()
            }
            UseStyle::Inline => match *self {
                Self::CxInterned(interned) => {
                    interned.print(printer).insert_name_before_def(pretty::Fragment::default())
                }
                Self::PlanItem(item) => printer
                    .error_style()
                    .apply(format!(
                        "/* undefined {} */_",
                        item.keyword_and_name_prefix().map_or_else(|s| s, |(s, _)| s)
                    ))
                    .into(),
                Self::DbgScope { .. }
                | Self::RegionLabel(_)
                | Self::RegionInput { .. }
                | Self::NodeOutput { .. }
                | Self::DataInstOutput(_) => "_".into(),

                Self::AlignmentAnchorForRegion(_)
                | Self::AlignmentAnchorForNode(_)
                | Self::AlignmentAnchorForDataInst(_) => unreachable!(),
            },
        }
    }

    fn print_as_def(&self, printer: &Printer<'_>) -> pretty::Fragment {
        self.print_as_ref_or_def(printer, true)
    }
}

impl Print for Use {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        self.print_as_ref_or_def(printer, false)
    }
}

// Interned/module-stored nodes dispatch through the `Use` impl above.
impl Print for Type {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        Use::CxInterned(CxInterned::Type(*self)).print(printer)
    }
}
impl Print for Const {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        Use::CxInterned(CxInterned::Const(*self)).print(printer)
    }
}
impl Print for GlobalVar {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        Use::PlanItem(PlanItem::GlobalVar(*self)).print(printer)
    }
}
impl Print for Func {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        Use::PlanItem(PlanItem::Func(*self)).print(printer)
    }
}

// NOTE(eddyb) the `Print` impl for `PlanItem` is for the top-level definition,
// *not* any uses (which go through the `Print` impls above).

impl Print for Plan<'_> {
    type Output = Versions<pretty::Fragment>;
    fn print(&self, printer: &Printer<'_>) -> Versions<pretty::Fragment> {
        self.print_all_items_and_or_root(printer, true, true)
    }
}

impl Plan<'_> {
    fn print_all_items_and_or_root(
        &self,
        printer: &Printer<'_>,
        print_all_items: bool,
        print_root: bool,
    ) -> Versions<pretty::Fragment> {
        enum PlanItemOrRoot {
            PlanItem(PlanItem),
            Root,
        }

        let all_items = printer
            .use_styles
            .keys()
            .filter_map(|&use_kind| match use_kind {
                Use::PlanItem(item) => Some(item),
                _ => None,
            })
            .map(PlanItemOrRoot::PlanItem);
        let root = [PlanItemOrRoot::Root].into_iter();
        let all_items_and_or_root = Some(all_items)
            .filter(|_| print_all_items)
            .into_iter()
            .flatten()
            .chain(Some(root).filter(|_| print_root).into_iter().flatten());

        let per_item_versions_with_repeat_count =
            all_items_and_or_root.map(|item_or_root| -> SmallVec<[_; 1]> {
                // Only print `PlanItem::AllCxInterned` once (it doesn't really have
                // per-version item definitions in the first place, anyway).
                if let PlanItemOrRoot::PlanItem(item @ PlanItem::AllCxInterned) = item_or_root {
                    item.keyword_and_name_prefix().unwrap_err();

                    return [(CxInterned::print_all(printer), self.versions.len())]
                        .into_iter()
                        .collect();
                }

                self.versions
                    .iter()
                    .map(move |version| match item_or_root {
                        PlanItemOrRoot::PlanItem(item) => version
                            .item_defs
                            .get(&item)
                            .map(|def| {
                                let prev_plan_item = printer.current_plan_item.replace(Some(item));
                                let printed_def = def.print(printer);
                                printer.current_plan_item.set(prev_plan_item);

                                printed_def.insert_name_before_def(
                                    Use::PlanItem(item).print_as_def(printer),
                                )
                            })
                            .unwrap_or_default(),
                        PlanItemOrRoot::Root => version.root.print(printer),
                    })
                    .dedup_with_count()
                    .map(|(repeat_count, fragment)| {
                        // FIXME(eddyb) consider rewriting intra-func anchors
                        // here, post-deduplication, to be unique per-version,
                        // though `multiversion` should probably handle it.

                        (fragment, repeat_count)
                    })
                    .collect()
            });

        // Unversioned, flatten the items.
        if self.versions.len() == 1 && self.versions[0].name.is_empty() {
            Versions::Single(pretty::Fragment::new(
                per_item_versions_with_repeat_count
                    .map(|mut versions_with_repeat_count| {
                        versions_with_repeat_count.pop().unwrap().0
                    })
                    .filter(|fragment| !fragment.nodes.is_empty())
                    .intersperse({
                        // Separate top-level definitions with empty lines.
                        // FIXME(eddyb) have an explicit `pretty::Node`
                        // for "vertical gap" instead.
                        "\n\n".into()
                    }),
            ))
        } else {
            Versions::Multiple {
                version_names: self.versions.iter().map(|v| v.name.clone()).collect(),
                per_row_versions_with_repeat_count: per_item_versions_with_repeat_count.collect(),
            }
        }
    }
}

impl Print for Module {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        if self.exports.is_empty() {
            return pretty::Fragment::default();
        }

        pretty::Fragment::new([
            printer.declarative_keyword_style().apply("export").into(),
            " ".into(),
            pretty::join_comma_sep(
                "{",
                self.exports
                    .iter()
                    .map(|(export_key, exportee)| {
                        pretty::Fragment::new([
                            export_key.print(printer),
                            ": ".into(),
                            exportee.print(printer),
                        ])
                    })
                    .map(|entry| {
                        pretty::Fragment::new([pretty::Node::ForceLineSeparation.into(), entry])
                    }),
                "}",
            ),
        ])
    }
}

impl Print for PlanItemDef<'_> {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        match self {
            Self::ModuleDialect(dialect) => dialect.print(printer),
            Self::ModuleDebugInfo(debug_info) => debug_info.print(printer),
            Self::GlobalVar(gv_decl) => gv_decl.print(printer),
            Self::Func(func_decl) => func_decl.print(printer),
        }
    }
}

impl Print for ModuleDialect {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let dialect = match self {
            Self::Spv(dialect) => dialect.print(printer),
        };

        AttrsAndDef {
            attrs: pretty::Fragment::default(),
            def_without_name: pretty::Fragment::new([" = ".into(), dialect]),
        }
    }
}
impl Print for spv::Dialect {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let Self {
            version_major,
            version_minor,
            capabilities,
            extensions,
            addressing_model,
            memory_model,
        } = self;

        let wk = &spv::spec::Spec::get().well_known;
        pretty::Fragment::new([
            printer
                .demote_style_for_namespace_prefix(printer.spv_base_style())
                .apply("spv.")
                .into(),
            printer.spv_base_style().apply("Module").into(),
            pretty::join_comma_sep(
                "(",
                [pretty::Fragment::new([
                    printer.pretty_named_argument_prefix("version"),
                    printer.numeric_literal_style().apply(format!("{version_major}")).into(),
                    ".".into(),
                    printer.numeric_literal_style().apply(format!("{version_minor}")).into(),
                ])]
                .into_iter()
                .chain((!extensions.is_empty()).then(|| {
                    pretty::Fragment::new([
                        printer.pretty_named_argument_prefix("extensions"),
                        pretty::join_comma_sep(
                            "{",
                            extensions.iter().map(|ext| printer.pretty_string_literal(ext)),
                            "}",
                        ),
                    ])
                }))
                .chain(
                    // FIXME(eddyb) consider a `spv.Capability.{A,B,C}` style.
                    (!capabilities.is_empty()).then(|| {
                        let cap_imms = |cap| [spv::Imm::Short(wk.Capability, cap)];

                        // HACK(eddyb) construct a custom `spv.Capability.{A,B,C}`.
                        let capability_namespace_prefix = printer
                            .pretty_spv_print_tokens_for_operand({
                                let mut tokens = spv::print::operand_from_imms(cap_imms(0));
                                assert!(matches!(
                                    tokens.tokens.pop(),
                                    Some(spv::print::Token::EnumerandName(_))
                                ));
                                tokens
                            });

                        let mut cap_names = capabilities.iter().map(|&cap| {
                            printer.pretty_spv_print_tokens_for_operand({
                                let mut tokens = spv::print::operand_from_imms(cap_imms(cap));
                                tokens.tokens.drain(..tokens.tokens.len() - 1);
                                assert!(matches!(
                                    tokens.tokens[..],
                                    [spv::print::Token::EnumerandName(_)]
                                ));
                                tokens
                            })
                        });

                        pretty::Fragment::new([
                            capability_namespace_prefix,
                            if cap_names.len() == 1 {
                                cap_names.next().unwrap()
                            } else {
                                pretty::join_comma_sep("{", cap_names, "}")
                            },
                        ])
                    }),
                )
                .chain(
                    (*addressing_model != wk.Logical)
                        .then(|| printer.pretty_spv_imm(wk.AddressingModel, *addressing_model)),
                )
                .chain([printer.pretty_spv_imm(wk.MemoryModel, *memory_model)]),
                ")",
            ),
        ])
    }
}

impl Print for ModuleDebugInfo {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let debug_info = match self {
            Self::Spv(debug_info) => debug_info.print(printer),
        };

        AttrsAndDef {
            attrs: pretty::Fragment::default(),
            def_without_name: pretty::Fragment::new([" = ".into(), debug_info]),
        }
    }
}

impl Print for spv::ModuleDebugInfo {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let Self {
            original_generator_magic,
            source_languages,
            source_extensions,
            module_processes,
        } = self;

        let wk = &spv::spec::Spec::get().well_known;
        pretty::Fragment::new([
            printer
                .demote_style_for_namespace_prefix(
                    printer.demote_style_for_namespace_prefix(printer.spv_base_style()),
                )
                .apply("spv.")
                .into(),
            printer
                .demote_style_for_namespace_prefix(printer.spv_base_style())
                .apply("Module.")
                .into(),
            printer.spv_base_style().apply("DebugInfo").into(),
            pretty::join_comma_sep(
                "(",
                [
                    original_generator_magic.map(|generator_magic| {
                        let (tool_id, tool_version) =
                            (generator_magic.get() >> 16, generator_magic.get() as u16);
                        pretty::Fragment::new([
                            printer.pretty_named_argument_prefix("generator"),
                            printer
                                .demote_style_for_namespace_prefix(printer.spv_base_style())
                                .apply("spv.")
                                .into(),
                            printer.spv_base_style().apply("Tool").into(),
                            pretty::join_comma_sep(
                                "(",
                                [
                                    Some(pretty::Fragment::new([
                                        printer.pretty_named_argument_prefix("id"),
                                        printer
                                            .numeric_literal_style()
                                            .apply(format!("{tool_id}"))
                                            .into(),
                                    ])),
                                    (tool_version != 0).then(|| {
                                        pretty::Fragment::new([
                                            printer.pretty_named_argument_prefix("version"),
                                            printer
                                                .numeric_literal_style()
                                                .apply(format!("{tool_version}"))
                                                .into(),
                                        ])
                                    }),
                                ]
                                .into_iter()
                                .flatten(),
                                ")",
                            ),
                        ])
                    }),
                    (!source_languages.is_empty()).then(|| {
                        pretty::Fragment::new([
                            printer.pretty_named_argument_prefix("source_languages"),
                            pretty::join_comma_sep(
                                "{",
                                source_languages
                                    .iter()
                                    .map(|(lang, sources)| {
                                        let spv::DebugSources { file_contents } = sources;
                                        pretty::Fragment::new([
                                            printer.pretty_spv_imm(wk.SourceLanguage, lang.lang),
                                            "(".into(),
                                            printer.pretty_named_argument_prefix("version"),
                                            printer
                                                .numeric_literal_style()
                                                .apply(format!("{}", lang.version))
                                                .into(),
                                            "): ".into(),
                                            pretty::join_comma_sep(
                                                "{",
                                                file_contents
                                                    .iter()
                                                    .map(|(&file, contents)| {
                                                        pretty::Fragment::new([
                                                            printer.pretty_string_literal(
                                                                &printer.cx[file],
                                                            ),
                                                            pretty::join_space(
                                                                ":",
                                                                [printer.pretty_string_literal(
                                                                    contents,
                                                                )],
                                                            ),
                                                        ])
                                                    })
                                                    .map(|entry| {
                                                        pretty::Fragment::new([
                                                            pretty::Node::ForceLineSeparation
                                                                .into(),
                                                            entry,
                                                        ])
                                                    }),
                                                "}",
                                            ),
                                        ])
                                    })
                                    .map(|entry| {
                                        pretty::Fragment::new([
                                            pretty::Node::ForceLineSeparation.into(),
                                            entry,
                                        ])
                                    }),
                                "}",
                            ),
                        ])
                    }),
                    (!source_extensions.is_empty()).then(|| {
                        pretty::Fragment::new([
                            printer.pretty_named_argument_prefix("source_extensions"),
                            pretty::join_comma_sep(
                                "[",
                                source_extensions
                                    .iter()
                                    .map(|ext| printer.pretty_string_literal(ext)),
                                "]",
                            ),
                        ])
                    }),
                    (!module_processes.is_empty()).then(|| {
                        pretty::Fragment::new([
                            printer.pretty_named_argument_prefix("module_processes"),
                            pretty::join_comma_sep(
                                "[",
                                module_processes
                                    .iter()
                                    .map(|proc| printer.pretty_string_literal(proc)),
                                "]",
                            ),
                        ])
                    }),
                ]
                .into_iter()
                .flatten(),
                ")",
            ),
        ])
    }
}

impl Print for ExportKey {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        match self {
            &Self::LinkName(name) => printer.pretty_string_literal(&printer.cx[name]),

            // HACK(eddyb) `interface_global_vars` should be recomputed by
            // `spv::lift` anyway, so hiding them here mimics that.
            Self::SpvEntryPoint { imms, interface_global_vars: _ } => {
                let wk = &spv::spec::Spec::get().well_known;

                printer.pretty_spv_inst(printer.spv_op_style(), wk.OpEntryPoint, imms, [None])
            }
        }
    }
}

impl Print for Exportee {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        match *self {
            Self::GlobalVar(gv) => gv.print(printer),
            Self::Func(func) => func.print(printer),
        }
    }
}

impl CxInterned {
    fn print_all(printer: &Printer<'_>) -> pretty::Fragment {
        let fragments = printer
            .use_styles
            .iter()
            .filter_map(|(&use_kind, use_style)| match (use_kind, use_style) {
                (Use::CxInterned(interned), UseStyle::Anon { .. } | UseStyle::Named { .. }) => {
                    Some(interned)
                }
                _ => None,
            })
            .map(|interned| {
                interned.print(printer).insert_name_before_def(pretty::Fragment::new([
                    Use::CxInterned(interned).print_as_def(printer),
                    " = ".into(),
                ]))
            })
            .intersperse({
                // Separate top-level definitions with empty lines.
                // FIXME(eddyb) have an explicit `pretty::Node`
                // for "vertical gap" instead.
                "\n\n".into()
            });

        pretty::Fragment::new(fragments)
    }
}

impl Print for CxInterned {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        match *self {
            Self::Type(ty) => printer.cx[ty].print(printer),
            Self::Const(ct) => printer.cx[ct].print(printer),
        }
    }
}

impl Print for AttrSet {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let AttrSetDef { attrs } = &printer.cx[*self];

        // Avoid showing `#[spv.OpName("...")]` when it's already in use as the
        // name of the definition (but keep it in all other cases).
        let spv_name_to_hide = printer.attrs_with_spv_name_in_use.get(self).copied();

        pretty::Fragment::new(
            attrs
                .iter()
                .filter(|attr| match attr {
                    Attr::SpvAnnotation(spv_inst) => Some(spv_inst) != spv_name_to_hide,
                    _ => true,
                })
                .map(|attr| attr.print(printer))
                .flat_map(|entry| [entry, pretty::Node::ForceLineSeparation.into()]),
        )
    }
}

// FIXME(eddyb) maybe this could implement `Print` but it'd still be awkward.
impl DbgScope {
    fn maybe_print_name_as_ref(&self, printer: &Printer<'_>) -> Option<pretty::Fragment> {
        self.maybe_print_name_as_ref_or_def(printer, false)
    }

    fn maybe_print_name_as_ref_or_def(
        &self,
        printer: &Printer<'_>,
        is_def: bool,
    ) -> Option<pretty::Fragment> {
        let parent_func =
            printer.current_plan_item.get().and_then(|plan_item| match plan_item {
                PlanItem::Func(func) => Some(func),
                _ => None,
            })?;

        let dbg_scope_use = Use::DbgScope { scope: *self, parent_func };
        printer
            .use_styles
            .contains_key(&dbg_scope_use)
            .then(|| dbg_scope_use.print_as_ref_or_def(printer, is_def))
    }

    fn maybe_print_def(&self, printer: &Printer<'_>) -> Option<pretty::Fragment> {
        let name = self.maybe_print_name_as_ref_or_def(printer, true)?;

        let mut def = self.print_with_prefix(printer, " = ");

        // HACK(eddyb) avoid breaking `insert_name_before_def` anchor hoisting.
        def.attrs.nodes.push(printer.comment_style().apply("// "));

        Some(def.insert_name_before_def(name))
    }

    // HACK(eddyb) `AttrsAndDef` used to separate the `// at ...` comment, which
    // is treated like `Attr::DbgSrcLoc`, from the "body" of the `DbgScope` def.
    fn print_with_prefix(&self, printer: &Printer<'_>, prefix: &'static str) -> AttrsAndDef {
        let mut s = String::new();
        s += prefix;

        let DbgScope::InlinedCalleeBody { callee_name, call_site } = *self;

        s += "inlined `";

        // HACK(eddyb) not trusting non-trivial strings to behave.
        let callee_name = &printer.cx[callee_name];
        if callee_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            s += callee_name;
        } else {
            s.extend(
                callee_name
                    .escape_debug()
                    .flat_map(|c| (c == '`').then_some('\\').into_iter().chain([c])),
            );
        }

        s += "` call";

        AttrsAndDef {
            attrs: call_site.print(printer),
            def_without_name: printer.comment_style().apply(s).into(),
        }
    }
}

impl Print for Attr {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let non_comment_attr = match self {
            &Attr::DbgSrcLoc(OrdAssertEq(dbg_src_loc)) => {
                let mut comment = SmallVec::<[_; 4]>::new();

                let DbgSrcLoc {
                    file_path,
                    start_line_col: (start_line, mut start_col),
                    end_line_col: (end_line, mut end_col),
                    inlined_callee_name_and_call_site: _,
                } = dbg_src_loc;

                let dbg_scope = DbgScope::try_from_dbg_src_loc(dbg_src_loc);
                let dbg_scope_name =
                    dbg_scope.and_then(|scope| scope.maybe_print_name_as_ref(printer));
                let has_dbg_scope_name = dbg_scope_name.is_some();

                // HACK(eddyb) Rust-GPU's column numbers seem
                // off-by-one wrt what e.g. VSCode expects
                // for `:line:col` syntax, but it's hard to
                // tell from the spec and `glslang` doesn't
                // even emit column numbers at all!
                start_col += 1;
                end_col += 1;

                let mut s = String::new();
                s += "// ";

                if let Some(dbg_scope_name) = dbg_scope_name {
                    // FIXME(eddyb) this might not be the best way to refer
                    // to the `DbgScope`, but it should work for now.
                    s += "in ";
                    comment.extend([
                        printer.comment_style().apply(mem::take(&mut s)).into(),
                        dbg_scope_name,
                    ]);

                    s += " ";
                }

                s += "at ";

                // HACK(eddyb) only skip string-quoting and escaping for
                // well-behaved file paths.
                let file_path = &printer.cx[file_path];
                if file_path.chars().all(|c| c.is_ascii_graphic() && c != ':') {
                    s += file_path;
                } else {
                    write!(s, "{file_path:?}").unwrap();
                }

                // HACK(eddyb) the syntaxes used are taken from VSCode, i.e.:
                // https://github.com/microsoft/vscode/blob/6b924c5/src/vs/workbench/contrib/terminalContrib/links/browser/terminalLinkParsing.ts#L75-L91
                // (using the most boring syntax possible for every situation).
                let is_quoted = s.ends_with('"');
                let is_range = (start_line, start_col) != (end_line, end_col);
                write!(
                    s,
                    "{}{start_line}{}{start_col}",
                    if is_quoted { ',' } else { ':' },
                    if is_quoted && is_range { '.' } else { ':' }
                )
                .unwrap();
                if is_range {
                    s += "-";
                    if start_line != end_line {
                        write!(s, "{end_line}.").unwrap();
                    }
                    write!(s, "{end_col}").unwrap();
                }
                comment.push(printer.comment_style().apply(s).into());

                let comment = pretty::Fragment::new(comment);

                if let Some(dbg_scope) = dbg_scope {
                    // HACK(eddyb) only print the `DbgScope` inline if
                    // it lacked a `Use::DbgScope` to refer to, earlier.
                    if !has_dbg_scope_name {
                        // HACK(eddyb) chain `DbgScope`s by putting more important
                        // details (`file:line:col`) at the start of each comment,
                        // and less important ones (e.g. an inlined callee's name),
                        // at the end (even if the `DbgScope` is the only part
                        // which is affected by the comment just above this one).
                        return dbg_scope
                            .print_with_prefix(printer, " in ")
                            .insert_name_before_def(comment);
                    }
                }

                return comment;
            }

            Attr::Diagnostics(diags) => {
                return pretty::Fragment::new(
                    diags
                        .0
                        .iter()
                        .map(|diag| {
                            let Diag { level, message } = diag;

                            // FIXME(eddyb) the plan was to use 💥/⮿/⚠
                            // for bug/error/warning, but it doesn't really
                            // render correctly, so allcaps it is for now.
                            let (icon, icon_color) = match level {
                                DiagLevel::Bug(_) => ("BUG", pretty::palettes::simple::MAGENTA),
                                DiagLevel::Error => ("ERR", pretty::palettes::simple::RED),
                                DiagLevel::Warning => ("WARN", pretty::palettes::simple::YELLOW),
                            };

                            let grayish =
                                |[r, g, b]: [u8; 3]| [(r / 2) + 64, (g / 2) + 64, (b / 2) + 64];
                            let comment_style = pretty::Styles::color(grayish(icon_color));

                            // FIXME(eddyb) maybe make this a link to the source code?
                            let bug_location_prefix = match level {
                                DiagLevel::Bug(location) => {
                                    let location = location.to_string();
                                    let location = match location.rsplit_once("/src/") {
                                        Some((_path_prefix, intra_src)) => intra_src,
                                        None => &location,
                                    };
                                    comment_style.apply(format!("[{location}] ")).into()
                                }
                                DiagLevel::Error | DiagLevel::Warning => {
                                    pretty::Fragment::default()
                                }
                            };

                            let mut printed_message = message.print(printer);

                            // HACK(eddyb) apply the right style to all the plain
                            // text parts of the already-printed message.
                            // FIXME(eddyb) consider merging the styles somewhat?
                            for node in &mut printed_message.nodes {
                                if let pretty::Node::Text(style @ None, _) = node {
                                    *style = Some(comment_style);
                                }
                            }

                            // HACK(eddyb) this would ideally use line comments,
                            // but adding the line prefix properly to everything
                            // is a bit of a pain without special `pretty` support.
                            pretty::Fragment::new([
                                comment_style.apply("/*"),
                                pretty::Node::BreakingOnlySpace,
                                pretty::Node::InlineOrIndentedBlock(vec![pretty::Fragment::new([
                                    pretty::Styles {
                                        thickness: Some(3),

                                        // HACK(eddyb) this allows larger "icons"
                                        // without adding gaps via `line-height`.
                                        subscript: true,
                                        size: Some(2),

                                        ..pretty::Styles::color(icon_color)
                                    }
                                    .apply(icon)
                                    .into(),
                                    " ".into(),
                                    bug_location_prefix,
                                    printed_message,
                                ])]),
                                pretty::Node::BreakingOnlySpace,
                                comment_style.apply("*/"),
                            ])
                        })
                        .intersperse(pretty::Node::ForceLineSeparation.into()),
                );
            }

            Attr::QPtr(attr) => {
                let (name, params_inputs) = match attr {
                    QPtrAttr::ToSpvPtrInput { input_idx, pointee } => (
                        "to_spv_ptr_input",
                        pretty::Fragment::new([pretty::join_comma_sep(
                            "(",
                            [
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("input_idx"),
                                    printer
                                        .numeric_literal_style()
                                        .apply(format!("{input_idx}"))
                                        .into(),
                                ]),
                                pointee.0.print(printer),
                            ],
                            ")",
                        )]),
                    ),

                    QPtrAttr::FromSpvPtrOutput { addr_space, pointee } => (
                        "from_spv_ptr_output",
                        pretty::join_comma_sep(
                            "(",
                            [addr_space.0.print(printer), pointee.0.print(printer)],
                            ")",
                        ),
                    ),

                    QPtrAttr::Usage(usage) => {
                        ("usage", pretty::join_comma_sep("(", [usage.0.print(printer)], ")"))
                    }
                };
                pretty::Fragment::new([
                    printer
                        .demote_style_for_namespace_prefix(printer.attr_style())
                        .apply("qptr.")
                        .into(),
                    printer.attr_style().apply(name).into(),
                    params_inputs,
                ])
            }

            Attr::SpvAnnotation(spv::Inst { opcode, imms }) => {
                let wk = &spv::spec::Spec::get().well_known;

                // HACK(eddyb) `#[spv.OpDecorate(...)]` is redundant (with its operand).
                if [wk.OpDecorate, wk.OpDecorateString, wk.OpExecutionMode].contains(opcode) {
                    printer.pretty_spv_operand_from_imms(imms.iter().copied())
                } else if *opcode == wk.OpName {
                    // HACK(eddyb) unlike `OpDecorate`, we can't just omit `OpName`,
                    // but pretending it's a SPIR-T-specific `#[name = "..."]`
                    // attribute should be good enough for now.
                    pretty::Fragment::new([
                        printer.attr_style().apply("name = ").into(),
                        printer.pretty_spv_operand_from_imms(imms.iter().copied()),
                    ])
                } else {
                    printer.pretty_spv_inst(printer.attr_style(), *opcode, imms, [None])
                }
            }
            &Attr::SpvBitflagsOperand(imm) => printer.pretty_spv_operand_from_imms([imm]),
        };
        pretty::Fragment::new([
            printer.attr_style().apply("#[").into(),
            non_comment_attr,
            printer.attr_style().apply("]").into(),
        ])
    }
}

impl Print for Vec<DiagMsgPart> {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        pretty::Fragment::new(self.iter().map(|part| match part {
            DiagMsgPart::Plain(text) => pretty::Node::Text(None, text.clone()).into(),
            DiagMsgPart::Attrs(attrs) => attrs.print(printer),
            DiagMsgPart::Type(ty) => ty.print(printer),
            DiagMsgPart::Const(ct) => ct.print(printer),
            DiagMsgPart::QPtrUsage(usage) => usage.print(printer),
        }))
    }
}

impl Print for QPtrUsage {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        match self {
            QPtrUsage::Handles(qptr::shapes::Handle::Opaque(ty)) => ty.print(printer),
            QPtrUsage::Handles(qptr::shapes::Handle::Buffer(_, data_usage)) => {
                pretty::Fragment::new([
                    printer.declarative_keyword_style().apply("buffer_data").into(),
                    pretty::join_comma_sep("(", [data_usage.print(printer)], ")"),
                ])
            }
            QPtrUsage::Memory(usage) => usage.print(printer),
        }
    }
}

impl Print for QPtrMemUsage {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        // FIXME(eddyb) should this be a helper on `Printer`?
        let num_lit = |x: u32| printer.numeric_literal_style().apply(format!("{x}")).into();

        match &self.kind {
            QPtrMemUsageKind::Unused => "_".into(),
            // FIXME(eddyb) should the distinction be noted?
            &QPtrMemUsageKind::StrictlyTyped(ty) | &QPtrMemUsageKind::DirectAccess(ty) => {
                ty.print(printer)
            }
            QPtrMemUsageKind::OffsetBase(entries) => pretty::join_comma_sep(
                "{",
                entries
                    .iter()
                    .map(|(&offset, sub_usage)| {
                        pretty::Fragment::new([
                            num_lit(offset),
                            "..".into(),
                            sub_usage
                                .max_size
                                .and_then(|max_size| offset.checked_add(max_size))
                                .map(num_lit)
                                .unwrap_or_default(),
                            " => ".into(),
                            sub_usage.print(printer),
                        ])
                    })
                    .map(|entry| {
                        pretty::Fragment::new([pretty::Node::ForceLineSeparation.into(), entry])
                    }),
                "}",
            ),
            QPtrMemUsageKind::DynOffsetBase { element, stride } => pretty::Fragment::new([
                "(".into(),
                num_lit(0),
                "..".into(),
                self.max_size
                    .map(|max_size| max_size / stride.get())
                    .map(num_lit)
                    .unwrap_or_default(),
                ") × ".into(),
                num_lit(stride.get()),
                " => ".into(),
                element.print(printer),
            ]),
        }
    }
}

impl Print for TypeDef {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let Self { attrs, kind } = self;

        let wk = &spv::spec::Spec::get().well_known;

        // FIXME(eddyb) should this be done by lowering SPIR-V types to SPIR-T?
        let kw = |kw| printer.declarative_keyword_style().apply(kw).into();
        let compact_def = if let &TypeKind::SpvInst {
            spv_inst: spv::Inst { opcode, ref imms },
            ref type_and_const_inputs,
        } = kind
        {
            if opcode == wk.OpTypeBool {
                Some(kw("bool".into()))
            } else if opcode == wk.OpTypeInt {
                let (width, signed) = match imms[..] {
                    [spv::Imm::Short(_, width), spv::Imm::Short(_, signedness)] => {
                        (width, signedness != 0)
                    }
                    _ => unreachable!(),
                };

                Some(if signed { kw(format!("s{width}")) } else { kw(format!("u{width}")) })
            } else if opcode == wk.OpTypeFloat {
                let width = match imms[..] {
                    [spv::Imm::Short(_, width)] => width,
                    _ => unreachable!(),
                };

                Some(kw(format!("f{width}")))
            } else if opcode == wk.OpTypeVector {
                let (elem_ty, elem_count) = match (&imms[..], &type_and_const_inputs[..]) {
                    (&[spv::Imm::Short(_, elem_count)], &[TypeOrConst::Type(elem_ty)]) => {
                        (elem_ty, elem_count)
                    }
                    _ => unreachable!(),
                };

                Some(pretty::Fragment::new([
                    elem_ty.print(printer),
                    "×".into(),
                    printer.numeric_literal_style().apply(format!("{elem_count}")).into(),
                ]))
            } else {
                None
            }
        } else {
            None
        };

        AttrsAndDef {
            attrs: attrs.print(printer),
            def_without_name: if let Some(def) = compact_def {
                def
            } else {
                match kind {
                    // FIXME(eddyb) should this be shortened to `qtr`?
                    TypeKind::QPtr => printer.declarative_keyword_style().apply("qptr").into(),

                    TypeKind::SpvInst { spv_inst, type_and_const_inputs } => printer
                        .pretty_spv_inst(
                            printer.spv_op_style(),
                            spv_inst.opcode,
                            &spv_inst.imms,
                            type_and_const_inputs.iter().map(|&ty_or_ct| match ty_or_ct {
                                TypeOrConst::Type(ty) => ty.print(printer),
                                TypeOrConst::Const(ct) => ct.print(printer),
                            }),
                        ),
                    TypeKind::SpvStringLiteralForExtInst => pretty::Fragment::new([
                        printer.error_style().apply("type_of").into(),
                        "(".into(),
                        printer.pretty_spv_opcode(printer.spv_op_style(), wk.OpString),
                        ")".into(),
                    ]),
                }
            },
        }
    }
}

impl Print for ConstDef {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let Self { attrs, ty, kind } = self;

        let wk = &spv::spec::Spec::get().well_known;

        let kw = |kw| printer.declarative_keyword_style().apply(kw).into();
        let literal_ty_suffix = |ty| {
            pretty::Styles {
                // HACK(eddyb) the exact type detracts from the value.
                color_opacity: Some(0.4),
                subscript: true,
                ..printer.declarative_keyword_style()
            }
            .apply(ty)
        };
        let compact_def = if let ConstKind::SpvInst { spv_inst_and_const_inputs } = kind {
            let (spv_inst, _const_inputs) = &**spv_inst_and_const_inputs;
            let &spv::Inst { opcode, ref imms } = spv_inst;

            if opcode == wk.OpConstantFalse {
                Some(kw("false"))
            } else if opcode == wk.OpConstantTrue {
                Some(kw("true"))
            } else if opcode == wk.OpConstant {
                // HACK(eddyb) it's simpler to only handle a limited subset of
                // integer/float bit-widths, for now.
                let raw_bits = match imms[..] {
                    [spv::Imm::Short(_, x)] => Some(u64::from(x)),
                    [spv::Imm::LongStart(_, lo), spv::Imm::LongCont(_, hi)] => {
                        Some(u64::from(lo) | (u64::from(hi) << 32))
                    }
                    _ => None,
                };

                if let (
                    Some(raw_bits),
                    &TypeKind::SpvInst {
                        spv_inst: spv::Inst { opcode: ty_opcode, imms: ref ty_imms },
                        ..
                    },
                ) = (raw_bits, &printer.cx[*ty].kind)
                {
                    if ty_opcode == wk.OpTypeInt {
                        let (width, signed) = match ty_imms[..] {
                            [spv::Imm::Short(_, width), spv::Imm::Short(_, signedness)] => {
                                (width, signedness != 0)
                            }
                            _ => unreachable!(),
                        };

                        if width <= 64 {
                            let (printed_value, ty) = if signed {
                                let sext_raw_bits =
                                    (raw_bits as u128 as i128) << (128 - width) >> (128 - width);
                                // FIXME(eddyb) consider supporting negative hex.
                                (
                                    if sext_raw_bits >= 0 {
                                        printer.pretty_numeric_literal_as_dec_or_hex(
                                            sext_raw_bits as u128,
                                        )
                                    } else {
                                        printer
                                            .numeric_literal_style()
                                            .apply(format!("{sext_raw_bits}"))
                                            .into()
                                    },
                                    format!("s{width}"),
                                )
                            } else {
                                (
                                    printer.pretty_numeric_literal_as_dec_or_hex(raw_bits.into()),
                                    format!("u{width}"),
                                )
                            };
                            Some(pretty::Fragment::new([
                                printed_value,
                                literal_ty_suffix(ty).into(),
                            ]))
                        } else {
                            None
                        }
                    } else if ty_opcode == wk.OpTypeFloat {
                        let width = match ty_imms[..] {
                            [spv::Imm::Short(_, width)] => width,
                            _ => unreachable!(),
                        };

                        /// Check that parsing the result of printing produces
                        /// the original bits of the floating-point value, and
                        /// only return `Some` if that is the case.
                        fn bitwise_roundtrip_float_print<
                            BITS: Copy + PartialEq,
                            FLOAT: std::fmt::Debug + std::str::FromStr,
                        >(
                            bits: BITS,
                            float_from_bits: impl FnOnce(BITS) -> FLOAT,
                            float_to_bits: impl FnOnce(FLOAT) -> BITS,
                        ) -> Option<String> {
                            let float = float_from_bits(bits);
                            Some(format!("{float:?}")).filter(|s| {
                                s.parse::<FLOAT>()
                                    .map(float_to_bits)
                                    .is_ok_and(|roundtrip_bits| roundtrip_bits == bits)
                            })
                        }

                        let printed_value = match width {
                            32 => bitwise_roundtrip_float_print(
                                raw_bits as u32,
                                f32::from_bits,
                                f32::to_bits,
                            ),
                            64 => bitwise_roundtrip_float_print(
                                raw_bits,
                                f64::from_bits,
                                f64::to_bits,
                            ),
                            _ => None,
                        };
                        printed_value.map(|s| {
                            pretty::Fragment::new([
                                printer.numeric_literal_style().apply(s),
                                literal_ty_suffix(format!("f{width}")),
                            ])
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        AttrsAndDef {
            attrs: attrs.print(printer),
            def_without_name: compact_def.unwrap_or_else(|| match kind {
                &ConstKind::PtrToGlobalVar(gv) => {
                    pretty::Fragment::new(["&".into(), gv.print(printer)])
                }
                ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                    let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                    pretty::Fragment::new([
                        printer.pretty_spv_inst(
                            printer.spv_op_style(),
                            spv_inst.opcode,
                            &spv_inst.imms,
                            const_inputs.iter().map(|ct| ct.print(printer)),
                        ),
                        printer.pretty_type_ascription_suffix(*ty),
                    ])
                }
                &ConstKind::SpvStringLiteralForExtInst(s) => pretty::Fragment::new([
                    printer.pretty_spv_opcode(printer.spv_op_style(), wk.OpString),
                    "(".into(),
                    printer.pretty_string_literal(&printer.cx[s]),
                    ")".into(),
                ]),
            }),
        }
    }
}

impl Print for Import {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        match self {
            &Self::LinkName(name) => pretty::Fragment::new([
                printer.declarative_keyword_style().apply("import").into(),
                " ".into(),
                printer.pretty_string_literal(&printer.cx[name]),
            ]),
        }
    }
}

impl Print for GlobalVarDecl {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let Self { attrs, type_of_ptr_to, shape, addr_space, def } = self;

        let wk = &spv::spec::Spec::get().well_known;

        // HACK(eddyb) to avoid too many syntax variations, most details (other
        // than the type, if present) use named arguments in `GV123(...)`.
        let mut details = SmallVec::<[_; 4]>::new();

        match addr_space {
            AddrSpace::Handles => {}
            AddrSpace::SpvStorageClass(_) => {
                details.push(addr_space.print(printer));
            }
        }

        // FIXME(eddyb) should this be a helper on `Printer`?
        let num_lit = |x: u32| printer.numeric_literal_style().apply(format!("{x}")).into();

        // FIXME(eddyb) should the pointer type be shown as something like
        // `&GV123: OpTypePointer(..., T123)` *after* the variable definition?
        // (but each reference can technically have a different type...)
        let (qptr_shape, spv_ptr_pointee_type) = match &printer.cx[*type_of_ptr_to].kind {
            TypeKind::QPtr => (shape.as_ref(), None),

            // HACK(eddyb) get the pointee type from SPIR-V `OpTypePointer`, but
            // ideally the `GlobalVarDecl` would hold that type itself.
            TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
                if spv_inst.opcode == wk.OpTypePointer =>
            {
                match type_and_const_inputs[..] {
                    [TypeOrConst::Type(pointee_type)] => (None, Some(pointee_type)),
                    _ => (None, None),
                }
            }

            _ => (None, None),
        };
        let ascribe_type = match qptr_shape {
            Some(qptr::shapes::GlobalVarShape::Handles { handle, fixed_count }) => {
                let handle = match handle {
                    qptr::shapes::Handle::Opaque(ty) => ty.print(printer),
                    qptr::shapes::Handle::Buffer(addr_space, buf) => pretty::Fragment::new([
                        printer.declarative_keyword_style().apply("buffer").into(),
                        pretty::join_comma_sep(
                            "(",
                            [
                                addr_space.print(printer),
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("size"),
                                    pretty::Fragment::new(
                                        [
                                            Some(buf.fixed_base.size)
                                                .filter(|&base_size| {
                                                    base_size > 0 || buf.dyn_unit_stride.is_none()
                                                })
                                                .map(num_lit),
                                            buf.dyn_unit_stride.map(|stride| {
                                                pretty::Fragment::new([
                                                    "N × ".into(),
                                                    num_lit(stride.get()),
                                                ])
                                            }),
                                        ]
                                        .into_iter()
                                        .flatten()
                                        .intersperse_with(|| " + ".into()),
                                    ),
                                ]),
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("align"),
                                    num_lit(buf.fixed_base.align),
                                ]),
                            ],
                            ")",
                        ),
                    ]),
                };

                let handles = if fixed_count.map_or(0, |c| c.get()) == 1 {
                    handle
                } else {
                    pretty::Fragment::new([
                        "[".into(),
                        fixed_count
                            .map(|count| {
                                pretty::Fragment::new([num_lit(count.get()), " × ".into()])
                            })
                            .unwrap_or_default(),
                        handle,
                        "]".into(),
                    ])
                };
                Some(handles)
            }
            Some(qptr::shapes::GlobalVarShape::UntypedData(mem_layout)) => {
                details.extend([
                    pretty::Fragment::new([
                        printer.pretty_named_argument_prefix("size"),
                        num_lit(mem_layout.size),
                    ]),
                    pretty::Fragment::new([
                        printer.pretty_named_argument_prefix("align"),
                        num_lit(mem_layout.align),
                    ]),
                ]);
                None
            }
            Some(qptr::shapes::GlobalVarShape::TypedInterface(ty)) => Some(ty.print(printer)),

            None => Some(match spv_ptr_pointee_type {
                Some(ty) => ty.print(printer),
                None => pretty::Fragment::new([
                    printer.error_style().apply("pointee_type_of").into(),
                    "(".into(),
                    type_of_ptr_to.print(printer),
                    ")".into(),
                ]),
            }),
        };

        let import = match def {
            // FIXME(eddyb) deduplicate with `FuncDecl`, and maybe consider
            // putting the import *before* the declaration, to end up with:
            // import "..."
            //   as global_var GV...
            DeclDef::Imported(import) => Some(import.print(printer)),
            DeclDef::Present(GlobalVarDefBody { initializer }) => {
                if let Some(initializer) = initializer {
                    details.push(pretty::Fragment::new([
                        printer.pretty_named_argument_prefix("init"),
                        initializer.print(printer),
                    ]));
                }
                None
            }
        };

        let def_without_name = pretty::Fragment::new(
            [
                (!details.is_empty()).then(|| pretty::join_comma_sep("(", details, ")")),
                ascribe_type.map(|ty| pretty::join_space(":", [ty])),
                import.map(|import| pretty::Fragment::new([" = ".into(), import])),
            ]
            .into_iter()
            .flatten(),
        );

        AttrsAndDef { attrs: attrs.print(printer), def_without_name }
    }
}

impl Print for AddrSpace {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        match *self {
            AddrSpace::Handles => printer.declarative_keyword_style().apply("handles").into(),
            AddrSpace::SpvStorageClass(sc) => {
                let wk = &spv::spec::Spec::get().well_known;
                printer.pretty_spv_imm(wk.StorageClass, sc)
            }
        }
    }
}

impl Print for FuncDecl {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let Self { attrs, ret_type, params, def } = self;

        let sig = pretty::Fragment::new([
            pretty::join_comma_sep(
                "(",
                params.iter().enumerate().map(|(i, param)| {
                    let param_name = match def {
                        DeclDef::Imported(_) => "_".into(),
                        DeclDef::Present(def) => Value::RegionInput {
                            region: def.body,
                            input_idx: i.try_into().unwrap(),
                        }
                        .print_as_def(printer),
                    };
                    param.print(printer).insert_name_before_def(param_name)
                }),
                ")",
            ),
            " -> ".into(),
            ret_type.print(printer),
        ]);

        let def_without_name = match def {
            // FIXME(eddyb) deduplicate with `GlobalVarDecl`, and maybe consider
            // putting the import *before* the declaration, to end up with:
            // import "..."
            //   as func F...
            DeclDef::Imported(import) => pretty::Fragment::new([
                sig,
                pretty::join_space(
                    "",
                    [pretty::Fragment::new(["= ".into(), import.print(printer)])],
                ),
            ]),

            // FIXME(eddyb) this can probably go into `impl Print for FuncDefBody`.
            DeclDef::Present(def) => pretty::Fragment::new([
                sig,
                " {".into(),
                pretty::Node::IndentedBlock(match &def.unstructured_cfg {
                    None => vec![def.at_body().print(printer)],
                    Some(cfg) => cfg
                        .rev_post_order(def)
                        .map(|region| {
                            let label = Use::RegionLabel(region);
                            let label_header = if printer.use_styles.contains_key(&label) {
                                let inputs = &def.at(region).def().inputs;
                                let label_inputs = if !inputs.is_empty() {
                                    pretty::join_comma_sep(
                                        "(",
                                        inputs.iter().enumerate().map(|(input_idx, input)| {
                                            input.print(printer).insert_name_before_def(
                                                Value::RegionInput {
                                                    region,
                                                    input_idx: input_idx.try_into().unwrap(),
                                                }
                                                .print_as_def(printer),
                                            )
                                        }),
                                        ")",
                                    )
                                } else {
                                    pretty::Fragment::default()
                                };

                                // FIXME(eddyb) `:` as used here for C-like "label syntax"
                                // interferes (in theory) with `e: T` "type ascription syntax".
                                pretty::Fragment::new([
                                    pretty::Node::ForceLineSeparation.into(),
                                    label.print_as_def(printer),
                                    label_inputs,
                                    ":".into(),
                                    pretty::Node::ForceLineSeparation.into(),
                                ])
                            } else {
                                pretty::Fragment::default()
                            };

                            pretty::Fragment::new([
                                label_header,
                                pretty::Node::IndentedBlock(vec![def.at(region).print(printer)])
                                    .into(),
                                cfg.control_inst_on_exit_from[region].print(printer),
                            ])
                        })
                        .intersperse({
                            // Separate (top-level) nodes with empty lines.
                            // FIXME(eddyb) have an explicit `pretty::Node`
                            // for "vertical gap" instead.
                            "\n\n".into()
                        })
                        .collect(),
                })
                .into(),
                "}".into(),
            ]),
        };

        AttrsAndDef { attrs: attrs.print(printer), def_without_name }
    }
}

impl Print for FuncParam {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let Self { attrs, ty } = *self;

        AttrsAndDef {
            attrs: attrs.print(printer),
            def_without_name: printer.pretty_type_ascription_suffix(ty),
        }
    }
}

impl Print for FuncAt<'_, Region> {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let RegionDef { inputs: _, children, outputs } = self.def();

        // NOTE(eddyb) `inputs` are always printed by the parent.

        // HACK(eddyb) have to track which `DbgScope`s have been printed already,
        // due to potential duplication due to different versions' placements.
        let mut seen_dbg_scope_defs = SmallIndexSet::<_, 4>::default();
        let dbg_scope_def_placements =
            printer.per_region_dbg_scope_def_placements.get(self.position);
        let mut dbg_scope_defs_at = |intra_region: DbgScopeDefPlaceInRegion| {
            let mut relevant_dbg_scope_defs = dbg_scope_def_placements
                .and_then(|placements| placements.get(&intra_region))
                .into_iter()
                .flat_map(|dbg_scopes| dbg_scopes.iter().copied())
                .filter(|&dbg_scope| seen_dbg_scope_defs.insert(dbg_scope))
                .filter_map(|dbg_scope| dbg_scope.maybe_print_def(printer));

            // HACK(eddyb) allow the caller to tell apart the empty case.
            let relevant_dbg_scope_defs =
                [relevant_dbg_scope_defs.next()?].into_iter().chain(relevant_dbg_scope_defs);

            Some(pretty::Fragment::new(
                relevant_dbg_scope_defs
                    .flat_map(|comment| [comment, pretty::Node::ForceLineSeparation.into()]),
            ))
        };

        let header = dbg_scope_defs_at(DbgScopeDefPlaceInRegion { before_node: None })
            .map(|dbg_scope_defs| {
                pretty::Fragment::new([
                    dbg_scope_defs,
                    // HACK(eddyb) separate hoisted `DbgScope`s with an empty line.
                    // FIXME(eddyb) have an explicit `pretty::Node`
                    // for "vertical gap" instead.
                    "\n\n".into(),
                ])
            })
            .unwrap_or_default();

        let body = (self.at(*children).into_iter())
            .map(|func_at_node| {
                pretty::Fragment::new([
                    dbg_scope_defs_at(DbgScopeDefPlaceInRegion {
                        before_node: Some(func_at_node.position),
                    })
                    .unwrap_or_default(),
                    func_at_node.print(printer),
                ])
            })
            .intersperse(pretty::Node::ForceLineSeparation.into());

        let outputs_footer = if !outputs.is_empty() {
            let mut outputs = outputs.iter().map(|v| v.print(printer));
            let outputs = if outputs.len() == 1 {
                outputs.next().unwrap()
            } else {
                pretty::join_comma_sep("(", outputs, ")")
            };
            pretty::Fragment::new([pretty::Node::ForceLineSeparation.into(), outputs])
        } else {
            pretty::Fragment::default()
        };

        pretty::Fragment::new(
            [Use::AlignmentAnchorForRegion(self.position).print_as_def(printer)]
                .into_iter()
                .chain([header])
                .chain(body)
                .chain([outputs_footer]),
        )
    }
}

impl Print for FuncAt<'_, Node> {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let node = self.position;
        let NodeDef { kind, outputs } = self.def();

        let outputs_header = if !outputs.is_empty() {
            let mut outputs = outputs.iter().enumerate().map(|(output_idx, output)| {
                output.print(printer).insert_name_before_def(
                    Value::NodeOutput { node, output_idx: output_idx.try_into().unwrap() }
                        .print_as_def(printer),
                )
            });
            let outputs_lhs = if outputs.len() == 1 {
                outputs.next().unwrap()
            } else {
                pretty::join_comma_sep("(", outputs, ")")
            };
            pretty::Fragment::new([outputs_lhs, " = ".into()])
        } else {
            pretty::Fragment::default()
        };

        // FIXME(eddyb) using `declarative_keyword_style` seems more
        // appropriate here, but it's harder to spot at a glance.
        let kw_style = printer.imperative_keyword_style();
        let kw = |kw| kw_style.apply(kw).into();
        let node_body = match kind {
            NodeKind::Block { insts } => {
                assert!(outputs.is_empty());

                pretty::Fragment::new(
                    self.at(*insts)
                        .into_iter()
                        .map(|func_at_inst| func_at_inst.print(printer))
                        .flat_map(|entry| [pretty::Node::ForceLineSeparation.into(), entry]),
                )
            }
            NodeKind::Select { kind, scrutinee, cases } => kind.print_with_scrutinee_and_cases(
                printer,
                kw_style,
                *scrutinee,
                cases.iter().map(|&case| self.at(case).print(printer)),
            ),
            NodeKind::Loop { initial_inputs, body, repeat_condition } => {
                assert!(outputs.is_empty());

                let inputs = &self.at(*body).def().inputs;
                assert_eq!(initial_inputs.len(), inputs.len());

                // FIXME(eddyb) this avoids customizing how `body` is printed,
                // by adding a `-> ...` suffix to it instead, e.g. this `body`:
                // ```
                // v3 = ...
                // v4 = ...
                // (v3, v4)
                // ```
                // may be printed like this, as part of a loop:
                // ```
                // loop(v1 <- 0, v2 <- false) {
                //   v3 = ...
                //   v4 = ...
                //   (v3, v4) -> (v1, v2)
                // }
                // ```
                // In the above example, `v1` and `v2` are the `inputs` of the
                // `body`, which start at `0`/`false`, and are replaced with
                // `v3`/`v4` after each iteration.
                let (inputs_header, body_suffix) = if !inputs.is_empty() {
                    let input_decls_and_uses =
                        inputs.iter().enumerate().map(|(input_idx, input)| {
                            (
                                input,
                                Value::RegionInput {
                                    region: *body,
                                    input_idx: input_idx.try_into().unwrap(),
                                },
                            )
                        });
                    (
                        pretty::join_comma_sep(
                            "(",
                            input_decls_and_uses.clone().zip(initial_inputs).map(
                                |((input_decl, input_use), initial)| {
                                    pretty::Fragment::new([
                                        input_decl.print(printer).insert_name_before_def(
                                            input_use.print_as_def(printer),
                                        ),
                                        " <- ".into(),
                                        initial.print(printer),
                                    ])
                                },
                            ),
                            ")",
                        ),
                        pretty::Fragment::new([" -> ".into(), {
                            let mut input_dests =
                                input_decls_and_uses.map(|(_, input_use)| input_use.print(printer));
                            if input_dests.len() == 1 {
                                input_dests.next().unwrap()
                            } else {
                                pretty::join_comma_sep("(", input_dests, ")")
                            }
                        }]),
                    )
                } else {
                    (pretty::Fragment::default(), pretty::Fragment::default())
                };

                // FIXME(eddyb) this is a weird mishmash of Rust and C syntax.
                pretty::Fragment::new([
                    kw("loop"),
                    inputs_header,
                    " {".into(),
                    pretty::Node::IndentedBlock(vec![pretty::Fragment::new([
                        self.at(*body).print(printer),
                        body_suffix,
                    ])])
                    .into(),
                    "} ".into(),
                    kw("while"),
                    " ".into(),
                    repeat_condition.print(printer),
                ])
            }
            NodeKind::ExitInvocation {
                kind: cfg::ExitInvocationKind::SpvInst(spv::Inst { opcode, imms }),
                inputs,
            } => printer.pretty_spv_inst(
                kw_style,
                *opcode,
                imms,
                inputs.iter().map(|v| v.print(printer)),
            ),
        };
        pretty::Fragment::new([
            Use::AlignmentAnchorForNode(self.position).print_as_def(printer),
            outputs_header,
            node_body,
        ])
    }
}

impl Print for RegionInputDecl {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let Self { attrs, ty } = *self;

        AttrsAndDef {
            attrs: attrs.print(printer),
            def_without_name: printer.pretty_type_ascription_suffix(ty),
        }
    }
}

impl Print for NodeOutputDecl {
    type Output = AttrsAndDef;
    fn print(&self, printer: &Printer<'_>) -> AttrsAndDef {
        let Self { attrs, ty } = *self;

        AttrsAndDef {
            attrs: attrs.print(printer),
            def_without_name: printer.pretty_type_ascription_suffix(ty),
        }
    }
}

impl Print for FuncAt<'_, DataInst> {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let DataInstDef { attrs, kind, inputs, output_type } = self.def();

        let attrs = attrs.print(printer);

        let mut output_use_to_print_as_lhs =
            output_type.map(|_| Use::DataInstOutput(self.position));

        let mut output_type_to_print = *output_type;

        let def_without_type = match kind {
            &DataInstKind::FuncCall(func) => pretty::Fragment::new([
                printer.declarative_keyword_style().apply("call").into(),
                " ".into(),
                func.print(printer),
                pretty::join_comma_sep("(", inputs.iter().map(|v| v.print(printer)), ")"),
            ]),

            DataInstKind::QPtr(op) => {
                let (qptr_input, extra_inputs) = match op {
                    // HACK(eddyb) `FuncLocalVar` should probably not even be in `QPtrOp`.
                    QPtrOp::FuncLocalVar(_) => (None, &inputs[..]),
                    _ => (Some(inputs[0]), &inputs[1..]),
                };
                let (name, extra_inputs): (_, SmallVec<[_; 1]>) = match op {
                    QPtrOp::FuncLocalVar(mem_layout) => {
                        assert!(extra_inputs.len() <= 1);
                        (
                            "func_local_var",
                            [
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("size"),
                                    printer
                                        .numeric_literal_style()
                                        .apply(mem_layout.size.to_string())
                                        .into(),
                                ]),
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("align"),
                                    printer
                                        .numeric_literal_style()
                                        .apply(mem_layout.align.to_string())
                                        .into(),
                                ]),
                            ]
                            .into_iter()
                            .chain(extra_inputs.first().map(|&init| {
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("initializer"),
                                    init.print(printer),
                                ])
                            }))
                            .collect(),
                        )
                    }

                    QPtrOp::HandleArrayIndex => {
                        assert_eq!(extra_inputs.len(), 1);
                        (
                            "handle_array_index",
                            [extra_inputs[0].print(printer)].into_iter().collect(),
                        )
                    }
                    QPtrOp::BufferData => {
                        assert_eq!(extra_inputs.len(), 0);
                        ("buffer_data", [].into_iter().collect())
                    }
                    &QPtrOp::BufferDynLen { fixed_base_size, dyn_unit_stride } => {
                        assert_eq!(extra_inputs.len(), 0);

                        // FIXME(eddyb) this isn't very nice, but without mapping
                        // to actual integer ops, there's not a lot of options.
                        (
                            "buffer_dyn_len",
                            [
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("fixed_base_size"),
                                    printer
                                        .numeric_literal_style()
                                        .apply(fixed_base_size.to_string())
                                        .into(),
                                ]),
                                pretty::Fragment::new([
                                    printer.pretty_named_argument_prefix("dyn_unit_stride"),
                                    printer
                                        .numeric_literal_style()
                                        .apply(dyn_unit_stride.to_string())
                                        .into(),
                                ]),
                            ]
                            .into_iter()
                            .collect(),
                        )
                    }

                    QPtrOp::Offset(offset) => {
                        assert_eq!(extra_inputs.len(), 0);
                        (
                            "offset",
                            [printer.numeric_literal_style().apply(format!("{offset}")).into()]
                                .into_iter()
                                .collect(),
                        )
                    }
                    &QPtrOp::DynOffset { stride, index_bounds: _ } => {
                        assert_eq!(extra_inputs.len(), 1);
                        (
                            "dyn_offset",
                            [pretty::Fragment::new([
                                extra_inputs[0].print(printer),
                                " × ".into(),
                                printer.numeric_literal_style().apply(format!("{stride}")).into(),
                            ])]
                            .into_iter()
                            .collect(),
                        )
                    }

                    QPtrOp::Load => {
                        assert_eq!(extra_inputs.len(), 0);
                        ("load", [].into_iter().collect())
                    }
                    QPtrOp::Store => {
                        assert_eq!(extra_inputs.len(), 1);
                        ("store", [extra_inputs[0].print(printer)].into_iter().collect())
                    }
                };

                pretty::Fragment::new([
                    printer
                        .demote_style_for_namespace_prefix(printer.declarative_keyword_style())
                        .apply("qptr.")
                        .into(),
                    printer.declarative_keyword_style().apply(name).into(),
                    pretty::join_comma_sep(
                        "(",
                        qptr_input.map(|v| v.print(printer)).into_iter().chain(extra_inputs),
                        ")",
                    ),
                ])
            }

            DataInstKind::SpvInst(inst) => printer.pretty_spv_inst(
                printer.spv_op_style(),
                inst.opcode,
                &inst.imms,
                inputs.iter().map(|v| v.print(printer)),
            ),
            &DataInstKind::SpvExtInst { ext_set, inst } => {
                let spv_spec = spv::spec::Spec::get();
                let wk = &spv_spec.well_known;

                // HACK(eddyb) hide `OpTypeVoid` types, as they're effectively
                // the default, and not meaningful *even if* the resulting
                // value is "used" in a kind of "untyped token" way.
                output_type_to_print = output_type_to_print.filter(|&ty| {
                    let is_void = match &printer.cx[ty].kind {
                        TypeKind::SpvInst { spv_inst, .. } => spv_inst.opcode == wk.OpTypeVoid,
                        _ => false,
                    };
                    !is_void
                });
                // HACK(eddyb) only keep around untyped outputs if they're used.
                if output_type_to_print.is_none() {
                    output_use_to_print_as_lhs = output_use_to_print_as_lhs.filter(|output_use| {
                        printer
                            .use_styles
                            .get(output_use)
                            .is_some_and(|style| !matches!(style, UseStyle::Inline))
                    });
                }

                // FIXME(eddyb) this may get expensive, cache it?
                let ext_set_name = &printer.cx[ext_set];
                let lowercase_ext_set_name = ext_set_name.to_ascii_lowercase();
                let (ext_set_alias, known_inst_desc) = (spv_spec
                    .get_ext_inst_set_by_lowercase_name(&lowercase_ext_set_name))
                .or_else(|| {
                    printer.cx.get_custom_ext_inst_set_by_lowercase_name(&lowercase_ext_set_name)
                })
                .map_or((&None, None), |ext_inst_set| {
                    // FIXME(eddyb) check that these aliases are unique
                    // across the entire output before using them!
                    (&ext_inst_set.short_alias, ext_inst_set.instructions.get(&inst))
                });

                // FIXME(eddyb) extract and separate out the version?
                let ext_set_name = ext_set_alias.as_deref().unwrap_or(ext_set_name);

                // HACK(eddyb) infinite iterator, only to be used with `zip`.
                let operand_names = known_inst_desc
                    .into_iter()
                    .flat_map(|inst_desc| inst_desc.operand_names.iter().map(|name| &name[..]))
                    .chain(std::iter::repeat(""));

                // HACK(eddyb) we only support two kinds of "pseudo-immediates"
                // (i.e. `Const`s used as immediates by extended instruction sets).
                enum PseudoImm<'a> {
                    Str(&'a str),
                    U32(u32),
                }
                let pseudo_imm_from_value = |v: Value| {
                    if let Value::Const(ct) = v {
                        match &printer.cx[ct].kind {
                            &ConstKind::SpvStringLiteralForExtInst(s) => {
                                return Some(PseudoImm::Str(&printer.cx[s]));
                            }
                            ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                                let (spv_inst, _const_inputs) = &**spv_inst_and_const_inputs;
                                if spv_inst.opcode == wk.OpConstant
                                    && let [spv::Imm::Short(_, x)] = spv_inst.imms[..]
                                {
                                    // HACK(eddyb) only allow unambiguously positive values.
                                    if i32::try_from(x).and_then(u32::try_from) == Ok(x) {
                                        return Some(PseudoImm::U32(x));
                                    }
                                }
                            }
                            ConstKind::PtrToGlobalVar(_) => {}
                        }
                    }
                    None
                };

                let debuginfo_with_pseudo_imm_inputs: Option<SmallVec<[_; 8]>> = known_inst_desc
                    .filter(|inst_desc| {
                        inst_desc.is_debuginfo && output_use_to_print_as_lhs.is_none()
                    })
                    .and_then(|_| inputs.iter().copied().map(pseudo_imm_from_value).collect());
                let printing_debuginfo_as_comment = debuginfo_with_pseudo_imm_inputs.is_some();

                let [spv_base_style, string_literal_style, numeric_literal_style] =
                    if printing_debuginfo_as_comment {
                        [printer.comment_style(); 3]
                    } else {
                        [
                            printer.spv_base_style(),
                            printer.string_literal_style(),
                            printer.numeric_literal_style(),
                        ]
                    };

                let inst_name_or_num = {
                    let (style, s) = match known_inst_desc {
                        Some(inst_desc) => (spv_base_style, inst_desc.name.clone()),
                        None => (numeric_literal_style, format!("{inst}").into()),
                    };
                    // HACK(eddyb) this overlaps a bit with `Printer::spv_op_style`.
                    pretty::Styles { thickness: Some(3), ..style }.apply(s)
                };

                pretty::Fragment::new([
                    if printing_debuginfo_as_comment {
                        printer.comment_style().apply("// ").into()
                    } else {
                        pretty::Fragment::default()
                    },
                    // HACK(eddyb) double/triple-demote to end up with `spv.extinst.A.B`,
                    // with increasing size from `spv.` to `extinst.` to `A.` to `B`.
                    printer
                        .demote_style_for_namespace_prefix(
                            printer.demote_style_for_namespace_prefix(
                                printer.demote_style_for_namespace_prefix(spv_base_style),
                            ),
                        )
                        .apply("spv.")
                        .into(),
                    printer
                        .demote_style_for_namespace_prefix(
                            printer.demote_style_for_namespace_prefix(spv_base_style),
                        )
                        .apply("extinst.")
                        .into(),
                    // HACK(eddyb) print it as a string still, since we don't sanitize it.
                    printer
                        .demote_style_for_namespace_prefix(string_literal_style)
                        .apply(format!("{ext_set_name:?}"))
                        .into(),
                    printer.demote_style_for_namespace_prefix(spv_base_style).apply(".").into(),
                    inst_name_or_num.into(),
                    if let Some(inputs) = debuginfo_with_pseudo_imm_inputs {
                        let style = printer.comment_style();
                        let inputs = inputs.into_iter().zip(operand_names).map(|(input, name)| {
                            pretty::Fragment::new([
                                Some(name)
                                    .filter(|name| !name.is_empty())
                                    .and_then(|name| {
                                        Some(printer.pretty_named_argument_prefix(
                                            printer.sanitize_spv_operand_name(name)?,
                                        ))
                                    })
                                    .unwrap_or_default(),
                                style
                                    .apply(match input {
                                        PseudoImm::Str(s) => format!("{s:?}"),
                                        PseudoImm::U32(x) => format!("{x}"),
                                    })
                                    .into(),
                            ])
                        });
                        pretty::Fragment::new(
                            ([style.apply("(").into()].into_iter())
                                .chain(inputs.intersperse(style.apply(", ").into()))
                                .chain([style.apply(")").into()]),
                        )
                    } else {
                        pretty::join_comma_sep(
                            "(",
                            inputs.iter().zip(operand_names).map(|(&input, name)| {
                                // HACK(eddyb) no need to wrap strings in `OpString(...)`.
                                let printed_input = match pseudo_imm_from_value(input) {
                                    Some(PseudoImm::Str(s)) => printer.pretty_string_literal(s),
                                    _ => input.print(printer),
                                };
                                let name = Some(name)
                                    .filter(|name| !name.is_empty())
                                    .and_then(|name| printer.sanitize_spv_operand_name(name));
                                if let Some(name) = name {
                                    pretty::Fragment::new([
                                        // HACK(eddyb) this duplicates part of
                                        // `Printer::pretty_named_argument_prefix`,
                                        // but the `pretty::join_space` is important.
                                        printer
                                            .named_argument_label_style()
                                            .apply(format!("{name}:"))
                                            .into(),
                                        pretty::join_space("", [printed_input]),
                                    ])
                                } else {
                                    printed_input
                                }
                            }),
                            ")",
                        )
                    },
                ])
            }
        };

        let def_without_name = pretty::Fragment::new([
            def_without_type,
            output_type_to_print
                .map(|ty| printer.pretty_type_ascription_suffix(ty))
                .unwrap_or_default(),
        ]);

        // FIXME(eddyb) this is quite verbose for prepending.
        let def_without_name = pretty::Fragment::new([
            Use::AlignmentAnchorForDataInst(self.position).print_as_def(printer),
            def_without_name,
        ]);

        AttrsAndDef { attrs, def_without_name }.insert_name_before_def(
            output_use_to_print_as_lhs
                .map(|output_use| {
                    pretty::Fragment::new([output_use.print_as_def(printer), " = ".into()])
                })
                .unwrap_or_default(),
        )
    }
}

impl Print for cfg::ControlInst {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        let Self { attrs, kind, inputs, targets, target_inputs } = self;

        let attrs = attrs.print(printer);

        let kw_style = printer.imperative_keyword_style();
        let kw = |kw| kw_style.apply(kw).into();

        let mut targets = targets.iter().map(|&target_region| {
            let mut target = pretty::Fragment::new([
                kw("branch"),
                " ".into(),
                Use::RegionLabel(target_region).print(printer),
            ]);
            if let Some(inputs) = target_inputs.get(&target_region) {
                target = pretty::Fragment::new([
                    target,
                    pretty::join_comma_sep("(", inputs.iter().map(|v| v.print(printer)), ")"),
                ]);
            }
            target
        });

        let def = match kind {
            cfg::ControlInstKind::Unreachable => {
                // FIXME(eddyb) use `targets.is_empty()` when that is stabilized.
                assert!(targets.len() == 0 && inputs.is_empty());
                kw("unreachable")
            }
            cfg::ControlInstKind::Return => {
                // FIXME(eddyb) use `targets.is_empty()` when that is stabilized.
                assert!(targets.len() == 0);
                match inputs[..] {
                    [] => kw("return"),
                    [v] => pretty::Fragment::new([kw("return"), " ".into(), v.print(printer)]),
                    _ => unreachable!(),
                }
            }
            cfg::ControlInstKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(spv::Inst {
                opcode,
                imms,
            })) => {
                // FIXME(eddyb) use `targets.is_empty()` when that is stabilized.
                assert!(targets.len() == 0);
                printer.pretty_spv_inst(
                    kw_style,
                    *opcode,
                    imms,
                    inputs.iter().map(|v| v.print(printer)),
                )
            }

            cfg::ControlInstKind::Branch => {
                assert_eq!((targets.len(), inputs.len()), (1, 0));
                targets.next().unwrap()
            }

            cfg::ControlInstKind::SelectBranch(kind) => {
                assert_eq!(inputs.len(), 1);
                kind.print_with_scrutinee_and_cases(printer, kw_style, inputs[0], targets)
            }
        };

        pretty::Fragment::new([attrs, def])
    }
}

impl SelectionKind {
    fn print_with_scrutinee_and_cases(
        &self,
        printer: &Printer<'_>,
        kw_style: pretty::Styles,
        scrutinee: Value,
        mut cases: impl ExactSizeIterator<Item = pretty::Fragment>,
    ) -> pretty::Fragment {
        let kw = |kw| kw_style.apply(kw).into();
        match *self {
            SelectionKind::BoolCond => {
                assert_eq!(cases.len(), 2);
                let [then_case, else_case] = [cases.next().unwrap(), cases.next().unwrap()];
                pretty::Fragment::new([
                    kw("if"),
                    " ".into(),
                    scrutinee.print(printer),
                    " {".into(),
                    pretty::Node::IndentedBlock(vec![then_case]).into(),
                    "} ".into(),
                    kw("else"),
                    " {".into(),
                    pretty::Node::IndentedBlock(vec![else_case]).into(),
                    "}".into(),
                ])
            }
            SelectionKind::SpvInst(spv::Inst { opcode, ref imms }) => {
                let header = printer.pretty_spv_inst(
                    kw_style,
                    opcode,
                    imms,
                    [Some(scrutinee.print(printer))]
                        .into_iter()
                        .chain((0..cases.len()).map(|_| None)),
                );

                pretty::Fragment::new([
                    header,
                    " {".into(),
                    pretty::Node::IndentedBlock(
                        cases
                            .map(|case| {
                                pretty::Fragment::new([
                                    pretty::Node::ForceLineSeparation.into(),
                                    // FIXME(eddyb) this should pull information out
                                    // of the instruction to be more precise.
                                    kw("case"),
                                    " => {".into(),
                                    pretty::Node::IndentedBlock(vec![case]).into(),
                                    "}".into(),
                                    pretty::Node::ForceLineSeparation.into(),
                                ])
                            })
                            .collect(),
                    )
                    .into(),
                    "}".into(),
                ])
            }
        }
    }
}

impl Value {
    fn print_as_def(&self, printer: &Printer<'_>) -> pretty::Fragment {
        Use::from(*self).print_as_def(printer)
    }
}

impl Print for Value {
    type Output = pretty::Fragment;
    fn print(&self, printer: &Printer<'_>) -> pretty::Fragment {
        Use::from(*self).print(printer)
    }
}
