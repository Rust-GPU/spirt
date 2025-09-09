//! SPIR-T to SPIR-V lifting.

use crate::cf::{self, SelectionKind};
use crate::func_at::FuncAt;
use crate::mem::MemOp;
use crate::spv::{self, spec};
use crate::visit::{InnerVisit, Visitor};
use crate::{
    AddrSpace, Attr, AttrSet, Const, ConstDef, ConstKind, Context, DataInst, DataInstDef,
    DataInstKind, DbgSrcLoc, DeclDef, ExportKey, Exportee, Func, FuncDecl, FuncDefBody, FuncParam,
    FxIndexMap, FxIndexSet, GlobalVar, GlobalVarDefBody, Import, Module, ModuleDebugInfo,
    ModuleDialect, Node, NodeDef, NodeKind, OrdAssertEq, Region, Type, TypeDef, TypeKind,
    TypeOrConst, Value, Var, VarDecl, VarKind, scalar,
};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::{io, iter, mem};

impl spv::Dialect {
    fn capability_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.capabilities.iter().map(move |&cap| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpCapability,
                imms: iter::once(spv::Imm::Short(wk.Capability, cap)).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }

    pub fn extension_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.extensions.iter().map(move |ext| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpExtension,
                imms: spv::encode_literal_string(ext).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }
}

impl spv::ModuleDebugInfo {
    fn source_extension_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.source_extensions.iter().map(move |ext| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpSourceExtension,
                imms: spv::encode_literal_string(ext).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }

    fn module_processed_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.module_processes.iter().map(move |proc| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpModuleProcessed,
                imms: spv::encode_literal_string(proc).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }
}

struct IdAllocator<'a, AI: FnMut() -> spv::Id> {
    cx: &'a Context,
    module: &'a Module,

    /// ID allocation callback, kept as a closure (instead of having its state
    /// be part of `IdAllocator`) to avoid misuse.
    alloc_id: AI,

    ids: ModuleIds<'a>,

    global_vars_seen: FxIndexSet<GlobalVar>,
}

#[derive(Default)]
struct ModuleIds<'a> {
    ext_inst_imports: BTreeMap<&'a str, spv::Id>,
    debug_strings: BTreeMap<&'a str, spv::Id>,

    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    globals: FxIndexMap<Global, spv::Id>,
    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    funcs: FxIndexMap<Func, FuncIds<'a>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Global {
    Type(Type),
    Const(Const),
}

// FIXME(eddyb) should this use ID ranges instead of `SmallVec<[spv::Id; 4]>`?
// FIXME(eddyb) this is inconsistently named with `FuncBodyLifting`.
struct FuncIds<'a> {
    spv_func_ret_type: Type,
    // FIXME(eddyb) should we even be interning an `OpTypeFunction` in `Context`?
    // (it's easier this way, but it could also be tracked in `ModuleIds`)
    spv_func_type: Type,

    func_id: spv::Id,
    param_ids: SmallVec<[spv::Id; 4]>,

    body: Option<FuncBodyLifting<'a>>,
}

impl<AI: FnMut() -> spv::Id> Visitor<'_> for IdAllocator<'_, AI> {
    fn visit_attr_set_use(&mut self, attrs: AttrSet) {
        self.visit_attr_set_def(&self.cx[attrs]);
    }
    fn visit_type_use(&mut self, ty: Type) {
        let global = Global::Type(ty);
        if self.ids.globals.contains_key(&global) {
            return;
        }
        let ty_def = &self.cx[ty];

        // HACK(eddyb) there isn't a great way to handle canonical types, but
        // perhaps this result should be recorded in `self.globals`?
        if let Some((_spv_inst, type_and_const_inputs)) =
            spv::Inst::from_canonical_type(self.cx, &ty_def.kind)
        {
            for ty_or_ct in type_and_const_inputs {
                match ty_or_ct {
                    TypeOrConst::Type(ty) => self.visit_type_use(ty),
                    TypeOrConst::Const(ct) => self.visit_const_use(ct),
                }
            }
        }

        match ty_def.kind {
            TypeKind::Scalar(_) | TypeKind::Vector(_) | TypeKind::SpvInst { .. } => {}

            // FIXME(eddyb) this should be a proper `Result`-based error instead,
            // and/or `spv::lift` should mutate the module for legalization.
            TypeKind::QPtr => {
                unreachable!("`TypeKind::QPtr` should be legalized away before lifting");
            }

            TypeKind::Thunk => {
                // HACK(eddyb) unstructured control-flow uses thunks.
                return;
            }

            TypeKind::SpvStringLiteralForExtInst => {
                unreachable!(
                    "`TypeKind::SpvStringLiteralForExtInst` should not be used \
                     as a type outside of `ConstKind::SpvStringLiteralForExtInst`"
                );
            }
        }

        self.visit_type_def(ty_def);
        self.ids.globals.insert(global, (self.alloc_id)());
    }
    fn visit_const_use(&mut self, ct: Const) {
        let global = Global::Const(ct);
        if self.ids.globals.contains_key(&global) {
            return;
        }
        let ct_def = &self.cx[ct];

        // HACK(eddyb) there isn't a great way to handle canonical consts, but
        // perhaps this result should be recorded in `self.globals`?
        if let Some((_spv_inst, const_inputs)) =
            spv::Inst::from_canonical_const(self.cx, &ct_def.kind)
        {
            for ct in const_inputs {
                self.visit_const_use(ct);
            }
        }

        match ct_def.kind {
            ConstKind::Undef if matches!(self.cx[ct_def.ty].kind, TypeKind::Thunk) => {
                // HACK(eddyb) unstructured control-flow may use `undef` thunks.
            }

            ConstKind::Undef
            | ConstKind::Scalar(_)
            | ConstKind::Vector(_)
            | ConstKind::PtrToGlobalVar { .. }
            | ConstKind::PtrToFunc(_)
            | ConstKind::SpvInst { .. } => {
                self.visit_const_def(ct_def);
                self.ids.globals.insert(global, (self.alloc_id)());
            }

            // HACK(eddyb) because this is an `OpString` and needs to go earlier
            // in the module than any `OpConstant*`, it needs to be special-cased,
            // without visiting its type, or an entry in `self.globals`.
            ConstKind::SpvStringLiteralForExtInst(s) => {
                let ConstDef { attrs, ty, kind: _ } = ct_def;

                assert!(*attrs == AttrSet::default());
                assert!(
                    self.cx[*ty]
                        == TypeDef {
                            attrs: AttrSet::default(),
                            kind: TypeKind::SpvStringLiteralForExtInst,
                        }
                );

                self.ids.debug_strings.entry(&self.cx[s]).or_insert_with(&mut self.alloc_id);
            }
        }
    }

    fn visit_global_var_use(&mut self, gv: GlobalVar) {
        if self.global_vars_seen.insert(gv) {
            self.visit_global_var_decl(&self.module.global_vars[gv]);
        }
    }
    fn visit_func_use(&mut self, func: Func) {
        if self.ids.funcs.contains_key(&func) {
            return;
        }
        let func_decl = &self.module.funcs[func];

        // Synthesize an `OpTypeFunction` type (that SPIR-T itself doesn't carry).
        let wk = &spec::Spec::get().well_known;
        let spv_func_ret_type = func_decl.ret_type;
        let spv_func_type = self.cx.intern(TypeKind::SpvInst {
            spv_inst: wk.OpTypeFunction.into(),
            type_and_const_inputs: iter::once(spv_func_ret_type)
                .chain(func_decl.params.iter().map(|param| param.ty))
                .map(TypeOrConst::Type)
                .collect(),
        });
        self.visit_type_use(spv_func_type);

        // NOTE(eddyb) inserting first produces a different function ordering
        // overall in the final module, but the order doesn't matter, and we
        // need to avoid infinite recursion for recursive functions.
        self.ids.funcs.insert(
            func,
            FuncIds {
                spv_func_ret_type,
                spv_func_type,
                func_id: (self.alloc_id)(),
                param_ids: func_decl.params.iter().map(|_| (self.alloc_id)()).collect(),
                body: None,
            },
        );

        self.visit_func_decl(func_decl);

        // Handle the body last, to minimize recursion hazards (see comment above).
        match &func_decl.def {
            DeclDef::Imported(_) => {}
            DeclDef::Present(func_def_body) => {
                let func_body_lifting = FuncBodyLifting::from_func_def_body(self, func_def_body);
                self.ids.funcs.get_mut(&func).unwrap().body = Some(func_body_lifting);
            }
        }
    }

    fn visit_spv_module_debug_info(&mut self, debug_info: &spv::ModuleDebugInfo) {
        for sources in debug_info.source_languages.values() {
            // The file operand of `OpSource` has to point to an `OpString`.
            for &s in sources.file_contents.keys() {
                self.ids.debug_strings.entry(&self.cx[s]).or_insert_with(&mut self.alloc_id);
            }
        }
    }
    fn visit_attr(&mut self, attr: &Attr) {
        match *attr {
            Attr::Diagnostics(_)
            | Attr::Mem(_)
            | Attr::QPtr(_)
            | Attr::SpvAnnotation { .. }
            | Attr::SpvBitflagsOperand(_) => {}
            Attr::DbgSrcLoc(OrdAssertEq(DbgSrcLoc { file_path, .. })) => {
                self.ids
                    .debug_strings
                    .entry(&self.cx[file_path])
                    .or_insert_with(&mut self.alloc_id);
            }
        }
        attr.inner_visit_with(self);
    }

    fn visit_node_def(&mut self, func_at_node: FuncAt<'_, Node>) {
        match func_at_node.def().kind {
            NodeKind::Select(_)
            | NodeKind::Loop { .. }
            | NodeKind::ExitInvocation(_)
            | DataInstKind::Scalar(_)
            | DataInstKind::Vector(_)
            | DataInstKind::Mem(MemOp::Load { offset: None } | MemOp::Store { offset: None })
            | DataInstKind::FuncCall(_)
            | DataInstKind::ThunkBind(_)
            | DataInstKind::SpvInst(_) => {}

            // FIXME(eddyb) this should be a proper `Result`-based error instead,
            // and/or `spv::lift` should mutate the module for legalization.
            DataInstKind::Mem(_) => {
                unreachable!("`DataInstKind::Mem` should be legalized away before lifting");
            }

            // FIXME(eddyb) this should be a proper `Result`-based error instead,
            // and/or `spv::lift` should mutate the module for legalization.
            DataInstKind::QPtr(_) => {
                unreachable!("`DataInstKind::QPtr` should be legalized away before lifting");
            }

            DataInstKind::SpvExtInst { ext_set, .. } => {
                self.ids
                    .ext_inst_imports
                    .entry(&self.cx[ext_set])
                    .or_insert_with(&mut self.alloc_id);
            }
        }
        func_at_node.inner_visit_with(self);
    }
}

// FIXME(eddyb) this is inconsistently named with `FuncIds`.
struct FuncBodyLifting<'a> {
    // HACK(eddyb) temporary workaround before it's clear how to map everything
    // to use the new `Var` abstraction effectively.
    vars: &'a crate::EntityDefs<Var>,

    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    region_inputs_source: FxHashMap<Region, RegionInputsSource>,
    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    data_inst_output_ids: FxHashMap<DataInst, spv::Id>,

    label_ids: FxHashMap<CfgPoint, spv::Id>,
    blocks: FxIndexMap<CfgPoint, BlockLifting<'a>>,
}

/// What determines the values for [`VarKind::RegionInput`]s, for a specific
/// region (effectively the subset of "region parents" that support inputs).
///
/// Note that this is not used when an unstructured `thunk` carries input values
/// for its target [`Region`] (which would itself have phis for its `inputs`).
enum RegionInputsSource {
    FuncParams,
    LoopHeaderPhis(Node),
}

/// Any of the possible points in structured or unstructured SPIR-T control-flow,
/// that may require a separate SPIR-V basic block.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CfgPoint {
    RegionEntry(Region),
    RegionExit(Region),

    NodeEntry(Node),
    NodeExit(Node),

    // HACK(eddyb) this is only needed to recover φ ("phi") semantics for
    // unstructured CFG edges, while letting SPIR-T use "BB args" semantics
    // (i.e. potentially different values for the same target `Region`).
    // NOTE(eddyb) this is much more relied upon after all `thunk` refactors.
    UnstructuredEdge(cf::unstructured::ControlEdge),
}

struct BlockLifting<'a> {
    phis: SmallVec<[Phi; 2]>,
    insts: SmallVec<[DataInst; 4]>,
    terminator: Terminator<'a>,
}

struct Phi {
    attrs: AttrSet,
    ty: Type,

    result_id: spv::Id,
    cases: FxIndexMap<CfgPoint, Value>,

    // HACK(eddyb) used for `Loop` `initial_inputs`, to indicate that any edge
    // to the `Loop` (other than the backedge, which is already in `cases`)
    // should automatically get an entry into `cases`, with this value.
    default_value: Option<Value>,
}

/// Similar to unstructured `thunk`s, except:
/// * `targets` use [`CfgPoint`]s instead of [`Region`]s, to be able to
///   reach any of the SPIR-V blocks being created during lifting
/// * φ ("phi") values can be provided for targets regardless of "which side" of
///   the structured control-flow they are for ("region input" vs "node output")
/// * optional `merge` (for `OpSelectionMerge`/`OpLoopMerge`)
/// * existing data is borrowed (from the [`FuncDefBody`](crate::FuncDefBody)),
///   wherever possible
struct Terminator<'a> {
    attrs: AttrSet,

    kind: TerminatorKind<'a>,

    // FIXME(eddyb) use `Cow` or something, but ideally the "owned" case always
    // has at most one input, so allocating a whole `Vec` for that seems unwise.
    inputs: SmallVec<[Value; 2]>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    targets: SmallVec<[CfgPoint; 4]>,

    target_phi_values: FxIndexMap<CfgPoint, &'a [Value]>,

    merge: Option<Merge<CfgPoint>>,
}

enum TerminatorKind<'a> {
    Unreachable,
    Return,
    Branch,
    SelectBranch(&'a cf::SelectionKind),

    // HACK(eddyb) this is the only case the unstructured CFG doesn't represent
    // using `thunk`s (as it has been moved to `NodeKind::ExitInvocation`).
    ExitInvocation(&'a cf::ExitInvocationKind),
}

#[derive(Copy, Clone)]
enum Merge<L> {
    Selection(L),

    Loop {
        /// The label just after the whole loop, i.e. the `break` target.
        loop_merge: L,

        /// A label that the back-edge block post-dominates, i.e. some point in
        /// the loop body where looping around is inevitable (modulo `break`ing
        /// out of the loop through a `do`-`while`-style conditional back-edge).
        ///
        /// SPIR-V calls this "the `continue` target", but unlike other aspects
        /// of SPIR-V "structured control-flow", there can be multiple valid
        /// choices (any that fit the post-dominator/"inevitability" definition).
        //
        // FIXME(eddyb) https://github.com/EmbarkStudios/spirt/pull/10 tried to
        // set this to the loop body entry, but that may not be valid if the loop
        // body actually diverges, because then the loop body exit will still be
        // post-dominating the back-edge *but* the loop body itself wouldn't have
        // any relationship between its entry and its *unreachable* exit.
        loop_continue: L,
    },
}

/// Helper type for deep traversal of the CFG (as a graph of [`CfgPoint`]s), which
/// tracks the necessary context for navigating a [`Region`]/[`Node`].
#[derive(Copy, Clone)]
struct CfgCursor<'p, P = CfgPoint> {
    point: P,
    parent: Option<&'p CfgCursor<'p, ControlParent>>,
}

enum ControlParent {
    Region(Region),
    Node(Node),
}

impl<'p> FuncAt<'_, CfgCursor<'p>> {
    /// Return the next [`CfgPoint`] (wrapped in [`CfgCursor`]) in a linear
    /// chain within structured control-flow (i.e. no branching to child regions).
    fn unique_successor(self) -> Option<CfgCursor<'p>> {
        let cursor = self.position;

        // HACK(eddyb) this skips past any tailing `thunk`-producing node, and
        // goes straight to the `RegionExit` instead.
        let filter_out_tail_thunk = |node| {
            let node_def = &self.nodes[node];
            let is_last_node = node_def.next_in_list().is_none();
            let is_tail_thunk = is_last_node
                && match node_def.kind {
                    NodeKind::ThunkBind(_) => true,
                    NodeKind::Select(_) => node_def.child_regions.iter().all(|&case| {
                        self.at(case).at_children().into_iter().exactly_one().is_ok_and(
                            |case_node| matches!(case_node.def().kind, NodeKind::ThunkBind(_)),
                        )
                    }),
                    _ => false,
                };
            (!is_tail_thunk).then_some(node)
        };

        match cursor.point {
            // Entering a `Region` enters its first `Node` child,
            // or exits the region right away (if it has no children).
            CfgPoint::RegionEntry(region) => Some(CfgCursor {
                point: match (self.at(region).def().children.iter().first)
                    .and_then(filter_out_tail_thunk)
                {
                    Some(first_child) => CfgPoint::NodeEntry(first_child),
                    None => CfgPoint::RegionExit(region),
                },
                parent: cursor.parent,
            }),

            // Exiting a `Region` exits its parent `Node`.
            CfgPoint::RegionExit(_) => cursor.parent.map(|parent| match parent.point {
                ControlParent::Region(_) => unreachable!(),
                ControlParent::Node(parent_node) => {
                    CfgCursor { point: CfgPoint::NodeExit(parent_node), parent: parent.parent }
                }
            }),
            CfgPoint::UnstructuredEdge { .. } => {
                assert!(cursor.parent.is_none());
                None
            }

            // Entering a `Node` depends entirely on the `NodeKind`.
            CfgPoint::NodeEntry(node) => match self.at(node).def().kind {
                NodeKind::Select { .. }
                | NodeKind::Loop { .. }
                | NodeKind::ExitInvocation { .. } => None,

                DataInstKind::Scalar(_)
                | DataInstKind::Vector(_)
                | DataInstKind::FuncCall(_)
                | DataInstKind::Mem(_)
                | DataInstKind::QPtr(_)
                | DataInstKind::ThunkBind(_)
                | DataInstKind::SpvInst(_)
                | DataInstKind::SpvExtInst { .. } => {
                    Some(CfgCursor { point: CfgPoint::NodeExit(node), parent: cursor.parent })
                }
            },

            // Exiting a `Node` chains to a sibling/parent.
            CfgPoint::NodeExit(node) => {
                Some(match self.nodes[node].next_in_list().and_then(filter_out_tail_thunk) {
                    // Enter the next sibling in the `Region`, if one exists.
                    Some(next_node) => {
                        CfgCursor { point: CfgPoint::NodeEntry(next_node), parent: cursor.parent }
                    }

                    // Exit the parent `Region`.
                    None => {
                        let parent = cursor.parent.unwrap();
                        match cursor.parent.unwrap().point {
                            ControlParent::Region(parent_region) => CfgCursor {
                                point: CfgPoint::RegionExit(parent_region),
                                parent: parent.parent,
                            },
                            ControlParent::Node(_) => unreachable!(),
                        }
                    }
                })
            }
        }
    }
}

impl FuncAt<'_, Region> {
    /// Traverse every [`CfgPoint`] (deeply) contained in this [`Region`],
    /// in reverse post-order (RPO), with `f` receiving each [`CfgPoint`]
    /// in turn (wrapped in [`CfgCursor`], for further traversal flexibility).
    ///
    /// RPO iteration over a CFG provides certain guarantees, most importantly
    /// that dominators are visited before the entire subgraph they dominate.
    fn rev_post_order_for_each(self, mut f: impl FnMut(CfgCursor<'_>)) {
        self.rev_post_order_for_each_inner(&mut f, None);
    }

    fn rev_post_order_for_each_inner(
        self,
        f: &mut impl FnMut(CfgCursor<'_>),
        parent: Option<&CfgCursor<'_, ControlParent>>,
    ) {
        let region = self.position;
        f(CfgCursor { point: CfgPoint::RegionEntry(region), parent });
        for func_at_node in self.at_children() {
            func_at_node.rev_post_order_for_each_inner(
                f,
                &CfgCursor { point: ControlParent::Region(region), parent },
            );
        }
        f(CfgCursor { point: CfgPoint::RegionExit(region), parent });
    }
}

impl FuncAt<'_, Node> {
    fn rev_post_order_for_each_inner(
        self,
        f: &mut impl FnMut(CfgCursor<'_>),
        parent: &CfgCursor<'_, ControlParent>,
    ) {
        let node = self.position;
        let parent = Some(parent);
        f(CfgCursor { point: CfgPoint::NodeEntry(node), parent });
        for &region in &self.def().child_regions {
            self.at(region).rev_post_order_for_each_inner(
                f,
                Some(&CfgCursor { point: ControlParent::Node(node), parent }),
            );
        }
        f(CfgCursor { point: CfgPoint::NodeExit(node), parent });
    }
}

impl<'a> FuncBodyLifting<'a> {
    fn from_func_def_body(
        id_allocator: &mut IdAllocator<'_, impl FnMut() -> spv::Id>,
        func_def_body: &'a FuncDefBody,
    ) -> Self {
        let cx = id_allocator.cx;

        let mut region_inputs_source = FxHashMap::default();
        region_inputs_source.insert(func_def_body.body, RegionInputsSource::FuncParams);
        let mut data_inst_output_ids = FxHashMap::default();

        // Create a SPIR-V block for every CFG point needing one.
        let mut blocks = FxIndexMap::default();
        let mut visit_cfg_point = |point_cursor: CfgCursor<'_>| {
            let point = point_cursor.point;

            let phis = match point {
                CfgPoint::RegionEntry(region) => {
                    if region_inputs_source.contains_key(&region) {
                        // Region inputs handled by the parent of the region.
                        SmallVec::new()
                    } else {
                        func_def_body
                            .at(region)
                            .def()
                            .inputs
                            .iter()
                            .map(|&input_var| {
                                let &VarDecl { attrs, ty, .. } = func_def_body.at(input_var).decl();
                                Phi {
                                    attrs,
                                    ty,

                                    result_id: (id_allocator.alloc_id)(),
                                    cases: FxIndexMap::default(),
                                    default_value: None,
                                }
                            })
                            .collect()
                    }
                }
                CfgPoint::RegionExit(_) | CfgPoint::UnstructuredEdge { .. } => SmallVec::new(),

                CfgPoint::NodeEntry(node) => {
                    let node_def = func_def_body.at(node).def();
                    match &node_def.kind {
                        // The backedge of a SPIR-V structured loop points to
                        // the "loop header", i.e. the `Entry` of the `Loop`,
                        // so that's where `body` `inputs` phis have to go.
                        NodeKind::Loop { .. } => {
                            let body = node_def.child_regions[0];
                            let loop_body_def = func_def_body.at(body).def();
                            let loop_body_inputs = &loop_body_def.inputs;

                            if !loop_body_inputs.is_empty() {
                                region_inputs_source
                                    .insert(body, RegionInputsSource::LoopHeaderPhis(node));
                            }

                            loop_body_inputs
                                .iter()
                                .zip_eq(&node_def.inputs)
                                .map(|(&input_var, &initial_value)| {
                                    let &VarDecl { attrs, ty, .. } =
                                        func_def_body.at(input_var).decl();
                                    Phi {
                                        attrs,
                                        ty,

                                        result_id: (id_allocator.alloc_id)(),
                                        cases: FxIndexMap::default(),
                                        default_value: Some(initial_value),
                                    }
                                })
                                .collect()
                        }
                        _ => SmallVec::new(),
                    }
                }
                CfgPoint::NodeExit(node) => {
                    let node_def = func_def_body.at(node).def();
                    if !node_def.child_regions.is_empty() {
                        node_def
                            .outputs
                            .iter()
                            .map(|&output_var| {
                                let &VarDecl { attrs, ty, .. } =
                                    func_def_body.at(output_var).decl();
                                Phi {
                                    attrs,
                                    ty,

                                    result_id: (id_allocator.alloc_id)(),
                                    cases: FxIndexMap::default(),
                                    default_value: None,
                                }
                            })
                            .collect()
                    } else {
                        SmallVec::new()
                    }
                }
            };

            let insts = match point {
                CfgPoint::NodeEntry(node) => {
                    let node_def = func_def_body.at(node).def();
                    match node_def.kind {
                        NodeKind::Select(_)
                        | NodeKind::Loop { .. }
                        | NodeKind::ExitInvocation(_) => SmallVec::new(),

                        DataInstKind::Scalar(_)
                        | DataInstKind::Vector(_)
                        | DataInstKind::FuncCall(_)
                        | DataInstKind::Mem(_)
                        | DataInstKind::QPtr(_)
                        | DataInstKind::SpvInst(_)
                        | DataInstKind::SpvExtInst { .. } => {
                            if !node_def.outputs.is_empty() {
                                data_inst_output_ids.insert(node, (id_allocator.alloc_id)());
                            }

                            [node].into_iter().collect()
                        }

                        DataInstKind::ThunkBind(_) => unreachable!(),
                    }
                }
                _ => SmallVec::new(),
            };

            // Get the terminator, or reconstruct it from structured control-flow.
            let terminator = match (point, func_def_body.at(point_cursor).unique_successor()) {
                // Exiting a `Region` w/o a structured parent.
                (CfgPoint::RegionExit(region), None) => {
                    let unstructured_cfg_thunk =
                        func_def_body.unstructured_cfg.as_ref().map(|cfg| {
                            (
                                cfg,
                                func_def_body
                                    .at(region)
                                    .def()
                                    .outputs
                                    .iter()
                                    .copied()
                                    .exactly_one()
                                    .ok()
                                    .unwrap(),
                            )
                        });
                    if let Some((cfg, thunk)) = unstructured_cfg_thunk {
                        // FIXME(eddyb) this partially overlaps with
                        // `ControlFlowGraph::edges_from_thunk_tailed_region`.
                        let thunk_node_def = match thunk {
                            Value::Var(thunk) => match func_def_body.at(thunk).decl().kind() {
                                VarKind::NodeOutput { node, output_idx: 0 } => {
                                    Ok(func_def_body.at(node).def())
                                }
                                _ => unreachable!(),
                            },
                            Value::Const(ct) => Err(ct),
                        };
                        let (kind, inputs) = match thunk_node_def {
                            Ok(NodeDef { kind: NodeKind::ThunkBind(_), .. }) | Err(_) => {
                                (TerminatorKind::Branch, [].into_iter().collect())
                            }
                            Ok(NodeDef { kind: NodeKind::Select(kind), inputs, .. }) => {
                                (
                                    TerminatorKind::SelectBranch(kind),
                                    // FIXME(eddyb) borrow these whenever possible.
                                    inputs.clone(),
                                )
                            }
                            Ok(_) => unreachable!(),
                        };

                        Terminator {
                            attrs: thunk_node_def.map(|def| def.attrs).unwrap_or_default(),
                            kind,
                            inputs,
                            // FIXME(eddyb) try limiting this to repeated target `Region`s
                            // which *also* pass different value inputs.
                            // NOTE(eddyb) this is much more relied upon,
                            // after all `thunk` refactors.
                            targets: cfg
                                .edges_from(func_def_body.at(region))
                                .map(CfgPoint::UnstructuredEdge)
                                .collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: None,
                        }
                    } else {
                        // Structured return out of the function body.
                        assert!(region == func_def_body.body);
                        Terminator {
                            attrs: AttrSet::default(),
                            kind: TerminatorKind::Return,
                            inputs: func_def_body.at_body().def().outputs.clone(),
                            targets: [].into_iter().collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: None,
                        }
                    }
                }
                (CfgPoint::UnstructuredEdge(edge), None) => {
                    let cfg = func_def_body.unstructured_cfg.as_ref().unwrap();
                    let (attrs, target, target_inputs) =
                        cfg.edge_attrs_target_and_inputs(func_def_body.at(edge));

                    let target = match target {
                        Ok(cf::unstructured::ControlTarget::Region(target)) => Ok(target),
                        Ok(cf::unstructured::ControlTarget::Return) => Err(TerminatorKind::Return),
                        Err(ct) => match cx[ct].kind {
                            ConstKind::Undef => Err(TerminatorKind::Unreachable),
                            _ => unreachable!(),
                        },
                    };
                    match target {
                        Ok(target) => Terminator {
                            attrs,
                            kind: TerminatorKind::Branch,
                            inputs: [].into_iter().collect(),
                            targets: [CfgPoint::RegionEntry(target)].into_iter().collect(),
                            target_phi_values: [(CfgPoint::RegionEntry(target), target_inputs)]
                                .into_iter()
                                .collect(),
                            merge: None,
                        },
                        Err(terminator_kind) => {
                            Terminator {
                                attrs,
                                kind: terminator_kind,
                                // FIXME(eddyb) borrow these whenever possible.
                                inputs: target_inputs.iter().copied().collect(),
                                targets: [].into_iter().collect(),
                                target_phi_values: FxIndexMap::default(),
                                merge: None,
                            }
                        }
                    }
                }

                // Entering a `Node` with child `Region`s (or diverging).
                (CfgPoint::NodeEntry(node), None) => {
                    let node_def = func_def_body.at(node).def();
                    match &node_def.kind {
                        NodeKind::Select(kind) => Terminator {
                            attrs: AttrSet::default(),
                            kind: TerminatorKind::SelectBranch(kind),
                            inputs: [node_def.inputs[0]].into_iter().collect(),
                            targets: node_def
                                .child_regions
                                .iter()
                                .map(|&case| CfgPoint::RegionEntry(case))
                                .collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: Some(Merge::Selection(CfgPoint::NodeExit(node))),
                        },

                        NodeKind::Loop { repeat_condition: _ } => {
                            let body = node_def.child_regions[0];
                            Terminator {
                                attrs: AttrSet::default(),
                                kind: TerminatorKind::Branch,
                                inputs: [].into_iter().collect(),
                                targets: [CfgPoint::RegionEntry(body)].into_iter().collect(),
                                target_phi_values: FxIndexMap::default(),
                                merge: Some(Merge::Loop {
                                    loop_merge: CfgPoint::NodeExit(node),
                                    // NOTE(eddyb) see the note on `Merge::Loop`'s
                                    // `loop_continue` field - in particular, for
                                    // SPIR-T loops, we *could* pick any point
                                    // before/after/between `body`'s `children`
                                    // and it should be valid *but* that had to be
                                    // reverted because it's only true in the absence
                                    // of divergence within the loop body itself!
                                    loop_continue: CfgPoint::RegionExit(body),
                                }),
                            }
                        }

                        NodeKind::ExitInvocation(kind) => Terminator {
                            attrs: AttrSet::default(),
                            kind: TerminatorKind::ExitInvocation(kind),
                            inputs: node_def.inputs.clone(),
                            targets: [].into_iter().collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: None,
                        },

                        DataInstKind::Scalar(_)
                        | DataInstKind::Vector(_)
                        | DataInstKind::FuncCall(_)
                        | DataInstKind::Mem(_)
                        | DataInstKind::QPtr(_)
                        | DataInstKind::ThunkBind(_)
                        | DataInstKind::SpvInst(_)
                        | DataInstKind::SpvExtInst { .. } => unreachable!(),
                    }
                }

                // Exiting a `Region` to the parent `Node`.
                (CfgPoint::RegionExit(region), Some(parent_exit_cursor)) => {
                    let region_outputs = Some(&func_def_body.at(region).def().outputs[..])
                        .filter(|outputs| !outputs.is_empty());

                    let parent_exit = parent_exit_cursor.point;
                    let parent_node = match parent_exit {
                        CfgPoint::NodeExit(parent_node) => parent_node,
                        _ => unreachable!(),
                    };

                    match func_def_body.at(parent_node).def().kind {
                        NodeKind::Select { .. } => Terminator {
                            attrs: AttrSet::default(),
                            kind: TerminatorKind::Branch,
                            inputs: [].into_iter().collect(),
                            targets: [parent_exit].into_iter().collect(),
                            target_phi_values: region_outputs
                                .map(|outputs| (parent_exit, outputs))
                                .into_iter()
                                .collect(),
                            merge: None,
                        },

                        NodeKind::Loop { repeat_condition } => {
                            let backedge = CfgPoint::NodeEntry(parent_node);
                            let mut target_phi_values = region_outputs
                                .map(|outputs| (backedge, outputs))
                                .into_iter()
                                .collect();

                            let is_infinite_loop = match repeat_condition {
                                Value::Const(cond) => {
                                    matches!(cx[cond].kind, ConstKind::Scalar(scalar::Const::TRUE))
                                }
                                Value::Var(_) => false,
                            };
                            if is_infinite_loop {
                                Terminator {
                                    attrs: AttrSet::default(),
                                    kind: TerminatorKind::Branch,
                                    inputs: [].into_iter().collect(),
                                    targets: [backedge].into_iter().collect(),
                                    target_phi_values,
                                    merge: None,
                                }
                            } else {
                                // FIXME(eddyb) this will cause redundant `OpPhi`s
                                // (in that SSA dominance rules do allow directly
                                // referencing values defined inside the loop body),
                                // they should be soundly optimized out somehow,
                                // maybe even reuse `cf::cfgssa` infrastructure?
                                if let Some(outputs) = region_outputs {
                                    target_phi_values.insert(parent_exit, outputs);
                                }
                                Terminator {
                                    attrs: AttrSet::default(),
                                    kind: TerminatorKind::SelectBranch(&SelectionKind::BoolCond),
                                    inputs: [repeat_condition].into_iter().collect(),
                                    targets: [backedge, parent_exit].into_iter().collect(),
                                    target_phi_values,
                                    merge: None,
                                }
                            }
                        }

                        NodeKind::ExitInvocation { .. }
                        | DataInstKind::Scalar(_)
                        | DataInstKind::Vector(_)
                        | DataInstKind::FuncCall(_)
                        | DataInstKind::Mem(_)
                        | DataInstKind::QPtr(_)
                        | DataInstKind::ThunkBind(_)
                        | DataInstKind::SpvInst(_)
                        | DataInstKind::SpvExtInst { .. } => unreachable!(),
                    }
                }

                // Siblings in the same `Region` (including the
                // implied edge from a `DataInst`'s `Entry` to its `Exit`).
                //
                // FIXME(eddyb) reduce the cost of generating then removing most
                // "basic blocks" (as each former-`DataInst` gets *two*!),
                // which should be pretty doable in the common case of getting
                // `NodeEntry(a), NodeExit(a), NodeEntry(b), NodeExit(b), ...`
                // from `rev_post_order_try_for_each` and/or introducing an
                // `unique_predecessor` helper (just like `unique_successor`).
                (_, Some(succ_cursor)) => Terminator {
                    attrs: AttrSet::default(),
                    kind: TerminatorKind::Branch,
                    inputs: [].into_iter().collect(),
                    targets: [succ_cursor.point].into_iter().collect(),
                    target_phi_values: FxIndexMap::default(),
                    merge: None,
                },

                // Impossible cases, they always return `(_, Some(_))`.
                (CfgPoint::RegionEntry(_) | CfgPoint::NodeExit(_), None) => {
                    unreachable!()
                }
            };

            blocks.insert(point, BlockLifting { phis, insts, terminator });
        };
        match &func_def_body.unstructured_cfg {
            None => {
                func_def_body.at_body().rev_post_order_for_each(visit_cfg_point);
            }
            Some(cfg) => {
                for region in cfg.rev_post_order(func_def_body) {
                    func_def_body.at(region).rev_post_order_for_each(&mut visit_cfg_point);

                    // FIXME(eddyb) try limiting this to repeated target `Region`s
                    // which *also* pass different value inputs.
                    // NOTE(eddyb) this is much more relied upon,
                    // after all `thunk` refactors.
                    for edge in cfg.edges_from(func_def_body.at(region)) {
                        visit_cfg_point(CfgCursor {
                            point: CfgPoint::UnstructuredEdge(edge),
                            parent: None,
                        });
                    }
                }
            }
        }

        // Count the number of "uses" of each block (each incoming edge, plus
        // `1` for the entry block), to help determine which blocks are part
        // of a linear branch chain (and potentially fusable), later on.
        //
        // FIXME(eddyb) use `EntityOrientedDenseMap` here.
        let mut use_counts = FxHashMap::<CfgPoint, usize>::default();
        use_counts.reserve(blocks.len());
        let all_edges = blocks.first().map(|(&entry_point, _)| entry_point).into_iter().chain(
            blocks.values().flat_map(|block| {
                block
                    .terminator
                    .merge
                    .iter()
                    .flat_map(|merge| {
                        let (a, b) = match merge {
                            Merge::Selection(a) => (a, None),
                            Merge::Loop { loop_merge: a, loop_continue: b } => (a, Some(b)),
                        };
                        [a].into_iter().chain(b)
                    })
                    .chain(&block.terminator.targets)
                    .copied()
            }),
        );
        for target in all_edges {
            *use_counts.entry(target).or_default() += 1;
        }

        // Fuse chains of linear branches, when there is no information being
        // lost by the fusion. This is done in reverse order, so that in e.g.
        // `a -> b -> c`, `b -> c` is fused first, then when the iteration
        // reaches `a`, it sees `a -> bc` and can further fuse that into one
        // `abc` block, without knowing about `b` and `c` themselves
        // (this is possible because RPO will always output `[a, b, c]`, when
        // `b` and `c` only have one predecessor each).
        //
        // FIXME(eddyb) while this could theoretically fuse certain kinds of
        // merge blocks (mostly loop bodies) into their unique precedessor, that
        // would require adjusting the `Merge` that points to them.
        //
        // HACK(eddyb) this takes advantage of `blocks` being an `IndexMap`,
        // to iterate at the same time as mutating other entries.
        for block_idx in (0..blocks.len()).rev() {
            // HACK(eddyb) elide empty cases of an `if`-`else`/`switch`, as
            // SPIR-V allows their targets to just be the whole merge block
            // (the same one that `OpSelectionMerge` describes).
            let block = &blocks[block_idx];
            if let (TerminatorKind::SelectBranch(_), Some(Merge::Selection(merge_point))) =
                (&block.terminator.kind, block.terminator.merge)
            {
                for target_idx in 0..block.terminator.targets.len() {
                    let block = &blocks[block_idx];
                    let target = block.terminator.targets[target_idx];
                    if !block
                        .terminator
                        .target_phi_values
                        .get(&target)
                        .copied()
                        .unwrap_or_default()
                        .is_empty()
                    {
                        continue;
                    }

                    let target_is_trivial_branch = {
                        let BlockLifting {
                            phis,
                            insts,
                            terminator:
                                Terminator { attrs, kind, inputs, targets, target_phi_values, merge },
                        } = &blocks[&target];

                        (phis.is_empty()
                            && insts.is_empty()
                            && *attrs == AttrSet::default()
                            && matches!(kind, TerminatorKind::Branch)
                            && inputs.is_empty()
                            && targets.len() == 1
                            && target_phi_values.is_empty()
                            && merge.is_none())
                        .then(|| targets[0])
                    };
                    if let Some(target_of_target) = target_is_trivial_branch
                        && target_of_target == merge_point
                    {
                        blocks[block_idx].terminator.targets[target_idx] = target_of_target;
                        *use_counts.get_mut(&target).unwrap() -= 1;
                        *use_counts.get_mut(&target_of_target).unwrap() += 1;
                    }
                }
            }

            let block = &blocks[block_idx];
            let is_trivial_branch = {
                let Terminator { attrs, kind, inputs, targets, target_phi_values, merge } =
                    &block.terminator;

                (*attrs == AttrSet::default()
                    && matches!(kind, TerminatorKind::Branch)
                    && inputs.is_empty()
                    && targets.len() == 1
                    && target_phi_values.is_empty()
                    && merge.is_none())
                .then(|| targets[0])
            };

            if let Some(target) = is_trivial_branch {
                let target_use_count = use_counts.get_mut(&target).unwrap();

                if *target_use_count == 1 {
                    let BlockLifting {
                        phis: ref target_phis,
                        insts: ref mut extra_insts,
                        terminator: ref mut new_terminator,
                    } = blocks[&target];

                    // FIXME(eddyb) check for block-level attributes, once/if
                    // they start being tracked.
                    if target_phis.is_empty() {
                        let extra_insts = mem::take(extra_insts);
                        let new_terminator = mem::replace(
                            new_terminator,
                            Terminator {
                                attrs: Default::default(),
                                kind: TerminatorKind::Unreachable,
                                inputs: Default::default(),
                                targets: Default::default(),
                                target_phi_values: Default::default(),
                                merge: None,
                            },
                        );
                        *target_use_count = 0;

                        let combined_block = &mut blocks[block_idx];
                        combined_block.insts.extend(extra_insts);
                        combined_block.terminator = new_terminator;
                    }
                }
            }
        }

        // Remove now-unused blocks.
        blocks.retain(|point, _| use_counts.get(point).is_some_and(|&count| count > 0));

        // Collect `OpPhi`s from other blocks' edges into each block.
        //
        // HACK(eddyb) this takes advantage of `blocks` being an `IndexMap`,
        // to iterate at the same time as mutating other entries.
        for source_block_idx in 0..blocks.len() {
            let (&source_point, source_block) = blocks.get_index(source_block_idx).unwrap();
            let targets = source_block.terminator.targets.clone();

            for target in targets {
                let source_values = {
                    let (_, source_block) = blocks.get_index(source_block_idx).unwrap();
                    source_block.terminator.target_phi_values.get(&target).copied()
                };
                let target_block = blocks.get_mut(&target).unwrap();
                for (i, target_phi) in target_block.phis.iter_mut().enumerate() {
                    use indexmap::map::Entry;

                    let source_value =
                        source_values.map(|values| values[i]).or(target_phi.default_value).unwrap();
                    match target_phi.cases.entry(source_point) {
                        Entry::Vacant(entry) => {
                            entry.insert(source_value);
                        }

                        // NOTE(eddyb) the only reason duplicates are allowed,
                        // is that `targets` may itself contain the same target
                        // multiple times (which would result in the same value).
                        Entry::Occupied(entry) => {
                            assert!(*entry.get() == source_value);
                        }
                    }
                }
            }
        }

        Self {
            vars: &func_def_body.vars,

            region_inputs_source,
            data_inst_output_ids,
            label_ids: blocks.keys().map(|&point| (point, (id_allocator.alloc_id)())).collect(),
            blocks,
        }
    }
}

/// Maybe-decorated "lazy" SPIR-V instruction, allowing separately emitting
/// decorations from attributes, and the instruction itself, without eagerly
/// allocating all the instructions.
#[derive(Copy, Clone)]
enum LazyInst<'a, 'b> {
    Global(Global),
    OpFunction {
        func_decl: &'a FuncDecl,
        func_ids: &'b FuncIds<'a>,
    },
    OpFunctionParameter {
        param_id: spv::Id,
        param: &'a FuncParam,
    },
    OpLabel {
        label_id: spv::Id,
    },
    OpPhi {
        parent_func_ids: &'b FuncIds<'a>,
        phi: &'b Phi,
    },
    DataInst {
        parent_func_ids: &'b FuncIds<'a>,
        result_id: Option<spv::Id>,
        data_inst_def: &'a DataInstDef,
    },
    Merge(Merge<spv::Id>),
    Terminator {
        parent_func_ids: &'b FuncIds<'a>,
        terminator: &'b Terminator<'a>,
    },
    OpFunctionEnd,
}

impl LazyInst<'_, '_> {
    fn result_id_attrs_and_import(
        self,
        module: &Module,
        ids: &ModuleIds<'_>,
    ) -> (Option<spv::Id>, AttrSet, Option<Import>) {
        let cx = module.cx_ref();

        #[allow(clippy::match_same_arms)]
        match self {
            Self::Global(global) => {
                let (attrs, import) = match global {
                    Global::Type(ty) => (cx[ty].attrs, None),
                    Global::Const(ct) => {
                        let ct_def = &cx[ct];
                        match ct_def.kind {
                            ConstKind::PtrToGlobalVar { global_var, offset: _ } => {
                                let gv_decl = &module.global_vars[global_var];
                                let import = match gv_decl.def {
                                    DeclDef::Imported(import) => Some(import),
                                    DeclDef::Present(_) => None,
                                };
                                (gv_decl.attrs, import)
                            }

                            ConstKind::Undef
                            | ConstKind::Scalar(_)
                            | ConstKind::Vector(_)
                            | ConstKind::PtrToFunc(_)
                            | ConstKind::SpvInst { .. } => (ct_def.attrs, None),

                            // Not inserted into `globals` while visiting.
                            ConstKind::SpvStringLiteralForExtInst(_) => unreachable!(),
                        }
                    }
                };
                (Some(ids.globals[&global]), attrs, import)
            }
            Self::OpFunction { func_decl, func_ids } => {
                let import = match func_decl.def {
                    DeclDef::Imported(import) => Some(import),
                    DeclDef::Present(_) => None,
                };
                (Some(func_ids.func_id), func_decl.attrs, import)
            }
            Self::OpFunctionParameter { param_id, param } => (Some(param_id), param.attrs, None),
            Self::OpLabel { label_id } => (Some(label_id), AttrSet::default(), None),
            Self::OpPhi { parent_func_ids: _, phi } => (Some(phi.result_id), phi.attrs, None),
            Self::DataInst { parent_func_ids: _, result_id, data_inst_def } => {
                (result_id, data_inst_def.attrs, None)
            }
            Self::Merge(_) => (None, AttrSet::default(), None),
            Self::Terminator { parent_func_ids: _, terminator } => (None, terminator.attrs, None),
            Self::OpFunctionEnd => (None, AttrSet::default(), None),
        }
    }

    fn to_inst_and_attrs(
        self,
        module: &Module,
        ids: &ModuleIds<'_>,
    ) -> (spv::InstWithIds, AttrSet) {
        let wk = &spec::Spec::get().well_known;
        let cx = module.cx_ref();

        let value_to_id = |parent_func_ids: &FuncIds<'_>, v| match v {
            Value::Const(ct) => match cx[ct].kind {
                ConstKind::SpvStringLiteralForExtInst(s) => ids.debug_strings[&cx[s]],

                _ => ids.globals[&Global::Const(ct)],
            },
            Value::Var(v) => {
                let parent_func_body_lifting = parent_func_ids.body.as_ref().unwrap();
                match parent_func_body_lifting.vars[v].kind() {
                    VarKind::RegionInput { region, input_idx } => {
                        let input_idx = usize::try_from(input_idx).unwrap();
                        match parent_func_body_lifting.region_inputs_source.get(&region) {
                            Some(RegionInputsSource::FuncParams) => {
                                parent_func_ids.param_ids[input_idx]
                            }
                            Some(&RegionInputsSource::LoopHeaderPhis(loop_node)) => {
                                parent_func_body_lifting.blocks[&CfgPoint::NodeEntry(loop_node)]
                                    .phis[input_idx]
                                    .result_id
                            }
                            None => {
                                parent_func_body_lifting.blocks[&CfgPoint::RegionEntry(region)].phis
                                    [input_idx]
                                    .result_id
                            }
                        }
                    }
                    VarKind::NodeOutput { node, output_idx } => {
                        if let Some(&data_inst_output_id) =
                            parent_func_body_lifting.data_inst_output_ids.get(&node)
                        {
                            // HACK(eddyb) multi-output instructions don't exist pre-disaggregate.
                            assert_eq!(output_idx, 0);
                            data_inst_output_id
                        } else {
                            parent_func_body_lifting.blocks[&CfgPoint::NodeExit(node)].phis
                                [usize::try_from(output_idx).unwrap()]
                            .result_id
                        }
                    }
                }
            }
        };

        let (result_id, attrs, _) = self.result_id_attrs_and_import(module, ids);
        let inst = match self {
            Self::Global(global) => match global {
                Global::Type(ty) => {
                    let ty_def = &cx[ty];
                    match spv::Inst::from_canonical_type(cx, &ty_def.kind)
                        .as_ref()
                        .ok_or(&ty_def.kind)
                    {
                        Err(TypeKind::Scalar(_) | TypeKind::Vector(_)) => {
                            unreachable!("should've been handled as canonical")
                        }

                        Ok((spv_inst, type_and_const_inputs))
                        | Err(TypeKind::SpvInst { spv_inst, type_and_const_inputs }) => {
                            spv::InstWithIds {
                                without_ids: spv_inst.clone(),
                                result_type_id: None,
                                result_id,
                                ids: type_and_const_inputs
                                    .iter()
                                    .map(|&ty_or_ct| {
                                        ids.globals[&match ty_or_ct {
                                            TypeOrConst::Type(ty) => Global::Type(ty),
                                            TypeOrConst::Const(ct) => Global::Const(ct),
                                        }]
                                    })
                                    .collect(),
                            }
                        }

                        // Not inserted into `globals` while visiting.
                        Err(
                            TypeKind::QPtr | TypeKind::Thunk | TypeKind::SpvStringLiteralForExtInst,
                        ) => {
                            unreachable!()
                        }
                    }
                }
                Global::Const(ct) => {
                    let ct_def = &cx[ct];
                    match spv::Inst::from_canonical_const(cx, &ct_def.kind).ok_or(&ct_def.kind) {
                        // FIXME(eddyb) this duplicates the `ConstKind::SpvInst`
                        // case, only due to an inability to pattern-match `Rc`.
                        Ok((spv_inst, const_inputs)) => spv::InstWithIds {
                            without_ids: spv_inst,
                            result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                            result_id,
                            ids: const_inputs
                                .iter()
                                .map(|&ct| ids.globals[&Global::Const(ct)])
                                .collect(),
                        },
                        Err(ConstKind::SpvInst { spv_inst_and_const_inputs }) => {
                            let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                            spv::InstWithIds {
                                without_ids: spv_inst.clone(),
                                result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                                result_id,
                                ids: const_inputs
                                    .iter()
                                    .map(|&ct| ids.globals[&Global::Const(ct)])
                                    .collect(),
                            }
                        }

                        Err(ConstKind::Undef | ConstKind::Scalar(_) | ConstKind::Vector(_)) => {
                            unreachable!("should've been handled as canonical")
                        }

                        Err(&ConstKind::PtrToGlobalVar { global_var, offset }) => {
                            assert_eq!(
                                offset, None,
                                "immediate offsets should be legalized away before lifting"
                            );

                            assert!(ct_def.attrs == AttrSet::default());

                            let gv_decl = &module.global_vars[global_var];

                            assert!(ct_def.ty == gv_decl.type_of_ptr_to);

                            let storage_class = match gv_decl.addr_space {
                                AddrSpace::Handles => {
                                    unreachable!(
                                        "`AddrSpace::Handles` should be legalized away before lifting"
                                    );
                                }
                                AddrSpace::SpvStorageClass(sc) => {
                                    spv::Imm::Short(wk.StorageClass, sc)
                                }
                            };
                            let initializer = match gv_decl.def {
                                DeclDef::Imported(_) => None,
                                DeclDef::Present(GlobalVarDefBody { initializer }) => initializer
                                    .map(|initializer| ids.globals[&Global::Const(initializer)]),
                            };
                            spv::InstWithIds {
                                without_ids: spv::Inst {
                                    opcode: wk.OpVariable,
                                    imms: iter::once(storage_class).collect(),
                                },
                                result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                                result_id,
                                ids: initializer.into_iter().collect(),
                            }
                        }

                        Err(&ConstKind::PtrToFunc(func)) => spv::InstWithIds {
                            without_ids: wk.OpConstantFunctionPointerINTEL.into(),
                            result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                            result_id,
                            ids: [ids.funcs[&func].func_id].into_iter().collect(),
                        },

                        // Not inserted into `globals` while visiting.
                        Err(ConstKind::SpvStringLiteralForExtInst(_)) => unreachable!(),
                    }
                }
            },
            Self::OpFunction { func_decl: _, func_ids } => {
                // FIXME(eddyb) make this less of a search and more of a
                // lookup by splitting attrs into key and value parts.
                let func_ctrl = cx[attrs]
                    .attrs
                    .iter()
                    .find_map(|attr| match *attr {
                        Attr::SpvBitflagsOperand(spv::Imm::Short(kind, word))
                            if kind == wk.FunctionControl =>
                        {
                            Some(word)
                        }
                        _ => None,
                    })
                    .unwrap_or(0);

                spv::InstWithIds {
                    without_ids: spv::Inst {
                        opcode: wk.OpFunction,
                        imms: iter::once(spv::Imm::Short(wk.FunctionControl, func_ctrl)).collect(),
                    },
                    result_type_id: Some(ids.globals[&Global::Type(func_ids.spv_func_ret_type)]),
                    result_id,
                    ids: iter::once(ids.globals[&Global::Type(func_ids.spv_func_type)]).collect(),
                }
            }
            Self::OpFunctionParameter { param_id: _, param } => spv::InstWithIds {
                without_ids: wk.OpFunctionParameter.into(),
                result_type_id: Some(ids.globals[&Global::Type(param.ty)]),
                result_id,
                ids: [].into_iter().collect(),
            },
            Self::OpLabel { label_id: _ } => spv::InstWithIds {
                without_ids: wk.OpLabel.into(),
                result_type_id: None,
                result_id,
                ids: [].into_iter().collect(),
            },
            Self::OpPhi { parent_func_ids, phi } => spv::InstWithIds {
                without_ids: wk.OpPhi.into(),
                result_type_id: Some(ids.globals[&Global::Type(phi.ty)]),
                result_id: Some(phi.result_id),
                ids: phi
                    .cases
                    .iter()
                    .flat_map(|(&source_point, &v)| {
                        [
                            value_to_id(parent_func_ids, v),
                            parent_func_ids.body.as_ref().unwrap().label_ids[&source_point],
                        ]
                    })
                    .collect(),
            },
            Self::DataInst { parent_func_ids, result_id: _, data_inst_def } => {
                let kind = &data_inst_def.kind;
                let (inst, extra_initial_id_operand) =
                    match spv::Inst::from_canonical_node_kind(kind).ok_or(kind) {
                        Ok(spv_inst) => (spv_inst, None),

                        Err(
                            NodeKind::Select(_)
                            | NodeKind::Loop { .. }
                            | NodeKind::ExitInvocation(_),
                        ) => unreachable!(),

                        Err(DataInstKind::Scalar(_) | DataInstKind::Vector(_)) => {
                            unreachable!("should've been handled as canonical")
                        }

                        Err(
                            DataInstKind::Mem(_)
                            | DataInstKind::QPtr(_)
                            | DataInstKind::ThunkBind(_),
                        ) => {
                            // Disallowed while visiting.
                            unreachable!()
                        }

                        Err(&DataInstKind::FuncCall(callee)) => {
                            (wk.OpFunctionCall.into(), Some(ids.funcs[&callee].func_id))
                        }
                        Err(DataInstKind::SpvInst(inst)) => (inst.clone(), None),
                        Err(&DataInstKind::SpvExtInst { ext_set, inst }) => (
                            spv::Inst {
                                opcode: wk.OpExtInst,
                                imms: iter::once(spv::Imm::Short(wk.LiteralExtInstInteger, inst))
                                    .collect(),
                            },
                            Some(ids.ext_inst_imports[&cx[ext_set]]),
                        ),
                    };
                spv::InstWithIds {
                    without_ids: inst,
                    // HACK(eddyb) multi-output instructions don't exist pre-disaggregate.
                    result_type_id: (data_inst_def.outputs.iter().at_most_one().ok().unwrap()).map(
                        |&o| {
                            ids.globals
                                [&Global::Type(parent_func_ids.body.as_ref().unwrap().vars[o].ty)]
                        },
                    ),
                    result_id,
                    ids: extra_initial_id_operand
                        .into_iter()
                        .chain(
                            data_inst_def.inputs.iter().map(|&v| value_to_id(parent_func_ids, v)),
                        )
                        .collect(),
                }
            }
            Self::Merge(Merge::Selection(merge_label_id)) => spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpSelectionMerge,
                    imms: [spv::Imm::Short(wk.SelectionControl, 0)].into_iter().collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [merge_label_id].into_iter().collect(),
            },
            Self::Merge(Merge::Loop {
                loop_merge: merge_label_id,
                loop_continue: continue_label_id,
            }) => spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpLoopMerge,
                    imms: [spv::Imm::Short(wk.LoopControl, 0)].into_iter().collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [merge_label_id, continue_label_id].into_iter().collect(),
            },
            Self::Terminator { parent_func_ids, terminator } => {
                let parent_func_body_lifting = parent_func_ids.body.as_ref().unwrap();
                let mut ids: SmallVec<[_; 4]> = terminator
                    .inputs
                    .iter()
                    .map(|&v| value_to_id(parent_func_ids, v))
                    .chain(
                        terminator
                            .targets
                            .iter()
                            .map(|&target| parent_func_body_lifting.label_ids[&target]),
                    )
                    .collect();

                // FIXME(eddyb) move some of this to `spv::canonical`.
                let inst = match terminator.kind {
                    TerminatorKind::Unreachable => wk.OpUnreachable.into(),
                    TerminatorKind::Return => {
                        if terminator.inputs.is_empty() {
                            wk.OpReturn.into()
                        } else {
                            wk.OpReturnValue.into()
                        }
                    }
                    TerminatorKind::ExitInvocation(cf::ExitInvocationKind::SpvInst(inst)) => {
                        inst.clone()
                    }

                    TerminatorKind::Branch => wk.OpBranch.into(),

                    TerminatorKind::SelectBranch(SelectionKind::BoolCond) => {
                        wk.OpBranchConditional.into()
                    }
                    TerminatorKind::SelectBranch(SelectionKind::Switch { case_consts }) => {
                        // HACK(eddyb) move the default case from last back to first.
                        let default_target = ids.pop().unwrap();
                        ids.insert(1, default_target);

                        spv::Inst {
                            opcode: wk.OpSwitch,
                            imms: case_consts
                                .iter()
                                .flat_map(|ct| ct.encode_as_spv_imms())
                                .collect(),
                        }
                    }
                };
                spv::InstWithIds { without_ids: inst, result_type_id: None, result_id: None, ids }
            }
            Self::OpFunctionEnd => spv::InstWithIds {
                without_ids: wk.OpFunctionEnd.into(),
                result_type_id: None,
                result_id: None,
                ids: [].into_iter().collect(),
            },
        };
        (inst, attrs)
    }
}

impl Module {
    pub fn lift_to_spv_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.lift_to_spv_module_emitter()?.write_to_spv_file(path)
    }

    pub fn lift_to_spv_module_emitter(&self) -> io::Result<spv::write::ModuleEmitter> {
        let spv_spec = spec::Spec::get();
        let wk = &spv_spec.well_known;

        let cx = self.cx();
        let (dialect, debug_info) = match (&self.dialect, &self.debug_info) {
            (ModuleDialect::Spv(dialect), ModuleDebugInfo::Spv(debug_info)) => {
                (dialect, debug_info)
            }

            // FIXME(eddyb) support by computing some valid "minimum viable"
            // `spv::Dialect`, or by taking it as additional input.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "not a SPIR-V module"));
            }
        };

        // Because `GlobalVar`s are given IDs by the `Const`s that point to them
        // (i.e. `ConstKind::PtrToGlobalVar`), any `GlobalVar`s in other positions
        // require extra care to ensure the ID-giving `Const` is visited.
        let global_var_to_id_giving_global = |gv| {
            let type_of_ptr_to_global_var = self.global_vars[gv].type_of_ptr_to;
            let ptr_to_global_var = cx.intern(ConstDef {
                attrs: AttrSet::default(),
                ty: type_of_ptr_to_global_var,
                kind: ConstKind::PtrToGlobalVar { global_var: gv, offset: None },
            });
            Global::Const(ptr_to_global_var)
        };

        // Collect uses scattered throughout the module, allocating IDs for them.
        let (ids, id_bound) = {
            let mut id_bound = NonZeroUsize::MIN;
            let mut id_allocator = IdAllocator {
                cx: &cx,
                module: self,
                alloc_id: || {
                    let id = id_bound;
                    id_bound =
                        id_bound.checked_add(1).expect("overflowing `usize` should be impossible");

                    // NOTE(eddyb) `MAX` is just a placeholder - the check for overflows
                    // is done below, after all IDs that may be allocated, have been
                    // (this is in order to not need this closure to return a `Result`).
                    id.try_into().unwrap_or(spv::Id::new(u32::MAX).unwrap())
                },
                ids: ModuleIds::default(),
                global_vars_seen: FxIndexSet::default(),
            };
            id_allocator.visit_module(self);

            // See comment on `global_var_to_id_giving_global` for why this is here.
            for &gv in &id_allocator.global_vars_seen {
                id_allocator
                    .ids
                    .globals
                    .entry(global_var_to_id_giving_global(gv))
                    .or_insert_with(&mut id_allocator.alloc_id);
            }

            let ids = id_allocator.ids;

            let id_bound = spv::Id::try_from(id_bound).ok().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ID bound of SPIR-V module doesn't fit in 32 bits",
                )
            })?;

            (ids, id_bound)
        };

        // HACK(eddyb) allow `move` closures below to reference `cx` or `ids`
        // without causing unwanted moves out of them.
        let (cx, ids) = (&*cx, &ids);

        let global_and_func_insts = ids.globals.keys().copied().map(LazyInst::Global).chain(
            ids.funcs.iter().flat_map(|(&func, func_ids)| {
                let func_decl = &self.funcs[func];
                let body_with_lifting = match (&func_decl.def, &func_ids.body) {
                    (DeclDef::Imported(_), None) => None,
                    (DeclDef::Present(def), Some(func_body_lifting)) => {
                        Some((def, func_body_lifting))
                    }
                    _ => unreachable!(),
                };

                let param_insts =
                    func_ids.param_ids.iter().zip(&func_decl.params).map(|(&param_id, param)| {
                        LazyInst::OpFunctionParameter { param_id, param }
                    });
                let body_insts = body_with_lifting.map(|(func_def_body, func_body_lifting)| {
                    func_body_lifting.blocks.iter().flat_map(move |(point, block)| {
                        let BlockLifting { phis, insts, terminator } = block;

                        iter::once(LazyInst::OpLabel {
                            label_id: func_body_lifting.label_ids[point],
                        })
                        .chain(
                            phis.iter()
                                .map(|phi| LazyInst::OpPhi { parent_func_ids: func_ids, phi }),
                        )
                        .chain(insts.iter().copied().map(move |inst| {
                            let data_inst_def = func_def_body.at(inst).def();
                            LazyInst::DataInst {
                                parent_func_ids: func_ids,
                                // HACK(eddyb) multi-output instructions don't exist pre-disaggregate.
                                result_id: (data_inst_def.outputs.iter().at_most_one().ok())
                                    .unwrap()
                                    .map(|_| func_body_lifting.data_inst_output_ids[&inst]),
                                data_inst_def,
                            }
                        }))
                        .chain(terminator.merge.map(|merge| {
                            LazyInst::Merge(match merge {
                                Merge::Selection(merge) => {
                                    Merge::Selection(func_body_lifting.label_ids[&merge])
                                }
                                Merge::Loop { loop_merge, loop_continue } => Merge::Loop {
                                    loop_merge: func_body_lifting.label_ids[&loop_merge],
                                    loop_continue: func_body_lifting.label_ids[&loop_continue],
                                },
                            })
                        }))
                        .chain([LazyInst::Terminator { parent_func_ids: func_ids, terminator }])
                    })
                });

                iter::once(LazyInst::OpFunction { func_decl, func_ids })
                    .chain(param_insts)
                    .chain(body_insts.into_iter().flatten())
                    .chain([LazyInst::OpFunctionEnd])
            }),
        );

        let reserved_inst_schema = 0;
        let header = [
            spv_spec.magic,
            (u32::from(dialect.version_major) << 16) | (u32::from(dialect.version_minor) << 8),
            debug_info.original_generator_magic.map_or(0, |x| x.get()),
            id_bound.get(),
            reserved_inst_schema,
        ];

        let mut emitter = spv::write::ModuleEmitter::with_header(header);

        for cap_inst in dialect.capability_insts() {
            emitter.push_inst(&cap_inst)?;
        }
        for ext_inst in dialect.extension_insts() {
            emitter.push_inst(&ext_inst)?;
        }
        for (&name, &id) in &ids.ext_inst_imports {
            emitter.push_inst(&spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpExtInstImport,
                    imms: spv::encode_literal_string(name).collect(),
                },
                result_type_id: None,
                result_id: Some(id),
                ids: [].into_iter().collect(),
            })?;
        }
        emitter.push_inst(&spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpMemoryModel,
                imms: [
                    spv::Imm::Short(wk.AddressingModel, dialect.addressing_model),
                    spv::Imm::Short(wk.MemoryModel, dialect.memory_model),
                ]
                .into_iter()
                .collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })?;

        // Collect the various sources of attributes.
        let mut entry_point_insts = vec![];
        let mut execution_mode_insts = vec![];
        let mut debug_name_insts = vec![];
        let mut decoration_insts = vec![];

        for lazy_inst in global_and_func_insts.clone() {
            let (result_id, attrs, import) = lazy_inst.result_id_attrs_and_import(self, ids);

            for attr in cx[attrs].attrs.iter() {
                match attr {
                    Attr::DbgSrcLoc(_)
                    | Attr::Diagnostics(_)
                    | Attr::Mem(_)
                    | Attr::QPtr(_)
                    | Attr::SpvBitflagsOperand(_) => {}
                    Attr::SpvAnnotation(inst @ spv::Inst { opcode, .. }) => {
                        let target_id = result_id.expect(
                            "FIXME: it shouldn't be possible to attach \
                                 attributes to instructions without an output",
                        );

                        let inst = spv::InstWithIds {
                            without_ids: inst.clone(),
                            result_type_id: None,
                            result_id: None,
                            ids: iter::once(target_id).collect(),
                        };

                        if [wk.OpExecutionMode, wk.OpExecutionModeId].contains(opcode) {
                            execution_mode_insts.push(inst);
                        } else if [wk.OpName, wk.OpMemberName].contains(opcode) {
                            debug_name_insts.push(inst);
                        } else {
                            decoration_insts.push(inst);
                        }
                    }
                }

                if let Some(import) = import {
                    let target_id = result_id.unwrap();
                    match import {
                        Import::LinkName(name) => {
                            decoration_insts.push(spv::InstWithIds {
                                without_ids: spv::Inst {
                                    opcode: wk.OpDecorate,
                                    imms: iter::once(spv::Imm::Short(
                                        wk.Decoration,
                                        wk.LinkageAttributes,
                                    ))
                                    .chain(spv::encode_literal_string(&cx[name]))
                                    .chain([spv::Imm::Short(wk.LinkageType, wk.Import)])
                                    .collect(),
                                },
                                result_type_id: None,
                                result_id: None,
                                ids: iter::once(target_id).collect(),
                            });
                        }
                    }
                }
            }
        }

        for (export_key, &exportee) in &self.exports {
            let target_id = match exportee {
                Exportee::GlobalVar(gv) => ids.globals[&global_var_to_id_giving_global(gv)],
                Exportee::Func(func) => ids.funcs[&func].func_id,
            };
            match export_key {
                &ExportKey::LinkName(name) => {
                    decoration_insts.push(spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpDecorate,
                            imms: iter::once(spv::Imm::Short(wk.Decoration, wk.LinkageAttributes))
                                .chain(spv::encode_literal_string(&cx[name]))
                                .chain([spv::Imm::Short(wk.LinkageType, wk.Export)])
                                .collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(target_id).collect(),
                    });
                }
                ExportKey::SpvEntryPoint { imms, interface_global_vars } => {
                    entry_point_insts.push(spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpEntryPoint,
                            imms: imms.iter().copied().collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(target_id)
                            .chain(
                                interface_global_vars
                                    .iter()
                                    .map(|&gv| ids.globals[&global_var_to_id_giving_global(gv)]),
                            )
                            .collect(),
                    });
                }
            }
        }

        // FIXME(eddyb) maybe make a helper for `push_inst` with an iterator?
        for entry_point_inst in entry_point_insts {
            emitter.push_inst(&entry_point_inst)?;
        }
        for execution_mode_inst in execution_mode_insts {
            emitter.push_inst(&execution_mode_inst)?;
        }

        for (&s, &id) in &ids.debug_strings {
            emitter.push_inst(&spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpString,
                    imms: spv::encode_literal_string(s).collect(),
                },
                result_type_id: None,
                result_id: Some(id),
                ids: [].into_iter().collect(),
            })?;
        }
        for (lang, sources) in &debug_info.source_languages {
            let lang_imms = || {
                [
                    spv::Imm::Short(wk.SourceLanguage, lang.lang),
                    spv::Imm::Short(wk.LiteralInteger, lang.version),
                ]
                .into_iter()
            };
            if sources.file_contents.is_empty() {
                emitter.push_inst(&spv::InstWithIds {
                    without_ids: spv::Inst { opcode: wk.OpSource, imms: lang_imms().collect() },
                    result_type_id: None,
                    result_id: None,
                    ids: [].into_iter().collect(),
                })?;
            } else {
                for (&file, contents) in &sources.file_contents {
                    // The maximum word count is `2**16 - 1`, the first word is
                    // taken up by the opcode & word count, and one extra byte is
                    // taken up by the nil byte at the end of the LiteralString.
                    const MAX_OP_SOURCE_CONT_CONTENTS_LEN: usize = (0xffff - 1) * 4 - 1;

                    // `OpSource` has 3 more operands than `OpSourceContinued`,
                    // and each of them take up exactly one word.
                    const MAX_OP_SOURCE_CONTENTS_LEN: usize =
                        MAX_OP_SOURCE_CONT_CONTENTS_LEN - 3 * 4;

                    let (contents_initial, mut contents_rest) =
                        contents.split_at(contents.len().min(MAX_OP_SOURCE_CONTENTS_LEN));

                    emitter.push_inst(&spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpSource,
                            imms: lang_imms()
                                .chain(spv::encode_literal_string(contents_initial))
                                .collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(ids.debug_strings[&cx[file]]).collect(),
                    })?;

                    while !contents_rest.is_empty() {
                        // FIXME(eddyb) test with UTF-8! this `split_at` should
                        // actually take *less* than the full possible size, to
                        // avoid cutting a UTF-8 sequence.
                        let (cont_chunk, rest) = contents_rest
                            .split_at(contents_rest.len().min(MAX_OP_SOURCE_CONT_CONTENTS_LEN));
                        contents_rest = rest;

                        emitter.push_inst(&spv::InstWithIds {
                            without_ids: spv::Inst {
                                opcode: wk.OpSourceContinued,
                                imms: spv::encode_literal_string(cont_chunk).collect(),
                            },
                            result_type_id: None,
                            result_id: None,
                            ids: [].into_iter().collect(),
                        })?;
                    }
                }
            }
        }
        for ext_inst in debug_info.source_extension_insts() {
            emitter.push_inst(&ext_inst)?;
        }
        for debug_name_inst in debug_name_insts {
            emitter.push_inst(&debug_name_inst)?;
        }
        for mod_proc_inst in debug_info.module_processed_insts() {
            emitter.push_inst(&mod_proc_inst)?;
        }

        for decoration_inst in decoration_insts {
            emitter.push_inst(&decoration_inst)?;
        }

        let mut current_debug_line = None;
        let mut current_block_id = None; // HACK(eddyb) for `current_debug_line` resets.
        for lazy_inst in global_and_func_insts {
            let (inst, attrs) = lazy_inst.to_inst_and_attrs(self, ids);

            // Reset line debuginfo when crossing/leaving blocks.
            let new_block_id = if inst.opcode == wk.OpLabel {
                Some(inst.result_id.unwrap())
            } else if inst.opcode == wk.OpFunctionEnd {
                None
            } else {
                current_block_id
            };
            if current_block_id != new_block_id {
                current_debug_line = None;
            }
            current_block_id = new_block_id;

            // Determine whether to emit `OpLine`/`OpNoLine` before `inst`,
            // in order to end up with the expected line debuginfo.
            // FIXME(eddyb) make this less of a search and more of a
            // lookup by splitting attrs into key and value parts.
            let new_debug_line = attrs.dbg_src_loc(cx).map(|dbg_src_loc| {
                (ids.debug_strings[&cx[dbg_src_loc.file_path]], dbg_src_loc.start_line_col)
            });
            if current_debug_line != new_debug_line {
                let (opcode, imms, ids) = match new_debug_line {
                    Some((file_path_id, (line, col))) => (
                        wk.OpLine,
                        [
                            spv::Imm::Short(wk.LiteralInteger, line),
                            spv::Imm::Short(wk.LiteralInteger, col),
                        ]
                        .into_iter()
                        .collect(),
                        iter::once(file_path_id).collect(),
                    ),
                    None => (wk.OpNoLine, [].into_iter().collect(), [].into_iter().collect()),
                };
                emitter.push_inst(&spv::InstWithIds {
                    without_ids: spv::Inst { opcode, imms },
                    result_type_id: None,
                    result_id: None,
                    ids,
                })?;
            }
            current_debug_line = new_debug_line;

            emitter.push_inst(&inst)?;
        }

        Ok(emitter)
    }
}
