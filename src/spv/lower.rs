//! SPIR-V to SPIR-T lowering.

use crate::cf::{self, SelectionKind};
use crate::spv::{self, spec};
// FIXME(eddyb) import more to avoid `crate::` everywhere.
use crate::func_at::{FuncAt, FuncAtMut};
use crate::{
    AddrSpace, Attr, AttrSet, Const, ConstDef, ConstKind, Context, DataInstDef, DataInstKind,
    DbgSrcLoc, DeclDef, Diag, EntityDefs, ExportKey, Exportee, Func, FuncDecl, FuncDefBody,
    FuncParam, FxIndexMap, GlobalVarDecl, GlobalVarDefBody, Import, InternedStr, Module, NodeDef,
    NodeKind, Region, RegionDef, Type, TypeDef, TypeKind, TypeOrConst, Value, Var, VarDecl, print,
    scalar,
};
use itertools::{Either, Itertools as _};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::{io, mem};

/// SPIR-T definition of a SPIR-V ID.
enum IdDef {
    Type(Type),
    Const(Const),

    /// Like `Const`, but for SPIR-V "aggregate" (`OpTypeStruct`/`OpTypeArray`)
    /// constants (e.g. `OpConstantComposite`s of those types, but also more
    /// general constants like `OpUndef`/`OpConstantNull` etc.).
    AggregateConst {
        // FIXME(eddyb) remove `whole_const` by always using the `leaves`.
        whole_const: Const,

        whole_type: Type,

        leaves: SmallVec<[Const; 2]>,
    },

    Func(Func),

    // HACK(eddyb) despite `FuncBody` deferring ID resolution to allow forward
    // references *between* functions, function pointer *constants* need a `Func`
    // long before any `OpFunction`s, so they're pre-defined as dummy imports.
    FuncForwardRef(Func),

    SpvExtInstImport(InternedStr),
    SpvDebugString(InternedStr),
}

impl IdDef {
    fn descr(&self, cx: &Context) -> String {
        match *self {
            // FIXME(eddyb) print these with some kind of "maximum depth",
            // instead of just describing the kind of definition.
            // FIXME(eddyb) replace these with the `Diag` embedding system.
            IdDef::Type(_) => "a type".into(),
            IdDef::Const(_) => "a constant".into(),
            IdDef::AggregateConst { .. } => "an aggregate constant".into(),

            IdDef::Func(_) | IdDef::FuncForwardRef(_) => "a function".into(),

            IdDef::SpvExtInstImport(name) => {
                format!("`OpExtInstImport {:?}`", &cx[name])
            }
            IdDef::SpvDebugString(s) => format!("`OpString {:?}`", &cx[s]),
        }
    }
}

impl Type {
    fn aggregate_component_leaf_range_and_type(
        self,
        cx: &Context,
        idx: u32,
    ) -> Option<(Range<usize>, Type)> {
        let (type_and_const_inputs, aggregate_shape) = match &cx[self].kind {
            TypeKind::SpvInst {
                spv_inst: _,
                type_and_const_inputs,
                value_lowering: spv::ValueLowering::Disaggregate(aggregate_shape),
            } => (type_and_const_inputs, aggregate_shape),
            _ => return None,
        };
        let expect_type = |ty_or_ct| match ty_or_ct {
            TypeOrConst::Type(ty) => ty,
            TypeOrConst::Const(_) => unreachable!(),
        };

        let idx_usize = idx as usize;
        let component_type = match aggregate_shape {
            spv::AggregateShape::Struct { .. } => {
                expect_type(*type_and_const_inputs.get(idx_usize)?)
            }
            &spv::AggregateShape::Array { fixed_len, .. } => {
                if idx >= fixed_len {
                    return None;
                }
                expect_type(type_and_const_inputs[0])
            }
        };
        let component_leaf_count = cx[component_type].disaggregated_leaf_count();

        let component_leaf_range = match aggregate_shape {
            spv::AggregateShape::Struct { per_field_leaf_range_end } => {
                let end = per_field_leaf_range_end[idx_usize] as usize;
                let start = end.checked_sub(component_leaf_count)?;
                start..end
            }
            spv::AggregateShape::Array { .. } => {
                let start = component_leaf_count.checked_mul(idx_usize)?;
                let end = start.checked_add(component_leaf_count)?;
                start..end
            }
        };
        Some((component_leaf_range, component_type))
    }

    // HACK(eddyb) `indices` is a `&mut` because it specifically only consumes
    // the indices it needs, so when this function returns `Some`, all remaining
    // indices will be left over for the caller to process itself.
    fn aggregate_component_path_leaf_range_and_type(
        self,
        cx: &Context,
        indices: &mut impl Iterator<Item = u32>,
    ) -> Option<(Range<usize>, Type)> {
        let (mut leaf_range, mut leaf_type) =
            self.aggregate_component_leaf_range_and_type(cx, indices.next()?)?;

        while let spv::ValueLowering::Disaggregate(_) = cx[leaf_type].spv_value_lowering() {
            let (sub_leaf_range, sub_leaf_type) = match indices.next() {
                Some(i) => leaf_type.aggregate_component_leaf_range_and_type(cx, i)?,
                None => break,
            };

            assert!(sub_leaf_range.end <= leaf_range.len());
            leaf_range.end = leaf_range.start + sub_leaf_range.end;
            leaf_range.start += sub_leaf_range.start;
            leaf_type = sub_leaf_type;
        }

        Some((leaf_range, leaf_type))
    }
}

/// Deferred export, needed because the IDs are initially forward refs.
enum Export {
    Linkage {
        name: InternedStr,
        target_id: spv::Id,
    },
    EntryPoint {
        func_id: spv::Id,
        imms: SmallVec<[spv::Imm; 2]>,
        interface_ids: SmallVec<[spv::Id; 4]>,
    },
}

/// Deferred [`FuncDefBody`], needed because some IDs are initially forward refs.
struct FuncBody {
    func_id: spv::Id,
    func: Func,
    insts: Vec<IntraFuncInst>,
}

struct IntraFuncInst {
    // Instruction aspects that can be pre-lowered:
    attrs: AttrSet,
    result_type: Option<Type>,

    without_ids: spv::Inst,

    // Instruction aspects that cannot be lowered initially (due to forward refs):
    result_id: Option<spv::Id>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    ids: SmallVec<[spv::Id; 4]>,
}

// FIXME(eddyb) stop abusing `io::Error` for error reporting and switch to `Diag`.
fn invalid(reason: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("malformed SPIR-V ({reason})"))
}

fn invalid_factory_for_spv_inst(
    inst: &spv::Inst,
    result_id: Option<spv::Id>,
    ids: &[spv::Id],
) -> impl Fn(&str) -> io::Error + use<> {
    let opcode = inst.opcode;
    let first_id_operand = ids.first().copied();
    move |msg: &str| {
        let result_prefix = result_id.map(|id| format!("%{id} = ")).unwrap_or_default();
        let operand_suffix = first_id_operand.map(|id| format!(" %{id} ...")).unwrap_or_default();
        invalid(&format!("in {result_prefix}{}{operand_suffix}: {msg}", opcode.name()))
    }
}

// FIXME(eddyb) provide more information about any normalization that happened:
// * stats about deduplication that occured through interning
// * sets of unused global vars and functions (and types+consts only they use)
// FIXME(eddyb) use `Diag` instead of `io::Error`, maybe with a return type like
// `Result<Module, IncompletelyLoweredModule>` where `IncompletelyLoweredModule`
// contains a `Module`, maps of all the SPIR-V IDs (to the SPIR-T definitions),
// global `Diag`s (where they can't be attached to specific `AttrSet`s), etc.
impl Module {
    pub fn lower_from_spv_file(cx: Rc<Context>, path: impl AsRef<Path>) -> io::Result<Self> {
        Self::lower_from_spv_module_parser(cx, spv::read::ModuleParser::read_from_spv_file(path)?)
    }

    pub fn lower_from_spv_bytes(cx: Rc<Context>, spv_bytes: Vec<u8>) -> io::Result<Self> {
        Self::lower_from_spv_module_parser(
            cx,
            spv::read::ModuleParser::read_from_spv_bytes(spv_bytes)?,
        )
    }

    pub fn lower_from_spv_module_parser(
        cx: Rc<Context>,
        parser: spv::read::ModuleParser,
    ) -> io::Result<Self> {
        let spv_spec = spec::Spec::get();
        let wk = &spv_spec.well_known;

        // HACK(eddyb) used to quickly check whether an `OpVariable` is global.
        let storage_class_function_imm = spv::Imm::Short(wk.StorageClass, wk.Function);

        // HACK(eddyb) used as the `FuncDecl` for an `IdDef::FuncForwardRef`.
        let dummy_decl_for_func_forward_ref = FuncDecl {
            attrs: {
                let mut attrs = AttrSet::default();
                attrs.push_diag(
                    &cx,
                    Diag::err(["function ID used as forward reference but never defined".into()]),
                );
                attrs
            },
            ret_types: [].into_iter().collect(),
            params: [].into_iter().collect(),
            def: DeclDef::Imported(Import::LinkName(cx.intern(""))),
        };
        // HACK(eddyb) no `PartialEq` on `FuncDecl`.
        let assert_is_dummy_decl_for_func_forward_ref = |decl: &FuncDecl| {
            let [expected, found] = [&dummy_decl_for_func_forward_ref, decl].map(
                |FuncDecl { attrs, ret_types, params, def }| {
                    let DeclDef::Imported(import) = def else {
                        unreachable!();
                    };
                    (attrs, ret_types, params, import)
                },
            );
            assert!(expected == found);
        };

        let mut module = {
            let [magic, version, generator_magic, id_bound, reserved_inst_schema] = parser.header;

            // Ensured above (this is the value after any endianness swapping).
            assert_eq!(magic, spv_spec.magic);

            let [version_reserved_hi, version_major, version_minor, version_reserved_lo] =
                version.to_be_bytes();

            if (version_reserved_lo, version_reserved_hi) != (0, 0) {
                return Err(invalid(&format!(
                    "version 0x{version:08x} is not in expected (0.major.minor.0) form"
                )));
            }

            // FIXME(eddyb) maybe use this somehow? (e.g. check IDs against it)
            let _ = id_bound;

            if reserved_inst_schema != 0 {
                return Err(invalid(&format!(
                    "unknown instruction schema {reserved_inst_schema} - only 0 is supported"
                )));
            }

            Self::new(
                cx.clone(),
                crate::ModuleDialect::Spv(spv::Dialect {
                    version_major,
                    version_minor,

                    capabilities: BTreeSet::new(),
                    extensions: BTreeSet::new(),

                    addressing_model: 0,
                    memory_model: 0,
                }),
                crate::ModuleDebugInfo::Spv(spv::ModuleDebugInfo {
                    original_generator_magic: NonZeroU32::new(generator_magic),

                    source_languages: BTreeMap::new(),
                    source_extensions: vec![],
                    module_processes: vec![],
                }),
            )
        };

        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum Seq {
            Capability,
            Extension,
            ExtInstImport,
            MemoryModel,
            EntryPoint,
            ExecutionMode,
            DebugStringAndSource,
            DebugName,
            DebugModuleProcessed,
            Decoration,

            // NOTE(eddyb) not its own section, but only a "checkpoint", forcing
            // instructions following `OpLine`/`OpNoLine` into later sections.
            DebugLine,

            TypeConstOrGlobalVar,
            Function,
        }
        let mut seq = None;

        let mut has_memory_model = false;
        let mut pending_attrs = FxHashMap::<spv::Id, crate::AttrSetDef>::default();
        let mut pending_imports = FxHashMap::<spv::Id, Import>::default();
        let mut pending_exports = vec![];
        let mut current_dbg_src_loc = None;
        let mut current_block_id = None; // HACK(eddyb) for `current_dbg_src_loc` resets.
        let mut id_defs = FxHashMap::default();
        let mut pending_func_bodies = vec![];
        let mut current_func_body = None;

        let mut spv_insts = parser.peekable();
        while let Some(mut inst) = spv_insts.next().transpose()? {
            let opcode = inst.opcode;

            let invalid = invalid_factory_for_spv_inst(&inst, inst.result_id, &inst.ids);

            // Handle line debuginfo early, as it doesn't have its own section,
            // but rather can go almost anywhere among globals and functions.
            //
            // FIXME(eddyb) also support debuginfo "extended instruction sets"
            // (e.g. `OpenCL.DebugInfo.100`, `NonSemantic.Shader.DebugInfo.100`)
            if [wk.OpLine, wk.OpNoLine].contains(&opcode) {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                current_dbg_src_loc = if opcode == wk.OpLine {
                    match (&inst.imms[..], &inst.ids[..]) {
                        (
                            &[spv::Imm::Short(l_kind, line), spv::Imm::Short(c_kind, col)],
                            &[file_path_id],
                        ) => {
                            assert_eq!([l_kind, c_kind], [wk.LiteralInteger; 2]);
                            let file_path = match id_defs.get(&file_path_id) {
                                Some(&IdDef::SpvDebugString(s)) => s,
                                _ => {
                                    return Err(invalid(&format!(
                                        "%{file_path_id} is not an OpString"
                                    )));
                                }
                            };
                            Some(DbgSrcLoc {
                                file_path,
                                start_line_col: (line, col),
                                end_line_col: (line, col),
                                inlined_callee_name_and_call_site: None,
                            })
                        }
                        _ => unreachable!(),
                    }
                } else {
                    assert!(inst.imms.is_empty() && inst.ids.is_empty());
                    None
                };

                // Advance to `Seq::DebugLine` if we're not there yet, forcing
                // any following instructions to not be in earlier sections.
                seq = seq.max(Some(Seq::DebugLine));
                continue;
            }

            // Reset line debuginfo when crossing/leaving blocks.
            let new_block_id = if opcode == wk.OpLabel {
                Some(inst.result_id.unwrap())
            } else if opcode == wk.OpFunctionEnd {
                None
            } else {
                current_block_id
            };
            if current_block_id != new_block_id {
                current_dbg_src_loc = None;
            }
            current_block_id = new_block_id;

            let mut attrs =
                inst.result_id.and_then(|id| pending_attrs.remove(&id)).unwrap_or_default();

            if let Some(dbg_src_loc) = current_dbg_src_loc {
                attrs.set_dbg_src_loc(dbg_src_loc);
            }

            // Take certain bitflags operands out of the instruction and rewrite
            // them into attributes instead.
            inst.imms.retain(|imm| match *imm {
                spv::Imm::Short(kind, word) if kind == wk.FunctionControl => {
                    if word != 0 {
                        attrs.attrs.insert(Attr::SpvBitflagsOperand(*imm));
                    }
                    false
                }
                _ => true,
            });

            let mut attrs = cx.intern(attrs);

            // FIXME(eddyb) move this kind of lookup into methods on some sort
            // of "lowering context" type.
            let result_type = inst
                .result_type_id
                .map(|type_id| match id_defs.get(&type_id) {
                    Some(&IdDef::Type(ty)) => Ok(ty),
                    Some(id_def) => Err(invalid(&format!(
                        "result type %{} should be a type, not a {}",
                        type_id,
                        id_def.descr(&cx)
                    ))),
                    None => Err(invalid(&format!("result type %{type_id} not defined"))),
                })
                .transpose()?;

            let inst_category = spv_spec.instructions[opcode].category;

            let next_seq = if opcode == wk.OpCapability {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let cap = match (&inst.imms[..], &inst.ids[..]) {
                    (&[spv::Imm::Short(kind, cap)], &[]) => {
                        assert_eq!(kind, wk.Capability);
                        cap
                    }
                    _ => unreachable!(),
                };

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.capabilities.insert(cap);
                    }
                }

                Seq::Capability
            } else if opcode == wk.OpExtension {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let ext = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.extensions.insert(ext);
                    }
                }

                Seq::Extension
            } else if opcode == wk.OpExtInstImport {
                assert!(inst.result_type_id.is_none() && inst.ids.is_empty());
                let id = inst.result_id.unwrap();
                let name = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                id_defs.insert(id, IdDef::SpvExtInstImport(cx.intern(name)));

                Seq::ExtInstImport
            } else if opcode == wk.OpMemoryModel {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (addressing_model, memory_model) = match (&inst.imms[..], &inst.ids[..]) {
                    (&[spv::Imm::Short(am_kind, am), spv::Imm::Short(mm_kind, mm)], &[]) => {
                        assert_eq!([am_kind, mm_kind], [wk.AddressingModel, wk.MemoryModel]);
                        (am, mm)
                    }
                    _ => unreachable!(),
                };

                if has_memory_model {
                    return Err(invalid("duplicate OpMemoryModel"));
                }
                has_memory_model = true;

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.addressing_model = addressing_model;
                        dialect.memory_model = memory_model;
                    }
                }

                Seq::MemoryModel
            } else if opcode == wk.OpString {
                assert!(inst.result_type_id.is_none() && inst.ids.is_empty());
                let id = inst.result_id.unwrap();
                let s = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                id_defs.insert(id, IdDef::SpvDebugString(cx.intern(s)));

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpSource {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (lang, version) = match inst.imms[..] {
                    [spv::Imm::Short(l_kind, lang), spv::Imm::Short(v_kind, version), ..] => {
                        assert_eq!([l_kind, v_kind], [wk.SourceLanguage, wk.LiteralInteger]);
                        (lang, version)
                    }
                    _ => unreachable!(),
                };

                let debug_sources = match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => debug_info
                        .source_languages
                        .entry(spv::DebugSourceLang { lang, version })
                        .or_default(),
                };

                match (&inst.imms[2..], &inst.ids[..]) {
                    (contents, &[file_path_id]) => {
                        let file_path = match id_defs.get(&file_path_id) {
                            Some(&IdDef::SpvDebugString(s)) => s,
                            _ => {
                                return Err(invalid(&format!(
                                    "%{file_path_id} is not an OpString"
                                )));
                            }
                        };
                        let mut contents = if contents.is_empty() {
                            String::new()
                        } else {
                            spv::extract_literal_string(contents)
                                .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?
                        };

                        // Absorb all following `OpSourceContinued` into `contents`.
                        while let Some(Ok(cont_inst)) = spv_insts.peek() {
                            if cont_inst.opcode != wk.OpSourceContinued {
                                break;
                            }
                            let cont_inst = spv_insts.next().unwrap().unwrap();

                            assert!(
                                cont_inst.result_type_id.is_none()
                                    && cont_inst.result_id.is_none()
                                    && cont_inst.ids.is_empty()
                            );
                            let cont_contents = spv::extract_literal_string(&cont_inst.imms)
                                .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;
                            contents += &cont_contents;
                        }

                        debug_sources.file_contents.insert(file_path, contents);
                    }
                    (&[], &[]) => {}
                    _ => unreachable!(),
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpSourceContinued {
                return Err(invalid("must follow OpSource"));
            } else if opcode == wk.OpSourceExtension {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let ext = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => {
                        debug_info.source_extensions.push(ext);
                    }
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpModuleProcessed {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let proc = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => {
                        debug_info.module_processes.push(proc);
                    }
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugModuleProcessed
            } else if opcode == wk.OpEntryPoint {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                pending_exports.push(Export::EntryPoint {
                    func_id: inst.ids[0],
                    imms: inst.without_ids.imms,
                    interface_ids: inst.ids[1..].iter().copied().collect(),
                });

                Seq::EntryPoint
            } else if [
                wk.OpExecutionMode,
                wk.OpExecutionModeId, // FIXME(eddyb) not actually supported
                wk.OpName,
                wk.OpMemberName,
                wk.OpDecorate,
                wk.OpMemberDecorate,
                wk.OpDecorateId, // FIXME(eddyb) not actually supported
                wk.OpDecorateString,
                wk.OpMemberDecorateString,
            ]
            .contains(&opcode)
            {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                let target_id = inst.ids[0];
                if inst.ids.len() > 1 {
                    return Err(invalid("unsupported decoration with ID"));
                }

                match inst.imms[..] {
                    // Special-case `OpDecorate LinkageAttributes ... Import|Export`.
                    [
                        decoration @ spv::Imm::Short(..),
                        ref name @ ..,
                        spv::Imm::Short(lt_kind, linkage_type),
                    ] if opcode == wk.OpDecorate
                        && decoration == spv::Imm::Short(wk.Decoration, wk.LinkageAttributes)
                        && lt_kind == wk.LinkageType
                        && [wk.Import, wk.Export].contains(&linkage_type) =>
                    {
                        let name = spv::extract_literal_string(name)
                            .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;
                        let name = cx.intern(name);

                        if linkage_type == wk.Import {
                            pending_imports.insert(target_id, Import::LinkName(name));
                        } else {
                            pending_exports.push(Export::Linkage { name, target_id });
                        }
                    }

                    _ => {
                        pending_attrs
                            .entry(target_id)
                            .or_default()
                            .attrs
                            .insert(Attr::SpvAnnotation(inst.without_ids));
                    }
                };

                if [wk.OpExecutionMode, wk.OpExecutionModeId].contains(&opcode) {
                    Seq::ExecutionMode
                } else if [wk.OpName, wk.OpMemberName].contains(&opcode) {
                    Seq::DebugName
                } else {
                    Seq::Decoration
                }
            } else if [wk.OpDecorationGroup, wk.OpGroupDecorate, wk.OpGroupMemberDecorate]
                .contains(&opcode)
            {
                return Err(invalid("unsupported decoration groups (officially deprecated)"));
            } else if opcode == wk.OpTypeForwardPointer {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (id, sc) = match (&inst.imms[..], &inst.ids[..]) {
                    (&[sc], &[id]) => (id, sc),
                    _ => unreachable!(),
                };

                // HACK(eddyb) this is not a proper implementation - one would
                // require fixpoint (aka "μ" aka "mu") types - but for now this
                // serves as a first approximation for a "deferred error".
                let ty = cx.intern(TypeDef {
                    attrs: mem::take(&mut attrs),
                    kind: TypeKind::SpvInst {
                        spv_inst: spv::Inst { opcode, imms: [sc].into_iter().collect() },
                        type_and_const_inputs: [].into_iter().collect(),
                        value_lowering: Default::default(),
                    },
                });
                id_defs.insert(id, IdDef::Type(ty));

                Seq::TypeConstOrGlobalVar
            } else if inst_category == spec::InstructionCategory::Type {
                assert!(inst.result_type_id.is_none());
                let id = inst.result_id.unwrap();
                let type_and_const_inputs: SmallVec<_> = inst
                    .ids
                    .iter()
                    .map(|&id| match id_defs.get(&id) {
                        Some(&IdDef::Type(ty)) => Ok(TypeOrConst::Type(ty)),
                        Some(&IdDef::Const(ct)) => Ok(TypeOrConst::Const(ct)),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .map(|result| {
                        result.map_err(|descr| {
                            invalid(&format!("unsupported use of {descr} in a type"))
                        })
                    })
                    .collect::<Result<_, _>>()?;

                let ty = cx.intern(TypeDef {
                    attrs: mem::take(&mut attrs),
                    kind: inst.without_ids.into_canonical_type_with(&cx, type_and_const_inputs),
                });
                id_defs.insert(id, IdDef::Type(ty));

                Seq::TypeConstOrGlobalVar
            } else if opcode == wk.OpConstantFunctionPointerINTEL {
                use std::collections::hash_map::Entry;

                let id = inst.result_id.unwrap();

                let func_id = inst.ids[0];
                let func = match id_defs.entry(func_id) {
                    Entry::Occupied(entry) => match entry.get() {
                        &IdDef::FuncForwardRef(func) => Ok(func),
                        id_def => Err(id_def.descr(&cx)),
                    },
                    Entry::Vacant(entry) => {
                        let func =
                            module.funcs.define(&cx, dummy_decl_for_func_forward_ref.clone());
                        entry.insert(IdDef::FuncForwardRef(func));
                        Ok(func)
                    }
                }
                .map_err(|descr| {
                    invalid(&format!(
                        "unsupported use of {descr} as the `OpConstantFunctionPointerINTEL` operand"
                    ))
                })?;

                let ct = cx.intern(ConstDef {
                    attrs: mem::take(&mut attrs),
                    ty: result_type.unwrap(),
                    kind: ConstKind::PtrToFunc(func),
                });
                id_defs.insert(id, IdDef::Const(ct));

                Seq::TypeConstOrGlobalVar
            } else if inst_category == spec::InstructionCategory::Const
                || inst.always_lower_as_const()
            {
                let id = inst.result_id.unwrap();

                let ty = result_type.unwrap();

                let mut aggregate_leaves = match cx[ty].spv_value_lowering() {
                    spv::ValueLowering::Direct => None,
                    spv::ValueLowering::Disaggregate(_) => {
                        // HACK(eddyb) this expands `OpUndef`/`OpConstantNull`.
                        // FIXME(eddyb) this could potentially create a very
                        // inefficient large array, even when the intent can
                        // be expressed much more compactly in theory.
                        if inst.lower_const_by_distributing_to_aggregate_leaves() {
                            assert_eq!(inst.ids.len(), 0);
                            Some(
                                ty.disaggregated_leaf_types(&cx)
                                    .map(|leaf_type| {
                                        cx.intern(ConstDef {
                                            attrs: Default::default(),
                                            ty: leaf_type,
                                            kind: inst
                                                .as_canonical_const(&cx, leaf_type, &[])
                                                .unwrap_or_else(|| ConstKind::SpvInst {
                                                    spv_inst_and_const_inputs: Rc::new((
                                                        inst.without_ids.clone(),
                                                        [].into_iter().collect(),
                                                    )),
                                                }),
                                        })
                                    })
                                    .collect(),
                            )
                        } else if [wk.OpConstantComposite, wk.OpSpecConstantComposite]
                            .contains(&opcode)
                        {
                            // NOTE(eddyb) actual leaves gathered below, while
                            // collecting `const_inputs`.
                            Some(SmallVec::with_capacity(cx[ty].disaggregated_leaf_count()))
                        } else {
                            attrs.push_diag(
                                &cx,
                                Diag::bug(["unsupported aggregate-producing constant".into()]),
                            );
                            None
                        }
                    }
                };

                let const_inputs: SmallVec<_> = inst
                    .ids
                    .iter()
                    .map(|&id| match id_defs.get(&id) {
                        Some(&IdDef::Const(ct)) => {
                            if let Some(aggregate_leaves) = &mut aggregate_leaves {
                                aggregate_leaves.push(ct);
                            }
                            Ok(ct)
                        }
                        Some(IdDef::AggregateConst { whole_const, whole_type: _, leaves }) => {
                            if let Some(aggregate_leaves) = &mut aggregate_leaves {
                                aggregate_leaves.extend(leaves.iter().copied());
                            }
                            Ok(*whole_const)
                        }
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .map(|result| {
                        result.map_err(|descr| {
                            invalid(&format!("unsupported use of {descr} in a constant"))
                        })
                    })
                    .collect::<Result<_, _>>()?;

                if let (spv::ValueLowering::Disaggregate(_), Some(leaves)) =
                    (cx[ty].spv_value_lowering(), &aggregate_leaves)
                    && cx[ty].disaggregated_leaf_count() != leaves.len()
                {
                    attrs.push_diag(
                        &cx,
                        Diag::err([format!(
                            "aggregate leaf count mismatch (expected {}, found {})",
                            cx[ty].disaggregated_leaf_count(),
                            leaves.len()
                        )
                        .into()]),
                    );
                    aggregate_leaves = None;
                }

                let ct = cx.intern(ConstDef {
                    attrs: mem::take(&mut attrs),
                    ty,
                    kind: inst.as_canonical_const(&cx, ty, &const_inputs).unwrap_or_else(|| {
                        ConstKind::SpvInst {
                            spv_inst_and_const_inputs: Rc::new((inst.without_ids, const_inputs)),
                        }
                    }),
                });
                id_defs.insert(
                    id,
                    match (cx[ty].spv_value_lowering(), aggregate_leaves) {
                        (spv::ValueLowering::Disaggregate(_), Some(leaves)) => {
                            // FIXME(eddyb) this may lose semantic `attrs` when
                            // `leaves` are directly used.
                            IdDef::AggregateConst { whole_const: ct, whole_type: ty, leaves }
                        }
                        _ => IdDef::Const(ct),
                    },
                );

                if inst_category != spec::InstructionCategory::Const {
                    // `OpUndef` can appear either among constants, or in a
                    // function, so at most advance `seq` to globals.
                    seq.max(Some(Seq::TypeConstOrGlobalVar)).unwrap()
                } else {
                    Seq::TypeConstOrGlobalVar
                }
            } else if opcode == wk.OpVariable && current_func_body.is_none() {
                let global_var_id = inst.result_id.unwrap();
                let type_of_ptr_to_global_var = result_type.unwrap();

                if inst.imms[0] == storage_class_function_imm {
                    return Err(invalid("`Function` storage class outside function"));
                }

                let storage_class = match inst.imms[..] {
                    [spv::Imm::Short(kind, storage_class)] => {
                        assert_eq!(kind, wk.StorageClass);
                        storage_class
                    }
                    _ => unreachable!(),
                };
                let initializer = match inst.ids[..] {
                    [initializer] => Some(initializer),
                    [] => None,
                    _ => unreachable!(),
                };

                let initializer = initializer
                    .map(|id| match id_defs.get(&id) {
                        Some(&IdDef::Const(ct)) => Ok(ct),
                        Some(&IdDef::AggregateConst { whole_const, .. }) => {
                            // FIXME(eddyb) disaggregate global initializers.
                            Ok(whole_const)
                        }
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .transpose()
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the initializer of a global variable"
                        ))
                    })?;

                let def = match pending_imports.remove(&global_var_id) {
                    Some(import @ Import::LinkName(name)) => {
                        if initializer.is_some() {
                            return Err(invalid(&format!(
                                "global variable with initializer decorated as `Import` of {:?}",
                                &cx[name]
                            )));
                        }
                        DeclDef::Imported(import)
                    }
                    None => DeclDef::Present(GlobalVarDefBody { initializer }),
                };

                let global_var = module.global_vars.define(
                    &cx,
                    GlobalVarDecl {
                        attrs: mem::take(&mut attrs),
                        type_of_ptr_to: type_of_ptr_to_global_var,
                        shape: None,
                        addr_space: AddrSpace::SpvStorageClass(storage_class),
                        def,
                    },
                );
                let ptr_to_global_var = cx.intern(ConstDef {
                    attrs: AttrSet::default(),
                    ty: type_of_ptr_to_global_var,
                    kind: ConstKind::PtrToGlobalVar { global_var, offset: None },
                });
                id_defs.insert(global_var_id, IdDef::Const(ptr_to_global_var));

                Seq::TypeConstOrGlobalVar
            } else if opcode == wk.OpFunction {
                if current_func_body.is_some() {
                    return Err(invalid("nested OpFunction while still in a function"));
                }

                let func_id = inst.result_id.unwrap();
                let func_ret_type = result_type.unwrap();

                let func_type_id = match (&inst.imms[..], &inst.ids[..]) {
                    // NOTE(eddyb) the `FunctionControl` operand is already gone,
                    // having been converted into an attribute above.
                    (&[], &[func_type_id]) => func_type_id,
                    _ => unreachable!(),
                };

                let (func_type_ret_type, func_type_param_types) =
                    match id_defs.get(&func_type_id) {
                        Some(&IdDef::Type(ty)) => match &cx[ty].kind {
                            TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
                                if spv_inst.opcode == wk.OpTypeFunction =>
                            {
                                let mut types =
                                    type_and_const_inputs.iter().map(|&ty_or_ct| match ty_or_ct {
                                        TypeOrConst::Type(ty) => ty,
                                        TypeOrConst::Const(_) => unreachable!(),
                                    });
                                Some((types.next().unwrap(), types))
                            }
                            _ => None,
                        },
                        _ => None,
                    }
                    .ok_or_else(|| {
                        invalid(&format!("function type %{func_type_id} not an `OpTypeFunction`"))
                    })?;

                if func_ret_type != func_type_ret_type {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(
                        &print::Plan::for_root(
                            &cx,
                            &Diag::err([
                                format!("in %{func_id}, ").into(),
                                "return type differs between `OpFunction` (".into(),
                                func_ret_type.into(),
                                ") and `OpTypeFunction` (".into(),
                                func_type_ret_type.into(),
                                ")".into(),
                            ])
                            .message,
                        )
                        .pretty_print()
                        .to_string(),
                    ));
                }

                let def = match pending_imports.remove(&func_id) {
                    Some(import) => DeclDef::Imported(import),
                    None => {
                        let mut regions = EntityDefs::default();
                        let body = regions.define(&cx, RegionDef::default());
                        DeclDef::Present(FuncDefBody {
                            regions,
                            nodes: Default::default(),
                            vars: Default::default(),
                            body,
                            unstructured_cfg: Some(cf::unstructured::ControlFlowGraph::default()),
                        })
                    }
                };

                // Always flatten aggregates in param and return types.
                let ret_types = match &cx[func_ret_type].kind {
                    // HACK(eddyb) `OpTypeVoid` special-cased here as if it were
                    // an aggregate with `0` leaves.
                    TypeKind::SpvInst { spv_inst: func_ret_type_spv_inst, .. }
                        if func_ret_type_spv_inst.opcode == wk.OpTypeVoid =>
                    {
                        [].into_iter().collect()
                    }

                    _ => func_ret_type.disaggregated_leaf_types(&cx).collect(),
                };
                let mut params = SmallVec::with_capacity(func_type_param_types.len());
                for param_type in func_type_param_types {
                    params.extend(
                        param_type
                            .disaggregated_leaf_types(&cx)
                            .map(|ty| FuncParam { attrs: AttrSet::default(), ty }),
                    );
                }

                let decl = FuncDecl { attrs: mem::take(&mut attrs), ret_types, params, def };

                let func = {
                    use std::collections::hash_map::Entry;

                    match id_defs.entry(func_id) {
                        Entry::Occupied(mut entry) => match entry.get() {
                            &IdDef::FuncForwardRef(func) => {
                                let decl_slot = &mut module.funcs[func];
                                assert_is_dummy_decl_for_func_forward_ref(decl_slot);
                                *decl_slot = decl;

                                entry.insert(IdDef::Func(func));
                                Ok(func)
                            }
                            id_def => Err(id_def.descr(&cx)),
                        },
                        Entry::Vacant(entry) => {
                            let func = module.funcs.define(&cx, decl);
                            entry.insert(IdDef::Func(func));
                            Ok(func)
                        }
                    }
                    .map_err(|descr| {
                        invalid(&format!("invalid redefinition of {descr} as a new function"))
                    })?
                };

                current_func_body = Some(FuncBody { func_id, func, insts: vec![] });

                Seq::Function
            } else if opcode == wk.OpFunctionEnd {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                assert!(inst.imms.is_empty() && inst.ids.is_empty());

                let func_body = current_func_body
                    .take()
                    .ok_or_else(|| invalid("nested OpFunction while still in a function"))?;

                pending_func_bodies.push(func_body);

                Seq::Function
            } else {
                let func_body = current_func_body
                    .as_mut()
                    .ok_or_else(|| invalid("expected only inside a function"))?;
                assert_eq!(seq, Some(Seq::Function));

                func_body.insts.push(IntraFuncInst {
                    attrs: mem::take(&mut attrs),
                    result_type,

                    without_ids: spv::Inst { opcode, imms: inst.without_ids.imms },
                    result_id: inst.result_id,
                    ids: inst.ids,
                });

                Seq::Function
            };
            if let Some(prev_seq) = seq
                && prev_seq > next_seq
            {
                return Err(invalid(&format!(
                    "out of order: {next_seq:?} instructions must precede {prev_seq:?} instructions"
                )));
            }
            seq = Some(next_seq);

            if attrs != Default::default() {
                return Err(invalid("unused decorations / line debuginfo"));
            }
        }

        if !has_memory_model {
            return Err(invalid("missing OpMemoryModel"));
        }

        if !pending_attrs.is_empty() {
            let ids = pending_attrs.keys().collect::<BTreeSet<_>>();
            return Err(invalid(&format!("decorated IDs never defined: {ids:?}")));
        }

        if current_func_body.is_some() {
            return Err(invalid("OpFunction without matching OpFunctionEnd"));
        }

        let entry_points: FxHashSet<_> = pending_exports
            .iter()
            .filter_map(|export| match *export {
                Export::Linkage { .. } => None,
                Export::EntryPoint { func_id, .. } => Some(func_id),
            })
            .collect();

        // Process function bodies, having seen the whole module.
        for func_body in pending_func_bodies {
            let FuncBody { func_id, func, insts: raw_insts } = func_body;

            let func_is_entry_point = entry_points.contains(&func_id);

            let func_decl = &mut module.funcs[func];

            #[derive(PartialEq, Eq, Hash)]
            struct PhiKey {
                source_block_id: spv::Id,
                target_block_id: spv::Id,
                // FIXME(eddyb) remove this, key phis only by the edge, and keep
                // a per-edge list of phi input `spv::Id`s (with validation for
                // missing entries/duplicates).
                target_phi_idx: u32,
            }

            struct BlockDetails {
                label_id: spv::Id,
                phi_count: usize,

                // FIXME(eddyb) how inefficient is `FxIndexMap<spv::Id, Type>`?
                // (vs e.g. a bitset combined with not duplicating `Type`s per-block?)
                cfgssa_inter_block_uses: FxIndexMap<spv::Id, Type>,
            }

            // Gather `OpLabel`s and `OpPhi`s early (so they can be random-accessed).
            let mut phi_to_values = FxIndexMap::<PhiKey, SmallVec<[spv::Id; 1]>>::default();
            // FIXME(eddyb) wouldn't `EntityOrientedDenseMap` make more sense?
            let mut block_details = FxIndexMap::<Region, BlockDetails>::default();
            let mut has_blocks = false;
            let mut cfgssa_def_map = {
                // FIXME(eddyb) in theory, this could be a toggle, but there is
                // very little value in allowing dominance-based SSA use rules.
                const SPIRT_CFGSSA_UNDOMINATE: bool = true;

                SPIRT_CFGSSA_UNDOMINATE.then(|| {
                    let mut def_map = cf::cfgssa::DefMap::new();

                    // HACK(eddyb) allow e.g. `OpFunctionParameter` to
                    // be treated like `OpPhi`s of the entry block.
                    if let DeclDef::Present(func_def_body) = &func_decl.def {
                        def_map.add_block(func_def_body.body);
                    }

                    def_map
                })
            };
            // FIXME(eddyb) remove the now-unnecessary indentation.
            // FIXME(eddyb) rethink this first part as "allocating `Var`s".
            {
                for raw_inst in &raw_insts {
                    let IntraFuncInst {
                        without_ids: spv::Inst { opcode, ref imms },
                        result_type,
                        result_id,
                        ..
                    } = *raw_inst;

                    if opcode == wk.OpFunctionParameter {
                        continue;
                    }

                    let is_entry_block = !has_blocks;
                    has_blocks = true;

                    let func_def_body = match &mut func_decl.def {
                        // Error will be emitted later, below.
                        DeclDef::Imported(_) => continue,
                        DeclDef::Present(def) => def,
                    };

                    if opcode == wk.OpLabel {
                        let block = if is_entry_block {
                            // A `Region` was defined earlier,
                            // to be able to create the `FuncDefBody`.
                            func_def_body.body
                        } else {
                            func_def_body.regions.define(&cx, RegionDef::default())
                        };
                        block_details.insert(
                            block,
                            BlockDetails {
                                label_id: result_id.unwrap(),
                                phi_count: 0,
                                cfgssa_inter_block_uses: Default::default(),
                            },
                        );
                    } else if opcode == wk.OpPhi {
                        let (_, block_details) = match block_details.last_mut() {
                            Some(entry) => entry,
                            // Error will be emitted later, below.
                            None => continue,
                        };

                        let phi_idx = block_details.phi_count;
                        block_details.phi_count = phi_idx.checked_add(1).unwrap();
                        let phi_idx = u32::try_from(phi_idx).unwrap();

                        assert!(imms.is_empty());
                        // FIXME(eddyb) use `array_chunks` when that's stable.
                        for value_and_source_block_id in raw_inst.ids.chunks(2) {
                            let &[value_id, source_block_id]: &[_; 2] =
                                value_and_source_block_id.try_into().unwrap();

                            phi_to_values
                                .entry(PhiKey {
                                    source_block_id,
                                    target_block_id: block_details.label_id,
                                    target_phi_idx: phi_idx,
                                })
                                .or_default()
                                .push(value_id);
                        }
                    }

                    if let Some(def_map) = &mut cfgssa_def_map
                        && let DeclDef::Present(func_def_body) = &func_decl.def
                    {
                        let current_block = match block_details.last() {
                            Some((&current_block, _)) => current_block,
                            // HACK(eddyb) ensure e.g. `OpFunctionParameter`
                            // are treated like `OpPhi`s of the entry block.
                            None => func_def_body.body,
                        };

                        if opcode == wk.OpLabel {
                            // HACK(eddyb) the entry block was already added.
                            if current_block != func_def_body.body {
                                def_map.add_block(current_block);
                            }
                            continue;
                        }

                        if let Some(id) = result_id {
                            // HACK(eddyb) ignore entry block defs, to avoid them
                            // being passed around the CFG (this could be done
                            // for every single "region", if they are computed,
                            // approximately *but accurately*, from the CFG).
                            if current_block != func_def_body.body {
                                def_map.add_def(current_block, id, result_type.unwrap());
                            }
                        }
                    }
                }
            }

            let mut params = SmallVec::<[_; 8]>::new();

            let mut func_def_body = if has_blocks {
                match &mut func_decl.def {
                    DeclDef::Imported(Import::LinkName(name)) => {
                        return Err(invalid(&format!(
                            "non-empty function %{} decorated as `Import` of {:?}",
                            func_id, &cx[*name]
                        )));
                    }
                    DeclDef::Present(def) => Some(def),
                }
            } else {
                match func_decl.def {
                    DeclDef::Imported(Import::LinkName(_)) => {}
                    DeclDef::Present(_) => {
                        // FIXME(remove) embed IDs in errors by moving them to the
                        // `let invalid = |...| ...;` closure that wraps insts.
                        return Err(invalid(&format!(
                            "function %{func_id} lacks any blocks, \
                             but isn't an import either"
                        )));
                    }
                }

                None
            };

            // HACK(eddyb) this is generic to allow `IdDef::AggregateConst`s
            // to be converted to `LocalIdDef::Value`s, inside `lookup_id`.
            enum LocalIdDef<VL = Either<VarRange, SmallVec<[Value; 4]>>> {
                Value { whole_type: Type, leaves: VL },
                BlockLabel(Region),
            }

            #[derive(Copy, Clone)]
            struct VarRange {
                start: Var,
                count: NonZeroU32,
            }

            impl VarRange {
                // FIXME(eddyb) make this return `&[Var]` instead, maybe even
                // have `FuncAt<VarRange>` implement `Deref<Target = [Var]>`.
                fn iter(self, func: FuncAt<'_, ()>) -> impl ExactSizeIterator<Item = Value> {
                    let start_decl = func.at(self.start).decl();
                    let all_vars_in_def_parent = start_decl.def_parent.either(
                        |region| &func.at(region).def().inputs,
                        |node| &func.at(node).def().outputs,
                    );
                    all_vars_in_def_parent[start_decl.def_idx as usize..]
                        [..self.count.get() as usize]
                        .iter()
                        .copied()
                        .map(Value::Var)
                }
            }

            let mut local_id_defs = FxIndexMap::<spv::Id, LocalIdDef>::default();

            // Labels can be forward-referenced, so always have them present.
            local_id_defs.extend(
                block_details
                    .iter()
                    .map(|(&region, details)| (details.label_id, LocalIdDef::BlockLabel(region))),
            );

            // HACK(eddyb) in order to reduce restructurization costs with lots
            // of conditional returns, wherever possible (i.e. in entry-points),
            // `OpReturn` gets turned into an `ExitInvocation`, but to avoid
            // doing that to the "true return" at the "end" of the function,
            // SPIR-V structured merge annotations are used to try to find that
            // one block which should end in an `OpReturn` (notably, all of this
            // is for SPIR-V that already has some structured control-flow, and
            // not the kind of SPIR-V that e.g. Rust-GPU might generate).
            // FIXME(eddyb) use a "structured control-flow recovery" analysis to
            // make this more principled.
            let mut whole_func_merge =
                func_def_body.as_ref().map(|func_def_body| func_def_body.body);

            // HACK(eddyb) an entire separate traversal is required to find
            // all inter-block uses, before any blocks get lowered to SPIR-T.
            let mut cfgssa_use_accumulator = cfgssa_def_map
                .as_ref()
                .filter(|_| func_def_body.is_some())
                .map(cf::cfgssa::UseAccumulator::new);
            if let Some(use_acc) = &mut cfgssa_use_accumulator {
                // HACK(eddyb) ensure e.g. `OpFunctionParameter`
                // are treated like `OpPhi`s of the entry block.
                let mut current_block = func_def_body.as_ref().unwrap().body;
                for raw_inst in &raw_insts {
                    let IntraFuncInst {
                        without_ids: spv::Inst { opcode, ref imms },
                        result_id,
                        ..
                    } = *raw_inst;

                    if opcode == wk.OpLabel {
                        current_block = match local_id_defs[&result_id.unwrap()] {
                            LocalIdDef::BlockLabel(region) => region,
                            LocalIdDef::Value { .. } => unreachable!(),
                        };
                        continue;
                    }

                    if opcode == wk.OpPhi {
                        assert!(imms.is_empty());
                        // FIXME(eddyb) use `array_chunks` when that's stable.
                        for value_and_source_block_id in raw_inst.ids.chunks(2) {
                            let &[value_id, source_block_id]: &[_; 2] =
                                value_and_source_block_id.try_into().unwrap();

                            if let Some(&LocalIdDef::BlockLabel(source_block)) =
                                local_id_defs.get(&source_block_id)
                            {
                                // HACK(eddyb) `value_id` would be explicitly used
                                // in `source_block`, in a "BB args" representation,
                                // but phis move the use to the edge's target.
                                use_acc.add_use(source_block, value_id);
                            }
                        }
                        continue;
                    }

                    // HACK(eddyb) while including merges as edges may seem useful,
                    // they don't participate in dominance (and thus SSA validity),
                    // and if there's any chance `current_block` is *not* the
                    // closest dominator of a merge, that merge could contain
                    // uses that don't belong/are illegal in `current_block`.
                    if [wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode) {
                        if whole_func_merge == Some(current_block)
                            && let Some(&LocalIdDef::BlockLabel(merge_block)) =
                                local_id_defs.get(&raw_inst.ids[0])
                        {
                            whole_func_merge = Some(merge_block);
                        }
                        continue;
                    }

                    for &id in &raw_inst.ids {
                        // HACK(eddyb) treat all mentions of `OpLabel` IDs as
                        // CFG edge targets, which turns out to be accurate,
                        // except for `OpPhi`/`OpSelectionMerge`/`OpLoopMerge`
                        // (which are already special-cased above).
                        if let Some(&LocalIdDef::BlockLabel(target_block)) = local_id_defs.get(&id)
                        {
                            if whole_func_merge == Some(current_block) {
                                // HACK(eddyb) always replacing `whole_func_merge`
                                // detects unstructured control-flow, and avoids
                                // keeping around some intermediary block which
                                // happened to be e.g. in an `OpBranch` chain,
                                // but doesn't end in `OpReturn`/`ExitInvocation`.
                                whole_func_merge = (opcode == wk.OpBranch).then_some(target_block);
                            }
                            use_acc.add_edge(current_block, target_block);
                        } else {
                            // HACK(eddyb) this heavily relies on `add_use(_, id)`
                            // ignoring `id`s which aren't recognized by `def_map`.
                            use_acc.add_use(current_block, id);
                        }
                    }
                }
            }
            if let Some(use_acc) = cfgssa_use_accumulator {
                for (block, inter_block_uses) in use_acc.into_inter_block_uses() {
                    block_details[&block].cfgssa_inter_block_uses = inter_block_uses;
                }
            }

            struct CurrentBlock<'a> {
                region: Region,

                // FIXME(eddyb) figure out a better name and/or organization for this.
                details: &'a BlockDetails,

                // HACK(eddyb) this is probably very inefficient but allows easy
                // access to inter-block-used IDs, in a form directly usable in
                // the current block (i.e. `Region` inputs).
                shadowed_local_id_defs: FxIndexMap<spv::Id, LocalIdDef>,
            }

            let mut current_block = None;
            for (raw_inst_idx, raw_inst) in raw_insts.iter().enumerate() {
                let lookahead_raw_inst =
                    |dist| raw_inst_idx.checked_add(dist).and_then(|i| raw_insts.get(i));

                let IntraFuncInst {
                    attrs,
                    result_type,
                    without_ids: spv::Inst { opcode, ref imms },
                    result_id,
                    ref ids,
                } = *raw_inst;

                let invalid = invalid_factory_for_spv_inst(&raw_inst.without_ids, result_id, ids);

                let is_last_in_block = lookahead_raw_inst(1)
                    .is_none_or(|next_raw_inst| next_raw_inst.without_ids.opcode == wk.OpLabel);

                // HACK(eddyb) this is handled early because it's the only case
                // where a `result_id` isn't a value, and `OpFunctionParameter`
                // wants to be able to use common value result helpers.
                if opcode == wk.OpLabel {
                    if is_last_in_block {
                        return Err(invalid("block lacks terminator instruction"));
                    }

                    // An empty `Region` was defined earlier,
                    // to be able to have an entry in `local_id_defs`.
                    let region = match local_id_defs[&result_id.unwrap()] {
                        LocalIdDef::BlockLabel(region) => region,
                        LocalIdDef::Value { .. } => unreachable!(),
                    };
                    let details = &block_details[&region];
                    assert_eq!(details.label_id, result_id.unwrap());
                    current_block = Some(CurrentBlock {
                        region,
                        details,

                        // HACK(eddyb) reuse `shadowed_local_id_defs` storage.
                        shadowed_local_id_defs: current_block
                            .take()
                            .map(|CurrentBlock { mut shadowed_local_id_defs, .. }| {
                                shadowed_local_id_defs.clear();
                                shadowed_local_id_defs
                            })
                            .unwrap_or_default(),
                    });
                    continue;
                }

                // Helper shared by `OpFunctionParameter` and `OpPhi`.
                let attrs_for_result_leaf = |leaf_type: Type| {
                    if result_type == Some(leaf_type) {
                        attrs
                    } else {
                        // FIXME(eddyb) this may lose semantic `attrs`.
                        AttrSet::default()
                    }
                };

                if opcode == wk.OpFunctionParameter {
                    let result_type = result_type.unwrap();

                    if current_block.is_some() {
                        return Err(invalid(
                            "out of order: `OpFunctionParameter`s should come \
                             before the function's blocks",
                        ));
                    }

                    assert!(imms.is_empty() && ids.is_empty());

                    let param_start = params.len();
                    params.extend(
                        result_type
                            .disaggregated_leaf_types(&cx)
                            .map(|ty| FuncParam { attrs: attrs_for_result_leaf(ty), ty }),
                    );
                    let param_end = params.len();

                    if let Some(func_def_body) = &mut func_def_body {
                        let body_inputs = &mut func_def_body.regions[func_def_body.body].inputs;
                        let start = u32::try_from(body_inputs.len()).unwrap();
                        body_inputs.extend(params[param_start..param_end].iter().zip(start..).map(
                            |(&FuncParam { attrs, ty }, def_idx)| {
                                func_def_body.vars.define(
                                    &cx,
                                    VarDecl {
                                        attrs,
                                        ty,

                                        def_parent: Either::Left(func_def_body.body),
                                        def_idx,
                                    },
                                )
                            },
                        ));
                        let end = u32::try_from(body_inputs.len()).unwrap();

                        local_id_defs.insert(
                            result_id.unwrap(),
                            LocalIdDef::Value {
                                whole_type: result_type,
                                leaves: NonZeroU32::new(end - start).map_or(
                                    Either::Right(SmallVec::new()),
                                    |count| {
                                        Either::Left(VarRange {
                                            start: body_inputs[start as usize],
                                            count,
                                        })
                                    },
                                ),
                            },
                        );
                    }
                    continue;
                }
                let func_def_body = func_def_body.as_deref_mut().unwrap();

                let current_block = current_block.as_mut().ok_or_else(|| {
                    invalid("out of order: not expected before the function's blocks")
                })?;

                // HACK(eddyb) the `Region` inputs for inter-block uses
                // have to be inserted just after all the `OpPhi`s' region inputs,
                // or right away (e.g. on `OpLabel`) when there are no `OpPhi`s,
                // so the easiest place to insert them is before handling the
                // first instruction in the block that's not `OpLabel`/`OpPhi`.
                if opcode != wk.OpPhi
                    && current_block.shadowed_local_id_defs.is_empty()
                    && !current_block.details.cfgssa_inter_block_uses.is_empty()
                {
                    let current_block_region_def = &mut func_def_body.regions[current_block.region];
                    assert!(current_block_region_def.children.is_empty());

                    current_block.shadowed_local_id_defs.extend(
                        current_block.details.cfgssa_inter_block_uses.iter().map(
                            |(&used_id, &ty)| {
                                let inputs = &mut current_block_region_def.inputs;
                                let start = u32::try_from(inputs.len()).unwrap();
                                inputs.extend(ty.disaggregated_leaf_types(&cx).zip(start..).map(
                                    |(ty, def_idx)| {
                                        func_def_body.vars.define(
                                            &cx,
                                            VarDecl {
                                                attrs: AttrSet::default(),
                                                ty,

                                                def_parent: Either::Left(current_block.region),
                                                def_idx,
                                            },
                                        )
                                    },
                                ));
                                let end = u32::try_from(inputs.len()).unwrap();

                                (
                                    used_id,
                                    LocalIdDef::Value {
                                        whole_type: ty,
                                        leaves: NonZeroU32::new(end - start).map_or(
                                            Either::Right(SmallVec::new()),
                                            |count| {
                                                Either::Left(VarRange {
                                                    start: inputs[start as usize],
                                                    count,
                                                })
                                            },
                                        ),
                                    },
                                )
                            },
                        ),
                    );
                }

                // HACK(eddyb) not relying on iterators, to allow `FuncAt` usage.
                #[derive(Copy, Clone)]
                enum Leaves<'a> {
                    VarRange(VarRange),
                    Values(&'a [Value]),

                    Const(Const),
                    Consts(&'a [Const]),
                }

                impl Leaves<'_> {
                    fn iter(self, func: FuncAt<'_, ()>) -> impl ExactSizeIterator<Item = Value> {
                        match self {
                            Leaves::VarRange(leaves) => {
                                Either::Left(Either::Left(leaves.iter(func)))
                            }
                            Leaves::Values(leaves) => {
                                Either::Left(Either::Right(leaves.iter().copied()))
                            }

                            Leaves::Const(ct) => {
                                Either::Right(Either::Left([Value::Const(ct)].into_iter()))
                            }
                            Leaves::Consts(leaves) => Either::Right(Either::Right(
                                leaves.iter().copied().map(Value::Const),
                            )),
                        }
                    }
                }

                // FIXME(eddyb) this returns `LocalIdDef` even for global values.
                let lookup_id = |id| match id_defs.get(&id) {
                    None | Some(IdDef::FuncForwardRef(_)) => {
                        let local_id_def = (current_block.shadowed_local_id_defs.get(&id))
                            .or_else(|| local_id_defs.get(&id))
                            .ok_or_else(|| {
                                // FIXME(eddyb) scan the rest of the function for any
                                // instructions returning this ID, to report an invalid
                                // forward reference (use before def).
                                invalid(&format!("undefined ID %{id}"))
                            })?;
                        // HACK(eddyb) change the type of `leaves` within
                        // `LocalIdDef::Value` to support consts
                        // (see `IdDef::AggregateConst` case just below).
                        Ok(match local_id_def {
                            LocalIdDef::Value { whole_type, leaves } => LocalIdDef::Value {
                                whole_type: *whole_type,
                                leaves: leaves.as_ref().either(
                                    |&leaves| Leaves::VarRange(leaves),
                                    |leaves| Leaves::Values(leaves),
                                ),
                            },
                            &LocalIdDef::BlockLabel(label) => LocalIdDef::BlockLabel(label),
                        })
                    }
                    Some(&IdDef::Const(ct)) => {
                        Ok(LocalIdDef::Value { whole_type: cx[ct].ty, leaves: Leaves::Const(ct) })
                    }
                    Some(IdDef::AggregateConst { whole_const: _, whole_type, leaves }) => {
                        Ok(LocalIdDef::Value {
                            whole_type: *whole_type,
                            leaves: Leaves::Consts(leaves),
                        })
                    }
                    Some(id_def @ IdDef::Type(_)) => Err(invalid(&format!(
                        "unsupported use of {} as an operand for \
                         an instruction in a function",
                        id_def.descr(&cx),
                    ))),
                    Some(id_def @ IdDef::Func(_)) => Err(invalid(&format!(
                        "unsupported use of {} outside `OpFunctionCall`",
                        id_def.descr(&cx),
                    ))),
                    Some(id_def @ IdDef::SpvDebugString(s)) => {
                        if opcode == wk.OpExtInst {
                            // HACK(eddyb) intern `OpString`s as `Const`s on
                            // the fly, as it's a less likely usage than the
                            // `OpLine` one.
                            let ty = cx.intern(TypeKind::SpvStringLiteralForExtInst);
                            let ct = cx.intern(ConstDef {
                                attrs: AttrSet::default(),
                                ty,
                                kind: ConstKind::SpvStringLiteralForExtInst(*s),
                            });
                            Ok(LocalIdDef::Value { whole_type: ty, leaves: Leaves::Const(ct) })
                        } else {
                            Err(invalid(&format!(
                                "unsupported use of {} outside `OpSource`, \
                                 `OpLine`, or `OpExtInst`",
                                id_def.descr(&cx),
                            )))
                        }
                    }
                    Some(id_def @ IdDef::SpvExtInstImport(_)) => Err(invalid(&format!(
                        "unsupported use of {} outside `OpExtInst`",
                        id_def.descr(&cx),
                    ))),
                };

                if is_last_in_block {
                    if opcode.def().category != spec::InstructionCategory::ControlFlow
                        || [wk.OpPhi, wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode)
                    {
                        return Err(invalid(
                            "non-control-flow instruction cannot be used \
                             as the terminator instruction of a block",
                        ));
                    }

                    let mut target_inputs = FxIndexMap::default();
                    let mut record_cfg_edge = |target_block| -> io::Result<()> {
                        use indexmap::map::Entry;

                        let target_block_details = &block_details[&target_block];

                        if target_block_details.phi_count == 0
                            && target_block_details.cfgssa_inter_block_uses.is_empty()
                        {
                            return Ok(());
                        }

                        // Only resolve `OpPhi`s exactly once (per target).
                        let target_inputs_entry = match target_inputs.entry(target_block) {
                            Entry::Occupied(_) => return Ok(()),
                            Entry::Vacant(entry) => entry,
                        };

                        let mut target_inputs = SmallVec::new();
                        for target_phi_idx in 0..target_block_details.phi_count {
                            let phi_key = PhiKey {
                                source_block_id: current_block.details.label_id,
                                target_block_id: target_block_details.label_id,
                                target_phi_idx: target_phi_idx.try_into().unwrap(),
                            };
                            let descr_phi_case = || {
                                format!(
                                    "`OpPhi` (#{} in %{})'s case for source block %{}",
                                    phi_key.target_phi_idx,
                                    phi_key.target_block_id,
                                    phi_key.source_block_id,
                                )
                            };

                            let phi_value_ids =
                                phi_to_values.swap_remove(&phi_key).unwrap_or_default();

                            let phi_value_id = match phi_value_ids[..] {
                                [] => {
                                    return Err(invalid(&format!(
                                        "{} is missing",
                                        descr_phi_case()
                                    )));
                                }
                                [id] => id,
                                [..] => {
                                    return Err(invalid(&format!(
                                        "{} is duplicated",
                                        descr_phi_case()
                                    )));
                                }
                            };

                            match lookup_id(phi_value_id)? {
                                LocalIdDef::Value { leaves, .. } => {
                                    target_inputs.extend(leaves.iter(func_def_body.at(())));
                                }
                                LocalIdDef::BlockLabel(_) => {
                                    return Err(invalid(&format!(
                                        "unsupported use of block label as the value for {}",
                                        descr_phi_case()
                                    )));
                                }
                            }
                        }

                        for &used_id in target_block_details.cfgssa_inter_block_uses.keys() {
                            match lookup_id(used_id)? {
                                LocalIdDef::Value { leaves, .. } => {
                                    target_inputs.extend(leaves.iter(func_def_body.at(())));
                                }
                                LocalIdDef::BlockLabel(_) => unreachable!(),
                            }
                        }

                        target_inputs_entry.insert(target_inputs);

                        Ok(())
                    };

                    // Split the operands into value inputs (e.g. a branch's
                    // condition or an `OpSwitch`'s selector) and target blocks.
                    let mut inputs = SmallVec::new();
                    let mut input_types = SmallVec::<[_; 2]>::new();
                    let mut targets = SmallVec::<[_; 4]>::new();
                    for &id in ids {
                        match lookup_id(id)? {
                            LocalIdDef::Value { whole_type, leaves, .. } => {
                                if !targets.is_empty() {
                                    return Err(invalid(
                                        "out of order: value operand \
                                         after target label ID",
                                    ));
                                }

                                match cx[whole_type].spv_value_lowering() {
                                    spv::ValueLowering::Direct => {}

                                    // Returns are "lossily" disaggregated, just like
                                    // function's signatures and calls to them.
                                    spv::ValueLowering::Disaggregate(_)
                                        if opcode == wk.OpReturnValue => {}

                                    spv::ValueLowering::Disaggregate(_) => {
                                        return Err(invalid(
                                            "unsupported aggregate value operand, \
                                             in non-return terminator instruction",
                                        ));
                                    }
                                }

                                inputs.extend(leaves.iter(func_def_body.at(())));
                                input_types.push(whole_type);
                            }
                            LocalIdDef::BlockLabel(target) => {
                                record_cfg_edge(target)?;
                                targets.push(target);
                            }
                        }
                    }

                    // FIXME(eddyb) collect targets in this form to
                    // begin with (instead of recombining them here).
                    let targets_with_inputs = targets.into_iter().map(|target| {
                        (
                            cf::unstructured::ControlTarget::Region(target),
                            target_inputs.get(&target).cloned().unwrap_or_default(),
                        )
                    });

                    // FIXME(eddyb) cache this.
                    let thunk_ty = cx.intern(TypeKind::Thunk);

                    let build_thunk =
                        |func_at_region: FuncAtMut<'_, Region>, (target, target_inputs)| {
                            let region = func_at_region.position;
                            let func = func_at_region.at(());

                            let thunk_node = func.nodes.define(
                                &cx,
                                NodeDef {
                                    attrs,
                                    kind: NodeKind::ThunkBind(target),
                                    inputs: target_inputs,
                                    child_regions: [].into_iter().collect(),
                                    outputs: [].into_iter().collect(),
                                }
                                .into(),
                            );
                            func.regions[region].children.insert_last(thunk_node, func.nodes);

                            let thunk_var = func.vars.define(
                                &cx,
                                VarDecl {
                                    attrs: AttrSet::default(),
                                    ty: thunk_ty,

                                    def_parent: Either::Right(thunk_node),
                                    def_idx: 0,
                                },
                            );
                            func.nodes[thunk_node].outputs.push(thunk_var);

                            Value::Var(thunk_var)
                        };

                    let (selection_kind, targets_with_inputs) = if opcode == wk.OpBranchConditional
                    {
                        assert_eq!((targets_with_inputs.len(), inputs.len()), (2, 1));
                        (Some(SelectionKind::BoolCond), Either::Left(targets_with_inputs))
                    } else if opcode == wk.OpSwitch {
                        assert_eq!(inputs.len(), 1);

                        // HACK(eddyb) `spv::read` has to "redundantly" validate
                        // that such a type is `OpTypeInt`/`OpTypeFloat`, but
                        // there is still a limitation when it comes to `scalar::Const`.
                        // FIXME(eddyb) don't hardcode the 128-bit limitation,
                        // but query `scalar::Const` somehow instead.
                        let scrutinee_type = input_types[0];
                        let scrutinee_type = scrutinee_type
                            .as_scalar(&cx)
                            .filter(|ty| {
                                matches!(ty, scalar::Type::UInt(_) | scalar::Type::SInt(_))
                                    && ty.bit_width() <= 128
                            })
                            .ok_or_else(|| {
                                invalid(
                                    &print::Plan::for_root(
                                        &cx,
                                        &Diag::err([
                                            "unsupported `OpSwitch` scrutinee type `".into(),
                                            scrutinee_type.into(),
                                            "`".into(),
                                        ])
                                        .message,
                                    )
                                    .pretty_print()
                                    .to_string(),
                                )
                            })?;

                        // FIXME(eddyb) move some of this to `spv::canonical`.
                        let imm_words_per_case =
                            usize::try_from(scrutinee_type.bit_width().div_ceil(32)).unwrap();

                        // NOTE(eddyb) these sanity-checks are redundant with `spv::read`.
                        assert_eq!(imms.len() % imm_words_per_case, 0);
                        assert_eq!(targets_with_inputs.len(), 1 + imms.len() / imm_words_per_case);

                        let case_consts = imms
                            .chunks(imm_words_per_case)
                            .map(|case_imms| {
                                scalar::Const::try_decode_from_spv_imms(scrutinee_type, case_imms)
                                    .ok_or_else(|| {
                                        invalid(&format!(
                                            "invalid {}-bit `OpSwitch` case constant",
                                            scrutinee_type.bit_width()
                                        ))
                                    })
                            })
                            .collect::<Result<_, _>>()?;

                        // HACK(eddyb) move the default case from first to last.
                        let targets_with_inputs = {
                            let mut original_targets = targets_with_inputs;
                            let default_target = original_targets.next().unwrap();
                            original_targets.chain([default_target])
                        };

                        (
                            Some(SelectionKind::Switch { case_consts }),
                            Either::Right(targets_with_inputs),
                        )
                    } else {
                        (None, Either::Left(targets_with_inputs))
                    };

                    // HACK(eddyb) see comment on `whole_func_merge`.
                    let treat_return_as_exit_invocation = opcode == wk.OpReturn
                        && func_is_entry_point
                        && whole_func_merge.is_some_and(|whole_func_merge| {
                            whole_func_merge != current_block.region
                        });

                    let target_thunk = if let Some(selection_kind) = selection_kind {
                        let cases = targets_with_inputs
                            .map(|target_with_inputs| {
                                let case = func_def_body.regions.define(&cx, RegionDef::default());
                                let thunk =
                                    build_thunk(func_def_body.at_mut(case), target_with_inputs);
                                func_def_body.regions[case].outputs.push(thunk);
                                case
                            })
                            .collect();

                        // FIXME(eddyb) move some of this to `spv::canonical`.
                        let select_node = func_def_body.nodes.define(
                            &cx,
                            NodeDef {
                                attrs,
                                kind: NodeKind::Select(selection_kind),
                                inputs: mem::take(&mut inputs),
                                child_regions: cases,
                                outputs: [].into_iter().collect(),
                            }
                            .into(),
                        );
                        func_def_body.regions[current_block.region]
                            .children
                            .insert_last(select_node, &mut func_def_body.nodes);

                        let select_thunk_var = func_def_body.vars.define(
                            &cx,
                            VarDecl {
                                attrs: AttrSet::default(),
                                ty: thunk_ty,

                                def_parent: Either::Right(select_node),
                                def_idx: 0,
                            },
                        );
                        func_def_body.nodes[select_node].outputs.push(select_thunk_var);

                        Value::Var(select_thunk_var)
                    } else if [wk.OpReturn, wk.OpReturnValue].contains(&opcode)
                        && !treat_return_as_exit_invocation
                    {
                        assert!(targets_with_inputs.count() == 0);
                        build_thunk(
                            func_def_body.at_mut(current_block.region),
                            (cf::unstructured::ControlTarget::Return, mem::take(&mut inputs)),
                        )
                    } else if opcode == wk.OpBranch {
                        build_thunk(
                            func_def_body.at_mut(current_block.region),
                            targets_with_inputs.exactly_one().ok().unwrap(),
                        )
                    } else {
                        if targets_with_inputs.count() > 0 {
                            return Err(invalid("unsupported control-flow instruction"));
                        }

                        if opcode != wk.OpUnreachable {
                            let node = func_def_body.nodes.define(
                                &cx,
                                NodeDef {
                                    attrs,
                                    kind: NodeKind::ExitInvocation(
                                        cf::ExitInvocationKind::SpvInst(
                                            raw_inst.without_ids.clone(),
                                        ),
                                    ),
                                    inputs: mem::take(&mut inputs),
                                    child_regions: [].into_iter().collect(),
                                    outputs: [].into_iter().collect(),
                                }
                                .into(),
                            );
                            func_def_body.regions[current_block.region]
                                .children
                                .insert_last(node, &mut func_def_body.nodes);
                        }

                        // FIXME(eddyb) cache this.
                        Value::Const(cx.intern(ConstDef {
                            attrs: AttrSet::default(),
                            ty: thunk_ty,
                            kind: ConstKind::Undef,
                        }))
                    };

                    assert_eq!(inputs.len(), 0);

                    func_def_body.regions[current_block.region].outputs =
                        [target_thunk].into_iter().collect();
                    continue;
                }

                if opcode == wk.OpPhi {
                    let current_block_region_def = &mut func_def_body.regions[current_block.region];

                    let result_type = result_type.unwrap();

                    if !current_block_region_def.children.is_empty() {
                        return Err(invalid(
                            "out of order: `OpPhi`s should come before \
                             the rest of the block's instructions",
                        ));
                    }

                    let inputs = &mut current_block_region_def.inputs;
                    let start = u32::try_from(inputs.len()).unwrap();
                    inputs.extend(result_type.disaggregated_leaf_types(&cx).zip(start..).map(
                        |(ty, def_idx)| {
                            func_def_body.vars.define(
                                &cx,
                                VarDecl {
                                    attrs: attrs_for_result_leaf(ty),
                                    ty,

                                    def_parent: Either::Left(current_block.region),
                                    def_idx,
                                },
                            )
                        },
                    ));
                    let end = u32::try_from(inputs.len()).unwrap();

                    local_id_defs.insert(
                        result_id.unwrap(),
                        LocalIdDef::Value {
                            whole_type: result_type,
                            leaves: NonZeroU32::new(end - start).map_or(
                                Either::Right(SmallVec::new()),
                                |count| {
                                    Either::Left(VarRange { start: inputs[start as usize], count })
                                },
                            ),
                        },
                    );
                    continue;
                }

                if [wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode) {
                    let is_second_to_last_in_block = lookahead_raw_inst(2)
                        .is_none_or(|next_raw_inst| next_raw_inst.without_ids.opcode == wk.OpLabel);

                    if !is_second_to_last_in_block {
                        return Err(invalid(
                            "out of order: a merge instruction should be the last \
                             instruction before the block's terminator",
                        ));
                    }

                    // HACK(eddyb) we want to at least record `OpLoopMerge`s'
                    // impact on the shape of a loop, for restructurization.
                    if opcode == wk.OpLoopMerge {
                        assert_eq!(ids.len(), 2);
                        let loop_merge_target = match lookup_id(ids[0])? {
                            LocalIdDef::Value { .. } => return Err(invalid("expected label ID")),
                            LocalIdDef::BlockLabel(target) => target,
                        };

                        func_def_body
                            .unstructured_cfg
                            .as_mut()
                            .unwrap()
                            .loop_merge_to_loop_header
                            .insert(loop_merge_target, current_block.region);
                    }

                    // HACK(eddyb) merges are mostly ignored - this may be lossy,
                    // especially wrt the `SelectionControl` and `LoopControl`
                    // operands, but it's not obvious how they should map to
                    // some "structured regions" replacement for the CFG.
                    continue;
                }

                // All control-flow instructions have been handled above.
                // Only `Node`s get generated below here.

                let append_node = |func: FuncAtMut<'_, ()>, node_def: NodeDef| {
                    let node = func.nodes.define(&cx, node_def.into());
                    func.regions[current_block.region].children.insert_last(node, func.nodes);
                    node
                };

                let lookup_value_id = |id| match lookup_id(id)? {
                    LocalIdDef::Value { whole_type, leaves } => Ok((whole_type, leaves)),
                    LocalIdDef::BlockLabel(_) => Err(invalid(
                        "unsupported use of block label as a value, \
                         in non-terminator instruction",
                    )),
                };

                // Special-case instructions which deal with aggregates as
                // "containers" for their leaves, and so have an effect which
                // can be interpreted eagerly on the disaggregated form.
                // FIXME(eddyb) this may lose semantic `attrs`
                let eagerly_lowered_result = if opcode == wk.OpCompositeConstruct {
                    let result_type = result_type.unwrap();

                    match cx[result_type].spv_value_lowering() {
                        spv::ValueLowering::Direct => None,
                        spv::ValueLowering::Disaggregate(_) => {
                            let mut all_leaves =
                                SmallVec::with_capacity(cx[result_type].disaggregated_leaf_count());
                            for &id in ids {
                                let (_, leaves) = lookup_value_id(id)?;
                                all_leaves.extend(leaves.iter(func_def_body.at(())));
                            }
                            if all_leaves.len() == cx[result_type].disaggregated_leaf_count() {
                                Some(LocalIdDef::Value {
                                    whole_type: result_type,
                                    leaves: Either::Right(all_leaves),
                                })
                            } else {
                                None
                            }
                        }
                    }
                } else if [wk.OpCompositeExtract, wk.OpCompositeInsert].contains(&opcode) {
                    let result_type = result_type.unwrap();

                    let (&composite_id, ids_without_last) = ids.split_last().unwrap();
                    let (composite_type, leaves) = lookup_value_id(composite_id)?;

                    // HACK(eddyb) `replace_component` and `rebuild_composite`
                    // are always both `None` or both `Some`, but splitting the
                    // two aspects of `OpCompositeInsert` makes it easier later.
                    let (component_type, replace_component, rebuild_composite);
                    match ids_without_last[..] {
                        [] => {
                            component_type = result_type;
                            replace_component = None;
                            rebuild_composite = None;
                        }
                        [replacement_component_id] => {
                            let (replacement_component_type, replacement_component_leaves) =
                                lookup_value_id(replacement_component_id)?;

                            component_type = replacement_component_type;
                            replace_component = Some(replacement_component_leaves);
                            rebuild_composite = Some(result_type);
                        }
                        _ => unreachable!(),
                    }

                    // HACK(eddyb) this is a `try {...}`-like use of a closure.
                    (|| {
                        if let Some(expected_type) = rebuild_composite
                            && composite_type != expected_type
                        {
                            return None;
                        }

                        let mut imms = imms.iter();
                        let (leaf_range, leaf_type) = match cx[composite_type].spv_value_lowering()
                        {
                            spv::ValueLowering::Direct => return None,
                            spv::ValueLowering::Disaggregate(_) => composite_type
                                .aggregate_component_path_leaf_range_and_type(
                                    &cx,
                                    &mut imms.by_ref().map(|&imm| match imm {
                                        spv::Imm::Short(_, i) => i,
                                        _ => unreachable!(),
                                    }),
                                )?,
                        };
                        let non_aggregate_indexing_imms = imms.as_slice();

                        if non_aggregate_indexing_imms.is_empty() && leaf_type != component_type {
                            return None;
                        }

                        let component_leaves = leaves
                            .iter(func_def_body.at(()))
                            .skip(leaf_range.start)
                            .take(leaf_range.len());

                        // If there's any leftover indices they must be indexing
                        // into a vector/matrix, which requires separate handling.
                        let component_leaves = if !non_aggregate_indexing_imms.is_empty() {
                            let non_aggregate_composite =
                                component_leaves.exactly_one().ok().unwrap();

                            let leaf_spv_inst = spv::Inst {
                                opcode,
                                imms: non_aggregate_indexing_imms.iter().copied().collect(),
                            };
                            let leaf_output_type = match rebuild_composite {
                                Some(_) => leaf_type,
                                None => component_type,
                            };
                            let leaf_inputs: SmallVec<[Value; 2]> = replace_component
                                .map(|replacement_leaves| {
                                    replacement_leaves
                                        .iter(func_def_body.at(()))
                                        .exactly_one()
                                        .ok()
                                        .unwrap()
                                })
                                .into_iter()
                                .chain([non_aggregate_composite])
                                .collect();
                            let leaf_kind = leaf_spv_inst
                                .as_canonical_node_kind(
                                    &cx,
                                    [leaf_output_type].into_iter(),
                                    leaf_inputs.iter().map(|&v| func_def_body.at(v).type_of(&cx)),
                                )
                                .unwrap_or(DataInstKind::SpvInst(
                                    leaf_spv_inst,
                                    spv::InstLowering::default(),
                                ));
                            let leaf_inst = append_node(
                                func_def_body.at_mut(()),
                                DataInstDef {
                                    attrs,
                                    kind: leaf_kind,
                                    inputs: leaf_inputs,
                                    child_regions: [].into_iter().collect(),
                                    outputs: [].into_iter().collect(),
                                },
                            );

                            let leaf_outputs = &mut func_def_body.nodes[leaf_inst].outputs;
                            let leaf_output_var = func_def_body.vars.define(
                                &cx,
                                VarDecl {
                                    // FIXME(eddyb) this may lose semantic `attrs`.
                                    attrs: AttrSet::default(),
                                    ty: leaf_output_type,

                                    def_parent: Either::Right(leaf_inst),
                                    def_idx: leaf_outputs.len().try_into().unwrap(),
                                },
                            );
                            leaf_outputs.push(leaf_output_var);

                            Either::Left([Value::Var(leaf_output_var)].into_iter())
                        } else {
                            Either::Right(replace_component.map_or(
                                Either::Left(component_leaves),
                                |replacement_leaves| {
                                    Either::Right(replacement_leaves.iter(func_def_body.at(())))
                                },
                            ))
                        };

                        assert_eq!(
                            component_leaves.len(),
                            cx[component_type].disaggregated_leaf_count()
                        );

                        let leaves = match rebuild_composite {
                            Some(_) => leaves
                                .iter(func_def_body.at(()))
                                .take(leaf_range.start)
                                .chain(component_leaves)
                                .chain(leaves.iter(func_def_body.at(())).skip(leaf_range.end))
                                .collect(),
                            None => component_leaves.collect(),
                        };

                        Some(LocalIdDef::Value {
                            whole_type: result_type,
                            // FIXME(eddyb) avoid allocating somehow, like
                            // try "recompressing" into a `VarRange`, or
                            // preserving that form throughout above?
                            leaves: Either::Right(leaves),
                        })
                    })()
                } else {
                    None
                };
                if let Some(def) = eagerly_lowered_result {
                    local_id_defs.insert(result_id.unwrap(), def);
                    continue;
                }

                let mut ids = &ids[..];
                let mut kind = if opcode == wk.OpFunctionCall {
                    assert!(imms.is_empty());
                    let callee_id = ids[0];
                    let maybe_callee = id_defs
                        .get(&callee_id)
                        .map(|id_def| match *id_def {
                            IdDef::Func(func) => Ok(func),
                            _ => Err(id_def.descr(&cx)),
                        })
                        .transpose()
                        .map_err(|descr| {
                            invalid(&format!(
                                "unsupported use of {descr} as the `OpFunctionCall` callee"
                            ))
                        })?;

                    match maybe_callee {
                        Some(callee) => {
                            ids = &ids[1..];
                            DataInstKind::FuncCall(callee)
                        }

                        // HACK(eddyb) this should be an error, but it shows
                        // up in Rust-GPU output (likely a zombie?).
                        None => DataInstKind::SpvInst(
                            raw_inst.without_ids.clone(),
                            spv::InstLowering::default(),
                        ),
                    }
                } else if opcode == wk.OpExtInst {
                    let ext_set_id = ids[0];
                    ids = &ids[1..];

                    let inst = match imms[..] {
                        [spv::Imm::Short(kind, inst)] => {
                            assert_eq!(kind, wk.LiteralExtInstInteger);
                            inst
                        }
                        _ => unreachable!(),
                    };

                    let ext_set = match id_defs.get(&ext_set_id) {
                        Some(&IdDef::SpvExtInstImport(name)) => Ok(name),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{ext_set_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the `OpExtInst` \
                             extended instruction set ID"
                        ))
                    })?;

                    DataInstKind::SpvExtInst {
                        ext_set,
                        inst,
                        lowering: spv::InstLowering::default(),
                    }
                } else {
                    DataInstKind::SpvInst(
                        raw_inst.without_ids.clone(),
                        spv::InstLowering::default(),
                    )
                };

                // HACK(eddyb) only factored out due to `kind`'s mutable borrow.
                let call_ret_type = match &kind {
                    DataInstKind::FuncCall(_) => true,
                    DataInstKind::SpvInst(spv_inst, _) => {
                        spv_inst.opcode == wk.OpFunctionPointerCallINTEL
                    }
                    _ => false,
                }
                .then(|| result_type.unwrap());

                let mut spv_inst_lowering = match &mut kind {
                    DataInstKind::SpvInst(_, lowering)
                    | DataInstKind::SpvExtInst { lowering, .. } => Some(lowering),

                    // NOTE(eddyb) function signatures and calls keep their
                    // disaggregation even when lifting back to SPIR-V, so
                    // no `spv::InstLowering` is tracked for them.
                    DataInstKind::FuncCall(_) => None,

                    NodeKind::Select(_)
                    | NodeKind::Loop { .. }
                    | NodeKind::ExitInvocation(_)
                    | DataInstKind::Scalar(_)
                    | DataInstKind::Vector(_)
                    | DataInstKind::Mem(_)
                    | DataInstKind::QPtr(_)
                    | DataInstKind::ThunkBind(_) => {
                        unreachable!()
                    }
                };

                let (output_count_u32, output_leaf_types) = result_id
                    .and_then(|_| {
                        let result_type = result_type.unwrap();

                        // HACK(eddyb) `OpTypeVoid` special-cased for calls
                        // as if it were an aggregate with `0` leaves.
                        let ret_void = call_ret_type.is_some_and(|ty| match &cx[ty].kind {
                            TypeKind::SpvInst { spv_inst: ret_type_spv_inst, .. } => {
                                ret_type_spv_inst.opcode == wk.OpTypeVoid
                            }
                            _ => false,
                        });

                        if let Some(spv_inst_lowering) = &mut spv_inst_lowering {
                            spv_inst_lowering.disaggregated_output =
                                match cx[result_type].spv_value_lowering() {
                                    // HACK(eddyb) `spv_inst_lowering` can only
                                    // coexist with `call_ret_type` for indirect
                                    // calls (`OpFunctionPointerCallINTEL`).
                                    spv::ValueLowering::Direct => ret_void,
                                    spv::ValueLowering::Disaggregate(_) => true,
                                }
                                .then_some(result_type);
                        }

                        (!ret_void).then_some(result_type)
                    })
                    .map_or((0, None), |result_type| {
                        (
                            cx[result_type].disaggregated_leaf_count_u32(),
                            Some(result_type.disaggregated_leaf_types(&cx)),
                        )
                    });
                let output_leaf_types = output_leaf_types.into_iter().flatten();

                let mut inputs = SmallVec::with_capacity(ids.len());
                for &id in ids {
                    let (whole_input_type, leaves) = lookup_value_id(id)?;

                    let start = u32::try_from(inputs.len()).unwrap();
                    inputs.extend(leaves.iter(func_def_body.at(())));
                    let end = u32::try_from(inputs.len()).unwrap();

                    if let spv::ValueLowering::Disaggregate(_) =
                        cx[whole_input_type].spv_value_lowering()
                        && let Some(lowering) = &mut spv_inst_lowering
                    {
                        lowering.disaggregated_inputs.push((start..end, whole_input_type));
                    }
                }

                let node = append_node(
                    func_def_body.at_mut(()),
                    NodeDef {
                        attrs,
                        kind,
                        inputs,
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    },
                );

                if let Some(result_id) = result_id {
                    let outputs = &mut func_def_body.nodes[node].outputs;
                    assert_eq!(outputs.len(), 0);
                    outputs.extend(output_leaf_types.zip_eq(0..output_count_u32).map(
                        |(ty, def_idx)| {
                            func_def_body.vars.define(
                                &cx,
                                VarDecl {
                                    // FIXME(eddyb) split attrs between output and inst.
                                    attrs: AttrSet::default(),
                                    ty,

                                    def_parent: Either::Right(node),
                                    def_idx,
                                },
                            )
                        },
                    ));

                    local_id_defs.insert(
                        result_id,
                        LocalIdDef::Value {
                            whole_type: result_type.unwrap(),
                            leaves: NonZeroU32::new(output_count_u32)
                                .map_or(Either::Right(SmallVec::new()), |count| {
                                    Either::Left(VarRange { start: outputs[0], count })
                                }),
                        },
                    );
                }

                // HACK(eddyb) doing this after defining the maybe-uncanonical
                // node, just to keep the iterators simpler.
                let node_def = &mut func_def_body.nodes[node];
                if let DataInstKind::SpvInst(spv_inst, lowering) = &node_def.kind
                    && lowering.disaggregated_inputs.is_empty()
                    && let Some(canonical_kind) = spv_inst.as_canonical_node_kind(
                        &cx,
                        node_def
                            .outputs
                            .iter()
                            .map(|&output_var| func_def_body.vars[output_var].ty),
                        node_def.inputs.iter().map(|&v| {
                            // HACK(eddyb) `func_def_body.at(v).type_of(cx)`
                            // equivalent, without running into borrow issues.
                            match v {
                                Value::Const(ct) => cx[ct].ty,
                                Value::Var(var) => func_def_body.vars[var].ty,
                            }
                        }),
                    )
                {
                    // FIXME(eddyb) sanity-check the number/types of inputs.
                    node_def.kind = canonical_kind;
                }
            }

            // FIXME(eddyb) all functions should have the appropriate number of
            // `OpFunctionParameter`, even imports.
            if !params.is_empty() {
                if func_decl.params.len() != params.len() {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(&format!(
                        "in %{}, param count differs between `OpTypeFunction` ({}) \
                         and `OpFunctionParameter`s ({})",
                        func_id,
                        func_decl.params.len(),
                        params.len(),
                    )));
                }

                for (i, (func_decl_param, param)) in
                    func_decl.params.iter_mut().zip(params).enumerate()
                {
                    func_decl_param.attrs = param.attrs;
                    if func_decl_param.ty != param.ty {
                        // FIXME(remove) embed IDs in errors by moving them to the
                        // `let invalid = |...| ...;` closure that wraps insts.
                        return Err(invalid(
                            &print::Plan::for_root(
                                &cx,
                                &Diag::err([
                                    format!("in %{func_id}, ").into(),
                                    format!("param #{i}'s type differs between `OpTypeFunction` (")
                                        .into(),
                                    func_decl_param.ty.into(),
                                    ") and `OpFunctionParameter` (".into(),
                                    param.ty.into(),
                                    ")".into(),
                                ])
                                .message,
                            )
                            .pretty_print()
                            .to_string(),
                        ));
                    }
                }
            }

            if !phi_to_values.is_empty() {
                let mut edges = phi_to_values
                    .keys()
                    .map(|key| format!("%{} -> %{}", key.source_block_id, key.target_block_id))
                    .collect::<Vec<_>>();
                edges.dedup();
                // FIXME(remove) embed IDs in errors by moving them to the
                // `let invalid = |...| ...;` closure that wraps insts.
                return Err(invalid(&format!(
                    "in %{}, `OpPhi`s refer to non-existent edges: {}",
                    func_id,
                    edges.join(", ")
                )));
            }

            // Sanity-check the entry block.
            if let Some(func_def_body) = func_def_body
                && block_details[&func_def_body.body].phi_count > 0
            {
                // FIXME(remove) embed IDs in errors by moving them to the
                // `let invalid = |...| ...;` closure that wraps insts.
                return Err(invalid(&format!("in %{func_id}, the entry block contains `OpPhi`s")));
            }
        }

        assert!(module.exports.is_empty());
        module.exports = pending_exports
            .into_iter()
            .map(|export| match export {
                Export::Linkage { name, target_id } => {
                    let exportee = match id_defs.get(&target_id) {
                        Some(id_def @ &IdDef::Const(ct)) => match cx[ct].kind {
                            ConstKind::PtrToGlobalVar { global_var, offset: None } => {
                                Ok(Exportee::GlobalVar(global_var))
                            }
                            _ => Err(id_def.descr(&cx)),
                        },
                        Some(&IdDef::Func(func)) => Ok(Exportee::Func(func)),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{target_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the `LinkageAttributes` target"
                        ))
                    })?;

                    Ok((ExportKey::LinkName(name), exportee))
                }

                Export::EntryPoint { func_id, imms, interface_ids } => {
                    let func = match id_defs.get(&func_id) {
                        Some(&IdDef::Func(func)) => Ok(func),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{func_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!("unsupported use of {descr} as the `OpEntryPoint` target"))
                    })?;
                    let interface_global_vars = interface_ids
                        .into_iter()
                        .map(|id| match id_defs.get(&id) {
                            Some(id_def @ &IdDef::Const(ct)) => match cx[ct].kind {
                                ConstKind::PtrToGlobalVar { global_var, offset: None } => {
                                    Ok(global_var)
                                }
                                _ => Err(id_def.descr(&cx)),
                            },
                            Some(id_def) => Err(id_def.descr(&cx)),
                            None => Err(format!("unknown ID %{id}")),
                        })
                        .map(|result| {
                            result.map_err(|descr| {
                                invalid(&format!(
                                    "unsupported use of {descr} as an \
                                     `OpEntryPoint` interface variable"
                                ))
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    Ok((
                        ExportKey::SpvEntryPoint { imms, interface_global_vars },
                        Exportee::Func(func),
                    ))
                }
            })
            .collect::<io::Result<_>>()?;

        Ok(module)
    }
}
