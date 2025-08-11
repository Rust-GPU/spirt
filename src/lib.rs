//! > <div style="font-size:small;border:1px solid;padding:1em;padding-top:0">
//! > <div align="center">
//! >
//! > ## `SPIR-🇹`
//! >
//! > **⋯🢒 🇹arget 🠆 🇹ransform 🠆 🇹ranslate ⋯🢒**
//! >
//! > </div><br>
//! >
//! > **SPIR-🇹** is a research project aimed at exploring shader-oriented IR designs
//! > derived from SPIR-V, and producing a framework around such an IR to facilitate
//! > advanced compilation pipelines, beyond what existing SPIR-V tooling allows for.
//! >
//! > 🚧 *This project is in active design and development, many details can and will change* 🚧
//! >
//! > </div>
//! >
//! > *&mdash;
#![cfg_attr(
    docsrs,
    // NOTE(eddyb) this requires updating `repository` before every release to
    // end in `/tree/` followed by the tag name, in order to be useful.
    doc = concat!(
        "[`", env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION"), "`'s `README`]",
        "(", env!("CARGO_PKG_REPOSITORY"), "#readme)*  "
    )
)]
#![cfg_attr(
    git_main_docs,
    doc = concat!(
        "[`", env!("CARGO_PKG_NAME"), " @ ", env!("GIT_MAIN_DESCRIBE"), "`'s `README`]",
        "(https://github.com/rust-gpu/spirt/tree/", env!("GIT_MAIN_COMMIT"), "#readme)*  "
    )
)]
#![cfg_attr(
    any(docsrs, git_main_docs),
    doc = "<sup>&nbsp;&nbsp;&nbsp;&nbsp;*(click through for the full version)*</sup>"
)]
// HACK(eddyb) this is only relevant for local builds (which don't need a link).
#![cfg_attr(
    not(any(docsrs, git_main_docs)),
    doc = concat!("`", env!("CARGO_PKG_NAME"), "`'s `README`*  ")
)]
//!
//! *Check out also [the `rust-gpu/spirt` GitHub repository](https://github.com/rust-gpu/spirt),
//! for any additional developments.*
//!
//! #### Notable types/modules
//!
//! ##### IR data types
// HACK(eddyb) using `(struct.Context.html)` to link `Context`, not `context::Context`.
//! * [`Context`](struct.Context.html): handles interning ([`Type`]s, [`Const`]s, etc.) and allocating entity handles
//! * [`Module`]: owns [`Func`]s and [`GlobalVar`]s (rooted by [`exports`](Module::exports))
//! * [`FuncDefBody`]: owns [`Region`]s and [`DataInst`]s (rooted by [`body`](FuncDefBody::body))
//!
//! ##### Utilities and passes
//! * [`print`](mod@print): pretty-printer with (styled and hyperlinked) HTML output
//! * [`spv::lower`]/[`spv::lift`]: conversion from/to SPIR-V
//! * [`cfg::Structurizer`]: (re)structurization from arbitrary control-flow
//!

// BEGIN - Embark standard lints v6 for Rust 1.55+
// do not change or add/remove here, but one can add exceptions after this section
// for more info see: <https://github.com/EmbarkStudios/rust-ecosystem/issues/59>
#![deny(unsafe_code)]
#![warn(
    clippy::all,
    clippy::await_holding_lock,
    clippy::char_lit_as_u8,
    clippy::checked_conversions,
    clippy::dbg_macro,
    clippy::debug_assert_with_mut_call,
    clippy::doc_markdown,
    clippy::empty_enum,
    clippy::enum_glob_use,
    clippy::exit,
    clippy::expl_impl_clone_on_copy,
    clippy::explicit_deref_methods,
    clippy::explicit_into_iter_loop,
    clippy::fallible_impl_from,
    clippy::filter_map_next,
    clippy::flat_map_option,
    clippy::float_cmp_const,
    clippy::fn_params_excessive_bools,
    clippy::from_iter_instead_of_collect,
    clippy::if_let_mutex,
    clippy::implicit_clone,
    clippy::imprecise_flops,
    clippy::inefficient_to_string,
    clippy::invalid_upcast_comparisons,
    clippy::large_digit_groups,
    clippy::large_stack_arrays,
    clippy::large_types_passed_by_value,
    clippy::let_unit_value,
    clippy::linkedlist,
    clippy::lossy_float_literal,
    clippy::macro_use_imports,
    clippy::manual_ok_or,
    clippy::map_err_ignore,
    clippy::map_flatten,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::match_wild_err_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::mem_forget,
    clippy::missing_enforced_import_renames,
    clippy::mut_mut,
    clippy::mutex_integer,
    clippy::needless_borrow,
    clippy::needless_continue,
    clippy::needless_for_each,
    clippy::option_option,
    clippy::path_buf_push_overwrite,
    clippy::ptr_as_ptr,
    clippy::rc_mutex,
    clippy::ref_option_ref,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::same_functions_in_if_condition,
    clippy::semicolon_if_nothing_returned,
    clippy::single_match_else,
    clippy::string_add_assign,
    clippy::string_add,
    clippy::string_lit_as_bytes,
    clippy::string_to_string,
    clippy::todo,
    clippy::trait_duplication_in_bounds,
    clippy::unimplemented,
    clippy::unnested_or_patterns,
    clippy::unused_self,
    clippy::useless_transmute,
    clippy::verbose_file_reads,
    clippy::zero_sized_map_values,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms
)]
// END - Embark standard lints v6 for Rust 1.55+
// crate-specific exceptions:
#![allow(
    // NOTE(eddyb) ignored for readability (`match` used when `if let` is too long).
    clippy::single_match_else,

    // NOTE(eddyb) ignored because it's misguided to suggest `let mut s = ...;`
    // and `s.push_str(...);` when `+` is equivalent and does not require `let`.
    clippy::string_add,

    // FIXME(eddyb) rework doc comments to conform to linted expectations.
    clippy::too_long_first_doc_paragraph,
)]
// NOTE(eddyb) this is stronger than the "Embark standard lints" above, because
// we almost never need `unsafe` code and this is a further "speed bump" to it.
#![forbid(unsafe_code)]

// NOTE(eddyb) all the modules are declared here, but they're documented "inside"
// (i.e. using inner doc comments).
pub mod cf;
mod context;
pub mod func_at;
pub mod print;
pub mod transform;
pub mod visit;
pub mod passes {
    //! IR transformations (typically whole-[`Module`](crate::Module)).
    //
    // NOTE(eddyb) inline `mod` to avoid adding APIs here, it's just namespacing.

    pub mod legalize;
    pub mod link;
    pub mod qptr;
}
pub mod mem;
pub mod qptr;
pub mod spv;

use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::rc::Rc;

// HACK(eddyb) work around the lack of `FxIndex{Map,Set}` type aliases elsewhere.
#[doc(hidden)]
type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
#[doc(hidden)]
type FxIndexSet<V> = indexmap::IndexSet<V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

// NOTE(eddyb) these reexports are all documented inside `context`.
// FIXME(eddyb) maybe make an `entity` module to move either the definitions,
// or at least the re-exports - an `ir` module might help too, organizationally?
pub use context::{
    Context, EntityDefs, EntityList, EntityListIter, EntityOrientedDenseMap, EntityOrientedMapKey,
};

/// Interned handle for a [`str`].
pub use context::InternedStr;

// HACK(eddyb) this only serves to disallow modifying the `cx` field of `Module`.
#[doc(hidden)]
mod sealed {
    use super::*;
    use std::rc::Rc;

    #[derive(Clone)]
    pub struct Module {
        /// Context used for everything interned, in this module.
        ///
        /// Notable choices made for this field:
        /// * private to disallow switching the context of a module
        /// * [`Rc`] sharing to allow multiple modules to use the same context
        ///   (`Context: !Sync` because of the interners so it can't be `Arc`)
        cx: Rc<Context>,

        pub dialect: ModuleDialect,
        pub debug_info: ModuleDebugInfo,

        pub global_vars: EntityDefs<GlobalVar>,
        pub funcs: EntityDefs<Func>,

        pub exports: FxIndexMap<ExportKey, Exportee>,
    }

    impl Module {
        pub fn new(cx: Rc<Context>, dialect: ModuleDialect, debug_info: ModuleDebugInfo) -> Self {
            Self {
                cx,

                dialect,
                debug_info,

                global_vars: Default::default(),
                funcs: Default::default(),

                exports: Default::default(),
            }
        }

        // FIXME(eddyb) `cx_ref` might be the better default in situations where
        // the module doesn't need to be modified, figure out if that's common.
        pub fn cx(&self) -> Rc<Context> {
            self.cx.clone()
        }

        pub fn cx_ref(&self) -> &Rc<Context> {
            &self.cx
        }
    }
}
pub use sealed::Module;

/// Semantic properties of a SPIR-T module (not tied to any declarations/definitions).
#[derive(Clone)]
pub enum ModuleDialect {
    Spv(spv::Dialect),
}

/// Non-semantic details (i.e. debuginfo) of a SPIR-Y module (not tied to any
/// declarations/definitions).
#[derive(Clone)]
pub enum ModuleDebugInfo {
    Spv(spv::ModuleDebugInfo),
}

/// An unique identifier (e.g. a link name, or "symbol") for a module export.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ExportKey {
    LinkName(InternedStr),

    SpvEntryPoint {
        imms: SmallVec<[spv::Imm; 2]>,
        // FIXME(eddyb) remove this by recomputing the interface vars.
        interface_global_vars: SmallVec<[GlobalVar; 4]>,
    },
}

/// A definition exported out of a module (see also [`ExportKey`]).
#[derive(Copy, Clone)]
pub enum Exportee {
    GlobalVar(GlobalVar),
    Func(Func),
}

/// Interned handle for an [`AttrSetDef`](crate::AttrSetDef)
/// (a set of [`Attr`](crate::Attr)s).
pub use context::AttrSet;

/// Definition for an [`AttrSet`]: a set of [`Attr`]s.
#[derive(Default, PartialEq, Eq, Hash)]
pub struct AttrSetDef {
    // FIXME(eddyb) consider "persistent datastructures" (e.g. the `im` crate).
    // FIXME(eddyb) use `BTreeMap<Attr, AttrValue>` and split some of the params
    // between the `Attr` and `AttrValue` based on specified uniquness.
    // FIXME(eddyb) don't put debuginfo in here, but rather at use sites
    // (for e.g. types, with component types also having the debuginfo
    // bundled at the use site of the composite type) in order to allow
    // deduplicating definitions that only differ in debuginfo, in SPIR-T,
    // and still lift SPIR-V with duplicate definitions, out of that.
    pub attrs: BTreeSet<Attr>,
}

impl AttrSetDef {
    pub fn dbg_src_loc(&self) -> Option<DbgSrcLoc> {
        // FIXME(eddyb) seriously consider moving to `BTreeMap` (see above).
        // HACK(eddyb) this assumes `Attr::DbgSrcLoc` is the first of `Attr`!
        match self.attrs.first() {
            Some(&Attr::DbgSrcLoc(OrdAssertEq(dbg_src_loc))) => Some(dbg_src_loc),
            _ => None,
        }
    }

    pub fn set_dbg_src_loc(&mut self, dbg_src_loc: DbgSrcLoc) {
        // FIXME(eddyb) seriously consider moving to `BTreeMap` (see above).
        // HACK(eddyb) this assumes `Attr::DbgSrcLoc` is the first of `Attr`!
        if let Some(Attr::DbgSrcLoc(_)) = self.attrs.first() {
            self.attrs.pop_first().unwrap();
        }
        self.attrs.insert(Attr::DbgSrcLoc(OrdAssertEq(dbg_src_loc)));
    }

    pub fn diags(&self) -> &[Diag] {
        // FIXME(eddyb) seriously consider moving to `BTreeMap` (see above).
        // HACK(eddyb) this assumes `Attr::Diagnostics` is the last of `Attr`!
        match self.attrs.last() {
            Some(Attr::Diagnostics(OrdAssertEq(diags))) => diags,
            _ => &[],
        }
    }

    pub fn mutate_diags(&mut self, f: impl FnOnce(&mut Vec<Diag>)) {
        // FIXME(eddyb) seriously consider moving to `BTreeMap` (see above).
        // HACK(eddyb) this assumes `Attr::Diagnostics` is the last of `Attr`!
        let mut attr = if let Some(Attr::Diagnostics(_)) = self.attrs.last() {
            self.attrs.pop_last().unwrap()
        } else {
            Attr::Diagnostics(OrdAssertEq(vec![]))
        };
        match &mut attr {
            Attr::Diagnostics(OrdAssertEq(diags)) => f(diags),
            _ => unreachable!(),
        }
        self.attrs.insert(attr);
    }

    // HACK(eddyb) these only exist to avoid changing code working with `AttrSetDef`s.
    pub fn push_diags(&mut self, new_diags: impl IntoIterator<Item = Diag>) {
        self.mutate_diags(|diags| diags.extend(new_diags));
    }
    pub fn push_diag(&mut self, diag: Diag) {
        self.push_diags([diag]);
    }
}

// FIXME(eddyb) should these methods be elsewhere?
impl AttrSet {
    // FIXME(eddyb) could these two methods have a better name?
    pub fn reintern_with(self, cx: &Context, f: impl FnOnce(&mut AttrSetDef)) -> Self {
        let mut new_attrs = AttrSetDef { attrs: cx[self].attrs.clone() };
        f(&mut new_attrs);
        cx.intern(new_attrs)
    }
    pub fn mutate(&mut self, cx: &Context, f: impl FnOnce(&mut AttrSetDef)) {
        *self = self.reintern_with(cx, f);
    }

    pub fn dbg_src_loc(self, cx: &Context) -> Option<DbgSrcLoc> {
        if self == AttrSet::default() {
            return None;
        }
        cx[self].dbg_src_loc()
    }

    pub fn set_dbg_src_loc(&mut self, cx: &Context, dbg_src_loc: DbgSrcLoc) {
        self.mutate(cx, |attrs| attrs.set_dbg_src_loc(dbg_src_loc));
    }

    pub fn diags(self, cx: &Context) -> &[Diag] {
        cx[self].diags()
    }

    pub fn push_diags(&mut self, cx: &Context, diags: impl IntoIterator<Item = Diag>) {
        self.mutate(cx, |attrs| attrs.push_diags(diags));
    }

    pub fn push_diag(&mut self, cx: &Context, diag: Diag) {
        self.push_diags(cx, [diag]);
    }
}

/// Any semantic or non-semantic (debuginfo) decoration/modifier, that can be
/// *optionally* applied to some declaration/definition.
///
/// Always used via [`AttrSetDef`] (interned as [`AttrSet`]).
//
// FIXME(eddyb) consider interning individual attrs, not just `AttrSet`s.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub enum Attr {
    // HACK(eddyb) this must be the first variant of `Attr` for the correctness
    // of `AttrSetDef::{dbg_src_loc,set_dbg_src_loc}`.
    DbgSrcLoc(OrdAssertEq<DbgSrcLoc>),

    /// Memory-specific attributes (see [`mem::MemAttr`]).
    #[from]
    Mem(mem::MemAttr),

    /// `QPtr`-specific attributes (see [`qptr::QPtrAttr`]).
    #[from]
    QPtr(qptr::QPtrAttr),

    SpvAnnotation(spv::Inst),

    /// Some SPIR-V instructions, like `OpFunction`, take a bitflags operand
    /// that is effectively an optimization over using `OpDecorate`.
    //
    // FIXME(eddyb) handle flags having further operands as parameters.
    SpvBitflagsOperand(spv::Imm),

    /// Can be used anywhere to record [`Diag`]nostics produced during a pass,
    /// while allowing the pass to continue (and its output to be pretty-printed).
    //
    // HACK(eddyb) this must be the last variant of `Attr` for the correctness
    // of`AttrSetDef::{diags,mutate_diags}` (this also helps with printing order).
    Diagnostics(OrdAssertEq<Vec<Diag>>),
}

/// Simple `file:line:column`-style debuginfo, similar to SPIR-V `OpLine`,
/// but also supporting `(line, column)` ranges, and inlined locations.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct DbgSrcLoc {
    pub file_path: InternedStr,

    // FIXME(eddyb) `Range` might make sense here but these are inclusive,
    // and `range::RangeInclusive` (the non-`Iterator` version of `a..=b`)
    // isn't stable (nor the type of `a..=b` expressions), yet.
    pub start_line_col: (u32, u32),
    pub end_line_col: (u32, u32),

    /// To describe locations originally in the callee of a call that was inlined,
    /// the name of the callee and attributes describing the callsite are used,
    /// where callsite attributes are expected to contain an [`Attr::DbgSrcLoc`].
    pub inlined_callee_name_and_call_site: Option<(InternedStr, AttrSet)>,
}

/// Diagnostics produced by SPIR-T passes, and recorded in [`Attr::Diagnostics`].
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Diag {
    pub level: DiagLevel,
    // FIXME(eddyb) this may want to be `SmallVec` and/or `Rc`?
    pub message: Vec<DiagMsgPart>,
}

impl Diag {
    pub fn new(level: DiagLevel, message: impl IntoIterator<Item = DiagMsgPart>) -> Self {
        Self { level, message: message.into_iter().collect() }
    }

    // FIMXE(eddyb) make macros more ergonomic than this, for interpolation.
    #[track_caller]
    pub fn bug(message: impl IntoIterator<Item = DiagMsgPart>) -> Self {
        Self::new(DiagLevel::Bug(std::panic::Location::caller()), message)
    }

    pub fn err(message: impl IntoIterator<Item = DiagMsgPart>) -> Self {
        Self::new(DiagLevel::Error, message)
    }

    pub fn warn(message: impl IntoIterator<Item = DiagMsgPart>) -> Self {
        Self::new(DiagLevel::Warning, message)
    }
}

/// The "severity" level of a [`Diag`]nostic.
///
/// Note: `Bug` diagnostics track their emission point for easier identification.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum DiagLevel {
    Bug(&'static std::panic::Location<'static>),
    Error,
    Warning,
}

/// One part of a [`Diag`]nostic message, allowing rich interpolation.
///
/// Note: [`visit::Visitor`] and [`transform::Transformer`] *do not* interact
/// with any interpolated information, and it's instead treated as "frozen" data.
#[derive(Clone, PartialEq, Eq, Hash, derive_more::From)]
// HACK(eddyb) this sets the default as "opt-out", to avoid `#[from(forward)]`
// on the `Plain` variant from making it "opt-in" for all variants.
#[from]
pub enum DiagMsgPart {
    #[from(forward)]
    Plain(Cow<'static, str>),

    // FIXME(eddyb) use `dyn Trait` instead of listing out a few cases.
    Attrs(AttrSet),
    Type(Type),
    Const(Const),
    MemAccesses(mem::MemAccesses),
}

/// Wrapper to limit `Ord` for interned index types (e.g. [`InternedStr`])
/// to only situations where the interned index reflects contents (i.e. equality).
//
// FIXME(eddyb) this is not ideal, and it might be more useful to replace the
// `BTreeSet<Attr>` with an `BTreeMap<Attr, AttrValue>`, where only `Attr` needs
// to be `Ord`, and the details that cannot be `Ord`, can be moved to `AttrValue`.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct OrdAssertEq<T>(pub T);

impl<T: Eq> PartialOrd for OrdAssertEq<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq> Ord for OrdAssertEq<T> {
    #[track_caller]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        assert!(
            self == other,
            "OrdAssertEq<{}>::cmp called with unequal values",
            std::any::type_name::<T>(),
        );
        std::cmp::Ordering::Equal
    }
}

/// Interned handle for a [`TypeDef`](crate::TypeDef).
pub use context::Type;

/// Definition for a [`Type`].
//
// FIXME(eddyb) maybe special-case some basic types like integers.
#[derive(PartialEq, Eq, Hash)]
pub struct TypeDef {
    pub attrs: AttrSet,
    pub kind: TypeKind,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    /// "Quasi-pointer", an untyped pointer-like abstract scalar that can represent
    /// both memory locations (in any address space) and other kinds of locations
    /// (e.g. SPIR-V `OpVariable`s in non-memory "storage classes").
    ///
    /// This flexibility can be used to represent pointers from source languages
    /// that expect/are defined to operate on untyped memory (C, C++, Rust, etc.),
    /// that can then be legalized away (e.g. via inlining) or even emulated.
    ///
    /// Information narrowing down how values of the type may be created/used
    /// (e.g. "points to variable `x`" or "accessed at offset `y`") can be found
    /// attached as `Attr`s on those `Value`s (see [`Attr::QPtr`]).
    //
    // FIXME(eddyb) a "refinement system" that's orthogonal from types, and kept
    // separately in e.g. `RegionInputDecl`, might be a better approach?
    QPtr,

    SpvInst {
        spv_inst: spv::Inst,
        // FIXME(eddyb) find a better name.
        type_and_const_inputs: SmallVec<[TypeOrConst; 2]>,
    },

    /// The type of a [`ConstKind::SpvStringLiteralForExtInst`] constant, i.e.
    /// a SPIR-V `OpString` with no actual type in SPIR-V.
    SpvStringLiteralForExtInst,
}

// HACK(eddyb) this behaves like an implicit conversion for `cx.intern(...)`.
impl context::InternInCx<Type> for TypeKind {
    fn intern_in_cx(self, cx: &Context) -> Type {
        cx.intern(TypeDef { attrs: Default::default(), kind: self })
    }
}

// HACK(eddyb) this is like `Either<Type, Const>`, only used in `TypeKind::SpvInst`,
// and only because SPIR-V type definitions can references both types and consts.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum TypeOrConst {
    Type(Type),
    Const(Const),
}

/// Interned handle for a [`ConstDef`](crate::ConstDef) (a constant value).
pub use context::Const;

/// Definition for a [`Const`]: a constant value.
//
// FIXME(eddyb) maybe special-case some basic consts like integer literals.
#[derive(PartialEq, Eq, Hash)]
pub struct ConstDef {
    pub attrs: AttrSet,
    pub ty: Type,
    pub kind: ConstKind,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ConstKind {
    PtrToGlobalVar(GlobalVar),

    // HACK(eddyb) this is a fallback case that should become increasingly rare
    // (especially wrt recursive consts), `Rc` means it can't bloat `ConstDef`.
    SpvInst {
        spv_inst_and_const_inputs: Rc<(spv::Inst, SmallVec<[Const; 4]>)>,
    },

    /// SPIR-V `OpString`, but only when used as an operand for an `OpExtInst`,
    /// which can't have literals itself - for non-string literals `OpConstant*`
    /// are readily usable, but only `OpString` is supported for string literals.
    SpvStringLiteralForExtInst(InternedStr),
}

/// Declarations ([`GlobalVarDecl`], [`FuncDecl`]) can contain a full definition,
/// or only be an import of a definition (e.g. from another module).
#[derive(Clone)]
pub enum DeclDef<D> {
    Imported(Import),
    Present(D),
}

/// An identifier (e.g. a link name, or "symbol") for an import declaration.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Import {
    LinkName(InternedStr),
}

/// Entity handle for a [`GlobalVarDecl`](crate::GlobalVarDecl) (a global variable).
pub use context::GlobalVar;

/// Declaration/definition for a [`GlobalVar`]: a global variable.
//
// FIXME(eddyb) mark any `GlobalVar` not *controlled* by the SPIR-V module
// (roughly: storage classes that don't allow initializers, i.e. most of them),
// as an "import" from "the shader interface", and therefore "externally visible",
// to implicitly distinguish it from `GlobalVar`s internal to the module
// (such as any constants that may need to be reshaped for legalization).
#[derive(Clone)]
pub struct GlobalVarDecl {
    pub attrs: AttrSet,

    /// The type of a pointer to the global variable (as opposed to the value type).
    // FIXME(eddyb) try to replace with value type (or at least have that too).
    pub type_of_ptr_to: Type,

    /// When `type_of_ptr_to` is `QPtr`, `shape` must be used to describe the
    /// global variable (see `GlobalVarShape`'s documentation for more details).
    pub shape: Option<mem::shapes::GlobalVarShape>,

    /// The address space the global variable will be allocated into.
    pub addr_space: AddrSpace,

    pub def: DeclDef<GlobalVarDefBody>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum AddrSpace {
    /// Placeholder for `GlobalVar`s with `GlobalVarShape::Handles`.
    ///
    /// In SPIR-V, this corresponds to `UniformConstant` for `Handle::Opaque`,
    /// or the buffer's storage class for `Handle::Buffer`.
    Handles,

    SpvStorageClass(u32),
}

/// The body of a [`GlobalVar`] definition.
#[derive(Clone)]
pub struct GlobalVarDefBody {
    /// If `Some`, the global variable will start out with the specified value.
    pub initializer: Option<Const>,
}

/// Entity handle for a [`FuncDecl`](crate::FuncDecl) (a function).
pub use context::Func;

/// Declaration/definition for a [`Func`]: a function.
#[derive(Clone)]
pub struct FuncDecl {
    pub attrs: AttrSet,

    pub ret_type: Type,

    pub params: SmallVec<[FuncParam; 2]>,

    pub def: DeclDef<FuncDefBody>,
}

#[derive(Copy, Clone)]
pub struct FuncParam {
    pub attrs: AttrSet,

    pub ty: Type,
}

/// The body of a [`Func`] definition.
//
// FIXME(eddyb) `FuncDefBody`/`func_def_body` are too long, find shorter names.
#[derive(Clone)]
pub struct FuncDefBody {
    pub regions: EntityDefs<Region>,
    pub nodes: EntityDefs<Node>,
    pub data_insts: EntityDefs<DataInst>,

    /// The [`Region`] representing the whole body of the function.
    ///
    /// Function parameters are provided via `body.inputs`, i.e. they can be
    /// only accessed with `Value::RegionInputs { region: body, idx }`.
    ///
    /// When `unstructured_cfg` is `None`, this includes the structured return
    /// of the function, with `body.outputs` as the returned values.
    pub body: Region,

    /// The unstructured (part of the) control-flow graph of the function.
    ///
    /// Only present if structurization wasn't attempted, or if was only partial
    /// (leaving behind a mix of structured and unstructured control-flow).
    ///
    /// When present, it starts at `body` (more specifically, its exit),
    /// effectively replacing the structured return `body` otherwise implies,
    /// with `body` (or rather, its `children`) always being fully structured.
    //
    // FIXME(eddyb) replace this with a new `NodeKind` variant.
    pub unstructured_cfg: Option<cf::unstructured::ControlFlowGraph>,
}

/// Entity handle for a [`RegionDef`](crate::RegionDef)
/// (a control-flow region).
///
/// A [`Region`] ("control-flow region") is a linear chain of [`Node`]s,
/// describing a single-entry single-exit (SESE) control-flow "region" (subgraph)
/// in a function's control-flow graph (CFG).
///
/// # Control-flow
///
/// In SPIR-T, two forms of control-flow are used:
/// * "structured": [`Region`]s and [`Node`]s in a "mutual tree"
///   * i.e. each such [`Region`] can only appear in exactly one [`Node`],
///     and each [`Node`] can only appear in exactly one [`Region`]
///   * a region is either the function's body, or used as part of [`Node`]
///     (e.g. the "then" case of an `if`-`else`), itself part of a larger region
///   * when inside a region, reaching any other part of the function (or any
///     other function on call stack) requires leaving through the region's
///     single exit (also called "merge") point, i.e. its execution is either:
///     * "convergent": the region completes and continues into its parent
///       [`Node`], or function (the latter being a "structured return")
///     * "divergent": execution gets stuck in the region (an infinite loop),
///       or is aborted (e.g. `OpTerminateInvocation` from SPIR-V)
/// * "unstructured": [`Region`]s which connect to other [`Region`]s
///   using [`cfg::ControlInst`](crate::cfg::ControlInst)s (as described by a
///   [`cfg::ControlFlowGraph`](crate::cfg::ControlFlowGraph))
///
/// When a function's entire body can be described by a single [`Region`],
/// that function is said to have (entirely) "structured control-flow".
///
/// Mixing "structured" and "unstructured" control-flow is supported because:
/// * during structurization, it allows structured subgraphs to remain connected
///   by the same CFG edges that were connecting smaller [`Region`]s before
/// * structurization doesn't have to fail in the cases it doesn't fully support
///   yet, but can instead result in a "maximally structured" function
///
/// Other IRs may use different "structured control-flow" definitions, notably:
/// * SPIR-V uses a laxer definition, that corresponds more to the constraints
///   of the GLSL language, and is single-entry multiple-exit (SEME) with
///   "alternate exits" consisting of `break`s out of `switch`es and loops,
///   and `return`s (making it non-trivial to inline one function into another)
/// * RVSDG inspired SPIR-T's design, but its regions are (acyclic) graphs, it
///   makes no distinction between control-flow and "computational" nodes, and
///   its execution order is determined by value/state dependencies alone
///   (SPIR-T may get closer to it in the future, but the initial compromise
///   was chosen to limit the effort of lowering/lifting from/to SPIR-V)
///
/// # Data-flow interactions
///
/// SPIR-T [`Value`](crate::Value)s follow "single static assignment" (SSA), just like SPIR-V:
/// * inside a function, any new value is produced (or "defined") as an output
///   of [`DataInst`]/[`Node`], and "uses" of that value are [`Value`](crate::Value)s
///   variants which refer to the defining [`DataInst`]/[`Node`] directly
///   (guaranteeing the "single" and "static" of "SSA", by construction)
/// * the definition of a value must "dominate" all of its uses
///   (i.e. in all possible execution paths, the definition precedes all uses)
///
/// But unlike SPIR-V, SPIR-T's structured control-flow has implications for SSA:
/// * dominance is simpler, so values defined in a [`Region`](crate::Region) can be used:
///   * later in that region, including in the region's `outputs`
///     (which allows "exporting" values out to the rest of the function)
///   * outside that region, but *only* if the parent [`Node`](crate::Node)
///     is a `Loop` (that is, when the region is a loop's body)
///     * this is an "emergent" property, stemming from the region having to
///       execute (at least once) before the parent [`Node`](crate::Node)
///       can complete, but is not is not ideal and should eventually be replaced
///       with passing all such values through loop (body) `outputs`
/// * instead of φ ("phi") nodes, SPIR-T uses region `outputs` to merge values
///   coming from separate control-flow paths (i.e. the cases of a `Select`),
///   and region `inputs` for passing values back along loop backedges
///   (additionally, the body's `inputs` are used for function parameters)
///   * like the "block arguments" alternative to SSA phi nodes (which some
///     other SSA IRs use), this has the advantage of keeping the uses of the
///     "source" values in their respective paths (where they're dominated),
///     instead of in the merge (where phi nodes require special-casing, as
///     their "uses" of all the "source" values would normally be illegal)
///   * in unstructured control-flow, region `inputs` are additionally used for
///     representing phi nodes, as [`cfg::ControlInst`](crate::cfg::ControlInst)s
///     passing values to their target regions
///     * all value uses across unstructured control-flow edges (i.e. not in the
///       same region containing the value definition) *require* explicit passing,
///       as unstructured control-flow [`Region`](crate::Region)s
///       do *not* themselves get *any* implied dominance relations from the
///       shape of the control-flow graph (unlike most typical CFG+SSA IRs)
pub use context::Region;

/// Definition for a [`Region`]: a control-flow region.
#[derive(Clone, Default)]
pub struct RegionDef {
    /// Inputs to this [`Region`]:
    /// * accessed using [`Value::RegionInput`]
    /// * values provided by the parent:
    ///   * when this is the function body: the function's parameters
    pub inputs: SmallVec<[RegionInputDecl; 2]>,

    pub children: EntityList<Node>,

    /// Output values from this [`Region`], provided to the parent:
    /// * when this is the function body: these are the structured return values
    /// * when this is a `Select` case: these are the values for the parent
    ///   [`Node`]'s outputs (accessed using [`Value::NodeOutput`])
    /// * when this is a `Loop` body: these are the values to be used for the
    ///   next loop iteration's body `inputs`
    ///   * **not** accessible through [`Value::NodeOutput`] on the `Loop`,
    ///     as it's both confusing regarding [`Value::RegionInput`], and
    ///     also there's nothing stopping body-defined values from directly being
    ///     used outside the loop (once that changes, this aspect can be flipped)
    pub outputs: SmallVec<[Value; 2]>,
}

#[derive(Copy, Clone)]
pub struct RegionInputDecl {
    pub attrs: AttrSet,

    pub ty: Type,
}

/// Entity handle for a [`NodeDef`](crate::NodeDef)
/// (a control-flow operator or leaf).
///
/// See [`Region`] docs for more on control-flow in SPIR-T.
pub use context::Node;

/// Definition for a [`Node`]: a control-flow operator or leaf.
///
/// See [`Region`] docs for more on control-flow in SPIR-T.
#[derive(Clone)]
pub struct NodeDef {
    pub kind: NodeKind,

    /// Outputs from this [`Node`]:
    /// * accessed using [`Value::NodeOutput`]
    /// * values provided by `region.outputs`, where `region` is the executed
    ///   child [`Region`]:
    ///   * when this is a `Select`: the case that was chosen
    pub outputs: SmallVec<[NodeOutputDecl; 2]>,
}

#[derive(Copy, Clone)]
pub struct NodeOutputDecl {
    pub attrs: AttrSet,

    pub ty: Type,
}

#[derive(Clone)]
pub enum NodeKind {
    /// Linear chain of [`DataInst`]s, executing in sequence.
    ///
    /// This is only an optimization over keeping [`DataInst`]s in [`Region`]
    /// linear chains directly, or even merging [`DataInst`] with [`Node`].
    Block {
        // FIXME(eddyb) should empty blocks be allowed? should `DataInst`s be
        // linked directly into the `Region` `children` list?
        insts: EntityList<DataInst>,
    },

    /// Choose one [`Region`] out of `cases` to execute, based on a single
    /// value input (`scrutinee`) interpreted according to [`SelectionKind`].
    ///
    /// This corresponds to "gamma" (`γ`) nodes in (R)VSDG, though those are
    /// sometimes limited only to a two-way selection on a boolean condition.
    Select { kind: cf::SelectionKind, scrutinee: Value, cases: SmallVec<[Region; 2]> },

    /// Execute `body` repeatedly, until `repeat_condition` evaluates to `false`.
    ///
    /// To represent "loop state", `body` can take `inputs`, getting values from:
    /// * on the first iteration: `initial_inputs`
    /// * on later iterations: `body`'s own `outputs` (from the last iteration)
    ///
    /// As the condition is checked only *after* the body, this type of loop is
    /// sometimes described as "tail-controlled", and is also equivalent to the
    /// C-like `do { body; } while(repeat_condition)` construct.
    ///
    /// This corresponds to "theta" (`θ`) nodes in (R)VSDG.
    Loop {
        initial_inputs: SmallVec<[Value; 2]>,

        body: Region,

        // FIXME(eddyb) should this be kept in `body.outputs`? (that would not
        // have any ambiguity as to whether it can see `body`-computed values)
        repeat_condition: Value,
    },

    /// Leave the current invocation, similar to returning from every function
    /// call in the stack (up to and including the entry-point), but potentially
    /// indicating a fatal error as well.
    //
    // FIXME(eddyb) make this less shader-controlflow-centric.
    ExitInvocation {
        kind: cf::ExitInvocationKind,

        // FIXME(eddyb) centralize `Value` inputs across `Node`s,
        // and only use stricter types for building/traversing the IR.
        inputs: SmallVec<[Value; 2]>,
    },
}

/// Entity handle for a [`DataInstDef`](crate::DataInstDef) (a leaf instruction).
pub use context::DataInst;

/// Definition for a [`DataInst`]: a leaf (non-control-flow) instruction.
//
// FIXME(eddyb) `DataInstKind::FuncCall` should probably be a `NodeKind`,
// but also `DataInst` vs `Node` is a purely artificial distinction.
#[derive(Clone)]
pub struct DataInstDef {
    pub attrs: AttrSet,

    pub kind: DataInstKind,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    pub inputs: SmallVec<[Value; 2]>,

    pub output_type: Option<Type>,
}

#[derive(Clone, PartialEq, Eq, Hash, derive_more::From)]
pub enum DataInstKind {
    // FIXME(eddyb) try to split this into recursive and non-recursive calls,
    // to avoid needing special handling for recursion where it's impossible.
    FuncCall(Func),

    /// Memory-specific operations (see [`mem::MemOp`]).
    #[from]
    Mem(mem::MemOp),

    /// `QPtr`-specific operations (see [`qptr::QPtrOp`]).
    #[from]
    QPtr(qptr::QPtrOp),

    // FIXME(eddyb) should this have `#[from]`?
    SpvInst(spv::Inst),
    SpvExtInst {
        ext_set: InternedStr,
        inst: u32,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    Const(Const),

    /// One of the inputs to a [`Region`]:
    /// * declared by `region.inputs[input_idx]`
    /// * value provided by the parent of the `region`:
    ///   * when `region` is the function body: `input_idx`th function parameter
    RegionInput {
        region: Region,
        input_idx: u32,
    },

    /// One of the outputs produced by a [`Node`]:
    /// * declared by `node.outputs[output_idx]`
    /// * value provided by `region.outputs[output_idx]`, where `region` is the
    ///   executed child [`Region`] (of `node`):
    ///   * when `node` is a `Select`: the case that was chosen
    NodeOutput {
        node: Node,
        output_idx: u32,
    },

    /// The output value of a [`DataInst`].
    DataInstOutput(DataInst),
}
