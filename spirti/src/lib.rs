use arrayvec::ArrayVec;
use itertools::Itertools;
use lazy_static::lazy_static;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use spirt::cf::SelectionKind;
use spirt::func_at::FuncAt;
use spirt::{
    AddrSpace, Attr, AttrSet, Const, ConstKind, Context, DataInstKind, DeclDef,
    EntityOrientedDenseMap, ExportKey, Exportee, Func, GlobalVar, GlobalVarDecl, InternedStr,
    Module, Node, NodeDef, NodeKind, Region, RegionDef, Type, TypeDef, TypeKind, TypeOrConst,
    Value, Var, scalar, spv, vector,
};
use std::cell::Cell;
use std::fmt::Write as _;
use std::num::{NonZeroU8, NonZeroU32};
use std::ops::{Range, RangeFrom};
use std::path::PathBuf;
use std::rc::Rc;
use std::{fmt, mem, ops};

// HACK(eddyb) provide access to the `spirt` crate w/o needing 2 dependencies.
pub use spirt;

// HACK(eddyb) work around the lack of `FxIndex{Map,Set}` type aliases elsewhere.
#[doc(hidden)]
type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
#[doc(hidden)]
type FxIndexSet<V> = indexmap::IndexSet<V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

// HACK(eddyb) `spv::spec::Spec` with extra `WellKnown`s.
macro_rules! def_spv_spec_with_extra_well_known {
    ($($group:ident: $ty:ty = [$($entry:ident),+ $(,)?]),+ $(,)?) => {
        pub struct SpvSpecWithExtras {
            __base_spec: &'static spv::spec::Spec,

            pub well_known: SpvWellKnownWithExtras,
        }

        #[allow(non_snake_case)]
        pub struct SpvWellKnownWithExtras {
            __base_well_known: &'static spv::spec::WellKnown,

            $($(pub $entry: $ty,)+)+
        }

        impl std::ops::Deref for SpvSpecWithExtras {
            type Target = spv::spec::Spec;
            fn deref(&self) -> &Self::Target {
                self.__base_spec
            }
        }

        impl std::ops::Deref for SpvWellKnownWithExtras {
            type Target = spv::spec::WellKnown;
            fn deref(&self) -> &Self::Target {
                self.__base_well_known
            }
        }

        impl SpvSpecWithExtras {
            #[inline(always)]
            #[must_use]
            pub fn get() -> &'static SpvSpecWithExtras {
                lazy_static! {
                    static ref SPEC: SpvSpecWithExtras = {
                        #[allow(non_camel_case_types)]
                        struct PerWellKnownGroup<$($group),+> {
                            $($group: $group),+
                        }

                        let spv_spec = spv::spec::Spec::get();
                        let wk = &spv_spec.well_known;

                        let storage_classes = match &spv_spec.operand_kinds[wk.StorageClass] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };
                        let decorations = match &spv_spec.operand_kinds[wk.Decoration] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };

                        let execution_models = match &spv_spec.operand_kinds[spv_spec.operand_kinds.lookup("ExecutionModel").unwrap()] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };
                        let execution_modes = match &spv_spec.operand_kinds[spv_spec.operand_kinds.lookup("ExecutionMode").unwrap()] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };
                        let builtins = match &spv_spec.operand_kinds[spv_spec.operand_kinds.lookup("BuiltIn").unwrap()] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };

                        let glsl_std_450_ops = spv_spec
                            .get_ext_inst_set_by_lowercase_name("glsl.std.450")
                            .unwrap()
                            .instructions
                            .iter()
                            .map(|(&op, inst_desc)| (&inst_desc.name[..], op))
                            .collect::<FxHashMap<_, _>>();

                        let lookup_fns = PerWellKnownGroup {
                            opcode: |name| spv_spec.instructions.lookup(name).unwrap(),
                            operand_kind: |name| spv_spec.operand_kinds.lookup(name).unwrap(),
                            storage_class: |name| storage_classes.lookup(name).unwrap().into(),
                            decoration: |name| decorations.lookup(name).unwrap().into(),
                            execution_model: |name| execution_models.lookup(name).unwrap().into(),
                            execution_mode: |name| execution_modes.lookup(name).unwrap().into(),
                            builtin: |name| builtins.lookup(name).unwrap().into(),
                            glsl_std_450_op: |name| glsl_std_450_ops.get(name).copied().unwrap(),
                        };

                        SpvSpecWithExtras {
                            __base_spec: spv_spec,

                            well_known: SpvWellKnownWithExtras {
                                __base_well_known: &spv_spec.well_known,

                                $($($entry: (lookup_fns.$group)(stringify!($entry)),)+)+
                            },
                        }
                    };
                }
                &SPEC
            }
        }
    };
}
def_spv_spec_with_extra_well_known! {
    opcode: spv::spec::Opcode = [
        OpExecutionMode,

        OpSpecConstant,

        OpSelect,

        OpAtomicLoad,
        OpAtomicCompareExchange,
        OpAtomicIAdd,
    ],
    operand_kind: spv::spec::OperandKind = [
        ExecutionModel,
        ExecutionMode,
    ],
    storage_class: u32 = [
        PushConstant,
        StorageBuffer,
    ],
    decoration: u32 = [
        BuiltIn,
        DescriptorSet,
        Binding,
        Location,
    ],
    execution_model: u32 = [
        Vertex,
        Fragment,
        GLCompute,
    ],
    execution_mode: u32 = [
        // FIXME(eddyb) use this for compute launches!
        LocalSize,
    ],
    builtin: u32 = [
        FragCoord,
        GlobalInvocationId,
        LocalInvocationId,
        WorkgroupId,
    ],
    glsl_std_450_op: u32 = [
        FAbs,
        Round,
        Exp,
        Sqrt,
        Sin,
        Cos,

        FMin,
        FMax,
        Pow,

        Fma,
    ],
}

pub fn run_from_file(in_file_path: PathBuf, out_file_path: Option<PathBuf>) {
    let wk = &SpvSpecWithExtras::get().well_known;

    let debug = DebugOptions::from_env();

    fn eprint_duration<R>(f: impl FnOnce() -> R) -> R {
        let start = std::time::Instant::now();
        let r = f();
        eprint!("[{:8.3}ms] ", start.elapsed().as_secs_f64() * 1000.0);
        r
    }

    let mut module = eprint_duration(|| {
        Module::lower_from_spv_file(Rc::new(Context::new()), &in_file_path).unwrap()
    });
    eprintln!("Module::lower_from_spv_file({})", in_file_path.display());

    let cx = module.cx();

    eprint_duration(|| spirt::passes::legalize::structurize_func_cfgs(&mut module));
    eprintln!("legalize::structurize_func_cfgs");

    if debug.hermetically_sealed_regions {
        eprint_duration(|| {
            use spirt::visit;

            // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
            let visit::AllUses { funcs, .. } = visit::AllUses::from_module(&module);

            for &func in &funcs {
                if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
                    spirt::cf::hermetic::seal(&cx, func_def_body.at_mut_body());
                }
            }
        });
        eprintln!("cf::hermetic::seal");
    }

    let mut interpreter =
        Interpreter::new(&module, debug, &spirt::mem::LayoutConfig::VULKAN_SCALAR_LAYOUT_LE);

    let print_exec_model = |exec_model| {
        spv::print::operand_from_imms([spv::Imm::Short(wk.ExecutionModel, exec_model)])
            .concat_to_plain_text()
    };

    // FIXME(eddyb) allow selecting the entry-point by name.
    let (_entry_name, entry_exec_model, entry) = module
        .exports
        .iter()
        .filter_map(|(export_key, exportee)| match (export_key, exportee) {
            (
                ExportKey::SpvEntryPoint { imms, interface_global_vars: _ },
                &Exportee::Func(func),
            ) => {
                let exec_model = match imms[0] {
                    spv::Imm::Short(kind, exec_model) => {
                        assert_eq!(kind, wk.ExecutionModel);
                        exec_model
                    }
                    _ => unreachable!(),
                };
                let name = spv::extract_literal_string(&imms[1..]).unwrap();

                Some((name, exec_model, func))
            }
            _ => None,
        })
        .exactly_one()
        .unwrap_or_else(|e| {
            panic!(
                "not exactly one entry-point:{}",
                e.map(|(name, exec_model, _)| format!(
                    "\n  {name:?} => {}",
                    print_exec_model(exec_model)
                ))
                .collect::<Vec<_>>()
                .concat()
            )
        });

    // FIXME(eddyb) allow customizing this in-depth (via CLI).
    let launch = if entry_exec_model == wk.Vertex {
        Launch::VertIndices(0..3)
    } else if entry_exec_model == wk.Fragment {
        let h = 1080;
        let w = h * 16 / 9;
        Launch::FragRect { width: NonZeroU32::new(w).unwrap(), height: NonZeroU32::new(h).unwrap() }
    } else if entry_exec_model == wk.GLCompute {
        Launch::Compute {
            local: [1, 1, 1].map(|x| NonZeroU32::new(x).unwrap()),
            global: [1, 1, 1].map(|x| NonZeroU32::new(x).unwrap()),
        }
    } else {
        todo!("{}", print_exec_model(entry_exec_model));
    };

    // FIXME(eddyb) allow customizing this in-depth (via CLI).
    // FIXME(eddyb) some kind of interpolation could allow generating an animation,
    // by feeding time & mouse coords etc. for e.g. the mouse shader example.
    if let Launch::FragRect { width, height } = launch {
        let mut push_constant_bytes = vec![0; 13 * 4];

        push_constant_bytes[0..4].copy_from_slice(&u32::to_le_bytes(width.get()));
        push_constant_bytes[4..8].copy_from_slice(&u32::to_le_bytes(height.get()));

        interpreter.bind_memory(BindSlot::PushConstant, push_constant_bytes);
    }

    let start = std::time::Instant::now();
    interpreter.launch = Some(launch.clone());
    interpreter.eval_call(entry, [].into_iter().collect());
    let elapsed = start.elapsed();

    // FIXME(eddyb) the use of "instruction" below migth be a bit inaccurate.
    eprintln!(
        "{:8.3}ms for {} instructions ({} = {} invocations each)",
        elapsed.as_secs_f64() * 1e3,
        interpreter.step_counter,
        launch,
        launch.invocation_count(),
    );
    eprintln!(
        "={:7.3}ms per instruction",
        elapsed.as_secs_f64() * 1e3 / (interpreter.step_counter as f64),
    );
    eprintln!(
        "={:7.1}ns per invocation",
        elapsed.as_secs_f64() * 1e9 / (launch.invocation_count() as f64),
    );
    eprintln!(
        "={:7.1}ns per instruction per invocation",
        elapsed.as_secs_f64() * 1e9
            / (interpreter.step_counter as f64)
            / (launch.invocation_count() as f64),
    );

    if let Launch::FragRect { width, height } = launch
        && let Some(out_file_path) = out_file_path
    {
        for &gv in &interpreter.global_vars_keys {
            let gv_decl = &module.global_vars[gv];
            if gv_decl.addr_space == AddrSpace::SpvStorageClass(wk.Output)
                && let Some(&[spv::Imm::Short(_, 0)]) = interpreter.get_spv_attr(
                    gv_decl.attrs,
                    wk.OpDecorate,
                    wk.Decoration,
                    wk.Location,
                )
                && let Some(&gv_alloc) = interpreter.global_vars.get(gv)
                && let [DynVal { kind: DynLeaf::Vector(elems), .. }] =
                    &interpreter.mem_state.as_ref().unwrap()[gv_alloc].leaves[..]
                && let [
                    DynScalar(f32::TYPE, DynScalarData::B32(r)),
                    DynScalar(f32::TYPE, DynScalarData::B32(g)),
                    DynScalar(f32::TYPE, DynScalarData::B32(b)),
                    DynScalar(f32::TYPE, DynScalarData::B32(a)),
                ] = &elems[..]
            {
                // FIXME(eddyb) use `rayon` to build a `Vec<u8>` here.
                image::RgbaImage::from_fn(width.get(), height.get(), |x, y| {
                    let i = usize::try_from(
                        y.checked_mul(width.get()).unwrap().checked_add(x).unwrap(),
                    )
                    .unwrap();
                    let get = |x: &DynData<u32>| {
                        let x = f32::from_bits(match x {
                            &DynData::Uniform(x) => x,
                            DynData::PerInvocation(xs) => xs[i],
                        })
                        .clamp(0.0, 1.0);

                        // apply the srgb OETF (i.e. do "linear to sRGB")
                        fn srgb_oetf(x: f32) -> f32 {
                            if x <= 0.0031308 {
                                x * 12.92
                            } else {
                                1.055 * x.powf(1.0 / 2.4) - 0.055
                            }
                        }

                        // FIXME(eddyb) do not convert alpha.
                        (srgb_oetf(x).clamp(0.0, 1.0) * 255.0) as u8
                    };
                    image::Rgba([get(r), get(g), get(b), get(a)])
                })
                .save(out_file_path)
                .unwrap();
                return;
            }
        }
        unreachable!("`f32x4` output global var not found");
    }
}

#[derive(Clone)]
enum DynData<T> {
    Uniform(T),
    // FIXME(eddyb) make this hierarchical to allow "subgroup-uniform" etc.
    PerInvocation(Rc<Vec<T>>),
    // HACK(eddyb) consider some "linear interpolation" for that doesn't store
    // the values, or some "min"/"max" tracking, to potentially uniformize conditions.
}

// FIXME(eddyb) test perf vs rayon (also, consider batching that LLVM may vectorize).
impl<T: Copy + Send + Sync> DynData<T> {
    fn map<U: Send>(self, f: impl Fn(T) -> U + Send + Sync) -> DynData<U> {
        match self {
            DynData::Uniform(x) => DynData::Uniform(f(x)),
            // FIXME(eddyb) also consider making uniform if all the results are
            // all equal, but that may be too expensive to check? (maybe it'd
            // make more sense if it were hierarchical, since it would already
            // be trying to group invocations, and have more impact on average)
            //
            // FIXME(eddyb) take advantage of by-value `self` to re-use the `Rc`
            // allocation in-place, whenever possible (would require knowing that
            // `T == U`, or some kind of "scalars all made of `u32` chunks" repr).
            #[cfg(feature = "rayon")]
            DynData::PerInvocation(xs) => {
                DynData::PerInvocation(Rc::new(xs.par_iter().copied().map(f).collect()))
            }
            #[cfg(not(feature = "rayon"))]
            DynData::PerInvocation(xs) => {
                DynData::PerInvocation(Rc::new(xs.iter().copied().map(f).collect()))
            }
        }
    }
    fn map2<U: Copy + Send + Sync, V: Send>(
        self,
        other: DynData<U>,
        f: impl Fn(T, U) -> V + Send + Sync,
    ) -> DynData<V> {
        match (self, other) {
            (x, DynData::Uniform(y)) => x.map(move |x| f(x, y)),
            (DynData::Uniform(x), y) => y.map(move |y| f(x, y)),
            (DynData::PerInvocation(xs), DynData::PerInvocation(ys)) => {
                DynData::PerInvocation(Rc::new(
                    {
                        #[cfg(feature = "rayon")]
                        {
                            xs.par_iter().zip_eq(&ys[..])
                        }
                        #[cfg(not(feature = "rayon"))]
                        {
                            xs.iter().zip_eq(&ys[..])
                        }
                    }
                    .map(|(&x, &y)| f(x, y))
                    .collect(),
                ))
            }
        }
    }
    fn map3<U: Copy + Send + Sync, V: Copy + Send + Sync, W: Send>(
        self,
        other: DynData<U>,
        other2: DynData<V>,
        f: impl Fn(T, U, V) -> W + Send + Sync,
    ) -> DynData<W> {
        match (self, other, other2) {
            (x, y, DynData::Uniform(z)) => x.map2(y, move |x, y| f(x, y, z)),
            (x, DynData::Uniform(y), z) => x.map2(z, move |x, z| f(x, y, z)),
            (DynData::Uniform(x), y, z) => y.map2(z, move |y, z| f(x, y, z)),
            (
                DynData::PerInvocation(xs),
                DynData::PerInvocation(ys),
                DynData::PerInvocation(zs),
            ) => DynData::PerInvocation(Rc::new(
                {
                    #[cfg(feature = "rayon")]
                    {
                        xs.par_iter().zip_eq(&ys[..]).zip_eq(&zs[..])
                    }
                    #[cfg(not(feature = "rayon"))]
                    {
                        xs.iter().zip_eq(&ys[..]).zip_eq(&zs[..])
                    }
                }
                .map(|((&x, &y), &z)| f(x, y, z))
                .collect(),
            )),
        }
    }
}

// HACK(eddyb) traits to call the `map2`/`map3` methods, but as `map`, on tuples.
trait DynDataMap2<A: Copy + Send + Sync, B: Copy + Send + Sync> {
    fn map<R: Send>(self, f: impl Fn(A, B) -> R + Send + Sync) -> DynData<R>;
}
trait DynDataMap3<A: Copy + Send + Sync, B: Copy + Send + Sync, C: Copy + Send + Sync> {
    fn map<R: Send>(self, f: impl Fn(A, B, C) -> R + Send + Sync) -> DynData<R>;
}

impl<A: Copy + Send + Sync, B: Copy + Send + Sync> DynDataMap2<A, B> for (DynData<A>, DynData<B>) {
    fn map<R: Send>(self, f: impl Fn(A, B) -> R + Send + Sync) -> DynData<R> {
        let (xs, ys) = self;
        xs.map2(ys, f)
    }
}
impl<A: Copy + Send + Sync, B: Copy + Send + Sync, C: Copy + Send + Sync> DynDataMap3<A, B, C>
    for (DynData<A>, DynData<B>, DynData<C>)
{
    fn map<R: Send>(self, f: impl Fn(A, B, C) -> R + Send + Sync) -> DynData<R> {
        let (xs, ys, zs) = self;
        xs.map3(ys, zs, f)
    }
}

// HACK(eddyb) only a tuple struct for pattern-matching.
#[derive(Clone)]
struct DynScalar(scalar::Type, DynScalarData);

#[derive(Clone)]
enum DynScalarData {
    Undef,

    // FIXME(eddyb) this should really use a bitset!
    Bool(DynData<bool>),

    // FIXME(eddyb) unify these further?
    B8(DynData<u8>),
    B16(DynData<u16>),
    B32(DynData<u32>),
    B64(DynData<u64>),
}

// FIXME(eddyb) adjust this to be able to represent `bool` with bitsets.
#[allow(clippy::type_complexity)]
trait Scalar: Sized {
    const TYPE: scalar::Type;

    type Repr;

    // HACK(eddyb) these return pairs of functions, only to help with inference.
    // HACK(eddyb) `fn` pointers used only because `impl Fn` breaks inference.
    fn packer() -> (fn(DynData<Self::Repr>) -> DynScalar, fn(Self) -> Self::Repr);
    fn unpacker() -> (fn(DynScalar) -> Option<DynData<Self::Repr>>, fn(Self::Repr) -> Self);
}

macro_rules! impl_scalar {
    ($($ty:ty => ($scalar_ty_variant:ident$(($($scalar_ty_arg:tt)+))?, $variant:ident($repr:ty))
        where pack($pack_in:ident) = $pack_out:expr, unpack($unpack_in:ident) = $unpack_out:expr $(,)?
    ;)+) => {
        $(impl Scalar for $ty {
            const TYPE: scalar::Type = scalar::Type::$scalar_ty_variant$(($($scalar_ty_arg)+))?;

            type Repr = $repr;

            fn packer() -> (fn(DynData<Self::Repr>) -> DynScalar, fn(Self) -> Self::Repr) {
                (
                    |data| DynScalar(Self::TYPE, DynScalarData::$variant(data)),
                    |$pack_in| $pack_out,
                )
            }
            fn unpacker() -> (fn(DynScalar) -> Option<DynData<Self::Repr>>, fn(Self::Repr) -> Self)
            {
                (
                    |scalar| match scalar {
                        DynScalar(Self::TYPE, data) => {
                            match data {
                                DynScalarData::Undef => None,
                                DynScalarData::$variant(data) => Some(data),
                                _ => unreachable!("wrong `DynScalarData` for `{}`", stringify!($ty)),
                            }
                        }
                        _ => None,
                    },
                    |$unpack_in| $unpack_out,
                )
            }
        })+
    };
}

impl_scalar! {
    bool => (Bool, Bool(bool)) where pack(x) = x, unpack(x) = x;
    u8 =>  (UInt(scalar::IntWidth::I8), B8(u8)) where pack(x) = x, unpack(x) = x;
    i8 =>  (SInt(scalar::IntWidth::I8), B8(u8)) where pack(x) = x as u8, unpack(x) = x as i8;
    u16 => (UInt(scalar::IntWidth::I16), B16(u16)) where pack(x) = x, unpack(x) = x;
    i16 => (SInt(scalar::IntWidth::I16), B16(u16)) where pack(x) = x as u16, unpack(x) = x as i16;
    u32 => (UInt(scalar::IntWidth::I32), B32(u32)) where pack(x) = x, unpack(x) = x;
    i32 => (SInt(scalar::IntWidth::I32), B32(u32)) where pack(x) = x as u32, unpack(x) = x as i32;
    u64 => (UInt(scalar::IntWidth::I64), B64(u64)) where pack(x) = x, unpack(x) = x;
    i64 => (SInt(scalar::IntWidth::I64), B64(u64)) where pack(x) = x as u64, unpack(x) = x as i64;
    f32 => (Float(scalar::FloatWidth::F32), B32(u32)) where pack(x) = f32::to_bits(x), unpack(x) = f32::from_bits(x);
    f64 => (Float(scalar::FloatWidth::F64), B64(u64)) where pack(x) = f64::to_bits(x), unpack(x) = f64::from_bits(x);
}

macro_rules! invoke_op {
    // HACK(eddyb) notation shorthand, allowing e.g. `(+)` instead of `|x, y| x + y`.
    ((! $op:tt)($x:expr, $y:expr)) => { !($x $op $y) };
    (($op:tt)($x:expr, $y:expr)) => { $x $op $y };

    // HACK(eddyb) adapter shorthand (used with specialized `U`/`S` for integers).
    (($ctor:ident($($closure:tt)+))($($args:ident),+ $(,)?)) => {
        $ctor(|$($args),+| invoke_op!(($($closure)+)($($args),+)))($($args),+)
    };

    ((|$($params:ident $(: $param_ty:ty)?),+ $(,)?| $body:expr)($($args:expr),+ $(,)?)) => {
        match ($($args,)+) {
            ($($params,)+) => {
                $($(let $params: $param_ty = $params;)?)+
                $body
            }
        }
    };
}

// FIXME(eddyb) come up with a better syntax, and ideally unify the cases.
macro_rules! try_dispatch_scalar {
    ($inputs:ident.match $op:ident => ($T:ty) -> _: $($pat:pat => ($($closure:tt)+)),+ $(,)?) => {{
        let (x,) = $inputs.into_iter().collect_tuple().unwrap();
        match $op {
            $($pat => {
                // HACK(eddyb) closure somewhat used as `try` block.
                let (input_as_dyn_data, input_from_repr) = <$T as Scalar>::unpacker();
                let (output_from_dyn_data, output_to_repr) = <_ as Scalar>::packer();
                None.or_else(|| {
                    let x = input_as_dyn_data(x)?;
                    Some(output_from_dyn_data(x.map(|x| {
                        let x = input_from_repr(x);
                        output_to_repr(invoke_op!(($($closure)+)(x)))
                    })))
                })
            })+
        }
    }};
    // FIXME(eddyb) use for int binops too.
    ($inputs:ident.match $op:ident => ($T:ty, $U:ty) -> _: $($pat:pat => ($($closure:tt)+)),+ $(,)?) => {{
        let (x, y) = $inputs.into_iter().collect_tuple().unwrap();
        match $op {
            $($pat => {
                // HACK(eddyb) closure somewhat used as `try` block.
                let (input_as_dyn_data, input_from_repr) = <$T as Scalar>::unpacker();
                let (output_from_dyn_data, output_to_repr) = <_ as Scalar>::packer();
                None.or_else(|| {
                    let x = input_as_dyn_data(x)?;
                    let y = input_as_dyn_data(y)?;
                    Some(output_from_dyn_data((x, y).map(|x, y| {
                        let [x, y] = [x, y].map(input_from_repr);
                        output_to_repr(invoke_op!(($($closure)+)(x, y)))
                    })))
                })
            }),+
        }
    }};
    ($inputs:ident.match $op:ident => ($T:ty, $U:ty, $V:ty) -> _: $($pat:pat => ($($closure:tt)+)),+ $(,)?) => {{
        let (x, y, z) = $inputs.into_iter().collect_tuple().unwrap();
        match $op {
            $($pat => {
                // HACK(eddyb) closure somewhat used as `try` block.
                let (input_as_dyn_data, input_from_repr) = <$T as Scalar>::unpacker();
                let (output_from_dyn_data, output_to_repr) = <_ as Scalar>::packer();
                None.or_else(|| {
                    let x = input_as_dyn_data(x)?;
                    let y = input_as_dyn_data(y)?;
                    let z = input_as_dyn_data(z)?;
                    Some(output_from_dyn_data((x, y, z).map(|x, y, z| {
                        let [x, y, z] = [x, y, z].map(input_from_repr);
                        output_to_repr(invoke_op!(($($closure)+)(x, y, z)))
                    })))
                })
            }),+
        }
    }};
}

#[derive(Clone)]
enum DynPtrBase {
    // FIXME(eddyb) any reason to ever have a different variant?
    Alloc(AllocId),
}

#[derive(Clone, derive_more::From, derive_more::TryInto)]
enum DynLeaf {
    #[from]
    #[try_into]
    Scalar(DynScalar),

    Vector(ArrayVec<DynScalar, 4>),

    Ptr {
        base: DynPtrBase,

        leaf_range: RangeFrom<DynData<usize>>,

        // FIXME(eddyb) consider flattening even vectors in `spirti`.
        vector_component: Option<u8>,
    },

    SpvStringLiteralForExtInst(InternedStr),
    SpvVoidTypedValueFromExtInst,
}

impl DynLeaf {
    fn is_uniform(&self) -> bool {
        let scalar_is_uniform = |x: &DynScalar| match &x.1 {
            DynScalarData::Undef => true,
            DynScalarData::Bool(x) => matches!(x, DynData::Uniform(_)),
            DynScalarData::B8(x) => matches!(x, DynData::Uniform(_)),
            DynScalarData::B16(x) => matches!(x, DynData::Uniform(_)),
            DynScalarData::B32(x) => matches!(x, DynData::Uniform(_)),
            DynScalarData::B64(x) => matches!(x, DynData::Uniform(_)),
        };
        match self {
            DynLeaf::Scalar(x) => scalar_is_uniform(x),
            DynLeaf::Vector(xs) => xs.iter().all(scalar_is_uniform),
            DynLeaf::Ptr { base: _, leaf_range, vector_component: _ } => {
                matches!(leaf_range.start, DynData::Uniform(_))
            }
            DynLeaf::SpvStringLiteralForExtInst(_) | DynLeaf::SpvVoidTypedValueFromExtInst => true,
        }
    }

    // HACK(eddyb) this picks the value(s) corresponding to a specific invocation,
    // replacing all `PerInvocation(xs)` with `Uniform(xs[invocation_idx])`.
    fn extract_invocation_as_uniform(&self, invocation_idx: usize) -> Self {
        fn get<T: Copy>(x: &DynData<T>, ii: usize) -> DynData<T> {
            DynData::Uniform(match x {
                &DynData::Uniform(x) => x,
                DynData::PerInvocation(xs) => xs[ii],
            })
        }
        let ii = invocation_idx;
        let extract_scalar = |x: &DynScalar| {
            let data = match &x.1 {
                DynScalarData::Undef => DynScalarData::Undef,
                DynScalarData::Bool(x) => DynScalarData::Bool(get(x, ii)),
                DynScalarData::B8(x) => DynScalarData::B8(get(x, ii)),
                DynScalarData::B16(x) => DynScalarData::B16(get(x, ii)),
                DynScalarData::B32(x) => DynScalarData::B32(get(x, ii)),
                DynScalarData::B64(x) => DynScalarData::B64(get(x, ii)),
            };
            DynScalar(x.0, data)
        };
        match self {
            DynLeaf::Scalar(x) => DynLeaf::Scalar(extract_scalar(x)),
            DynLeaf::Vector(xs) => DynLeaf::Vector(xs.into_iter().map(extract_scalar).collect()),
            DynLeaf::Ptr { base, leaf_range, vector_component } => DynLeaf::Ptr {
                base: base.clone(),
                leaf_range: get(&leaf_range.start, ii)..,
                vector_component: *vector_component,
            },
            &DynLeaf::SpvStringLiteralForExtInst(s) => DynLeaf::SpvStringLiteralForExtInst(s),
            DynLeaf::SpvVoidTypedValueFromExtInst => DynLeaf::SpvVoidTypedValueFromExtInst,
        }
    }
}

// FIXME(eddyb) this is all over the place, reorganize!
fn vec_distribute<VDS: FromIterator<DynScalar>, VDV: IntoIterator<Item = DynLeaf>>(
    per_elem: impl Fn(VDS) -> DynScalar,
) -> impl Fn(VDV) -> DynLeaf {
    move |inputs| {
        let mut inputs: SmallVec<[_; 4]> = inputs
            .into_iter()
            .map(|v| match v {
                DynLeaf::Vector(xs) => xs.into_iter(),
                _ => unreachable!(),
            })
            .collect();
        let elem_count = inputs[0].len();
        inputs.iter().for_each(|xs| assert_eq!(xs.len(), elem_count));

        DynLeaf::Vector(
            (0..elem_count)
                .map(|_| per_elem(inputs.iter_mut().map(|xs| xs.next().unwrap()).collect()))
                .collect(),
        )
    }
}
fn scalar_or_vec_distribute<VDS: FromIterator<DynScalar>, VDV: IntoIterator<Item = DynLeaf>>(
    per_scalar: impl Fn(VDS) -> DynScalar,
) -> impl Fn(VDV) -> DynLeaf {
    move |inputs| {
        let mut inputs = inputs.into_iter().peekable();
        match inputs.peek().unwrap() {
            DynLeaf::Scalar(_) => {
                DynLeaf::Scalar(per_scalar(inputs.map(|x| x.try_into().unwrap()).collect()))
            }
            DynLeaf::Vector(_) => vec_distribute(&per_scalar)(inputs),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone)]
pub struct DynVal {
    ty: Type,
    // FIXME(eddyb) reconsider whether the field name and type name fit here.
    kind: DynLeaf,

    /// Assigning unique IDs in [`Interpreter::new_val`], to each new `DynVal`,
    /// allows telling apart otherwise-identical values (i.e. taint-tracking).
    uniq_id: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BindSlot {
    PushConstant,
    StorageBuffer { descriptor_set: u32, binding: u32 },
}

enum BindState {
    // FIXME(eddyb) support leaving parts of buffers `undef`?
    Unclaimed { init: Vec<u8> },
    ClaimedBy(GlobalVar),
}

#[derive(Clone)]
pub enum Launch {
    VertIndices(Range<u32>),
    FragRect { width: NonZeroU32, height: NonZeroU32 },
    Compute { local: [NonZeroU32; 3], global: [NonZeroU32; 3] },
}

impl fmt::Display for Launch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Launch::VertIndices(indices) => write!(f, "vertex({indices:?})"),
            Launch::FragRect { width, height } => write!(f, "fragment({width}px × {height}px)"),
            Launch::Compute { local: [lx, ly, lz], global: [gx, gy, gz] } => {
                write!(f, "compute({lx}×{ly}×{lz} × {gx}×{gy}×{gz})")
            }
        }
    }
}

impl Launch {
    fn invocation_count(&self) -> usize {
        match self {
            Launch::VertIndices(indices) => indices.len(),
            &Launch::FragRect { width, height } => [width, height]
                .into_iter()
                .map(|x| usize::try_from(x.get()).unwrap())
                .reduce(|a, b| a.checked_mul(b).unwrap())
                .unwrap(),
            &Launch::Compute { local, global } => (local.into_iter().chain(global))
                .map(|x| usize::try_from(x.get()).unwrap())
                .reduce(|a, b| a.checked_mul(b).unwrap())
                .unwrap(),
        }
    }
}

#[derive(Default)]
struct MemState {
    slots: Vec<AllocSlot>,
    next_uniq_seq_id: u64,
    last_dead_slot_idx: Option<usize>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct AllocId {
    /// Index in [`MemState::slots`].
    slot_idx: usize,

    /// `uniq_seq_id` must equal `slots[slot_idx].uniq_seq_id` for this to
    /// still refer to the a live allocation (otherwise, it's use-after-free).
    uniq_seq_id: u64,
}

enum AllocSlot {
    Dead { next_dead_slot_idx: Option<usize> },
    Alive { uniq_seq_id: u64, data: AllocData },
}

// FIXME(eddyb) use something more like `miri` (via `qptr`?).
// TODO(eddyb) distinguish between per-invocation (Private) memory and
// shared (Workgroup/Global/buffer) memory, with different semantics!
struct AllocData {
    ty: Type,
    leaves: Vec<DynVal>,

    // HACK(eddyb) this is a temporary workaround for the lack of proper memory
    // scoping (by forcing `eval_mem_store` to keep all `leaves` uniform).
    globally_shared: bool,
}

impl ops::Index<AllocId> for MemState {
    type Output = AllocData;
    fn index(&self, alloc: AllocId) -> &Self::Output {
        match &self.slots[alloc.slot_idx] {
            AllocSlot::Alive { uniq_seq_id, data } => {
                assert_eq!(alloc.uniq_seq_id, *uniq_seq_id);
                data
            }
            AllocSlot::Dead { .. } => {
                panic!("use-after-free on alloc in slot {}", alloc.slot_idx);
            }
        }
    }
}

impl ops::IndexMut<AllocId> for MemState {
    fn index_mut(&mut self, alloc: AllocId) -> &mut Self::Output {
        match &mut self.slots[alloc.slot_idx] {
            AllocSlot::Alive { uniq_seq_id, data } => {
                assert_eq!(alloc.uniq_seq_id, *uniq_seq_id);
                data
            }
            AllocSlot::Dead { .. } => {
                panic!("use-after-free on alloc in slot {}", alloc.slot_idx);
            }
        }
    }
}

impl MemState {
    fn alloc(&mut self, data: AllocData) -> AllocId {
        let uniq_seq_id = self.next_uniq_seq_id;
        self.next_uniq_seq_id = uniq_seq_id.checked_add(1).unwrap();

        let slot_idx = self.last_dead_slot_idx.take().unwrap_or_else(|| {
            let next_slot_idx = self.slots.len();
            self.slots.push(AllocSlot::Dead { next_dead_slot_idx: None });
            next_slot_idx
        });

        let AllocSlot::Dead { next_dead_slot_idx } =
            mem::replace(&mut self.slots[slot_idx], AllocSlot::Alive { uniq_seq_id, data })
        else {
            unreachable!()
        };
        self.last_dead_slot_idx = next_dead_slot_idx;

        AllocId { slot_idx, uniq_seq_id }
    }

    fn dealloc(&mut self, alloc: AllocId) {
        let AllocSlot::Alive { uniq_seq_id, data: _ } = mem::replace(
            &mut self.slots[alloc.slot_idx],
            AllocSlot::Dead { next_dead_slot_idx: self.last_dead_slot_idx.replace(alloc.slot_idx) },
        ) else {
            panic!("double-free on alloc in slot {}", alloc.slot_idx);
        };
        assert_eq!(alloc.uniq_seq_id, uniq_seq_id);
    }
}

#[derive(Default)]
struct CallFrame {
    var_values: EntityOrientedDenseMap<Var, DynVal>,

    dealloc_on_exit: Vec<AllocId>,
}

#[derive(Default)]
pub struct DebugOptions {
    trace_all: bool,
    trace_every_nth_step: Option<usize>,

    // FIXME(eddyb) make the time threshold configurable.
    trace_slow: bool,

    drop_vals_on_region_exit: bool,
    hermetically_sealed_regions: bool,

    // HACK(eddyb) this is only used by e.g. infinite loop detection.
    transiently_trace_all: bool,
}

impl DebugOptions {
    pub fn from_env() -> Self {
        let mut debug = Self::default();
        if let Some(opts) = std::env::var_os("SPIRTI_DEBUG")
            && !opts.is_empty()
        {
            for opt in opts.into_string().unwrap().split(',') {
                if let Some(n) = opt.strip_prefix("trace-every-nth=") {
                    debug.trace_every_nth_step = Some(n.parse().unwrap());
                } else {
                    match opt {
                        "trace-all" => debug.trace_all = true,
                        "trace-slow" => debug.trace_slow = true,
                        "drop-vals-on-region-exit" => debug.drop_vals_on_region_exit = true,
                        "hermetically-sealed-regions" => debug.hermetically_sealed_regions = true,
                        _ => panic!("unknown `SPIRTI_DEBUG` option `{opt}`"),
                    }
                }
            }
        }
        debug
    }
}

pub struct Interpreter<'a> {
    wk: &'static SpvWellKnownWithExtras,

    // FIXME(eddyb) group these into something like an `interned` field.
    glsl_std_450: InternedStr,
    non_semantic_debug_printf: InternedStr,

    layout_cache: spirt::mem::layout::LayoutCache<'a>,

    module: &'a Module,
    bindings: FxIndexMap<BindSlot, BindState>,

    // FIXME(eddyb) maybe move all of these fields into a "debug state"?
    debug: DebugOptions,
    next_val_uniq_id: Cell<u64>,
    step_counter: usize,
    // HACK(eddyb) this avoids clobbering the last line on stderr, by anything
    // other than the same instruction (if nothing else had been output yet).
    stderr_left_dirty_by_step: Option<usize>,

    // HACK(eddyb) this allows a way to iterate `global_vars`, without having
    // to make `global_vars` itself some kind of e.g. `FxIndexMap`.
    global_vars_keys: FxIndexSet<GlobalVar>,

    global_vars: EntityOrientedDenseMap<GlobalVar, AllocId>,
    call_stack: Vec<CallFrame>,

    // FIXME(eddyb) maybe merge this and `mem_state` into a `GlobalState`?
    // TODO(eddyb) make this private by having an `eval_entry_launch`.
    pub launch: Option<Launch>,

    // HACK(eddyb) the `Option` allows this to be "stolen" and put back,
    // mimicking linear state semantics in (R)VSDG.
    mem_state: Option<Box<MemState>>,

    // FIXME(eddyb) move this into a `GlobalState` alongside above fields.
    tangle: DynData<bool>,
}

impl<'a> Interpreter<'a> {
    pub fn new(
        module: &'a Module,
        debug: DebugOptions,
        layout_config: &'a spirt::mem::LayoutConfig,
    ) -> Self {
        let cx = module.cx_ref();

        Interpreter {
            wk: &SpvSpecWithExtras::get().well_known,
            glsl_std_450: cx.intern("GLSL.std.450"),
            non_semantic_debug_printf: cx.intern("NonSemantic.DebugPrintf"),

            layout_cache: spirt::mem::layout::LayoutCache::new(cx.clone(), layout_config),

            module,
            bindings: Default::default(),

            debug,
            next_val_uniq_id: Default::default(),
            step_counter: Default::default(),
            stderr_left_dirty_by_step: Default::default(),

            global_vars_keys: Default::default(),
            global_vars: Default::default(),
            call_stack: Default::default(),

            launch: None,
            mem_state: Some(Default::default()),
            tangle: DynData::Uniform(true),
        }
    }

    fn cx(&self) -> &'a Context {
        self.module.cx_ref()
    }

    pub fn bind_memory(&mut self, slot: BindSlot, bytes: Vec<u8>) {
        assert!(self.bindings.insert(slot, BindState::Unclaimed { init: bytes }).is_none());
    }

    pub fn read_bound_memory(
        &self,
        slot: BindSlot,
    ) -> Result<Vec<u8>, spirt::mem::const_data::ConstData<DynVal>> {
        let cx = self.cx();

        let data = match &self.bindings[&slot] {
            // FIXME(eddyb) maybe avoid clone (e.g. return `Cow<'_, [u8]>`).
            BindState::Unclaimed { init } => return Ok(init.clone()),
            &BindState::ClaimedBy(gv) => &self.mem_state.as_ref().unwrap()[self.global_vars[gv]],
        };

        let layout = match self.layout_cache.layout_of(data.ty) {
            Ok(spirt::mem::layout::TypeLayout::Handle(handle)) => match handle {
                spirt::mem::shapes::Handle::Opaque(_) => todo!(),
                spirt::mem::shapes::Handle::Buffer(_, buf_layout) => buf_layout,
            },
            Ok(spirt::mem::layout::TypeLayout::HandleArray(..)) => todo!(),
            Ok(spirt::mem::layout::TypeLayout::Concrete(_)) => todo!(),
            Err(_) => todo!(),
        };

        // HACK(eddyb) this slightly duplicates `qptr::lower`.
        // FIXME(eddyb) consider some kind of helper (maybe for `AllocData`?)
        // which relates leaves to their offsets, without touching layouts directly.
        let mut leaves = data.leaves.iter();
        let mut read_data =
            spirt::mem::const_data::ConstData::new(layout.mem_layout.fixed_base.size);
        let result = layout.deeply_flatten_if(
            0,
            // Whether `candidate_layout` is an aggregate (to recurse into).
            &|candidate_layout| {
                matches!(
                    &cx[candidate_layout.original_type].kind,
                    TypeKind::SpvInst { value_lowering: spv::ValueLowering::Disaggregate(_), .. }
                )
            },
            &mut |leaf_offset, leaf| {
                let leaf_offset = u32::try_from(leaf_offset).unwrap();

                let dyn_leaf = leaves.next().expect("alloc had fewer leaves than layout");

                assert!(dyn_leaf.ty == leaf.original_type);

                let leaf_size = NonZeroU32::new(leaf.mem_layout.fixed_base.size).unwrap();

                let mut total_written_range = leaf_offset..leaf_offset;

                // FIXME(eddyb) deduplicate with other instances of this.
                let dyn_scalar_as_const_kind = |x: &DynScalar| {
                    let x_bits = match *x {
                        DynScalar(_, DynScalarData::Bool(DynData::Uniform(x))) => x as u128,
                        DynScalar(_, DynScalarData::B8(DynData::Uniform(x))) => x.into(),
                        DynScalar(_, DynScalarData::B16(DynData::Uniform(x))) => x.into(),
                        DynScalar(_, DynScalarData::B32(DynData::Uniform(x))) => x.into(),
                        DynScalar(_, DynScalarData::B64(DynData::Uniform(x))) => x.into(),

                        DynScalar(_, DynScalarData::Undef) => return Some(ConstKind::Undef),
                        _ => return None,
                    };

                    Some(ConstKind::Scalar(scalar::Const::from_bits(x.0, x_bits)))
                };
                let dyn_leaf_as_const_kind = |x: &DynLeaf| match x {
                    DynLeaf::Scalar(x) => dyn_scalar_as_const_kind(x),
                    DynLeaf::Vector(elems) => elems
                        .into_iter()
                        .map(dyn_scalar_as_const_kind)
                        .collect::<Option<ArrayVec<_, 4>>>()
                        .and_then(|elems| {
                            if elems.iter().all(|x| matches!(x, ConstKind::Undef)) {
                                Some(ConstKind::Undef)
                            } else {
                                let elems: ArrayVec<_, 4> = elems
                                    .into_iter()
                                    .map(|x| match x {
                                        ConstKind::Scalar(x) => Some(x),
                                        _ => None,
                                    })
                                    .collect::<Option<_>>()?;
                                let ty = vector::Type {
                                    elem: elems
                                        .iter()
                                        .map(|x| x.ty())
                                        .dedup()
                                        .exactly_one()
                                        .ok()?,
                                    elem_count: NonZeroU8::new(elems.len().try_into().ok()?)
                                        .unwrap(),
                                };
                                Some(ConstKind::Vector(vector::Const::from_elems(ty, elems)))
                            }
                        }),
                    &DynLeaf::SpvStringLiteralForExtInst(s) => {
                        Some(ConstKind::SpvStringLiteralForExtInst(s))
                    }
                    DynLeaf::Ptr { .. } | DynLeaf::SpvVoidTypedValueFromExtInst => None,
                };

                // HACK(eddyb) helper shared by `Scalar` and `Vector`.
                let mut write_next_scalar = |leaf_scalar: scalar::Const| {
                    // FIXME(eddyb) try harder to avoid panicking due to out-of-bounds
                    // offsets caused by e.g. malformed layouts (and/or guarantee certain
                    // invariants for types that didn't error during layout computation).
                    let written_range = read_data
                        .write_scalar(
                            total_written_range.end,
                            leaf_scalar,
                            self.layout_cache.config,
                        )
                        .unwrap();
                    total_written_range.end = written_range.end;
                };

                match dyn_leaf_as_const_kind(&dyn_leaf.kind) {
                    Some(ConstKind::Undef) => {
                        return Ok(());
                    }

                    Some(ConstKind::Scalar(leaf_scalar)) => {
                        write_next_scalar(leaf_scalar);
                    }

                    Some(ConstKind::Vector(leaf_vector)) => {
                        for elem in leaf_vector.elems() {
                            write_next_scalar(elem);
                        }
                    }

                    // FIXME(eddyb) try harder to avoid panicking due to out-of-bounds
                    // offsets caused by e.g. malformed layouts (and/or guarantee certain
                    // invariants for types that didn't error during layout computation).
                    _ => {
                        read_data.write_symbolic(leaf_offset, leaf_size, dyn_leaf.clone()).unwrap();
                        total_written_range.end += leaf_size.get();
                    }
                }

                assert_eq!(total_written_range, leaf_offset..(leaf_offset + leaf_size.get()));

                Ok(())
            },
        );
        result.ok().unwrap();

        assert!(leaves.next().is_none(), "alloc had more leaves than layout");

        let data_bytes = match read_data.read(0..read_data.size()).exactly_one() {
            // FIXME(eddyb) avoid the reallocation here, somehow?
            Ok(spirt::mem::const_data::Part::Bytes(bytes)) => Some(bytes.to_vec()),
            _ => None,
        };

        data_bytes.ok_or(read_data)
    }

    pub fn eval_call(&mut self, f: Func, args: SmallVec<[DynVal; 4]>) -> SmallVec<[DynVal; 4]> {
        let func_def_body = match &self.module.funcs[f].def {
            DeclDef::Imported(import) => unreachable!(
                "calling import {:?}",
                match import {
                    &spirt::Import::LinkName(name) => &self.cx()[name],
                }
            ),
            DeclDef::Present(def) => def,
        };

        self.call_stack.push(CallFrame::default());
        let ret_vals = self.eval_region(func_def_body.at_body(), args);
        let frame = self.call_stack.pop().unwrap();
        let mem_state = self.mem_state.as_mut().unwrap();
        for alloc in frame.dealloc_on_exit {
            mem_state.dealloc(alloc);
        }
        ret_vals
    }
    fn eval_region(
        &mut self,
        func_at_region: FuncAt<'_, Region>,
        inputs: SmallVec<[DynVal; 4]>,
    ) -> SmallVec<[DynVal; 4]> {
        let (outputs, ()) = self.eval_region_with(func_at_region, inputs, |_| ());
        outputs
    }
    fn eval_region_with<R>(
        &mut self,
        func_at_region: FuncAt<'_, Region>,
        inputs: SmallVec<[DynVal; 4]>,

        // HACK(eddyb) only needed for the `Loop` exit condition.
        before_exit: impl FnOnce(&mut Self) -> R,
    ) -> (SmallVec<[DynVal; 4]>, R) {
        let RegionDef { inputs: input_vars, children: _, outputs } = func_at_region.def();

        assert_eq!(input_vars.len(), inputs.len());
        for (&input_var, input) in input_vars.iter().zip_eq(inputs) {
            self.call_stack.last_mut().unwrap().var_values.insert(input_var, input);
        }

        for func_at_node in func_at_region.at_children() {
            self.eval_node(func_at_node);
        }

        let outputs = outputs.iter().map(|&v| self.eval_value(v)).collect();
        let extra = before_exit(self);

        if self.debug.drop_vals_on_region_exit {
            let vars_defined_in_region = input_vars
                .iter()
                .chain(
                    func_at_region
                        .at_children()
                        .into_iter()
                        .flat_map(|fan| fan.def().outputs.iter()),
                )
                .copied();

            let var_values = &mut self.call_stack.last_mut().unwrap().var_values;
            for var in vars_defined_in_region {
                // FIXME(eddyb) should this use a tombstone instead?
                var_values.remove(var).unwrap();
            }
        }

        (outputs, extra)
    }
    fn eval_node(&mut self, func_at_node: FuncAt<'_, Node>) {
        let cx = self.cx();

        let step = self.step_counter;
        self.step_counter += 1;

        let NodeDef { attrs, kind, inputs, child_regions: _, outputs: output_vars } =
            func_at_node.def();
        let func = func_at_node.at(());

        let inputs: SmallVec<[_; 4]> = inputs.iter().map(|&v| self.eval_value(v)).collect();

        // TODO(eddyb) the details used here are relatively shallow, consider
        // either parameterizing `DynLeaf` by what's currently `DynScalar`,
        // or coming up with some other representation, that avoids cloning
        // allocations (and/or keeping `Rc` refcounts), for e.g. `dbg_trace_header`
        // to be always retroactively computable (even in the face of regalloc).
        let dbg_val = |v: &DynVal| {
            let details = (|| match &v.kind {
                DynLeaf::Scalar(DynScalar(_, DynScalarData::Undef)) => Some("undef".to_string()),
                DynLeaf::Scalar(x) => {
                    let x_bits = match *x {
                        DynScalar(_, DynScalarData::Bool(DynData::Uniform(x))) => x as u128,
                        DynScalar(_, DynScalarData::B8(DynData::Uniform(x))) => x.into(),
                        DynScalar(_, DynScalarData::B16(DynData::Uniform(x))) => x.into(),
                        DynScalar(_, DynScalarData::B32(DynData::Uniform(x))) => x.into(),
                        DynScalar(_, DynScalarData::B64(DynData::Uniform(x))) => x.into(),
                        _ => return None,
                    };

                    let ct: Const = cx.intern(scalar::Const::from_bits(x.0, x_bits));
                    Some(spirt::print::Plan::for_root(cx, &ct).pretty_print().to_string())
                }
                DynLeaf::Ptr {
                    base: DynPtrBase::Alloc(alloc_id),
                    leaf_range,
                    vector_component,
                } => {
                    let leaf_range = match leaf_range {
                        RangeFrom { start: DynData::Uniform(start) } => start..,
                        _ => return None,
                    };
                    let mut s = format!("&alloc{}[{leaf_range:?}]", alloc_id.uniq_seq_id);
                    if let Some(i) = vector_component {
                        write!(s, ".{i}").unwrap();
                    }
                    Some(s)
                }
                DynLeaf::Vector(_)
                | DynLeaf::SpvStringLiteralForExtInst(_)
                | DynLeaf::SpvVoidTypedValueFromExtInst => None,
            })();
            let mut s = format!(
                "v{}",
                // FIXME(eddyb) this should be a reusable abstraction.
                v.uniq_id
                    .to_string()
                    .chars()
                    .map(|d| char::from_u32(('₀' as u32) + d.to_digit(10).unwrap()).unwrap())
                    .collect::<String>()
            );
            // FIXME(eddyb) figure out the best way to print this.
            if let Some(details) = details {
                s += "→";
                s += &details;
            }
            s
        };

        let always_dbg_trace = self.debug.trace_all
            || self.debug.transiently_trace_all
            || self.debug.trace_every_nth_step.is_some_and(|n| self.step_counter.is_multiple_of(n));
        let may_dbg_trace = always_dbg_trace || self.debug.trace_slow;
        let dbg_trace_header = may_dbg_trace.then(|| {
            let name = match kind {
                NodeKind::Select(SelectionKind::BoolCond) => "if",
                NodeKind::Select(SelectionKind::Switch { .. }) => "switch",
                NodeKind::Loop { .. } => "loop",

                DataInstKind::Scalar(op) => op.name(),
                DataInstKind::Vector(op) => match op {
                    vector::Op::Distribute(op) => op.name(),
                    vector::Op::Reduce(op) => op.name(),
                    vector::Op::Whole(op) => op.name(),
                },
                DataInstKind::FuncCall(_) => "call",
                DataInstKind::Mem(_) | DataInstKind::QPtr(_) | DataInstKind::ThunkBind(_) => {
                    unreachable!()
                }
                NodeKind::ExitInvocation(spirt::cf::ExitInvocationKind::SpvInst(spv_inst))
                | DataInstKind::SpvInst(spv_inst, _) => spv_inst.opcode.name(),
                &DataInstKind::SpvExtInst { ext_set, inst, .. } => {
                    &spv::spec::Spec::get()
                        .get_ext_inst_set_by_lowercase_name(&cx[ext_set].to_ascii_lowercase())
                        .unwrap()
                        .instructions[&inst]
                        .name
                }
            };

            let mut s = format!("{name}(");
            for (i, input) in inputs.iter().enumerate() {
                if i > 0 {
                    s += ", ";
                }
                s += &dbg_val(input);
            }
            s += ")";

            // TODO(eddyb) provide similar detail in other cases (e.g. SPIR-V imms).
            // FIXME(eddyb) is it worth including `=> ⋯` for every case?
            if let NodeKind::Select(SelectionKind::Switch { case_consts }) = kind {
                s += " {";
                for &ct in case_consts {
                    let ct: Const = self.cx().intern(ct);
                    s += " ";
                    s += &spirt::print::Plan::for_root(self.cx(), &ct).pretty_print().to_string();
                    s += " => ⋯,";
                }
                s += " _ => ⋯ }";
            }

            s
        });
        if always_dbg_trace {
            if self.stderr_left_dirty_by_step.take().is_some() {
                eprintln!();
            }

            let attrs = spirt::print::Plan::for_root(cx, attrs).pretty_print().to_string();
            if !attrs.is_empty() {
                eprintln!("{attrs}");
            }
            eprint!("╭→#{step} {}", dbg_trace_header.as_ref().unwrap());

            // FIXME(eddyb) is it useful to omit this on completion?
            if !output_vars.is_empty() {
                eprint!(": ");
                if output_vars.len() >= 2 {
                    eprint!("(");
                }
                for (i, &output_var) in output_vars.iter().enumerate() {
                    if i > 0 {
                        eprint!(", ");
                    }
                    let ty = func.vars[output_var].ty;
                    // HACK(eddyb) hope the type is self-contained.
                    let ty = spirt::print::Plan::for_root(cx, &ty).pretty_print().to_string();
                    let ty = if !ty.contains('\n') && ty.len() < 8 { &ty } else { "⋯" };
                    eprint!("{ty}");
                }
                if output_vars.len() >= 2 {
                    eprint!(")");
                }
            }

            self.stderr_left_dirty_by_step = Some(step);
        }

        // FIXME(eddyb) eagerly extract the `MemState` based on `kind`,
        // and use it to treat side-effectful ops as never uniform.
        let dbg_trace_has_non_uniform_inputs = may_dbg_trace
            && inputs.iter().any(|input| {
                let non_uniform_scalar = |scalar: &DynScalar| match scalar.1 {
                    DynScalarData::Undef
                    | DynScalarData::Bool(DynData::Uniform(_))
                    | DynScalarData::B8(DynData::Uniform(_))
                    | DynScalarData::B16(DynData::Uniform(_))
                    | DynScalarData::B32(DynData::Uniform(_))
                    | DynScalarData::B64(DynData::Uniform(_)) => false,

                    DynScalarData::Bool(DynData::PerInvocation(_))
                    | DynScalarData::B8(DynData::PerInvocation(_))
                    | DynScalarData::B16(DynData::PerInvocation(_))
                    | DynScalarData::B32(DynData::PerInvocation(_))
                    | DynScalarData::B64(DynData::PerInvocation(_)) => true,
                };
                match &input.kind {
                    DynLeaf::Scalar(x) => non_uniform_scalar(x),
                    DynLeaf::Vector(xs) => xs.iter().any(non_uniform_scalar),
                    DynLeaf::Ptr { .. }
                    | DynLeaf::SpvStringLiteralForExtInst(_)
                    | DynLeaf::SpvVoidTypedValueFromExtInst => false,
                }
            });

        let start = std::time::Instant::now();

        let outputs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.eval_node_inner(func_at_node, inputs)
        }))
        .unwrap_or_else(|payload| {
            // NOTE(eddyb) the default panic handler will have already printed
            // the panic message, so this will end up being shown *after*.
            // FIXME(eddyb) should this be shown for every enclosing node?
            eprintln!("⮬ [spirti] PANIC during step #{step}");

            // FIXME(eddyb) consider using tombstones for output variables,
            // and continuing execution, skipping over any nodes that touch
            // tombstones in any way (if panic tombstones are *not* included
            // in `DynLeaf`, but e.g. `var_values` holding `Result` instead,
            // there would be no need for special handling anywhere).
            std::panic::resume_unwind(payload)
        });

        let elapsed = start.elapsed();

        if may_dbg_trace {
            let elapsed_us = elapsed.as_secs_f64() * 1e6;

            let assumed_instances = if dbg_trace_has_non_uniform_inputs {
                self.launch.as_ref().unwrap().invocation_count()
            } else {
                1
            };
            let elapsed_ns_per_instance = elapsed_us * 1e3 / (assumed_instances as f64);

            // HACK(eddyb) arbitrary metric, to keep `self.debug.trace_slow`
            // from always tracing everything uniform, just because there's
            // not enough amortization of interpretation happening there.
            let estimate_elapsed_ns_per_1k_invocations = || {
                if dbg_trace_has_non_uniform_inputs {
                    elapsed_ns_per_instance * 1e3
                } else {
                    elapsed_ns_per_instance
                }
            };

            let stderr_left_dirty_by_step = self.stderr_left_dirty_by_step.take();

            if always_dbg_trace
                || self.debug.trace_slow
                    && self.step_counter == step + 1
                    && estimate_elapsed_ns_per_1k_invocations() >= 4e3
            {
                let dbg_trace_header = dbg_trace_header.as_ref().unwrap();

                // FIXME(eddyb) this condition is overly convoluted, and seems
                // to also suggest that `self.stderr_left_dirty_by_step`
                // being `Option` is unnecessary, as when it's `Some(past_step)`,
                // the `past_step` value will always be `self.step_counter - 1`.
                if stderr_left_dirty_by_step == Some(step)
                    || !always_dbg_trace && self.step_counter == step + 1
                {
                    let elapsed = if assumed_instances == 1 || elapsed_ns_per_instance < 0.1 {
                        format!("{elapsed_us:.3}µs")
                    } else {
                        format!("{elapsed_ns_per_instance:.1}ns × {assumed_instances}")
                    };
                    // HACK(eddyb) this elides the step # (does it matter?).
                    eprint!("\r┊{elapsed:>20} │ {dbg_trace_header}");
                } else {
                    // HACK(eddyb) this elides the elapsed duration, but that's
                    // less relevant (maybe include it somewhere else?).
                    eprint!("╰←#{step}");

                    // TODO(eddyb) this combination is never possible, now.
                    if !always_dbg_trace {
                        // HACK(eddyb) this elides the `dbg_trace_header`,
                        // if it was already printed, as it may be confusing
                        // (and the step # should be enough to find it).
                        eprint!(" {dbg_trace_header}");
                    }
                }

                if !outputs.is_empty() {
                    eprint!(" => ");
                    if outputs.len() >= 2 {
                        eprint!("(");
                    }
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            eprint!(", ");
                        }
                        eprint!("{}", dbg_val(output));
                    }
                    if outputs.len() >= 2 {
                        eprint!(")");
                    }
                }
                eprintln!();
            } else {
                assert_eq!(stderr_left_dirty_by_step, None);
            }
        }

        assert_eq!(output_vars.len(), outputs.len());
        for (&output_var, output) in output_vars.iter().zip_eq(outputs) {
            {
                let found = output.ty;
                let expected = func.vars[output_var].ty;

                let print = |ty| spirt::print::Plan::for_root(self.cx(), &ty).pretty_print();
                assert!(
                    found == expected
                        || matches!(
                            [found, expected].map(|x| self.as_spv_ptr_type(x)),
                            // HACK(eddyb) only compare `AddrSpace` for pointers.
                            [Some((found_as, _)), Some((expected_as, _))] if found_as == expected_as
                        ),
                    "   found `{}`\n\
                     expected `{}`",
                    print(found),
                    print(expected),
                );
            }

            self.call_stack.last_mut().unwrap().var_values.insert(output_var, output);
        }
    }

    // FIXME(eddyb) move around e.g. tracing logic, to take advantage of this split.
    fn eval_node_inner(
        &mut self,
        func_at_node: FuncAt<'_, Node>,
        inputs: SmallVec<[DynVal; 4]>,
    ) -> SmallVec<[DynVal; 4]> {
        let cx = self.cx();

        let NodeDef { attrs: _, inputs: _, kind, child_regions, outputs: output_vars } =
            func_at_node.def();
        let output_types = output_vars.iter().map(|&var| func_at_node.vars[var].ty);

        // TODO(eddyb) share this with `dbg_trace_header` in `eval_node`.
        let describe_spv_inst = || match kind {
            DataInstKind::SpvInst(spv_inst, _) => {
                format!(
                    "SPIR-V instruction {}({})",
                    spv_inst.opcode.name(),
                    spv::print::inst_operands(
                        spv_inst.opcode,
                        spv_inst.imms.iter().copied(),
                        func_at_node
                            .def()
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(i, _)| format!("<input #{i}>"))
                    )
                    .map(|operand| operand.concat_to_plain_text())
                    .collect::<Vec<_>>()
                    .join(", ")
                )
            }
            &DataInstKind::SpvExtInst { ext_set, inst, .. } => {
                let ext_set = &cx[ext_set];
                format!(
                    "SPIR-V extended ({ext_set}) instruction #{inst} ({:?})",
                    spv::spec::Spec::get()
                        .get_ext_inst_set_by_lowercase_name(&ext_set.to_ascii_lowercase())
                        .and_then(|ext_inst_set_desc| Some(
                            &ext_inst_set_desc.instructions.get(&inst)?.name
                        )),
                )
            }
            _ => unreachable!(),
        };

        match kind {
            NodeKind::Select(kind) => {
                let selector = inputs[0].clone();
                let selector_scalar = match selector.kind {
                    DynLeaf::Scalar(selector) => selector,
                    _ => unreachable!(),
                };

                let case_consts = match kind {
                    SelectionKind::BoolCond => &[scalar::Const::TRUE][..],
                    SelectionKind::Switch { case_consts } => case_consts,
                };

                let case_inputs = if self.debug.hermetically_sealed_regions {
                    inputs
                } else {
                    [].into_iter().collect()
                };

                let mut already_handled =
                    DynScalar(scalar::Type::Bool, DynScalarData::Bool(DynData::Uniform(false)));
                let mut partial_outputs = None;

                for (maybe_case_const, &case) in
                    case_consts.iter().map(Some).chain([None]).zip_eq(child_regions)
                {
                    // FIXME(eddyb) optimize the chains of operations used here.
                    let case_taken = match maybe_case_const {
                        Some(&ct) => self.eval_scalar_op(
                            if selector_scalar.0 == scalar::Type::Bool {
                                scalar::BoolBinOp::Eq.into()
                            } else {
                                scalar::IntBinOp::Eq.into()
                            },
                            [selector_scalar.clone(), self.eval_scalar_const(ct)]
                                .into_iter()
                                .collect(),
                            scalar::Type::Bool,
                        ),
                        None => self.eval_scalar_op(
                            scalar::BoolUnOp::Not.into(),
                            [already_handled.clone()].into_iter().collect(),
                            scalar::Type::Bool,
                        ),
                    };

                    already_handled = self.eval_scalar_op(
                        scalar::BoolBinOp::Or.into(),
                        [already_handled.clone(), case_taken.clone()].into_iter().collect(),
                        scalar::Type::Bool,
                    );

                    let case_taken_data = match &case_taken {
                        DynScalar(_, DynScalarData::Undef) => {
                            panic!("undefined behavior: branching on `undef` condition");
                        }
                        DynScalar(_, DynScalarData::Bool(x)) => x.clone(),
                        _ => unreachable!(),
                    };

                    if let DynData::Uniform(false) = case_taken_data {
                        continue;
                    }

                    // FIXME(eddyb) use `eval_scalar_op` for this?
                    let outer_tangle = self.tangle.clone();
                    self.tangle = (self.tangle.clone(), case_taken_data.clone()).map(|x, y| x & y);

                    let case_outputs = self.eval_region(func_at_node.at(case), case_inputs.clone());

                    self.tangle = outer_tangle;

                    if let DynData::Uniform(true) = case_taken_data {
                        // FIXME(eddyb) this should be a noop at this point.
                        return case_outputs;
                    }

                    partial_outputs = Some(match partial_outputs {
                        Some(prev_outputs) => case_outputs
                            .into_iter()
                            .zip_eq(prev_outputs)
                            .map(|(case_output, prev_output)| {
                                let ty = case_output.ty;
                                self.eval_select(
                                    DynLeaf::Scalar(case_taken.clone()),
                                    case_output,
                                    prev_output,
                                    ty,
                                )
                            })
                            .collect(),
                        None => case_outputs,
                    });
                }

                partial_outputs.unwrap()
            }
            NodeKind::Loop { repeat_condition } => {
                let mut loop_state = inputs;

                let mut iter_count = 0u64;
                loop {
                    iter_count += 1;
                    let outer_transiently_trace_all = self.debug.transiently_trace_all;
                    if iter_count.is_multiple_of(10_000) {
                        eprintln!(
                            "WARN: slow loop! ({iter_count} so far) - forcing tracing for one iteration:"
                        );
                        self.debug.transiently_trace_all = true;
                    }

                    let rep_cond;
                    (loop_state, rep_cond) = self.eval_region_with(
                        func_at_node.at(child_regions[0]),
                        loop_state,
                        |this| this.eval_value(*repeat_condition),
                    );

                    self.debug.transiently_trace_all = outer_transiently_trace_all;

                    self.step_counter += 1;

                    let rep_cond = match rep_cond.kind {
                        DynLeaf::Scalar(DynScalar(bool::TYPE, DynScalarData::Bool(cond))) => cond,
                        _ => unreachable!(),
                    };

                    // HACK(eddyb) try really hard to avoid per-invocation decisions.
                    let rep_cond = match rep_cond {
                        DynData::Uniform(cond) => cond,
                        DynData::PerInvocation(conds)
                            if {
                                let first = conds[0];
                                #[cfg(feature = "rayon")]
                                {
                                    conds[1..].par_iter().all(|&c| c == first)
                                }
                                #[cfg(not(feature = "rayon"))]
                                {
                                    conds[1..].iter().all(|&c| c == first)
                                }
                            } =>
                        {
                            conds[0]
                        }
                        DynData::PerInvocation(_) => todo!("per-invocation `loop` repeat decision"),
                    };
                    if !rep_cond {
                        break;
                    }
                }
                loop_state
            }
            NodeKind::ExitInvocation(_) => todo!(),

            // TODO(eddyb) move these into `eval_scalar_op`.
            DataInstKind::Scalar(scalar::Op::IntBinary(scalar::IntBinOp::CarryingAdd)) => {
                let (x, y) = inputs.into_iter().map(|x| x.kind).collect_tuple().unwrap();
                match (x, y) {
                    (
                        DynLeaf::Scalar(DynScalar(u32::TYPE, DynScalarData::B32(x))),
                        DynLeaf::Scalar(DynScalar(u32::TYPE, DynScalarData::B32(y))),
                    ) => [
                        (x.clone(), y.clone()).map(|x, y| x.wrapping_add(y)),
                        (x, y).map(|x, y| x.overflowing_add(y).1 as u32),
                    ]
                    .into_iter()
                    .map(|x| {
                        self.new_val(
                            // FIXME(eddyb) reuse from input types?
                            cx.intern(u32::TYPE),
                            DynLeaf::Scalar(DynScalar(u32::TYPE, DynScalarData::B32(x))),
                        )
                    })
                    .collect(),
                    _ => unreachable!(),
                }
            }
            DataInstKind::Scalar(scalar::Op::IntBinary(scalar::IntBinOp::WideningMulU)) => {
                let (x, y) = inputs.into_iter().map(|x| x.kind).collect_tuple().unwrap();
                match (x, y) {
                    (
                        DynLeaf::Scalar(DynScalar(u32::TYPE, DynScalarData::B32(x))),
                        DynLeaf::Scalar(DynScalar(u32::TYPE, DynScalarData::B32(y))),
                    ) => [
                        (x.clone(), y.clone()).map(|x, y| x.wrapping_mul(y)),
                        (x, y).map(|x, y| ((x as u64).wrapping_mul(y as u64) >> 32) as u32),
                    ]
                    .into_iter()
                    .map(|x| {
                        self.new_val(
                            // FIXME(eddyb) reuse from input types?
                            cx.intern(u32::TYPE),
                            DynLeaf::Scalar(DynScalar(u32::TYPE, DynScalarData::B32(x))),
                        )
                    })
                    .collect(),
                    _ => unreachable!(),
                }
            }
            &DataInstKind::Scalar(op) => {
                let output_type = output_types.exactly_one().ok().unwrap();
                [self.new_val(
                    output_type,
                    self.eval_scalar_op(
                        op,
                        inputs.into_iter().map(|x| x.kind.try_into().unwrap()).collect(),
                        output_type.as_scalar(self.cx()).unwrap(),
                    )
                    .into(),
                )]
                .into_iter()
                .collect()
            }
            &DataInstKind::Vector(op) => {
                let output_type = output_types.exactly_one().ok().unwrap();
                let inputs: SmallVec<[_; 4]> = inputs.into_iter().map(|x| x.kind).collect();
                let output = match op {
                    vector::Op::Distribute(op) => {
                        let output_type = output_type.as_vector(self.cx()).unwrap();
                        vec_distribute(|inputs| self.eval_scalar_op(op, inputs, output_type.elem))(
                            inputs,
                        )
                    }
                    vector::Op::Reduce(op) => match op {
                        vector::ReduceOp::Dot => todo!(),
                        vector::ReduceOp::Any => todo!(),
                        vector::ReduceOp::All => todo!(),
                    },
                    vector::Op::Whole(op) => match op {
                        vector::WholeOp::New => DynLeaf::Vector(
                            inputs.into_iter().map(|x| x.try_into().unwrap()).collect(),
                        ),
                        vector::WholeOp::Extract { elem_idx } => {
                            match inputs.into_iter().collect_tuple().unwrap() {
                                (DynLeaf::Vector(elems),) => DynLeaf::Scalar(
                                    elems.into_iter().nth(usize::from(elem_idx)).unwrap(),
                                ),
                                _ => unreachable!(),
                            }
                        }
                        vector::WholeOp::Insert { elem_idx } => {
                            let (new_elem, val) = inputs.into_iter().collect_tuple().unwrap();
                            match val {
                                DynLeaf::Vector(mut elems) => {
                                    elems[usize::from(elem_idx)] = new_elem.try_into().unwrap();
                                    DynLeaf::Vector(elems)
                                }
                                _ => unreachable!(),
                            }
                        }
                        vector::WholeOp::DynExtract => todo!(),
                        vector::WholeOp::DynInsert => todo!(),
                        vector::WholeOp::Mul => match inputs.into_iter().collect_tuple().unwrap() {
                            (DynLeaf::Vector(elems), DynLeaf::Scalar(factor)) => DynLeaf::Vector(
                                elems
                                    .into_iter()
                                    .map(|x| {
                                        self.eval_scalar_op(
                                            scalar::Op::FloatBinary(scalar::FloatBinOp::Mul),
                                            [x, factor.clone()].into_iter().collect(),
                                            factor.0,
                                        )
                                    })
                                    .collect(),
                            ),
                            _ => unreachable!(),
                        },
                    },
                };
                [self.new_val(output_type, output)].into_iter().collect()
            }
            &DataInstKind::FuncCall(callee) => self.eval_call(callee, inputs),
            DataInstKind::Mem(_) | DataInstKind::QPtr(_) | DataInstKind::ThunkBind(_) => todo!(),
            DataInstKind::SpvInst(spv_inst, _) => {
                let output_type = output_types.at_most_one().ok().unwrap();
                let output = if spv_inst.opcode == self.wk.OpVariable {
                    let mut init = {
                        let init_ty = self.as_spv_ptr_type(output_type.unwrap()).unwrap().1;
                        AllocData {
                            ty: init_ty,
                            leaves: init_ty
                                .disaggregated_leaf_types(cx)
                                .map(|leaf_type| self.eval_undef_const(leaf_type))
                                .collect(),

                            globally_shared: false,
                        }
                    };
                    if !inputs.is_empty() {
                        assert_eq!(init.leaves.len(), inputs.len());
                        for (init_slot, input_leaf) in init.leaves.iter_mut().zip_eq(inputs) {
                            assert!(init_slot.ty == input_leaf.ty);
                            *init_slot = input_leaf;
                        }
                    }
                    let alloc = self.mem_state.as_mut().unwrap().alloc(init);
                    self.call_stack.last_mut().unwrap().dealloc_on_exit.push(alloc);
                    Some(self.new_val(
                        output_type.unwrap(),
                        DynLeaf::Ptr {
                            base: DynPtrBase::Alloc(alloc),
                            leaf_range: DynData::Uniform(0)..,
                            vector_component: None,
                        },
                    ))
                } else if spv_inst.opcode == self.wk.OpLoad {
                    let (ptr,) = inputs.into_iter().collect_tuple().unwrap();
                    Some(self.eval_mem_load(self.mem_state.as_ref().unwrap(), ptr))
                } else if spv_inst.opcode == self.wk.OpStore {
                    let (ptr, val) = inputs.into_iter().collect_tuple().unwrap();

                    let mut mem_state = self.mem_state.take().unwrap();
                    self.eval_mem_store(&mut mem_state, ptr, val);
                    self.mem_state = Some(mem_state);

                    None
                } else if spv_inst.opcode == self.wk.OpCopyMemory {
                    // HACK(eddyb) this avoids supporting non-leaf loads/stores.
                    let (dst, src) = inputs.into_iter().collect_tuple().unwrap();

                    let pointee_type = {
                        let (_, dst_pointee_type) = self.as_spv_ptr_type(dst.ty).unwrap();
                        let (_, src_pointee_type) = self.as_spv_ptr_type(src.ty).unwrap();
                        assert!(dst_pointee_type == src_pointee_type);
                        src_pointee_type
                    };

                    let mut mem_state = self.mem_state.take().unwrap();
                    for (i, leaf_type) in pointee_type.disaggregated_leaf_types(cx).enumerate() {
                        let [leaf_dst, leaf_src] = [&dst, &src].map(|ptr| {
                            let mut leaf_ptr = ptr.clone();
                            leaf_ptr.ty = self.modify_spv_ptr_type(ptr.ty, leaf_type);
                            match &mut leaf_ptr.kind {
                                DynLeaf::Ptr { leaf_range, .. } => {
                                    *leaf_range = leaf_range
                                        .start
                                        .clone()
                                        .map(|x| x.checked_add(i).unwrap())..;
                                }
                                _ => unreachable!(),
                            }
                            leaf_ptr
                        });
                        let leaf = self.eval_mem_load(&mem_state, leaf_src);
                        self.eval_mem_store(&mut mem_state, leaf_dst, leaf);
                    }
                    self.mem_state = Some(mem_state);

                    None
                } else if spv_inst.opcode == self.wk.OpAtomicLoad {
                    Some(self.eval_mem_load(self.mem_state.as_ref().unwrap(), inputs[0].clone()))
                } else if spv_inst.opcode == self.wk.OpAtomicCompareExchange {
                    let mut mem_state = self.mem_state.take().unwrap();

                    // TODO(eddyb) actually handle this per-invocation, not once.
                    let old = self.eval_mem_load(&mem_state, inputs[0].clone());
                    let new = match (&old, &inputs[4], &inputs[5]) {
                        (
                            DynVal { kind: DynLeaf::Scalar(old_scalar), .. },
                            new_if_eq,
                            DynVal { kind: DynLeaf::Scalar(cmp_scalar), .. },
                        ) => self.eval_select(
                            DynLeaf::Scalar(self.eval_scalar_op(
                                scalar::IntBinOp::Eq.into(),
                                [old_scalar.clone(), cmp_scalar.clone()].into_iter().collect(),
                                bool::TYPE,
                            )),
                            new_if_eq.clone(),
                            old.clone(),
                            output_type.unwrap(),
                        ),
                        _ => unreachable!(),
                    };
                    self.eval_mem_store(&mut mem_state, inputs[0].clone(), new);
                    self.mem_state = Some(mem_state);

                    Some(old)
                } else if spv_inst.opcode == self.wk.OpAtomicIAdd {
                    let mut mem_state = self.mem_state.take().unwrap();

                    // TODO(eddyb) actually handle this per-invocation, not once.
                    let old = self.eval_mem_load(&mem_state, inputs[0].clone());
                    let new = match (&old.kind, &inputs[3].kind) {
                        (DynLeaf::Scalar(old_scalar), DynLeaf::Scalar(addend)) => self.new_val(
                            old.ty,
                            DynLeaf::Scalar(self.eval_scalar_op(
                                scalar::IntBinOp::Add.into(),
                                [old_scalar.clone(), addend.clone()].into_iter().collect(),
                                old_scalar.0,
                            )),
                        ),
                        _ => unreachable!(),
                    };
                    self.eval_mem_store(&mut mem_state, inputs[0].clone(), new);
                    self.mem_state = Some(mem_state);

                    Some(old)
                } else if spv_inst.opcode == self.wk.OpAccessChain
                    || spv_inst.opcode == self.wk.OpInBoundsAccessChain
                {
                    let mut inputs = inputs.into_iter();
                    let original_ptr = inputs.next().unwrap();

                    let DynLeaf::Ptr { base, mut leaf_range, mut vector_component } =
                        original_ptr.kind
                    else {
                        unreachable!();
                    };
                    let (_, mut pointee_type) = self.as_spv_ptr_type(original_ptr.ty).unwrap();

                    while let Some(i) = inputs.next() {
                        let i = match i.kind {
                            DynLeaf::Scalar(DynScalar(u32::TYPE, DynScalarData::B32(i))) => i,
                            _ => unreachable!(),
                        };

                        if let Some(ty) = pointee_type.as_vector(cx) {
                            let i = match i {
                                DynData::Uniform(i) => i,
                                _ => todo!("non-uniform vector index in OpAccessChain"),
                            };
                            assert!(i < ty.elem_count.get().into());
                            assert_eq!(vector_component, None);
                            vector_component = Some(i.try_into().unwrap());

                            // HACK(eddyb) avoids needing to update `pointee_type`
                            // (which would require interning `ty.elem`).
                            assert!(inputs.next().is_none());
                            break;
                        }

                        let static_or_dyn_idx = match (i, &cx[pointee_type].kind) {
                            (DynData::Uniform(i), _) => Ok(i),
                            (
                                i,
                                TypeKind::SpvInst {
                                    value_lowering:
                                        spv::ValueLowering::Disaggregate(spv::AggregateShape::Array {
                                            ..
                                        }),
                                    ..
                                },
                            ) => Err(i),
                            _ => unreachable!(
                                "non-uniform OpAccessChain into {}",
                                spirt::print::Plan::for_root(cx, &pointee_type)
                                    .pretty_print()
                                    .to_string()
                            ),
                        };

                        let (component_type, component_leaf_range) = pointee_type
                            .aggregate_component_type_and_leaf_range(
                                cx,
                                static_or_dyn_idx.as_ref().ok().copied().unwrap_or(0),
                            )
                            .unwrap();
                        assert_eq!(
                            component_leaf_range.len(),
                            cx[component_type].disaggregated_leaf_count()
                        );

                        leaf_range = match static_or_dyn_idx {
                            Ok(_) => {
                                leaf_range
                                    .start
                                    .map(|x| x.checked_add(component_leaf_range.start).unwrap())..
                            }
                            Err(dyn_idx) => {
                                assert_eq!(component_leaf_range.start, 0);
                                let stride = component_leaf_range.end;

                                (leaf_range.start, dyn_idx).map(|start, i| {
                                    start
                                        .checked_add(
                                            usize::try_from(i)
                                                .unwrap()
                                                .checked_mul(stride)
                                                .unwrap(),
                                        )
                                        .unwrap()
                                })..
                            }
                        };

                        pointee_type = component_type;
                    }
                    let final_ptr_type =
                        self.modify_spv_ptr_type(output_type.unwrap(), pointee_type);
                    Some(self.new_val(
                        final_ptr_type,
                        DynLeaf::Ptr { base, leaf_range, vector_component },
                    ))
                } else if spv_inst.opcode == self.wk.OpArrayLength {
                    let (_, pointee_type) = self.as_spv_ptr_type(inputs[0].ty).unwrap();
                    let [spv::Imm::Short(_, array_field_idx)] = spv_inst.imms[..] else {
                        unreachable!();
                    };

                    let array_field_type = match &cx[pointee_type].kind {
                        TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
                            if spv_inst.opcode == self.wk.OpTypeStruct =>
                        {
                            match type_and_const_inputs[usize::try_from(array_field_idx).unwrap()] {
                                TypeOrConst::Type(ty) => ty,
                                TypeOrConst::Const(_) => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };

                    let array_length = match &cx[array_field_type].kind {
                        // HACK(eddyb) this relies on something else having already
                        // replaced `OpTypeRuntimeArray` with `OpTypeArray`.
                        TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
                            if spv_inst.opcode == self.wk.OpTypeArray =>
                        {
                            match type_and_const_inputs[..] {
                                [TypeOrConst::Type(_), TypeOrConst::Const(len)] => len,
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };

                    Some(self.eval_value(Value::Const(array_length)))
                } else if spv_inst.opcode == self.wk.OpSelect {
                    let (cond, t, e) = inputs.into_iter().collect_tuple().unwrap();
                    Some(self.eval_select(cond.kind, t, e, output_type.unwrap()))
                } else if spv_inst.opcode == self.wk.OpBitcast {
                    let (x,) = inputs.into_iter().collect_tuple().unwrap();
                    Some(match (x.kind, &cx[output_type.unwrap()].kind) {
                        (
                            DynLeaf::Scalar(DynScalar(in_type, in_data)),
                            &TypeKind::Scalar(out_type),
                        ) if in_type.bit_width() == out_type.bit_width() => self.new_val(
                            output_type.unwrap(),
                            DynLeaf::Scalar(DynScalar(out_type, in_data)),
                        ),
                        _ => unreachable!(),
                    })
                } else {
                    todo!("unsupported {}", describe_spv_inst())
                };
                output.into_iter().collect()
            }

            &DataInstKind::SpvExtInst { ext_set, inst, .. } if ext_set == self.glsl_std_450 => {
                let inputs = inputs.into_iter().map(|x| x.kind);
                let output_type = output_types.exactly_one().ok().unwrap();

                macro_rules! float_unop {
                    ($($f:tt)+) => {
                        scalar_or_vec_distribute(|inputs: ArrayVec<_, 1>| {
                            let op = ();
                            try_dispatch_scalar! { inputs.match op => (f32) -> _:
                                () => ($($f)+)
                            }.unwrap()
                        })(inputs)
                    };
                }
                macro_rules! float_binop {
                    ($($f:tt)+) => {
                        scalar_or_vec_distribute(|inputs: ArrayVec<_, 2>| {
                            let op = ();
                            try_dispatch_scalar! { inputs.match op => (f32, f32) -> _:
                                () => ($($f)+)
                            }.unwrap()
                        })(inputs)
                    };
                }
                // FIXME(eddyb) this is a silly name, find a better one (`op3`? overload macro?).
                macro_rules! float_triop {
                    ($($f:tt)+) => {
                        scalar_or_vec_distribute(|inputs: ArrayVec<_, 3>| {
                            let op = ();
                            None.or_else(|| {
                                // FIXME(eddyb) remove cloning by only attempting
                                // one dispatch (based on input/output types).
                                let inputs = inputs.clone();

                                try_dispatch_scalar! { inputs.match op => (f32, f32, f32) -> _:
                                    () => ($($f)+)
                                }
                            }).or_else(|| {
                                try_dispatch_scalar! { inputs.match op => (f64, f64, f64) -> _:
                                    () => ($($f)+)
                                }
                            }).unwrap()
                        })(inputs)
                    };
                }

                let output = if inst == self.wk.FAbs {
                    float_unop!(|x| x.abs())
                } else if inst == self.wk.Round {
                    float_unop!(|x| x.round())
                } else if inst == self.wk.Exp {
                    float_unop!(|x| x.exp())
                } else if inst == self.wk.Sqrt {
                    float_unop!(|x| x.sqrt())
                } else if inst == self.wk.Sin {
                    float_unop!(|x| x.sin())
                } else if inst == self.wk.Cos {
                    float_unop!(|x| x.cos())
                } else if inst == self.wk.FMin {
                    float_binop!(|x, y| x.min(y))
                } else if inst == self.wk.FMax {
                    float_binop!(|x, y| x.max(y))
                } else if inst == self.wk.Pow {
                    float_binop!(|x, y| x.powf(y))
                } else if inst == self.wk.Fma {
                    float_triop!(|x, y, z| x.mul_add(y, z))
                } else {
                    todo!("unsupported {}", describe_spv_inst())
                };
                [self.new_val(output_type, output)].into_iter().collect()
            }

            &DataInstKind::SpvExtInst { ext_set, inst, .. }
                if ext_set == self.non_semantic_debug_printf && inst == 1 =>
            {
                let output_type = output_types.exactly_one().ok().unwrap();

                match inputs[0].clone().kind {
                    DynLeaf::SpvStringLiteralForExtInst(s) => {
                        if self.stderr_left_dirty_by_step.take().is_some() {
                            eprintln!();
                        }

                        let s = &self.cx()[s];
                        match (s, &inputs[1..]) {
                            (
                                "%u",
                                [
                                    DynVal {
                                        kind:
                                            DynLeaf::Scalar(DynScalar(
                                                _,
                                                DynScalarData::B32(DynData::Uniform(x)),
                                            )),
                                        ..
                                    },
                                ],
                            ) => {
                                eprintln!("{x}");
                            }
                            _ => eprintln!("{s}"),
                        }
                    }
                    _ => unreachable!(),
                }

                [self.new_val(output_type, DynLeaf::SpvVoidTypedValueFromExtInst)]
                    .into_iter()
                    .collect()
            }

            &DataInstKind::SpvExtInst { .. } => {
                todo!("unsupported {}", describe_spv_inst());
            }
        }
    }
    fn eval_mem_load(&self, mem_state: &MemState, ptr: DynVal) -> DynVal {
        let DynLeaf::Ptr { base: DynPtrBase::Alloc(alloc), leaf_range, vector_component } =
            ptr.kind
        else {
            unreachable!();
        };

        let alloc_leaves = &mem_state[alloc].leaves;

        let loaded_leaf_count =
            self.cx()[self.as_spv_ptr_type(ptr.ty).unwrap().1].disaggregated_leaf_count();
        let loaded_leaves: SmallVec<[_; 4]> = (0..loaded_leaf_count)
            .map(|leaf_idx| match &leaf_range.start {
                &DynData::Uniform(start) => {
                    alloc_leaves[start.checked_add(leaf_idx).unwrap()].clone()
                }
                DynData::PerInvocation(starts) => {
                    let extract_leaf_for_invocation = |(ii, start): (usize, usize)| {
                        let leaf: &DynVal = &alloc_leaves[start.checked_add(leaf_idx).unwrap()];
                        (
                            leaf.ty,
                            leaf.kind.extract_invocation_as_uniform(ii),
                        )
                    };

                    // FIXME(eddyb) this could be parallelized, with some effort.
                    let mut per_invocation_leaf =
                        starts.iter().copied().enumerate().map(extract_leaf_for_invocation);
                    let (leaf0_ty, leaf0) = per_invocation_leaf.next().unwrap();
                    let per_invocation_leaf = [leaf0.clone()].into_iter().chain(
                        per_invocation_leaf.map(|(leaf_ty, leaf)| {
                            assert!(leaf_ty == leaf0_ty);
                            leaf
                        }),
                    );

                    fn collect_scalar(
                        mut per_invocation_scalar: impl Iterator<Item = DynScalar>,
                    ) -> DynScalar {
                        let DynScalar(scalar0_ty, scalar0_data) =
                            per_invocation_scalar.next().unwrap();
                        let mut per_invocation_data = [scalar0_data.clone()].into_iter().chain(
                            per_invocation_scalar.map(|DynScalar(ty, data)| {
                                assert!(ty == scalar0_ty);
                                data
                            }),
                        );

                        macro_rules! collect {
                            ($($variant:ident),+) => {
                                match scalar0_data {
                                    // FIXME(eddyb) move `undef`s into a separate mask!
                                    DynScalarData::Undef => {
                                        assert!(per_invocation_data.all(|x| matches!(x, DynScalarData::Undef)));
                                        DynScalarData::Undef
                                    }
                                    $(DynScalarData::$variant(_) => DynScalarData::$variant(DynData::PerInvocation(
                                        Rc::new(per_invocation_data.map(|x| match x {
                                            DynScalarData::$variant(DynData::Uniform(x)) => x,
                                            _ => unreachable!(),
                                        }).collect()),
                                    ))),+
                                }
                            };
                        }
                        DynScalar(scalar0_ty, collect!(Bool, B8, B16, B32, B64))
                    }

                    let leaf = match leaf0 {
                        DynLeaf::Scalar(_) => {
                            DynLeaf::Scalar(collect_scalar(per_invocation_leaf.map(|x| match x {
                                DynLeaf::Scalar(x) => x,
                                _ => unreachable!(),
                            })))
                        }
                        DynLeaf::Vector(elems) => {
                            DynLeaf::Vector((0..elems.len()).map(|elem_idx| {
                                collect_scalar(per_invocation_leaf.clone().map(|x| match x {
                                    DynLeaf::Vector(xs) => xs[elem_idx].clone(),
                                    _ => unreachable!(),
                                }))
                            }).collect())
                        }

                        // FIXME(eddyb) could `AllocData`' `leaves` field
                        // support only `DynScalar`, instead of any `DynLeaf`?
                        DynLeaf::Ptr { .. }
                        | DynLeaf::SpvStringLiteralForExtInst(_)
                        | DynLeaf::SpvVoidTypedValueFromExtInst => unreachable!(),
                    };
                    self.new_val(leaf0_ty, leaf)
                }
            })
            .collect();

        // TODO(eddyb) support accessing multiple leaves!
        assert_eq!(loaded_leaves.len(), 1);
        let leaf = loaded_leaves.into_iter().exactly_one().ok().unwrap();

        // FIXME(eddyb) avoid having to create new `DynVal`s because of vectors.
        if let Some(i) = vector_component {
            let DynLeaf::Vector(elems) = leaf.kind else {
                unreachable!();
            };

            let elem = elems[usize::from(i)].clone();
            self.new_val(self.cx().intern(elem.0), DynLeaf::Scalar(elem))
        } else {
            leaf
        }
    }
    fn eval_mem_store(&self, mem_state: &mut MemState, ptr: DynVal, stored_leaf: DynVal) {
        let DynLeaf::Ptr { base: DynPtrBase::Alloc(alloc), leaf_range, vector_component } =
            ptr.kind
        else {
            unreachable!();
        };

        // TODO(eddyb) support accessing multiple leaves!
        assert_eq!(
            self.cx()[self.as_spv_ptr_type(ptr.ty).unwrap().1].disaggregated_leaf_count(),
            1
        );

        assert_eq!(vector_component, None);

        let alloc_data = &mut mem_state[alloc];

        // HACK(eddyb) this is the only place where anything like address space
        // semantics, is handled in `spirti` currently, at all, and it's only
        // really just treating all stored locations as selecting one uniform
        // value to keep, from all invocations storing there.
        if alloc_data.globally_shared {
            let tangle_mask = match &self.tangle {
                DynData::Uniform(false) => return,
                DynData::Uniform(true) => None,
                DynData::PerInvocation(mask) => Some(&mask[..]),
            };

            let mut store_for_invocation = |ii: usize, start| {
                if let Some(mask) = tangle_mask
                    && !mask[ii]
                {
                    return;
                }
                alloc_data.leaves[start] = if stored_leaf.kind.is_uniform() {
                    stored_leaf.clone()
                } else {
                    // FIXME(eddyb) should this require an atomic store?
                    // (and does it matter which invocation is picked?)
                    self.new_val(stored_leaf.ty, stored_leaf.kind.extract_invocation_as_uniform(ii))
                };
            };
            match &leaf_range.start {
                &DynData::Uniform(start) => {
                    store_for_invocation(
                        tangle_mask.and_then(|mask| mask.iter().position(|&x| x)).unwrap_or(0),
                        start,
                    );
                }
                DynData::PerInvocation(starts) => {
                    for (ii, start) in starts.iter().copied().enumerate() {
                        store_for_invocation(ii, start);
                    }
                }
            }
        } else {
            match leaf_range.start {
                DynData::Uniform(start) => {
                    let slot = &mut alloc_data.leaves[start];
                    *slot = self.eval_select(
                        DynLeaf::Scalar(DynScalar(
                            scalar::Type::Bool,
                            DynScalarData::Bool(self.tangle.clone()),
                        )),
                        stored_leaf,
                        slot.clone(),
                        slot.ty,
                    );
                }
                _ => todo!("non-uniform indexed store"),
            }
        }
    }
    fn eval_select(
        &self,
        cond: DynLeaf,
        val_if_true: DynVal,
        val_if_false: DynVal,
        output_type: Type,
    ) -> DynVal {
        let cond = match cond {
            DynLeaf::Scalar(DynScalar(bool::TYPE, cond)) => cond,
            _ => unreachable!(),
        };
        fn select<T>(cond: bool, t: T, e: T) -> T {
            if cond { t } else { e }
        }
        match (cond, val_if_true, val_if_false) {
            // TODO(eddyb) is this UB? or just `undef` output?
            // FIXME(eddyb) this happens because of `qptr::lift`
            // using `OpSelect` to turn booleans into integers.
            (DynScalarData::Undef, _, _) => self.eval_undef_const(output_type),

            (DynScalarData::Bool(DynData::Uniform(cond)), t, e) => select(cond, t, e),

            (
                DynScalarData::Bool(cond),
                DynVal { ty: t_ty, kind: DynLeaf::Scalar(DynScalar(t_sty, t)), .. },
                DynVal { ty: e_ty, kind: DynLeaf::Scalar(DynScalar(e_sty, e)), .. },
            ) if t_ty == e_ty && e_sty == t_sty => self.new_val(
                t_ty,
                DynLeaf::Scalar(DynScalar(
                    t_sty,
                    match (t, e) {
                        (DynScalarData::Undef, DynScalarData::Undef) => DynScalarData::Undef,
                        (DynScalarData::Undef, _) | (_, DynScalarData::Undef) => todo!(),
                        (DynScalarData::Bool(t), DynScalarData::Bool(e)) => {
                            DynScalarData::Bool((cond, t, e).map(select))
                        }
                        (DynScalarData::B32(t), DynScalarData::B32(e)) => {
                            DynScalarData::B32((cond, t, e).map(select))
                        }
                        _ => unreachable!(),
                    },
                )),
            ),

            _ => todo!(),
        }
    }
    fn eval_scalar_op(
        &self,
        op: scalar::Op,
        inputs: ArrayVec<DynScalar, 2>,
        output_type: scalar::Type,
    ) -> DynScalar {
        let input_types: ArrayVec<_, 2> = inputs.iter().map(|&DynScalar(ty, _)| ty).collect();

        if inputs.iter().any(|x| matches!(x, DynScalar(_, DynScalarData::Undef))) {
            // FIXME(eddyb) also check for potential UB (shift/div/rem RHSes).
            let output = match op {
                // TODO(eddyb) try byte-level `undef` tracking, if nothing else.
                scalar::Op::IntBinary(scalar::IntBinOp::And) => match &inputs[..] {
                    [DynScalar(_, DynScalarData::Undef), DynScalar(_, DynScalarData::Undef)] => {
                        DynScalarData::Undef
                    }

                    _ => self.eval_scalar_const(scalar::Const::from_bits(output_type, 0)).1,
                },
                scalar::Op::IntBinary(
                    scalar::IntBinOp::ShrU | scalar::IntBinOp::ShrS | scalar::IntBinOp::Shl,
                ) => match &inputs[..] {
                    [_, DynScalar(_, DynScalarData::Undef)] => DynScalarData::Undef,

                    _ => self.eval_scalar_const(scalar::Const::from_bits(output_type, 0)).1,
                },

                _ => DynScalarData::Undef,
            };
            return DynScalar(output_type, output);
        }

        // HACK(eddyb) see `scalar::Op::FloatBinary` (needed for `CmpOrUnord`).
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        match op {
            scalar::Op::BoolUnary(op) => {
                // FIXME(eddyb) move import into macro.
                use scalar::BoolUnOp::*;
                try_dispatch_scalar! { inputs.match op => (bool) -> _:
                    Not => (|x| !x)
                }
            }
            scalar::Op::BoolBinary(op) => {
                // FIXME(eddyb) move import into macro.
                use scalar::BoolBinOp::*;
                try_dispatch_scalar! { inputs.match op => (bool, bool) -> _:
                    Eq => (==),
                    Ne => (!=),
                    Or => (|),
                    And => (&),
                }
            }
            scalar::Op::IntUnary(op) => {
                let (x,) = inputs.into_iter().collect_tuple().unwrap();
                match op {
                    scalar::IntUnOp::Neg => {
                        assert!(x.0 == output_type);
                        Some(DynScalar(
                            output_type,
                            match x.1 {
                                DynScalarData::Undef => DynScalarData::Undef,
                                DynScalarData::Bool(_) => unreachable!(),
                                DynScalarData::B8(x) => {
                                    DynScalarData::B8(x.map(|x| x.wrapping_neg()))
                                }
                                DynScalarData::B16(x) => {
                                    DynScalarData::B16(x.map(|x| x.wrapping_neg()))
                                }
                                DynScalarData::B32(x) => {
                                    DynScalarData::B32(x.map(|x| x.wrapping_neg()))
                                }
                                DynScalarData::B64(x) => {
                                    DynScalarData::B64(x.map(|x| x.wrapping_neg()))
                                }
                            },
                        ))
                    }
                    scalar::IntUnOp::Not => {
                        assert!(x.0 == output_type);
                        Some(DynScalar(
                            output_type,
                            match x.1 {
                                DynScalarData::Undef => DynScalarData::Undef,
                                DynScalarData::Bool(_) => unreachable!(),
                                DynScalarData::B8(x) => DynScalarData::B8(x.map(|x| !x)),
                                DynScalarData::B16(x) => DynScalarData::B16(x.map(|x| !x)),
                                DynScalarData::B32(x) => DynScalarData::B32(x.map(|x| !x)),
                                DynScalarData::B64(x) => DynScalarData::B64(x.map(|x| !x)),
                            },
                        ))
                    }
                    scalar::IntUnOp::CountOnes => Some(DynScalar(
                        scalar::Type::U32,
                        match x.1 {
                            DynScalarData::Undef => DynScalarData::Undef,
                            DynScalarData::Bool(_) => unreachable!(),
                            DynScalarData::B8(x) => DynScalarData::B32(x.map(|x| x.count_ones())),
                            DynScalarData::B16(x) => DynScalarData::B32(x.map(|x| x.count_ones())),
                            DynScalarData::B32(x) => DynScalarData::B32(x.map(|x| x.count_ones())),
                            DynScalarData::B64(x) => DynScalarData::B32(x.map(|x| x.count_ones())),
                        },
                    )),
                    scalar::IntUnOp::TruncOrZeroExtend => Some(DynScalar(
                        output_type,
                        match x.1 {
                            DynScalarData::Undef => DynScalarData::Undef,
                            DynScalarData::B8(x) if output_type.bit_width() == 16 => {
                                DynScalarData::B16(x.map(|x| x as u16))
                            }
                            DynScalarData::B8(x) if output_type.bit_width() == 32 => {
                                DynScalarData::B32(x.map(|x| x as u32))
                            }
                            DynScalarData::B8(x) if output_type.bit_width() == 64 => {
                                DynScalarData::B64(x.map(|x| x as u64))
                            }
                            DynScalarData::B16(x) if output_type.bit_width() == 8 => {
                                DynScalarData::B8(x.map(|x| x as u8))
                            }
                            DynScalarData::B16(x) if output_type.bit_width() == 32 => {
                                DynScalarData::B32(x.map(|x| x as u32))
                            }
                            DynScalarData::B16(x) if output_type.bit_width() == 64 => {
                                DynScalarData::B64(x.map(|x| x as u64))
                            }
                            DynScalarData::B32(x) if output_type.bit_width() == 8 => {
                                DynScalarData::B8(x.map(|x| x as u8))
                            }
                            DynScalarData::B32(x) if output_type.bit_width() == 16 => {
                                DynScalarData::B16(x.map(|x| x as u16))
                            }
                            DynScalarData::B32(x) if output_type.bit_width() == 64 => {
                                DynScalarData::B64(x.map(|x| x as u64))
                            }
                            DynScalarData::B64(x) if output_type.bit_width() == 16 => {
                                DynScalarData::B16(x.map(|x| x as u16))
                            }
                            DynScalarData::B64(x) if output_type.bit_width() == 8 => {
                                DynScalarData::B8(x.map(|x| x as u8))
                            }
                            DynScalarData::B64(x) if output_type.bit_width() == 32 => {
                                DynScalarData::B32(x.map(|x| x as u32))
                            }
                            _ => todo!(
                                "zext_or_trunc({} -> {})",
                                x.0.bit_width(),
                                output_type.bit_width()
                            ),
                        },
                    )),
                    scalar::IntUnOp::TruncOrSignExtend => Some(DynScalar(
                        output_type,
                        match x.1 {
                            DynScalarData::Undef => DynScalarData::Undef,
                            DynScalarData::B8(x) if output_type.bit_width() == 16 => {
                                DynScalarData::B16(x.map(|x| x as i8 as u16))
                            }
                            DynScalarData::B8(x) if output_type.bit_width() == 32 => {
                                DynScalarData::B32(x.map(|x| x as i8 as u32))
                            }
                            DynScalarData::B8(x) if output_type.bit_width() == 64 => {
                                DynScalarData::B64(x.map(|x| x as i8 as u64))
                            }
                            DynScalarData::B16(x) if output_type.bit_width() == 8 => {
                                DynScalarData::B8(x.map(|x| x as i16 as u8))
                            }
                            DynScalarData::B16(x) if output_type.bit_width() == 32 => {
                                DynScalarData::B32(x.map(|x| x as i16 as u32))
                            }
                            DynScalarData::B16(x) if output_type.bit_width() == 64 => {
                                DynScalarData::B64(x.map(|x| x as i16 as u64))
                            }
                            DynScalarData::B32(x) if output_type.bit_width() == 8 => {
                                DynScalarData::B8(x.map(|x| x as i32 as u8))
                            }
                            DynScalarData::B32(x) if output_type.bit_width() == 16 => {
                                DynScalarData::B16(x.map(|x| x as i32 as u16))
                            }
                            DynScalarData::B32(x) if output_type.bit_width() == 64 => {
                                DynScalarData::B64(x.map(|x| x as i32 as u64))
                            }
                            DynScalarData::B64(x) if output_type.bit_width() == 16 => {
                                DynScalarData::B16(x.map(|x| x as i64 as u16))
                            }
                            DynScalarData::B64(x) if output_type.bit_width() == 8 => {
                                DynScalarData::B8(x.map(|x| x as i64 as u8))
                            }
                            DynScalarData::B64(x) if output_type.bit_width() == 32 => {
                                DynScalarData::B32(x.map(|x| x as i64 as u32))
                            }
                            _ => todo!(
                                "sext_or_trunc({} -> {})",
                                x.0.bit_width(),
                                output_type.bit_width()
                            ),
                        },
                    )),
                }
            }
            // FIXME(eddyb) wrap everything in `Wrapping`.
            scalar::Op::IntBinary(op) => {
                // FIXME(eddyb) maybe use `num-traits` to automate away a lot of this?
                trait Int {
                    type UInt: Int;
                    type SInt: Int;
                }
                trait CastTo<T> {
                    fn cast_to(self) -> T;
                }
                impl<T> CastTo<T> for T {
                    fn cast_to(self) -> T {
                        self
                    }
                }

                macro_rules! int_cast_impls {
                    ($($U:ident <=> $S:ident),+ $(,)?) => {$(
                        impl Int for $U { type UInt = $U; type SInt = $S; }
                        impl Int for $S { type UInt = $U; type SInt = $S; }
                        impl CastTo<$U> for $S { fn cast_to(self) -> $U { self as $U } }
                        impl CastTo<$S> for $U { fn cast_to(self) -> $S { self as $S } }
                    )+}
                }
                int_cast_impls! {
                    u8 <=> i8,
                    u16 <=> i16,
                    u32 <=> i32,
                    u64 <=> i64,
                }

                // HACK(eddyb) comparison ops make this necessary at all.
                trait MaybeReverseCastOutput<I: CastTo<T>, T> {
                    type Output;
                    fn maybe_reverse_cast_output(self) -> Self::Output;
                }
                impl<I: CastTo<T>, T: Int> MaybeReverseCastOutput<I, T> for bool {
                    type Output = Self;
                    fn maybe_reverse_cast_output(self) -> Self {
                        self
                    }
                }
                impl<I: CastTo<T>, T: CastTo<I>> MaybeReverseCastOutput<I, T> for T {
                    type Output = I;
                    fn maybe_reverse_cast_output(self) -> I {
                        self.cast_to()
                    }
                }

                // HACK(eddyb) allow forcing a signedness.
                #[allow(non_snake_case)]
                fn U<I: Int + CastTo<I::UInt>, R: MaybeReverseCastOutput<I, I::UInt>>(
                    f: impl Fn(I::UInt, I::UInt) -> R,
                ) -> impl Fn(I, I) -> <R as MaybeReverseCastOutput<I, I::UInt>>::Output
                {
                    move |x, y| f(x.cast_to(), y.cast_to()).maybe_reverse_cast_output()
                }
                #[allow(non_snake_case)]
                fn S<I: Int + CastTo<I::SInt>, R: MaybeReverseCastOutput<I, I::SInt>>(
                    f: impl Fn(I::SInt, I::SInt) -> R,
                ) -> impl Fn(I, I) -> <R as MaybeReverseCastOutput<I, I::SInt>>::Output
                {
                    move |x, y| f(x.cast_to(), y.cast_to()).maybe_reverse_cast_output()
                }

                macro_rules! dispatch_int_binop {
                    ($inputs:ident.match $op:ident: $($pat:pat => ($($f:tt)+)),+ $(,)?) => {{
                        let (x, y) = $inputs.into_iter().collect_tuple().unwrap();
                        let (u8_input_as_dyn_data, u8_input_from_repr) = <u8 as Scalar>::unpacker();
                        let (i8_input_as_dyn_data, i8_input_from_repr) = <i8 as Scalar>::unpacker();
                        let (u16_input_as_dyn_data, u16_input_from_repr) = <u16 as Scalar>::unpacker();
                        let (i16_input_as_dyn_data, i16_input_from_repr) = <i16 as Scalar>::unpacker();
                        let (u32_input_as_dyn_data, u32_input_from_repr) = <u32 as Scalar>::unpacker();
                        let (i32_input_as_dyn_data, i32_input_from_repr) = <i32 as Scalar>::unpacker();
                        let (u64_input_as_dyn_data, u64_input_from_repr) = <u64 as Scalar>::unpacker();
                        let (i64_input_as_dyn_data, i64_input_from_repr) = <i64 as Scalar>::unpacker();
                        match $op {
                            $($pat => {
                                // HACK(eddyb) closures somewhat used as `try` blocks.
                                None.or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = u8_input_as_dyn_data(x.clone())?;
                                    let y = u8_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(u8_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                }).or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = i8_input_as_dyn_data(x.clone())?;
                                    let y = i8_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(i8_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                }).or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = u16_input_as_dyn_data(x.clone())?;
                                    let y = u16_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(u16_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                }).or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = i16_input_as_dyn_data(x.clone())?;
                                    let y = i16_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(i16_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                }).or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = u32_input_as_dyn_data(x.clone())?;
                                    let y = u32_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(u32_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                }).or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = i32_input_as_dyn_data(x.clone())?;
                                    let y = i32_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(i32_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                }).or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = u64_input_as_dyn_data(x.clone())?;
                                    let y = u64_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(u64_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                }).or_else(|| {
                                    let (output_from_dyn_data, output_to_repr) =
                                        <_ as Scalar>::packer();

                                    // FIXME(eddyb) remove the clones here if possible.
                                    let x = i64_input_as_dyn_data(x.clone())?;
                                    let y = i64_input_as_dyn_data(y.clone())?;
                                    Some(output_from_dyn_data((x, y).map(|x, y| {
                                        let [x, y] = [x, y].map(i64_input_from_repr);
                                        output_to_repr(invoke_op!(($($f)+)(x, y)))
                                    })))
                                })
                            }),+
                        }
                    }};
                }

                #[track_caller]
                fn todo<T>(_: T) -> T {
                    todo!()
                }

                use scalar::IntBinOp::*;
                let mut inputs = inputs;
                if let Add | Sub | Mul = op {
                    let (x, y) = inputs.iter_mut().map(|x| &mut x.0).collect_tuple().unwrap();
                    if *x != *y
                        && let [scalar::Type::SInt(xw), scalar::Type::UInt(yw)]
                        | [scalar::Type::UInt(xw), scalar::Type::SInt(yw)] = [*x, *y]
                        && xw == yw
                    {
                        *y = *x;
                    }
                }
                // HACK(eddyb) inefficient but easy.
                if let ShrU | ShrS | Shl = op && input_types[0] != input_types[1] {
                    // HACK(eddyb) sanity-checking shifts.
                    if let DynScalar(_, DynScalarData::B32(DynData::Uniform(rhs))) = inputs[1] {
                        assert_eq!(rhs, rhs & (input_types[0].bit_width() - 1));
                    }

                    inputs[1] = if input_types[0].bit_width() == input_types[1].bit_width() {
                        DynScalar(input_types[0], inputs[1].1.clone())
                    } else {
                        self.eval_scalar_op(
                            scalar::IntUnOp::TruncOrZeroExtend.into(),
                            [inputs[1].clone()].into_iter().collect(),
                            input_types[0],
                        )
                    };
                }
                dispatch_int_binop! { inputs.match op:
                    Add => (+),
                    Sub => (-),
                    Mul => (*),
                    DivU => (U(/)),
                    DivS => (S(/)),
                    ModU => (U(%)),
                    RemS => (S(%)),
                    ModS => (S(|_x, _y| todo(_x))),
                    ShrU => (U(>>)),
                    ShrS => (S(>>)),
                    Shl => (<<),
                    Or => (|),
                    Xor => (^),
                    And => (&),
                    // FIXME(eddyb) these are unreachable (handled separately).
                    CarryingAdd => (|_x, _y| todo(_x)),
                    BorrowingSub => (|_x, _y| todo(_x)),
                    WideningMulU => (U(|_x, _y| todo(_x))),
                    WideningMulS => (S(|_x, _y| todo(_x))),
                    Eq => (==),
                    Ne => (!=),
                    GtU => (U(>)),
                    GtS => (S(>)),
                    GeU => (U(>=)),
                    GeS => (S(>=)),
                    LtU => (U(<)),
                    LtS => (S(<)),
                    LeU => (U(<=)),
                    LeS => (S(<=)),
                }
            }
            scalar::Op::FloatUnary(op) => {
                #[track_caller]
                fn todo<T>() -> T {
                    todo!()
                }

                use scalar::FloatUnOp::*;
                let mut inputs = inputs;
                // HACK(eddyb) inefficient but easy.
                if let FromUInt = op && input_types[0].bit_width() < 64 {
                    inputs[0] = self.eval_scalar_op(
                        scalar::IntUnOp::TruncOrZeroExtend.into(),
                        [inputs[0].clone()].into_iter().collect(),
                        u64::TYPE,
                    );
                }
                if let FromSInt = op && input_types[0].bit_width() < 64 {
                    inputs[0] = self.eval_scalar_op(
                        scalar::IntUnOp::TruncOrSignExtend.into(),
                        [inputs[0].clone()].into_iter().collect(),
                        i64::TYPE,
                    );
                }
                if let ToUInt | ToSInt = op && input_types[0].bit_width() < 64 {
                    inputs[0] = self.eval_scalar_op(
                        scalar::FloatUnOp::Convert.into(),
                        [inputs[0].clone()].into_iter().collect(),
                        f64::TYPE,
                    );
                }
                // FIXME(eddyb) partition these by "signature shape" first,
                // including declaring custom `enum`s to streamline dispatch.
                let op_and_output_type = (op, output_type);
                try_dispatch_scalar! { inputs.match op_and_output_type => (_) -> _:
                    (Neg, f64::TYPE) => (|x: f64| -x),
                    (Neg, _) => (|x: f32| -x),
                    (IsNan, _) => (|x: f32| x.is_nan()),
                    (IsInf, _) => (|x: f32| x.is_infinite()),
                    // FIXME(eddyb) handle both signed and unsigned here.
                    (FromUInt, f32::TYPE) => (|x: u64| x as f32),
                    (FromSInt, f32::TYPE) => (|x: i64| x as f32),
                    (FromUInt, f64::TYPE) => (|x: u64| x as f64),
                    (FromSInt, f64::TYPE) => (|x: i64| x as f64),
                    (FromUInt, _) => (|_x: u64| todo::<f32>()),
                    (FromSInt, _) => (|_x: i64| todo::<f32>()),
                    (ToUInt, u8::TYPE) => (|x: f64| x as u8),
                    (ToSInt, i8::TYPE) => (|x: f64| x as i8),
                    (ToUInt, u16::TYPE) => (|x: f64| x as u16),
                    (ToSInt, i16::TYPE) => (|x: f64| x as i16),
                    (ToUInt, u32::TYPE) => (|x: f64| x as u32),
                    (ToSInt, i32::TYPE) => (|x: f64| x as i32),
                    (ToUInt, u64::TYPE) => (|x: f64| x as u64),
                    (ToSInt, i64::TYPE) => (|x: f64| x as i64),
                    (ToUInt, _) => (|_x: f64| todo::<u32>()),
                    (ToSInt, _) => (|_x: f64| todo::<i32>()),
                    (Convert, f64::TYPE) => (|x: f32| x as f64),
                    (Convert, f32::TYPE) => (|x: f64| x as f32),
                    (Convert, _) => (|_x: f32| todo::<f32>()),
                    (QuantizeAsF16, _) => (|_x: f32| todo::<f32>()),
                }
            }
            scalar::Op::FloatBinary(op) => {
                #[track_caller]
                fn todo<T>() -> T {
                    todo!()
                }

                // FIXME(eddyb) use a generic function here, instead.
                macro_rules! try_dispatch_float_binop {
                    ($($F:ty),+ $(,)?) => {
                        None $(.or_else(|| {
                            // FIXME(eddyb) remove cloning by only attempting
                            // one dispatch (based on `input_types`/`output_type`).
                            let inputs = inputs.clone();

                            // FIXME(eddyb) move import into macro.
                            use scalar::{FloatBinOp::*, FloatCmp::*};
                            try_dispatch_scalar! { inputs.match op => ($F, $F) -> _:
                                Add => (+),
                                Sub => (-),
                                Mul => (*),
                                Div => (/),
                                Rem => (%),
                                Mod => (|_x, _y| todo::<$F>()),
                                Cmp(Eq) => (==),
                                Cmp(Ne) => (!=),
                                Cmp(Lt) => (<),
                                Cmp(Gt) => (>),
                                Cmp(Le) => (<=),
                                Cmp(Ge) => (>=),
                                // HACK(eddyb) all of these negate the opposite comparison,
                                // which flips unordered from always `false` to always `true`.
                                CmpOrUnord(Eq) => (! !=),
                                CmpOrUnord(Ne) => (! ==),
                                CmpOrUnord(Lt) => (! >=),
                                CmpOrUnord(Gt) => (! <=),
                                CmpOrUnord(Le) => (! >),
                                CmpOrUnord(Ge) => (! <),
                            }
                        }))+
                    }
                }
                try_dispatch_float_binop!(f32, f64)
            }
        }.unwrap_or_else(|| {
            let cx = self.cx();
            let input_types = input_types.iter().map(|&ty| {
                let ty: Type = cx.intern(ty);
                spirt::print::Plan::for_root(cx, &ty).pretty_print().to_string()
            }).collect::<ArrayVec<_, 2>>().join(", ");
            unreachable!("unsupported `{}` with `({input_types})` inputs", op.name());
        })
    }
    fn eval_value(&mut self, value: Value) -> DynVal {
        match value {
            Value::Const(ct) => {
                let ct_def = &self.cx()[ct];
                let kind = match &ct_def.kind {
                    ConstKind::Undef => self.eval_undef_const(ct_def.ty).kind,
                    &ConstKind::Scalar(ct) => DynLeaf::Scalar(self.eval_scalar_const(ct)),
                    ConstKind::Vector(ct) => {
                        DynLeaf::Vector(ct.elems().map(|ct| self.eval_scalar_const(ct)).collect())
                    }
                    &ConstKind::PtrToGlobalVar { global_var, offset: None } => {
                        // FIXME(eddyb) move this into e.g. `eval_node`, handling
                        // all `Value::Const(PtrToGlobalVar {..})` inputs early.
                        let alloc = self.lazy_init_global_var(global_var);
                        let alloc_data = &self.mem_state.as_ref().unwrap()[alloc];

                        let kind = DynLeaf::Ptr {
                            base: DynPtrBase::Alloc(alloc),
                            leaf_range: DynData::Uniform(0)..,
                            vector_component: None,
                        };

                        // HACK(eddyb) allow allocations to have different types
                        // from their declarations (e.g. for `OpTypeRuntimeArray`).
                        if alloc_data.ty != self.as_spv_ptr_type(ct_def.ty).unwrap().1 {
                            return self
                                .new_val(self.modify_spv_ptr_type(ct_def.ty, alloc_data.ty), kind);
                        }

                        kind
                    }
                    ConstKind::PtrToGlobalVar { global_var: _, offset: Some(_) } => todo!(),
                    ConstKind::PtrToFunc(_) => todo!(),
                    ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                        let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                        // FIXME(eddyb) support plugging in specialization values.
                        if spv_inst.opcode == self.wk.OpSpecConstant {
                            assert_eq!(const_inputs.len(), 0);
                            match (&self.cx()[ct_def.ty].kind, &spv_inst.imms[..]) {
                                (&TypeKind::Scalar(ty), &[spv::Imm::Short(_, bits)]) => {
                                    assert!(ty.bit_width() <= 32);
                                    DynLeaf::Scalar(self.eval_scalar_const(
                                        scalar::Const::from_bits(ty, bits.into()),
                                    ))
                                }
                                _ => todo!(
                                    "unsupported `OpSpecConstant` type `{}`",
                                    spirt::print::Plan::for_root(self.cx(), &ct_def.ty)
                                        .pretty_print()
                                ),
                            }
                        } else {
                            todo!(
                                "unsupported const `{}`",
                                spirt::print::Plan::for_root(self.cx(), &ct).pretty_print()
                            )
                        }
                    }
                    &ConstKind::SpvStringLiteralForExtInst(s) => {
                        DynLeaf::SpvStringLiteralForExtInst(s)
                    }
                };
                self.new_val(ct_def.ty, kind)
            }
            Value::Var(var) => self.call_stack.last().unwrap().var_values[var].clone(),
        }
    }
    fn eval_undef_const(&self, ty: Type) -> DynVal {
        let kind = match self.cx()[ty].kind {
            TypeKind::Scalar(ty) => DynLeaf::Scalar(DynScalar(ty, DynScalarData::Undef)),
            TypeKind::Vector(ty) => DynLeaf::Vector(
                (0..ty.elem_count.get())
                    .map(|_| DynScalar(ty.elem, DynScalarData::Undef))
                    .collect(),
            ),
            _ => todo!(
                "unsupported `undef` of type `{}`",
                spirt::print::Plan::for_root(self.cx(), &ty).pretty_print()
            ),
        };
        self.new_val(ty, kind)
    }
    fn eval_scalar_const(&self, ct: scalar::Const) -> DynScalar {
        let ty = ct.ty();
        DynScalar(
            ty,
            match ct {
                scalar::Const::FALSE => DynScalarData::Bool(DynData::Uniform(false)),
                scalar::Const::TRUE => DynScalarData::Bool(DynData::Uniform(true)),
                _ => match ty.bit_width() {
                    8 => DynScalarData::B8(DynData::Uniform(u8::try_from(ct.bits()).unwrap())),
                    16 => DynScalarData::B16(DynData::Uniform(u16::try_from(ct.bits()).unwrap())),
                    32 => DynScalarData::B32(DynData::Uniform(u32::try_from(ct.bits()).unwrap())),
                    64 => DynScalarData::B64(DynData::Uniform(u64::try_from(ct.bits()).unwrap())),
                    _ => {
                        let ct: Const = self.cx().intern(ct);
                        todo!(
                            "unsupported const `{}`",
                            spirt::print::Plan::for_root(self.cx(), &ct).pretty_print()
                        )
                    }
                },
            },
        )
    }
    fn lazy_init_global_var(&mut self, gv: GlobalVar) -> AllocId {
        if let Some(&alloc) = self.global_vars.get(gv) {
            return alloc;
        }

        let gv_decl = &self.module.global_vars[gv];
        let init = match &gv_decl.def {
            DeclDef::Imported(_) => None,
            DeclDef::Present(def) => def.initializer.as_ref(),
        };
        let init = init.map(|init| {
            let (ty, leaves) = match init {
                &spirt::GlobalVarInit::Direct(init) => {
                    let init = self.eval_value(Value::Const(init));
                    (init.ty, vec![init])
                }
                spirt::GlobalVarInit::SpvAggregate { ty, leaves } => {
                    (*ty, leaves.iter().map(|&leaf| self.eval_value(Value::Const(leaf))).collect())
                }
                spirt::GlobalVarInit::Data(_) => unreachable!(),
            };
            AllocData {
                ty,
                leaves,
                // FIXME(eddyb) check the storage class.
                globally_shared: false,
            }
        });
        let init =
            init.unwrap_or_else(|| self.claim_binding_or_default_init_global_var(gv, gv_decl));

        let alloc = self.mem_state.as_mut().unwrap().alloc(init);
        self.global_vars.insert(gv, alloc);
        self.global_vars_keys.insert(gv);
        alloc
    }
    fn bind_slot_of_global_var(&self, gv_decl: &GlobalVarDecl) -> Option<BindSlot> {
        if gv_decl.addr_space == AddrSpace::SpvStorageClass(self.wk.PushConstant) {
            return Some(BindSlot::PushConstant);
        }

        let get_decoration = |decoration| {
            self.get_spv_attr(gv_decl.attrs, self.wk.OpDecorate, self.wk.Decoration, decoration)
                .map(|imms| match imms {
                    &[spv::Imm::Short(_, x)] => x,
                    _ => unreachable!(),
                })
        };

        if gv_decl.addr_space == AddrSpace::SpvStorageClass(self.wk.StorageBuffer)
            && let Some(descriptor_set) = get_decoration(self.wk.DescriptorSet)
            && let Some(binding) = get_decoration(self.wk.Binding)
        {
            return Some(BindSlot::StorageBuffer { descriptor_set, binding });
        }

        None
    }
    fn claim_binding_or_default_init_global_var(
        &mut self,
        gv: GlobalVar,
        gv_decl: &GlobalVarDecl,
    ) -> AllocData {
        let cx = self.cx();
        let init_ty = self.as_spv_ptr_type(gv_decl.type_of_ptr_to).unwrap().1;

        if gv_decl.addr_space == AddrSpace::SpvStorageClass(self.wk.Input)
            && let Some(&[spv::Imm::Short(builtin_kind, builtin)]) = self.get_spv_attr(
                gv_decl.attrs,
                self.wk.OpDecorate,
                self.wk.Decoration,
                self.wk.BuiltIn,
            )
        {
            let print_builtin = || {
                spv::print::operand_from_imms([spv::Imm::Short(builtin_kind, builtin)])
                    .concat_to_plain_text()
            };
            let init = if builtin == self.wk.FragCoord {
                match self.launch.as_ref().unwrap() {
                    Launch::FragRect { width, height } => DynLeaf::Vector(
                        [
                            DynData::PerInvocation(Rc::new(
                                (0..height.get())
                                    .flat_map(|_| 0..width.get())
                                    .map(|x| (x as f32) + 0.5)
                                    .map(f32::to_bits)
                                    .collect(),
                            )),
                            DynData::PerInvocation(Rc::new(
                                (0..height.get())
                                    .map(|y| (y as f32) + 0.5)
                                    .flat_map(|y| (0..width.get()).map(move |_| y))
                                    .map(f32::to_bits)
                                    .collect(),
                            )),
                            DynData::Uniform(f32::to_bits(0.0)),
                            DynData::Uniform(f32::to_bits(1.0)),
                        ]
                        .map(|f32_data| DynScalar(f32::TYPE, DynScalarData::B32(f32_data)))
                        .into(),
                    ),
                    _ => unreachable!("{} used outside Fragment shader", print_builtin()),
                }
            } else if [self.wk.GlobalInvocationId, self.wk.LocalInvocationId, self.wk.WorkgroupId]
                .contains(&builtin)
            {
                match self.launch.as_ref().unwrap() {
                    &Launch::Compute { local: [lx, ly, lz], global: [gx, gy, gz] } => {
                        let [fx, fy, fz] = [
                            lx.checked_mul(gx).unwrap(),
                            ly.checked_mul(gy).unwrap(),
                            lz.checked_mul(gz).unwrap(),
                        ];
                        let flat_ids = (0..fx.get()).flat_map(|x| {
                            (0..fy.get()).flat_map(move |y| (0..fz.get()).map(move |z| [x, y, z]))
                        });

                        let wanted_xyz = match builtin {
                            _ if builtin == self.wk.GlobalInvocationId => [fx, fy, fz],
                            _ if builtin == self.wk.LocalInvocationId => [lx, ly, lz],
                            _ if builtin == self.wk.WorkgroupId => [gx, gy, gz],
                            _ => unreachable!(),
                        };

                        DynLeaf::Vector(
                            (0..3)
                                .map(|i| {
                                    if wanted_xyz[i].get() == 1 {
                                        DynData::Uniform(0)
                                    } else {
                                        DynData::PerInvocation(Rc::new(
                                            flat_ids
                                                .clone()
                                                .map(|f_xyz| {
                                                    let f_i = f_xyz[i];
                                                    let l_i = [lx, ly, lz][i];
                                                    match builtin {
                                                        _ if builtin
                                                            == self.wk.GlobalInvocationId =>
                                                        {
                                                            f_i
                                                        }
                                                        _ if builtin
                                                            == self.wk.LocalInvocationId =>
                                                        {
                                                            f_i % l_i
                                                        }
                                                        _ if builtin == self.wk.WorkgroupId => {
                                                            f_i / l_i
                                                        }
                                                        _ => unreachable!(),
                                                    }
                                                })
                                                .collect(),
                                        ))
                                    }
                                })
                                .map(|u32_data| DynScalar(u32::TYPE, DynScalarData::B32(u32_data)))
                                .collect(),
                        )
                    }
                    _ => unreachable!("{} used outside Compute shader", print_builtin()),
                }
            } else {
                todo!("unknown {}", print_builtin());
            };
            return AllocData {
                ty: init_ty,
                leaves: vec![self.new_val(init_ty, init)],
                globally_shared: false,
            };
        }

        if let Some(bind_slot) = self.bind_slot_of_global_var(gv_decl) {
            let init_bytes = match self.bindings.insert(bind_slot, BindState::ClaimedBy(gv)) {
                None => panic!("{bind_slot:?} initializer not provided"),
                Some(BindState::ClaimedBy(_)) => {
                    panic!("{bind_slot:?} aliased by multiple globals")
                }
                Some(BindState::Unclaimed { init }) => init,
            };

            let mut ty = init_ty;

            // HACK(eddyb) rewrite `StorageBuffer` "blocks" ending in `OpTypeRuntimeArray`.
            let ty_def = &cx[ty];
            if let Ok(spirt::mem::layout::TypeLayout::Handle(spirt::mem::shapes::Handle::Buffer(
                _,
                buf_layout,
            ))) = self.layout_cache.layout_of(ty)
                && let Some(buf_stride) = buf_layout.mem_layout.dyn_unit_stride
                && let TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. } = &ty_def.kind
                && spv_inst.opcode == self.wk.OpTypeStruct
                && let Some(&TypeOrConst::Type(last_field_type)) = type_and_const_inputs.last()
                && let TypeKind::SpvInst { spv_inst, .. } = &cx[last_field_type].kind
                && spv_inst.opcode == self.wk.OpTypeRuntimeArray
            {
                let dyn_len = (init_bytes.len() as u32)
                    .saturating_sub(buf_layout.mem_layout.fixed_base.size)
                    / buf_stride.get();
                assert_eq!(
                    init_bytes.len(),
                    (dyn_len * buf_stride.get() + buf_layout.mem_layout.fixed_base.size) as usize
                );

                let mut ty_def = TypeDef { attrs: ty_def.attrs, kind: ty_def.kind.clone() };
                let TypeKind::SpvInst { spv_inst, type_and_const_inputs, value_lowering } =
                    &mut ty_def.kind
                else {
                    unreachable!();
                };
                let TypeOrConst::Type(last_field_type) = type_and_const_inputs.last_mut().unwrap()
                else {
                    unreachable!();
                };

                *last_field_type = {
                    let ty_def = &cx[*last_field_type];
                    let mut ty_def = TypeDef { attrs: ty_def.attrs, kind: ty_def.kind.clone() };

                    let TypeKind::SpvInst { spv_inst, type_and_const_inputs, value_lowering } =
                        &mut ty_def.kind
                    else {
                        unreachable!();
                    };
                    assert!(spv_inst.opcode == self.wk.OpTypeRuntimeArray);
                    spv_inst.opcode = self.wk.OpTypeArray;
                    type_and_const_inputs
                        .push(TypeOrConst::Const(cx.intern(scalar::Const::from_u32(dyn_len))));
                    *value_lowering = spv::ValueLowering::Disaggregate(
                        spv::AggregateShape::compute(cx, spv_inst, type_and_const_inputs).unwrap(),
                    );
                    cx.intern(ty_def)
                };

                *value_lowering = spv::ValueLowering::Disaggregate(
                    spv::AggregateShape::compute(cx, spv_inst, type_and_const_inputs).unwrap(),
                );
                ty = cx.intern(ty_def);
            }

            // FIXME(eddyb) avoid the `ty` vs `init_ty` confusion, and maybe
            // even call `layout_of` only once (also used by the `if let` above).
            let layout = match self.layout_cache.layout_of(ty) {
                Ok(spirt::mem::layout::TypeLayout::Handle(handle)) => match handle {
                    spirt::mem::shapes::Handle::Opaque(_) => todo!(),
                    spirt::mem::shapes::Handle::Buffer(_, buf_layout) => buf_layout,
                },
                Ok(spirt::mem::layout::TypeLayout::HandleArray(..)) => todo!(),
                Ok(spirt::mem::layout::TypeLayout::Concrete(_)) => todo!(),
                Err(_) => todo!(),
            };

            // HACK(eddyb) a lot of this is duplicated from `qptr::lift`.
            // FIXME(eddyb) consider some kind of helper (maybe for `AllocData`?)
            // which relates leaves to their offsets, without touching layouts directly.
            let mut leaves = vec![];
            let result = layout.deeply_flatten_if(
                0,
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
                    let leaf_offset = u32::try_from(leaf_offset).unwrap();

                    let leaf_size = NonZeroU32::new(leaf.mem_layout.fixed_base.size).unwrap();

                    // FIXME(eddyb) avoid out-of-bounds panics with malformed layouts
                    // (and/or guarantee certain invariants in layouts that didn't error).
                    let bytes = &init_bytes
                        [(leaf_offset as usize)..((leaf_offset + leaf_size.get()) as usize)];

                    let mut total_read_scalar_size = 0;
                    let mut read_next_scalar = |leaf_scalar_type: scalar::Type| {
                        let byte_len = match leaf_scalar_type {
                            scalar::Type::Bool => {
                                self.layout_cache.config.abstract_bool_size_align.0
                            }
                            scalar::Type::SInt(_)
                            | scalar::Type::UInt(_)
                            | scalar::Type::Float(_) => {
                                let bit_width = leaf_scalar_type.bit_width();
                                assert_eq!(bit_width % 8, 0);
                                bit_width / 8
                            }
                        } as usize;

                        let mut copied_bytes = [0; 16];
                        copied_bytes[..byte_len]
                            .copy_from_slice(&bytes[total_read_scalar_size..][..byte_len]);
                        if self.layout_cache.config.is_big_endian {
                            copied_bytes[..byte_len].reverse();
                        }
                        let bits = u128::from_le_bytes(copied_bytes);

                        let leaf_scalar =
                            scalar::Const::try_from_bits(leaf_scalar_type, bits).unwrap();

                        total_read_scalar_size += byte_len;

                        self.eval_scalar_const(leaf_scalar)
                    };

                    let dyn_leaf = match cx[leaf.original_type].kind {
                        TypeKind::Scalar(ty) => DynLeaf::Scalar(read_next_scalar(ty)),
                        TypeKind::Vector(ty) => DynLeaf::Vector(
                            (0..ty.elem_count.get()).map(|_| read_next_scalar(ty.elem)).collect(),
                        ),
                        _ => todo!(),
                    };

                    assert_eq!(total_read_scalar_size, bytes.len());

                    leaves.push(self.new_val(leaf.original_type, dyn_leaf));

                    Ok(())
                },
            );
            result.ok().unwrap();

            let expected_leaf_count = cx[layout.original_type].disaggregated_leaf_count();
            assert_eq!(leaves.len(), expected_leaf_count);

            return AllocData {
                ty: layout.original_type,
                leaves,
                globally_shared: match bind_slot {
                    BindSlot::PushConstant | BindSlot::StorageBuffer { .. } => true,
                },
            };
        }

        AllocData {
            ty: init_ty,
            leaves: init_ty
                .disaggregated_leaf_types(cx)
                .map(|leaf_type| self.eval_undef_const(leaf_type))
                .collect(),

            // FIXME(eddyb) check the storage class.
            globally_shared: false,
        }
    }

    #[track_caller]
    fn new_val(&self, ty: Type, kind: DynLeaf) -> DynVal {
        let cx = self.cx();

        let type_def = &cx[ty];
        let valid = match &type_def.kind {
            &TypeKind::Scalar(expected) => match &kind {
                DynLeaf::Scalar(DynScalar(found, found_data)) if expected == *found => {
                    matches!(
                        (expected.bit_width(), found_data),
                        (_, DynScalarData::Undef)
                            | (1, DynScalarData::Bool(_))
                            | (8, DynScalarData::B8(_))
                            | (16, DynScalarData::B16(_))
                            | (32, DynScalarData::B32(_))
                            | (64, DynScalarData::B64(_))
                    )
                }
                _ => false,
            },
            TypeKind::Vector(_) => matches!(kind, DynLeaf::Vector(_)),
            TypeKind::QPtr => matches!(kind, DynLeaf::Ptr { .. }),
            TypeKind::Thunk => false,
            TypeKind::SpvInst { spv_inst, .. } => match kind {
                DynLeaf::Ptr { .. } => self.as_spv_ptr_type(ty).is_some(),
                DynLeaf::SpvVoidTypedValueFromExtInst => spv_inst.opcode == self.wk.OpTypeVoid,
                _ => false,
            },
            TypeKind::SpvStringLiteralForExtInst => {
                matches!(kind, DynLeaf::SpvStringLiteralForExtInst(_))
            }
        };
        if !valid {
            let variant = match kind {
                DynLeaf::Scalar(DynScalar(_, data)) => match data {
                    DynScalarData::Undef => "Scalar(Undef)",
                    DynScalarData::Bool(_) => "Scalar(Bool)",
                    DynScalarData::B8(_) => "Scalar(B8)",
                    DynScalarData::B16(_) => "Scalar(B16)",
                    DynScalarData::B32(_) => "Scalar(B32)",
                    DynScalarData::B64(_) => "Scalar(B64)",
                },
                DynLeaf::Vector(_) => "Vector",
                DynLeaf::Ptr { .. } => "Ptr",
                DynLeaf::SpvStringLiteralForExtInst(_) => "SpvStringLiteralForExtInst",
                DynLeaf::SpvVoidTypedValueFromExtInst => "SpvVoidTypedValueFromExtInst",
            };
            unreachable!(
                "`DynLeaf::{variant}` invalid for type `{}`",
                spirt::print::Plan::for_root(cx, &ty).pretty_print()
            )
        }

        let uniq_id = self.next_val_uniq_id.get();
        self.next_val_uniq_id.set(uniq_id.checked_add(1).unwrap());

        DynVal { ty, kind, uniq_id }
    }

    // FIXME(eddyb) should this be a method on `AttrSet`?
    pub fn get_spv_attr(
        &self,
        attrs: AttrSet,
        opcode: spv::spec::Opcode,
        operand_kind: spv::spec::OperandKind,
        operand: u32,
    ) -> Option<&'a [spv::Imm]> {
        for attr in &self.cx()[attrs].attrs {
            if let Attr::SpvAnnotation(spv_inst) = attr
                && spv_inst.opcode == opcode
                && let Some(imms) =
                    spv_inst.imms.strip_prefix(&[spv::Imm::Short(operand_kind, operand)])
            {
                return Some(imms);
            }
        }

        None
    }

    fn as_spv_ptr_type(&self, ty: Type) -> Option<(AddrSpace, Type)> {
        match &self.cx()[ty].kind {
            TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
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

    fn modify_spv_ptr_type(&self, ptr_type: Type, new_pointee_type: Type) -> Type {
        let cx = self.cx();
        let ty_def = &cx[ptr_type];
        let mut ty_def = TypeDef { attrs: ty_def.attrs, kind: ty_def.kind.clone() };
        match &mut ty_def.kind {
            TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
                if spv_inst.opcode == self.wk.OpTypePointer =>
            {
                type_and_const_inputs[0] = TypeOrConst::Type(new_pointee_type);
                cx.intern(ty_def)
            }
            _ => unreachable!(),
        }
    }
}
