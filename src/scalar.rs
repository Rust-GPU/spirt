//! Scalar (`bool`, integer, and floating-point) types and associated functionality.
//!
//! **Note**: pointers are never scalars (like SPIR-V, but unlike other IRs).

use arrayvec::ArrayVec;
use itertools::Itertools;

// HACK(eddyb) this could be some `struct` with private fields, but this `enum`
// is only 2 bytes in size, and has better ergonomics overall.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Bool,
    SInt(IntWidth),
    UInt(IntWidth),

    // FIXME(eddyb) SPIR-V added a "Floating Point Encoding" optional operand
    // to `OpTypeFloat`, for non-IEEE floating-point formats, find a way to
    // also support those here (maybe replacing `FloatWidth` entirely?).
    Float(FloatWidth),
}

impl Type {
    // HACK(eddyb) only common widths, as a convenience, expand as-needed.
    pub const S32: Type = Type::SInt(IntWidth::I32);
    pub const U32: Type = Type::UInt(IntWidth::I32);
    pub const F16: Type = Type::Float(FloatWidth::F16);
    pub const F32: Type = Type::Float(FloatWidth::F32);
    pub const F64: Type = Type::Float(FloatWidth::F64);

    pub const fn bit_width(self) -> u32 {
        match self {
            Type::Bool => 1,
            Type::SInt(w) | Type::UInt(w) => w.bits(),
            Type::Float(w) => w.bits(),
        }
    }
}

/// Bit-width of a supported integer type (only power-of-two multiples of a byte).
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct IntWidth {
    // HACK(eddyb) this is so compact that only 3 bits of this byte are used
    // to encode integer types from `i8` to `i128`, and so `Type` could all fit
    // in one byte, but that'd need a new `enum` for `Bool`/`{S,U}Int`/`Float`.
    log2_bytes: u8,
}

impl IntWidth {
    pub const I8: Self = Self::try_from_bits_unwrap(8);
    pub const I16: Self = Self::try_from_bits_unwrap(16);
    pub const I32: Self = Self::try_from_bits_unwrap(32);
    pub const I64: Self = Self::try_from_bits_unwrap(64);
    pub const I128: Self = Self::try_from_bits_unwrap(128);

    // FIXME(eddyb) remove when `Option::unwrap` is stabilized.
    const fn try_from_bits_unwrap(bits: u32) -> Self {
        match Self::try_from_bits(bits) {
            Some(w) => w,
            None => unreachable!(),
        }
    }

    pub const fn try_from_bits(bits: u32) -> Option<Self> {
        if !bits.is_multiple_of(8) {
            return None;
        }
        let bytes = bits / 8;
        match bytes.checked_ilog2() {
            Some(log2_bytes_u32) => {
                let log2_bytes = log2_bytes_u32 as u8;
                assert!(log2_bytes as u32 == log2_bytes_u32);
                Some(Self { log2_bytes })
            }
            None => None,
        }
    }

    pub const fn bits(self) -> u32 {
        8 * (1 << self.log2_bytes)
    }
}

/// Bit-width of a supported floating-point type (only power-of-two multiples of a byte).
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct FloatWidth(IntWidth);

impl FloatWidth {
    pub const F16: Self = Self::try_from_bits_unwrap(16);
    pub const F32: Self = Self::try_from_bits_unwrap(32);
    pub const F64: Self = Self::try_from_bits_unwrap(64);

    // FIXME(eddyb) remove when `Option::unwrap` is stabilized.
    const fn try_from_bits_unwrap(bits: u32) -> Self {
        match Self::try_from_bits(bits) {
            Some(w) => w,
            None => unreachable!(),
        }
    }

    pub const fn try_from_bits(bits: u32) -> Option<Self> {
        match IntWidth::try_from_bits(bits) {
            Some(w) => Some(Self(w)),
            None => None,
        }
    }

    pub const fn bits(self) -> u32 {
        self.0.bits()
    }
}

// FIXME(eddyb) document the 128-bit limitations.
// HACK(eddyb) `(Type, u128)` would waste almost half its size on padding, and
// packing will only impact accessing the `bits`, while allowing e.g. being
// wrapped in an outer `enum`, before reaching the same size as `(u128, u128)`.
#[repr(Rust, packed)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Const {
    ty: Type,
    bits: u128,
}

impl Const {
    pub const FALSE: Const = Const::from_bool(false);
    pub const TRUE: Const = Const::from_bool(true);

    // FIXME(eddyb) document the panic conditions.
    // FIXME(eddyb) make this public?
    const fn from_bits_trunc(ty: Type, bits: u128) -> Const {
        // FIXME(eddyb) this ensures `Const`s cannot be created when that could
        // potentially need more than 128 bits for e.g. constant-folding.
        let width = ty.bit_width();
        assert!(width <= 128);

        Const { ty, bits: bits & (!0u128 >> (128 - width)) }
    }

    // FIXME(eddyb) document the panic conditions.
    pub const fn from_bits(ty: Type, bits: u128) -> Const {
        let ct_trunc = Const::from_bits_trunc(ty, bits);
        assert!(ct_trunc.bits == bits);
        ct_trunc
    }

    pub const fn try_from_bits(ty: Type, bits: u128) -> Option<Const> {
        let ct_trunc = Const::from_bits_trunc(ty, bits);
        if ct_trunc.bits == bits { Some(ct_trunc) } else { None }
    }

    pub const fn from_bool(v: bool) -> Const {
        Const::from_bits(Type::Bool, v as u128)
    }

    pub const fn from_u32(v: u32) -> Const {
        Const::from_bits(Type::U32, v as u128)
    }

    /// Returns `Some(ct)` iff `ty` is `{S,U}Int` and can represent `v: i128`
    /// (i.e. `ct` has the same sign and absolute value as `v` does).
    pub fn int_try_from_i128(ty: Type, v: i128) -> Option<Const> {
        let ct_trunc = Const::from_bits_trunc(ty, v as u128);
        (ct_trunc.int_as_i128() == Some(v)).then_some(ct_trunc)
    }

    pub const fn ty(&self) -> Type {
        self.ty
    }

    pub const fn bits(&self) -> u128 {
        self.bits
    }

    // FIXME(eddyb) make this public?
    fn try_bit_cast_to(&self, ty: Type) -> Option<Const> {
        (self.ty.bit_width() == ty.bit_width()).then_some(Const { ty, ..*self })
    }

    /// Returns `Some(v)` iff `self` is `{S,U}Int` and representable by `v: i128`
    /// (i.e. `self` has the same sign and absolute value as `v` does).
    pub fn int_as_i128(&self) -> Option<i128> {
        match self.ty {
            Type::Bool | Type::Float(_) => None,
            Type::SInt(_) => {
                let width = self.ty.bit_width();
                Some((self.bits as i128) << (128 - width) >> (128 - width))
            }
            Type::UInt(_) => self.bits.try_into().ok(),
        }
    }

    /// Returns `Some(v)` iff `self` is `{S,U}Int` and representable by `v: u128`
    /// (i.e. `self` is positive and has the same absolute value as `v` does).
    pub fn int_as_u128(&self) -> Option<u128> {
        match self.ty {
            Type::Bool | Type::Float(_) => None,
            Type::SInt(_) => self.int_as_i128()?.try_into().ok(),
            Type::UInt(_) => Some(self.bits),
        }
    }

    /// Returns `Some(v)` iff `self` is `{S,U}Int` and representable by `v: i32`
    /// (i.e. `self` has the same sign and absolute value as `v` does).
    pub fn int_as_i32(&self) -> Option<i32> {
        self.int_as_i128()?.try_into().ok()
    }

    /// Returns `Some(v)` iff `self` is `{S,U}Int` and representable by `v: u32`
    /// (i.e. `self` is positive and has the same absolute value as `v` does).
    pub fn int_as_u32(&self) -> Option<u32> {
        self.int_as_u128()?.try_into().ok()
    }
}

/// Pure operations with scalar inputs and outputs.
//
// FIXME(eddyb) these are not some "perfect" grouping, but allow for more
// flexibility in users of this `enum` (and its component `enum`s).
#[derive(Copy, Clone, PartialEq, Eq, Hash, derive_more::From)]
pub enum Op {
    BoolUnary(BoolUnOp),
    BoolBinary(BoolBinOp),

    IntUnary(IntUnOp),
    IntBinary(IntBinOp),

    FloatUnary(FloatUnOp),
    FloatBinary(FloatBinOp),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum BoolUnOp {
    Not,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum BoolBinOp {
    Eq,
    // FIXME(eddyb) should this be `Xor` instead?
    Ne,
    Or,
    And,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum IntUnOp {
    Neg,
    Not,
    CountOnes,

    // FIXME(eddyb) ideally `Trunc` should be separated and common.
    TruncOrZeroExtend,
    TruncOrSignExtend,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum IntBinOp {
    // I×I→I
    Add,
    Sub,
    Mul,
    DivU,
    DivS,
    ModU,
    RemS,
    ModS,
    ShrU,
    ShrS,
    Shl,
    Or,
    Xor,
    And,

    // I×I→I×I
    CarryingAdd,
    BorrowingSub,
    WideningMulU,
    WideningMulS,

    // I×I→B
    Eq,
    Ne,
    // FIXME(eddyb) deduplicate between signed and unsigned.
    GtU,
    GtS,
    GeU,
    GeS,
    LtU,
    LtS,
    LeU,
    LeS,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum FloatUnOp {
    // F→F
    Neg,

    // F→B
    IsNan,
    IsInf,

    // FIXME(eddyb) these are a complicated mix of signatures.
    FromUInt,
    FromSInt,
    ToUInt,
    ToSInt,
    Convert,
    QuantizeAsF16,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum FloatBinOp {
    // F×F→F
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Mod,

    // F×F→B
    Cmp(FloatCmp),
    // FIXME(eddyb) this doesn't properly convey that this is effectively the
    // boolean flip of the opposite comparison, e.g. `CmpOrUnord(Ge)` is really
    // a fused version of `Not(Cmp(Lt))`, because `x < y` is never `true` for
    // unordered `x` and `y` (i.e. `PartialOrd::partial_cmp(x, y) == None`),
    // but that maps to `!(x < y)` always being `true` for unordered `x` and `y`,
    // and thus `x >= y` is only equivalent for the ordered cases.
    CmpOrUnord(FloatCmp),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum FloatCmp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

pub enum EvalError {
    // FIXME(eddyb) provide more detail.
    OpSignatureMismatch,

    UnsupportedFloatWidth(FloatWidth),

    // FIXME(eddyb) is there a better name for this?
    FloatException,

    // FIXME(eddyb) not exactly an error, and can be replaced with `undef`.
    PoisonOutput,

    UndefinedBehavior { cause: &'static str },
}

impl Op {
    pub fn output_count(self) -> usize {
        match self {
            Op::IntBinary(op) => op.output_count(),
            _ => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Op::BoolUnary(op) => op.name(),
            Op::BoolBinary(op) => op.name(),

            Op::IntUnary(op) => op.name(),
            Op::IntBinary(op) => op.name(),

            Op::FloatUnary(op) => op.name(),
            Op::FloatBinary(op) => op.name(),
        }
    }

    pub fn try_eval(
        self,
        inputs: &[Const],
        output_types: &[Type],
    ) -> Result<ArrayVec<Const, 2>, EvalError> {
        let single_output = match (self, inputs, output_types) {
            (Op::BoolUnary(op), &[Const { ty: Type::Bool, bits: x @ (0..=1) }], &[Type::Bool]) => {
                Const::from_bool(op.eval(x != 0))
            }
            (
                Op::BoolBinary(op),
                &[
                    Const { ty: Type::Bool, bits: a @ (0..=1) },
                    Const { ty: Type::Bool, bits: b @ (0..=1) },
                ],
                &[Type::Bool],
            ) => Const::from_bool(op.eval(a != 0, b != 0)),
            (Op::IntUnary(op), &[x], &[output_type]) => op.try_eval(x, output_type)?,
            (Op::IntBinary(op), &[a, b], _) => return op.try_eval(a, b, output_types),
            (Op::FloatUnary(op), &[x], &[output_type]) => op.try_eval(x, output_type)?,
            (Op::FloatBinary(op), &[a, b], &[output_type]) => op.try_eval(a, b, output_type)?,
            _ => return Err(EvalError::OpSignatureMismatch),
        };
        Ok([single_output].into_iter().collect())
    }
}

impl BoolUnOp {
    pub fn name(self) -> &'static str {
        match self {
            BoolUnOp::Not => "bool.not",
        }
    }

    pub fn eval(self, x: bool) -> bool {
        match self {
            BoolUnOp::Not => !x,
        }
    }
}

impl BoolBinOp {
    pub fn name(self) -> &'static str {
        match self {
            BoolBinOp::Eq => "bool.eq",
            BoolBinOp::Ne => "bool.ne",
            BoolBinOp::Or => "bool.or",
            BoolBinOp::And => "bool.and",
        }
    }

    pub fn eval(self, a: bool, b: bool) -> bool {
        match self {
            BoolBinOp::Eq => a == b,
            BoolBinOp::Ne => a != b,
            BoolBinOp::Or => a | b,
            BoolBinOp::And => a & b,
        }
    }
}

impl IntUnOp {
    pub fn name(self) -> &'static str {
        match self {
            IntUnOp::Neg => "i.neg",
            IntUnOp::Not => "i.not",
            IntUnOp::CountOnes => "i.count_ones",

            IntUnOp::TruncOrZeroExtend => "u.trunc_or_zext",
            IntUnOp::TruncOrSignExtend => "s.trunc_or_sext",
        }
    }

    pub fn try_eval(self, x: Const, output_type: Type) -> Result<Const, EvalError> {
        // FIXME(eddyb) try to dedup these helpers with `IntBinOp`.
        let int_width = |ty| match ty {
            Type::UInt(w) | Type::SInt(w) => Ok(w),
            _ => Err(EvalError::OpSignatureMismatch),
        };
        let output_width = int_width(output_type)?;

        let x_width = int_width(x.ty())?;
        let (x, x_s) =
            (x.bits(), x.try_bit_cast_to(Type::SInt(x_width)).unwrap().int_as_i128().unwrap());

        let valid_widths = output_width == x_width
            || matches!(
                self,
                IntUnOp::CountOnes | IntUnOp::TruncOrZeroExtend | IntUnOp::TruncOrSignExtend
            );
        if !valid_widths {
            return Err(EvalError::OpSignatureMismatch);
        }

        let output_bits = match self {
            IntUnOp::Neg => x_s.wrapping_neg() as u128,
            IntUnOp::Not => !x,
            IntUnOp::CountOnes => x.count_ones().into(),
            IntUnOp::TruncOrZeroExtend => x,
            IntUnOp::TruncOrSignExtend => x_s as u128,
        };
        Ok(Const::from_bits_trunc(output_type, output_bits))
    }
}

impl IntBinOp {
    pub fn output_count(self) -> usize {
        // FIXME(eddyb) should these 4 go into a different `enum`?
        match self {
            IntBinOp::CarryingAdd
            | IntBinOp::BorrowingSub
            | IntBinOp::WideningMulU
            | IntBinOp::WideningMulS => 2,
            _ => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            IntBinOp::Add => "i.add",
            IntBinOp::Sub => "i.sub",
            IntBinOp::Mul => "i.mul",
            IntBinOp::DivU => "u.div",
            IntBinOp::DivS => "s.div",
            IntBinOp::ModU => "u.mod",
            IntBinOp::RemS => "s.rem",
            IntBinOp::ModS => "s.mod",
            IntBinOp::ShrU => "u.shr",
            IntBinOp::ShrS => "s.shr",
            IntBinOp::Shl => "i.shl",
            IntBinOp::Or => "i.or",
            IntBinOp::Xor => "i.xor",
            IntBinOp::And => "i.and",
            IntBinOp::CarryingAdd => "i.carrying_add",
            IntBinOp::BorrowingSub => "i.borrowing_sub",
            IntBinOp::WideningMulU => "u.widening_mul",
            IntBinOp::WideningMulS => "s.widening_mul",
            IntBinOp::Eq => "i.eq",
            IntBinOp::Ne => "i.ne",
            IntBinOp::GtU => "u.gt",
            IntBinOp::GtS => "s.gt",
            IntBinOp::GeU => "u.ge",
            IntBinOp::GeS => "s.ge",
            IntBinOp::LtU => "u.lt",
            IntBinOp::LtS => "s.lt",
            IntBinOp::LeU => "u.le",
            IntBinOp::LeS => "s.le",
        }
    }

    pub fn try_eval(
        self,
        a: Const,
        b: Const,
        output_types: &[Type],
    ) -> Result<ArrayVec<Const, 2>, EvalError> {
        let output_type = output_types
            .iter()
            .copied()
            .dedup()
            .exactly_one()
            .ok()
            .filter(|_| output_types.len() == self.output_count())
            .ok_or(EvalError::OpSignatureMismatch)?;

        // FIXME(eddyb) try to dedup these helpers with `IntUnOp`.
        let int_width = |ty| match ty {
            Type::UInt(w) | Type::SInt(w) => Ok(w),
            _ => Err(EvalError::OpSignatureMismatch),
        };
        let output_width = match self {
            // FIXME(eddyb) should comparisons be handled separately?
            IntBinOp::Eq
            | IntBinOp::Ne
            | IntBinOp::GtU
            | IntBinOp::GtS
            | IntBinOp::GeU
            | IntBinOp::GeS
            | IntBinOp::LtU
            | IntBinOp::LtS
            | IntBinOp::LeU
            | IntBinOp::LeS => None,

            _ => Some(int_width(output_type)?),
        };

        let as_u128_i128 = |x: Const| {
            let x_width = int_width(x.ty())?;
            Ok((
                x_width,
                x.bits(),
                x.try_bit_cast_to(Type::SInt(x_width)).unwrap().int_as_i128().unwrap(),
            ))
        };
        let (a_width, a, a_s) = as_u128_i128(a)?;
        let (b_width, b, b_s) = as_u128_i128(b)?;

        let valid_widths = output_width.is_none_or(|w| w == a_width)
            && (a_width == b_width
                || matches!(self, IntBinOp::ShrU | IntBinOp::ShrS | IntBinOp::Shl));
        if !valid_widths {
            return Err(EvalError::OpSignatureMismatch);
        }

        let div_ub_err = EvalError::UndefinedBehavior {
            cause: if b_s == 0 { "division by 0" } else { "signed division overflow" },
        };
        let b_as_shift_amount =
            || u32::try_from(b).ok().filter(|&b| b < a_width.bits()).ok_or(EvalError::PoisonOutput);

        // FIXME(eddyb) replace with `u128::widening_mul` when it stabilizes.
        fn u128_widening_mul(a: u128, b: u128) -> (u128, u128) {
            // HACK(eddyb) the code below extracts `lo` and `hi`,
            // such that `lo + 2¹²⁸hi` is equal to this expansion of `a · b`:
            // `(al + 2⁶⁴ah) · (bl + 2⁶⁴bh) = al·bl + 2⁶⁴(al·bh + ah·bl) + 2¹²⁸(ah·bh)`
            let [(al, ah), (bl, bh)] = [a, b].map(|x| (x as u64 as u128, x >> 64));
            let [[al_bl, al_bh], [ah_bl, ah_bh]] =
                [al, ah].map(|a| [bl, bh].map(|b| a.checked_mul(b).unwrap()));

            let (mid, mid_carry) = al_bh.overflowing_add(ah_bl);
            let (lo, lo_carry) = al_bl.overflowing_add(mid << 64);
            let hi = [ah_bh, mid >> 64, (mid_carry as u128) << 64, lo_carry as u128]
                .into_iter()
                .reduce(|a, b| a.checked_add(b).unwrap())
                .unwrap();

            assert_eq!(lo, a.wrapping_mul(b));

            (lo, hi)
        }

        // FIXME(eddyb) replace with `i128::widening_mul` when it stabilizes.
        fn i128_widening_mul(a: i128, b: i128) -> (u128, i128) {
            // HACK(eddyb) to avoid duplication and signedness subtleties,
            // the sign is handled on top of the unsigned implementation above.
            let (abs_lo, abs_hi) = u128_widening_mul(a.unsigned_abs(), b.unsigned_abs());
            if a.signum() * b.signum() == -1 {
                // HACK(eddyb) `-x` is equivalent to `(!x).wrapping_add(1)`,
                // which can be directly applied to a double-width integer.
                let (lo, lo_carry) = (!abs_lo).overflowing_add(1);
                (lo, (!abs_hi).wrapping_add(lo_carry as u128) as i128)
            } else {
                (abs_lo, abs_hi as i128)
            }
        }

        let wide_result = |[lo, hi]: [u128; 2]| {
            // HACK(eddyb) `lo + 2¹²⁸hi` form a 256-bit result, but the true
            // result for an N-bit operation will only match those two halves
            // for N=128, for smaller N both halves can be found in `lo`.
            let width = output_width.unwrap().bits();
            let hi = if width == 128 || {
                // HACK(eddyb) because subtraction overflow is centered around `0`,
                // and not `2^N`, the 128-bit `hi` is already the correct top half,
                // and it's not obvious how to otherwise get that correct value,
                // without this (otherwise quite annoying) special-case.
                self == IntBinOp::BorrowingSub
            } {
                hi
            } else {
                lo.checked_shr(width).unwrap()
            };

            Ok([lo, hi].map(|x| Const::from_bits_trunc(output_type, x)).into_iter().collect())
        };

        // HACK(eddyb) can't trust `checked_{div,rem}` to handle the "MIN" part
        // correctly, because `iN::MIN as i128 != i128::MIN` for `N < 128`.
        if let IntBinOp::DivS | IntBinOp::RemS | IntBinOp::ModS = self
            && a_s == -1 << (a_width.bits() - 1)
            && b_s == -1
        {
            return Err(div_ub_err);
        }

        let output_bits = match self {
            IntBinOp::Add => a.wrapping_add(b),
            IntBinOp::Sub => a.wrapping_sub(b),
            IntBinOp::Mul => a.wrapping_mul(b),
            IntBinOp::DivU => a.checked_div(b).ok_or(div_ub_err)?,
            IntBinOp::DivS => a_s.checked_div(b_s).ok_or(div_ub_err)? as u128,
            IntBinOp::ModU => a.checked_rem(b).ok_or(div_ub_err)?,
            IntBinOp::RemS => a_s.checked_rem(b_s).ok_or(div_ub_err)? as u128,
            IntBinOp::ModS => {
                let rem_s = a_s.checked_rem(b_s).ok_or(div_ub_err)?;
                let mod_s = if rem_s.signum() * b_s.signum() == -1 {
                    // |b_s| > |rem_s|, so |b_s + rem_s| = |b_s| - |rem_s|, and
                    // the sum will have sign of `b_s` (as required by SPIR-V).
                    rem_s.checked_add(b_s).unwrap()
                } else {
                    rem_s
                };
                mod_s as u128
            }
            IntBinOp::ShrU => a.checked_shr(b_as_shift_amount()?).unwrap(),
            IntBinOp::ShrS => a_s.checked_shr(b_as_shift_amount()?).unwrap() as u128,
            IntBinOp::Shl => a.checked_shl(b_as_shift_amount()?).unwrap(),
            IntBinOp::Or => a | b,
            IntBinOp::Xor => a ^ b,
            IntBinOp::And => a & b,
            IntBinOp::CarryingAdd => {
                let (lo, hi) = a.overflowing_add(b);
                return wide_result([lo, hi as u128]);
            }
            IntBinOp::BorrowingSub => {
                let (lo, hi) = a.overflowing_sub(b);
                return wide_result([lo, hi as u128]);
            }
            IntBinOp::WideningMulU => {
                let (lo, hi) = u128_widening_mul(a, b);
                return wide_result([lo, hi]);
            }
            IntBinOp::WideningMulS => {
                let (lo, hi) = i128_widening_mul(a_s, b_s);
                return wide_result([lo, hi as u128]);
            }
            IntBinOp::Eq => (a == b) as u128,
            IntBinOp::Ne => (a != b) as u128,
            IntBinOp::GtU => (a > b) as u128,
            IntBinOp::GtS => (a_s > b_s) as u128,
            IntBinOp::GeU => (a >= b) as u128,
            IntBinOp::GeS => (a_s >= b_s) as u128,
            IntBinOp::LtU => (a < b) as u128,
            IntBinOp::LtS => (a_s < b_s) as u128,
            IntBinOp::LeU => (a <= b) as u128,
            IntBinOp::LeS => (a_s <= b_s) as u128,
        };
        Ok([Const::from_bits_trunc(output_type, output_bits)].into_iter().collect())
    }
}

impl FloatUnOp {
    pub fn name(self) -> &'static str {
        match self {
            FloatUnOp::Neg => "f.neg",

            FloatUnOp::IsNan => "f.is_nan",
            FloatUnOp::IsInf => "f.is_inf",

            FloatUnOp::FromUInt => "f.from_uint",
            FloatUnOp::FromSInt => "f.from_sint",
            FloatUnOp::ToUInt => "f.to_uint",
            FloatUnOp::ToSInt => "f.to_sint",
            FloatUnOp::Convert => "f.convert",
            FloatUnOp::QuantizeAsF16 => "f.quantize_as_f16",
        }
    }

    pub fn try_eval(self, x: Const, output_type: Type) -> Result<Const, EvalError> {
        let float_type = match self {
            FloatUnOp::Neg
            | FloatUnOp::IsNan
            | FloatUnOp::IsInf
            | FloatUnOp::ToUInt
            | FloatUnOp::ToSInt
            | FloatUnOp::Convert => x.ty(),
            FloatUnOp::FromUInt | FloatUnOp::FromSInt => output_type,
            FloatUnOp::QuantizeAsF16 => Type::F32,
        };

        match float_type {
            Type::F16 => self.try_eval_specialized::<rustc_apfloat::ieee::Half>(x, output_type),
            Type::F32 => self.try_eval_specialized::<rustc_apfloat::ieee::Single>(x, output_type),
            Type::F64 => self.try_eval_specialized::<rustc_apfloat::ieee::Double>(x, output_type),
            Type::Float(w) => Err(EvalError::UnsupportedFloatWidth(w)),
            _ => Err(EvalError::OpSignatureMismatch),
        }
    }

    fn try_eval_specialized<F>(self, x: Const, output_type: Type) -> Result<Const, EvalError>
    where
        F: rustc_apfloat::Float
            + rustc_apfloat::FloatConvert<rustc_apfloat::ieee::Half>
            + rustc_apfloat::FloatConvert<rustc_apfloat::ieee::Single>
            + rustc_apfloat::FloatConvert<rustc_apfloat::ieee::Double>,
        rustc_apfloat::ieee::Half: rustc_apfloat::FloatConvert<F>,
    {
        use rustc_apfloat::{Float, FloatConvert, Status, StatusAnd};

        // HACK(eddyb) more convenient conversion helper.
        fn convert<T: FloatConvert<U>, U: Float>(x: T) -> StatusAnd<U> {
            x.convert(&mut false)
        }

        let int_width = |ty| match ty {
            Type::UInt(w) | Type::SInt(w) => Ok(w),
            _ => Err(EvalError::OpSignatureMismatch),
        };

        // FIXME(eddyb) try to dedup these helpers with `FloatBinOp`.
        let expected_float_type =
            Type::Float(FloatWidth::try_from_bits(F::BITS.try_into().unwrap()).unwrap());
        let f_from_const = |x: Const| {
            if x.ty() != expected_float_type {
                return Err(EvalError::OpSignatureMismatch);
            }
            Ok(F::from_bits(x.bits()))
        };
        let const_f = |x: F| Const::from_bits(expected_float_type, x.to_bits());
        let const_bool = |x: bool| Const::from_bits(Type::Bool, x as u128);

        let status_and_output = match self {
            FloatUnOp::Neg => Status::OK.and(-f_from_const(x)?).map(const_f),
            FloatUnOp::IsNan => Status::OK.and(f_from_const(x)?.is_nan()).map(const_bool),
            FloatUnOp::IsInf => Status::OK.and(f_from_const(x)?.is_infinite()).map(const_bool),
            FloatUnOp::FromUInt => {
                F::from_u128(x.int_as_u128().ok_or(EvalError::OpSignatureMismatch)?).map(const_f)
            }
            FloatUnOp::FromSInt => {
                F::from_i128(x.int_as_i128().ok_or(EvalError::OpSignatureMismatch)?).map(const_f)
            }
            FloatUnOp::ToUInt => {
                let width = int_width(output_type)?;
                f_from_const(x)?
                    .to_u128(width.bits() as usize)
                    .map(|r| Const::from_bits(Type::UInt(width), r))
            }
            FloatUnOp::ToSInt => {
                let width = int_width(output_type)?;
                f_from_const(x)?
                    .to_i128(width.bits() as usize)
                    .map(|r| Const::int_try_from_i128(Type::SInt(width), r).unwrap())
            }
            FloatUnOp::Convert => {
                let x = f_from_const(x)?;
                let status_and_output_bits = match output_type {
                    Type::F16 => convert::<_, rustc_apfloat::ieee::Half>(x).map(|r| r.to_bits()),
                    Type::F32 => convert::<_, rustc_apfloat::ieee::Single>(x).map(|r| r.to_bits()),
                    Type::F64 => convert::<_, rustc_apfloat::ieee::Double>(x).map(|r| r.to_bits()),
                    Type::Float(w) => return Err(EvalError::UnsupportedFloatWidth(w)),
                    _ => return Err(EvalError::OpSignatureMismatch),
                };
                status_and_output_bits.map(|output_bits| Const::from_bits(output_type, output_bits))
            }
            FloatUnOp::QuantizeAsF16 => convert::<_, rustc_apfloat::ieee::Half>(f_from_const(x)?)
                .map(|x_f16| convert::<_, F>(x_f16).value)
                .map(const_f),
        };

        if status_and_output.status.intersects(Status::INVALID_OP | Status::DIV_BY_ZERO) {
            return Err(EvalError::FloatException);
        }

        let output = status_and_output.value;
        if output.ty() != output_type {
            return Err(EvalError::OpSignatureMismatch);
        }
        Ok(output)
    }
}

impl FloatBinOp {
    pub fn name(self) -> &'static str {
        match self {
            FloatBinOp::Add => "f.add",
            FloatBinOp::Sub => "f.sub",
            FloatBinOp::Mul => "f.mul",
            FloatBinOp::Div => "f.div",
            FloatBinOp::Rem => "f.rem",
            FloatBinOp::Mod => "f.mod",
            FloatBinOp::Cmp(FloatCmp::Eq) => "f.eq",
            FloatBinOp::Cmp(FloatCmp::Ne) => "f.ne",
            FloatBinOp::Cmp(FloatCmp::Lt) => "f.lt",
            FloatBinOp::Cmp(FloatCmp::Gt) => "f.gt",
            FloatBinOp::Cmp(FloatCmp::Le) => "f.le",
            FloatBinOp::Cmp(FloatCmp::Ge) => "f.ge",
            FloatBinOp::CmpOrUnord(FloatCmp::Eq) => "f.eq_or_unord",
            FloatBinOp::CmpOrUnord(FloatCmp::Ne) => "f.ne_or_unord",
            FloatBinOp::CmpOrUnord(FloatCmp::Lt) => "f.lt_or_unord",
            FloatBinOp::CmpOrUnord(FloatCmp::Gt) => "f.gt_or_unord",
            FloatBinOp::CmpOrUnord(FloatCmp::Le) => "f.le_or_unord",
            FloatBinOp::CmpOrUnord(FloatCmp::Ge) => "f.ge_or_unord",
        }
    }

    pub fn try_eval(self, a: Const, b: Const, output_type: Type) -> Result<Const, EvalError> {
        if a.ty() != b.ty() {
            return Err(EvalError::OpSignatureMismatch);
        }

        match a.ty() {
            Type::F16 => self.try_eval_specialized::<rustc_apfloat::ieee::Half>(a, b, output_type),
            Type::F32 => {
                self.try_eval_specialized::<rustc_apfloat::ieee::Single>(a, b, output_type)
            }
            Type::F64 => {
                self.try_eval_specialized::<rustc_apfloat::ieee::Double>(a, b, output_type)
            }
            Type::Float(w) => Err(EvalError::UnsupportedFloatWidth(w)),
            _ => Err(EvalError::OpSignatureMismatch),
        }
    }

    fn try_eval_specialized<F: rustc_apfloat::Float>(
        self,
        a: Const,
        b: Const,
        output_type: Type,
    ) -> Result<Const, EvalError> {
        use rustc_apfloat::Status;

        // FIXME(eddyb) try to dedup these helpers with `FloatBinOp`.
        let expected_float_type =
            Type::Float(FloatWidth::try_from_bits(F::BITS.try_into().unwrap()).unwrap());
        let f_from_const = |x: Const| {
            if x.ty() != expected_float_type {
                return Err(EvalError::OpSignatureMismatch);
            }
            Ok(F::from_bits(x.bits()))
        };
        let const_f = |x: F| Const::from_bits(expected_float_type, x.to_bits());
        let const_bool = |x: bool| Const::from_bits(Type::Bool, x as u128);

        let status_and_output = match self {
            FloatBinOp::Add => (f_from_const(a)? + f_from_const(b)?).map(const_f),
            FloatBinOp::Sub => (f_from_const(a)? - f_from_const(b)?).map(const_f),
            FloatBinOp::Mul => (f_from_const(a)? * f_from_const(b)?).map(const_f),
            FloatBinOp::Div => (f_from_const(a)? / f_from_const(b)?).map(const_f),
            FloatBinOp::Rem => (f_from_const(a)? % f_from_const(b)?).map(const_f),
            FloatBinOp::Mod => {
                let (a, b) = (f_from_const(a)?, f_from_const(b)?);
                (a % b)
                    .map(|rem| {
                        if !rem.is_zero() && rem.is_negative() != b.is_negative() {
                            // |b| > |rem|, so |b + rem| = |b| - |rem|, and the sum
                            // will have sign of `b` (as required by SPIR-V).
                            (rem + b).value
                        } else {
                            rem
                        }
                    })
                    .map(const_f)
            }
            FloatBinOp::Cmp(cmp) => {
                Status::OK.and(cmp.eval(&f_from_const(a)?, &f_from_const(b)?)).map(const_bool)
            }
            // HACK(eddyb) see comment on `FloatBinOp::CmpOrUnord` for an explanation.
            FloatBinOp::CmpOrUnord(cmp) => Status::OK
                .and((!cmp).eval(&f_from_const(a)?, &f_from_const(b)?))
                .map(|r| const_bool(!r)),
        };

        if status_and_output.status.intersects(Status::INVALID_OP | Status::DIV_BY_ZERO) {
            return Err(EvalError::FloatException);
        }

        let output = status_and_output.value;
        if output.ty() != output_type {
            return Err(EvalError::OpSignatureMismatch);
        }
        Ok(output)
    }
}

// HACK(eddyb) see comment on `FloatBinOp::CmpOrUnord` for why this "flipping"
// is useful - i.e. `FloatBinOp::CmpOrUnord(cmp)` is equivalent to first applying
// `FloatBinOp::Cmp(!cmp)` then passing its result to `BoolUnOp::Not`.
impl std::ops::Not for FloatCmp {
    type Output = FloatCmp;
    fn not(self) -> FloatCmp {
        match self {
            FloatCmp::Eq => FloatCmp::Ne,
            FloatCmp::Ne => FloatCmp::Eq,
            FloatCmp::Lt => FloatCmp::Ge,
            FloatCmp::Gt => FloatCmp::Le,
            FloatCmp::Le => FloatCmp::Gt,
            FloatCmp::Ge => FloatCmp::Lt,
        }
    }
}

impl FloatCmp {
    fn eval<T: PartialOrd>(self, a: &T, b: &T) -> bool {
        match self {
            FloatCmp::Eq => *a == *b,
            FloatCmp::Ne => *a != *b,
            FloatCmp::Lt => *a < *b,
            FloatCmp::Gt => *a > *b,
            FloatCmp::Le => *a <= *b,
            FloatCmp::Ge => *a >= *b,
        }
    }
}
