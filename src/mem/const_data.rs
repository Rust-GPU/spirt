//! Constant data efficiently mixing concrete bytes with symbolic values.

use crate::scalar;
use itertools::Itertools;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::ops::Range;
use std::{iter, mem};

/// Constant data "blob" or "chunk", where each byte can be part of:
/// - uninitialized areas (e.g. SPIR-V `OpUndef`)
/// - concrete data (i.e. `u8` values)
/// - symbolic values of type `V` (spanning some number of bytes)
///
/// This is similar to (and inspired by), [`rustc`'s `mir::interpret::Allocation`](
/// https://rustc-dev-guide.rust-lang.org/const-eval/interpret.html#memory),
/// which only has abstract pointers as symbolic values, encoded as "relocations"
/// (i.e. concrete data contains the respective offset for each abstract pointer,
/// whereas here the symbolic values are completely disjoint with concrete data).
#[derive(Clone)]
pub struct ConstData<V> {
    /// The bit `init[i / 64] & (1 << (i % 64))` is set iff byte offset `i` is
    /// initialized, either with concrete data, or as part of a symbolic value.
    //
    // FIXME(eddyb) come up with a centralized "bitset"/"bitvec" instead.
    init: Box<[u64]>,

    /// Concrete data bytes, with each byte only used when `init` indicates
    /// it is initialized *and* no symbolic value overlaps it. Unused bytes can
    /// have any values in `bytes`, as they're guaranteed to be always ignored.
    data: Box<[u8]>,

    /// Non-overlapping set of symbolic `V` values, forming an "overlay" on top
    /// of the concrete data bytes, with `syms[offset] = (size, value)`
    /// indicating bytes `offset..(offset + size)` are occupied by `value`.
    syms: BTreeMap<u32, (NonZeroU32, V)>,

    /// Largest symbolic value size, i.e. `syms.values().map(|(size, _)| size).max()`.
    //
    // FIXME(eddyb) this is only needed to help with scanning overlaps in `syms`,
    // and because there is no inherent limit on the size of symbolic values.
    max_sym_size: NonZeroU32,
}

/// One uniform "slice" of a `ConstData` (*not* mixing value categories).
#[derive(Clone)]
pub enum Part<'a, V> {
    Uninit {
        size: NonZeroU32,
    },
    Bytes(&'a [u8]),
    Symbolic {
        size: NonZeroU32,
        /// This is only the full `value` if `maybe_partial_slice == 0..size`.
        maybe_partial_slice: Range<u32>,
        value: V,
    },
}

impl<V> Part<'_, V> {
    // FIXME(eddyb) should there just be a common `size` field?
    pub fn size(&self) -> NonZeroU32 {
        match *self {
            Part::Uninit { size } | Part::Symbolic { size, .. } => size,
            Part::Bytes(bytes) => NonZeroU32::new(bytes.len().try_into().unwrap()).unwrap(),
        }
    }
}

/// Error type for write operations, emitted when they would otherwise cause a
/// partial overwrite of a symbolic value, if allowed to take effect.
#[derive(Debug)]
pub struct PartialSymbolicOverlap {
    pub offsets: Range<u32>,
}

// FIXME(eddyb) come up with a nicer abstraction for bitvecs, or use a crate.
fn bitrange_word_chunks(range: Range<u32>) -> (Range<usize>, impl Iterator<Item = Range<u8>>) {
    // HACK(eddyb) `/ 64` and `% 64` work directly for inclusive positions,
    // but it's more useful to be able to use `Range` in general.
    let (first, last) = (range.start, range.end - 1);
    let words = (first / 64)..((last / 64) + 1);
    (
        (words.start as usize)..(words.end as usize),
        words.map(move |i| {
            let [first_in_word, last_in_word] = [0, 63]
                .map(|offset_in_word| ((i * 64 + offset_in_word).clamp(first, last) % 64) as u8);
            first_in_word..(last_in_word + 1)
        }),
    )
}

impl<V: Clone> ConstData<V> {
    pub fn new(size: u32) -> Self {
        let size = size as usize;
        Self {
            init: vec![0; size.div_ceil(64)].into_boxed_slice(),
            data: vec![0; size].into_boxed_slice(),
            syms: BTreeMap::new(),
            max_sym_size: NonZeroU32::new(1).unwrap(),
        }
    }

    // HACK(eddyb) only used by `qptr::legalize` for fusing `GlobalVar`s.
    pub fn grow(&mut self, new_size: u32) {
        assert!(new_size >= self.size());
        let new_size = new_size as usize;

        let mut init = mem::take(&mut self.init).into_vec();
        let mut data = mem::take(&mut self.data).into_vec();
        init.extend((init.len()..new_size.div_ceil(64)).map(|_| 0));
        data.extend((data.len()..new_size).map(|_| 0));
        self.init = init.into_boxed_slice();
        self.data = data.into_boxed_slice();
    }

    pub fn size(&self) -> u32 {
        self.data.len() as u32
    }

    // HACK(eddyb) only needed for `visit`.
    pub fn used_symbolic_values(&self) -> impl Iterator<Item = &V> {
        self.syms.values().map(|(_, v)| v)
    }

    // HACK(eddyb) only needed for `transform`.
    pub fn used_symbolic_values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.syms.values_mut().map(|(_, v)| v)
    }

    pub fn read(&self, range: Range<u32>) -> impl Iterator<Item = Part<'_, V>> {
        // HACK(eddyb) trigger bounds-checking panics.
        let _ = &self.data[(range.start as usize)..(range.end as usize)];

        // HACK(eddyb) the range has to be extended backwards, because a partial
        // overlap could exist, i.e. `range.start` being in the middle of a value,
        // but then irrelevant values have to be ignored.
        let mut syms = self
            .syms
            .range(range.start.saturating_sub(self.max_sym_size.get() - 1)..range.end)
            .map(|(&offset, (size, value))| (offset..(offset + size.get()), value.clone()))
            .peekable();
        while let Some((sym_range, _)) = syms.peek() {
            if sym_range.end > range.start {
                break;
            }
            syms.next().unwrap();
        }

        let mut part_start = range.start;
        iter::from_fn(move || {
            if part_start >= range.end {
                return None;
            }
            let next_sym_range = syms.peek().cloned().map_or(range.end..range.end, |(r, _)| r);

            let max_part_end = if next_sym_range.contains(&part_start) {
                next_sym_range.end
            } else {
                next_sym_range.start
            };
            // FIXME(eddyb) come up with a nicer abstraction for bitvecs, or use a crate.
            let (part_is_init, part_size) = {
                let (words, word_bitslices) = bitrange_word_chunks(part_start..max_part_end);
                let mut init_runs = self.init[words]
                    .iter()
                    .zip_eq(word_bitslices)
                    .flat_map(|(&word, word_bitslice)| {
                        let mut remaining_word =
                            (word & (!0 >> (64 - word_bitslice.end))) >> word_bitslice.start;
                        let mut remaining_bit_len = word_bitslice.len() as u32;

                        iter::from_fn(move || {
                            if remaining_bit_len == 0 {
                                return None;
                            }
                            let is_set = (remaining_word & 1) != 0;
                            let run_len = if is_set {
                                remaining_word.trailing_ones()
                            } else {
                                remaining_word.trailing_zeros().min(remaining_bit_len)
                            };
                            // HACK(eddyb) work around overlong shifts.
                            remaining_word >>= 1;
                            remaining_word >>= run_len - 1;
                            remaining_bit_len -= run_len;
                            Some((is_set, NonZeroU32::new(run_len).unwrap()))
                        })
                    })
                    .coalesce(|(a, a_run), (b, b_run)| {
                        if a == b {
                            Ok((a, a_run.checked_add(b_run.get()).unwrap()))
                        } else {
                            Err(((a, a_run), (b, b_run)))
                        }
                    });

                init_runs.next().unwrap()
            };

            let part_end = part_start + part_size.get();
            let part = if !part_is_init {
                Part::Uninit { size: part_size }
            } else if next_sym_range.contains(&part_start) {
                let (sym_range, value) = syms.next().unwrap();
                // HACK(eddyb) ensure slicing is caused by `range`, *not* `init`.
                assert_eq!(
                    part_start..part_end,
                    sym_range.start.clamp(range.start, range.end)
                        ..sym_range.end.clamp(range.start, range.end)
                );
                Part::Symbolic {
                    size: NonZeroU32::new(sym_range.len() as u32).unwrap(),
                    maybe_partial_slice: (part_start - sym_range.start)
                        ..(part_end - sym_range.start),
                    value,
                }
            } else {
                Part::Bytes(&self.data[(part_start as usize)..(part_end as usize)])
            };
            part_start = part_end;
            Some(part)
        })
    }

    /// Helper for `write_bytes` and `write_symbolic`, which only modifies `self`
    /// (removing fully overwritten symbolic values, and setting `init` bits),
    /// when it can guarantee it will return `Ok(())` (i.e. after error checks).
    fn try_init(&mut self, range: Range<u32>) -> Result<(), PartialSymbolicOverlap> {
        // HACK(eddyb) trigger bounds-checking panics.
        let _ = &self.data[(range.start as usize)..(range.end as usize)];

        // HACK(eddyb) the range has to be extended backwards, because a partial
        // overlap could exit, i.e. `range.start` being in the middle of a value,
        // but then irrelevant values have to be ignored.
        let syms_ranges = self
            .syms
            .range(range.start.saturating_sub(self.max_sym_size.get() - 1)..range.end)
            .map(|(&offset, &(size, _))| offset..(offset + size.get()));

        // FIXME(eddyb) this is a bit inefficient but we don't have
        // cursors, so we have to buffer the `BTreeMap` keys here.
        let mut fully_overwritten_sym_offsets = SmallVec::<[u32; 16]>::new();
        for sym_range in syms_ranges {
            let overlap = sym_range.start.clamp(range.start, range.end)
                ..sym_range.end.clamp(range.start, range.end);
            if overlap.is_empty() {
                continue;
            }
            if overlap == sym_range {
                fully_overwritten_sym_offsets.push(sym_range.start);
            } else {
                return Err(PartialSymbolicOverlap { offsets: overlap });
            }
        }
        for offset in fully_overwritten_sym_offsets {
            self.syms.remove(&offset);
        }

        // FIXME(eddyb) come up with a nicer abstraction for bitvecs, or use a crate.
        {
            let (words, word_bitslices) = bitrange_word_chunks(range);
            for (word, word_bitslice) in self.init[words].iter_mut().zip(word_bitslices) {
                *word |= (!0 << word_bitslice.start) & (!0 >> (64 - word_bitslice.end));
            }
        }

        Ok(())
    }

    // HACK(eddyb) returns the written range (should it be just a byte length?).
    pub fn write_bytes(
        &mut self,
        offset: u32,
        bytes: &[u8],
    ) -> Result<Range<u32>, PartialSymbolicOverlap> {
        let range = offset..(offset + u32::try_from(bytes.len()).unwrap());
        self.try_init(range.clone())?;
        self.data[(range.start as usize)..(range.end as usize)].copy_from_slice(bytes);
        Ok(range)
    }

    // FIXME(eddyb) should this take an offset range instead?
    pub fn write_symbolic(
        &mut self,
        offset: u32,
        size: NonZeroU32,
        value: V,
    ) -> Result<(), PartialSymbolicOverlap> {
        let range = offset..(offset + size.get());
        self.try_init(range.clone())?;
        self.syms.insert(offset, (size, value));
        self.max_sym_size = self.max_sym_size.max(size);
        Ok(())
    }

    // HACK(eddyb) returns the written range (should it be just a byte length?).
    pub fn write_scalar(
        &mut self,
        offset: u32,
        scalar: scalar::Const,
        layout_config: &crate::mem::LayoutConfig,
    ) -> Result<Range<u32>, PartialSymbolicOverlap> {
        let byte_len = match scalar.ty() {
            scalar::Type::Bool => layout_config.abstract_bool_size_align.0,
            scalar::Type::SInt(_) | scalar::Type::UInt(_) | scalar::Type::Float(_) => {
                let bit_width = scalar.ty().bit_width();
                assert_eq!(bit_width % 8, 0);
                bit_width / 8
            }
        };
        let mut bytes = scalar.bits().to_le_bytes();
        let bytes = &mut bytes[..byte_len as usize];
        if layout_config.is_big_endian {
            bytes.reverse();
        }
        self.write_bytes(offset, bytes)
    }
}
