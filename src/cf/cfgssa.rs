//! Tools for working with control-flow graphs that contain SSA dataflow
//! (often abbreviated to `CFG<SSA>` or similar).
//!
//! The defining characteristic of SSA dataflow in a control-flow graph is that
//! SSA definitions (of values, e.g. the result of an instruction) are "visible"
//! from all the CFG locations they dominate (i.e. the locations that can only
//! be reached by passing through the definition first), and can therefore be
//! directly used arbitrarily far away in the CFG with no annotations required
//! anywhere in the CFG between the definition and its uses.
//!
//! While "def dominates use" is sufficient to ensure the value can traverse
//! the necessary paths (between def and use) in the CFG, a lot of care must
//! be taken to preserve the correctness of such implicit dataflow across all
//! transformations, and it's overall far more fragile than the local dataflow
//! of e.g. phi nodes (or their alternative "block arguments"), or in SPIR-T's
//! case, `Region` inputs and `Node` outputs (inspired by RVSDG,
//! which has even stricter isolation/locality in its regions).

use crate::{FxIndexMap, FxIndexSet};
use itertools::Either;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::hash::Hash;

// HACK(eddyb) to be able to propagate many uses at once while avoiding expensive
// hierchical indexing (and because the parent block of a def is significant),
// each block's defs get chunked, with chunks being the size of `FixedBitSet`
// (i.e. each bit tracks one def in the chunk, and can be propagated together).
// FIXME(eddyb) in theory, a sparse bitset could expose some of its sparseness
// to allow chunked addressing/iteration/etc. (but that requires more API design).
const CHUNK_SIZE: usize = data::FixedBitSet::SIZE;

#[derive(Copy, Clone, PartialEq, Eq, Hash, derive_more::From, derive_more::Into)]
struct BlockIdx(usize);
#[derive(Copy, Clone, PartialEq, Eq, Hash, derive_more::From, derive_more::Into)]
struct ChunkIdx(usize);
#[derive(Copy, Clone, PartialEq, Eq, Hash, derive_more::From, derive_more::Into)]
struct DefIdx(usize);

impl DefIdx {
    fn chunk(self) -> ChunkIdx {
        ChunkIdx(self.0 / CHUNK_SIZE)
    }
}

/// All blocks and ddefinitions they contain, which have to be computed first,
/// and remain immutable, because where a value is defined (or whether it's at
/// all part of the function itself) can have non-monotonic effects elsewhere.
pub struct DefMap<BlockId, DefId, DefType> {
    blocks_by_id: FxIndexMap<BlockId, BlockDef>,
    // FIXME(eddyb) should this contain `BlockIdx` instead?
    chunk_to_block_id: data::KeyedVec<ChunkIdx, BlockId>,
    chunk_defs: data::KeyedVec<ChunkIdx, [Option<(DefId, DefType)>; CHUNK_SIZE]>,
    def_id_to_def_idx: FxHashMap<DefId, DefIdx>,
}

struct BlockDef {
    last_def_idx: DefIdx,
}

impl<BlockId: Copy + Eq + Hash, DefId: Copy + Eq + Hash, DefType: Copy> Default
    for DefMap<BlockId, DefId, DefType>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<BlockId: Copy + Eq + Hash, DefId: Copy + Eq + Hash, DefType: Copy>
    DefMap<BlockId, DefId, DefType>
{
    pub fn new() -> Self {
        Self {
            blocks_by_id: Default::default(),
            chunk_to_block_id: Default::default(),
            chunk_defs: Default::default(),
            def_id_to_def_idx: Default::default(),
        }
    }

    pub fn add_block(&mut self, block_id: BlockId) {
        // FIXME(eddyb) disallow accidental re-insertion.
        self.blocks_by_id.insert(block_id, BlockDef { last_def_idx: DefIdx(!0) });
    }

    pub fn add_def(&mut self, block_id: BlockId, def_id: DefId, def_type: DefType) {
        // HACK(eddyb) optimize for repeated definitions in the same block.
        let block = match self.blocks_by_id.last_mut() {
            Some((&last_block_id, last_block)) if last_block_id == block_id => last_block,
            _ => &mut self.blocks_by_id[&block_id],
        };
        let def_idx = Some(DefIdx(block.last_def_idx.0.wrapping_add(1)))
            .filter(|def_idx| def_idx.chunk() == block.last_def_idx.chunk())
            .unwrap_or_else(|| {
                let chunk_idx = self.chunk_to_block_id.push(block_id);
                assert!(chunk_idx == self.chunk_defs.push([None; CHUNK_SIZE]));
                DefIdx(chunk_idx.0 * CHUNK_SIZE)
            });
        block.last_def_idx = def_idx;

        self.chunk_defs[def_idx.chunk()][def_idx.0 % CHUNK_SIZE] = Some((def_id, def_type));

        // FIXME(eddyb) disallow accidental re-insertion.
        self.def_id_to_def_idx.insert(def_id, def_idx);
    }

    fn block_id_from_idx(&self, block_idx: BlockIdx) -> BlockId {
        *self.blocks_by_id.get_index(block_idx.0).unwrap().0
    }
}

/// Incremental tracker for definition uses (and CFG edges between blocks),
/// accumulating the complete set of transitive uses for each block, also known
/// as the SSA "live set" (corresponding to the starting position of each block).
pub struct UseAccumulator<'a, BlockId, DefId, DefType> {
    def_map: &'a DefMap<BlockId, DefId, DefType>,

    blocks: data::KeyedVec<BlockIdx, BlockAcc>,

    // HACK(eddyb) optimize for repeated uses from the same block.
    most_recent_block_idx: BlockIdx,

    /// Every `block_idx` with non-empty `blocks[block_idx].dirty_chunks`,
    /// and used for breadth-first propagation through predecessors.
    //
    // FIXME(eddyb) some traversal orders might be more effective, but also
    // the "chunk" granularity might itself be enough to paper over that?
    propagate_queue: VecDeque<BlockIdx>,
}

#[derive(Default)]
struct BlockAcc {
    // FIXME(eddyb) should this use a bitset? seems likely to be inefficient
    preds: FxIndexSet<BlockIdx>,

    /// All definitions used in this block (or any other block reachable from it),
    /// excluding its own definitions, and represented as a sparse bitset.
    uses: data::SparseMap<ChunkIdx, data::FixedBitSet<usize>>,

    /// All chunks `c` where `uses[c]` has changed since this block has last
    /// propagated any of its `uses` to its predecessors.
    dirty_chunks: data::BitSet<ChunkIdx>,
}

enum AddUsesSource {
    New(DefIdx),
    PropagateBackwardsAcrossEdge { target: BlockIdx, only_dirty: bool },
}

impl<'a, BlockId: Copy + Eq + Hash, DefId: Copy + Eq + Hash, DefType: Copy>
    UseAccumulator<'a, BlockId, DefId, DefType>
{
    pub fn new(def_map: &'a DefMap<BlockId, DefId, DefType>) -> Self {
        Self {
            def_map,

            blocks: data::KeyedVec::from_fn(..BlockIdx(def_map.blocks_by_id.len()), |_| {
                Default::default()
            }),

            most_recent_block_idx: BlockIdx(0),

            propagate_queue: VecDeque::new(),
        }
    }

    // FIXME(eddyb) how inefficient is `FxIndexMap<DefId, DefType>`?
    // (vs e.g. a bitset combined with not duplicating `DefType`s per-block?)
    // FIXME(eddyb) naming might not be enough to clarify the semantics,
    // might be useful to use the liveness (e.g. "live set") jargon?
    pub fn into_inter_block_uses(
        mut self,
    ) -> impl Iterator<Item = (BlockId, FxIndexMap<DefId, DefType>)> + 'a {
        self.propagate();

        assert!(self.propagate_queue.is_empty());

        self.blocks.into_iter().map(|(block_idx, block_acc)| {
            assert!(block_acc.dirty_chunks.is_empty());

            (
                self.def_map.block_id_from_idx(block_idx),
                block_acc
                    .uses
                    .iter()
                    .flat_map(|(chunk_idx, chunk_uses)| {
                        let chunk_defs = &self.def_map.chunk_defs[chunk_idx];
                        chunk_uses.keys().map(move |i| chunk_defs[i].unwrap())
                    })
                    .collect(),
            )
        })
    }

    pub fn add_use(&mut self, block_id: BlockId, used_def_id: DefId) {
        // FIXME(eddyb) use `let ... else`?
        let &used_def_idx = match self.def_map.def_id_to_def_idx.get(&used_def_id) {
            // HACK(eddyb) silently ignoring unrecognized defs.
            None => return,
            Some(def_idx) => def_idx,
        };

        // Intra-block uses are not tracked.
        if self.def_map.chunk_to_block_id[used_def_idx.chunk()] == block_id {
            return;
        }

        let block_idx = Some(self.most_recent_block_idx)
            .filter(|&block_idx| self.def_map.block_id_from_idx(block_idx) == block_id)
            .unwrap_or_else(|| {
                BlockIdx(self.def_map.blocks_by_id.get_index_of(&block_id).unwrap())
            });

        self.add_uses_to(block_idx, AddUsesSource::New(used_def_idx));
    }

    pub fn add_edge(&mut self, source_block_id: BlockId, target_block_id: BlockId) {
        // Self-loops require no tracking (could never introduce more uses).
        if source_block_id == target_block_id {
            return;
        }

        // FIXME(eddyb) is this necessary? (the concern is that dirty-tracking
        // might get confused, but it shouldn't actually be an issue)
        self.propagate();

        let [source_block_idx, target_block_idx] = [source_block_id, target_block_id]
            .map(|block_id| BlockIdx(self.def_map.blocks_by_id.get_index_of(&block_id).unwrap()));

        if self.blocks[target_block_idx].preds.insert(source_block_idx) {
            self.add_uses_to(
                source_block_idx,
                AddUsesSource::PropagateBackwardsAcrossEdge {
                    target: target_block_idx,
                    only_dirty: false,
                },
            );
        }
    }

    fn propagate(&mut self) {
        while let Some(block_idx) = self.propagate_queue.pop_front() {
            for i in 0..self.blocks[block_idx].preds.len() {
                let pred_block_idx = self.blocks[block_idx].preds[i];

                self.add_uses_to(
                    pred_block_idx,
                    AddUsesSource::PropagateBackwardsAcrossEdge {
                        target: block_idx,
                        only_dirty: true,
                    },
                );
            }
            self.blocks[block_idx].dirty_chunks.clear();
        }
    }

    fn add_uses_to(&mut self, block_idx: BlockIdx, uses: AddUsesSource) {
        // FIXME(eddyb) make this unnecessary for a comparison later, perhaps?
        let block_id = self.def_map.block_id_from_idx(block_idx);

        let mut new_uses;
        let (block_acc, chunked_uses) = match uses {
            AddUsesSource::New(def_idx) => {
                new_uses = data::FixedBitSet::new();
                new_uses.insert(def_idx.0 % CHUNK_SIZE, ());
                (
                    &mut self.blocks[block_idx],
                    Either::Left([(def_idx.chunk(), &new_uses)].into_iter()),
                )
            }
            AddUsesSource::PropagateBackwardsAcrossEdge { target, only_dirty } => {
                let [block_acc, target_block_acc] =
                    self.blocks.get_mut2([block_idx, target]).unwrap();

                (
                    block_acc,
                    Either::Right(if only_dirty {
                        Either::Left(target_block_acc.dirty_chunks.iter().map(|(chunk_idx, _)| {
                            (chunk_idx, target_block_acc.uses.get(chunk_idx).unwrap())
                        }))
                    } else {
                        Either::Right(target_block_acc.uses.iter())
                    }),
                )
            }
        };

        let block_was_dirty = !block_acc.dirty_chunks.is_empty();
        for (chunk_idx, new_uses) in chunked_uses {
            // Use tracking terminates in the defining block.
            if self.def_map.chunk_to_block_id[chunk_idx] == block_id {
                continue;
            }

            let uses = block_acc.uses.entry(chunk_idx).or_default();

            let old_and_new_uses = uses.union(new_uses);
            if *uses != old_and_new_uses {
                *uses = old_and_new_uses;
                block_acc.dirty_chunks.entry(chunk_idx).insert(());
            }
        }
        if !block_was_dirty && !block_acc.dirty_chunks.is_empty() {
            self.propagate_queue.push_back(block_idx);
        }
    }
}

// HACK(eddyb) the hierarchy/breadth/sparsity/etc. of these data structures is
// somewhat arbitrary, but they should do better than naive non-sparse solutions.
// FIMXE(eddyb) attempt to fine-tune this for realistic workloads.
// FIXME(eddyb) move this out of here.
mod data {
    use smallvec::SmallVec;
    use std::marker::PhantomData;
    use std::{iter, mem, ops};

    // FIXME(eddyb) should this be `FlatVec`? also, does it belong here?
    pub struct KeyedVec<K, T> {
        vec: Vec<T>,
        _marker: PhantomData<K>,
    }

    impl<K: Copy + Into<usize> + From<usize>, T> Default for KeyedVec<K, T> {
        fn default() -> Self {
            Self::new()
        }
    }
    impl<K: Copy + Into<usize> + From<usize>, T> KeyedVec<K, T> {
        pub fn new() -> Self {
            Self { vec: vec![], _marker: PhantomData }
        }
        pub fn from_fn(keys: ops::RangeTo<K>, mut f: impl FnMut(K) -> T) -> Self {
            KeyedVec {
                vec: (0..keys.end.into()).map(|i| f(K::from(i))).collect(),
                _marker: PhantomData,
            }
        }
        pub fn push(&mut self, x: T) -> K {
            let k = K::from(self.vec.len());
            self.vec.push(x);
            k
        }
        pub fn into_iter(self) -> impl Iterator<Item = (K, T)> {
            self.vec.into_iter().enumerate().map(|(i, x)| (K::from(i), x))
        }

        // FIXME(eddyb) replace this when `get_many_mut` gets stabilizes.
        pub fn get_mut2(&mut self, keys: [K; 2]) -> Option<[&mut T; 2]> {
            let [k_i, k_j] = keys;
            let (i, j) = (k_i.into(), k_j.into());
            if i > j {
                let [y, x] = self.get_mut2([k_j, k_i])?;
                return Some([x, y]);
            }
            if i == j || j >= self.vec.len() {
                return None;
            }

            let (xs, ys) = self.vec.split_at_mut(j);
            Some([&mut xs[i], &mut ys[0]])
        }
    }
    impl<K: Copy + Into<usize> + From<usize>, T> ops::Index<K> for KeyedVec<K, T> {
        type Output = T;
        fn index(&self, k: K) -> &T {
            &self.vec[k.into()]
        }
    }
    impl<K: Copy + Into<usize> + From<usize>, T> ops::IndexMut<K> for KeyedVec<K, T> {
        fn index_mut(&mut self, k: K) -> &mut T {
            &mut self.vec[k.into()]
        }
    }

    // HACK(eddyb) abstraction to enable code sharing between maps and sets.
    pub trait ValueStorage<V> {
        // HACK(eddyb) most of the need for this arises from avoidance of
        // `unsafe` code (i.e. `MaybeUninit<V>` could suffice in most cases).
        type Slot: Default;
        fn slot_unwrap(slot: Self::Slot) -> V;
        fn slot_unwrap_ref(slot: &Self::Slot) -> &V;
        fn slot_unwrap_mut(slot: &mut Self::Slot) -> &mut V;
        fn slot_insert(slot: &mut Self::Slot, v: V) -> &mut V;

        // FIXME(eddyb) ideally whether this allocates would be size-based.
        // FIXME(eddyb) the name and APIs probably don't make it clear this is
        // for holding some number of `Self::Slot`s specifically.
        type LazyBox<T>;
        fn lazy_box_default<T>(default: impl Fn() -> T) -> Self::LazyBox<T>;
        fn lazy_box_unwrap_ref<T>(lb: &Self::LazyBox<T>) -> &T;
        fn lazy_box_unwrap_mut_or_alloc<T>(
            lb: &mut Self::LazyBox<T>,
            default: impl Fn() -> T,
        ) -> &mut T;
    }

    pub enum IgnoreValue {}
    impl ValueStorage<()> for IgnoreValue {
        type Slot = ();
        fn slot_unwrap(_: ()) {}
        fn slot_unwrap_ref(_: &()) -> &() {
            &()
        }
        fn slot_unwrap_mut(slot: &mut ()) -> &mut () {
            slot
        }
        fn slot_insert(slot: &mut (), _: ()) -> &mut () {
            slot
        }

        type LazyBox<T> = T;
        fn lazy_box_default<T>(default: impl Fn() -> T) -> T {
            default()
        }
        fn lazy_box_unwrap_ref<T>(lb: &T) -> &T {
            lb
        }
        fn lazy_box_unwrap_mut_or_alloc<T>(lb: &mut T, _: impl Fn() -> T) -> &mut T {
            lb
        }
    }

    pub enum EfficientValue {}
    impl<V: Default> ValueStorage<V> for EfficientValue {
        type Slot = V;
        fn slot_unwrap(slot: V) -> V {
            slot
        }
        fn slot_unwrap_ref(slot: &V) -> &V {
            slot
        }
        fn slot_unwrap_mut(slot: &mut V) -> &mut V {
            slot
        }
        fn slot_insert(slot: &mut V, v: V) -> &mut V {
            *slot = v;
            slot
        }

        // FIXME(eddyb) this is far from "efficient", maybe this part belong
        // in another `trait`, or some better automation could be found?
        type LazyBox<T> = Option<Box<T>>;
        fn lazy_box_default<T>(_: impl Fn() -> T) -> Option<Box<T>> {
            None
        }
        fn lazy_box_unwrap_ref<T>(lb: &Option<Box<T>>) -> &T {
            lb.as_deref().unwrap()
        }
        fn lazy_box_unwrap_mut_or_alloc<T>(
            lb: &mut Option<Box<T>>,
            default: impl Fn() -> T,
        ) -> &mut T {
            lb.get_or_insert_with(|| Box::new(default()))
        }
    }

    // HACK(eddyb) most of the need for this arises from avoidance of
    // `unsafe` code (i.e. `MaybeUninit<V>` could suffice in most cases).
    // FIXME(eddyb) figure out if keeping this around is useful at all.
    #[allow(unused)]
    pub enum WrapNonDefaultValueInOption {}
    impl<V> ValueStorage<V> for WrapNonDefaultValueInOption {
        type Slot = Option<V>;
        fn slot_unwrap(slot: Option<V>) -> V {
            slot.unwrap()
        }
        fn slot_unwrap_ref(slot: &Option<V>) -> &V {
            slot.as_ref().unwrap()
        }
        fn slot_unwrap_mut(slot: &mut Option<V>) -> &mut V {
            slot.as_mut().unwrap()
        }
        fn slot_insert(slot: &mut Option<V>, v: V) -> &mut V {
            slot.insert(v)
        }

        type LazyBox<T> = Option<Box<T>>;
        fn lazy_box_default<T>(_: impl Fn() -> T) -> Option<Box<T>> {
            None
        }
        fn lazy_box_unwrap_ref<T>(lb: &Option<Box<T>>) -> &T {
            lb.as_deref().unwrap()
        }
        fn lazy_box_unwrap_mut_or_alloc<T>(
            lb: &mut Option<Box<T>>,
            default: impl Fn() -> T,
        ) -> &mut T {
            lb.get_or_insert_with(|| Box::new(default()))
        }
    }

    // FIXME(eddyb) maybe make this parameterizable?
    type FixedBitSetUint = u64;

    pub type FixedBitSet<K> = FixedFlatMap<K, (), IgnoreValue>;
    const _: () =
        assert!(mem::size_of::<FixedBitSet<usize>>() == mem::size_of::<FixedBitSetUint>());

    pub struct FixedFlatMap<K, V, VS: ValueStorage<V> = EfficientValue> {
        occupied: FixedBitSetUint,
        slots: VS::LazyBox<[VS::Slot; FixedBitSet::SIZE]>,
        _marker: PhantomData<K>,
    }

    impl FixedBitSet<usize> {
        pub const SIZE: usize = {
            let bit_width = mem::size_of::<FixedBitSetUint>() * 8;
            assert!(FixedBitSetUint::count_ones(!0) == bit_width as u32);
            bit_width
        };
    }
    impl<K> PartialEq for FixedBitSet<K> {
        fn eq(&self, other: &Self) -> bool {
            self.occupied == other.occupied
        }
    }
    impl<K: Copy + Into<usize> + From<usize>> FixedBitSet<K> {
        pub fn union(&self, other: &Self) -> Self {
            Self { occupied: self.occupied | other.occupied, ..Self::default() }
        }
    }

    impl<K: Copy + Into<usize> + From<usize>, V, VS: ValueStorage<V>> Default
        for FixedFlatMap<K, V, VS>
    {
        fn default() -> Self {
            Self::new()
        }
    }
    impl<K: Copy + Into<usize> + From<usize>, V, VS: ValueStorage<V>> FixedFlatMap<K, V, VS> {
        pub fn new() -> Self {
            Self {
                occupied: 0,
                slots: VS::lazy_box_default(|| std::array::from_fn(|_| Default::default())),
                _marker: PhantomData,
            }
        }
        pub fn contains(&self, k: K) -> bool {
            u32::try_from(k.into()).ok().and_then(|k| Some(self.occupied.checked_shr(k)? & 1))
                == Some(1)
        }
        pub fn get(&self, k: K) -> Option<&V> {
            self.contains(k)
                .then(|| VS::slot_unwrap_ref(&VS::lazy_box_unwrap_ref(&self.slots)[k.into()]))
        }
        pub fn entry(&mut self, k: K) -> FixedFlatMapEntry<'_, V, VS> {
            let k = k.into();
            let key_mask = FixedBitSetUint::checked_shl(1, u32::try_from(k).unwrap()).unwrap();
            FixedFlatMapEntry {
                key_mask,
                occupied: &mut self.occupied,
                slot: &mut VS::lazy_box_unwrap_mut_or_alloc(&mut self.slots, || {
                    std::array::from_fn(|_| Default::default())
                })[k],
            }
        }

        pub fn insert(&mut self, k: K, v: V) {
            self.entry(k).insert(v);
        }
        fn remove(&mut self, k: K) -> Option<V> {
            self.contains(k).then(|| self.entry(k).remove().unwrap())
        }

        pub fn keys(&self) -> impl Iterator<Item = K> + use<K, V, VS> {
            let mut i = 0;
            let mut remaining = self.occupied;
            iter::from_fn(move || {
                (remaining != 0).then(|| {
                    let gap = remaining.trailing_zeros() as usize;
                    i += gap;
                    remaining >>= gap;

                    let k = K::from(i);

                    // Skip the lowest bit (which should always be `1` here).
                    i += 1;
                    remaining >>= 1;

                    k
                })
            })
        }
        pub fn iter(&self) -> impl Iterator<Item = (K, &V)> + '_ {
            self.keys().map(|k| (k, self.get(k).unwrap()))
        }
        pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
            self.keys().map(|k| (k, self.remove(k).unwrap()))
        }

        // FIXME(eddyb) does this fully replace `drain`?
        pub fn clear(&mut self) {
            // FIXME(eddyb) theoretically this could be more efficient, but
            // it doesn't seem worth it wrt `VS` abstraction complexity.
            for _ in self.drain() {}
        }
    }

    pub struct FixedFlatMapEntry<'a, V, VS: ValueStorage<V> = EfficientValue> {
        key_mask: FixedBitSetUint,
        occupied: &'a mut FixedBitSetUint,
        // FIXME(eddyb) in theory, this forces the `Box` to be allocated even
        // when it might not be needed, so it optimizes for e.g. insertion.
        slot: &'a mut VS::Slot,
    }

    impl<'a, V, VS: ValueStorage<V>> FixedFlatMapEntry<'a, V, VS> {
        pub fn occupied(&self) -> bool {
            (*self.occupied & self.key_mask) != 0
        }
        fn into_mut(self) -> Option<&'a mut V> {
            self.occupied().then(|| VS::slot_unwrap_mut(self.slot))
        }
        pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
            if self.occupied() { self.into_mut().unwrap() } else { self.insert(f()) }
        }
        pub fn insert(self, v: V) -> &'a mut V {
            *self.occupied |= self.key_mask;
            VS::slot_insert(self.slot, v)
        }
        pub fn remove(&mut self) -> Option<V> {
            self.occupied().then(|| {
                *self.occupied &= !self.key_mask;
                VS::slot_unwrap(mem::take(self.slot))
            })
        }
    }

    // FIXME(eddyb) not a sparse bitset because of how `SparseMap` is only sparse
    // wrt not allocating space to store values until needed, but `BitSet` has no
    // real values (and uses `IgnoreValue` to completely remove that allocation).
    pub type BitSet<K> = SparseMap<K, (), IgnoreValue>;

    pub struct SparseMap<K, V, VS: ValueStorage<V> = EfficientValue> {
        // NOTE(eddyb) this is really efficient when the keys don't exceed
        // `FixedBitSet::SIZE`, and this can be further amplified by using
        // e.g. `SparseMap<_, FixedFlatMap<_, V>>`, which adds another layer
        // of sparseness (more concretely, if `FixedBitSet::SIZE == 64`, then
        // that combination can effectively handle up to 64*64 = 4096 entries
        // without causing the outermost `SmallFlatMap` to allocate a `Vec`).
        outer_map: SmallVec<[FixedFlatMap<usize, V, VS>; 1]>,
        count: usize,
        _marker: PhantomData<K>,
    }

    impl<K: Copy + Into<usize> + From<usize>, V, VS: ValueStorage<V>> Default for SparseMap<K, V, VS> {
        fn default() -> Self {
            Self::new()
        }
    }
    impl<K: Copy + Into<usize> + From<usize>, V, VS: ValueStorage<V>> SparseMap<K, V, VS> {
        pub fn new() -> Self {
            Self { outer_map: SmallVec::new(), count: 0, _marker: PhantomData }
        }
        pub fn is_empty(&self) -> bool {
            self.count == 0
        }
        pub fn get(&self, k: K) -> Option<&V> {
            let k = k.into();
            let (outer_key, inner_key) = (k / FixedBitSet::SIZE, k % FixedBitSet::SIZE);
            self.outer_map.get(outer_key)?.get(inner_key)
        }
        pub fn entry(&mut self, k: K) -> SparseMapEntry<'_, V, VS> {
            let k = k.into();
            let (outer_key, inner_key) = (k / FixedBitSet::SIZE, k % FixedBitSet::SIZE);
            let needed_outer_len = outer_key + 1;
            if self.outer_map.len() < needed_outer_len {
                self.outer_map.resize_with(needed_outer_len, Default::default);
            }
            SparseMapEntry {
                inner: self.outer_map[outer_key].entry(inner_key),
                count: &mut self.count,
            }
        }

        pub fn iter(&self) -> impl Iterator<Item = (K, &V)> + '_ {
            (!self.is_empty())
                .then(|| {
                    self.outer_map.iter().enumerate().flat_map(|(outer_key, inner_map)| {
                        inner_map.iter().map(move |(inner_key, v)| {
                            (K::from(outer_key * FixedBitSet::SIZE + inner_key), v)
                        })
                    })
                })
                .into_iter()
                .flatten()
        }

        pub fn clear(&mut self) {
            for inner_map in &mut self.outer_map {
                inner_map.clear();
            }
            self.count = 0;
        }
    }

    pub struct SparseMapEntry<'a, V, VS: ValueStorage<V> = EfficientValue> {
        inner: FixedFlatMapEntry<'a, V, VS>,
        count: &'a mut usize,
    }

    impl<'a, V, VS: ValueStorage<V>> SparseMapEntry<'a, V, VS> {
        pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
            self.inner.or_insert_with(|| {
                *self.count += 1;
                f()
            })
        }
        #[allow(clippy::unwrap_or_default)]
        pub fn or_default(self) -> &'a mut V
        where
            V: Default,
        {
            self.or_insert_with(V::default)
        }
        pub fn insert(self, v: V) {
            if !self.inner.occupied() {
                *self.count += 1;
            }
            self.inner.insert(v);
        }
    }
}
