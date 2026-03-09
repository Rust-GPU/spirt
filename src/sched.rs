//! Scheduling of execution and resources (e.g. register allocation).
//
// NOTE(eddyb) while SPIR-T is still strictly ordered, it might not be in
// the future, and a RVSDG-like approach would require an explicit ordering.

use crate::func_at::FuncAt;
use crate::{EntityOrientedDenseMap, FxIndexSet, Node, NodeKind, Region, Value, Var};
use std::collections::VecDeque;

// FIXME(eddyb) try to separate {control,data}-flow?
#[derive(Default)]
pub struct Schedule {
    pub regions: EntityOrientedDenseMap<Region, RegionSchedule>,
    pub vars: EntityOrientedDenseMap<Var, VarSchedule>,
}

pub struct RegionSchedule {
    pub nodes: FxIndexSet<Node>,
}

pub struct VarSchedule {
    pub def_region: Region,
    pub def_pos: DefPos,
    pub last_use_pos: Option<UsePos>,
}

// FIXME(eddyb) this could be more space-efficient using appropriate niches.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum DefPos {
    RegionInput,
    NodeOutput { node_sched_idx: u32 },
}

// FIXME(eddyb) this could be more space-efficient using appropriate niches.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum UsePos {
    NodeInput { node_sched_idx: u32 },
    RegionOutput,

    // HACK(eddyb) `Loop` repeat conditions are effectively extra body outputs.
    LoopRepeatCond,
}

impl Schedule {
    pub fn compute(func_at_region: FuncAt<'_, Region>) -> Self {
        let mut sched = Schedule { regions: Default::default(), vars: Default::default() };

        let func = func_at_region.at(());

        let mut queue = VecDeque::new();
        queue.push_back((func_at_region.position, None));
        while let Some((region, parent_node)) = queue.pop_front() {
            let func_at_region = func.at(region);
            let region_def = func_at_region.def();

            sched.add_defs(&region_def.inputs, region, DefPos::RegionInput);

            let mut region_sched = RegionSchedule { nodes: Default::default() };
            for func_at_node in func_at_region.at_children() {
                let node = func_at_node.position;
                let node_sched_idx = u32::try_from(region_sched.nodes.len()).unwrap();
                // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
                // uniqueness like this (maybe using a new method that always does it?).
                assert!(region_sched.nodes.insert(node));

                let node_def = func_at_node.def();

                sched.add_uses(&node_def.inputs, region, UsePos::NodeInput { node_sched_idx });
                sched.add_defs(&node_def.outputs, region, DefPos::NodeOutput { node_sched_idx });
                queue.extend(node_def.child_regions.iter().map(|&r| (r, Some(node))));
            }

            sched.add_uses(&region_def.outputs, region, UsePos::RegionOutput);

            if let Some(parent_node) = parent_node {
                // HACK(eddyb) `Loop` repeat conditions are effectively extra body outputs.
                if let NodeKind::Loop { repeat_condition } = func.nodes[parent_node].kind {
                    sched.add_uses(&[repeat_condition], region, UsePos::LoopRepeatCond);
                }
            }

            // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
            // uniqueness like this (maybe using a new method that always does it?).
            assert!(sched.regions.insert(region, region_sched).is_none());
        }

        sched
    }

    fn add_defs(&mut self, defined_vars: &[Var], def_region: Region, def_pos: DefPos) {
        for &var in defined_vars {
            // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
            // uniqueness like this (maybe using a new method that always does it?).
            assert!(
                self.vars
                    .insert(var, VarSchedule { def_region, def_pos, last_use_pos: None })
                    .is_none()
            );
        }
    }

    fn add_uses(&mut self, used_values: &[Value], use_region: Region, use_pos: UsePos) {
        for &v in used_values {
            match v {
                Value::Const(_) => {}
                Value::Var(var) => {
                    let var_sched = &mut self.vars[var];
                    assert!(
                        var_sched.def_region == use_region,
                        "cross-region use found, `cf::hermetic::seal` must be applied first!"
                    );

                    // FIXME(eddyb) consider asserting monotonicity here.
                    var_sched.last_use_pos = Some(use_pos);
                }
            }
        }
    }
}
