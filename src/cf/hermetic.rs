//! RVSDG-like "fully hermetic" regions can only use own inputs and node outputs.
//
// FIXME(eddyb) always keep SPIR-T in this form (e.g. for better local reasoning).

use crate::func_at::FuncAtMut;
use crate::{Context, EntityOrientedDenseMap, Node, NodeKind, Region, Value, Var, VarDecl};
use itertools::{Either, Itertools as _};

pub fn seal(cx: &Context, func_at_region: FuncAtMut<'_, Region>) {
    let region = func_at_region.position;
    Sealer { cx, tracked_vars: Default::default() }.seal_region(
        func_at_region.at(()),
        region,
        None,
    );
}

struct Sealer<'a> {
    cx: &'a Context,

    tracked_vars: EntityOrientedDenseMap<Var, TrackedVar>,
}

struct TrackedVar {
    /// Defining region (for a region input `Var`), or the parent region
    /// of the defining node (for a node output `Var`).
    def_region: Region,

    /// If this `Var` is ever used from a region other than `def_region`,
    /// `use_proxy` will cache the "proxy variable" used to access its value
    /// from within `use_region`, during the non-reentrant part of its sealing.
    ///
    /// **Note**: to avoid an additional `Option`, `use_proxy` is initialized
    /// to the original `Var` (also making `use_region == def_region`).
    //
    // HACK(eddyb) no save+restore mechanism is necessary for this caching
    // mechanism, thanks to the non-reentrant (i.e. shallow) traversal at
    // the end of `Sealer::seal_region`.
    use_proxy: Var,

    /// Defining region of `use_proxy` (equal to `def_region`, initially).
    use_region: Region,
}

/// Push `new_capture` to the node inputs, create all the necessary matching
/// `Var`s for child regions' inputs (plus all the `Loop`-specific dataflow),
/// and return the `def_idx` of the new `Var`s.
fn capture_value(cx: &Context, func_at_node: FuncAtMut<'_, Node>, new_capture: Value) -> u32 {
    let node = func_at_node.position;
    let mut func = func_at_node.at(());

    let ty = func.reborrow().freeze().at(new_capture).type_of(cx);

    let node_def = &mut func.nodes[node];

    let def_idx = u32::try_from(node_def.inputs.len()).unwrap();
    node_def.inputs.push(new_capture);

    for &child_region in &node_def.child_regions {
        let var = func.vars.define(
            cx,
            VarDecl {
                attrs: Default::default(),
                ty,
                def_parent: Either::Left(child_region),
                def_idx,
            },
        );

        let child_region_def = &mut func.regions[child_region];
        assert_eq!(child_region_def.inputs.len(), def_idx as usize);
        child_region_def.inputs.push(var);

        if let NodeKind::Loop { .. } = node_def.kind {
            assert_eq!(child_region_def.outputs.len(), def_idx as usize);
            child_region_def.outputs.push(Value::Var(var));
        }
    }

    if let NodeKind::Loop { .. } = node_def.kind {
        let var = func.vars.define(
            cx,
            VarDecl { attrs: Default::default(), ty, def_parent: Either::Right(node), def_idx },
        );

        assert_eq!(node_def.outputs.len(), def_idx as usize);
        node_def.outputs.push(var);
    }

    def_idx
}

impl Sealer<'_> {
    fn seal_region(
        &mut self,
        mut func: FuncAtMut<'_, ()>,
        region: Region,
        parent_node: Option<Node>,
    ) {
        // Start by recursively sealing all nested regions, bubbling up
        // all inter-region uses into node inputs (in this region).
        let track_defined_vars = |this: &mut Self, vars: &[Var]| {
            for &var in vars {
                // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
                // uniqueness like this (maybe using a new method that always does it?).
                assert!(
                    this.tracked_vars
                        .insert(
                            var,
                            TrackedVar { def_region: region, use_proxy: var, use_region: region },
                        )
                        .is_none()
                );
            }
        };
        track_defined_vars(self, &func.regions[region].inputs);
        {
            let mut func_at_children = func.reborrow().at(region).at_children().into_iter();
            while let Some(func_at_child_node) = func_at_children.next() {
                let node = func_at_child_node.position;
                let mut func = func_at_child_node.at(());

                let node_def = &mut func.nodes[node];

                // HACK(eddyb) temporary sanity check.
                for &v in &node_def.inputs {
                    if let Value::Var(v) = v {
                        assert!(self.tracked_vars.get(v).is_some());
                    }
                }

                // HACK(eddyb) prepare non-hermetic `Select` nodes by adding
                // an initial input `Var` to every child region.
                if let NodeKind::Select(_) = node_def.kind
                    && node_def.inputs.len() == 1
                    && node_def
                        .child_regions
                        .iter()
                        .map(|&case| func.regions[case].inputs.len())
                        .dedup()
                        .exactly_one()
                        .ok()
                        .unwrap()
                        == 0
                {
                    let selector = node_def.inputs.pop().unwrap();
                    assert_eq!(capture_value(self.cx, func.reborrow().at(node), selector), 0);
                }

                for i in 0..func.nodes[node].child_regions.len() {
                    let child_region = func.nodes[node].child_regions[i];

                    self.seal_region(func.reborrow(), child_region, Some(node));
                }

                track_defined_vars(self, &func.nodes[node].outputs);
            }

            // HACK(eddyb) temporary sanity check.
            for &v in &func.regions[region].outputs {
                if let Value::Var(v) = v {
                    assert!(self.tracked_vars.get(v).is_some());
                }
            }
        }

        // Reuse any (incidental) proxies that already exist, including those
        // added by previous sibling regions (of e.g. a `Select` node).
        if let Some(parent_node) = parent_node {
            let parent_node_def = &func.nodes[parent_node];
            let region_def = &func.regions[region];

            for (i, (&region_input, &parent_node_input)) in
                region_def.inputs.iter().zip_eq(&parent_node_def.inputs).enumerate()
            {
                let Value::Var(var) = parent_node_input else {
                    continue;
                };

                match parent_node_def.kind {
                    NodeKind::Select(_) => {}
                    NodeKind::Loop { .. } => {
                        if region_def.outputs[i] != Value::Var(region_input) {
                            continue;
                        }
                    }
                    _ => unreachable!(),
                }

                let TrackedVar { use_proxy, use_region, .. } = &mut self.tracked_vars[var];

                if *use_region == region {
                    continue;
                }

                *use_proxy = region_input;
                *use_region = region;
            }
        }

        // Finish sealing the region, by processing all (shallow) `Value` uses.
        let mut proxy_value_if_inter_region_use = |mut func: FuncAtMut<'_, ()>, v: Value| {
            let Value::Var(var) = v else {
                return None;
            };

            let TrackedVar { def_region, use_proxy, use_region } = &mut self.tracked_vars[var];

            if *def_region == region {
                return None;
            }

            if *use_region != region {
                let def_idx = capture_value(self.cx, func.reborrow().at(parent_node.unwrap()), v);
                *use_proxy = func.regions[region].inputs[def_idx as usize];
            }

            Some(Value::Var(*use_proxy))
        };

        // HACK(eddyb) sealing a `Loop` body results in additional body outputs,
        // which can be ignored (and doing so avoids adding them to `tracked_vars`).
        let original_output_count = func.regions[region].outputs.len();

        {
            let mut func_at_children = func.reborrow().at(region).at_children().into_iter();
            while let Some(func_at_child_node) = func_at_children.next() {
                let node = func_at_child_node.position;
                let mut func = func_at_child_node.at(());

                for i in 0..func.nodes[node].inputs.len() {
                    let input = func.nodes[node].inputs[i];
                    if let Some(proxy) = proxy_value_if_inter_region_use(func.reborrow(), input) {
                        func.nodes[node].inputs[i] = proxy;
                    }
                }
            }
        }
        for i in 0..original_output_count {
            let output = func.regions[region].outputs[i];
            if let Some(proxy) = proxy_value_if_inter_region_use(func.reborrow(), output) {
                func.regions[region].outputs[i] = proxy;
            }
        }
        // HACK(eddyb) `Loop` repeat conditions are effectively extra body outputs.
        if let Some(parent_node) = parent_node
            && let NodeKind::Loop { repeat_condition } = func.nodes[parent_node].kind
            && let Some(proxy) = proxy_value_if_inter_region_use(func.reborrow(), repeat_condition)
        {
            func.nodes[parent_node].kind = NodeKind::Loop { repeat_condition: proxy };
        }
    }
}
