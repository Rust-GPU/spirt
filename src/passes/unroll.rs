//! Loop unrolling optimization pass.
//!
//! Analyzes every `Loop` node in the IR and unrolls it when the trip count is
//! statically determinable and the cost/benefit analysis is positive.
//!
//! # Profitability model
//!
//! Three metrics are estimated before committing to an unroll:
//! * **Body size** – total `DataInst` count inside the body (recursively).
//! * **Register pressure** – `loop_state_vars × trip_count`, where
//!   `loop_state_vars` is the number of values that flow across loop
//!   iterations (`initial_inputs.len()`).  Captures how many simultaneous
//!   live values unrolling produces.
//! * **Code growth** – `body_size × trip_count`.
//!
//! Each metric is compared against a per-loop threshold.
//!
//! # Trip-count detection
//!
//! Two strategies are tried in order:
//! 1. **Direct condition**: `repeat_condition` is (or is derived from) a
//!    `ULessThan / SLessThan / INotEqual` of a loop-input against a constant.
//! 2. **Body scan**: walk every top-level `Block` in the body looking for the
//!    same comparison patterns.
//!

use crate::visit::{InnerVisit, Visitor};
use crate::{
    AttrSet, Const, ConstKind, Context, DataInst, DataInstDef, DataInstKind, DeclDef, EntityList,
    Func, FuncDefBody, FxIndexSet, Module, Node, NodeDef, NodeKind, Region, RegionDef,
    RegionInputDecl, Value, spv,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct UnrollConfig {
    /// Maximum body `DataInst` count.  Bodies larger than this are not unrolled
    pub max_body_insts: usize,
    /// Maximum total `DataInst`s after unrolling (`body_size x trip_count`).
    pub max_unrolled_insts: usize,
    /// Maximum trip count considered for unrolling.
    pub max_trip_count: u32,
    /// Maximum `loop_state_vars x trip_count` (register-pressure).
    pub max_register_pressure: usize,
}

impl Default for UnrollConfig {
    fn default() -> Self {
        Self {
            max_body_insts: 32,
            max_unrolled_insts: 128,
            max_trip_count: 16,
            max_register_pressure: 32,
        }
    }
}

/// Analyse every `Loop` node in `module` and unroll those that are
/// profitable.  Uses [`UnrollConfig::default`] thresholds.
pub fn unroll_loops(module: &mut Module) {
    unroll_loops_with_config(module, &UnrollConfig::default());
}

pub fn unroll_loops_with_config(module: &mut Module, config: &UnrollConfig) {
    let cx = module.cx();
    let mut seen_funcs = FxIndexSet::default();
    collect_funcs(module, &mut seen_funcs);
    for &func in &seen_funcs {
        if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
            unroll_loops_in_func(&cx, func_def_body, config);
        }
    }
}

fn collect_funcs(module: &Module, seen_funcs: &mut FxIndexSet<Func>) {
    struct FuncCollector<'a> {
        module: &'a Module,
        seen_funcs: &'a mut FxIndexSet<Func>,
    }
    impl Visitor<'_> for FuncCollector<'_> {
        fn visit_attr_set_use(&mut self, _: AttrSet) {}
        fn visit_type_use(&mut self, _: crate::Type) {}
        fn visit_const_use(&mut self, _: Const) {}
        fn visit_global_var_use(&mut self, _: crate::GlobalVar) {}
        fn visit_func_use(&mut self, func: Func) {
            if self.seen_funcs.insert(func) {
                self.visit_func_decl(&self.module.funcs[func]);
            }
        }
    }
    let mut col = FuncCollector { module, seen_funcs };
    for (_, &exp) in &module.exports {
        exp.inner_visit_with(&mut col);
    }
}

fn unroll_loops_in_func(cx: &Context, func: &mut FuncDefBody, default_config: &UnrollConfig) {
    let candidates = collect_loop_candidates(func);

    for (parent_region, loop_node) in candidates {
        // trip-count detection.
        let Some(trip_count) = detect_trip_count(cx, func, loop_node) else { continue };
        if trip_count == 0 {
            func.regions[parent_region].children.remove(loop_node, &mut func.nodes);
            continue;
        }
        if trip_count > default_config.max_trip_count {
            continue;
        }

        // cost analysis.
        let NodeKind::Loop { ref initial_inputs, body, .. } = func.nodes[loop_node].kind else {
            unreachable!()
        };
        let body_size = count_body_insts(func, body);
        let loop_state = initial_inputs.len();

        if !is_profitable(default_config, body_size, loop_state, trip_count) {
            continue;
        }

        inline_unrolled_iterations(cx, func, parent_region, loop_node, trip_count);
    }
}

/// walk every region recursively, collecting `(parent_region, loop_node)` in
/// post-order (inner loops first so they are eligible to be unrolled before
/// their containing loop is considered).
fn collect_loop_candidates(func: &FuncDefBody) -> Vec<(Region, Node)> {
    let mut out = Vec::new();
    visit_region_for_loops(func, func.body, &mut out);
    out
}

fn visit_region_for_loops(func: &FuncDefBody, region: Region, out: &mut Vec<(Region, Node)>) {
    let mut iter = func.regions[region].children.iter();
    while let Some((node, rest)) = iter.split_first(&func.nodes) {
        iter = rest;
        match &func.nodes[node].kind {
            NodeKind::Block { .. } | NodeKind::ExitInvocation { .. } => {}
            NodeKind::Select { cases, .. } => {
                for &case in cases.clone().iter() {
                    visit_region_for_loops(func, case, out);
                }
            }
            NodeKind::Loop { body, .. } => {
                visit_region_for_loops(func, *body, out);
                out.push((region, node));
            }
        }
    }
}

fn count_body_insts(func: &FuncDefBody, region: Region) -> usize {
    let mut n = 0;
    count_region_insts(func, region, &mut n);
    n
}

fn count_region_insts(func: &FuncDefBody, region: Region, n: &mut usize) {
    let mut iter = func.regions[region].children.iter();
    while let Some((node, rest)) = iter.split_first(&func.nodes) {
        iter = rest;
        match &func.nodes[node].kind {
            NodeKind::Block { insts } => {
                let mut it = insts.iter();
                while let Some((_, r)) = it.split_first(&func.data_insts) {
                    it = r;
                    *n += 1;
                }
            }
            NodeKind::Select { cases, .. } => {
                for &case in cases.iter() {
                    count_region_insts(func, case, n);
                }
            }
            NodeKind::Loop { body, .. } => {
                count_region_insts(func, *body, n);
            }
            NodeKind::ExitInvocation { .. } => {}
        }
    }
}

fn is_profitable(cfg: &UnrollConfig, body_size: usize, loop_state: usize, trip_count: u32) -> bool {
    let tc = trip_count as usize;
    body_size <= cfg.max_body_insts
        && body_size.saturating_mul(tc) <= cfg.max_unrolled_insts
        && loop_state.saturating_mul(tc) <= cfg.max_register_pressure
}

/// try to determine the exact number of times the loop body should execute.
///
/// Returns `None` if no static trip count could be found.
fn detect_trip_count(cx: &Context, func: &FuncDefBody, loop_node: Node) -> Option<u32> {
    let NodeKind::Loop { initial_inputs, body, repeat_condition } = &func.nodes[loop_node].kind
    else {
        return None;
    };
    let (body, repeat_condition) = (*body, *repeat_condition);
    let initial_inputs: SmallVec<[Value; 2]> = initial_inputs.clone();

    // follow the repeat_condition directly.
    if let Some(tc) = trip_count_from_condition(cx, func, repeat_condition, body, &initial_inputs) {
        return Some(tc);
    }

    // scan the body's top-level Blocks for a comparison.
    trip_count_from_body_scan(cx, func, body, &initial_inputs)
}

/// try to derive the trip count by inspecting `repeat_condition`.
///
/// Handles:
/// * `DataInstOutput(cmp)` – direct comparison `DataInst`.
/// * `NodeOutput{select, idx}` – the condition comes out of a Select;
///   we inspect the branches to find a constant-false arm and pull the real
///   condition from the other arm.
fn trip_count_from_condition(
    cx: &Context,
    func: &FuncDefBody,
    cond: Value,
    body: Region,
    initial_inputs: &[Value],
) -> Option<u32> {
    match cond {
        Value::DataInstOutput(inst) => {
            let def = &*func.data_insts[inst];
            try_trip_count_from_cmp(cx, func, def, body, initial_inputs)
        }

        // if the condition is the output of a Select, peek into its branches:
        // one arm might be `false` (constant) while the other carries the real
        // comparison, e.g.:
        //   `(_, cond) = if guard { (_, real_cmp) } else { (_, false) }`
        Value::NodeOutput { node, output_idx } => {
            let NodeKind::Select { cases, .. } = &func.nodes[node].kind else {
                return None;
            };
            for &case in cases.iter() {
                let case_outputs = &func.regions[case].outputs;
                let case_val = *case_outputs.get(output_idx as usize)?;
                // Skip constant-false arms.
                if is_const_false(cx, case_val) {
                    continue;
                }
                if let Some(tc) =
                    trip_count_from_condition(cx, func, case_val, body, initial_inputs)
                {
                    return Some(tc);
                }
            }
            None
        }

        _ => None,
    }
}

/// If `inst` is a `ULessThan / SLessThan / INotEqual(lhs, N_const)` where
/// `lhs` is a loop input (or `IAdd(loop_input, 1)`), return the trip count.
fn try_trip_count_from_cmp(
    cx: &Context,
    func: &FuncDefBody,
    inst: &DataInstDef,
    body: Region,
    initial_inputs: &[Value],
) -> Option<u32> {
    let DataInstKind::SpvInst(ref spv_inst) = inst.kind else {
        return None;
    };
    let op = spv_inst.opcode.name();
    let is_lt = matches!(op, "OpULessThan" | "OpSLessThan");
    let is_ne = op == "OpINotEqual";
    if !is_lt && !is_ne {
        return None;
    }

    let [lhs, rhs] = inst.inputs.as_slice() else { return None };

    //  rust-gpu packs range state into a struct; the comparison is
    //   `OpCompositeExtract(input, counter_field) < OpCompositeExtract(input, bound_field)`
    // where the initial value of `input` is `OpConstantComposite(init, bound)`.
    if (is_lt || is_ne)
        && let (Some((lhs_base, lhs_field)), Some((rhs_base, rhs_field))) =
            (follow_composite_extract(func, *lhs), follow_composite_extract(func, *rhs))
        && let (
            Value::RegionInput { region: lr, input_idx: li },
            Value::RegionInput { region: rr, input_idx: ri },
        ) = (lhs_base, rhs_base)
        && lr == body
        && rr == body
        && li == ri
    {
        let init_composite = *initial_inputs.get(li as usize)?;
        let init_val = extract_composite_field_const(cx, init_composite, lhs_field)?;
        let bound = extract_composite_field_const(cx, init_composite, rhs_field)?;
        return u64_trip_count(init_val, bound, is_ne);
    }

    // handle `OpCompositeExtract(input, field) OP const` for scalar structs.
    if (is_lt || is_ne)
        && let Some((base, field)) = follow_composite_extract(func, *lhs)
        && let Value::RegionInput { region, input_idx } = base
        && region == body
    {
        let init_composite = *initial_inputs.get(input_idx as usize)?;
        let init_val = extract_composite_field_const(cx, init_composite, field)?;
        let upper = extract_u64_const(cx, *rhs)?;
        return u64_trip_count(init_val, upper, is_ne);
    }

    let upper = extract_u64_const(cx, *rhs)?;

    // `loop_input < N`  →  N+1 iterations (guard included).
    if let Value::RegionInput { region, input_idx } = lhs
        && *region == body
    {
        let init = initial_inputs.get(*input_idx as usize)?;
        let init_val = extract_u64_const(cx, *init)?;
        return u64_trip_count(init_val, upper, is_ne);
    }

    //  `IAdd(loop_input, 1) < N`  →  N iterations (no guard needed).
    if is_lt && let Value::DataInstOutput(iadd_inst) = lhs {
        let iadd = &*func.data_insts[*iadd_inst];
        let DataInstKind::SpvInst(ref iadd_op) = iadd.kind else {
            return None;
        };
        if iadd_op.opcode.name() != "OpIAdd" {
            return None;
        }
        let [a, b] = iadd.inputs.as_slice() else { return None };
        let (loop_var, step) = if is_loop_region_input(body, *a) {
            (*a, *b)
        } else if is_loop_region_input(body, *b) {
            (*b, *a)
        } else {
            return None;
        };
        if extract_u64_const(cx, step) != Some(1) {
            return None;
        }
        let Value::RegionInput { input_idx, .. } = loop_var else {
            return None;
        };
        let init_val = extract_u64_const(cx, *initial_inputs.get(input_idx as usize)?)?;
        return upper.checked_sub(init_val).and_then(|d| u32::try_from(d).ok());
    }

    None
}

/// compute trip count from 64-bit init/bound, returning `None` if the result
/// overflows `u32` (these loops will probably exceed `max_trip_count` anyway).
fn u64_trip_count(init: u64, bound: u64, is_ne: bool) -> Option<u32> {
    let diff = bound.checked_sub(init)?;
    let tc = if is_ne { diff } else { diff.checked_add(1)? };
    u32::try_from(tc).ok()
}

/// scan top-level `Block` `DataInsts` of `body` for a usable comparison.
fn trip_count_from_body_scan(
    cx: &Context,
    func: &FuncDefBody,
    body: Region,
    initial_inputs: &[Value],
) -> Option<u32> {
    let mut iter = func.regions[body].children.iter();
    while let Some((node, rest)) = iter.split_first(&func.nodes) {
        iter = rest;
        let NodeKind::Block { insts } = func.nodes[node].kind else { continue };
        let mut it = insts.iter();
        while let Some((inst, ir)) = it.split_first(&func.data_insts) {
            it = ir;
            if let Some(tc) =
                try_trip_count_from_cmp(cx, func, &func.data_insts[inst], body, initial_inputs)
            {
                return Some(tc);
            }
        }
    }
    None
}

fn is_loop_region_input(body: Region, v: Value) -> bool {
    matches!(v, Value::RegionInput { region, .. } if region == body)
}

fn is_const_false(cx: &Context, v: Value) -> bool {
    let Value::Const(ct) = v else { return false };
    let ConstKind::SpvInst { ref spv_inst_and_const_inputs } = cx[ct].kind else {
        return false;
    };
    let (spv_inst, _) = &**spv_inst_and_const_inputs;
    spv_inst.opcode.name() == "OpConstantFalse"
}

fn extract_u64_const(cx: &Context, v: Value) -> Option<u64> {
    let Value::Const(ct) = v else { return None };
    let ConstKind::SpvInst { ref spv_inst_and_const_inputs } = cx[ct].kind else {
        return None;
    };
    let (spv_inst, _) = &**spv_inst_and_const_inputs;
    if spv_inst.opcode.name() != "OpConstant" {
        return None;
    }
    match spv_inst.imms.as_slice() {
        // 32-bit constant — widen so callers can use one function for both widths.
        [spv::Imm::Short(_, lo)] => Some(*lo as u64),
        // 64-bit constant: lo word first, hi word second.
        [spv::Imm::LongStart(_, lo), spv::Imm::LongCont(_, hi)] => {
            Some((*lo as u64) | ((*hi as u64) << 32))
        }
        _ => None,
    }
}

fn follow_composite_extract(func: &FuncDefBody, v: Value) -> Option<(Value, u32)> {
    let Value::DataInstOutput(inst) = v else { return None };
    let def = &*func.data_insts[inst];
    let DataInstKind::SpvInst(ref spv_inst) = def.kind else { return None };
    if spv_inst.opcode.name() != "OpCompositeExtract" {
        return None;
    }
    let [base] = def.inputs.as_slice() else { return None };
    let field = match spv_inst.imms.as_slice() {
        [spv::Imm::Short(_, idx)] => *idx,
        _ => return None,
    };
    Some((*base, field))
}

/// extract `field` from an `OpConstantComposite` constant.
fn extract_composite_field_const(cx: &Context, v: Value, field: u32) -> Option<u64> {
    let Value::Const(ct) = v else { return None };
    let ConstKind::SpvInst { ref spv_inst_and_const_inputs } = cx[ct].kind else {
        return None;
    };
    let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
    if spv_inst.opcode.name() != "OpConstantComposite" {
        return None;
    }
    let elem = const_inputs.get(field as usize)?;
    extract_u64_const(cx, Value::Const(*elem))
}

/// replace `loop_node` in `parent_region.children` with `trip_count` inlined
/// copies of the loop body, each wired up with outputs from the previous copy.
fn inline_unrolled_iterations(
    cx: &Context,
    func: &mut FuncDefBody,
    parent_region: Region,
    loop_node: Node,
    trip_count: u32,
) {
    let (initial_inputs, body) = {
        let NodeKind::Loop { ref initial_inputs, body, .. } = func.nodes[loop_node].kind else {
            unreachable!()
        };
        (initial_inputs.clone(), body)
    };

    let mut current_values: Vec<Value> = initial_inputs.into_iter().collect();

    for _ in 0..trip_count {
        let mut value_map = FxHashMap::default();
        for (idx, &v) in current_values.iter().enumerate() {
            value_map.insert(Value::RegionInput { region: body, input_idx: idx as u32 }, v);
        }
        let body_outputs =
            clone_region_children_before(cx, func, body, parent_region, loop_node, &mut value_map);
        current_values = body_outputs;
    }

    func.regions[parent_region].children.remove(loop_node, &mut func.nodes);
}

fn clone_region_children_before(
    cx: &Context,
    func: &mut FuncDefBody,
    src_region: Region,
    dest_region: Region,
    before_node: Node,
    value_map: &mut FxHashMap<Value, Value>,
) -> Vec<Value> {
    let children: Vec<Node> = collect_children(func, src_region);
    for old_node in children {
        let new_node = clone_control_node(cx, func, old_node, value_map);
        func.regions[dest_region].children.insert_before(new_node, before_node, &mut func.nodes);
    }
    let outputs = func.regions[src_region].outputs.clone();
    outputs.iter().map(|v| map_value(*v, value_map)).collect()
}

fn clone_control_node(
    cx: &Context,
    func: &mut FuncDefBody,
    old_node: Node,
    value_map: &mut FxHashMap<Value, Value>,
) -> Node {
    let old_def: NodeDef = (*func.nodes[old_node]).clone();

    let new_kind = match old_def.kind {
        NodeKind::Block { insts } => {
            NodeKind::Block { insts: clone_data_inst_list(cx, func, insts, value_map) }
        }
        NodeKind::Select { kind, scrutinee, cases } => NodeKind::Select {
            kind,
            scrutinee: map_value(scrutinee, value_map),
            cases: cases.iter().map(|&c| clone_region_standalone(cx, func, c, value_map)).collect(),
        },
        NodeKind::Loop { initial_inputs, body, repeat_condition } => NodeKind::Loop {
            initial_inputs: initial_inputs.iter().map(|&v| map_value(v, value_map)).collect(),
            body: clone_region_standalone(cx, func, body, value_map),
            repeat_condition: map_value(repeat_condition, value_map),
        },
        NodeKind::ExitInvocation { kind, inputs } => NodeKind::ExitInvocation {
            kind,
            inputs: inputs.iter().map(|&v| map_value(v, value_map)).collect(),
        },
    };

    let new_node =
        func.nodes.define(cx, NodeDef { kind: new_kind, outputs: old_def.outputs.clone() }.into());

    for output_idx in 0..old_def.outputs.len() as u32 {
        value_map.insert(
            Value::NodeOutput { node: old_node, output_idx },
            Value::NodeOutput { node: new_node, output_idx },
        );
    }
    new_node
}

fn clone_region_standalone(
    cx: &Context,
    func: &mut FuncDefBody,
    src: Region,
    outer_map: &mut FxHashMap<Value, Value>,
) -> Region {
    let input_decls: SmallVec<[RegionInputDecl; 2]> = func.regions[src].inputs.clone();
    let src_outputs: SmallVec<[Value; 2]> = func.regions[src].outputs.clone();

    let new_region = func.regions.define(
        cx,
        RegionDef {
            inputs: input_decls.clone(),
            children: EntityList::empty(),
            outputs: SmallVec::new(),
        },
    );

    let mut inner_map = outer_map.clone();
    for (idx, _) in input_decls.iter().enumerate() {
        inner_map.insert(
            Value::RegionInput { region: src, input_idx: idx as u32 },
            Value::RegionInput { region: new_region, input_idx: idx as u32 },
        );
    }

    for old_node in collect_children(func, src) {
        let new_node = clone_control_node(cx, func, old_node, &mut inner_map);
        func.regions[new_region].children.insert_last(new_node, &mut func.nodes);
    }

    func.regions[new_region].outputs =
        src_outputs.iter().map(|&v| map_value(v, &inner_map)).collect();

    for (k, v) in &inner_map {
        outer_map.entry(*k).or_insert(*v);
    }
    new_region
}

fn clone_data_inst_list(
    cx: &Context,
    func: &mut FuncDefBody,
    insts: EntityList<DataInst>,
    value_map: &mut FxHashMap<Value, Value>,
) -> EntityList<DataInst> {
    let mut new_list = EntityList::empty();
    let mut iter = insts.iter();

    while let Some((inst, rest)) = iter.split_first(&func.data_insts) {
        iter = rest;
        let old: DataInstDef = (*func.data_insts[inst]).clone();
        let new_inputs = old.inputs.iter().map(|&v| map_value(v, value_map)).collect();
        let new_inst = func.data_insts.define(
            cx,
            DataInstDef {
                attrs: old.attrs,
                kind: old.kind,
                inputs: new_inputs,
                output_type: old.output_type,
            }
            .into(),
        );
        new_list.insert_last(new_inst, &mut func.data_insts);
        value_map.insert(Value::DataInstOutput(inst), Value::DataInstOutput(new_inst));
    }
    new_list
}

fn collect_children(func: &FuncDefBody, region: Region) -> Vec<Node> {
    let mut v = Vec::new();
    let mut iter = func.regions[region].children.iter();
    while let Some((node, rest)) = iter.split_first(&func.nodes) {
        iter = rest;
        v.push(node);
    }
    v
}

fn map_value(v: Value, m: &FxHashMap<Value, Value>) -> Value {
    *m.get(&v).unwrap_or(&v)
}
