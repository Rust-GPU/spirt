use std::collections::HashMap;
use std::rc::Rc;

use spirt::spv;

struct Block {
    label: spv::Id,
    insts: Vec<spv::InstWithIds>,
}

fn lifted_blocks_from_fixture() -> Vec<Block> {
    let cx = Rc::new(spirt::Context::new());
    let mut module = spirt::Module::lower_from_spv_bytes(
        cx,
        include_bytes!("data/control_flow_mem2reg_undef_rust.spvbin").to_vec(),
    )
    .unwrap();

    spirt::passes::link::minimize_exports(&mut module, |export_key| {
        matches!(export_key, spirt::ExportKey::SpvEntryPoint { .. })
    });
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    spirt::passes::link::resolve_imports(&mut module);

    let emitted = module.lift_to_spv_module_emitter().unwrap();
    let spv_bytes = emitted.words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let parser = spv::read::ModuleParser::read_from_spv_bytes(spv_bytes).unwrap();

    let wk = &spv::spec::Spec::get().well_known;

    let mut in_first_function = false;
    let mut blocks = Vec::new();
    let mut current: Option<Block> = None;

    for inst in parser {
        let inst = inst.unwrap();

        if inst.opcode == wk.OpFunction {
            if in_first_function {
                continue;
            }
            in_first_function = true;
            continue;
        }
        if !in_first_function {
            continue;
        }
        if inst.opcode == wk.OpFunctionEnd {
            if let Some(block) = current.take() {
                blocks.push(block);
            }
            break;
        }
        if inst.opcode == wk.OpLabel {
            if let Some(block) = current.take() {
                blocks.push(block);
            }
            current = Some(Block { label: inst.result_id.unwrap(), insts: Vec::new() });
            continue;
        }

        if let Some(block) = &mut current {
            block.insts.push(inst);
        }
    }

    blocks
}

fn block_is_trivial_branch_to(wk: &spv::spec::WellKnown, block: &Block, target: spv::Id) -> bool {
    matches!(block.insts.as_slice(), [terminator] if terminator.opcode == wk.OpBranch && terminator.ids.as_slice() == [target])
}

fn block_terminator(block: &Block) -> Option<&spv::InstWithIds> {
    block.insts.last()
}

fn find_bad_loop_continue_shortcut_shape(blocks: &[Block]) -> bool {
    let wk = &spv::spec::Spec::get().well_known;

    let by_label = blocks.iter().map(|b| (b.label, b)).collect::<HashMap<_, _>>();

    let loop_merge_and_continue = |block: &Block| {
        block.insts.iter().find(|inst| inst.opcode == wk.OpLoopMerge).and_then(|inst| {
            match inst.ids.as_slice() {
                [loop_merge, loop_continue] => Some((*loop_merge, *loop_continue)),
                _ => None,
            }
        })
    };

    let selection_merge = |block: &Block| {
        block.insts.iter().find(|inst| inst.opcode == wk.OpSelectionMerge).and_then(|inst| {
            match inst.ids.as_slice() {
                [merge] => Some(*merge),
                _ => None,
            }
        })
    };

    for header in blocks {
        let Some((loop_merge, loop_continue)) = loop_merge_and_continue(header) else {
            continue;
        };

        let Some(header_term) = block_terminator(header) else {
            continue;
        };
        if !(header_term.opcode == wk.OpBranch && header_term.ids.len() == 1) {
            continue;
        }
        let body = header_term.ids[0];

        let Some(body_block) = by_label.get(&body) else {
            continue;
        };
        let Some(body_merge) = selection_merge(body_block) else {
            continue;
        };
        let Some(body_term) = block_terminator(body_block) else {
            continue;
        };
        if !(body_term.opcode == wk.OpBranchConditional && body_term.ids.len() == 3) {
            continue;
        }

        let body_cond = body_term.ids[0];
        let body_t0 = body_term.ids[1];
        let body_t1 = body_term.ids[2];

        let pass_target = if by_label
            .get(&body_t0)
            .is_some_and(|b| block_is_trivial_branch_to(wk, b, body_merge))
        {
            body_t0
        } else if by_label
            .get(&body_t1)
            .is_some_and(|b| block_is_trivial_branch_to(wk, b, body_merge))
        {
            body_t1
        } else {
            continue;
        };

        let Some(body_merge_block) = by_label.get(&body_merge) else {
            continue;
        };
        let Some(body_merge_term) = block_terminator(body_merge_block) else {
            continue;
        };
        if !(body_merge_term.opcode == wk.OpBranch
            && body_merge_term.ids.as_slice() == [loop_continue])
        {
            continue;
        }

        let Some(loop_continue_block) = by_label.get(&loop_continue) else {
            continue;
        };
        let Some(loop_continue_term) = block_terminator(loop_continue_block) else {
            continue;
        };
        if !(loop_continue_term.opcode == wk.OpBranchConditional
            && loop_continue_term.ids.len() == 3
            && loop_continue_term.ids[0] == body_cond)
        {
            continue;
        }

        let continue_targets = [loop_continue_term.ids[1], loop_continue_term.ids[2]];
        if !continue_targets.contains(&header.label) || !continue_targets.contains(&loop_merge) {
            continue;
        }

        eprintln!(
            "found shortcut shape: header=%{:?} body=%{:?} pass=%{:?} body_merge=%{:?} continue=%{:?} merge=%{:?}",
            header.label, body, pass_target, body_merge, loop_continue, loop_merge,
        );
        return true;
    }

    false
}

#[test]
fn no_loop_continue_shortcut_shape_after_lift() {
    let blocks = lifted_blocks_from_fixture();

    assert!(
        !find_bad_loop_continue_shortcut_shape(&blocks),
        "found loop-continue shortcut shape that triggers control_flow_mem2reg_undef miscompile"
    );
}
