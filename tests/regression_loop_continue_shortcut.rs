use std::collections::HashMap;
use std::rc::Rc;

use spirt::spv;

struct Block {
    label: spv::Id,
    insts: Vec<spv::InstWithIds>,
}

struct LiftedFixture {
    blocks: Vec<Block>,
    undef_ids: std::collections::BTreeSet<spv::Id>,
}

fn lifted_fixture_from_spv_fixture(spv_bytes: &[u8]) -> LiftedFixture {
    let cx = Rc::new(spirt::Context::new());
    let mut module = spirt::Module::lower_from_spv_bytes(cx, spv_bytes.to_vec()).unwrap();

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
    let mut undef_ids = std::collections::BTreeSet::new();

    for inst in parser {
        let inst = inst.unwrap();

        if inst.opcode == wk.OpUndef {
            if let Some(result_id) = inst.result_id {
                undef_ids.insert(result_id);
            }
        }

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

    LiftedFixture { blocks, undef_ids }
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

fn find_loop_carried_undef_from_shortcut(
    blocks: &[Block],
    undef_ids: &std::collections::BTreeSet<spv::Id>,
) -> bool {
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

        let (pass_target, _work_target) = if by_label
            .get(&body_t0)
            .is_some_and(|b| block_is_trivial_branch_to(wk, b, body_merge))
        {
            (body_t0, body_t1)
        } else if by_label
            .get(&body_t1)
            .is_some_and(|b| block_is_trivial_branch_to(wk, b, body_merge))
        {
            (body_t1, body_t0)
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

        let mut backedge_values = std::collections::BTreeSet::new();
        for inst in &header.insts {
            if inst.opcode != wk.OpPhi {
                continue;
            }
            for pair in inst.ids.chunks_exact(2) {
                let incoming_value = pair[0];
                let incoming_pred = pair[1];
                if incoming_pred == loop_continue {
                    backedge_values.insert(incoming_value);
                }
            }
        }

        for inst in &body_merge_block.insts {
            if inst.opcode != wk.OpPhi {
                continue;
            }
            let Some(phi_result) = inst.result_id else {
                continue;
            };
            if !backedge_values.contains(&phi_result) {
                continue;
            }

            for pair in inst.ids.chunks_exact(2) {
                let incoming_value = pair[0];
                let incoming_pred = pair[1];
                if incoming_pred != pass_target {
                    continue;
                }

                if undef_ids.contains(&incoming_value) {
                    eprintln!(
                        "found loop-carried undef via shortcut: header=%{:?} body=%{:?} pass=%{:?} body_merge=%{:?} continue=%{:?} phi=%{:?}",
                        header.label, body, pass_target, body_merge, loop_continue, phi_result
                    );
                    return true;
                }
            }
        }
    }

    false
}

#[test]
fn no_loop_continue_shortcut_shape_after_lift() {
    let fixtures: [(&str, &[u8]); 4] = [
        (
            "loop-continue-shortcut.repro",
            include_bytes!("data/loop-continue-shortcut.repro.spvbin"),
        ),
        (
            "loop-continue-shortcut-pretest-len.repro",
            include_bytes!("data/loop-continue-shortcut-pretest-len.repro.spvbin"),
        ),
        (
            "loop-continue-shortcut-nested.repro",
            include_bytes!("data/loop-continue-shortcut-nested.repro.spvbin"),
        ),
        (
            "loop-continue-shortcut-nested-len.repro",
            include_bytes!("data/loop-continue-shortcut-nested-len.repro.spvbin"),
        ),
    ];

    let mut offenders = Vec::new();
    for (name, spv_bytes) in fixtures {
        let fixture = lifted_fixture_from_spv_fixture(spv_bytes);
        if find_bad_loop_continue_shortcut_shape(&fixture.blocks) {
            offenders.push(name);
        }
    }

    assert!(
        offenders.is_empty(),
        "found loop-continue shortcut shape that can mis-handle loop edge values in fixtures: {offenders:?}"
    );
}

#[test]
fn detector_does_not_trigger_on_non_loop_fixture() {
    let fixtures: [(&str, &[u8]); 3] = [
        ("basic.frag.glsl.dbg", include_bytes!("data/basic.frag.glsl.dbg.spvbin")),
        (
            "loop-continue-shortcut-control-posttest",
            include_bytes!("data/loop-continue-shortcut-control-posttest.spvbin"),
        ),
        (
            "loop-continue-shortcut-control-noloop",
            include_bytes!("data/loop-continue-shortcut-control-noloop.spvbin"),
        ),
    ];

    let mut offenders = Vec::new();
    for (name, spv_bytes) in fixtures {
        let fixture = lifted_fixture_from_spv_fixture(spv_bytes);
        if find_bad_loop_continue_shortcut_shape(&fixture.blocks) {
            offenders.push(name);
        }
    }

    assert!(offenders.is_empty(), "detector matched control fixtures unexpectedly: {offenders:?}");
}

#[test]
fn no_loop_carried_undef_from_shortcut_after_lift() {
    let repro_fixtures: [(&str, &[u8]); 4] = [
        (
            "loop-continue-shortcut.repro",
            include_bytes!("data/loop-continue-shortcut.repro.spvbin"),
        ),
        (
            "loop-continue-shortcut-pretest-len.repro",
            include_bytes!("data/loop-continue-shortcut-pretest-len.repro.spvbin"),
        ),
        (
            "loop-continue-shortcut-nested.repro",
            include_bytes!("data/loop-continue-shortcut-nested.repro.spvbin"),
        ),
        (
            "loop-continue-shortcut-nested-len.repro",
            include_bytes!("data/loop-continue-shortcut-nested-len.repro.spvbin"),
        ),
    ];

    let mut offenders = Vec::new();
    for (name, spv_bytes) in repro_fixtures {
        let fixture = lifted_fixture_from_spv_fixture(spv_bytes);
        if find_loop_carried_undef_from_shortcut(&fixture.blocks, &fixture.undef_ids) {
            offenders.push(name);
        }
    }

    assert!(
        offenders.is_empty(),
        "found loop-carried undef values through shortcut fixtures: {offenders:?}"
    );

    let control_fixtures: [(&str, &[u8]); 3] = [
        ("basic.frag.glsl.dbg", include_bytes!("data/basic.frag.glsl.dbg.spvbin")),
        (
            "loop-continue-shortcut-control-posttest",
            include_bytes!("data/loop-continue-shortcut-control-posttest.spvbin"),
        ),
        (
            "loop-continue-shortcut-control-noloop",
            include_bytes!("data/loop-continue-shortcut-control-noloop.spvbin"),
        ),
    ];
    let mut false_positives = Vec::new();
    for (name, spv_bytes) in control_fixtures {
        let fixture = lifted_fixture_from_spv_fixture(spv_bytes);
        if find_loop_carried_undef_from_shortcut(&fixture.blocks, &fixture.undef_ids) {
            false_positives.push(name);
        }
    }
    assert!(
        false_positives.is_empty(),
        "semantic detector matched control fixtures unexpectedly: {false_positives:?}"
    );
}
