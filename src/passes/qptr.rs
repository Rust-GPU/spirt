//! [`QPtr`](crate::TypeKind::QPtr) transforms.

use crate::visit;
use crate::{DeclDef, Module, qptr};

pub fn lower_from_spv_ptrs(module: &mut Module, layout_config: &crate::mem::LayoutConfig) {
    let cx = &module.cx();

    // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
    let visit::AllUses { global_vars, funcs, .. } = visit::AllUses::from_module(module);

    let lowerer = qptr::lower::LowerFromSpvPtrs::new(cx.clone(), layout_config);
    for &global_var in &global_vars {
        lowerer.lower_global_var(&mut module.global_vars[global_var]);
    }
    for &func in &funcs {
        lowerer.lower_func(&mut module.funcs[func]);
    }
}

// FIXME(eddyb) split this into separate passes, but the looping complicates things.
pub fn partition_and_propagate(module: &mut Module, layout_config: &crate::mem::LayoutConfig) {
    let cx = &module.cx();

    // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
    let visit::AllUses { funcs, .. } = visit::AllUses::from_module(module);

    for func in funcs {
        if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
            // FIXME(eddyb) reuse `LayoutCache` and whatnot, between functions,
            // or at least iterations of this loop.
            loop {
                qptr::simplify::partition_locals_in_func(cx.clone(), layout_config, func_def_body);

                let report = qptr::simplify::propagate_contents_of_locals_in_func(
                    cx.clone(),
                    layout_config,
                    func_def_body,
                );
                if !report.any_qptrs_propagated {
                    break;
                }
            }
        }
    }
}

// FIXME(eddyb) this doesn't really belong in `qptr`.
pub fn analyze_mem_accesses(module: &mut Module, layout_config: &crate::mem::LayoutConfig) {
    crate::mem::analyze::GatherAccesses::new(module.cx(), layout_config)
        .gather_accesses_in_module(module);
}

pub fn lift_to_spv_ptrs(module: &mut Module, layout_config: &crate::mem::LayoutConfig) {
    let cx = &module.cx();

    // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
    let visit::AllUses { global_vars, funcs, .. } = visit::AllUses::from_module(module);

    let lifter = qptr::lift::LiftToSpvPtrs::new(cx.clone(), layout_config);
    for &global_var in &global_vars {
        lifter.lift_global_var(&mut module.global_vars[global_var]);
    }
    lifter.lift_all_funcs(module, funcs);
}
