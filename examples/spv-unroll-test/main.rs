use std::path::Path;
use std::rc::Rc;

fn main() -> std::io::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let in_file = match args.as_slice() {
        [_, f] => f.clone(),
        _ => {
            eprintln!("usage: {} FILE.spv", args[0]);
            std::process::exit(1);
        }
    };
    let in_path = Path::new(&in_file);

    let cx = Rc::new(spirt::Context::new());
    let mut module = spirt::Module::lower_from_spv_file(cx, in_path)?;

    spirt::passes::legalize::structurize_func_cfgs(&mut module);

    println!("{}", spirt::print::Plan::for_module(&module).pretty_print());
    println!("new optimzied --------");

    spirt::passes::unroll::unroll_loops(&mut module);
    println!("{}", spirt::print::Plan::for_module(&module).pretty_print());

    Ok(())
}
