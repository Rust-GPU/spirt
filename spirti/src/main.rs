use std::path::Path;

fn main() {
    match &std::env::args().collect::<Vec<_>>()[..] {
        [_, in_file] => {
            spirti::run_from_file(Path::new(in_file).to_path_buf(), None);
        }
        [_, in_file, out_file] => {
            spirti::run_from_file(
                Path::new(in_file).to_path_buf(),
                Some(Path::new(out_file).to_path_buf()),
            );
        }
        args => {
            eprintln!("Usage: {} IN_FILE [OUT_FILE]", args[0]);
            std::process::exit(1);
        }
    }
}
