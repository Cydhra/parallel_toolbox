use std::env;
use std::process::exit;
use std::time::Instant;

use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

fn main() {
    let args: Vec<String> = env::args().collect();
    let initial_data_amount = if args.len() > 2 {
        eprintln!("usage: {} [sample-size]", args[0]);
        exit(1);
    } else if args.len() == 2 {
        if args[1].eq_ignore_ascii_case("--help") {
            println!("usage: {} [sample-size]", args[0]);
            exit(0);
        }

        usize::from_str_radix(&args[1], 10).expect("first argument should be an integer")
    } else {
        1 << 26
    };

    // generate random data
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..u64::MAX);


    let mut data = Vec::with_capacity(initial_data_amount);
    for _ in 0..initial_data_amount {
        data.push(rng.sample(&uniform));
    }

    // call sorting algorithm
    let start = Instant::now();
    data.sort_unstable();
    println!(
        "single thread sorted {} qwords in {} ms",
        data.len(),
        start.elapsed().as_millis()
    );
}
