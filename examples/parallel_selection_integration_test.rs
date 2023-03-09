use mpi::datatype::PartitionMut;
use mpi::traits::{Communicator, CommunicatorCollectives};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use std::borrow::Borrow;
use std::env;
use std::process::exit;
use std::time::Instant;

use parallel_toolbox::p_select_k;

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

    // initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let world_size = world.size() as usize;

    // generate random data
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..u64::MAX);

    let mut data = Vec::with_capacity(initial_data_amount / world_size);
    for _ in 0..initial_data_amount / world_size {
        data.push(rng.sample(&uniform));
    }

    // call sorting algorithm
    let start = Instant::now();
    let selected_data = p_select_k(&world, &data, world_size);

    println!(
        "process {} chose {} of {} qwords in {} ms",
        rank,
        selected_data.len(),
        initial_data_amount,
        start.elapsed().as_millis()
    );

    // check that selected values are actually the smallest values
    let local_count = selected_data.len();
    let mut global_counts = vec![0; world_size];
    world.all_gather_into(&local_count, &mut global_counts);

    let displs: Vec<i32> = global_counts
        .iter()
        .scan(0i32, |acc, i| {
            let t = *acc;
            *acc += *i;
            Some(t)
        })
        .collect();

    let mut all_selected_data_vec = vec![0; world_size];
    let mut all_data = PartitionMut::new(
        &mut all_selected_data_vec,
        global_counts.borrow(),
        displs.borrow(),
    );
    world.all_gather_varcount_into(&selected_data, &mut all_data);

    all_selected_data_vec.sort_unstable();

    for n in data {
        assert!(n >= all_selected_data_vec[0]);
        if n < all_selected_data_vec[all_selected_data_vec.len() - 1] {
            assert!(all_selected_data_vec.contains(&n));
        }
    }
}
