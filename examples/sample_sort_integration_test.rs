use mpi::traits::{Communicator, Destination, Source};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use std::env;
use std::process::exit;
use std::time::Instant;

use parallel_toolbox::{sample_quick_sort};

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
    let data = sample_quick_sort(&world, &mut data, initial_data_amount);
    println!(
        "process {} sorted {} of {} qwords in {} ms",
        rank,
        data.len(),
        initial_data_amount,
        start.elapsed().as_millis()
    );

    // check data is sorted locally
    for i in 0..data.len() - 1 {
        assert!(data[i] <= data[i + 1]);
    }

    // check data is sorted across PUs
    let mut next_elem = 0u64;
    mpi::request::scope(|scope| {
        let req = if rank > 0 {
            Some(
                world
                    .process_at_rank((rank - 1) as i32)
                    .immediate_send(scope, &data[0]),
            )
        } else {
            None
        };

        if rank < world_size - 1 {
            world
                .process_at_rank((rank + 1) as i32)
                .immediate_receive_into(scope, &mut next_elem)
                .wait();
        }

        req.and_then(|it| Some(it.wait()));
    });

    if rank < world_size - 1 {
        assert!(*data.last().unwrap() <= next_elem);
    }
}
