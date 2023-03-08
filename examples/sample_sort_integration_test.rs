use mpi::traits::{Communicator, Destination, Source};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use std::time::Instant;

use parallel_toolbox::p_sample_sort;

fn main() {
    // generate random data
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..u64::MAX);
    let initial_data_amount = 1 << 22;

    let mut data = vec![0u64; initial_data_amount];
    for i in 0..data.len() {
        data[i] = rng.sample(&uniform);
    }

    // initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let world_size = world.size() as usize;

    // call sorting algorithm
    let start = Instant::now();
    let data = p_sample_sort(&world, &data, data.len() * world_size);
    println!(
        "process {} sorted {} of {} qwords in {} ms",
        rank,
        data.len(),
        world_size * initial_data_amount,
        start.elapsed().as_millis()
    );

    for i in 0..data.len() - 1 {
        assert!(data[i] <= data[i + 1]);
    }

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
