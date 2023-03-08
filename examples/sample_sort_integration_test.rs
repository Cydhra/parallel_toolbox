use mpi::traits::{Communicator, Destination, Source};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

use parallel_toolbox::p_sample_sort;

fn main() {
    // generate random data
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..u64::MAX);

    let mut data = vec![0u64; 1 << 12];
    for i in 0..data.len() {
        data[i] = rng.sample(&uniform);
    }

    // initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let world_size = world.size() as usize;

    let data = p_sample_sort(&world, &data, data.len() * world_size);

    for i in 0..data.len() - 1 {
        assert!(data[i] <= data[i + 1]);
    }

    let mut next_elem = 0u64;
    mpi::request::scope(|scope| {
        let req = if rank > 0 {
            Some(world.process_at_rank((rank - 1) as i32).immediate_send(scope, &data[0]))
        } else {
            None
        };

        if rank < world_size - 1 {
            world.process_at_rank((rank + 1) as i32).immediate_receive_into(scope, &mut next_elem).wait();
        }

        req.and_then(|it| Some(it.wait()));
    });

    if rank < world_size - 1 {
        assert!(*data.last().unwrap() <= next_elem);
    }
}