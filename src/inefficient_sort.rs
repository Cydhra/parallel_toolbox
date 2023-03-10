use std::borrow::Borrow;

use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::SystemCommunicator;
use mpi::traits::{Communicator, Root};

/// A very inefficient sorting algorithm that just gathers all data, sorts it locally and
/// re-distributes it. This is technically worse than theoretical alternatives like
/// fast-inefficient-sort (which requires p = n² processors) and matrix-sort (which requires a
/// square number of processors), but this algorithm is applicable regardless of number of processors.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data of this process. Must not be very much and must be of equal size on all
/// processes. The buffer will be overwritten with the sorted data
pub fn p_inefficient_sort(comm: &SystemCommunicator, data: &mut [u64]) {
    let rank = comm.rank() as usize;
    let world_size = comm.size() as usize;
    let mut recv_buffer = vec![0u64; data.len() * world_size];
    let root = comm.process_at_rank(0);

    if rank == 0 {
        root.gather_into_root(data, &mut recv_buffer);
        recv_buffer.sort_unstable();
        root.scatter_into_root(&recv_buffer, data);
    } else {
        root.gather_into(data);
        root.scatter_into(data);
    }
}

/// A very inefficient ranking algorithm that just gathers all data, ranks it locally, and reports
/// rankings back to the processing units. Input data need not be of equal length
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data of this process
/// - `ranking` output parameter for the ranking
pub fn p_inefficient_rank(comm: &SystemCommunicator, data: &[u64], ranking: &mut [u64]) {
    assert_eq!(data.len(), ranking.len());

    let world_size = comm.size() as usize;
    let rank = comm.rank() as usize;
    let root_process = comm.process_at_rank(0);

    // collect counts and displs
    if rank == 0 {
        let mut counts: Vec<i32> = vec![0; world_size];
        root_process.gather_into_root(&(data.len() as i32), &mut counts);

        let displs: Vec<i32> = counts
            .iter()
            .scan(0, |acc, i| {
                let tmp = *acc;
                *acc += *i;
                Some(tmp)
            })
            .collect();

        let mut all_data_vec =
            vec![0u64; (displs[displs.len() - 1] + counts[counts.len() - 1]) as usize];
        let mut all_data = PartitionMut::new(&mut all_data_vec, counts.borrow(), displs.borrow());

        root_process.gather_varcount_into_root(data, &mut all_data);

        let mut sorted_data = all_data_vec.iter().cloned().collect::<Vec<_>>();
        sorted_data.sort_unstable();

        for i in 0..all_data_vec.len() {
            all_data_vec[i] = sorted_data.partition_point(|x| *x < all_data_vec[i]) as u64;
        }

        let all_data = Partition::new(&all_data_vec, counts.borrow(), displs.borrow());
        root_process.scatter_varcount_into_root(&all_data, ranking);
    } else {
        root_process.gather_into(&(data.len() as i32));
        root_process.gather_varcount_into(data);
        root_process.scatter_varcount_into(ranking);
    }
}

pub fn p_matrix_sort(comm: &SystemCommunicator, data: &mut [u64]) {
    let rank = comm.rank() as usize;
    let world_size = comm.size() as usize;
    let matrix_size = f64::sqrt(world_size as f64) as usize;
    assert_eq!(
        matrix_size * matrix_size,
        world_size,
        "matrix sort only works if the processor count is a square number"
    );

    let row = rank / matrix_size;
    let column = rank % matrix_size;

    // todo split comm into groups
}

#[cfg(test)]
mod tests {
    use crate::p_inefficient_sort;

    #[test]
    fn test_inefficient_sort() {
        let mut data = [234, 23, 4, 234, 23, 4, 234, 23, 2, 1362, 6, 1, 36, 7];
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        p_inefficient_sort(&world, &mut data);
        let expected = [1, 2, 4, 4, 6, 7, 23, 23, 23, 36, 234, 234, 234, 1362u64];
        assert_eq!(expected.len(), data.len());
        expected
            .iter()
            .zip(data.iter())
            .for_each(|(i, j)| assert_eq!(*i, *j));
    }
}
