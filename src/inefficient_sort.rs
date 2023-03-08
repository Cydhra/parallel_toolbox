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
        expected.iter().zip(data.iter()).for_each(|(i, j)| assert_eq!(*i, *j));
    }
}