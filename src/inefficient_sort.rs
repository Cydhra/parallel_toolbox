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
    let mut recv_buffer = vec![0, data.len() * world_size];

    if rank == 0 {
        comm.this_process().gather_into_root(data, &mut recv_buffer);
        recv_buffer.sort_unstable();
        comm.this_process().scatter_into_root(&recv_buffer, data);
    } else {
        comm.process_at_rank(0i32).gather_into(data);
        comm.process_at_rank(0i32).scatter_into(data);
    }
}