use std::borrow::Borrow;

use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::SystemCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

const INEFFICIENT_SORT_THRESHOLD: usize = 512;

/// Sort a set of numerical data of which each processing unit has one part. The sorting algorithm
/// is optimized for distributed memory, i.e. data communication is kept to a minimum. After the
/// call, processing units have a sorted slice of data, which are all approximately of equal size, and
/// sorted in ascending order with respect to each processing unit's rank.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data local to this processor
/// - `total_data` total amount of data distributed over all processors. This value need not be
/// exact, however underestimating will lead to a worse distribution of data across processors, and
/// overestimating will slow down the algorithm.
pub fn p_sample_sort(comm: &SystemCommunicator, data: &[u64], total_data: usize) -> Vec<u64> {
    let processes = comm.size() as usize;
    let data_per_client: f32 = total_data as f32 / processes as f32;
    let sample_size = (16f32 * data_per_client.ln()) as usize;

    let mut local_sample = vec![0u64; sample_size];
    let mut pivots = vec![0u64; processes - 1];

    select_sample(data, &mut local_sample);
    select_pivots(comm, &mut local_sample, &mut pivots);

    // prepare p buffers for recipients
    let mut send_buffers = vec![Vec::with_capacity(data_per_client as usize); processes];

    // bucket sort
    for n in &*data {
        let j = pivots.partition_point(|&x| x < *n);
        send_buffers[j].push(*n)
    }

    // append all send buffers into a contiguous buffer for MPI, and store counts and displacements
    let mut partioned_buffer = Vec::with_capacity(data.len());
    let mut counts: Vec<i32> = Vec::with_capacity(processes);
    for mut buffer in send_buffers.into_iter() {
        counts.push(buffer.len() as i32);
        partioned_buffer.append(&mut buffer);
    }

    // calculate offsets in partitioned buffer
    let displs: Vec<i32> = counts.iter().scan(0, |acc, i| {
        let tmp = *acc;
        *acc += *i;
        Some(tmp)
    }).collect();

    // exchange counts and displacements to allow receiving prepared data
    let mut recv_counts: Vec<i32> = vec![0; processes];
    comm.all_to_all_into(&counts, &mut recv_counts);
    let recv_displs: Vec<i32> = recv_counts.iter().scan(0, |acc, i| {
        let tmp = *acc;
        *acc += *i;
        Some(tmp)
    }).collect();

    let mut recv_buffer = vec![0; recv_counts.iter().sum::<i32>() as usize];
    let mut recv_partition = PartitionMut::new(&mut recv_buffer, recv_counts.borrow(), recv_displs.borrow());

    // all to all exchange data and then quicksort it locally
    let partition = Partition::new(&partioned_buffer, counts.borrow(), displs.borrow());
    comm.all_to_all_varcount_into(&partition, &mut recv_partition);
    recv_buffer.sort_unstable();

    return recv_buffer;
}

/// Select a uniformly random sample from a dataset. The given buffer will be filled with the sample
///
/// # Parameters
/// - `data` data to select from
/// - `buf` output buffer which is filled with the sample
fn select_sample(data: &[u64], buf: &mut [u64]) {
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..data.len());

    for i in 0..buf.len() {
        buf[i] = data[rng.sample(uniform)]
    }
}

/// Select p - 1 pivots from a set of data samples that can then be used to perform a sample sort
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` data sample from which to select pivots
/// - `out` output buffer to store the pivots in. This must be a slice of length exactly equal to
/// the communicator's size minus one.
fn select_pivots(comm: &SystemCommunicator, data: &mut [u64], out: &mut [u64]) {
    let sample_len = data.len();
    let proc_count = comm.size() as usize;

    assert_eq!((out.len() + 1), proc_count, "output buffer must have capacity for number of processors minus 1");

    if data.len() <= INEFFICIENT_SORT_THRESHOLD {
        let root = comm.process_at_rank(0);

        // gather all samples on root, sort them, select pivots
        if comm.rank() == root.rank() {
            let mut buffer = vec![0u64; sample_len * proc_count];
            root.gather_into_root(data, &mut buffer);
            buffer.sort_unstable();

            for i in 0..(proc_count - 1) {
                out[i] = buffer[(i + 1) * sample_len];
            }
        } else {
            root.gather_into(data);
        }

        // broadcast pivots
        root.broadcast_into(out);
    } else {
        // sort sample, select pivots on each processor, gossip them
        let sorted_sample = p_sample_sort(comm, data, sample_len * proc_count);
        let local_pivot = sorted_sample[sorted_sample.len() - 1];
        comm.all_gather_into(&local_pivot, out);
    }
}

#[cfg(test)]
mod tests {
    use crate::p_sample_sort;

    #[test]
    fn test_sample_sort() {
        // this is just a sanity check. Since only one client is participating, the actual algorithm isn't tested

        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let mut data = [6, 30, 574, 16, 2342, 53, 5, 4935, 3, 4];
        let result = p_sample_sort(&world, &mut data, 10);
        let expected = [3, 4, 5, 6, 16, 30, 53, 574, 2342, 4935];

        assert_eq!(expected.len(), result.len(), "Result has wrong size");
        assert!(expected.iter().zip(result.iter()).all(|(a, b)| a == b), "Result does not match expected array");
    }
}