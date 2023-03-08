use mpi::datatype::PartitionMut;
use mpi::topology::SystemCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

const INEFFICIENT_SORT_THRESHOLD: usize = 128;

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
pub fn p_sample_sort(comm: &SystemCommunicator, data: &mut [u64], total_data: usize) -> Vec<u64> {
    let data_per_client: f32 = total_data as f32 / comm.size() as f32;
    let sample_size = (16f32 * data_per_client.ln()) as usize;
    let mut local_sample = vec![0u64; sample_size];
    let mut pivots = vec![0u64; (comm.size() - 1) as usize];

    select_sample(data, &mut local_sample);
    select_pivots(comm, &mut local_sample, &mut pivots);

    let mut send_buffers = vec![Vec::with_capacity(data_per_client as usize)];
    for n in data {
        let j = pivots.partition_point(|&x| x < *n);
        send_buffers[j].push(*n)
    }

    // todo the partitioning of data
    comm.all_to_all_varcount_into()

    todo!()
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
        let local_pivot = sorted_sample[sample_len - 1];
        comm.all_gather_into(&local_pivot, out);
    }
}