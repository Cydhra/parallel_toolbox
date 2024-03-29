use std::borrow::Borrow;

use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::{Buffer, BufferMut, Communicator, CommunicatorCollectives, Equivalence, Root};
use num::Zero;

use crate::util::select_sample;

// TODO perform some benchmarks to set this constant to an appropriate value
const INEFFICIENT_SORT_THRESHOLD: usize = 512;

/// Sort a set of numerical data of which each processing unit has one part. The sorting algorithm
/// is optimized for distributed memory, i.e. data communication is kept to a minimum. After the
/// call, processing units have a sorted slice of data, which are all approximately of equal size, and
/// sorted in ascending order with respect to each processing unit's rank.
///
/// The local sort is performed by quick sort. An alternative using radix sort is available with the
/// `rdxsort` feature.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data local to this processor
/// - `total_data` total amount of data distributed over all processors. This value need not be
/// exact, however underestimating will lead to a worse distribution of data across processors, and
/// overestimating will slow down the algorithm.
pub fn sample_quick_sort<T>(comm: &dyn Communicator, data: &mut [T], total_data: usize) -> Vec<T>
where
    T: Clone + Equivalence + Ord + Zero,
    [T]: Buffer + BufferMut,
    Vec<T>: BufferMut,
{
    sample_generic_sort(comm, data, total_data, <[T]>::sort_unstable)
}


/// Sort a set of numerical data of which each processing unit has one part. The sorting algorithm
/// is optimized for distributed memory, i.e. data communication is kept to a minimum. After the
/// call, processing units have a sorted slice of data, which are all approximately of equal size, and
/// sorted in ascending order with respect to each processing unit's rank.
///
/// The local sort is performed by radix sort rather than quick sort. This is faster for large
/// datasets, but slower for small datasets, especially with large data-types.
///
/// The initial sample sort is still performed by a comparative bucket sort, to ensure equal
/// distribution of data across processors.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data local to this processor
/// - `total_data` total amount of data distributed over all processors. This value need not be
/// exact, however underestimating will lead to a worse distribution of data across processors, and
/// overestimating will slow down the algorithm.
#[cfg(feature = "rdxsort")]
pub fn sample_radix_sort<T>(comm: &dyn Communicator, data: &mut [T], total_data: usize) -> Vec<T>
    where
        T: Clone + Equivalence + Ord + Zero + rdxsort::RdxSortTemplate,
        [T]: Buffer + BufferMut,
        Vec<T>: BufferMut,
{

    sample_generic_sort(comm, data, total_data, rdxsort::RdxSort::rdxsort)
}

/// Sort a set of numerical data of which each processing unit has one part. The sorting algorithm
/// is optimized for distributed memory, i.e. data communication is kept to a minimum. After the
/// call, processing units have a sorted slice of data, which are all approximately of equal size, and
/// sorted in ascending order with respect to each processing unit's rank.
///
/// The data is locally sorted by a sorting algorithm given as parameter. This allows for using
/// different sorting algorithms, such as radix sort. The default sorting algorithm is quick sort,
/// which is available using the convenience function `sample_quick_sort`, another option is radix
/// sort, which is available using the convenience function `sample_radix_sort` when using the
/// `rdxsort` feature.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data local to this processor
/// - `total_data` total amount of data distributed over all processors. This value need not be
/// exact, however underestimating will lead to a worse distribution of data across processors, and
/// overestimating will slow down the algorithm.
/// - `local_sort` sorting algorithm to use for sorting the local data. This function must sort the
/// data in place.
pub fn sample_generic_sort<T>(comm: &dyn Communicator, data: &mut [T], total_data: usize, local_sort: fn(&mut [T])) -> Vec<T>
    where
        T: Clone + Equivalence + Ord + Zero,
        [T]: Buffer + BufferMut,
        Vec<T>: BufferMut,
{
    let processes = comm.size() as usize;
    let data_per_client: f32 = total_data as f32 / processes as f32;
    let sample_size = (16f32 * data_per_client.ln()) as usize;

    let mut local_sample = vec![T::zero(); sample_size];
    let mut pivots = vec![T::zero(); processes - 1];

    select_sample(data, &mut local_sample);
    select_pivots(comm, &mut local_sample, &mut pivots);

    // prepare p buffers for recipients
    let mut send_buffers = vec![Vec::with_capacity(data_per_client as usize); processes];

    // bucket sort
    for n in &*data {
        let j = pivots.partition_point(|x| x.clone() < *n);
        send_buffers[j].push(n.clone())
    }

    // append all send buffers into a contiguous buffer for MPI, and store counts and displacements
    let mut counts: Vec<i32> = Vec::with_capacity(processes);
    let mut displs: Vec<i32> = Vec::with_capacity(processes);
    let mut offset: usize = 0;

    for buffer in send_buffers.into_iter() {
        counts.push(buffer.len() as i32);
        displs.push(offset as i32);

        data[offset..offset + buffer.len()].clone_from_slice(&buffer[..]);
        offset += buffer.len();
    }

    // exchange counts and displacements to allow receiving prepared data
    let mut recv_counts: Vec<i32> = vec![0; processes];
    comm.all_to_all_into(&counts, &mut recv_counts);
    let recv_displs: Vec<i32> = recv_counts
        .iter()
        .scan(0, |acc, i| {
            let tmp = *acc;
            *acc += *i;
            Some(tmp)
        })
        .collect();

    let mut recv_buffer = vec![T::zero(); recv_counts.iter().sum::<i32>() as usize];
    let mut recv_partition =
        PartitionMut::new(&mut recv_buffer, recv_counts.borrow(), recv_displs.borrow());

    // all to all exchange data and then quicksort it locally
    let partition = Partition::new(data, counts.borrow(), displs.borrow());
    comm.all_to_all_varcount_into(&partition, &mut recv_partition);
    local_sort(&mut recv_buffer);

    return recv_buffer;
}

/// Select p - 1 pivots from a set of data samples that can then be used to perform a sample sort
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` data sample from which to select pivots
/// - `out` output buffer to store the pivots in. This must be a slice of length exactly equal to
/// the communicator's size minus one.
fn select_pivots<T>(comm: &dyn Communicator, data: &mut [T], out: &mut [T])
where
    T: Clone + Equivalence + Ord + Zero,
    [T]: Buffer + BufferMut,
    Vec<T>: BufferMut,
{
    let sample_len = data.len();
    let proc_count = comm.size() as usize;

    assert_eq!(
        (out.len() + 1),
        proc_count,
        "output buffer must have capacity for number of processors minus 1"
    );

    if data.len() <= INEFFICIENT_SORT_THRESHOLD {
        let root = comm.process_at_rank(0);

        // gather all samples on root, sort them, select pivots
        if comm.rank() == root.rank() {
            let mut buffer = vec![T::zero(); sample_len * proc_count];
            root.gather_into_root(data, &mut buffer);
            buffer.sort_unstable();

            for i in 0..(proc_count - 1) {
                out[i] = buffer[(i + 1) * sample_len].clone();
            }
        } else {
            root.gather_into(data);
        }

        // broadcast pivots
        root.broadcast_into(out);
    } else {
        // sort sample, select pivots on each processor, gossip them
        let sorted_sample = sample_quick_sort(comm, data, sample_len * proc_count);
        let local_pivot = sorted_sample[sorted_sample.len() - 1].clone();
        comm.all_gather_into(&local_pivot, out);
    }
}

#[cfg(test)]
mod tests {
    use crate::sample_quick_sort;
    use rusty_fork::rusty_fork_test;

    rusty_fork_test! {
        #[test]
        fn test_sample_sort() {
            // this is just a sanity check. Since only one client is participating, the actual algorithm isn't tested

            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            let mut data = [6, 30, 574, 16, 2342, 53, 5, 4935, 3, 4];
            let result = sample_quick_sort(&world, &mut data, 10);
            let expected = [3, 4, 5, 6, 16, 30, 53, 574, 2342, 4935];

            assert_eq!(expected.len(), result.len(), "Result has wrong size");
            assert!(
                expected.iter().zip(result.iter()).all(|(a, b)| a == b),
                "Result does not match expected array"
            );
        }
    }

    #[cfg(feature = "rdxsort")]
    rusty_fork_test! {
        #[test]
        fn test_radix_sort() {
            // this is just a sanity check. Since only one client is participating, the actual algorithm isn't tested

            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            let mut data = [6, 30, 574, 16, 2342, 53, 5, 4935, 3, 4];
            let result = sample_quick_sort(&world, &mut data, 10);
            let expected = [3, 4, 5, 6, 16, 30, 53, 574, 2342, 4935];

            assert_eq!(expected.len(), result.len(), "Result has wrong size");
            assert!(
                expected.iter().zip(result.iter()).all(|(a, b)| a == b),
                "Result does not match expected array"
            );
        }
    }
}
