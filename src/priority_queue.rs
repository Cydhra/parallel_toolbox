use crate::util::select_sample;
use crate::p_inefficient_rank;
use mpi::collective::SystemOperation;
use mpi::topology::SystemCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence, Root};
use std::cmp::{max, min};

const LOCAL_SORT_THRESHOLD: usize = 512;

/// Las-Vegas-algorithm to select the k smallest elements in a distributed set. Chernoff bound guarantees
/// constant number of recursions with high probability. Algorithm by Blum et al. 1972.
/// Guarantees that all data is still present in input array, but may be sorted.
///
/// # Parameters
/// - `data` a mutable slice of data. May be sorted in-place, hence the mutability
/// - `k` amount of elements to select from `data`. Must not be greater than length of `data`.
///
/// # Returns
/// A vector of the k smallest elements of the input data slice. Not sorted.
pub fn select_k(data: &mut [u64], k: usize) -> Vec<u64> {
    assert!(data.len() >= k, "cannot select more elements than present");

    let delta = 16; // tuning parameter

    // sanity optimization
    if data.len() == k {
        return data.to_vec();
    }

    if data.len() < LOCAL_SORT_THRESHOLD {
        data.sort_unstable();
        return data[0..k].to_vec();
    }

    let mut sample = vec![0u64; LOCAL_SORT_THRESHOLD];
    select_sample(data, &mut sample);
    sample.sort_unstable();

    let ratio = k as f64 / data.len() as f64;
    let bound = (ratio * LOCAL_SORT_THRESHOLD as f64) as usize;

    let upper = sample[min(bound + delta, sample.len() - 1)];
    let lower = sample[max(bound as isize - delta as isize, 0) as usize];

    let mut lower_bucket = Vec::with_capacity(k + delta);
    let mut middle_bucket = Vec::with_capacity(2 * delta);
    let mut upper_bucket = Vec::with_capacity(data.len() - k - delta);

    for n in data {
        if *n < lower {
            lower_bucket.push(*n);
        } else if *n > upper {
            upper_bucket.push(*n);
        } else {
            middle_bucket.push(*n);
        }
    }

    if lower_bucket.len() > k {
        select_k(&mut lower_bucket, k)
    } else {
        if lower_bucket.len() + middle_bucket.len() < k {
            lower_bucket.append(&mut middle_bucket);
            lower_bucket.append(&mut select_k(
                &mut upper_bucket,
                k - lower_bucket.len() - middle_bucket.len(),
            ));
            lower_bucket
        } else {
            lower_bucket.append(&mut select_k(&mut middle_bucket, k - lower_bucket.len()));
            lower_bucket
        }
    }
}

/// Parallel adaption of Blum et al.'s Las-Vegas-algorithm to select the k smallest elements in a
/// distributed set. Chernoff bound guarantees constant number of recursions with high probability.
/// Input data must not be an empty slice.
pub fn p_select_k(comm: &SystemCommunicator, data: &[u64], k: usize) -> Vec<u64> {
    let sample_size = 8; // tuning parameter
    let delta = 2; // tuning parameter

    // check if the data amount is small enough to do a local search
    let mut global_size = 0;
    comm.all_reduce_into(&data.len(), &mut global_size, SystemOperation::sum());
    if global_size < LOCAL_SORT_THRESHOLD {
        let mut ranking = vec![0; data.len()];
        p_inefficient_rank(comm, data, &mut ranking);

        // todo this will return too many elements if some elements have the same rank
        return data
            .iter()
            .zip(ranking.iter())
            .filter(|(_, rank)| **rank < k as u64)
            .map(|(itm, _)| *itm)
            .collect();
    }

    // prepare local buckets and recurse on them
    let mut lower_bucket = Vec::new();
    let mut middle_bucket = Vec::new();
    let mut upper_bucket = Vec::new();

    if data.len() > 0 {
        let mut sample = vec![0; sample_size];
        select_sample(data, &mut sample);

        let ratio = k as f64 / data.len() as f64;
        let bound = (ratio * sample_size as f64) as usize;
        let (lower_pivot, upper_pivot) = local_pivot_search(comm, &sample, bound, delta);

        // local bucket sort
        for n in data {
            if *n < lower_pivot {
                lower_bucket.push(*n);
            } else if *n > upper_pivot {
                upper_bucket.push(*n);
            } else {
                middle_bucket.push(*n);
            }
        }
    } else {
        // participate at local pivot search without contribution, the contributed zeros are ignored in pivot search
        local_pivot_search(comm, &vec![0; sample_size], 0, 0);
    }

    let mut low_bucket_size = 0;
    let mut mid_bucket_size = 0;

    // todo replace with a single reduce that operates on vectors
    comm.all_reduce_into(
        &lower_bucket.len(),
        &mut low_bucket_size,
        SystemOperation::sum(),
    );
    comm.all_reduce_into(
        &middle_bucket.len(),
        &mut mid_bucket_size,
        SystemOperation::sum(),
    );

    if low_bucket_size > k {
        p_select_k(comm, &mut lower_bucket, k)
    } else {
        if low_bucket_size + mid_bucket_size < k {
            lower_bucket.append(&mut middle_bucket);
            lower_bucket.append(&mut p_select_k(
                comm,
                &mut upper_bucket,
                k - low_bucket_size - mid_bucket_size,
            ));
            lower_bucket
        } else {
            lower_bucket.append(&mut p_select_k(
                comm,
                &mut middle_bucket,
                k - low_bucket_size,
            ));
            lower_bucket
        }
    }
}

/// Adaption of inefficient sorting algorithm that collects all elements on one processor, but instead of
/// redistributing them, it will select the pivots for the selection algorithm and broadcast only those.
/// # Parameters
/// - `comm` mpi communicator
/// - `sample` sample buffer, must be of equal length in all clients
/// - `bound` the estimated k-bound in the sample
/// - `delta` the delta value added and subtracted from the estimated bound to gauge the location of the k boundary
fn local_pivot_search(
    comm: &SystemCommunicator,
    sample: &[u64],
    bound: usize,
    delta: usize,
) -> (u64, u64) {
    let rank = comm.rank() as usize;
    let world_size = comm.size() as usize;
    let mut recv_buffer = vec![0u64; sample.len() * world_size];

    let root = comm.process_at_rank(0);

    #[derive(Equivalence)]
    struct Pivots(u64, u64);
    let mut pivots = Pivots(0, 0);

    if rank == 0 {
        root.gather_into_root(sample, &mut recv_buffer);
        recv_buffer.sort_unstable();

        let mut lower_bound = max(bound as isize - delta as isize, 0) as usize;
        let mut upper_bound = min(bound + delta, sample.len() - 1);

        // if the buffer contains values greater than zero, dont accept zero as a pivot, which might happen if some
        // clients did not provide pivots
        if recv_buffer[recv_buffer.len() - 1] > 0 {
            while recv_buffer[lower_bound] == 0 {
                lower_bound += 1;
            }

            while upper_bound <= lower_bound && upper_bound < recv_buffer.len() - 1 {
                upper_bound += 1;
            }
        }

        pivots.0 = recv_buffer[lower_bound];
        pivots.1 = recv_buffer[upper_bound];

        root.broadcast_into(&mut pivots);
    } else {
        root.gather_into(sample);
        root.broadcast_into(&mut pivots);
    }

    (pivots.0, pivots.1)
}

#[cfg(test)]
mod tests {
    use super::select_k;
    use super::LOCAL_SORT_THRESHOLD;
    use crate::p_select_k;
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_select_k() {
        let select_single = select_k(&mut [1], 1);
        assert_eq!(1, select_single.len());
        assert_eq!(1, select_single[0]);

        let select_multiple = select_k(&mut [1, 10, 3, 5, 6, 1, 2], 2);
        assert_eq!(2, select_multiple.len());
        assert_eq!(1, select_multiple[0]);
        assert_eq!(1, select_multiple[1]);

        let select_none = select_k(&mut [], 0);
        assert_eq!(0, select_none.len());

        let mut rng = thread_rng();
        let uniform = Uniform::from(10..100);
        let mut data = Vec::with_capacity(2 * LOCAL_SORT_THRESHOLD);
        for _ in 0..2 * LOCAL_SORT_THRESHOLD {
            data.push(rng.sample(&uniform));
        }
        data.push(1);
        data.push(2);
        data.push(3);
        let smallest = select_k(&mut data, 3);
        assert_eq!(3, smallest.len());
        assert!(smallest.contains(&1) && smallest.contains(&2) && smallest.contains(&3))
    }

    /// These are some sanity checks. Since only one process is used, the parallel features aren't tested
    #[test]
    fn test_p_select_k() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let select_single = p_select_k(&world, &[1], 1);
        assert_eq!(1, select_single.len());
        assert_eq!(1, select_single[0]);

        let select_multiple = p_select_k(&world, &[1, 10, 3, 5, 6, 1, 2], 2);
        assert_eq!(2, select_multiple.len());
        assert_eq!(1, select_multiple[0]);
        assert_eq!(1, select_multiple[1]);

        let select_none = p_select_k(&world, &[], 0);
        assert_eq!(0, select_none.len());

        let mut rng = thread_rng();
        let uniform = Uniform::from(10..100);
        let mut data = Vec::with_capacity(2 * LOCAL_SORT_THRESHOLD);
        for _ in 0..2 * LOCAL_SORT_THRESHOLD {
            data.push(rng.sample(&uniform));
        }
        data.push(1);
        data.push(2);
        data.push(3);
        let smallest = p_select_k(&world, &data, 3);
        assert_eq!(3, smallest.len());
        assert!(smallest.contains(&1) && smallest.contains(&2) && smallest.contains(&3))
    }
}
