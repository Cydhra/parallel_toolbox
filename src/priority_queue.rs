use crate::util::select_sample;
use mpi::topology::SystemCommunicator;
use mpi::traits::Communicator;
use std::cmp::{max, min};
use crate::p_inefficient_sort;

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
            lower_bucket.append(&mut select_k(&mut upper_bucket, k - lower_bucket.len()));
            lower_bucket
        } else {
            lower_bucket.append(&mut select_k(&mut middle_bucket, k - lower_bucket.len()));
            lower_bucket
        }
    }
}

/// Parallel adaption of Blum et al.'s Las-Vegas-algorithm to select the k smallest elements in a
/// distributed set. Chernoff bound guarantees constant number of recursions with high probability.
/// Guarantees that all data is still present in input array, but may be sorted.
///
pub fn p_select_k(comm: &SystemCommunicator, data: &mut [u64], k: usize) {
    let world_size = comm.size() as usize;
    let sample_size = 8; // tuning parameter
    let delta = 2; // tuning parameter
    let mut sample = vec![0; sample_size];

    select_sample(data, &mut sample);

    // todo instead of inefficient sort, redistribute and then complicated bound selection, select
    //  the bound while sorting, OR use ranking instead and broadcast the bounds
    p_inefficient_sort(comm, &mut sample);

    let ratio = k as f64 / data.len() as f64;
    let bound = (ratio * LOCAL_SORT_THRESHOLD as f64) as usize;

    // local bucket sort

    // select elements using recursion and inefficient sorting
}

#[cfg(test)]
mod tests {
    use super::select_k;
    use super::LOCAL_SORT_THRESHOLD;
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
}
