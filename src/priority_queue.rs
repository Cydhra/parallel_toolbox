use crate::p_sample_sort;
use crate::util::select_sample;

const LOCAL_SORT_THRESHOLD: usize = 512;

/// Las-Vegas-algorithm to select the k smallest elements in a distributed set. Chernoff bound guarantees
/// constant number of recursions with high probability. Algorithm by Blum et al. 1972.
fn select_k(data: &mut [u64], k: usize) -> Vec<u64> {
    let delta = 16; // tuning parameter

    if data.len() < LOCAL_SORT_THRESHOLD {
        data.sort_unstable();
        return data[0..k].to_vec();
    }

    let mut sample = vec![0u64; LOCAL_SORT_THRESHOLD];
    select_sample(data, &mut sample);
    sample.sort_unstable();

    let ratio = k as f64 / data.len() as f64;
    let upper = sample[ratio * LOCAL_SORT_THRESHOLD as usize + delta];
    let lower = sample[ratio * LOCAL_SORT_THRESHOLD as usize - delta];

    let mut lower_bucket = Vec::with_capacity(k + delta);
    let mut middle_bucket = Vec::with_capacity(2 * delta);
    let mut upper_bucket = Vec::with_capacity(data.len() - k - delta);

    for n in data {
        if n < lower {
            lower_bucket.push(*n);
        } else if n > upper {
            upper_bucket.push(*n);
        } else {
            middle_bucket.push(*n);
        }
    }
    todo!()
}