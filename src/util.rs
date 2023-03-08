use rand::distributions::Uniform;
use rand::{Rng, thread_rng};

/// Select a uniformly random sample from a dataset. The given buffer will be filled with the sample
///
/// # Parameters
/// - `data` data to select from
/// - `buf` output buffer which is filled with the sample
pub(crate) fn select_sample(data: &[u64], buf: &mut [u64]) {
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..data.len());

    for i in 0..buf.len() {
        buf[i] = data[rng.sample(uniform)]
    }
}