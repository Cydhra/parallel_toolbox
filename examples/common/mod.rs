use rand::distributions::Uniform;
use rand::{Rng, thread_rng};

/// Generates a vector of random data.
pub(crate) fn generate_random_data(amount: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..u64::MAX);

    let mut data = Vec::with_capacity(amount);
    for _ in 0..amount {
        data.push(rng.sample(&uniform));
    }

    data
}