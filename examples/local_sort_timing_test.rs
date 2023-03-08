use std::time::Instant;

use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

fn main() {
    // generate random data
    let mut rng = thread_rng();
    let uniform = Uniform::from(0..u64::MAX);
    let initial_data_amount = 1 << 26;

    let mut data = vec![0u64; initial_data_amount];
    for i in 0..data.len() {
        data[i] = rng.sample(&uniform);
    }

    // call sorting algorithm
    let start = Instant::now();
    data.sort_unstable();
    println!(
        "single thread sorted {} qwords in {} ms",
        data.len(),
        start.elapsed().as_millis()
    );
}
