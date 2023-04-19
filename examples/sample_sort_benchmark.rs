use std::mem::size_of;
use criterion::{BenchmarkId, Criterion, Throughput};
use mpi::collective::Root;
use mpi::traits::Communicator;
use mpirion::*;

use parallel_toolbox::sample_quick_sort;

mod common;

fn sample_sort_bench(c: &mut Criterion, world: &dyn Communicator) {
    let mut group = c.benchmark_group("sample_sort");
    group.sample_size(10);

    static MB: usize = 1_000_000;

    for size in [1*MB, 2*MB, 4*MB, 8*MB, 16*MB, 32*MB, 64*MB, 128*MB, 256*MB] {
        group.throughput(Throughput::Bytes((size * size_of::<u64>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("throughput", format!("{}MB", 4 * size / MB)),
            &size,
            |b, &size| {
                mpirion_bench!(sample_sort_kernel, b, world, size);
            }
        );
    }

    group.finish();
}

fn setup(_comm: &dyn Communicator, size: usize) -> Vec<u64> {
    common::generate_random_data(size)
}

fn sample_sort_kernel(comm: &dyn Communicator, data: &mut Vec<u64>) {
    let length = data.len();
    sample_quick_sort(comm, data, length * comm.size() as usize);
}

mpirion_kernel!(sample_sort_kernel, setup, usize);
mpirion_group!(benches, sample_sort_bench);
mpirion_main!(benches, sample_sort_kernel);
