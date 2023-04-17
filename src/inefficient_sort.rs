use mpi::collective::SystemOperation;
use std::borrow::Borrow;

use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use mpi::Rank;

/// A very inefficient sorting algorithm that just gathers all data, sorts it locally and
/// re-distributes it. This is technically worse than the theoretical alternative and matrix-sort
/// (which requires two integers that divide the processor count),
/// but this algorithm is applicable with no knowledge of processor count.
/// This algorithm is more efficient than the variant that accepts a variable amount of data.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data of this process. Must not be very much and must be of equal size on all
/// processes. The buffer will be overwritten with the sorted data
pub fn inefficient_sort(comm: &dyn Communicator, data: &mut [u64]) {
    let rank = comm.rank() as usize;
    let world_size = comm.size() as usize;
    let mut recv_buffer = vec![0u64; data.len() * world_size];
    let root = comm.process_at_rank(0);

    if rank == 0 {
        root.gather_into_root(data, &mut recv_buffer);
        recv_buffer.sort_unstable();
        root.scatter_into_root(&recv_buffer, data);
    } else {
        root.gather_into(data);
        root.scatter_into(data);
    }
}

/// A very inefficient sorting algorithm that just gathers all data, sorts it locally and
/// re-distributes it. This is technically worse than the theoretical alternative and matrix-sort
/// (which requires two integers that divide the processor count),
/// but this algorithm is applicable with no knowledge of processor count.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data of this process.
///
/// # Returns
/// A vector containing the p-th part of the sorted data, where p is the count of processes
pub fn inefficient_sort_var(comm: &dyn Communicator, data: &[u64]) -> Vec<u64> {
    let rank = comm.rank() as usize;
    let world_size = comm.size() as usize;

    let data_size = data.len() as i32;
    if rank == 0 {
        // gather all data sizes and then gather all data
        let mut counts = vec![0i32; world_size];
        comm.this_process().gather_into_root(&data_size, &mut counts);

        let mut displs = vec![0i32; world_size];
        for i in 1..world_size {
            displs[i] = displs[i - 1] + counts[i - 1];
        }

        let mut recv_buffer =
            vec![0u64; (displs[world_size - 1] + counts[world_size - 1]) as usize];
        let mut partition = PartitionMut::new(&mut recv_buffer, counts.borrow(), displs.borrow());
        comm.this_process().gather_varcount_into_root(data, &mut partition);

        // sort the data
        recv_buffer.sort_unstable();

        // calculate new counts by dividing the data into equal parts and adding the remainder on
        // the first processes
        let mut counts = vec![(recv_buffer.len() / world_size) as i32; world_size];
        counts[0..recv_buffer.len() % world_size]
            .iter_mut()
            .for_each(|x| *x += 1);
        for i in 1..world_size {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
        let partition = Partition::new(&recv_buffer, counts.borrow(), displs.borrow());

        // scatter the sorted data back to the processes
        let mut data_size: i32 = 0;
        comm.this_process().scatter_into_root(&counts, &mut data_size);
        let mut data = vec![0u64; data_size as usize];
        comm.this_process().scatter_varcount_into_root(&partition, &mut data);
        data
    } else {
        comm.process_at_rank(0).gather_into(&data_size);
        comm.process_at_rank(0).gather_varcount_into(data);

        let mut data_size: i32 = 0;
        comm.process_at_rank(0).scatter_into(&mut data_size);
        let mut data = vec![0u64; data_size as usize];
        comm.process_at_rank(0).scatter_varcount_into(&mut data);
        data
    }
}

/// A very inefficient ranking algorithm that just gathers all data, ranks it locally, and reports
/// rankings back to the processing units. Input data length must be equal between all clients.
/// This algorithm requires less communication than the `inefficient_rank_var` alternative.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data of this process
/// - `ranking` output parameter for the ranking, must be of equal size as the data slice
pub fn inefficient_rank(comm: &dyn Communicator, data: &[u64], ranking: &mut [u64]) {
    assert_eq!(data.len(), ranking.len());

    let rank = comm.rank() as usize;
    let world_size = comm.size() as usize;
    let mut recv_buffer = vec![0u64; data.len() * world_size];
    let mut rank_buffer = Vec::with_capacity(recv_buffer.len());

    let root = comm.process_at_rank(0);

    if rank == 0 {
        root.gather_into_root(data, &mut recv_buffer);
        let mut unsorted_buffer = vec![0u64; recv_buffer.len()];
        unsorted_buffer.copy_from_slice(&recv_buffer);
        recv_buffer.sort_unstable();

        for n in unsorted_buffer {
            rank_buffer.push(recv_buffer.partition_point(|x| *x < n));
        }

        root.scatter_into_root(&rank_buffer, ranking);
    } else {
        root.gather_into(data);
        root.scatter_into(ranking);
    }
}

/// A very inefficient ranking algorithm that just gathers all data, ranks it locally, and reports
/// rankings back to the processing units. Input data length may vary between clients.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` partial data of this process
/// - `ranking` output parameter for the ranking
pub fn inefficient_rank_var(comm: &dyn Communicator, data: &[u64], ranking: &mut [u64]) {
    assert_eq!(data.len(), ranking.len());

    let world_size = comm.size() as usize;
    let rank = comm.rank() as usize;
    let root_process = comm.process_at_rank(0);

    // collect counts and displs
    if rank == 0 {
        let mut counts: Vec<i32> = vec![0; world_size];
        root_process.gather_into_root(&(data.len() as i32), &mut counts);

        let displs: Vec<i32> = counts
            .iter()
            .scan(0, |acc, i| {
                let tmp = *acc;
                *acc += *i;
                Some(tmp)
            })
            .collect();

        let mut all_data_vec =
            vec![0u64; (displs[displs.len() - 1] + counts[counts.len() - 1]) as usize];
        let mut all_data = PartitionMut::new(&mut all_data_vec, counts.borrow(), displs.borrow());

        root_process.gather_varcount_into_root(data, &mut all_data);

        let mut sorted_data = all_data_vec.iter().cloned().collect::<Vec<_>>();
        sorted_data.sort_unstable();

        for i in 0..all_data_vec.len() {
            all_data_vec[i] = sorted_data.partition_point(|x| *x < all_data_vec[i]) as u64;
        }

        let all_data = Partition::new(&all_data_vec, counts.borrow(), displs.borrow());
        root_process.scatter_varcount_into_root(&all_data, ranking);
    } else {
        root_process.gather_into(&(data.len() as i32));
        root_process.gather_varcount_into(data);
        root_process.scatter_varcount_into(ranking);
    }
}

/// Ranking algorithm that arranges processors in a matrix and ranks a subset of data per row. In the end, all ranks
/// are scattered to the input processors. All input slices must be of equal size for this algorithm to work.
/// If processors are not a square number, it will lead to undefined behavior and most likely result in memory
/// corruption.
///
/// # Parameters
/// - `comm` mpi communicator
/// - `data` small data slice, same length on all clients
/// - `ranks` output slice for rank data, same length as `data` slice.
pub fn matrix_rank(comm: &dyn Communicator, data: &[u64], ranks: &mut [u64]) {
    assert_eq!(data.len(), ranks.len());

    let rank = comm.rank() as usize;
    let world_size = comm.size() as usize;
    let matrix_size = f64::sqrt(world_size as f64) as usize;
    assert_eq!(
        matrix_size * matrix_size,
        world_size,
        "matrix sort only works if the processor count is a square number"
    );

    // split processors into cubic matrix pattern along rows and columns
    let row = rank / matrix_size;
    let column = rank % matrix_size;

    let row_group = comm
        .split_by_subgroup_collective(
            &comm.group().include(
                &(matrix_size * row..matrix_size * (row + 1))
                    .into_iter()
                    .map(|i| i as Rank)
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();

    let column_group = comm
        .split_by_subgroup_collective(
            &comm.group().include(
                &((column..column + world_size - matrix_size + 1)
                    .step_by(matrix_size)
                    .into_iter()
                    .map(|i| i as Rank)
                    .collect::<Vec<_>>()),
            ),
        )
        .unwrap();

    let mut column_data = vec![0u64; matrix_size * data.len()];
    let mut row_data = vec![0u64; matrix_size * data.len()];
    column_group.all_gather_into(data, &mut column_data);

    // broadcast the received data along the row i from the cell (i, i).
    if row == column {
        row_data.copy_from_slice(&column_data);
        row_group.this_process().broadcast_into(&mut row_data);
    } else {
        row_group
            .process_at_rank(row as Rank)
            .broadcast_into(&mut row_data);
    }

    // sort column data
    column_data.sort_unstable();

    // locally calculate ranking
    let mut row_ranks = Vec::with_capacity(row_data.len());
    for n in row_data {
        row_ranks.push(column_data.partition_point(|x| *x < n))
    }

    // reduce ranking for row i in cell (i, i) and then re-scatter it along the column so all clients receive their
    // final ranking
    let mut global_ranks = vec![0u64; row_ranks.len()];
    if column == row {
        row_group.this_process().reduce_into_root(
            &row_ranks,
            &mut global_ranks,
            SystemOperation::sum(),
        );
        column_group
            .this_process()
            .scatter_into_root(&global_ranks, ranks);
    } else {
        row_group
            .process_at_rank(row as Rank)
            .reduce_into(&row_ranks, SystemOperation::sum());
        column_group
            .process_at_rank(column as Rank)
            .scatter_into(ranks);
    }
}

/// Those are just some sanity checks for the algorithms.
/// They are not exhaustive and only check for obvious regressions.
#[cfg(test)]
mod tests {
    use crate::{
        inefficient_rank, inefficient_rank_var, inefficient_sort, inefficient_sort_var, matrix_rank,
    };
    use rusty_fork::rusty_fork_test;

    rusty_fork_test! {
        #[test]
        fn test_inefficient_sort() {
            let mut data = [234, 23, 4, 234, 23, 4, 234, 23, 2, 1362, 6, 1, 36, 7];
            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            inefficient_sort(&world, &mut data);
            let expected = [1, 2, 4, 4, 6, 7, 23, 23, 23, 36, 234, 234, 234, 1362u64];
            assert_eq!(expected.len(), data.len());
            expected
                .iter()
                .zip(data.iter())
                .for_each(|(i, j)| assert_eq!(*i, *j));
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_inefficient_sort_var() {
            let data = [234, 23, 4, 234, 23, 4, 234, 23, 2, 1362, 6, 1, 36, 7];
            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            let data = inefficient_sort_var(&world, &data);
            let expected = [1, 2, 4, 4, 6, 7, 23, 23, 23, 36, 234, 234, 234, 1362u64];
            assert_eq!(expected.len(), data.len());
            expected
                .iter()
                .zip(data.iter())
                .for_each(|(i, j)| assert_eq!(*i, *j));
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_inefficient_rank() {
            let data = [234, 23, 4, 235, 24];
            let mut ranking = vec![0u64; data.len()];

            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            inefficient_rank(&world, &data, &mut ranking);
            let expected = [3, 1, 0, 4, 2];
            assert_eq!(expected.len(), ranking.len());
            expected
                .iter()
                .zip(ranking.iter())
                .for_each(|(i, j)| assert_eq!(*i, *j));
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_inefficient_rank_var_with_ties() {
            let data = [1, 1, 4, 5, 24];
            let mut ranking = vec![0u64; data.len()];

            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            inefficient_rank_var(&world, &data, &mut ranking);
            assert!(ranking[0] == 0 || ranking[0] == 1);
            assert!(ranking[1] == 0 || ranking[1] == 1);
            assert!(ranking[0] != ranking[1]);
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_inefficient_rank_var() {
            let data = [234, 23, 4, 235, 24];
            let mut ranking = vec![0u64; data.len()];

            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            inefficient_rank_var(&world, &data, &mut ranking);
            let expected = [3, 1, 0, 4, 2];
            assert_eq!(expected.len(), ranking.len());
            expected
                .iter()
                .zip(ranking.iter())
                .for_each(|(i, j)| assert_eq!(*i, *j));
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_inefficient_rank_with_ties() {
            let data = [1, 1, 4, 5, 24];
            let mut ranking = vec![0u64; data.len()];

            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            inefficient_rank(&world, &data, &mut ranking);
            assert!(ranking[0] == 0 || ranking[0] == 1);
            assert!(ranking[1] == 0 || ranking[1] == 1);
            assert!(ranking[0] != ranking[1]);
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_matrix_rank() {
            let data = [1, 2, 3, 4, 5, 6, 7, 8, 9];
            let mut ranking = vec![0u64; data.len()];

            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            matrix_rank(&world, &data, &mut ranking);

            let expected = [0, 1, 2, 3, 4, 5, 6, 7, 8];
            assert_eq!(expected.len(), ranking.len());
            expected
                .iter()
                .zip(ranking.iter())
                .for_each(|(i, j)| assert_eq!(*i, *j));
        }
    }
}
