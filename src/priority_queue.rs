use crate::parallel_select_k;
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::borrow::Borrow;
use std::cmp::{min, Reverse};
use std::collections::BinaryHeap;

pub struct ParallelPriorityQueue<'a, const OVERSAMPLING: usize> {
    communicator: &'a dyn Communicator,
    bin_heap: BinaryHeap<Reverse<u64>>,
    selection_pool: Vec<u64>,
    rng: ThreadRng,
}

// TODO figure out in benchmarks what a good default oversampling factor is
impl<'a> ParallelPriorityQueue<'a, 4> {
    /// Create a new ParallelPriorityQueue with default oversampling factor of 4
    pub fn new_default(comm: &'a dyn Communicator) -> ParallelPriorityQueue<'a, 4> {
        ParallelPriorityQueue::new(comm)
    }
}

impl<'a, const OVERSAMPLING: usize> ParallelPriorityQueue<'a, OVERSAMPLING> {
    pub fn new(comm: &'a dyn Communicator) -> ParallelPriorityQueue<'a, OVERSAMPLING> {
        let rt_world_size = (comm.size() as f64).sqrt() as usize;

        ParallelPriorityQueue {
            communicator: comm,
            bin_heap: BinaryHeap::new(),
            selection_pool: Vec::with_capacity(OVERSAMPLING * rt_world_size * 2),
            rng: thread_rng(),
        }
    }

    /// insert an element into the selection pool. If the selection pool would exceed its capacity,
    /// the selection pool will be flushed into the local queue, and the new element will be inserted
    /// into the queue as well.
    fn local_insert(&mut self, e: u64) {
        let rt_world_size = (self.communicator.size() as f64).sqrt() as usize;

        // if the element is larger than the minimum element in the local queue, insert it directly
        // because we don't need it before the current minimum
        if e > self.local_peek().unwrap_or(u64::MAX) {
            self.bin_heap.push(Reverse(e));
            return;
        }

        // otherwise insert it into the selection pool, unless the selection pool is full.
        if self.selection_pool.len() < OVERSAMPLING * rt_world_size * 2 {
            self.selection_pool.push(e);
        } else {
            // if the selection pool is full, flush it into the local queue and insert the new
            // element into the local queue as well
            self.selection_pool.iter().for_each(|&e| {
                self.bin_heap.push(Reverse(e));
            });
            self.selection_pool.clear();
            self.bin_heap.push(Reverse(e));
        }
    }

    /// Delete the minimum element from the local queue
    fn local_delete_min(&mut self) -> Option<u64> {
        self.bin_heap.pop().map(|Reverse(e)| e)
    }

    /// Peek at the minimum element in the local queue
    fn local_peek(&self) -> Option<u64> {
        self.bin_heap.peek().map(|Reverse(e)| *e)
    }

    /// Insert a constant amount of elements per processing unit. Insertion will redistribute the items among
    /// processors to ensure uniform data distribution
    ///
    /// # Parameters
    /// - `elements` a small number of elements to insert. They will be redistributed among processors
    pub fn insert(&mut self, elements: &[u64]) {
        let world_size = self.communicator.size() as usize;
        let elements = elements.to_vec();

        // generate random processor indices for each element
        let mut indices = vec![0; elements.len()];
        indices.fill_with(|| self.rng.gen_range(0..world_size));

        // sort elements by processor index
        let mut buf = elements
            .into_iter()
            .zip(indices.into_iter())
            .collect::<Vec<_>>();
        buf.sort_unstable_by_key(|a| a.1);

        // inform each processor of how many elements it will receive
        let mut send_counts = vec![0; world_size];
        let mut send_displs = vec![0; world_size];

        for (_, i) in buf.iter() {
            send_counts[*i] += 1;
        }

        for i in 0..world_size - 1 {
            send_displs[i + 1] = send_displs[i] + send_counts[i];
        }

        let mut recv_counts: Vec<i32> = vec![0; world_size];
        self.communicator
            .all_to_all_into(&send_counts, &mut recv_counts);
        let recv_displs: Vec<i32> = recv_counts
            .iter()
            .scan(0, |acc, i| {
                let tmp = *acc;
                *acc += *i;
                Some(tmp)
            })
            .collect();

        // allocate buffer for received elements
        let mut recv_buffer = vec![0u64; recv_counts.iter().sum::<i32>() as usize];
        let mut recv_partition =
            PartitionMut::new(&mut recv_buffer, recv_counts.borrow(), recv_displs.borrow());
        let send_data = buf.iter().map(|(e, _)| *e).collect::<Vec<_>>();

        // distribute elements among processors and insert them into the local queue
        let partition = Partition::new(&send_data, send_counts.borrow(), send_displs.borrow());
        self.communicator
            .all_to_all_varcount_into(&partition, &mut recv_partition);
        recv_buffer.iter().for_each(|e| self.local_insert(*e));
    }

    /// Delete the p smallest elements in the priority queue and distribute them among all p processing units.
    pub fn delete_min(&mut self) -> u64 {
        let world_size = self.communicator.size() as usize;
        let rt_world_size = (world_size as f64).sqrt() as usize;

        // ensure the selection pool has enough elements
        if self.selection_pool.len() < rt_world_size * OVERSAMPLING {
            for _ in 0..min(rt_world_size * OVERSAMPLING, self.bin_heap.len()) {
                let e = self.local_delete_min().unwrap();
                self.selection_pool.push(e);
            }
        }

        // perform select_k on the selection pool and distribute the result among all p processing units
        let mut smallest_elements =
            parallel_select_k(self.communicator, &self.selection_pool, world_size);

        // remove the selected elements from the pool buffer, but make sure to retain duplicates
        // of the largest selected element, if they have not been selected multiple times
        if !smallest_elements.is_empty() {
            smallest_elements.sort_unstable();
            let largest_element = *smallest_elements.last().unwrap();
            let mut largest_element_amount = smallest_elements.len()
                - smallest_elements.partition_point(|e| *e < largest_element);
            self.selection_pool.retain(|e| {
                if *e == largest_element {
                    if largest_element_amount > 0 {
                        largest_element_amount -= 1;
                        false
                    } else {
                        true
                    }
                } else {
                    !smallest_elements.contains(e)
                }
            });
        }

        // TODO maybe we should periodically flush the selection pool into the bin heap?
        //  theoretically that should be unnecessary because of the insertion strategy, which
        //  should ensure that the expensive edge case doesn't happen more often than without the
        //  persistent selection pool, but I don't know if there is another reason to do it

        // enumerate the smallest element using a prefix sum and distribute the elements among the processors accordingly
        let mut prefix_sum = 0usize;
        self.communicator.scan_into(
            &smallest_elements.len(),
            &mut prefix_sum,
            SystemOperation::sum(),
        );

        let mut send_counts = vec![0; world_size];
        let mut send_displs = vec![0; world_size];

        #[allow(clippy::needless_range_loop)] // easier to read
        for i in prefix_sum - smallest_elements.len()..prefix_sum {
            send_counts[i] = 1;
        }

        for i in 0..world_size - 1 {
            send_displs[i + 1] = send_displs[i] + send_counts[i];
        }

        let send_partition = Partition::new(
            &smallest_elements,
            send_counts.borrow(),
            send_displs.borrow(),
        );
        let mut recv_buffer: u64 = 0;
        let mut recv_partition = PartitionMut::new(&mut recv_buffer, [1].borrow(), [0].borrow());

        self.communicator
            .all_to_all_varcount_into(&send_partition, &mut recv_partition);

        // check that no processor has received a larger element than the smallest local element
        let mut repeat_operation_flag = 0u8;
        if recv_buffer > self.local_peek().unwrap_or(u64::MAX) {
            self.communicator.all_reduce_into(
                &1u8,
                &mut repeat_operation_flag,
                SystemOperation::max(),
            );

            // add more elements to the selection pool, so the smallest element is guaranteed to be
            // in it, and any more elements that might or might not be smaller as well
            for _ in 0..min(rt_world_size * OVERSAMPLING, self.bin_heap.len()) {
                let e = self.local_delete_min().unwrap();
                self.selection_pool.push(e);
            }
        } else {
            self.communicator.all_reduce_into(
                &0u8,
                &mut repeat_operation_flag,
                SystemOperation::max(),
            );
        }

        if repeat_operation_flag == 1 {
            // re-insert the received element into the queue and repeat the operation,
            // while all processors where the operation failed have increased their selection
            // pool size
            self.local_insert(recv_buffer);
            self.delete_min()
        } else {
            recv_buffer
        }
    }
}

#[cfg(test)]
mod tests {
    ///! These tests are sanity checks for the parallel priority queue implementation. They do not
    ///! check for the correctness of the parallel implementation, but rather for the correctness of
    ///! the local implementation.
    use super::*;
    use rusty_fork::rusty_fork_test;

    rusty_fork_test! {
        #[test]
        fn test_insert() {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            let mut pq = ParallelPriorityQueue::new_default(&world);

            let mut elements = vec![0u64; 10];
            elements.fill_with(|| rand::random());
            pq.insert(&elements);
            assert_eq!(
                pq.bin_heap.len(),
                elements.len(),
                "wrong amount inserted into the queue"
            );

            // check that all elements have been inserted correctly
            elements.sort();
            elements.iter().for_each(|e| {
                assert_eq!(
                    *e,
                    pq.bin_heap.pop().unwrap().0,
                    "unexpected element inserted into the queue"
                )
            });
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_delete_min() {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            let mut pq = ParallelPriorityQueue::new_default(&world);

            let mut elements = vec![0u64; 10];
            elements.fill_with(|| rand::random());

            pq.insert(&elements);

            let min = pq.delete_min();
            elements.sort();

            assert_eq!(
                min, elements[0],
                "deleteMin did not return the smallest element"
            );
            assert_eq!(
                pq.bin_heap.len() + pq.selection_pool.len(),
                elements.len() - 1,
                "deleteMin deleted too many elements"
            );
        }
    }

    rusty_fork_test! {
        #[test]
        fn test_delete_min_duplicates() {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            let mut pq = ParallelPriorityQueue::new_default(&world);
            let elements = vec![10u64; 10];
            pq.insert(&elements);
            pq.delete_min();

            assert_eq!(
                pq.bin_heap.len() + pq.selection_pool.len(),
                elements.len() - 1,
                "deleteMin deleted too many elements"
            );
        }
    }
}
