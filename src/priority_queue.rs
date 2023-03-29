use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::SystemCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives};
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct ParallelPriorityQueue<'a> {
    communicator: &'a SystemCommunicator,
    bin_heap: BinaryHeap<Reverse<u64>>,
    rng: ThreadRng,
}

impl<'a> ParallelPriorityQueue<'a> {
    pub fn new(comm: &'a SystemCommunicator) -> ParallelPriorityQueue<'a> {
        ParallelPriorityQueue {
            communicator: comm,
            bin_heap: BinaryHeap::new(),
            rng: thread_rng(),
        }
    }

    /// insert an element into the local queue
    fn local_insert(&mut self, e: u64) {
        self.bin_heap.push(Reverse(e))
    }

    /// Insert a constant amount of elements per processing unit. Insertion will redistribute the items among
    /// processors to ensure uniform data distribution
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
        todo!("not yet implemented")
    }
}
