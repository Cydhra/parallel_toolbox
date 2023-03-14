use std::cmp::Reverse;
use std::collections::BinaryHeap;
use mpi::topology::SystemCommunicator;
use rand::{Rng, thread_rng};
use rand::rngs::ThreadRng;

pub struct ParallelPriorityQueue<'a> {
    communicator: &'a SystemCommunicator,
    bin_heap: BinaryHeap<Reverse<u64>>,
    rng: ThreadRng,
}

impl <'a> ParallelPriorityQueue<'a> {

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
        todo!("not yet implemented");
    }

    /// Delete the p smallest elements in the priority queue and distribute them among all p processing units.
    pub fn delete_min(&mut self) -> u64 {
        todo!("not yet implemented")
    }
}