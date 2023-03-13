# Parallel Toolbox
An arbitrary assortment of distributed algorithms implemented on MPI.
Focus is on ranking, sorting and basic distributed datastructures.
Documentation is provided for each algorithm outlining its use case and trade-offs.

At the moment, tuning parameters are chosen arbitrarily and without much benchmarking, 
but I plan to remedy that in the future.
Further, those parameters will get exposed,
so applications demanding specific tuning of algorithms can easily achieve it.

## About Safety
This library does not use `unsafe` itself,
but (naturally) makes heavy use of [rsmpi](https://github.com/rsmpi/rsmpi)
which is a thin binding and therefore abundantly unsafe.
The binding lib makes no attempt at sanitizing input, and
it is therefore extremely easy to provoke a segfault by passing wrong buffer sizes to MPI
or by confusing data types.

**Do not use this library in security-critical applications.**

## Algorithms
#### Inefficient Ranking and Sorting
A few inefficient routines for ranking and sorting are provided,
which are reused in more efficient algorithms as the base case for recursions.
* `inefficient_sort` sends all data to one processor, which will sort and redistribute it. It is the most
  inefficient and slowest algorithm, but works in all cases
* `inefficient_rank` sends all data to one processor, which will calculate ranks and return them to the original 
  processors. Otherwise it is the same design as `inefficient_sort`
* `matrix_rank` requires a square number of processors, but is theoretically more efficient than alternatives
  

#### Sample Sort
As an efficient all-purpose sorting algorithm the toolbox provides `sample_sort`.
It expects roughly equally sized slices of data on all clients,
and will sort the data according to the MPI processor ranks.
It guarantees constant amount of recursions with high probability,
and data distribution will be no worse than $(1 + \epsilon)\frac{n}{p}$ 
where $n$ is the total data amount, $p$ is the number of processors and $\epsilon$ is a tuning parameter.