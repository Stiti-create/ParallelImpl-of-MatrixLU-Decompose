### LU Decomposition of Matrix using Row-Pivoting
* This repository contains Sequential and Parallel implementations of LU decomposition using row-pivoting that use Gaussian elimination to factor a dense N x N matrix into an upper-triangular one and a lower-triangular one, in C++.
* Includes analysis of comparisions of sequential, pthread and openmp implementations.

### Instructions to run the code

1. To clean log files and executables: `make clean`
2. To change problem size and number of threads:
    * Change the value of N and PTHREAD_COUNT from `constants.h` accordingly
3. To compile the different modes of LU decomposition:
    * Sequential Implementation: `make seq`
    * Parallel Implementation using pthreads: `make pth`
    * Parallel Implementation using openmp: `make omp`
    * All implementations compile: `make all`
4. Run the different codes:
    * ./sequential
    * ./pth_impl
    * ./omp_impl
5. You can view the time taken by different implementations in `log.txt`

