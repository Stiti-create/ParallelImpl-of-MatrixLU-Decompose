### LU Decomposition of Matrix using Row-Pivoting
Parallel implementations of LU decomposition that use Gaussian elimination to factor a dense N x N matrix into an upper-triangular one and a lower-triangular one. Includes analysis of comparisions of sequential, pthread and openmp implementations.

### Instructions to run the code

1. To clean log files and executables: `make clean`
2. To generate a random matrix of required size:
    * Change the value of N from `constants.h` accordingly
    * Run the command `make gen`
3. To run the different modes of LU decomposition:
    * Sequential Implementation: `make seq`
    * Parallel Implementation using pthreads: `make pth`
    * Parallel Implementation using openmp: `make omp`
4. You can view the time taken by different implementations in `log.txt`

