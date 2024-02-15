all:
	rm -f sequential omp_impl pth_impl
	g++ sequential.cpp -o sequential
	g++ omp_impl.cpp -o omp_impl -fopenmp
	g++ pthread_impl.cpp -o pth_impl -lpthread

seq:
	g++ sequential.cpp -o sequential 

pth:
	g++ pthread_impl.cpp -o pth_impl -lpthread

omp:
	g++ omp_impl.cpp -o omp_impl -fopenmp

debug: 
	g++ -g pthread_impl.cpp -o pth_impl -lpthread
	gdb pth_impl

clean:
	rm -f debug.txt debug_lu_verify.txt
	rm -f sequential pth_impl omp_impl



	