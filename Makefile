all:
	rm -f matrix_gen sequential
	rm -f matrix.txt debug.txt
	g++ matrix_gen.cpp -o matrix_gen
	./matrix_gen
	g++ sequential.cpp -o sequential
	./sequential

gen:
	rm -f matrix_gen matrix.txt
	g++ matrix_gen.cpp -o matrix_gen
	./matrix_gen

seq:
	g++ sequential.cpp -o sequential 
	./sequential

pth:
	g++ pthread_impl.cpp -o pth_impl -lpthread
	./pth_impl

omp:
	g++ omp_impl.cpp -o omp_impl -fopenmp
	./omp_impl

debug: 
	g++ -g pthread_impl.cpp -o pth_impl -lpthread
	gdb pth_impl

clean:
	rm -f debug.txt debug_lu_verify.txt 
	rm -f matrix_gen sequential pth_impl omp_impl



	