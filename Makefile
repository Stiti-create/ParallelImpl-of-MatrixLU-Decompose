all:
	rm -f matrix_gen sequential
	rm -f matrix.txt debug.txt
	g++ matrix_gen.cpp -o matrix_gen
	./matrix_gen
	g++ sequential.cpp -o sequential
	./sequential

gen:
	g++ matrix_gen.cpp -o matrix_gen
	./matrix_gen

seq:
	g++ sequential.cpp -o sequential 
	./sequential

clean:
	rm -f matrix_gen sequential
	rm -f matrix.txt debug.txt
	