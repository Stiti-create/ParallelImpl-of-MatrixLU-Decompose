run:
	g++ matrix_gen.cpp -o matrix_gen
	./matrix_gen
	g++ sequential.cpp -o sequential 
	./sequential

clean:
	rm -f matrix_gen sequential
	rm -f matrix.txt debug.txt
	