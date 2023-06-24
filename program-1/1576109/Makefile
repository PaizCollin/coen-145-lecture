CC = g++

default: matmult_omp

matmult_omp: $(SRC)
	${CC} -O3 -Wall -Wextra -fopenmp -o $@ matmult_omp.cpp

clean:
	-rm -f matmult_omp
