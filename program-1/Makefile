CC = g++

default: matmult matmult_omp

matmult: matmult.c
	${CC} -O3 -Wall -Wextra -o $@ matmult.c

matmult_omp: $(SRC)
	${CC} -O3 -Wall -Wextra -fopenmp -o $@ matmult_omp.cpp

clean:
	-rm -f matmult
	-rm -f matmult_omp
