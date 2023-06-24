CC = g++

default: sparsematmult

sparsematmult: sparsematmult.cpp
	${CC} -O3 -Wall -Wextra -Wno-unused-parameter -fopenmp -o $@ sparsematmult.cpp

clean:
	-rm -f sparsematmult
