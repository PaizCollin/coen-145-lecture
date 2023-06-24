SRC := dijkstra.cpp
MPISRC := dijkstra_mpi.cpp
OMPSRC := dijkstra_omp.cpp

default: dijkstra_omp

dijkstra: $(SRC)
	g++ -O0 -Wall -Wextra -o $@ $<

dijkstra_mpi: $(MPISRC)
	mpic++ -O0 -Wall -Wextra -Wno-cast-function-type -o $@ $<

dijkstra_omp: $(OMPSRC)
	g++ -fopenmp -O0 -Wall -Wextra -o $@ $<

clean: 
	rm -f dijkstra dijkstra_mpi dijkstra_omp
