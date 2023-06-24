#!/bin/bash
declare -a graphs=(1000 5000 10000 20000)
declare -a threads=(1 2 4 8 16 28)

for i in ${graphs[@]}
do
    echo "Beginning tests on graph ($i)"
    for j in ${threads[@]}
    do
        echo "Testing $j thread(s)"
        ./dijkstra_omp ${i}.graph 100 ${i}_${j}_omp.out $j
    done
    echo 
done
