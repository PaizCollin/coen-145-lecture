#!/bin/bash
declare -i nrows=10000
declare -i ncols=10000
declare -i ncols2=10000
declare -a factors=(0.05 0.1 0.15 0.2)
declare -a threads=(1 2 4 8 12 14 16 20 24 28)

for i in ${factors[@]}
do
    echo "Runtime" >> $i.csv
    echo "Beginning test ($nrows $ncols $ncols2 $i)"
    for j in ${threads[@]}
    do
        echo "Testing $j thread"
        ./sparsematmult $nrows $ncols $ncols2 $i -t $j
    done
done
