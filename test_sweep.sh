#!/bin/bash

max_test=32
cur_test=0

inc=$1
run_id=$2
t_len=$3

mkdir -p out

outdir="sweep_out/${t_len}_${run_id}"
mkdir -p $outdir

while [[ $cur_test -le $max_test ]]; do
	test_idx=$(($inc+$cur_test))

	echo "=$cur_test:$test_idx:$t_len="

	# see README.md for explanation of cl args
	./bin/a-test $t_len $test_idx 32 1 1

	mv -f out/a_data.out "${outdir}/a_data_${test_idx}.out"
	mv -f out/ab_data.out "${outdir}/ab_data_${test_idx}.out"

	((cur_test+=1))
done

cur_test=0
while [[ $cur_test -le $max_test ]]; do
	test_idx=$(($inc+$cur_test))

	echo "=$cur_test:$test_idx:$t_len="

	# see README.md for explanation of cl args	
	./bin/i-test $t_len $test_idx 32 1 1 

	mv -f out/i_data.out "${outdir}/i_data_${test_idx}.out"
	mv -f out/ib_data.out "${outdir}/ib_data_${test_idx}.out"

	((cur_test+=1))
done
