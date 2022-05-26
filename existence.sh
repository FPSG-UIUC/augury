#!/bin/zsh
# Evaluate mean access times and prefetch distance.

source run_env.sh

# run the test
for train_size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192;
do
	./test_sweep.sh "existence_sweep_$train_size" 1 $train_size
done

# plot results
python ./pool_avgs.py -c plot_cfgs/sweep.cfg sweep_out --list sweep_out/existence_sweep_* --save_file existencesweep.csv --debug
