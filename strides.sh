#!/bin/zsh
# Evaluate mean access times and prefetch distance.

source run_env.sh

# run the test
for train_stride in 1 2 3 4 6 8 12 16 24 32 48 64 128 256 512;
do
	./test_sweep.sh "stride_sweep_128" $(($train_stride * 16)) 128
done

# plot results
python ./pool_avgs.py -c plot_cfgs/density.cfg sweep_out --list sweep_out/stride_sweep_128* --save_file stridesweep.csv --debug
