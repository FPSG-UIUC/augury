#!/bin/zsh
# Evaluate mean access times and prefetch distance.

source run_env.sh

# run the test
for train_size in {4072..4120};
do
	./test_sweep.sh "page_sweep_$train_size" 1 $train_size
done

# plot results
python ./pool_avgs.py -c plot_cfgs/pagebound.cfg sweep_out --list sweep_out/page_sweep_* --save_file pagesweep.csv --debug
