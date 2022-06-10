#!/bin/zsh

source run_env.sh

arg_fail() {
	echo 'First argument should be run id'
	echo 'Second argument should be stride'
	echo 'Third argument should be number of pointers to train on'
	exit 1
}

max_test=80
max_retries=4
cur_test=0

# inc=$1; shift
inc=0
run_id=$1; shift
stride=$1; shift
train_len=$1; shift
[[ -z $inc ]] && arg_fail
[[ -z $run_id ]] && arg_fail
[[ -z $stride ]] && arg_fail
[[ -z $train_len ]] && arg_fail

mkdir -p out  # used by binary

outdir="sweep_out/${run_id}_${stride}"
mkdir -p $outdir  # used for final result

out_file="${outdir}/${run_id}_sweep.out"

echo "Saving results to ${outdir} and $out_file"

while [[ $cur_test -le $max_test ]]; do
	test_idx=$(($inc+$cur_test))
	# ((test_idx*=$stride))

	echo "${GREEN}================$cur_test : $test_idx : $stride : $train_len================${NOCOLOR}"

	# run experiment and get data.
	# Repeat until AoP activates.
	aop_active=0
	retries=1
	cur_train_len=$train_len
	while [[ aop_active -eq 0 && $retries -lt $max_retries ]]; do
		[[ $retries -gt 1 ]] && echo \
			"${RED}${stride}_${test_idx} Attempt #$retries/$max_retries${NOCOLOR}"

		successful=0
		echo "Running " $BIN $cur_train_len $test_idx $REPETITIONS $CORE $stride
		while [[ $successful -eq 0 ]]; do
			successful=1
			# can use sudo because it apparently gives better core pinning; but it's
			# unecessary
			$BIN $cur_train_len $test_idx $REPETITIONS $CORE $stride || \
				successful=0
			echo ""
		done

		aop_active=1
		./get_avg.py "out/baseline.out" "out/aop.out" \
			"${run_id}_${cur_train_len}_${stride}_${test_idx}" \
			--append_save --omit --save_file $out_file || aop_active=0

		((retries += 1))
		if [[ $retries -eq $max_retries && $aop_active -eq 0 ]] then
			./get_avg.py "out/baseline.out" "out/aop.out" \
				"${run_id}_${cur_train_len}_${stride}_${test_idx}" \
				--append_save --omit --save_file $out_file --save_anyway
			echo "Abandoning ${stride} after $retries attempts"
		fi

		# will never fall in this case if max_retries < 10
		if [[ $(($retries % 10)) -eq 0  && $aop_active -eq 0 ]] then
			((cur_train_len *= 2))
			echo "${RED}Increasing number of train pointers to ${cur_train_len}${NOCOLOR}"
		fi
	done

	mv -f out/aop.out "${outdir}/aop_${test_idx}.out"
	mv -f out/baseline.out "${outdir}/baseline_${test_idx}.out"

	((cur_test+=1))
done
