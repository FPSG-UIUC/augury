#!/bin/zsh

source run_env.sh

max_test=80
max_retries=10

stride=1
train_len=256
run_id="addr_sweep_${train_len}"

if [ ! -f addrs.dat ]; then
		echo "Generating addrs.dat"
		./make_addrs.py
fi

export  FORCEADDR=1
make -B

mkdir -p out

outdir="sweep_out/${run_id}_${stride}"
mkdir -p $outdir

out_file="${outdir}/${run_id}_sweep.out"

echo "Saving results to ${outdir} and $out_file"

while read addr; do
	echo "${GREEN}================ $addr ================${NOCOLOR}"

	# run experiment and get data. Repeat until AoP activates
	aop_active=0
	retries=1

	while [[ aop_active -eq 0 && $retries -lt $max_retries ]]; do
		[[ $retries -gt 1 ]] && echo \
			"${RED}${addr} Attempt #$retries/$max_retries${NOCOLOR}"

		successful=0
		alloc_retries=1
		while [[ $successful -eq 0 && $alloc_retries -lt $max_retries ]]; do
			successful=1
			$BIN $train_len 1 32 4 $stride $addr || successful=0

			((alloc_retries += 1))
		done

		((retries += 1))
		if [[ $successful -eq 0 ]]; then
			echo "${RED}${addr} Failed to allocate${NOCOLOR}"
			continue
		fi

		aop_active=1
		./get_avg.py "out/baseline.out" "out/aop.out" \
			"${run_id}" --append_save --omit --save_file $out_file || aop_active=0

		if [[ $retries -eq $max_retries && $aop_active -eq 0 ]] then
			./get_avg.py "out/baseline.out" "out/aop.out" \
				"${run_id}" --append_save --omit --save_file $out_file --save_anyway

			echo "Abandoning ${addr} after $retries attempts"
		fi
	done

	if [[ -f "out/aop.out" ]]; then
		mv -f out/aop.out "${outdir}/aop_${addr}.out"
		mv -f out/baseline.out "${outdir}/baseline_${addr}.out"
	fi
done < addrs.dat
