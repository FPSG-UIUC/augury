#!/bin/bash
echo "Running for sizes 2 4 8 16 32 64 128 256 512 1024"
echo "We read proc/cpuinfo (or sysctl -a on macOS) and uname, fyi"
echo "Each size takes approximately the same amount of time"
echo "----------------"
# > cat /proc/cpuinfo  | grep "model name" | head -n1 | tr ' ' '_' | tr -d '\t'
# model_name__:_AMD_EPYC_7501_32-Core_Processor
# > sysctl -a | grep "brand_string" | head -n1 | tr ' ' '_' | tr -d '\t'
# machdep.cpu.brand_string:_Intel(R)_Core(TM)_i5-8257U_CPU_@_1.40GHz
if [[ -e /proc/cpuinfo ]]; then
    echo reading /proc/cpuinfo
    NAME=$(cat /proc/cpuinfo  | grep "model name" | head -n1 | tr ' ' '_' | tr -d '\t')
else
    echo running sysctl -a
    NAME=$(sysctl -a  | grep "brand_string" | head -n1 | tr ' ' '_' | tr -d '\t')
fi

for x in 2 4 8 16 32 64 128 256 512 1024; do echo "Starting size $x"; ./test_sweep.sh 0 $NAME ${x}; done;

uname -a > sweep_out/uname
echo $NAME > sweep_out/cpuinfo
tarname="longsweep"$(echo $NAME | cut -d ":" -f 2 | cut -d "@" -f1)
tar -czf "$tarname.tar.gz" sweep_out/

