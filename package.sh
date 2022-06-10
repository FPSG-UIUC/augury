#!/bin/bash
dname="dmp_existence_experiment_package_long"
mkdir -p $dname/bin
ARCH=x86 make
ARCH=x86 _OBJ=imp make
cp bin/imp $dname/bin/i-test
cp bin/augury $dname/bin/a-test
cp runme.sh $dname/
cp package_sweep.sh $dname/
tar -czf $dname.tar.gz $dname/
