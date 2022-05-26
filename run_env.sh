#!/bin/zsh

eval "$(conda shell.bash hook)"
conda activate augury

BIN=./bin/augury
# BIN=./bin/imp
CORE=4 # Cores {4..7} are firestorm. Change to {0..3} for icestorm.
REPETITIONS=32

quit_fail() {
	echo "${RED}[RUN FAIL] $1${NOCOLOR}"
	exit 1
}

if [ ! -f $BIN ]; then
	quit_fail "$BIN not found"
fi

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"
