# Experiment code for Augury: Using Data Memory-Dependent Prefetchers to Leak Data at Rest

This git repo contains the code for running the experiments and PoCs
provided in our paper, `Augury: Using Data Memory-Dependent
Prefetchers to Leak Data at Rest`. For more info or to read the paper,
please visit our [webpage](https://www.prefetchers.info/ "Augury
webpage and paper link").

## Main experiment code

### Building the code

Running `make` should be enough to build the code. We have only tested
the code and compiler specific features using `clang`. All of this
should work on `gcc` as well, but we haven't verified this.

There are additional build options available for configuring in the
`Makefile` depending on how you want to run the experiments. For
example, the variable `KEXT` determines if the experiment code will
use the kernel extension to enable and use the M1's performance
monitoring counters (`KEXT=1`) or the mach syscall
`mach_absolute_time` (`KEXT=0`) to record access times.

### Running the code

#### Quick and Dirty

Various runscripts are included which can be used to directly run the
experiments included in the paper.
These are:
 - `existence.sh`: Runs the AOP-DMP experiments used to generate Figures 4 and 7. To run IMP-DMP experiments, change the `BIN` in `run_env.sh`.
 - `strides.sh`: Runs the experiments used to generate Figure 8. It will not generate a graph for this figure, but will dump the necessary data as a csv.
 - `pagebound.sh`: Runs the experiments used to generate Figure 9.
 - `addrs.sh`: Runs the experiments used to characterize the behaviour around `0x280000000-0xffff3f2b0000`.

These files will all attempt to call the plotter to generate figures. This step is not necessary for regular usage and depends on the included Conda Environment being setup (`augury_env.yml`).

If you encounter any problems with these scripts, we instead recommend following the process below.

#### Details of our Experiments

The main experiment code referenced in our paper's sections `Existence
of the M1 DMP` and `Reverse engineering the M1 DMP` is contained in
the file `augury.c`. After running the build, the binary will be
located at `bin/augury`.

You can run the experiment using the following command in this directory:

```
./bin/augury <num_train_ptrs> <offs_past_train_buf> <repetitions>
<coreID> <ptrs_per_cacheline_density>
```

To get a better feel of what these command line args mean, consider
the following pseudocode simplified from that contained in `augury.c`:

```C
// ...snip setup...
pin_cpu(coreID); // Cores 0-3 are icestorm, 4-7 firestorm
// ...snip setup...

for (size_t rep = 0; rep < repetitions; ++rep) {

	/* train loop */
	for (size_t ii = 0; ii < num_train_ptrs; ++ii) {
		*aop[ii * ptrs_per_cacheline_density];
	}

	/* time the access to test pointer */
	size_t time1 = CURRENT_TIME;
	*aop[(num_train_ptrs + offs_past_train_buf - 1) * ptrs_per_cacheline_density];
	size_t time2 = CURRENT_TIME;

	access_times[rep] = time2 - time1;
}
```

In this code snippet:

* `<coreID>` determines whether the experiment runs on the Icestorm (0
  through 3 inclusive) or Firestorm (4 through 7 inclusive) cores.

* `<repetitions>` controls how many times the main experiment loop is
  run and how many times we time the test pointer access.

* `<num_train_ptrs>` controls how many pointers are stored and
  accessed in the array of pointers.

* `<offs_past_train_buf>` controls which pointer past the end of the
array of pointers is accessed for timing. If `<offs_past_train_buf>`
is greater than the prefetch depth of the DMP, then access times
should be slow, if it is less than the prefetch depth, then the
pointer should have been prefetched and accessing it should be
fast. Note that this means the array of pointers (`aop` in the
pseudocode) is allocated enough room to store at least
`<num_train_ptrs> + <offs_past_train_buf>` pointers. Note from the
pseudocode that `<offs_past_train_buf> =  0` means that we test the
access time to the last training pointer in the AoP and that
`<offs_past_train_buf> = 1` is the index of the *first* pointer
*after* the training pointers.

* `<ptrs_per_cacheline_density>` sets how many pointers are stored in
a L2 cacheline. Since the Apple M1 has 128 byte L2 cachelines and
pointers are 8 bytes wide, `<ptrs_per_cacheline_density>` should be
between 1 and 16 (`16 == 128/8`) inclusive.

A short-running example configuration we've been running is the
following. If your machine has a DMP, then this should result in a
fast train and test access time compared to the baseline (see next
subsection of this README).

```
./bin/augury 64 3 5 7 16
```

### Data analysis and plotting scripts

After running the augury binary, results of the experiment (program)
are logged in the `./out` directory. The `./out` directory should
contain two files: `./out/baseline.out` and `./out/aop.out`. Each of
these files is a two-column csv file with a header. The first column,
named `test`, specifies the test access time, i.e., `time2 - time1` in the
above pseudocode. The second column, named `train`, specifies the time
to iterate through all of the `<num_train_ptrs>` train pointers, i.e.,
the time to run the inner for-loop in the above pseudocode.

## PoCs

### SLH PoC

The SLH PoC is given in `slh-poc.c`, and the experiment binary after
building is `bin/slh-poc`. This binary requires no command line
arguments to run. Shortly after running the binary, you will be
prompted to pick between 3 pointers. The pointer you select will be
placed slightly past the bound of an array of pointers. All of the
pointers are then kicked out of cache.

After iterating through the array of pointers and activating the DMP,
the selected pointer will be prefetched by the DMP *but never accessed
by an instruction*. After iterating through the array of pointers, the
 program then checks the access times of each of the pointers. At the
end of the program execution, these access times are printed.

If the machine running the code has a DMP, then the access times of
the pointer you selected should be roughly around the time of an L2
cache access time.

### Out-of-bounds read PoC

We mention an out-of-bound read PoC in the paper, but we do not
provide an explicit binary for it here. The out-of-bounds read PoC is
identical to the SLH poc but without the
`-mspeculative-load-hardening` compiler flag.

### Testing validity of pointers (ASLR break) PoC

The ASLR PoC is given in `aslr-poc.c`, and the experiment binary after
building is `bin/aslr-poc`. Our PoC's code is adapted from [this](
https://github.com/Eugnis/spectre-attack "Spectre attack example
source link") Spectre attack example for x86-64.

This PoC binary also takes no command line args but will prompt you to
enter a pointer which the PoC will test for being a valid, accessible
virtual address. The binary will print some valid virtual addresses it
knows of for you to select, or you can pick a random one and hope that
it is not valid.

At the end of the program's execution, the program will print the
access times to a bunch of memory addresses. If the pointer you
selected was valid, then the access time of the test pointer specified
at the end of `main` should be fast (roughly around the time of an L2
cache access time). If you do not change the test pointer (the one
specified at the end of `main`), then this pointer should be
`array2[103 * 512]` in the printed access times.

# Plotting Results from Other Systems
After testing another system, plot the AOP and IMP data using:
```
./pool_avgs.py -c sku.cfg sku_sweep --list [aop_filenames] --save_file [output_file]
./pool_avgs.py -c sku_imp.cfg sku_sweep --list [imp_filenames] --save_file [output_file]
```

These scripts will only work if the filenames are unchanged.

## Packaging the experiment code

In our paper, we mention running existence experiments for both the
2-level, pointer-chasing DMP and the indirection-based DMP. The code
package that we used to test across a bunch of X86_64 and ARM
processors is all of the C code in this repo and the `package.sh`,
`runme.sh`, and `test_sweep.sh` shell scripts. These shell scripts
have been tested to work with both `zsh` and `bash`. 

To run the packaged up experiment code, run `bash package.sh` and the
packaged up experiment code will be in the directory specified in
`package.sh` (by default it is
`dmp_existence_experiment_package_long`). In this directory, there are
the `runme.sh` and `test_sweep.sh` scripts. Run `bash runme.sh`, and
after a while, the results for different AoP sizes will be dumped in
the directory. Result files named like `a...data` and `i...data`
contain the data from the runs for the AoP (2-level, pointer-chasing
DMP) and the IMP (multi-level, indirection-based DMP)
respectively. The files `ab...data` and `ib...data` contain the same
types of data but for the respective baseline runs (see the Existence
section of our paper for more info).

If the rest of the above instructions have already worked for you, you
probably **do not need to run this packaged experiment code**. It is
just here for convenience if you want to package the code up to send
to your friends with cool processors for running like we did.
