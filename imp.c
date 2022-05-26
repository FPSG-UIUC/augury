#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "util/util.h"


static int ipow(int base, int exp)
{
	if (exp == 0)
		return 1;
	return base;
}


#ifndef MACH
volatile int stop_timer;
volatile uint64_t ticks;


void *clock_thread(void *in)
{
	int core_ID = *(int *)in;
	pin_cpu(core_ID);

	while (!stop_timer) {
		ticks++;
	}

	return 0;
}
#endif


void print_size(unsigned bytes) {
    if (bytes <= KB(1)) {
        printf("%uB", bytes);
    } else if (bytes <= MB(1)) {
        printf("%uKB", TO_KB(bytes));
    } else {
        printf("%uMB", TO_MB(bytes));
    }
}


int main(int argc, char **argv)
{
	// Check arguments
	if (argc != 5 && argc != 6) {
		fprintf(stderr, "Wrong Input! ");
		fprintf(stderr, "Enter: %s "
				"<num_of_train_indices> <offs_past_train_buf> <repetitions> "
				"<core_ID>\n",
				argv[0]);
		exit(1);
	}

	// Parse core ID
	int core_ID;
	sscanf(argv[4], "%d", &core_ID);
	if (core_ID > 7 || core_ID < 0) {
        fprintf(stderr, "Wrong core! core_ID should be less than %d and more "
                "than 0!\n", 7);
		exit(1);
	}

#ifndef MACH
#warning "Using threaded timer"
	// Start clock thread
	pthread_t timer_thread;
	stop_timer = 0;
    // run timer thread on adjacent core
	int core_ID_timer = (core_ID <= 6) ? core_ID + 1 : core_ID - 1;
	pthread_create(&timer_thread, NULL, clock_thread, (void *) &core_ID_timer);
	while (ticks == 0) {}
	printf("Started Clock Thread\n");
#endif

	// Make the experiment run on the requested core
	pin_cpu(core_ID);

	// Parse offset of the test pointer
	int offs_past_train_buf;
	sscanf(argv[2], "%d", &offs_past_train_buf);

	// Parse the number of training pointers
	int num_of_train_indices;
	sscanf(argv[1], "%d", &num_of_train_indices);
	int num_of_test_indices = offs_past_train_buf < 2 + 80 ? 2 + 80 : num_of_test_indices;
	int total_num_of_indices =
		(num_of_train_indices < 8192 ? 8192 : num_of_train_indices) + num_of_test_indices;

	// Set up the DMP array (IMP)
	int size_of_memory_touched_in_training =
		(num_of_train_indices < 8192 ? 8192 : num_of_train_indices) * CACHE_LINE_SIZE;
	int size_of_memory_checked_in_testing = num_of_test_indices * CACHE_LINE_SIZE;
	int total_memory_allocated_for_imp = size_of_memory_touched_in_training +
		size_of_memory_checked_in_testing;

	// Allocate memory for the AoP
	/* const uintptr_t forced_address = 0x100f0c000; */
	/* volatile uint64_t **aop = mmap((void*)forced_address, */
	/* 		total_memory_allocated_for_imp, PROT_READ | PROT_WRITE, */
	/* 		MAP_ANON | MAP_FIXED | MAP_PRIVATE, -1, 0); */
	volatile uint64_t *imp = mmap(0, total_memory_allocated_for_imp,
            PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	assert(imp != MAP_FAILED);
	/* memset(aop, 0, total_memory_allocated_for_imp); */
    /* aop+=8; */

    // // Allocate memory for a 2 MB-aligned AoP (comment out above 2 lines if
    // // so)
    // volatile uint64_t **aop = mmap(0, total_memory_allocated_for_imp +
    // MB(2), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	// memset(aop, 0, total_memory_allocated_for_imp + MB(2));
	// uint64_t two_mb_mask = 0x200000 - 1;
	// for (uint64_t i = 0; i < MB(4); i++) {
	// 	if ((((uint64_t)aop + i) & two_mb_mask) == 0) {
	// 		aop = (volatile uint64_t **)((uint64_t)aop + i);
	// 		printf("[+] Found 2MB boundary at %lx\n", ((uint64_t)aop + i));
	// 		break;
	// 	}
	// }

	printf("[+] Num train indices: %d\n", num_of_train_indices);
	printf("[+] Total num indices: %d\n", total_num_of_indices);
	printf("[+] IMP is at %p-%p (", imp, imp + total_memory_allocated_for_imp);
	print_size(total_memory_allocated_for_imp);
	printf(")\n");

	// uint64_t paddr_start, paddr_end;
	// lkmc_pagemap_virt_to_phys_user(&paddr_start, (uintptr_t)aop);
	// lkmc_pagemap_virt_to_phys_user(&paddr_end, (uintptr_t)(uint64_t
	// **)((uint64_t)aop + total_memory_allocated_for_imp));
	// printf("[+] Physical address range of aop: 0x%lx-0x%lx\n", paddr_start,
	// paddr_end);

	// Set up the data array that the AoP holds ptrs to
	// For the data array we allocate a fixed amount of memory that
	// amounts to `PRNG_m` cache lines. Note that, depending on the
	// aop size, we may not need those many cache lines.
	int size_of_data_array = CACHE_LINE_SIZE * PRNG_m;
	/* int shared_data = open("data_file", O_RDWR); */
	volatile uint64_t *data_buffer = mmap(0, size_of_data_array,
			PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	assert((uintptr_t)data_buffer != 0x280000000);
	assert(data_buffer != MAP_FAILED);

	printf("[+] Allocated data_buffer\n");

	// Fill the data buffer with random data
	srand(time(NULL));
	for (uint64_t i = 0; i < size_of_data_array / sizeof(uint64_t); i++) {
		data_buffer[i] = rand() & (MSB_MASK - 1);
	}

	printf("[+] Data buf size is at %p-%p (", data_buffer,
			(uint64_t *)((uint64_t)data_buffer + size_of_data_array));
	print_size(size_of_data_array);
	printf(")\n");
	printf("[+] Data buf fits %d cache lines and %ld uint64ts\n", PRNG_m,
			size_of_data_array / sizeof(*data_buffer));

	// Make pointers in the AoP point to pseudo-random locations in the data
	// array
	uint64_t rand_idx = 1;
	uint64_t corresponding_idx_in_data_buffer;
	uint32_t u64s_per_cacheline = CACHE_LINE_SIZE / sizeof(uint64_t); // == 16
	for (int i = 0; i < total_num_of_indices; i+=1) {
		// Make i-th cache line in the aop point to somewhere in the data
		// buffer

		/* dprint("[dbg-alloc] %d\n", i); */

		corresponding_idx_in_data_buffer = rand_idx * u64s_per_cacheline;
		imp[i] = corresponding_idx_in_data_buffer;

		/* dprint("[dbg-alloc] (%d/%d) aop[%u]\n", i, total_num_of_indices, */
		/*         i); */
		/* dprint("[dbg-alloc]\t(%p)\n", */
		/* 	   &aop[i]); */
		/* dprint("[dbg-alloc]\t&data_buffer[%llu]\n", */
		/*         corresponding_idx_in_data_buffer); */
		/* dprint("[dbg-alloc]\t(%p)\n", */
		/*         &data_buffer[corresponding_idx_in_data_buffer]); */

		/* if (i < num_of_train_indices) { */
			dprint("[dbg-alloc] (%d) imp[%u](%p) <- data_buffer[%llu](%p)\n",
					i, i,
					&imp[i],
					corresponding_idx_in_data_buffer,
					&data_buffer[corresponding_idx_in_data_buffer]);
		/* } */

		// Print the last aop pointer accessed during training
		if (i == num_of_train_indices - 1) {
			printf("[+] Last access of training will be imp[%u](%p)=%llu\n",
					i, &imp[i], imp[i]);
			/* dprint("[dbg-alloc] Reset stride from %llu ", stride); */
			/* stride = 1; */
			/* dprint("to %llu\n", stride); */
		}

		// Update index into the data buffer for next aop pointer
		rand_idx = prng(rand_idx);
	}

	// For preventing unwanted compiler optimizations and adding
	// data dependencies between instructions.
	uint64_t __trash = 0;

    // Allocate a large array (8 times the cache) that we can access to flush
    // the entire cache.
	//
    // High performance cores: 12MB of shared L2 cache, 192KB L1 instruction
    // cache, and 128KB L1 data cache.
    // High efficiency cores: 4MB of shared L2 cache, 128KB L1 instruction
    // cache, and 64KB L1 data cache.
	int size_of_thrash_array = (MB(12) + KB(128)) * 8;
	volatile uint64_t *thrash_arr = mmap(0, size_of_thrash_array,
            PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);

	// Fill the thrash array with random data
	for (uint64_t i = 0; i < size_of_thrash_array / sizeof(uint64_t); i++) {
		thrash_arr[i] = rand() & (MSB_MASK - 1);
	}

	// Precompute the test pointer
	rand_idx = 1;
    for (uint32_t j = 0; j < num_of_train_indices + offs_past_train_buf - 1;
            j++) {
		rand_idx = prng(rand_idx);
    }
	uint64_t test_idx_in_data_buffer = rand_idx * u64s_per_cacheline;

    printf("[+] Address to test is data_buffer[%llu] (%p) = %llu\n",
            test_idx_in_data_buffer,
            &data_buffer[test_idx_in_data_buffer],
            data_buffer[test_idx_in_data_buffer]);
    printf("[+] The above should match the content of imp[%u]\n",
					 (num_of_train_indices + offs_past_train_buf - 1));
    printf("\t(%p)",
           &imp[(num_of_train_indices + offs_past_train_buf - 1)]);
    printf("\t= %llu\n",
           imp[(num_of_train_indices + offs_past_train_buf - 1)]);

		// uint64_t paddr;
		// lkmc_pagemap_virt_to_phys_user(&paddr, (uintptr_t)aop);
		// lkmc_pagemap_virt_to_phys_user(&paddr_end,
		// (uintptr_t)(&aop[(num_of_train_indices + offs_past_train_buf - 1)]));
		// printf("[+] Physical address of test: 0x%lx\n", paddr);

    // Double-check that the data idx we will test corresponds to the test
    // pointer of the aop we want
		assert(imp[(num_of_train_indices + offs_past_train_buf - 1)] ==
				test_idx_in_data_buffer);

	// Parse the number of repetitions
	int repetitions = 0;
	sscanf(argv[3], "%d", &repetitions);

	// Allocate array for results
	uint64_t *times_to_load_test_ptr_baseline = malloc(repetitions * sizeof(uint64_t));
	uint64_t *times_to_load_test_idx_imp = malloc(repetitions * sizeof(uint64_t));
	uint64_t *times_to_load_train_ptr_baseline = malloc(repetitions * sizeof(uint64_t));
	uint64_t *times_to_load_train_idx_imp = malloc(repetitions * sizeof(uint64_t));

	uint64_t *curr_test_base = times_to_load_test_ptr_baseline;
	uint64_t *curr_test_aop = times_to_load_test_idx_imp;
	uint64_t *curr_train_base = times_to_load_train_ptr_baseline;
	uint64_t *curr_train_aop = times_to_load_train_idx_imp;

	// Vars for use in time loop
	uint64_t T1 = 0, T2 = 0;
	uint64_t train_time;

	printf("[+] Starting experiment\n");

	uint8_t imp_mode = 0;

	// Collect data
	for (uint32_t i = 0; i < repetitions * 2; i++) {

		// Avoid speculation
		MEM_BARRIER;

		// Thrash the cache
        for (uint32_t j = 0; j < size_of_thrash_array / sizeof(uint64_t) - 2;
                j++) {
			__trash += (thrash_arr[j] ^ __trash) & 0b1111;
			__trash += (thrash_arr[j + 1] ^ __trash) & 0b1111;
			__trash += (thrash_arr[j + 2] ^ __trash) & 0b1111;
		}

		// // Time entire aop training
#ifdef MACH
#warning "Using system timer"
		T1 = get_time_nano(__trash & MSB_MASK);
#else
		T1 = *(&ticks + (__trash & MSB_MASK));
#endif
		__trash = (__trash + T1) & MSB_MASK;

		// Avoid speculation
		MEM_BARRIER;

		// Training loop
		// Alternate accesses through the AOP or through the data array
		imp_mode = imp_mode ^ 0x1;
		/* printf("%u-->", imp_mode); */
		uint32_t imp_idx = 0;
		rand_idx = 1;
		for (uint32_t j = 0; j < num_of_train_indices /* + 1 */; j++) {
			__trash = (rand_idx + imp_idx + j) | (__trash & MSB_MASK);

			// Direct data access
			// When imp_mode == 1 we always load data_buffer[16]
			// Otherwise, when imp_mode == 0, we follow the data accesses
			// in the order we filled the AOP
			corresponding_idx_in_data_buffer = rand_idx * (1 - imp_mode) *
				u64s_per_cacheline;

			dprint("[dbg] (%u,%u) Accessed imp[%llu](%p)-->%llu / "
					" data_buffer[%llu]: %p==%p?\n", i, j,
					(imp_idx) | (__trash & MSB_MASK),
					&imp[(imp_idx) | (__trash & MSB_MASK)],
					imp[(imp_idx) | (__trash & MSB_MASK)],
					corresponding_idx_in_data_buffer | (__trash & MSB_MASK),
					&data_buffer[corresponding_idx_in_data_buffer | (__trash & MSB_MASK)],
					data_buffer + (corresponding_idx_in_data_buffer | (__trash & MSB_MASK)));

			__trash = data_buffer[corresponding_idx_in_data_buffer | (__trash & MSB_MASK)];

			// Data access through IMP.
			// The modulo operation prevents speculatively accessing data past the end of the
			// array during training.
			__trash = data_buffer[imp[(imp_idx % num_of_train_indices) | (__trash &
					MSB_MASK)]];

			// Compute indices for the next loop iteration
			imp_idx += imp_mode | (__trash & MSB_MASK);
			rand_idx = prng(rand_idx | (__trash & MSB_MASK));
		}

		// Avoid speculation
		MEM_BARRIER;

		// Time entire aop training
#ifdef MACH
		T2 = get_time_nano(__trash & MSB_MASK);
#else
		T2 = *(&ticks + (__trash & MSB_MASK));
#endif
		train_time = T2 - T1;

		// Wait a bit for any training loads to fully complete
        /* int retval = nanosleep((const struct timespec[]){{0, 10000000L}}, */
        /*         NULL); */

		// Time loading a single test ptr
#ifdef MACH
		T1 = get_time_nano((__trash + test_idx_in_data_buffer) & MSB_MASK);
#else
		T1 = *(&ticks + ((__trash + test_idx_in_data_buffer) & MSB_MASK)
				+ (train_time & MSB_MASK));
#endif

		__trash = data_buffer[test_idx_in_data_buffer | (T1 & MSB_MASK)];
				// __trash = *aop[(num_of_train_indices + offs_past_train_buf - 1) | (T1 &
				// MSB_MASK)];

#ifdef MACH
		T2 = get_time_nano(__trash & MSB_MASK);
#else
		T2 = *(&ticks + (__trash & MSB_MASK));
#endif

		assert(((__trash + test_idx_in_data_buffer) & MSB_MASK) == 0);
		assert((T1 & MSB_MASK) == 0);
		assert((__trash & MSB_MASK) == 0);

		// Avoid speculation
		MEM_BARRIER;

		/* printf("[+] The extra access was at aop[%u] (%p) = %p -> %" PRIx64 */
		/*         "\n", (imp_idx), */
		/*         &aop[(imp_idx)], */
		/*         aop[(imp_idx)], */
		/*         *aop[(imp_idx)]); */

		// Store the time that it took to load the test ptr
		if (imp_mode == 0) {
			/* times_to_load_test_ptr_baseline[i / 2] = T2 - T1; */
			*curr_test_base = T2 - T1;
			/* times_to_load_train_ptr_baseline[i / 2] = train_time; */
			*curr_train_base = train_time;
			/* printf("%u-->%u (%llu : %llu) (%llu : %llu)\n", imp_mode, i/2, */
			/* 		*curr_test_base, T2 - T1, */
			/* 		*curr_train_base, train_time); */
			curr_test_base++;
			curr_train_base++;
		} else {
			/* times_to_load_test_idx_imp[i / 2] = T2 - T1; */
			*curr_test_aop = T2 - T1;
			/* times_to_load_train_idx_imp[i / 2] = train_time; */
			*curr_train_aop = train_time;
			/* printf("%u-->%u (%llu : %llu) (%llu : %llu)\n", imp_mode, i/2, */
			/* 		*curr_test_aop, T2 - T1, */
			/* 		*curr_train_aop, train_time); */
			curr_test_aop++;
			curr_train_aop++;
		}

		// Avoid speculation
		MEM_BARRIER;
	}

	printf("[+] Done! Storing results\n");

	// Open output file
	FILE *output_file_baseline = fopen("./out/baseline.out", "w");
	FILE *output_file_imp = fopen("./out/aop.out", "w");
	if (output_file_baseline == NULL || output_file_imp == NULL) {
		perror("output files");
	}

	// Store measurements
	fprintf(output_file_baseline, "%p,%p\n", imp, data_buffer);
	fprintf(output_file_imp, "%p,%p\n", imp, data_buffer);
	fprintf(output_file_baseline, "test,train\n");
	fprintf(output_file_imp, "test,train\n");
	for (uint32_t i = 0; i < repetitions; i++) {
		fprintf(output_file_baseline, "%llu,%llu\n",
                times_to_load_test_ptr_baseline[i],
                times_to_load_train_ptr_baseline[i]);
		fprintf(output_file_imp, "%llu,%llu\n",
				times_to_load_test_idx_imp[i],
				times_to_load_train_idx_imp[i]);

	}

	// Clean up
	fclose(output_file_baseline);
	fclose(output_file_imp);

#ifndef MACH
	// Stop timer thread
	stop_timer = 1;
	pthread_join(timer_thread, NULL);
#endif
}
