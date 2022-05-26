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


/* #define KB(x) (x * 1024) */
/* #define MB(x) (x * 1024 * 1024) */
/* #define TO_KB(x) (x/1024) */
/* #define TO_MB(x) (x/1024/1024) */


void print_size(unsigned bytes) {
	if (bytes <= KB(1)) {
		printf("%uB", bytes);
	} else if (bytes <= MB(1)) {
		printf("%uKB", TO_KB(bytes));
	} else {
		printf("%uMB", TO_MB(bytes));
	}
}


#if defined (MACH) && defined (KEXT)
#warning "Both MACH and KEXT defined!"
#endif


static void
sighandler_local()
{
	/* printf("SIGILL\n"); */
	/* printf ("[+] Current core: %d\n", get_current_core ()); */

	assert(0 && "Process changed cores; PMC no longer available."
            "This is more likely to happen with large train sizes."
            "If this happens too often, consider using the MACH timer instead.");
	/* printf("Setting up timer again... "); */
	/* init_timer(); */
	/* printf("done setting up timer.\n"); */

	/* printf("Jumping... "); */
	/* longjmp(retry, 1); */
	/* printf("done jumping.\n"); */
}


int main(int argc, char **argv)
{
	// Check arguments
#ifdef FORCE_ADDR
    if (!(argc == 6 || argc == 7))  // optional: forced data buffer address
#else
    if (argc != 6)
#endif
    {
        fprintf(stderr, "Wrong Input! \n");
        fprintf(stderr, "Enter: %s "
                "<num_of_train_pointers> <offs_past_train_buf> <repetitions> "
                "<core_ID> <AOP cl density> "
                "[data base address (if built with FORCE_ADDR)]\n",
                argv[0]);
        exit(1);
    }

#ifdef FORCE_ADDR
    // Parse address to set data base addresses to if one is specified.
    uintptr_t data_base_addr = (uintptr_t)NULL;
    if (argc == 7) {
        sscanf(argv[6], "%lx", &data_base_addr);
    }
    printf("[+] data_buffer base address: 0x%lx\n", data_base_addr);
#endif

    // Parse core ID
    int core_ID;
    sscanf(argv[4], "%d", &core_ID);
    if (core_ID > 7 || core_ID < 0) {
        fprintf(stderr, "Wrong core! core_ID should be less than %d and more "
                "than 0!\n", 7);
        exit(1);
    }

#if !defined (MACH) && !defined (KEXT)
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
    int num_of_train_pointers;
    sscanf(argv[1], "%d", &num_of_train_pointers);
    uint32_t u64_ptrs_per_cacheline;
    sscanf(argv[5], "%u", &u64_ptrs_per_cacheline);
    /* int num_of_test_pointers = 2 + offs_past_train_buf; */
    int num_of_test_pointers = offs_past_train_buf < 2 + 80 ? 2 + 80 :
        offs_past_train_buf;
    int total_num_of_pointers =
        (num_of_train_pointers < 8192 ? 8192 : num_of_train_pointers) +
        num_of_test_pointers;

    // Set up the DMP array (AoP)
    int size_of_memory_touched_in_training =
        (num_of_train_pointers < 8192 ? 8192 : num_of_train_pointers) *
        u64_ptrs_per_cacheline * CACHE_LINE_SIZE; // one pointer per cache line

    int size_of_memory_checked_in_testing = num_of_test_pointers *
        u64_ptrs_per_cacheline * CACHE_LINE_SIZE; // one pointer per cache line

    int total_memory_allocated_for_aop = size_of_memory_touched_in_training +
        size_of_memory_checked_in_testing;

    // Allocate memory for the AoP
    /* const uintptr_t forced_address = 0x100f0c000; */
    /* volatile uint64_t **aop = mmap((void*)forced_address, */
    /* 		total_memory_allocated_for_aop, PROT_READ | PROT_WRITE, */
    /* 		MAP_ANON | MAP_FIXED | MAP_PRIVATE, -1, 0); */
#ifdef N2MB
    volatile uint64_t **aop = mmap(0, total_memory_allocated_for_aop,
            PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(aop != MAP_FAILED);
    memset(aop, 0, total_memory_allocated_for_aop);

#else
    // Allocate memory for a 2 MB-aligned AoP
    volatile uint64_t **aop = mmap(0, total_memory_allocated_for_aop +
            MB(2), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    memset(aop, 0, total_memory_allocated_for_aop + MB(2));
    uint64_t two_mb_mask = 0x200000 - 1;
    for (uint64_t i = 0; i < MB(2); i++) {
        if ((((uint64_t)aop + i) & two_mb_mask) == 0) {
            aop = (volatile uint64_t **)((uint64_t)aop + i);
            printf("[+] Aligned to 2MB boundary at %p\n", aop);
            break;
        }
    }
#endif
    assert(aop != MAP_FAILED);

    printf("[+] Num train ptrs: %d\n", num_of_train_pointers);
    printf("[+] Stride len: %d\n", u64_ptrs_per_cacheline);
    printf("[+] Allocated ptrs: %d\n", total_num_of_pointers);
    printf("[+] AOP is at %p-%p (", aop,
            (uint64_t **)((uint64_t)aop + total_memory_allocated_for_aop));
    print_size(total_memory_allocated_for_aop);
    printf(")\n");

    // uint64_t paddr_start, paddr_end;
    // lkmc_pagemap_virt_to_phys_user(&paddr_start, (uintptr_t)aop);
    // lkmc_pagemap_virt_to_phys_user(&paddr_end, (uintptr_t)(uint64_t
    // **)((uint64_t)aop + total_memory_allocated_for_aop));
    // printf("[+] Physical address range of aop: 0x%lx-0x%lx\n", paddr_start,
    // paddr_end);

    // Set up the data array that the AoP holds ptrs to
    // For the data array we allocate a fixed amount of memory that
    // amounts to `PRNG_m` cache lines. Note that, depending on the
    // aop size, we may not need those many cache lines.
    int size_of_data_array = CACHE_LINE_SIZE * PRNG_m;

    /* int shared_data = open("data_file", O_RDWR); */
#ifdef FORCE_ADDR
#warning "Forcing data_buffer address"
    /* const uintptr_t forced_data_address = 0x123800000; */
    /* const uintptr_t forced_data_address = 0x280000000; */
    volatile uint64_t *data_buffer = mmap((void*)data_base_addr,
            size_of_data_array, PROT_READ | PROT_WRITE,
            MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);

    /* assert((uintptr_t)data_buffer > (uintptr_t)(data_base_addr - (2 << 24))); */
    /* assert((uintptr_t)data_buffer < (uintptr_t)(data_base_addr + (2 << 24))); */
    /* printf("[+] Acceptable data range: %lx < %p < %lx\n", */
    /*         data_base_addr - (2 << 24), data_buffer, */
    /*         data_base_addr + (2 << 24)); */

#else

    volatile uint64_t *data_buffer = mmap(0, size_of_data_array,
            PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    assert((uintptr_t)data_buffer != 0x280000000);
#endif

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
    /* uint32_t u64_ptrs_per_cacheline = CACHE_LINE_SIZE / sizeof(uint64_t *); */
    for (int i = 0; i < total_num_of_pointers; i+=1) {
        // Make i-th cache line in the aop point to somewhere in the data
        // buffer

        /* dprint("[dbg-alloc] %d\n", i); */

        corresponding_idx_in_data_buffer = rand_idx * u64s_per_cacheline;
        aop[i * u64_ptrs_per_cacheline] =
            &data_buffer[corresponding_idx_in_data_buffer];

        /* dprint("[dbg-alloc] (%d/%d) aop[%u]\n", i, total_num_of_pointers, */
        /*         i * u64_ptrs_per_cacheline); */
        /* dprint("[dbg-alloc]\t(%p)\n", */
        /* 	   &aop[i * u64_ptrs_per_cacheline]); */
        /* dprint("[dbg-alloc]\t&data_buffer[%llu]\n", */
        /*         corresponding_idx_in_data_buffer); */
        /* dprint("[dbg-alloc]\t(%p)\n", */
        /*         &data_buffer[corresponding_idx_in_data_buffer]); */

        if (i < num_of_train_pointers + num_of_test_pointers) {
            dprint("[dbg-alloc] (%d) aop[%u](%p) <- &data_buffer[%llu](%p)\n",
                    i, i * u64_ptrs_per_cacheline,
                    &aop[i * u64_ptrs_per_cacheline],
                    corresponding_idx_in_data_buffer,
                    &data_buffer[corresponding_idx_in_data_buffer]);
        }

        // Print the last aop pointer accessed during training
        if (i == num_of_train_pointers - 1) {
            printf("[+] Last access of training will be aop[%u](%p)=%p\n",
                    i * u64_ptrs_per_cacheline,
                    &aop[i * u64_ptrs_per_cacheline],
                    aop[i * u64_ptrs_per_cacheline]);
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
    for (uint32_t j = 0; j < num_of_train_pointers + offs_past_train_buf - 1; j++) {
        rand_idx = prng(rand_idx);
    }
    uint64_t test_idx_in_data_buffer = rand_idx * u64s_per_cacheline;

    printf("[+] Address to test is data_buffer[%llu] (%p) = %llu\n",
            test_idx_in_data_buffer,
            &data_buffer[test_idx_in_data_buffer],
            data_buffer[test_idx_in_data_buffer]);
    printf("[+] The above should match the content of aop[%u]\n",
            (num_of_train_pointers + offs_past_train_buf - 1) *
            u64_ptrs_per_cacheline);
    printf("\t(%p)",
            &aop[(num_of_train_pointers + offs_past_train_buf - 1) *
            u64_ptrs_per_cacheline]);
    printf("\t= %p -> %llu\n",
            aop[(num_of_train_pointers + offs_past_train_buf - 1) *
            u64_ptrs_per_cacheline],
            *aop[(num_of_train_pointers + offs_past_train_buf - 1) *
            u64_ptrs_per_cacheline]);

    // uint64_t paddr;
    // lkmc_pagemap_virt_to_phys_user(&paddr, (uintptr_t)aop);
    // lkmc_pagemap_virt_to_phys_user(&paddr_end,
    // (uintptr_t)(&aop[(num_of_train_pointers + offs_past_train_buf - 1) *
    // u64_ptrs_per_cacheline]));
    // printf("[+] Physical address of test: 0x%lx\n", paddr);

    // Double-check that the data idx we will test corresponds to the test
    // pointer of the aop we want
    assert(aop[(num_of_train_pointers + offs_past_train_buf - 1) *
            u64_ptrs_per_cacheline] ==
            &data_buffer[test_idx_in_data_buffer]);

    // Parse the number of repetitions
    int repetitions = 0;
    sscanf(argv[3], "%d", &repetitions);

    // Allocate array for results
    uint64_t *times_to_load_test_ptr_baseline = malloc(repetitions *
            sizeof(uint64_t));
    uint64_t *times_to_load_test_ptr_aop = malloc(repetitions *
            sizeof(uint64_t));
    uint64_t *times_to_load_train_ptr_baseline = malloc(repetitions *
            sizeof(uint64_t));
    uint64_t *times_to_load_train_ptr_aop = malloc(repetitions *
            sizeof(uint64_t));

    uint64_t *curr_test_base = times_to_load_test_ptr_baseline;
    uint64_t *curr_test_aop = times_to_load_test_ptr_aop;
    uint64_t *curr_train_base = times_to_load_train_ptr_baseline;
    uint64_t *curr_train_aop = times_to_load_train_ptr_aop;

    // Vars for use in time loop
    uint64_t T1 = 0, T2 = 0;
    uint64_t train_time;

    printf("[+] Starting experiment\n");

#ifdef KEXT
    // Register a SIGILL handler to reinitialize the timer if needed
    // Needed in case of a core switch
    printf("[+] Initializing the PMC\n");
    signal(SIGILL, sighandler_local);
    init_timer();
#endif

    uint8_t aop_mode = 0;

    // Collect data
    for (uint32_t i = 0; i < repetitions * 2; i++) {

        if (aop_mode == 1) {
            // fill first 5 lines of aop with the same pointer
            for (int i = 0; i * u64_ptrs_per_cacheline < 16 * 5;
                    i++) {
                dprint("Setting aop[%i] = 0\n",
                        i * u64_ptrs_per_cacheline);
                aop[i * u64_ptrs_per_cacheline] = data_buffer;
            }
        } else {
            // fill first 5 lines of aop with addresses
            rand_idx = 1;
            for (int i = 0; i * u64_ptrs_per_cacheline < 16 * 5;
                    i++) {
                corresponding_idx_in_data_buffer = rand_idx * u64s_per_cacheline;
                dprint("Setting aop[%i] = %p\n",
                        i * u64_ptrs_per_cacheline,
                        &data_buffer[corresponding_idx_in_data_buffer]);
                aop[i * u64_ptrs_per_cacheline] =
                    &data_buffer[corresponding_idx_in_data_buffer];
                rand_idx = prng(rand_idx);
            }
        }

        // Avoid speculation
        MEM_BARRIER;

        // Thrash the cache
        for (uint32_t j = 0; j < size_of_thrash_array / sizeof(uint64_t) - 2;
                j++) {
            __trash += (thrash_arr[j] ^ __trash) & 0b1111;
            __trash += (thrash_arr[j + 1] ^ __trash) & 0b1111;
            __trash += (thrash_arr[j + 2] ^ __trash) & 0b1111;
        }

        // Time entire aop training
#ifdef MACH
#warning "Using system timer"
        T1 = get_time_nano(__trash & MSB_MASK);
#else
#ifdef KEXT
#warning "Using kext timer"
        dprint("[debug] Reading PMC...\n");
        T1 = READ_TIMER;
        dprint("[debug] Read PMC\n");
#else
#warning "Using threaded timer"
		T1 = *(&ticks + (__trash & MSB_MASK));
#endif
#endif

        __trash = (__trash + T1) & MSB_MASK;

        // Avoid speculation
        MEM_BARRIER;

        // Training loop
        // Alternate accesses through the AOP or through the data array
        aop_mode = aop_mode ^ 0x1;
        /* printf("%u-->", aop_mode); */
        uint32_t aop_idx = 0;
        rand_idx = 1;
        for (uint32_t j = 0; j < num_of_train_pointers /* + 1 */; j++) {
            __trash = (rand_idx + aop_idx + j) | (__trash & MSB_MASK);

            // Direct data access
            // When aop_mode == 1 we always load data_buffer[16]
            // Otherwise, when aop_mode == 0, we follow the data accesses
            // in the order we filled the AOP
            corresponding_idx_in_data_buffer = rand_idx * (1 - aop_mode) *
                u64s_per_cacheline;

            dprint("[dbg] (%u,%u) Accessed aop[%llu](%p)-->%p / "
                    " data_buffer[%llu](%p)\n", i, j,
                    (aop_idx) * u64_ptrs_per_cacheline |
                    (__trash & MSB_MASK),
                    &aop[(aop_idx) * u64_ptrs_per_cacheline |
                    (__trash & MSB_MASK)],
                    aop[(aop_idx) * u64_ptrs_per_cacheline |
                    (__trash & MSB_MASK)],
                    corresponding_idx_in_data_buffer | (__trash & MSB_MASK),
                    &data_buffer[corresponding_idx_in_data_buffer | (__trash &
                        MSB_MASK)]);

            __trash = data_buffer[
                corresponding_idx_in_data_buffer | (__trash & MSB_MASK)];

            // Data access through AoP
            // When aop_mode == 1, we always load the next pointer in the aop
            // Otherwise, when aop_mode == 0, we always load aop[0] (which is =
            // data_buffer[16])
            // The modulo operation prevents speculatively accessing data past
            // the end of the array during training.
            __trash = *aop[(aop_idx % num_of_train_pointers) *
                u64_ptrs_per_cacheline | (__trash & MSB_MASK)];

            /* uint64_t inc = j != (num_of_train_pointers-1) ? aop_mode : 16385 * */
            /*     aop_mode; */

            // Compute indices for the next loop iteration
            aop_idx += aop_mode | (__trash & MSB_MASK);
            rand_idx = prng(rand_idx | (__trash & MSB_MASK));
        }

        // Access one extra pointer after 2 MB to test aliasing
        /* __trash = *aop[(aop_idx + 16384) * u64_ptrs_per_cacheline | (__trash & */
        /* 		MSB_MASK)]; */

        // Avoid speculation
        MEM_BARRIER;

        // Time entire aop training
#ifdef MACH
		T2 = get_time_nano(__trash & MSB_MASK);
#else
#ifdef KEXT
        T2 = READ_TIMER;
#else  // threaded
		T2 = *(&ticks + (__trash & MSB_MASK));
#endif
#endif

        train_time = T2 - T1;

        // Wait a bit for any training loads to fully complete
        /* int retval = nanosleep((const struct timespec[]){{0, 10000000L}}, */
        /*         NULL); */

        // Time loading a single test ptr
#ifdef MACH
        T1 = get_time_nano((__trash + test_idx_in_data_buffer) & MSB_MASK);
#else
#ifdef KEXT
        T1 = READ_TIMER;
#endif
        T1 = *(&ticks + ((__trash + test_idx_in_data_buffer) & MSB_MASK) +
                (train_time & MSB_MASK));
#endif

        __trash = data_buffer[test_idx_in_data_buffer | (T1 & MSB_MASK)];
        // __trash = *aop[(num_of_train_pointers + offs_past_train_buf - 1) *
        // u64_ptrs_per_cacheline | (T1 & MSB_MASK)];

#ifdef MACH
		T2 = get_time_nano(__trash & MSB_MASK);
#else
#ifdef KEXT
        T2 = READ_TIMER;
#else  // threaded
		T2 = *(&ticks + (__trash & MSB_MASK));
#endif
#endif

        assert(((__trash + test_idx_in_data_buffer) & MSB_MASK) == 0);
        assert((T1 & MSB_MASK) == 0);
        assert((__trash & MSB_MASK) == 0);

        // Avoid speculation
        MEM_BARRIER;

        /* printf("[+] The extra access was at aop[%u] (%p) = %p -> %" PRIx64 */
        /*         "\n", (aop_idx) * u64_ptrs_per_cacheline, */
        /*         &aop[(aop_idx) * u64_ptrs_per_cacheline], */
        /*         aop[(aop_idx) * u64_ptrs_per_cacheline], */
        /*         *aop[(aop_idx) * u64_ptrs_per_cacheline]); */

        // Store the time that it took to load the test ptr
        if (aop_mode == 0) {
            /* times_to_load_test_ptr_baseline[i / 2] = T2 - T1; */
            *curr_test_base = T2 - T1;
            /* times_to_load_train_ptr_baseline[i / 2] = train_time; */
            *curr_train_base = train_time;
            /* printf("%u-->%u (%llu : %llu) (%llu : %llu)\n", aop_mode, i/2, */
            /* 		*curr_test_base, T2 - T1, */
            /* 		*curr_train_base, train_time); */
            curr_test_base++;
            curr_train_base++;
        } else {
            /* times_to_load_test_ptr_aop[i / 2] = T2 - T1; */
            *curr_test_aop = T2 - T1;
            /* times_to_load_train_ptr_aop[i / 2] = train_time; */
            *curr_train_aop = train_time;
            /* printf("%u-->%u (%llu : %llu) (%llu : %llu)\n", aop_mode, i/2, */
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
    FILE *output_file_aop = fopen("./out/aop.out", "w");
    if (output_file_baseline == NULL || output_file_aop == NULL) {
        perror("output files");
    }

    // Store measurements
    fprintf(output_file_baseline, "%p,%p\n", aop, data_buffer);
    fprintf(output_file_aop, "%p,%p\n", aop, data_buffer);
    fprintf(output_file_baseline, "test,train\n");
    fprintf(output_file_aop, "test,train\n");
    for (uint32_t i = 0; i < repetitions; i++) {
        fprintf(output_file_baseline, "%llu,%llu\n",
                times_to_load_test_ptr_baseline[i],
                times_to_load_train_ptr_baseline[i]);
        fprintf(output_file_aop, "%llu,%llu\n",
                times_to_load_test_ptr_aop[i],
                times_to_load_train_ptr_aop[i]);

    }

    // Clean up
    fclose(output_file_baseline);
    fclose(output_file_aop);

#if !defined (MACH) && !defined (KEXT)
    // Stop timer thread
    stop_timer = 1;
    pthread_join(timer_thread, NULL);
#endif
}
