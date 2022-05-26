#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include "util/util.h"

#ifdef __MACH__
#include <setjmp.h>
jmp_buf retry;
#endif // __MACH__

#ifdef __MACH__
#include <inttypes.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <setjmp.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#define SREG_PMCR0 "S3_1_c15_c0_0"
#define SREG_PMCR1 "S3_1_c15_c1_0"
#define SREG_PMESR0 "S3_1_c15_c5_0"
#define SREG_PMESR1 "S3_1_c15_c6_0"
#define SREG_PMC0 "S3_2_c15_c0_0"
#define SREG_PMC1 "S3_2_c15_c1_0"
#define SREG_PMC2 "S3_2_c15_c2_0"
#define SREG_PMC3 "S3_2_c15_c3_0"
#define SREG_PMC4 "S3_2_c15_c4_0"
#define SREG_PMC5 "S3_2_c15_c5_0"
#define SREG_PMC6 "S3_2_c15_c6_0"
#define SREG_PMC7 "S3_2_c15_c7_0"
#define SREG_PMC8 "S3_2_c15_c9_0"
#define SREG_PMC9 "S3_2_c15_c10_0"

#define ENABLE 0x3003400ff4ff
#define DISABLE 0x3000400ff403

#define MSB_MASK 0x8000000000000000
#define DEP(x) (x & MSB_MASK)

#define SREG_WRITE(SR, V)                                                      \
    __asm__ volatile("msr " SR ", %0 \r\n isb \r\n dsb sy \r\n"                \
                     :                                                         \
                     : "r"((uint64_t)V))
#define SREG_READ(SR)                                                          \
    ({                                                                         \
        uint64_t VAL = 0;                                                      \
        __asm__ volatile("isb \r\n mrs %0, " SR " \r\n isb \r\n dsb sy \r\n"   \
                         : "=r"(VAL));                                         \
        VAL;                                                                   \
    })
#define INST_SYNC asm volatile("ISB")
#define DATA_SYNC asm volatile("DSB SY")
#define MEM_BARRIER                                                            \
    DATA_SYNC;                                                                 \
    INST_SYNC

#define RESET_TIMER SREG_WRITE(SREG_PMC0, 0)
#define READ_TIMER SREG_READ(SREG_PMC0)

#define KB(x) (x * 1024)
#define MB(x) (x * 1024 * 1024)

// Use a Lehmer RNG as PRNG
// https://en.wikipedia.org/wiki/Lehmer_random_number_generator
#define PNRG_a 75
#define PRNG_m 8388617
#define prng(x) ((PNRG_a * x) % PRNG_m)

#ifdef DEBUGG
#define dprint(...) printf(__VA_ARGS__)
#define dassert(x) assert(x)
#else
#define dprint(...)                                                            \
    do {                                                                       \
    } while (0)
#define dassert(x)                                                             \
    do {                                                                       \
    } while (0)
#endif // DEBUGG

static void set_sysctl(const char *name, uint64_t v) {
    if (sysctlbyname(name, NULL, 0, &v, sizeof v)) {
        printf("set_sysctl: sysctlbyname failed\n");
        exit(1);
    }
}

#ifdef KEXT
static void sighandler() {
    init_timer();
    longjmp(retry, 1);
}
#endif //KEXT

static inline uint64_t mach_get_time_nano(int32_t zero_dependency) {
    uint64_t t = zero_dependency;
#ifdef KEXT
    t += READ_TIMER;
#else
    t += mach_absolute_time();
#endif // KEXT
    return t;
}
#endif // __MACH__

#define CACHE_LINE_SIZE 128 /* bytes */

static int ipow(int base, int exp) {
    if (exp == 0)
        return 1;
    return base;
}

void fill_buffer_with_rand(uint64_t *buf, size_t bufsz) {
    for (size_t i = 0; i < bufsz / sizeof(uint64_t); ++i)
        buf[i] = rand() & (MSB_MASK - 1);
}

int main(int argc, char **argv) {
    // pthread_t timer_thread;
    // stop_timer = 0;
    // int core_ID_timer = core_ID + 1;
    // pthread_create(&timer_thread, NULL, clock_thread, (void *)
    // &core_ID_timer); while (ticks == 0) {} printf("Started Clock Thread\n");
    // (OS X) Set up signal handler for PMC
    srand(time(NULL));

    // Make this code run on a performance core; cores 0-3 are icestorm and
    // cores 4-7 are firestorm
    pin_cpu(6);

    // Set up signal handler for SIGILL on timer context switches
#ifdef KEXT
    signal(SIGILL, sighandler);
#endif // KEXT

    // Parse the number of training pointers
    int num_of_train_pointers = 256;
    int num_of_test_pointers = 16;
    int total_num_of_pointers = num_of_train_pointers + num_of_test_pointers;

    // Set up the DMP array (AoP)
    int size_of_memory_touched_in_training =
        num_of_train_pointers * CACHE_LINE_SIZE; // one pointer per cache line
    int size_of_memory_checked_in_testing =
        num_of_test_pointers * CACHE_LINE_SIZE; // one pointer per cache line
    int total_memory_allocated_for_aop =
        size_of_memory_touched_in_training + size_of_memory_checked_in_testing;
    // volatile uint64_t **aop = aligned_alloc(MB(2),
    // total_memory_allocated_for_aop);
    volatile uint64_t **aop =
        mmap(0, total_memory_allocated_for_aop, PROT_READ | PROT_WRITE,
             MAP_ANON | MAP_PRIVATE, -1, 0);
    /* aop += 1; */
    assert(aop != MAP_FAILED);
    assert((uintptr_t)(aop + size_of_memory_touched_in_training / 8) %
           (1 << 21)); // Ensure not against 2MB bound
    memset(aop, 0, total_memory_allocated_for_aop);

    printf("[+] Num train ptrs: %d\n", num_of_train_pointers);
    printf("[+] Total num ptrs: %d\n", total_num_of_pointers);
    printf("[+] AOP is at %p-%p (%d bytes)\n", aop,
           (uint64_t **)((uint64_t)aop + total_memory_allocated_for_aop),
           total_memory_allocated_for_aop);

    // Set up the data array that the AoP holds ptrs to
    // For the data array we allocate a fixed amount of memory that
    // amounts to `PRNG_m` cache lines. Note that, depending on the
    // aop size, we may not need those many cache lines.
    int size_of_data_array = CACHE_LINE_SIZE * PRNG_m;
    volatile uint64_t *data_buffer =
        mmap(0, size_of_data_array, PROT_READ | PROT_WRITE,
             MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(data_buffer != MAP_FAILED);
    assert((uintptr_t)data_buffer != 0x280000000);
    fill_buffer_with_rand(data_buffer, size_of_data_array);
    printf("[+] Allocated data_buffer\n");
    printf("[+] Data buf size is at %p-%p (%d bytes)\n", data_buffer,
           (uint64_t *)((uint64_t)data_buffer + size_of_data_array),
           size_of_data_array);
    printf("[+] Data buf fits %d cache lines and %ld uint64ts\n", PRNG_m,
           size_of_data_array / sizeof(*data_buffer));

    size_t offset_const = 128;
    size_t testbuffer_sz = 16384 * 256;
    uint8_t *testbuffer = mmap(0, testbuffer_sz, PROT_READ | PROT_WRITE,
                               MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(testbuffer != MAP_FAILED);
    fill_buffer_with_rand((uint64_t *)testbuffer, testbuffer_sz);

    uint8_t *test1 = testbuffer + (rand() % 256) * 16384 - offset_const;
    uint8_t *test2 = testbuffer + (rand() % 256) * 16384 - offset_const;
    uint8_t *test3 = testbuffer + (rand() % 256) * 16384 - offset_const;
    size_t chosen = 0;

    printf("[+] 0 - %p\n", test1 + offset_const);
    printf("[+] 1 - %p\n", test2 + offset_const);
    printf("[+] 2 - %p\n", test3 + offset_const);
    printf("[+] pick a pointer 0, 1, or 2: ");
    scanf("%zu", &chosen);
    assert(0 <= chosen && chosen <= 2);

    uint8_t *test_p = (uint8_t *)((0 == chosen) * (uintptr_t)test1 |
                                  (1 == chosen) * (uintptr_t)test2 |
                                  (2 == chosen) * (uintptr_t)test3);
    printf("[+] you picked %zu (%p)\n", chosen, test_p + offset_const);

    // Make pointers in the AoP point to pseudo-random locations in the data
    // array
    uint64_t rand_idx = 1;
    uint64_t corresponding_idx_in_data_buffer;
    uint32_t u64s_per_cacheline =
        CACHE_LINE_SIZE / sizeof(uint64_t); // should be 16
    uint32_t u64_ptrs_per_cacheline = CACHE_LINE_SIZE / sizeof(uint64_t *);
    for (int i = 0; i < total_num_of_pointers; i++) {

        // Make i-th cache line in the aop point to somewhere in the data buffer
        corresponding_idx_in_data_buffer = rand_idx * u64s_per_cacheline;
        aop[i * u64_ptrs_per_cacheline] =
            &data_buffer[corresponding_idx_in_data_buffer];

        // Make all of the test pointers the pointer we are trying to guess
        if (i >= num_of_train_pointers) {
            aop[i * u64_ptrs_per_cacheline] =
                (uint64_t *)(test_p + offset_const);
        }

        // Update index into the data buffer for next aop pointer
        rand_idx = prng(rand_idx);
    }

    fprintf(
        stdout,
        "[+] we've set the %u pointers after the first %u train pointers to:\n",
        total_num_of_pointers - num_of_train_pointers, num_of_train_pointers);
    for (size_t i = num_of_train_pointers; i < total_num_of_pointers; ++i)
        printf("[+] aop[%zu * %u] <- %p\n", i, u64s_per_cacheline,
               aop[i * u64s_per_cacheline]);
    printf("[+] now we will iterate over the first %u "
           "train pointers to do a flush+reload on these three available "
           "pointers\n",
           num_of_train_pointers);

    // Get rid of test_p for now just to make sure we aren't accidentally
    // getting it some way other through the DMP cache side channel
    test_p = NULL;
    chosen = 14; // Some nonsense magic number just greater than # pointers
                 // available for guessing

    // For preventing unwanted compiler optimizations and adding
    // data dependencies between instructions.
    uint64_t __trash = 0;

    // Allocate a large array (12 times the cache) that we can access to flush
    // the entire cache. High performance cores: 12MB of shared L2 cache, 192KB
    // L1 instruction cache, and 128KB L1 data cache. High efficiency cores: 4MB
    // of shared L2 cache, 128KB L1 instruction cache, and 64KB L1 data cache.
    int size_of_thrash_array = (MB(12) + KB(128)) * 12;
    volatile uint64_t *thrash_arr =
        mmap(0, size_of_thrash_array, PROT_READ | PROT_WRITE,
             MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(thrash_arr != MAP_FAILED);
    fill_buffer_with_rand(thrash_arr, size_of_thrash_array);

    // Allocate array for results and timing data
    size_t repetitions = 3; // repetitions per test pointr guess
    uint64_t *times_to_load_test_ptr_baseline =
        calloc(repetitions, sizeof(*times_to_load_test_ptr_baseline));
    uint64_t *times_to_load_test_ptr_aop =
        calloc(repetitions, sizeof(*times_to_load_test_ptr_aop));
    assert(times_to_load_test_ptr_aop && times_to_load_test_ptr_baseline);

    uint64_t **times = calloc(3, sizeof(uint64_t *));
    assert(times);
    for (size_t i = 0; i < 3; i++) {
        times[i] = calloc(repetitions, sizeof(uint64_t));
        assert(times[i]);
    }

    // Vars for use in time loop
    uint64_t T1 = 0, T2 = 0;

    printf("[+] Starting experiment\n");

#ifdef KEXT
    setjmp(retry);
    init_timer();
#endif // KEXT

    // Main experiment loop
    for (size_t ptr_no = 0; ptr_no < 3; ptr_no++) {
        // Set test_p to either test1, test2, or test3 depending on what ptr_no
        // is
        test_p = (uint8_t *)((0 == ptr_no) * (uintptr_t)test1 |
                             (1 == ptr_no) * (uintptr_t)test2 |
                             (2 == ptr_no) * (uintptr_t)test3);

        // Collect data
        for (uint32_t i = 0; i < repetitions; i++) {
            // Avoid speculation
            MEM_BARRIER;

            // Thrash the cache
            for (uint32_t j = 0;
                 j < size_of_thrash_array / sizeof(uint64_t) - 2; j++) {
                __trash += (thrash_arr[j] ^ __trash) & 0b1111;
                __trash += (thrash_arr[j + 1] ^ __trash) & 0b1111;
                __trash += (thrash_arr[j + 2] ^ __trash) & 0b1111;
            }

            // Avoid speculation
            MEM_BARRIER;

            for (uint32_t j = 0; j < num_of_train_pointers; j++) {
                __trash +=
                    *aop[(j % num_of_train_pointers) * u64_ptrs_per_cacheline |
                         (__trash & MSB_MASK)];
            }

            // Avoid speculation
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;
            MEM_BARRIER;

            // Time loading a single test ptr
            T1 = DEP(__trash);
            T1 += mach_get_time_nano(T1);

            __trash += MSB_MASK & *((test_p + offset_const) + (T1 & MSB_MASK) +
                                    (__trash & MSB_MASK));

            T2 = DEP(__trash) | DEP(T1);
            T2 += mach_get_time_nano(T2);

            assert((T1 & MSB_MASK) == 0);
            assert((__trash & MSB_MASK) == 0);

            // Avoid speculation
            MEM_BARRIER;

            // Store the time that it took to load the test ptr
            times[ptr_no][i] = T2 - T1;
        }
    }

    setjmp(retry);

    printf("[+] Done!\n");
    printf("[+] Times were: \n");
    for (size_t ptr_no = 0; ptr_no < 3; ptr_no++) {
        printf("\t%zu:\n", ptr_no);
        for (size_t i = 0; i < repetitions; ++i) {
            printf("\t\t%llu\n", times[ptr_no][i]);
        }
    }
    printf("\n");
}
