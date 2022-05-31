#ifndef _UTIL_H
#define _UTIL_H

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

#if __x86_64__ || __amd64__

// All three defined just in case some code is calling
// just insn sync or just data sync
#define INST_SYNC asm volatile("cpuid")
#define DATA_SYNC asm volatile("cpuid")
#define MEM_BARRIER asm volatile("cpuid")

#else
// Instruction Synchronization Barrier flushes the pipeline in the processor,
// so that all instructions following the ISB are fetched from cache or memory,
// after the instruction has been completed.
#define INST_SYNC asm volatile("ISB")

// Data Synchronization Barrier acts as a special kind of memory barrier. No
// instruction in program order after this instruction executes until this
// instruction completes. This instruction completes when:
// - All explicit memory accesses before this instruction complete.
// - All Cache, Branch predictor and TLB maintenance operations before this
//   instruction complete
#define DATA_SYNC asm volatile("DSB SY")

#define MEM_BARRIER \
	DATA_SYNC;      \
	INST_SYNC
#endif // __x86_64__ || __amd64__

#define MSB_MASK 0x8000000000000000

#define KB(x) (x * 1024)
#define MB(x) (x * 1024 * 1024)
#define TO_KB(x) (x/1024)
#define TO_MB(x) (x/1024/1024)

// void print_size(unsigned bytes) {
//     if (bytes <= KB(1)) {
//         printf("%uB", bytes);
//     } else if (bytes <= MB(1)) {
//         printf("%uKB", TO_KB(bytes));
//     } else {
//         printf("%uMB", TO_MB(bytes));
//     }
// }

// Use a Lehmer RNG as PRNG
// https://en.wikipedia.org/wiki/Lehmer_random_number_generator
#define PNRG_a 75
#define PRNG_m 8388617
#define prng(x) ((PNRG_a * x) % PRNG_m)

#ifdef DEBUGG
#define dprint(...) printf(__VA_ARGS__)
#define dassert(x) assert(x)
#else
#define dprint(...) \
	do {            \
	} while (0)
#define dassert(x) \
	do {           \
	} while (0)
#endif // DEBUGG

//---------------------------------------------------------------------
// OS-agnostic
//---------------------------------------------------------------------

void zeroize(void *pointer, size_t size_data);

uint64_t rand_uint64_slow(void);

struct Node {
	void *address;
	struct Node *next;
};

void append_string_to_linked_list(struct Node **head, void *addr);

#define CACHE_LINE_SIZE 128 /* bytes */

//---------------------------------------------------------------------
// OS-specific
//---------------------------------------------------------------------

void pin_cpu(size_t core_ID);
#ifdef MACH
uint64_t get_time_nano(int zero_dependency);

#elif defined KEXT
static void set_sysctl(const char*, uint64_t);
// static void sighandler();
void init_timer();
void init_retry_barrier();
void set_retry_barrier();
void reset_timer();

// KEXT Stuff
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

#define SREG_WRITE(SR, V)                                       \
	__asm__ volatile("msr " SR ", %0 \r\n isb \r\n dsb sy \r\n" \
					 :                                          \
					 : "r"((uint64_t)V))

#define SREG_READ(SR)                                                        \
	({                                                                       \
		uint64_t VAL = 0;                                                    \
		__asm__ volatile("isb \r\n mrs %0, " SR " \r\n isb \r\n dsb sy \r\n" \
						 : "=r"(VAL));                                       \
		VAL;                                                                 \
	})

#define RESET_TIMER SREG_WRITE(SREG_PMC0, 0)
#define READ_TIMER SREG_READ(SREG_PMC0)
#endif // KEXT

#endif // _UTIL_H
