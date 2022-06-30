
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <setjmp.h>
#include <string.h>
#include <pthread.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;
typedef int32_t i32;

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

#define SREG_WRITE(SR, V)                                                      \
  __asm__ volatile("msr " SR ", %0 \r\n isb \r\n dsb sy \r\n" : : "r"((u64)V))
#define SREG_READ(SR)                                                          \
  ({                                                                           \
    uint64_t VAL = 0;                                                          \
    __asm__ volatile("isb \r\n mrs %0, " SR " \r\n isb \r\n dsb sy \r\n" : "=r"(VAL));     \
    VAL;                                                                       \
  })
#define RESET_TIMER SREG_WRITE(SREG_PMC0, 0)
#define READ_TIMER SREG_READ(SREG_PMC0)

#define clflush(x) *x+=1

#define INST_SYNC asm volatile ("ISB")
// Instruction Synchronization Barrier flushes the pipeline in the processor,
// so that all instructions following the ISB are fetched from cache or memory,
// after the instruction has been completed.
#define DATA_SYNC asm volatile ("DSB SY")
// Data Synchronization Barrier acts as a special kind of memory barrier. No
// instruction in program order after this instruction executes until this
// instruction completes. This instruction completes when:
// - All explicit memory accesses before this instruction complete.
// - All Cache, Branch predictor and TLB maintenance operations before this
//   instruction complete
#define MEM_BARRIER DATA_SYNC; INST_SYNC
#define MSB_MASK 0x80000000

#define LOAD(pointer) asm volatile ("LDR %[val], [%[ptr]]\n\t"  \
                                    : [val] "=r" (val)          \
                                    : [ptr] "r" (pointer)       \
                                    );

static void
set_sysctl (const char *name, u64 v)
{
  if (sysctlbyname(name, NULL, 0, &v, sizeof v)) {
    printf("set_sysctl: sysctlbyname failed\n");
    exit(1);
  }
}

#define OFFSET 128 * 1024 // 2 ** 7 * 2 ** 10 = 2 ** 17
static u64 inline
access_evset (u64* target, u32 set_size)
{
  MEM_BARRIER;
  volatile u64 val = 0;

  for (u32 s=1; s<set_size; s++)
    LOAD(target + (s * (OFFSET) | (val & 0x80000000)));

  MEM_BARRIER;
  return val & 0x80000000;
}

static void
thrash_cache (u64* e_arr, u32 len)
{
  u64 *e_curr = e_arr;

  for (u32 w=0; w<8; w++)
    for (u64 *th_end = e_arr + len; e_curr < th_end; e_curr+=(128/sizeof(u64)))
      clflush(e_curr);
}

void
init_timer (void)
{
  set_sysctl("kern.pmcr0", 0x3003400ff4ff);
  SREG_WRITE(SREG_PMCR1, 0x3000003ff00);
  SREG_WRITE(SREG_PMCR0, DISABLE);
  SREG_WRITE(SREG_PMESR0, 0x02);
  SREG_WRITE(SREG_PMESR1, 0x02);
  SREG_WRITE(SREG_PMC0, 0);
  MEM_BARRIER;
}

static int
get_current_core (void)
{
  return SREG_READ("TPIDRRO_EL0") & 7;
}

static jmp_buf retry;

static void
sighandler()
{
  printf ("SIGILL\n");
  /* printf ("Current core: %d\n", get_current_core ()); */
  init_timer ();
  longjmp (retry, 1);
}

static i32
was_l2_miss_timing (u64 *times, u64 ntimes)
{
  i32 all_misses = 1;
  u64 time_threshold = 150;

  for (u32 i = 0; i < ntimes; ++i)
    all_misses = all_misses && (times[i] >= time_threshold);

  return all_misses;
}

static u32
candidates_remaining (u8 *leave_outs, u64 num_leave_outs, i32 ncands)
{
  i32 left_out = 0;
  for (u32 i = 0; i < num_leave_outs; ++i)
    left_out += leave_outs[i];
  return ncands - left_out;
}

int
get_evset (u64 *target_p, u64 **evset_buff, u64 ***final_evset, u64 *evset_size)
{
  signal(SIGILL, sighandler);
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

  u64 ARR_SIZE = 2097289 * 32;
  u64 thrash_len = ARR_SIZE * 2 * sizeof (u64);
  u64 *thrash_arr = aligned_alloc (128, thrash_len);
  assert (thrash_arr);

  u64 *test_buffer = aligned_alloc (64, ARR_SIZE * sizeof (u64));
  assert (test_buffer);

  // Buffers for collecting timing data
  u64 fgr = 8;
  u64 *miss_data = malloc (sizeof (u64) * fgr);
  u64 *L1_hit_data = malloc (sizeof (u64) * fgr);
  u64 *L2_hit_data = malloc (sizeof (u64) * fgr);
  assert (miss_data && L1_hit_data && L2_hit_data);

  /* Start L2 eviction set init */
  // Pages are 16384 = 2 ** 14
  // L2 line size is 128 = 2 ** 7 = l = L2_LINE_SIZE_LOG
  // L2 size is 4194304 (from `sysctl -a`) = 2 ** 22
  // L2 num lines is 2 ** 22 / 2 ** 7 = 2 ** 15
  // Guessing assoc is 8-way, so 2 ** 15 lines / 2 ** 3 lines / set =
  //   2 ** 12 sets = 4096
  // -> bits 7 through 14 (8 bits) are attacker controlled = gamma
  // -> candidate addresses should coincide on bits 7 through 14
  // from "Selection of Initial Search Space" section of "Theory and Practice.."
  // randomly sample from a very large buffer, 100 addresses that are 2 ** (gamma + l)
  // 2 ** (gamma + l) = 2 ** (8 + 7) = 32768 -> buffer should be at least
  // 100 * 32768 = 3276800 bytes wide
#define L2_SIZE 4194304
#define L2_LINE_SIZE 128
#define L2_LINE_SIZE_LOG 7
#define L2_ASSOC_GUESS 48
#define L2_LINES 32768 /* 2 ** 15 */

#define BUF_BYTES (1 << 28)
#define STRIDE (1 << 14)
#define CANDS (BUF_BYTES / STRIDE)

#define RAND(x) (1031 * x + 17377) % CANDS
#define CONG_RAND(x) ((17 * x + 61) % L2_ASSOC_GUESS)
#define CAND_RAND(x) (17377 * x + 3673) % CANDS
#define SEED 773

#define L2_MISS_THRESH 150
#define IS_L2_MISS(x) (x >= L2_MISS_THRESH)
#define ATT_CONTROL_BITS 0b0011111110000000
#define CLEAR_BITS 0xFFFFFFFFFFFF0000

  u64 **candidates = calloc (CANDS, sizeof (u64*));

  /* u64 *buffer = aligned_alloc (L2_LINE_SIZE, BUF_BYTES); */
  u64 *buffer = mmap (0, BUF_BYTES, PROT_READ | PROT_WRITE,
                      MAP_ANON | MAP_PRIVATE, 0, 0);
  u64 *min_p = buffer;
  u64 *max_p = buffer + (BUF_BYTES / sizeof (u64));

  u64 candidate_idx;
  u8 *leave_out = calloc (CANDS, sizeof (u8));
  u64 cands_left = CANDS;
  u64 cur_rand = RAND (SEED);

  u64 removal_round = 0;
  u64 removal_idx = CAND_RAND (removal_round);
  u64 **congruencies = calloc (L2_ASSOC_GUESS, sizeof (u64*));
  u64 num_congr = 0;
  assert (buffer && "Couldn't allocate for L2 evsets test buffer");
  assert (candidates && "Couldn't allocate for candidate buffer");
  assert (leave_out && "Couldn't allocate for leave out buffer");
  memset (buffer, 0, BUF_BYTES);

  for (u32 i = 0; i < CANDS; ++i)
    {
      assert (cur_rand != 0);

      u64 *cur = buffer + (STRIDE * cur_rand) / sizeof (u64);
      cur = (u64*) (((uintptr_t) cur) & ((uintptr_t) CLEAR_BITS));
      cur = (u64*) (((uintptr_t) cur) | (((uintptr_t) target_p) & ((uintptr_t) ATT_CONTROL_BITS)));
      assert (cur <= max_p);
      assert (min_p <= cur);

      candidates[i] = cur;

      printf ("[+] candidates[%u] <- %p\n", i, candidates[i]);
      cur_rand = RAND (cur_rand);
    }
  printf ("[+] Buffer is %p-%p (size %d)\n", min_p, max_p, BUF_BYTES);
  printf ("[+] target_p is %p\n", target_p);
  /* End L2 eviction set init */

  // Vars for use in time loop
  u64 T1;
  u64 T2;
  u64 time_delta_1 = 0; // For measuring main mem access
  u64 time_delta_2 = 0; // for measuring L1 access
  u64 time_delta_3 = 0; // for measuring L2 access
  i32 __trash; // used to add dependencies and avoid optimizing out code
  i32 is_first_round = 1, l2_evicted = 0;

  /* Set up PMU for timing */
  setjmp (retry);
  init_timer ();
  printf ("finished setting up control and selector regs\n");
  /* end set up PMU */

  while (num_congr < L2_ASSOC_GUESS && cands_left > 0)
    {
      for (u32 rep = 0; rep < fgr; ++rep)
        {
          /* thrash_cache (thrash_arr, thrash_len / sizeof (u64)); */
          RESET_TIMER;
          MEM_BARRIER;

          /* Put target_p in L2 */
          T1 = READ_TIMER;
          __trash = *target_p | (T1 & MSB_MASK); // __trash = 0;
          T2 = READ_TIMER | (__trash & MSB_MASK) | (T1 & MSB_MASK);
          time_delta_1 = T2 - T1;
          MEM_BARRIER;
          access_evset (target_p + (__trash & MSB_MASK), 9);

          MEM_BARRIER;

          /* Access the so-far-found-to-be congruent ptrs */
          for (u32 i = 0; i < num_congr; ++i)
            {
              __trash += *congruencies[i] | (__trash & MSB_MASK);
            }
          /* Access the remaining evset */
          for (u32 i = 0; i < CANDS; ++i)
            {
              u64 cur = CAND_RAND (i);
              if (!leave_out[cur])
                __trash += *candidates[cur] | (__trash & MSB_MASK);
            }

          RESET_TIMER;
          MEM_BARRIER;

          /* Time access to main mem or L2 */
          T1 = READ_TIMER | (time_delta_1 & MSB_MASK) | (__trash & MSB_MASK);
          __trash = *target_p | (T1 & MSB_MASK);
          T2 = READ_TIMER | (__trash & MSB_MASK) | (T1 & MSB_MASK);
          time_delta_2 = T2 - T1;

          MEM_BARRIER;

          /* Record times */
          miss_data[rep] = time_delta_1;
          L2_hit_data[rep] = time_delta_2;
        }

      /* Check the recorded times. I think that only doing
         the experiment once might be good enough since there
         is little deviation (+- 3 cycles) compared to the
         timing differences (L1 is ~68, L2 is ~80, mem is 170-300)*/
      l2_evicted = was_l2_miss_timing (L2_hit_data, fgr);
      setjmp (retry);

      if (is_first_round && !l2_evicted)
        {
          printf ("[-] Didn't guess first candidate set right\n");
          __trash = 1;
          goto cleanup;
        }
      else if (is_first_round)
        {
          /* Important to distinguish this case because nothing was
             removed to test for congruency in round 1. */
          is_first_round = 0;
          leave_out[removal_idx] = 1;
        }
      else
        {
          if (l2_evicted)
            {
              /* There are at least a other congruent addresses or
                 it just isn't congruent to target_p */
              cands_left = candidates_remaining (leave_out, CANDS, CANDS);
              if (cands_left == 0)
                printf ("[+] 0 cands left -> num_congr is %llu\n", num_congr);
            }
          else
            {
              printf ("[+] Candidate %llu is a part of %p's evset!\n",
                      removal_idx, target_p);
              congruencies[num_congr++] = candidates[removal_idx];
            }

          /* On to the next one */
          removal_round++;
          removal_idx = CAND_RAND (removal_round);
          leave_out[removal_idx] = 1;
          setjmp (retry);
        }
    }

  /* Test to see if the minimal evset is actually an evset */
  /* This is just the body of the minimizing loop C&P'd */
  u64 tests = 20;
  u64 *evicted_times = calloc (tests, sizeof (u64));
  setjmp (retry);

  /* To clear out effects from previous logging */
  thrash_cache (thrash_arr, thrash_len / sizeof (u64));

  for (u32 rep = 0; rep < tests; ++rep)
    {
      RESET_TIMER;
      MEM_BARRIER;

      /* Put target_p in L2 */
      T1 = READ_TIMER;
      __trash = *target_p | (T1 & MSB_MASK); // __trash = 0;
      T2 = READ_TIMER | (__trash & MSB_MASK) | (T1 & MSB_MASK);
      time_delta_1 = T2 - T1;
      MEM_BARRIER;
      access_evset (target_p + (__trash & MSB_MASK), 9);

      MEM_BARRIER;

      /* Access candidate evset */
      for (u32 i = 0; i < num_congr; ++i)
        __trash += *congruencies[i] | (__trash & MSB_MASK) | (time_delta_1 & MSB_MASK);

      RESET_TIMER;
      MEM_BARRIER;

      /* Time access to main mem or L2 */
      T1 = READ_TIMER | (time_delta_1 & MSB_MASK) | (__trash & MSB_MASK);
      __trash = *target_p | (T1 & MSB_MASK);
      T2 = READ_TIMER | (__trash & MSB_MASK) | (T1 & MSB_MASK);
      time_delta_2 = T2 - T1;

      MEM_BARRIER;

      evicted_times[rep] = time_delta_2;
    }

  l2_evicted = was_l2_miss_timing (evicted_times, tests);
  if (l2_evicted)
    {
      printf ("[+] Found eviction set : %d\n", l2_evicted);
      *evset_buff = buffer;
      *final_evset = congruencies;
      *evset_size = num_congr;
    }
  else
    {
      printf ("[+] Didn't find eviction set : %d\n", l2_evicted);
      assert (!munmap (buffer, BUF_BYTES));
      free (congruencies);
      *evset_buff = 0;
      *final_evset = 0;
      *evset_size = 0;
    }
  /* End minimal evset test */

cleanup:
  free (miss_data);
  free (evicted_times);
  free (L1_hit_data);
  free (L2_hit_data);
  free (test_buffer);
  free (thrash_arr);
  free (candidates);
  free (leave_out);
  return __trash;
}

/* Usage */
/* int */
/* main (void) */
/* { */
/*   u64 *evict_ptr = aligned_alloc (128, 16384); */
/*   u64 *evset_buf; */
/*   u64 evset_size = 0; */
/*   u64 **evset; */

/*   evict_ptr += 16384 / 16; */
/*   printf ("[+] evict ptr is %p\n", evict_ptr); */

/*   i32 res = 1; */
/*   while (res || evset_size == 0) */
/*     res = get_evset (evict_ptr, &evset_buf, &evset, &evset_size); */

/*   for (u32 i = 0; i < evset_size; ++i) */
/*     printf ("[+] In main: evset[%u] <- %p\n", i, evset[i]); */

/*   return 0; */
/* } */
