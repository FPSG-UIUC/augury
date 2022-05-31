#include "util.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include <mach/mach.h>
#include <mach/mach_time.h>
#include <setjmp.h>
#include <sys/mman.h>
#include <sys/sysctl.h>

void pin_cpu(size_t core_no)
{
	if (core_no <= 3) { // ICESTORM
		pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
	} else if (core_no <= 7) { // FIRESTORM
		pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
	} else {
		assert(0 && "error! make sure 0 <= core_no <= 7");
	}
}

uint64_t (*mach_ptr)(void) = &mach_absolute_time;
uint64_t (*ns_ptr)(clockid_t) = &clock_gettime_nsec_np;

uint64_t get_time_nano(int32_t zero_dependency)
{
    /* MEM_BARRIER; */
	uint64_t t = zero_dependency;
	/* t += mach_absolute_time(); */
    t += (*(ns_ptr + zero_dependency))(CLOCK_UPTIME_RAW);
	/* t += READ_TIMER; */
    /* MEM_BARRIER; */
	return t;
}

//---------------------------------------------------------------------
// Kext specific stuff below
//---------------------------------------------------------------------

#if __arm__ && KEXT

static void set_sysctl(const char *name, uint64_t v)
{
	if (sysctlbyname(name, NULL, 0, &v, sizeof v)) {
		printf("set_sysctl: sysctlbyname failed\n");
		exit(1);
	}
}

static void
sighandler()
{
	/* printf("SIGILL\n"); */
	/* printf ("[+] Current core: %d\n", get_current_core ()); */

	assert(0);
	/* printf("Setting up timer again... "); */
	/* init_timer(); */
	/* printf("done setting up timer.\n"); */

	/* printf("Jumping... "); */
	/* longjmp(retry, 1); */
	/* printf("done jumping.\n"); */
}

void init_timer(void)
{

	MEM_BARRIER;
	set_sysctl("kern.pmcr0", 0x3003400ff4ff);
	MEM_BARRIER;
	SREG_WRITE(SREG_PMCR1, 0x3000003ff00);
	SREG_WRITE(SREG_PMCR0, DISABLE);
	SREG_WRITE(SREG_PMESR0, 0x02);
	SREG_WRITE(SREG_PMESR1, 0x02);
	SREG_WRITE(SREG_PMC0, 0);
	SREG_READ(SREG_PMC0);
	MEM_BARRIER;
}

void init_retry_barrier(void)
{
	signal(SIGILL, sighandler);
}

void set_retry_barrier(void)
{
	assert(0);
	/* setjmp(retry); */
}

void reset_timer(void)
{
	RESET_TIMER;
}
#endif // __arm__ && KEXT
