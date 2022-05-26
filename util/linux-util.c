#include "util.h"

#include <sys/resource.h>
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>

void pin_cpu(size_t core_ID)
{
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(core_ID, &set);
	if (sched_setaffinity(0, sizeof(cpu_set_t), &set) < 0) {
		printf("Unable to Set Affinity\n");
		exit(EXIT_FAILURE);
	}

	// Set the scheduling priority to high to avoid interruptions
	// (lower priorities cause more favorable scheduling, and -20 is the max)
	setpriority(PRIO_PROCESS, 0, -20);
}

uint64_t get_time_nano(int zero_dependency)
{
	struct timespec ts;
	ts.tv_nsec = (zero_dependency);
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return 1000000000 * ts.tv_sec + (uint64_t)ts.tv_nsec;
}

// /* Format documented at:
//  * https://github.com/torvalds/linux/blob/v4.9/Documentation/vm/pagemap.txt
//  */
// typedef struct {
// 	uint64_t pfn : 55;
// 	unsigned int soft_dirty : 1;
// 	unsigned int file_page : 1;
// 	unsigned int swapped : 1;
// 	unsigned int present : 1;
// } LkmcPagemapEntry;

// /* Parse the pagemap entry for the given virtual address.
//  *
//  * @param[out] entry      the parsed entry
//  * @param[in]  pagemap_fd file descriptor to an open /proc/pid/pagemap file
//  * @param[in]  vaddr      virtual address to get entry for
//  * @return                0 for success, 1 for failure
//  */
// static int lkmc_pagemap_get_entry(LkmcPagemapEntry *entry, int pagemap_fd, uintptr_t vaddr)
// {
// 	size_t nread;
// 	ssize_t ret;
// 	uint64_t data;
// 	uintptr_t vpn;

// 	vpn = vaddr / sysconf(_SC_PAGE_SIZE);
// 	nread = 0;
// 	while (nread < sizeof(data)) {
// 		ret = pread(
// 			pagemap_fd,
// 			((uint8_t *)&data) + nread,
// 			sizeof(data) - nread,
// 			vpn * sizeof(data) + nread);
// 		nread += ret;
// 		if (ret <= 0) {
// 			return 1;
// 		}
// 	}
// 	entry->pfn = data & (((uint64_t)1 << 55) - 1);
// 	entry->soft_dirty = (data >> 55) & 1;
// 	entry->file_page = (data >> 61) & 1;
// 	entry->swapped = (data >> 62) & 1;
// 	entry->present = (data >> 63) & 1;
// 	return 0;
// }

// /* Convert the given virtual address to physical using /proc/PID/pagemap.
//  *
//  * @param[out] paddr physical address
//  * @param[in]  pid   process to convert for
//  * @param[in]  vaddr virtual address to get entry for
//  * @return           0 for success, 1 for failure
//  */
// int lkmc_pagemap_virt_to_phys_user(uintptr_t *paddr, uintptr_t vaddr)
// {
// 	int pagemap_fd;

// 	pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
// 	if (pagemap_fd < 0) {
// 		return 1;
// 	}
// 	LkmcPagemapEntry entry;
// 	if (lkmc_pagemap_get_entry(&entry, pagemap_fd, vaddr)) {
// 		return 1;
// 	}
// 	close(pagemap_fd);
// 	*paddr = (entry.pfn * sysconf(_SC_PAGE_SIZE)) + (vaddr % sysconf(_SC_PAGE_SIZE));
// 	return 0;
// }
