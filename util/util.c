#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

#include "util.h"

/*
 * Appends the given string to the linked list which is pointed to by the given head
 */
void append_string_to_linked_list(struct Node **head, void *addr)
{
	struct Node *current = *head;

	// Create the new node to append to the linked list
	struct Node *new_node = malloc(sizeof(*new_node));
	new_node->address = addr;
	new_node->next = NULL;

	// If the linked list is empty, just make the head to be this new node
	if (current == NULL)
		*head = new_node;

	// Otherwise, go till the last node and append the new node after it
	else {
		while (current->next != NULL)
			current = current->next;

		current->next = new_node;
	}
}

/*
 * Function that sets memory to zero and is not optimized out by the compiler.
 */
void zeroize(void *pointer, size_t size_data)
{
	volatile uint8_t *p = pointer;
	while (size_data--)
		*p++ = 0;
}

/*
 * Generates a random uint64_t value
 */
uint64_t rand_uint64_slow(void)
{
	uint64_t r = 0;
	for (int i = 0; i < 64; i++) {
		r = r * 2 + rand() % 2;
	}
	return r;
}
