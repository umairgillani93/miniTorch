#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>

typedef struct Arena {
	char *base;
	size_t size;
	size_t offset;
} Arena;

// prototype definitions
void arena_init(Arena *A, size_t size);
void arena_reset(Arena *A);
void *arena_alloc(Arena *A, size_t size);


#endif
