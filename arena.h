#ifndef ARENA_H
#define ARENA_H

typedef struct Arena {
	char *base;
	size_t size;
	size_t offset;
} Arena;

// prototype definitions
void arena_init(Arena *A, int size);
void *arena_alloc(Arena *A, int size);

#endif
