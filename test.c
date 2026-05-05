#include <stdio.h>
#include <stdlib.h>

typedef struct {
	char *base;
	int size;
	int offset;
} Arena;

void arena_init(Arena *A, int size) {
	A->base = (char *)malloc(size);
	A->size = size;
	A->offset = 0;
}

void* arena_alloc(Arena *A, int size) {
	void *ptr = A->base + A->offset;
	A->offset += size * sizeof(int);
	return ptr;
}	


int main() {
	int SIZE = 1024 * 1024;

	Arena *A = malloc(sizeof(Arena));;
	arena_init(A, SIZE);
	printf("Arean created!\n");


	char *op = arena_alloc(A, sizeof(char));
	op = "MATMUL";
	printf("%s\n", op);

	return 0;

}
