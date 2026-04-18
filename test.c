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

	int arr_size = 4;
	int *arr = (int *)arena_alloc(A, arr_size);
	for (int i = 0; i < arr_size; i++) {
		printf("%d\n", arr[i]);
	}

	return 0;

}
