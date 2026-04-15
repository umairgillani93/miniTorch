#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "tensor.h"
#include "feed_forward_nn.h"
#include "attention2.h"
#include "arena.h"
#include "config.h"


void arena_init(Arena *A, int size) {
	A->base = (char *)malloc(size);
	A->size = size;
	A->offset = 0;
}

void *arena_alloc(Arena *A, int size) {
	void *ptr = A->base + A->offset;
	A->offset += size; // This should be size in bytes
	return ptr;
}	

int main() {
	Arena *A = malloc(sizeof(Arena));
	int SIZE = 1024 * 1024 * 1024;
	arena_init(A, SIZE);
	int ndim = 2;
	int shape[2] = {SEQ_LEN, EMB_DIM};
	Tensor *x = tensor_create_new(A, ndim, shape);
	Tensor *y = tensor_create_new(A, ndim, shape);

	Tensor *z = tensor_matmul(x, y);;

	MHA *m = mha_create_new(A, HEADS, SEQ_LEN, EMB_DIM);
	printf("heads: %d\n", m->num_heads);

	int wq_size = tensor_size(m->wq);
	for (int i = 0; i < wq_size; i++) {
		printf("item: %f ", m->wq->data[i]);
	}
	printf("\n");

	return 0;
}
