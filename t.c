#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "config.h"
#include "arena.h"
#include "model.h"

#define RAND_FLOAT (float) rand() / (float) RAND_MAX

int main() {
	Arena *A = malloc(sizeof(Arena));
	arena_init(A, ARENA_SIZE);
	int ndim = 2;
	int shape[2] = {SEQ_LEN, EMB_DIM};

	Tensor *x = tensor_create_new(A, ndim, shape);
	tensor_randomize_weights(x);

	tensor_get(x);

	return 0;
}
