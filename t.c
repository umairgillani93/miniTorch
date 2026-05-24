#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"
#include "config.h"
#include "arena.h"



Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target) {
	int rows = pred->shape[0];
	int cols = pred->shape[1];
	int ndim = pred->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *sub = tensor_subtract(A, pred, target);
	Tensor *sq = tensor_square(A, sub, sub);
	Tensor *mu = tensor_mean(A, sq);

	Tensor *out = tensor_expand_cols(A, mu, cols);

	return out;

}


int main() {

	srand(time(NULL));

	Arena *A = malloc(sizeof(Arena));
	arena_init(A, ARENA_SIZE);
	printf("Arena initilized\n");


	int ndim = 2;
	int *shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = 16;
	shape[1] = 32;

	Tensor *x = tensor_fill_ones(A, ndim, shape);
	x->requires_grad = true;

	tensor_get_2d(x);
	return 0;

}
