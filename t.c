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

	// Tensor graph metadata
	//if (pred->requires_grad || target->requires_grad) {
	//	out->requires_grad = true;
	//	out->num_parents = 2;
	//	out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
	//	out->parents[0] = pred;
	//	out->parents[1] = target;

	//	// define out Operations
	//	Op *op = arena_alloc(A, sizeof(Op));
	//	op->backward = tensor_mse_backward;
	//	op->type = MSE;
	//	op->name = "OP_MSE";

	//	// out grad
	//	out->grad = tensor_create_new(A, ndim, out_shape);
	//	out->cols = cols
	//}

	return out;

}


int main() {

	srand(time(NULL));

	Arena *A = malloc(sizeof(Arena));
	arena_init(A, ARENA_SIZE);
	printf("Arena initilized\n");

	int features = 32;

	LayerNorm *ln = layer_norm_create_new(A, features);
	printf("Iniitiazing parameters for Layer Norm:\n");
	layer_norm_init_params(ln);

	int ndim = 2;
	int *shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = 16;
	shape[1] = 32;

	Tensor *x = tensor_create_new(A, ndim, shape);
	x->requires_grad = true;
	tensor_randomize(x);

	Tensor *y = tensor_create_new(A, ndim, shape);
	tensor_randomize(y);

	Tensor *z = tensor_mse_loss(A, ln, x);
	//tensor_get_2d(z);
	//tensor_shape_2d(z);;
	tensor_metadata(z);
	return 0;

}
