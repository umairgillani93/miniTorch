#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "config.h"
#include "arena.h"



int main() {
	srand(time(NULL));
	Arena *A = malloc(sizeof(Arena));
	size_t SIZE = 1024 * 1024;
	arena_init(A, ARENA_SIZE);

	int ndim = 2;
	int *shape_x = arena_alloc(A, ndim * sizeof(int)); // takes arena pointer and size
	int *shape_y = arena_alloc(A, ndim * sizeof(int));

	shape_x[0] = 16;
	shape_x[1] = 32;

	shape_y[0] = 32;
	shape_y[1] = 16;

	Tensor *x = tensor_create_new(A, ndim, shape_x);
	tensor_randomize_weights(x);
	x->requires_grad = true;
	//printf("tensor x: \n");
	Tensor *y = tensor_create_new(A,ndim,  shape_y);
	tensor_randomize_weights(y);
	Tensor *z = tensor_matmul(A, x, y);
	tensor_get_2d(z);
	return 0;
}


