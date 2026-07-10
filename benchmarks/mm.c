#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "config.h"
#include "arena.h"


Tensor *compare_matmul(Arena *A, Tensor *x, Tensor *y) {
	Tensor *z = tensor_matmul(A, x, y);
	return z;
}



int main() {
	srand(time(NULL));
	Arena *A = malloc(sizeof(Arena));
	size_t SIZE = 1024 * 1024;
	arena_init(A, ARENA_SIZE);

	int ndim = 2;
	int *shape_x = arena_alloc(A, ndim * sizeof(int)); // takes arena pointer and size
	int *shape_y = arena_alloc(A, ndim * sizeof(int));

	shape_x[0] = 1024;
	shape_x[1] = 1024;

	shape_y[0] = 1024;
	shape_y[1] = 1024;

	Tensor *x = tensor_create_new(A, ndim, shape_x);
	tensor_randomize_weights(x);
	x->requires_grad = true;
	//printf("tensor x: \n");
	Tensor *y = tensor_create_new(A,ndim,  shape_y);
	tensor_randomize_weights(y);

	clock_t start_time = clock();
	Tensor *z = compare_matmul(A, x, y);
	clock_t end_time = clock();
	double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
	printf("total time in ms: %f\n", time_taken * 1000);
	return 0;
}


