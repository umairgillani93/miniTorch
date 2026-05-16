#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>
#include "arena.h"
#include "config.h"
#include "tensor.h"

int main() {
	int ndim = 2;
	int *shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = 16;
	shape[1] = 32;
	Tensor *x = tensor_create_new(A, ndim, shape);
	tensor_randomize(x);

	tensor_shape_2d(x);
	
}
