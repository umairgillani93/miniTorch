#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "arena.h"
#include "config.h"

typedef struct Tensor Tensor;
Tensor *tensor_create(Arena *A, int ndim, int *shape);
Tensor *tensor_matmul(Arena *A, Tensor *x, Tensor *y);

typedef struct Op{
	void (*backward)(struct Tensor *self);
} Op;

typedef struct Tensor{
	int ndim;
	int *shape;
	int *stride;
	float *data;
	float *grad;

	// New parameters
	bool requires_grad;
	Op *operations;
	Tensor **parents;
	int num_parents;
} Tensor;


Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b) {
	// let's say bot tensors are off same shape
	assert(a->shape[1] == b->shape[0]);
	int a_rows = a->shape[0];
	int a_cols = a->shape[1];
	int b_rows = b->shape[0];
	int b_cols = b->shape[1];

	int *out_shape = arena_alloc(A, a->ndim * sizeof(int));
	out_shape[0] = a_rows;
	out_shape[1] = b_cols;

	Tensor *out = tensor_create(A, a->ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		// 1. NEED TO SAVE THE PARENTS
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		out->parents[1] = b;
		out->requires_grad = true;
		
		// 2. NEED TO POPULATE THE grad
		out->grad = NULL;
		
		// 3. Need to SAVE THE OPERATIONS for computation graph
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = matmul;
		out->operations = op;

	}

	for (int r = 0; r < a_rows; r++) {
		for (int c = 0; c < b_cols; c++) {
			float sum = 0.0f;
			for (int k = 0; k < a_cols; k++) {
				sum += (a->data[(r * a_cols + k)] *
					 	b->data[(k * b_cols + c)]);
			}
			out->data[r * b_cols + c] = sum;
		}
	}
	return out;
}

Tensor *tensor_create(Arena *A, int ndim, int *shape) {
	// Imagine tensor has float values
	Tensor *t = arena_alloc(A, sizeof(Tensor)); 

	t->shape = arena_alloc(A, ndim * sizeof(int));
	t->stride = arena_alloc(A, ndim * sizeof(int));
	t->ndim = ndim;

	t->stride[ndim - 1] = 1;

	for (int i = 0; i < ndim; i++) {
		t->shape[i] = shape[i];
	}

	for (int i = ndim - 2; i >= 0; i--) {
		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
	}
	int numel = t->shape[0] * t->shape[1];
	t->data = arena_alloc(A, numel * sizeof(float));

	// autograde Node parameters
	t->requires_grad = false;
	t->grad = NULL;
	t->parents = NULL;
	t->operations= NULL;
	t->num_parents = 0;
	return t;
}

int main() {
	Arena *A = malloc(sizeof(Arena));
	size_t SIZE = 1024 * 1024;
	arena_init(A, ARENA_SIZE);
	printf("Arena allocated\n");
	int ndim = 2;
	int *shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = 16;
	shape[1] = 32;
	Tensor *x = tensor_create(A, ndim, shape);
	Tensor *y = tensor_create(A, ndim, shape);
	Tensor *z = tensor_matmul(A, x, y);
		
	printf("%d\n", z->requires_grad);

	int rows = 16;
	int cols = 32;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			printf("%f ", z->data[r * cols + c]);
		}
		printf("\n");
	}

	return 0;
}

