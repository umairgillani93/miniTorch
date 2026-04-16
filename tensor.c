#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include "tensor.h"
#include "attention2.h"
#include "feed_forward_nn.h"
#include "arena.h"

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX
#define EMB_DIM 32 
#define SEQ_LEN 10
#define BATCH_SIZE 2
#define EPS 1e-5

void tensor_fill_zeros(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = 0.0f;
	}
}

bool is_exploding(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		float v = x->data[i];
		if (isnan(v) || isinf(v)) {
			return true;
		}
	}
	return false;
}

void tensor_add_inplace(Tensor **a, Tensor **b) {
	assert((*a)->shape != (*b)->shape);
	int rows = (*a)->shape[0];
	int cols = (*a)->shape[1];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * cols + j;
			(*a)->data[idx] = (*b)->data[i];
		}
	}
}



void tensor_check(char *name, Tensor *x) {
	if (is_exploding(x)) {
		printf("NaN/Inf detected in: %s\n", name);
		exit(1);
	}
}


Tensor *tensor_scaler_multiplication(Tensor *x, float val) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i * cols + rows] = val * x->data[i * cols + rows];
	}
	return x;
}

Tensor *tensor_scaler_addition(Tensor *x, float val) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i * cols + rows] = val + x->data[i * cols + rows];
	}
	return x;
}
	

Tensor *tensor_add(Tensor *a, Tensor *b) {
	assert(a->shape != b->shape);
	int ndim = 2;
	int rows = a->shape[0];
	int cols = a->shape[1];
	int shape[2] = {rows, cols};
	Tensor *res = tensor_create(ndim, shape);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * res->shape[1] + j;
			res->data[idx] = a->data[idx] + b->data[idx];
		}
	}
	return res;
}

Tensor *relu_backward(Tensor *da1, Tensor *h1) {
	Tensor *dh1 = tensor_create_weights(h1->ndim, h1->shape);
	int size = tensor_size(h1);

	for (int i = 0; i < size; i++) {
		if (h1->data[i] > 0) {
			dh1->data[i] = da1->data[i];
		}
		else {
			dh1->data[i] = 0.0f;
		}
	}
	return dh1;
}

float loss_value(Tensor *pred, Tensor *target) {
	float squared_err = 0.0f;
	int size = tensor_size(pred);
	
	for (int i = 0; i < size; i++) {
		float diff =  (pred->data[i] - target->data[i]);
		squared_err += (diff * diff);
	}
	return squared_err / size;
}

Tensor *tensor_mse_loss(Tensor *pred, Tensor *target) {
	Tensor *grad = tensor_create_weights(pred->ndim, pred->shape);	
	int size = tensor_size(pred);
	
	for (int i = 0; i < size; i++) {
		grad->data[i] = 2.0f * (pred->data[i] - target->data[i]);
	}
	return grad; 
}

Tensor *tensor_create(int ndim, int *shape) {
	Tensor *t = malloc(sizeof(Tensor));
	if (!t) {
		fprintf(stderr, "something's wrong with memory allocation\n-> aborting..");
		return NULL;
	}
	t->shape = malloc(ndim * sizeof(int));
	t->stride = malloc(ndim * sizeof(int));
	t->ndim = ndim;


	// define the shape of Tensor
	for (int i = 0; i < ndim; i++) {
		t->shape[i] = shape[i];
	}
	// calcuate size of tensor in self-contained fashion
	int size = 1;
	for (int i = 0; i < ndim; i++) {
		size *= shape[i];
	}
	//printf("Size of tensor: %d\n", size);
	//ndim - 1 > is always 1, fastest changing dimension
	// for next ones wer reveser loop and assign
	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
	t->stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; i--) {
		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
	}
	//printf("Stride: %d, %d, %d\n", t->stride[0], t->stride[1], t->stride[2]);
	// define the data now
	t->data = malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		t->data[i] = (rand() % 10) + 1.0f;
		// printf("%f ", t->data[i]);
	}	

	return t;
}

Tensor *tensor_create_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	t->ndim = ndim;
	t->shape = arena_alloc(A, ndim * sizeof(int));
	t->stride = arena_alloc(A, ndim * sizeof(int));

	// define the shape of Tensor
	int total = 1;
	for (int i = ndim - 1; i >= 0; i--) {
		t->shape[i] = shape[i];
		t->stride[i] = total;
		total *= shape[i];
	}
	t->data = arena_alloc(A, total * sizeof(float));
	return t;
}

//Tensor *tensor_create(int ndim, int *shape) {
//	Tensor *t = malloc(sizeof(Tensor));
//	if (!t) {
//		fprintf(stderr, "something's wrong with memory allocation\n-> aborting..");
//		return NULL;
//	}
//	t->shape = malloc(ndim * sizeof(int));
//	t->stride = malloc(ndim * sizeof(int));
//	t->ndim = ndim;
//
//
//	// define the shape of Tensor
//	for (int i = 0; i < ndim; i++) {
//		t->shape[i] = shape[i];
//	}
//	// calcuate size of tensor in self-contained fashion
//	int size = 1;
//	for (int i = 0; i < ndim; i++) {
//		size *= shape[i];
//	}
//	//printf("Size of tensor: %d\n", size);
//	//ndim - 1 > is always 1, fastest changing dimension
//	// for next ones wer reveser loop and assign
//	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
//	t->stride[ndim - 1] = 1;
//	for (int i = ndim - 2; i >= 0; i--) {
//		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
//	}
//	//printf("Stride: %d, %d, %d\n", t->stride[0], t->stride[1], t->stride[2]);
//	// define the data now
//	t->data = malloc(size * sizeof(float));
//	for (int i = 0; i < size; i++) {
//		t->data[i] = (rand() % 10) + 1.0f;
//		// printf("%f ", t->data[i]);
//	}	
//
//	return t;
//}

Tensor *tensor_create_weights(int ndim, int *shape) {
	Tensor *t = malloc(sizeof(Tensor));
	if (!t) {
		fprintf(stderr, "something's wrong with memory allocation\n-> aborting..");
		return NULL;
	}
	t->shape = malloc(ndim * sizeof(int));
	t->stride = malloc(ndim * sizeof(int));
	t->ndim = ndim;


	// define the shape of Tensor
	for (int i = 0; i < ndim; i++) {
		t->shape[i] = shape[i];
	}
	// calcuate size of tensor in self-contained fashion
	int size = 1;
	for (int i = 0; i < ndim; i++) {
		size *= shape[i];
	}
	//printf("Size of tensor: %d\n", size);
	//ndim - 1 > is always 1, fastest changing dimension
	// for next ones wer reveser loop and assign
	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
	t->stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; i--) {
		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
	}
	//printf("Stride: %d, %d, %d\n", t->stride[0], t->stride[1], t->stride[2]);
	// define the data now
	t->data = malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		t->data[i] = RAND_FLOAT;
		// printf("%f ", t->data[i]);
	}	

	return t;
}


Tensor *tensor_create_weights_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	t->shape = arena_alloc(A, ndim * sizeof(int));
	t->stride = arena_alloc(A, ndim * sizeof(int));
	t->ndim = ndim;

	int total = 1;
	for (int i = ndim - 1; i >= 0; i--) {
		t->shape[i] = shape[i];
		t->stride[i] = total;
		total *= shape[i];
	}

	t->data = arena_alloc(A, total * sizeof(float));

	return t;
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
	int rows_a = a->shape[0];
	int cols_a = a->shape[1];

	int rows_b = b->shape[0];
	int cols_b = b->shape[1];

	// resultant tensor having shape (rows_a, cols_b);
	int ndim_r = 2;
	int *shape_r = malloc(2 * sizeof(int));
	shape_r[0] = a->shape[0];
	shape_r[1] = b->shape[1];

	Tensor *r = tensor_create(ndim_r, shape_r);
	//printf("Created resultant tensor\n");

	for (int i = 0; i < rows_a; i++) {
		for (int j = 0; j < cols_b; j++) {
			float sum = 0.0f;
			for (int k = 0; k < cols_a; k++) {
				sum += (a->data[i * a->stride[0] + k * a->stride[1]]  * b->data[k * b->stride[0] + j * b->stride[1]]);
			}
			r->data[i * r->stride[0] + j * r->stride[1]] = sum;
		}
	}
	return r;
}

Tensor *tensor_softmax(Tensor *t) {
	Tensor *r = malloc(sizeof(Tensor));
	if (!r) {return NULL;}
	r->shape = t->shape;
	r->stride = t->stride;
	r->ndim = t->ndim;
	r->data = malloc(r->shape[0] * r->shape[1] * sizeof(float));

	int rows = t->shape[0];
	int cols = t->shape[1];

	for (int i = 0; i < rows; i++) {
		float max = -INFINITY;
		for (int j = 0; j < cols; j++) {
			if (t->data[i * cols + j] > max) {
				max = t->data[i * cols + j];
			}
		}

		float sum = 0.0f;
		for (int j = 0; j < cols; j++) {
			sum += expf(t->data[i * cols + j] - max);
		}

		// now find division
		for (int k = 0; k < cols; k++) {
			r->data[i * cols + k] = expf(t->data[i * cols + k] - max) / sum;
		}
	}
	return r;
}

void tensor_free(Tensor *t) {
	if (!t) return;
	free(t->data);
	free(t->stride);
	free(t->shape);
	free(t);
	//printf("Freed successfully!\n");
}

void tensor_get(Tensor *t) {
	if (!t) return;
	int size = 1;
	for (int i = 0; i < t->ndim; i++) {
		size *= t->shape[i];
	}
	//printf("size: %d\n", size);
	for (int i = 0; i < size; i++) {
		printf("%0.2f ", t->data[i]);
	}		
}

Tensor *tensor_transpose(Tensor *a) {
	int ndim = 2;
	int *shape = malloc(ndim * sizeof(int));
	shape[0] = a->shape[1];
	shape[1] = a->shape[0];
	Tensor *t = tensor_create(ndim, shape);
	
	int rows_a = a->shape[0];
	int cols_a = a->shape[1];
	for (int i = 0; i < rows_a; i++) {
		for (int j = 0; j < cols_a; j++) {
			t->data[j * t->stride[0] + i * t->stride[1]] = a->data[i * a->stride[0] + j * a->stride[1]];
		}
	}

	return t;
}

int tensor_size(Tensor *t) {
	int size = 1;
	for (int i = 0; i < t->ndim; i++) {
		size *= t->shape[i];
	}
	//printf("Tensor size: %d\n", size);
	return size;
}	

void tensor_shape(Tensor *t) {
	printf("(%d, %d)\n", t->shape[0], t->shape[1]);
}

//int main() {
//	Arena *A = malloc(sizeof(Arena));
//	int SIZE = 1024 * 1024 * 1024;
//	arena_init(A, SIZE);
//	int ndim = 2;
//	int shape[2] = {SEQ_LEN, EMB_DIM};
//	Tensor *x = tensor_create_new(A, ndim, shape);
//	int size = tensor_size(x);
//	for (int i = 0; i < size; i++) {
//		printf("%f\n", x->data[i]);
//	}
//}
