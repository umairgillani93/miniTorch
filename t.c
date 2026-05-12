#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>
#include "arena.h"
#include "config.h"
#include "tensor.h"

static size_t GLOBAL_TENSOR_ID = 0;


//typedef enum {
//	ADD,
//	MUL,
//	MATMUL,
//	RELU,
//	SUB,
//	DIV,
//	EXP,
//	LOG
//} OpType;
//
//typedef struct Op {
//	OpType type;
//	const char *name;
//	void(*backward)(Arena *A, struct Tensor *self);
//} Op;
//
//typedef struct Tensor {
//	int id;
//	int *shape;
//	int *stride;
//	int ndim;
//	float *data;
//	
//	// New parameters
//	Tensor *grad;
//	bool requires_grad;
//	Op *operations; // Added name, and type as well!
//	Tensor **parents;
//	int num_parents;
//} Tensor;


//void tensor_metadata(Tensor *x) {
//	// prints tensor shape
//	printf("shape: \n");
//	tensor_shape_2d(x);
//
//	printf("stride: \n");
//	tensor_stride_2d(x);
//	print("tensor dimension: \n", x->ndim);
//	print("Requires gradient: %d\n", x->requires_grad);
//
//	if (x->requires_grad) {
//		print("Created by: \n", x->operations->name);
//		print("Backward Function: \n", x->operations->type);
//		print("Num Parents: \n", x->num_parents);
//	}
//	
//	else {
//		fprintf(stderr, "Requires grad = false, so no operations to show");
//	}
//
//}


//Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b) {
//	// let's say bot tensors are off same shape
//	assert(a->shape[1] == b->shape[0]);
//	int a_rows = a->shape[0];
//	int a_cols = a->shape[1];
//	int b_rows = b->shape[0];
//	int b_cols = b->shape[1];
//
//	int *out_shape = arena_alloc(A, a->ndim * sizeof(int));
//	out_shape[0] = a_rows;
//	out_shape[1] = b_cols;
//
//	Tensor *out = tensor_create_new(A, a->ndim, out_shape);
//
//	if (a->requires_grad || b->requires_grad) {
//		out->requires_grad = true;
//		// 1. NEED TO SAVE THE PARENTS
//		out->num_parents = 2;
//		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
//		out->parents[0] = a;
//		out->parents[1] = b;
//		out->requires_grad = true;
//		
//		// 2. NEED TO POPULATE THE grad
//		int out_size = a->shape[0] * b->shape[1];
//		out->grad = arena_alloc(A, out_size * sizeof(float));
//		
//		// 3. Need to SAVE THE OPERATIONS for computation graph
//		Op *op = arena_alloc(A, sizeof(Op));
//		op->backward = tensor_matmul_backward;
//		op->type = MATMUL;
//		op->name = "OP_Times";
//		out->operations = op;
//
//	}
//
//	for (int r = 0; r < a_rows; r++) {
//		for (int c = 0; c < b_cols; c++) {
//			float sum = 0.0f;
//			for (int k = 0; k < a_cols; k++) {
//				sum += (a->data[(r * a_cols + k)] *
//					 	b->data[(k * b_cols + c)]);
//			}
//			out->data[r * b_cols + c] = sum;
//		}
//	}
//	return out;
//}
//
//Tensor *tensor_add(Arena *A, Tensor *a, Tensor *b) {
//	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
//	int ndim = a->ndim;
//	int rows = a->shape[0];
//	int cols = a->shape[1];
//	int *out_shape = arena_alloc(A, ndim * sizeof(int));
//	out_shape[0] = rows;
//	out_shape[1] = cols;
//
//	Tensor *out = tensor_create_new(A, ndim, out_shape);
//
//	if (a->requires_grad || b->requires_grad) {
//		out->requires_grad = true;
//
//		// out parents
//		out->num_parents = 2;
//		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
//		out->parents[0] = a;
//		out->parents[1] = b;
//
//		// Operations
//		Op *op = arena_alloc(A, sizeof(Op));
//		op->backward = tensor_add_backward;
//		op->type = ADD; // helps visualizing the computational graph
//		op->name = "OP_ADD"; // helps with logs and monitoring
//		out->operations = op;
//
//		// gradients
//		out->grad = arena_alloc(A, rows * cols * sizeof(float));
//
//	}
//
//	for (int r = 0; r < rows; r++) {
//		for (int c = 0; c < cols; c++) {
//			out->data[r * cols + c] = a->data[r * cols + c] + b->data[r * cols + c];
//		}
//	}
//	return out;
//}

Tensor *tensor_create_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	t->id = GLOBAL_TENSOR_ID++;
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
	// For autograd
	t->data = arena_alloc(A, total * sizeof(float));
	t->parents = NULL;
	t->operations = NULL;
	t->grad = NULL;
	t->num_parents = 0;
	return t;
}


//typedef struct Tensor Tensor;
//Tensor *tensor_create(Arena *A, int ndim, int *shape);
//Tensor *tensor_matmul_forward(Arena *A, Tensor *x, Tensor *y);
//void tensor_matmul_backward(Tensor *self);
//
//typedef struct Op {
//	void (*backward)(struct Tensor *self);
//} Op;
//
//typedef struct Tensor{
//	int ndim;
//	int *shape;
//	int *stride;
//	float *data;
//	float *grad;
//
//	// New parameters
//	bool requires_grad;
//	Op *operations;
//	Tensor **parents;
//	int num_parents;
//} Tensor;
//
//void tensor_matmul_backward(Tensor *self) {
//	// TODO: implement backward later
//}
//
//void tensor_randomize(Tensor *x) {
//	size_t size = tensor_size(x);
//	for (int i = 0; i < size; i++) {
//		x->data[i] = (rand() % 10) + 1.0f;
//	}
//}
//
//Tensor *tensor_matmul_forward(Arena *A, Tensor *a, Tensor *b) {
//	// let's say bot tensors are off same shape
//	assert(a->shape[1] == b->shape[0]);
//	int a_rows = a->shape[0];
//	int a_cols = a->shape[1];
//	int b_rows = b->shape[0];
//	int b_cols = b->shape[1];
//
//	int *out_shape = arena_alloc(A, a->ndim * sizeof(int));
//	out_shape[0] = a_rows;
//	out_shape[1] = b_cols;
//
//	Tensor *out = tensor_create(A, a->ndim, out_shape);
//
//	if (a->requires_grad || b->requires_grad) {
//		out->requires_grad = true;
//		// 1. NEED TO SAVE THE PARENTS
//		out->num_parents = 2;
//		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
//		out->parents[0] = a;
//		out->parents[1] = b;
//		out->requires_grad = true;
//		
//		// 2. NEED TO POPULATE THE grad
//		int out_size = a->shape[0] * b->shape[1];
//		out->grad = arena_alloc(A, out_size * sizeof(float));
//		
//		// 3. Need to SAVE THE OPERATIONS for computation graph
//		Op *op = arena_alloc(A, sizeof(Op));
//		op->backward = tensor_matmul_backward;
//		out->operations = op;
//
//	}
//
//	for (int r = 0; r < a_rows; r++) {
//		for (int c = 0; c < b_cols; c++) {
//			float sum = 0.0f;
//			for (int k = 0; k < a_cols; k++) {
//				sum += (a->data[(r * a_cols + k)] *
//					 	b->data[(k * b_cols + c)]);
//			}
//			out->data[r * b_cols + c] = sum;
//		}
//	}
//	return out;
//}
//
//Tensor *tensor_create(Arena *A, int ndim, int *shape) {
//	// Imagine tensor has float values
//	Tensor *t = arena_alloc(A, sizeof(Tensor)); 
//
//	t->shape = arena_alloc(A, ndim * sizeof(int));
//	t->stride = arena_alloc(A, ndim * sizeof(int));
//	t->ndim = ndim;
//
//	t->stride[ndim - 1] = 1;
//
//	for (int i = 0; i < ndim; i++) {
//		t->shape[i] = shape[i];
//	}
//
//	for (int i = ndim - 2; i >= 0; i--) {
//		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
//	}
//	int numel = t->shape[0] * t->shape[1];
//	t->data = arena_alloc(A, numel * sizeof(float));
//
//	// autograde Node parameters
//	t->requires_grad = false;
//	t->grad = NULL;
//	t->parents = NULL;
//	t->operations= NULL;
//	t->num_parents = 0;
//	return t;
//}
//
//
///*
// * Mean intution:
// * Tensor *tensor_mean(Tensor *a) {
// * 	 int out_dim = 1;
// * 	 int out_shape[2] = {rows, 1};
// * 	 Tensor *out = tensor_create(A, out_dim, out_shape);
// *   int rows = a->shape[0];
// *   int cols = a->shape[1];
// *   int row_mean = 0;
// *   for (int r = 0; r < rows; r++) {
// *     int *row_ptr = a->data + r * cols;
// *     int row_sum = 0;
// *     for (int r = 0; r < cols; r++) {
// *       row_sum += row_ptr[r];
// *     } 
// *     row_mean = row_sum / cols;
// *     out->data[r] = row_mean;
// *   }
// * }
// */
//
////Tensor *tensor_mean(arena *a, tensor *a) {
////	// computes row-wise mean 
////	// out_shape = (rows, 1)
////	int rows = a->shape[0];
////	int cols = a->shape[1];
////	int out_shape[2] = {rows, 1};
////	Tensor *out = arena_alloc(A, rows * sizeof(float));
////
////	for (int r = 0; r < rows; r++) {
////		float *row = a->data + r * cols;
////		float row_sum = 0.0f;
////		float row_mean = 0.0f;
////		for (int c = 0; c < cols; c++) {
////			row_sum += row[c];
////		}
////		row_mean = row_sum / cols;
////		out->data[r * cols] = row_mean;
////	}
////	return out;
////}
////
//
//int tensor_size(Tensor *t) {
//	int rows = t->shape[0];
//	int cols = t->shape[1];
//	int size = rows * cols;
//	return size;
//}


//void *_add_backward(Tensor *x, Tensor *y, Tensor *loss) {
//	// partial derivatives of loss w.r.t parameters
//	// Q = 3 * x ^ 3 - y ^ 2
//	// w.r.t. x -> 9 * x ^ 2
//	// w.r.t. y -> -2 * y
//	Tensor *x_sqrd = tensor_square(x);
//	x->grad = tensor_scaler_multiplication(x_sqrd, (float) 9);
//	Tensor *y_sqrd = tensor_square(y);
//	y->grad = tensor_scaler_multiplication(y_sqrd, (float) -2);
//}


//void tensor_matmul_backward(Tensor *x, Tensor *y,  Tensor *currNode) {
//	// dL/dx = currNode->grad @ y.T using python convention in C :p
//	Tensor *yt = tensor_transpose(y);
//	Tensor *dx = tensor_matmul(currNode-grad, yt);
//	tensor_add_inplace(x->grad, dx);
//
//	// dL/dy = x.T @ currNode->grad 
//	Tensor *xt = tensor_transpose(x);
//	Tensor *dy = tensor_matmul(xt, currNode->grad);
//	tensor_add_inplace(y->grad, dy);
//}

int main() {
	srand(time(NULL));
	Arena *A = malloc(sizeof(Arena));
	size_t SIZE = 1024 * 1024;
	arena_init(A, ARENA_SIZE);
	printf("Arena allocated\n");
	int ndim = 2;
	int *shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = 16;
	shape[1] = 32;

	Tensor *x = tensor_create_new(A, ndim, shape);
	Tensor *y = tensor_create_new(A, ndim, shape);
	Tensor *z = tensor_create_new(A, ndim, shape);

	x->requires_grad = true;
	y->requires_grad = true;

	tensor_randomize(x);
	tensor_randomize(y);
	tensor_randomize(z);

	// Computational graph
	Tensor *a = tensor_add(A, x, y);
	Tensor *zt = tensor_transpose(z);
	Tensor *b = tensor_matmul(A, a, zt);

	tensor_metadata(z);

		
	//Tensor *zt = tensor_transpose(z);
	//Tensor *b = tensor_matmul(A, a, zt);

	// so can we say like
	// if we have x @ y = z
	// and z + f = c 
	// f(c) = L (loss)
	// now dL/dc = grad_c
	// dL/df = dL/dc * dc/df = grad_c * dc/df => dL/df = grad_f
	// dL/dz = dL/dc * dc/dz = grad_c * dc/dz => dL/dz = grad_z
	// dL/dx = dL/dc * dc/dz * dz/dx => grad_z * dz/dx
	// dL/dy = dL/dc * dc/dz * dz/dy => grad_z * dz/dy


	return 0;
}
