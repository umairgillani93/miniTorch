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
#include "config.h"

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX
#define EPS 1e-5


void tensor_fill_zeros(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = 0.0f;
	}
}

void clip_gradient(Tensor *x) {
    int size = tensor_size(x);
    float threshold = 1.0f;
    float MX = 0.0f;
    bool has_bad = false;

    // Pass 1 — detect NaN/Inf and find max abs gradient
    for (int i = 0; i < size; i++) {
        float g = x->data[i];

        if (!isfinite(g)) {
            has_bad = true;
            break;
        }

        float v = fabsf(g);
        if (v > MX) MX = v;
    }

    // If NaN/Inf found → zero gradients and STOP
    if (has_bad) {
        for (int i = 0; i < size; i++)
            x->data[i] = 0.0f;
        return;
    }

    // Pass 2 — clip if too large
    if (MX > threshold) {
        float scale = threshold / MX;   // compute once
        for (int i = 0; i < size; i++)
            x->data[i] *= scale;
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

void tensor_randomize_weights(Tensor *x) {
	size_t size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = RAND_FLOAT;
	}
}

void tensor_randomize(Tensor *x) {
	size_t size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = (rand() % 10) + 1.0f;
	}
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
	

Tensor *tensor_add(Arena *A, Tensor *a, Tensor *b) {
	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
	int ndim = a->ndim;
	int rows = a->shape[0];
	int cols = a->shape[1];
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;

		// out parents
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));

		// Operations
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_add_backward;
		out->operations = op;

		// gradients
		out->grad = arena_alloc(A, rows * cols * sizeof(float));

	}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] + b->data[r * cols + c];
		}
	}
	return out;
}

Tensor *tensor_subtract(Arena *A, Tensor *a, Tensor *b) {
	assert((a->shape[0] == b->shape[0]) && (a->shape[1] == b->shape[1]));
	int ndim = a->ndim;
	int rows = a->shape[0];
	int cols = a->shape[1];
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;

		// out parents
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));

		// Operations
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_add_backward;
		out->operations = op;

		// gradients
		out->grad = arena_alloc(A, rows * cols * sizeof(float));

	}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] - b->data[r * cols + c];
		}
	}
	return out;
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

Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target) {
	Tensor *grad = tensor_create_weights_new(A, pred->ndim, pred->shape);	
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
	// For autograd
	t->data = arena_alloc(A, total * sizeof(float));
	t->parents = NULL;
	t->operations = NULL;
	t->grad = NULL;
	t->num_parents = 0;
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
	// For autograd
	t->data = arena_alloc(A, total * sizeof(float));
	t->parents = NULL;
	t->operations = NULL;
	t->grad = NULL;
	t->num_parents = 0;

	return t;
}

Tensor *tensor_matmul_forward(Arena *A, Tensor *a, Tensor *b) {
	// let's say bot tensors are off same shape
	assert(a->shape[1] == b->shape[0]);
	int a_rows = a->shape[0];
	int a_cols = a->shape[1];
	int b_rows = b->shape[0];
	int b_cols = b->shape[1];

	int *out_shape = arena_alloc(A, a->ndim * sizeof(int));
	out_shape[0] = a_rows;
	out_shape[1] = b_cols;

	Tensor *out = tensor_create_new(A, a->ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		// 1. NEED TO SAVE THE PARENTS
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		out->parents[1] = b;
		out->requires_grad = true;
		
		// 2. NEED TO POPULATE THE grad
		int out_size = a->shape[0] * b->shape[1];
		out->grad = arena_alloc(A, out_size * sizeof(float));
		
		// 3. Need to SAVE THE OPERATIONS for computation graph
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_matmul_backward;
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

//Tensor *tensor_matmul_forward(Arena *A, Tensor *a, Tensor *b) {
//	int rows_a = a->shape[0];
//	int cols_a = a->shape[1];
//
//	int rows_b = b->shape[0];
//	int cols_b = b->shape[1];
//
//	// resultant tensor having shape (rows_a, cols_b);
//	int ndim_r = 2;
//	int *shape_r = arena_alloc(A, ndim_r * sizeof(int));
//	shape_r[0] = a->shape[0];
//	shape_r[1] = b->shape[1];
//
//	Tensor *r = tensor_create_new(A, ndim_r, shape_r);
//	//printf("Created resultant tensor\n");
//
//	for (int i = 0; i < rows_a; i++) {
//		for (int j = 0; j < cols_b; j++) {
//			float sum = 0.0f;
//			for (int k = 0; k < cols_a; k++) {
//				sum += (a->data[i * a->stride[0] + k * a->stride[1]]  * b->data[k * b->stride[0] + j * b->stride[1]]);
//			}
//			r->data[i * r->stride[0] + j * r->stride[1]] = sum;
//		}
//	}
//	return r;
//}

Tensor *tensor_softmax_forward(Arena *A, Tensor *t) {
	Tensor *r = arena_alloc(A, sizeof(Tensor));
	if (!r) {return NULL;}
	r->shape = t->shape;
	r->stride = t->stride;
	r->ndim = t->ndim;
	r->data = arena_alloc(A, r->shape[0] * r->shape[1] * sizeof(float));

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

//void tensor_free(Tensor *t) {
//	if (!t) return;
//	free(t->data);
//	free(t->stride);
//	free(t->shape);
//	free(t);
//	//printf("Freed successfully!\n");
//}

void tensor_get_2d(Tensor *t) {
	if (!t) return;
	int rows = t->shape[0];
	int cols = t->shape[1];
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			printf("%0.2f ", t->data[r * cols + c]);
		}
		printf("\n");
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

void tensor_shape_2d(Tensor *t) {
	printf("(%d, %d)\n", t->shape[0], t->shape[1]);
}

// Auto-grad methods
Tensor *tensor_mean(Arena *A, Tensor *a) {
	// computes row-wise mean 
	// out_shape = (rows, 1)
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = 1;
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad) {
		// Build computation graph
		// define requires_grad = true for out;
		out->requires_grad = true;

		// define number of parents
		out->num_parents = 1;
		out->parents = arena_alloc(A, sizeof(Tensor *));;
		out->parents[0] = a;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_mean_backward;
		out->operations = op;

		// define gradients matrix
		out->grad = arena_alloc(A, rows * 1 * sizeof(float));
	}

	for (int r = 0; r < rows; r++) {
		float *row = a->data + r * cols;
		float row_sum = 0.0f;
		float row_mean = 0.0f;
		for (int c = 0; c < cols; c++) {
			row_sum += row[c];
		}
		row_mean = row_sum / cols;
		out->data[r] = row_mean; // cols = 1, c = 0 so r * 1 + 0 = r
	}
	return out;
}

Tensor *tensor_expand_cols(Arena *A, Tensor *m, int out_cols) {
	// Takes the mean value for each row 
	// and expands it column number of times to match the other tensor
	// reduce_mean = [m1],
	//               [m2],
	//               [m3]
	// expand_mean = [m1, m1, m1]
	//               [m2, m2, m2]
	//               [m3, m3, m3]
	
	assert(m->ndim == 2);
	assert(m->shape[1] == 1);
	int rows = m->shape[0];
	int ndim = 2;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = out_cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (m->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 1;
		out->parents = arena_alloc(A, sizeof(Tensor *));
		out->parents[0] = m;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_expand_cols_backward;
		out->operations = op;
		out->grad = arena_alloc(A, rows * out_cols * sizeof(float));
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		float v = m->data[r];
		for (int c = 0; c < out_cols; c++) {
			out->data[r * out_cols + c] = v;
		}
	}
	return out;
}

Tensor *tensor_square(Arena *A, Tensor *a, Tensor *b) {
	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		out->parents[1] = b;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_square_backward;
		out->operations = op;
		out->grad = arena_alloc(A, rows * cols * sizeof(float));
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] * b->data[r * cols + c];
		}
	}
	return out;

}

Tensor *tensor_sqrt(Arena *A, Tensor *a) {
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = 2;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_sqrt_backward;
		out->operations = op;
		out->grad = arena_alloc(A, rows * cols * sizeof(float));
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = sqrt(a->data[r * cols + c] + EPS); // Need to CORRECT. Apply another tensor_add() here.
		}
	}
	return out;
}

Tensor *tensor_div(Arena *A, Tensor *a, Tensor *b) {
	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		out->parents[1] = b;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_square_backward;
		out->operations = op;
		out->grad = arena_alloc(A, rows * cols * sizeof(float));
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] / b->data[r * cols + c];
		}
	}
	return out;

}

void tensor_sqrt_backward(Tensor *x) {
	// Will be implemented later IA
}

void tensor_square_backward(Tensor *x) {
	// Will be implemented later IA
}

void tensor_expand_cols_backward(Tensor *x) {
}

void tensor_mean_backward(Tensor *x) {
	// will implement later. IA
}

void tensor_matmul_backward(Tensor *x) {
	// Will be implemented later. IA
}

void tensor_add_backward(Tensor *x) {
	// Will be implemented later. IA
}


int main() {
	Arena *A = malloc(sizeof(Arena));
	int SIZE = 1024 * 1024 * 1024;
	arena_init(A, SIZE);
	int ndim = 2;
	int shape[2] = {SEQ_LEN, EMB_DIM};
	Tensor *x = tensor_create_new(A, ndim, shape);
	Tensor *y = tensor_create_new(A, ndim, shape);

	tensor_randomize(x);
	tensor_randomize(y);

	Tensor *mean = tensor_mean(A, x);
	Tensor *mean_exp = tensor_expand_cols(A, mean, x->shape[1]);
	Tensor *diff = tensor_subtract(A, x, mean_exp);
	Tensor *sq = tensor_square(A, diff, diff);
	Tensor *var = tensor_mean(A, sq);
	Tensor *var_exp = tensor_expand_cols(A, var, x->shape[1]);
	//Tensor *eps = tensor_fill_like(A, var_exp, 1e-5);
	//Tensor *var_eps = tensor_add(A, var_exp, eps);
	Tensor *std = tensor_sqrt(A, var_exp);
	Tensor *out = tensor_div(A, diff, std);

	tensor_shape_2d(out);
	tensor_get_2d(out);

	return 0;
}
