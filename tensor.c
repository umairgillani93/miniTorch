#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include "tensor.h"
#include "attention2.h"
#include "feed_forward_nn.h"
#include "layer_norm.h"
#include "arena.h"
#include "config.h"

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX
#define EPS 1e-5

static size_t GLOBAL_TENSOR_ID = 0;


// Backpropagation Intuition:
 	// so can we say like
	// if we have x @ y = z
	// and z + f = c
	// f(c) = L (loss)
	// now dL/dc = grad_c
	// dL/df = dL/dc * dc/df = grad_c * dc/df => dL/df = grad_f
	// dL/dz = dL/dc * dc/dz = grad_c * dc/dz => dL/dz = grad_z
	// dL/dx = dL/dc * dc/dz * dz/dx => grad_z * dz/dx
	// dL/dy = dL/dc * dc/dz * dz/dy => grad_z * dz/dy
	//
	//


void dfs(Tensor *root, bool *visited) {
	if (visited[root->id]) {
		return;
	}
	else {
		tensor_metadata(root);
		if (root->num_parents) {
			int p = root->num_parents;
			for (int i = 0; i < p; i++) {
				dfs(root->parents[i], visited);
			}
		}
	}
}

void tensor_metadata(Tensor *x) {
	// prints tensor shape
	printf("Tensor Id: %d\n", x->id);
	printf("shape: \n");
	tensor_shape_2d(x);

	printf("stride: \n");
	tensor_stride_2d(x);
	printf("tensor dimension: %d\n", x->ndim);
	printf("Requires gradient: %d\n", x->requires_grad);

	if (x->requires_grad) {
		printf("Created by: %s\n", x->operations->name);
		printf("Backward Function: %d\n", x->operations->type);
		printf("Back Pointer: %p\n", *x->operations->backward);
		printf("Num Parents: %d\n", x->num_parents);
	}
	
	else {
		fprintf(stderr, "<tensor_metadata> Error: Requires grad = false, so no operations to show\n");
	}

}
	
void tensor_add_backward(Arena *A, Tensor *currNode) {
	Tensor *x = currNode->parents[0];
	Tensor *y = currNode->parents[1];

	// by adding up the Tensors, whatever change happens in 
	// the Tensor will have linear impact on the currNode;
	// i.e if we raise Tensor (a) or Tensor (b) by samll about
	// that same change will be reflacted in currNode
	// dz/da = 1;
	// dz/db = 1;

	int x_ndim = x->ndim;
	int y_ndim = y->ndim;

	x->grad = tensor_create_new(A, x_ndim, x->shape);
	y->grad = tensor_create_new(A, y_ndim, y->shape);
	tensor_randomize_weights(x->grad);
	tensor_randomize_weights(y->grad);

	// For addition operads derivate, we accumulate the 
	// derivate of whatever comes like currNode -> grad

	int rows = currNode->shape[0];
	int cols = currNode->shape[1];
	int ndim = currNode->ndim;


	assert((x->grad->shape[0] == currNode->grad->shape[0]) && (x->grad->shape[1] == currNode->grad->shape[1]));
	assert((y->grad->shape[0] == currNode->grad->shape[0]) && (y->grad->shape[1] == currNode->grad->shape[1]));

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			x->grad->data[r * cols + c] += currNode->grad->data[r * cols + c];
			y->grad->data[r * cols + c] += currNode->grad->data[r * cols + c];
		}
	}

	printf("x grad shape: \n");
	tensor_shape_2d(x->grad);

	printf("y grad shape: \n");
	tensor_shape_2d(y->grad);
}


void tensor_matmul_backward(Arena *A, Tensor *currNode) {
	Tensor *x = currNode->parents[0];
	Tensor *y = currNode->parents[1];

	// dL/dx = grad_prev @ y.T
	Tensor *yt = tensor_transpose(A, y);

	Tensor *dx = tensor_matmul(A, y, currNode->grad);
	//Tensor *dxt = tensor_transpose(dx);
	Tensor *dxt = tensor_transpose(A, dx);

	// Accumulate x now and first initilize 'x->grad'
	x->grad = tensor_create_new(A, x->ndim, x->shape); 
	tensor_randomize(x->grad);

	size_t size_x = x->grad->shape[0] * x->grad->shape[1];
	for (int i = 0; i < size_x; i++) {
		x->grad->data[i] += dxt->data[i];
	}

	// dL/dy = x.T @ grad_prev 
	Tensor *xt = tensor_transpose(A, x);
	Tensor *dy = tensor_matmul(A, currNode->grad, x);

	printf("dy shape: \n");
	tensor_shape_2d(dy);

	// Accumulate y now and first initilize 'y->grad'
	y->grad = tensor_create_new(A, y->ndim, y->shape); 
	tensor_randomize(y->grad);
	size_t size_y = y->grad->shape[0] * y->grad->shape[1];
	for (int i = 0; i < size_y; i++) {
		y->grad->data[i] += dy->data[i];
	}
	printf("y shape: \n");
	tensor_shape_2d(y);

	printf("y->grad shape: \n");
	tensor_shape_2d(y->grad);
}


Tensor *tensor_relu(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;
	
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_relu_backward;
		out->operations = op;
		out->grad = arena_alloc(A, rows * cols * sizeof(float));
	}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if (x->data[r * cols + c] > 0) {
				out->data[r * cols + c] = x->data[r * cols + c];
			}
			else {
				out->data[r * cols + c] = 0.0f;
			}
		}
	}
	return out;
}



Tensor *tensor_fill_val(Arena *A, Tensor *x, int v) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;
	
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	// No computational graph for this
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = v;
		}
	}
	return out;
}

void tensor_fill_ones(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = 1.0f;
	}
}

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

float max_element(float *row, int cols) {
	float mx = row[0];
	for (int i = 1; i < cols; i++) {
		if (row[i] > row[i - 1]) {
			mx = row[i];
		}
	}
	return mx;
}

Tensor *tensor_row_max(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1]; // row sum shrinks cols dimention
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	// Create output tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_softmax_backward;
		out->operations = op;
		out->grad = arena_alloc(A, rows * cols * sizeof(float));
	}

	for (int r = 0; r < rows; r++) {
		float *row = x->data + r * cols;
		float MX = max_element(row, cols);
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = MX;
		}
	}
	return out;
}

Tensor *tensor_exp(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1]; // row sum shrinks cols dimention
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	// Create output tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	//if (x->requires_grad) {
	//	out->requires_grad = true;
	//	out->num_parents = 1;
	//	out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
	//	out->parents[0] = x;
	//	Op *op = arena_alloc(A, sizeof(Op));
	//	op->backward = tensor_softmax_backward;
	//	out->operations = op;
	//	out->grad = arena_alloc(A, rows * cols * sizeof(float));
	//}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = expf(x->data[r * cols + c]);
		}
	}
	return out;
}

Tensor *tensor_row_sum(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1]; // row sum shrinks cols dimention
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, rows * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = 1;

	// Create output tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_softmax_backward;
		out->operations = op;
		out->grad = arena_alloc(A, rows * sizeof(float));
	}

	for (int r = 0; r < rows; r++) {
		float sum = 0.0f;
		for (int c = 0; c < cols; c++) {
			sum += x->data[r * cols + c];
		}
		out->data[r] = sum;
	}
	return out;
}

Tensor *tensor_fill_like(Arena *A, Tensor *x, double eps) {
	// No grad = true for this
	// Hence this is not the part of Computational graph
	printf("rows: %d\n", x->shape[0]);
	printf("cols : %d\n", x->shape[1]);
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = x->shape[0]; // rows of out
	out_shape[1] = x->shape[1]; // cols of out
															//
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = x->data[r *cols + c] + eps;
		}
	}
	return out;
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

void tensor_accumulate(Tensor *x, Tensor *grad) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			x->grad->data[r * cols + c] += grad->data[r * cols + c];
		}
	}
}
		

//void tensor_add_inplace(Tensor **a, Tensor **b) {
//	assert((*a)->shape != (*b)->shape);
//	int rows = (*a)->shape[0];
//	int cols = (*a)->shape[1];
//	for (int i = 0; i < rows; i++) {
//		for (int j = 0; j < cols; j++) {
//			int idx = i * cols + j;
//			(*a)->data[idx] = (*b)->data[i];
//		}
//	}
//}



void tensor_check(char *name, Tensor *x) {
	if (is_exploding(x)) {
		printf("NaN/Inf detected in: %s\n", name);
		exit(1);
	}
}


// for rows expansion
Tensor *tensor_expand_rows(Arena *A, Tensor *a, int out_rows) {
	assert(a->shape[0] = 1); // has only single row
	int cols = a->shape[1];
	int ndim = a->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = out_rows;
	out_shape[1] = cols;
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	// computing logic
	for (int r = 0; r < out_rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[c];
		}
	}
	return out;
}

Tensor *tensor_scalling(Arena *A, Tensor *a, Tensor *b) {
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

Tensor *tensor_scaler_div(Arena *A, Tensor *x, float val) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = x->data[r * cols + c] / val;
		}
	}
	return out;
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

Tensor *tensor_scaler_addition(Arena *A, Tensor *x, float val) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = x->data[r * cols + c] + val;
		}
	}
	return out;
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
	out->id = GLOBAL_TENSOR_ID++;

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;

		// out parents
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		out->parents[1] = b;

		// Operations
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_add_backward;
		op->type = ADD; // helps visualizing the computational graph
		op->name = "OP_ADD"; // helps with logs and monitoring
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

//Tensor *relu_backward(Tensor *da1, Tensor *h1) {
//	Tensor *dh1 = tensor_create_weights(h1->ndim, h1->shape);
//	int size = tensor_size(h1);
//
//	for (int i = 0; i < size; i++) {
//		if (h1->data[i] > 0) {
//			dh1->data[i] = da1->data[i];
//		}
//		else {
//			dh1->data[i] = 0.0f;
//		}
//	}
//	return dh1;
//}

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

	Tensor *out = tensor_create_new(A, a->ndim, out_shape);
	out->id = GLOBAL_TENSOR_ID++;

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
		op->type = MATMUL;
		op->name = "OP_Times";
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

Tensor *tensor_softmax(Arena *A, Tensor *x) {
	/*
	 * Tensor *max = tensor_row_max(A, Tensor *x)
	 * Tensor *max_exp = tensor_expand_cols(A, Tensor *x)
	 * Tensor *shifted = tensor_sub(A, max_exp, num_cols);
	 */
		int rows = x->shape[0];
    int cols = x->shape[1];

    Tensor *row_max = tensor_row_max(A, x);
    //Tensor *row_max_expanded = tensor_expand_cols(A, row_max, cols);
    Tensor *shifted = tensor_subtract(A, x, row_max);
    Tensor *exp = tensor_exp(A, shifted);

    Tensor *row_sum = tensor_row_sum(A, exp);

    Tensor *row_sum_expanded = tensor_expand_cols(A, row_sum, cols);
    Tensor *out = tensor_div(A, exp, row_sum_expanded);

    return out;
}

	//int rows = x->shape[0];
	//int cols = x->shape[1];
	//int ndim = x->ndim;
	//int *out_shape = arena_alloc(A, ndim * sizeof(int));
	//out_shape[0] = rows;
	//out_shape[1] = cols;

	//Tensor *out = tensor_create_new(A, ndim, out_shape);

	//if (x->requires_grad) {
	//	out->requires_grad = true;
	//	out->num_parents = 1;
	//	out->parents = arena_alloc(A, sizeof(Tensor *));
	//	out->parents[0] = x;
	//	Op *op = arena_alloc(A, sizeof(Op));
	//	op->backward = tensor_softmax_backward;
	//	out->operations = op;
	//	out->grad = arena_alloc(A, rows * cols * sizeof(float));
	//}

	//// softmax(x) = e(x[i])/sum(row); // for each row
	//
	//Tensor *exp = tensor_exp(A, x);
	//Tensor *row_sum = tensor_row_sum(A, exp);
	//Tensor *row_sum_exp = tensor_expand_cols(A, row_sum, cols);
	//out = tensor_div(A, exp, row_sum_exp);

	//return out;
//}

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

Tensor *tensor_transpose(Arena *A, Tensor *a) {
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = 2;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = cols;
	out_shape[1] = rows;
	Tensor *t = tensor_create_new(A, ndim, out_shape);
	t->id = GLOBAL_TENSOR_ID++;

	if (a->requires_grad) {
		t->requires_grad = true;
		t->num_parents = 1;
		t->parents = arena_alloc(A, t->num_parents * sizeof(Tensor *));
		t->parents[0] = a;
		t->grad = arena_alloc(A, rows * cols * sizeof(float));
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_transpose_backward;
		op->type = TRANSPOSE;
		op->name = "OP_Transpose";
		t->operations = op;
	}
	
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

void tensor_stride_2d(Tensor *t) {
	printf("(%d, %d)\n", t->stride[0], t->stride[1]);
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

//void tensor_matmul_backward(Arena *A, Tensor *x) {
//	// Will be implemented later. IA
//}

//void tensor_add_backward(tensor *x) {
//	// will be implemented later. ia
//}

void tensor_scalling_backward(Tensor *x) {
	// Will be implemented later. IA
}

void tensor_softmax_backward(Tensor *x) {
	// Will be implemented later. IA
}

void tensor_relu_backward(Tensor *x) {
	// Will be implemented later. IA
}

void tensor_slice_cols_backward(Tensor *x) {
	// Will be implemented later. IA
}

void tensor_transpose_backward(Tensor *x) {
	// Will be implemented later. IA
}

bool tensor_equal(Tensor *x, Tensor *y) {
	int rows = x->shape[0];
	int cols = x->shape[1];

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if (x->data[r * cols + c] != y->data[r * cols + c]) {
				return false;
			}
		}
	}
	return true;
}

Tensor *tensor_concat(Arena *A, Tensor **heads, int num_heads) {
	int rows = heads[0]->shape[0];
	int cols = heads[0]->shape[1];
	int ndim = heads[0]->ndim;
	int out_cols = cols * num_heads;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = out_cols; // 8 * 4 = 32;
	
	// create out tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	// core logic
	for (int k = 0; k < num_heads; k++) {
		Tensor *head = heads[k]; // head chunk
	
		for (int r= 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out->data[r * out_cols + k * cols + c] =
				 	head->data[r * cols + c];
			}
		}
	}
	return out;
}


Tensor *tensor_slice_cols(Arena *A, Tensor *x, int k, int dk) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = dk;
	// If actual tensor 'x' has shape (16, 32)
	// out tensor will be sliced version of it 
	// and will have shape (16, dk);
	
	// create output tensor now
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	// build computational graph
	if (x->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 1;
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_slice_cols_backward;
		out->grad = arena_alloc(A, rows * dk * sizeof(float));
	}

	// core logic here
	// slices the main tensor 'x' and populates tensor 'out'
	// [[1,2,3,4,5,6,7,8],
	//	  [1,2,3,4,5,6,7,8],
	//		 [1,2,3,4,5,6,7,8],
	//	  [1,2,3,4,5,6,7,8]]
	
	// using pointer arithematic
	//float *start = x->data + k * rows * dk;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < dk; c++) {
			out->data[r * dk + c] = x->data[r * cols + k * dk + c];
		}
	}
	return out;
}

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

	tensor_randomize(x);
	tensor_randomize(y);
	tensor_randomize(z);

	x->requires_grad = true;
	y->requires_grad = true;
	z->requires_grad = true;

	Tensor *zt = tensor_transpose(A, z);

	// Computational graph
	Tensor *a = tensor_add(A, x, y);
	Tensor *b = tensor_matmul(A, a, zt);

	int S = 10;
	bool visited[S];

	for (int i = 0; i < S; i++) {
		visited[i] = false;
	}

	dfs(b, visited);

	//printf("z data: \n");
	//tensor_metadata(z);

	/*
	 * Tensor as nodes
	 * Tensor operations as Nodes
	 * X(Node)----|some opeartion| (NOde)----> Y(Node)
	 * shape, stride, id, operations, name, backward function...
	 */
	
	

	return 0;

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
}

