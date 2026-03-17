#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "tensor.h"

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX
#define EMB_DIM 32 
#define SEQ_LEN 10
#define BATCH_SIZE 2
#define EPS 1e-5

Tensor *tensor_fill_zeros(Tensor **x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x[i] = 0.0f;
	}
}

void sgd_optimizer(Tensor **w, Tensor **dw), {
	// simple SGD Soptmizer 
	// w = w - lr * dw;
	assert(shape(w) == shape(dw));
	int size = tensor_size(w);
	for (int i = 0; i < size; i++) {
		w[i] = w[i] - lr * dw[i];
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
	printf("Tensor size: %d\n", size);
	return size;
}	

void tensor_shape(Tensor *t) {
	printf("(%d, %d)\n", t->shape[0], t->shape[1]);
}

//int main() {
//	int seed = 32;
//	srand(seed);
//
//	int ndim = 2;
//	//Tensor *tokens = malloc(sizeof(Tensor));
//	//Tensor *q_weights = malloc(sizeof(Tensor));
//	//Tensor *k_weights = malloc(sizeof(Tensor));
//	//Tensor *v_weights = malloc(sizeof(Tensor));
//
//	int *shape_tokens = malloc(ndim * sizeof(int));
//	int *shape_q_weights = malloc(ndim * sizeof(float));
//	int *shape_k_weights = malloc(ndim * sizeof(float));
//	int *shape_v_weights = malloc(ndim * sizeof(float));
//
//	// define shape_tokens
//
//	shape_tokens[0] = SEQ_LEN;
//	shape_tokens[1] = EMB_DIM; // this is for token embeddings
//	
//	// consider this for a single token query weights
//	shape_q_weights[0] = EMB_DIM;
//	shape_q_weights[1] = EMB_DIM;
//
//	// key weights
//	shape_k_weights[0] = EMB_DIM;
//	shape_k_weights[1] = EMB_DIM;
//
//	// value weights
//	shape_v_weights[0] = EMB_DIM;
//	shape_v_weights[1] = EMB_DIM;
//
//
//	/* let's say we have a single token word "sky"
//	 * this token will have it's own embedding
//	 * and this embedding will have it's own projections all of them query, key and value
//	 * say "sky" has embedding -> e and has q_proj -> q and has key_proj -> k and has v_proj -> value
//	 * so that becomes e * wq = Q, e * wk = K and e * wv = V
//	 * then onwords we calculate the attention for each single token
//	 *
//	 */
//
//	// Attention(Q, K, V) = softmax(q * transpose(k) / sqrt(dk)) * v
//	Tensor *tokens = tensor_create(ndim, shape_tokens);
//	Tensor *qw = tensor_create_weights(ndim, shape_q_weights);	
//	Tensor *kw = tensor_create_weights(ndim, shape_k_weights);
//	Tensor *vw = tensor_create_weights(ndim, shape_v_weights);
//
//	Tensor *qwt = transpose(qw);
//	tensor_shape(tokens);
//	tensor_shape(qwt);
//	Tensor *q_pro = matmul(tokens, qwt);
//	tensor_shape(q_pro);
//
//
//	return 0;
//}
