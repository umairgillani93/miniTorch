#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"

#define SEQ_LEN 10
#define EMB_DIM 32
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

FFN *ffn_create(int input_dim, int hidden_dim) {
	int ndim = 2;
	int shape1[2] = {input_dim, hidden_dim};
	int shape2[2] = {hidden_dim, input_dim};

	FFN *f = malloc(sizeof(FFN));
	f->w1 = tensor_create_weights(ndim, shape1);
	f->w2 = tensor_create_weights(ndim, shape2);

	return f;
}

Tensor *ffn_forward(Tensor *x, FFN *f) {
	Tensor *h1 = tensor_matmul(x, f->w1);
	Tensor *a1 = relu(h1);
	Tensor *out = tensor_matmul(a1, f->w2);
	return out;
}	


Tensor *relu(Tensor *x) {
	int size = x->shape[0] * x->shape[1];
	for (int i = 0; i < size; i++) {
		float val = MAX(0, x->data[i]);
		x->data[i] = val;
	}
	return x;
}

Tensor *forward(Tensor *x) {
	int shape1[2] = {32, 128};
	int shape2[2] = {128, 32};
	Tensor *w1 = tensor_create_weights(2, shape1);
	Tensor *w2 = tensor_create_weights(2, shape2);
	Tensor *h1 = tensor_matmul(x, w1);
	Tensor *a1 = relu(h1);
	Tensor *out = tensor_matmul(a1, w2);

	return out;
}	

int main() {
	int ndim = 2;
	int *shape_tokens = malloc(ndim * sizeof(int));
	if (!shape_tokens) {
		fprintf(stderr, "Something wrong with memory allocation\n");
		return 0;
	}
	shape_tokens[0] = SEQ_LEN;
	shape_tokens[1] = EMB_DIM;

	Tensor *tokens = tensor_create(ndim, shape_tokens);

	if (!tokens) {
		fprintf(stderr, "Something wrong with memory allocation\n");
		return 0;
	}

	FFN *f = ffn_create(32, EMB_DIM);
	Tensor *res = ffn_forward(ffn, tokens);
	tensor_get(res);

	return 0;
}

