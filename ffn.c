#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"

#define SEQ_LEN 10
#define EMB_DIM 32
#define MAX(a, b) (((a) > (b)) ? (a) : (b))


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

	/* 
	 * we have some tensor of shape (SEQ_LEN, EMB_DIM)
	 * we want to perform some FFNN overit
	 * and what FFN does is it takes the values of tensor "FLATTEN"
	 * and assign some "WEIGHTS" to the flatten tensor
	 * and does 'Y = WX + B' for each "PERCEPTRON"
	 *
	 */


	if (!tokens) {
		fprintf(stderr, "Something wrong with memory allocation\n");
		return 0;
	}

	Tensor *res = forward(tokens);
	tensor_get(res);
	tensor_shape(res);
	return 0;
}

