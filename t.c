#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "attention2.h"


int main() {

	int seq_len = 10;
	int emb_dim = 32;
	int shape[2] = {seq_len, emb_dim};
	int heads = 8;
	int ndim = 2;

	Tensor *t = tensor_create(2, shape);
	int size = tensor_size(t);
	int cols = t->shape[1];
	int rows = t->shape[0];

	for (int i = 0; i < rows; i++) {
		float *row = t->data + i * cols;
		float row_sum = 0.0f;
		for (int j = 0; j < cols; j++) {
			row_sum += (dA[j] * A[j]);
			float val = dA[j] - row_sum;
		}


	}


	return 0;
	
}

