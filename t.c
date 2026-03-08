#include <stdio.h>
#include "tensor.h"
#include <stdlib.h>

Tensor **tensor_slice_cols(Tensor *t, int dk)  {
	int heads = 8;
	Tensor **mha = malloc(heads * sizeof(Tensor *));
	int rows = t->shape[0];
	int cols = t->shape[1];
	int shape[2] = {10, 4};

	for (int k = 0; k < heads; k++) {
		Tensor *head = tensor_create(2, shape);
		float *base = t->data + (k * dk);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				float *src = (base + (i * cols) + j);
				head->data[i * dk + j] = *src;
			}
		}
		mha[k] = head;
	}
	return mha;
}

int main() {

	int shape[2] = {10, 32};
	int ndim = 2;
	Tensor *t = tensor_create(2, shape);
	Tensor **s = tensor_slice_cols(t, 4);

	return 0;
}

