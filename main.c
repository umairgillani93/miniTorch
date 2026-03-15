#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"
#include "config.h"

int main() {
	int ndim = 2;
	srand(time(NULL));
	int shape[2] = {SEQ_LEN, EMB_DIM};
	
	Tensor *t = tensor_create(2, shape);
	int size = tensor_size(t);
	int stride = EMB_DIM * 2;

	for (int i = 0; i < size / stride; i++) {
		float *row_idx = t->data + i * EMB_DIM;
		for (int j = 0; j < stride; j++) {
			printf("%f ", row_idx[j]);
		}
		printf("\n");
	}

	return 0;
}
