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
	
	Tensor *T = tensor_create(2, shape);
	int size = tensor_size(T);
	int stride = EMB_DIM * 2; // we need chunks of rows
	
	int batch_size = 10;

	for (int b = 0; b < SEQ_LEN/batch_size; b++) { // divide the whole sequence in large batch chunks
		// now insdie each row we have [10, 32] 2d tensor
		// creating a Tensor of 2D with [100, 32] size
		int ndim = 2;
		int rows = SEQ_LEN / batch_size;
		int cols = EMB_DIM;
		int shape_2d[2] = {rows, cols};
		Tensor *t = tensor_create(ndim, shape_2d);
		int tensor_idx = 0;
		for (int r = 0; r < batch_size; r++) {
			for (int c = 0; c < EMB_DIM; c++) {
				int idx = (b * batch_size + r) * EMB_DIM + c;
				tensor_idx = idx;
			}
		}

		t->data[0] = T->data[tensor_idx];
		tensor_shape(t);
	}

	return 0;
}
