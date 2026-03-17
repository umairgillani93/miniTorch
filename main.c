#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
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
	//int num_batches = SEQ_LEN / BATCH_SIZE;

	int num_chunks = SEQ_LEN / BATCH_SIZE;
	for (int b = 0; b < num_chunks; b++) {

    float *batch_ptr = T->data + b * BATCH_SIZE * EMB_DIM;

		int shape_local[2] = {BATCH_SIZE, EMB_DIM};
    Tensor *batch_tensor = tensor_create(2, shape_local);
		memcpy(batch_tensor->data, batch_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));

		MHA *m = mha_create(HEADS, BATCH_SIZE, EMB_DIM);
		//tensor_shape(batch_tensor);
		//tensor_shape(m->wq);
		//tensor_shape(m->Q);
		//printf("heads: %d\n", m->num_heads);
		//printf("dk: %d\n", m->dk);
		//break;
		Tensor *multi_head = mha_forward(batch_tensor, m);
		memcpy(batch_ptr, batch_tensor->data, BATCH_SIZE * EMB_DIM * sizeof(float));
		tensor_shape(multi_head);
}

	return 0;
}
