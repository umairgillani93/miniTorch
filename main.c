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
	
	// Creating actual data tensor
	Tensor *T = tensor_create(2, shape);
	int size = tensor_size(T);


	// Create global MHA
	MHA *M = mha_create(HEADS, SEQ_LEN, EMB_DIM);

	// Target tensor to compare the output against
	int shape_target[2] = {SEQ_LEN, EMB_DIM};
	Tensor *target = tensor_create(ndim, shape_target);

	// define batches for Actual tensor
	int num_chunks = SEQ_LEN / BATCH_SIZE;
	int EPOCHS = 2;

		for (int b = 0; b < num_chunks; b++) {

			float *batch_ptr = T->data + b * BATCH_SIZE * EMB_DIM;
			float *target_ptr = target->data + b * BATCH_SIZE * EMB_DIM;

			int shape_local[2] = {BATCH_SIZE, EMB_DIM};
			Tensor *batch_tensor = tensor_create(2, shape_local);
			Tensor *target_batch = tensor_create(2, shape_local);
			
			memcpy(batch_tensor->data, batch_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));
			memcpy(target_batch->data, target_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));

			MHA *m_batch = mha_create(HEADS, BATCH_SIZE, EMB_DIM);

			Tensor *attn_score = mha_forward(batch_tensor, m_batch);

			// Apply layer_norm
			Tensor *ln1 = layer_norm(attn_score);

			// Create FFN feed-forward NN and run ffn_forward pass
			FFN *f = ffn_create(EMB_DIM, 128);
			Tensor *ffn_ln = ffn_forward(ln1, f);
			
			// Apply layer_norm
			Tensor *ln2 = layer_norm(ffn_ln);
			
			Tensor *loss = tensor_mse_loss(ln2, target_batch);

			Tensor *ffn_backpass = ffn_backward(f, batch_tensor, loss);
			
			Tensor *mha_backpass = mha_backward(m_batch, ffn_backpass, batch_tensor);



			// Copy the chunks back to main tensor
			//memcpy(batch_ptr, ln2->data, BATCH_SIZE * EMB_DIM * sizeof(float));

			// free memory
			//tensor_free(batch_tensor);
			//tensor_free(attn_score);
			//tensor_free(ln1);
			//tensor_free(ffn_ln);
			//tensor_free(ln2);
		}

	return 0;
}
