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
	int shape_input[2] = {SEQ_LEN, EMB_DIM};
	int shape_target[2] = {SEQ_LEN, EMB_DIM};

	// Create input data tensor
	Tensor *input = tensor_create(ndim, shape_input);
	
	// Create target tensor we want to compare out results with 
	Tensor *target = tensor_create(ndim, shape_input);

	// Create multi-head attention 
	MHA *M = mha_create(HEADS, SEQ_LEN, EMB_DIM);

	// Create FFN network
	FFN *F = ffn_create(EMB_DIM, HIDDEN_DIM);

	int batches = SEQ_LEN / BATCH_SIZE;

	for (int b = 0; b < batches; b++) {
		float *batch_ptr = input->data + (b * BATCH_SIZE * EMB_DIM);
		int shape_batch[2] = {BATCH_SIZE, EMB_DIM};
		
		// Create input and target tensor dummy and copy ptr index data
		Tensor *batch_tensor = tensor_create(2, shape_batch);
		Tensor *target_tensor = tensor_create(2, shape_batch);
		memcpy(batch_tensor->data, batch_ptr, EMB_DIM * BATCH_SIZE * sizeof(float));
		memcpy(target->data, batch_ptr, EMB_DIM * BATCH_SIZE * sizeof(float));


		// Run forward pass 
		Tensor *att = mha_forward(batch_tensor, M);
		Tensor *ln1 = layer_norm_forward(att);
		Tensor *ffn_out = ffn_forward(ln1, F);
		Tensor *ln2 = layer_norm_forward(ffn_out);


		// Calculating loss 
		Tensor *loss_grad = tensor_mse_loss(ln2, target_tensor);
		float loss = loss_value(ln2, target_tensor);
		printf("Loss value: %f \n", loss);

		// Running Backward pass
		Tensor *d_ffn_out = ffn_backward(F, ln2, loss_grad);
		Tensor *d_mha_out = mha_backward(M, d_ffn_out, batch_tensor);

		// Updating weights
		sgd_optimizer(F->w1, F->dw1, LR);
		sgd_optimizer(F->w2, F->dw2, LR);
		sgd_optimizer(F->a1, F->da1, LR);
		sgd_optimizer(F->h1, F->dh1, LR);

		tensor_fill_zeros(F->dw1);
		tensor_fill_zeros(F->dw2);
		tensor_fill_zeros(F->da1);
		tensor_fill_zeros(F->dh1);

		sgd_optimizer(M->wq, M->dwq, LR);
		sgd_optimizer(M->wk, M->dwk, LR);
		sgd_optimizer(M->wv, M->dwv, LR);

		tensor_fill_zeros(M->dwq);
		tensor_fill_zeros(M->dwk);
		tensor_fill_zeros(M->dwv);

		printf("Batch completed!\n");


	}
	return 0;
	
}

