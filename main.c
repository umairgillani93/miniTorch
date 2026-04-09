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

	for (int e = 1; e <= EPOCHS; e++) {
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
			LayerNorm *L1 = layer_norm_create(EMB_DIM);
			Tensor *ln1 = layer_norm_forward(L1, attn_score);
			//printf("LayerNorm #1 ran successfully!\n");

			// Create FFN feed-forward NN and run ffn_forward pass
			FFN *f = ffn_create(EMB_DIM, 128);
			Tensor *ffn_ln = ffn_forward(ln1, f);

			// Apply layer_norm
			LayerNorm *L2 = layer_norm_create(EMB_DIM);
			Tensor *ln2 = layer_norm_forward(L2, ffn_ln);
			//tensor_shape(ln2);
			//printf("LayerNorm #2 ran successfully!\n");
		
			Tensor *loss = tensor_mse_loss(ln2, target_batch);
			float loss_to_show = loss_value(ln2, target_batch);

			Tensor *dx_for_ffn = tensor_create(2, shape_local);
			Tensor *dx_for_mha = tensor_create(2, shape_local);
			//printf("All good\n");
			layer_norm_backward(L2, ffn_ln, loss, dx_for_ffn, LR);
			Tensor *ffn_backpass = ffn_backward(f, ln1, dx_for_ffn);
			layer_norm_backward(L1, attn_score, ffn_backpass, dx_for_mha, LR);
			
			Tensor *mha_backpass = mha_backward(m_batch, dx_for_mha, batch_tensor);

			//printf("w1 shape: \n");
			//tensor_shape(f->w1);

			//printf("dw1 shape: \n");
			//tensor_shape(f->dw1);

			clip_gradient(f->dw1);
			clip_gradient(f->dw2);
			clip_gradient(f->da1);
			clip_gradient(f->dh1);
			sgd_optimizer(f->w1, f->dw1, LR);
			sgd_optimizer(f->w2, f->dw2, LR);
			sgd_optimizer(f->a1, f->da1, LR);
			sgd_optimizer(f->h1, f->dh1, LR);

			tensor_fill_zeros(f->dw1);
			tensor_fill_zeros(f->dw2);
			tensor_fill_zeros(f->da1);
			tensor_fill_zeros(f->dh1);

			clip_gradient(m_batch->dwq);
			clip_gradient(m_batch->dwk);
			clip_gradient(m_batch->dwv);
			sgd_optimizer(m_batch->wq, m_batch->dwq, LR);
			sgd_optimizer(m_batch->wk, m_batch->dwk, LR);
			sgd_optimizer(m_batch->wv, m_batch->dwv, LR);

			tensor_fill_zeros(m_batch->dwq);
			tensor_fill_zeros(m_batch->dwk);
			tensor_fill_zeros(m_batch->dwv);

			clip_gradient(L1->d_gamma);
			clip_gradient(L1->d_beta);
			clip_gradient(L2->d_gamma);
			clip_gradient(L2->d_beta);

			sgd_optimizer(L1->gamma, L1->d_gamma, LR);
			sgd_optimizer(L1->beta,  L1->d_beta,  LR);
			sgd_optimizer(L2->gamma, L2->d_gamma, LR);
			sgd_optimizer(L2->beta,  L2->d_beta,  LR);

			// Zero out gradients
			tensor_fill_zeros(L1->d_gamma);
			tensor_fill_zeros(L1->d_beta);
			tensor_fill_zeros(L2->d_gamma);
			tensor_fill_zeros(L2->d_beta);

			// Copy the chunks back to main tensor
			memcpy(batch_ptr, mha_backpass->data, BATCH_SIZE * EMB_DIM * sizeof(float));
			//printf("batch tensor copied to the main\n");

			// free memory
			//tensor_free(batch_tensor);
			//tensor_free(attn_score);
			//tensor_free(ln1);
			//tensor_free(ffn_ln);
			//tensor_free(ln2);
			//printf("Running Epoch: %d, Batch: %d\n", e, b);
			if (b % 10 == 0) {
				printf("Loss: %f, after Epochs: %d\n", loss_to_show, e);
				printf("Batch competed!\n");
				//printf("Batch competed!\n");
			}
		}
	}
	printf("Traning finished!\n");
	//tensor_shape(T);

	return 0;
}
