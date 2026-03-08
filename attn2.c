#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "tensor.h"
#include "attention2.h"

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX
#define EMB_DIM 32 // out model dimension, i.e Embedding size for each token
#define SEQ_LEN 10 // assume these are 10 tokens converted into token IDs
#define BATCH_SIZE 2
#define EPS 1e-5

Tensor *multihead_attention(Tensor *t, int heads, int seq_len, int emb_dim, int ndim) {
	
	// final pointer array storing Tensor pointers corresponding to each "head" tensor
	Tensor **arr= malloc(heads * sizeof(Tensor *));
	
	MHA *mha = mha_create(heads, seq_len, emb_dim);
	mha->Q = tensor_matmul(t, mha->wq);
	mha->K = tensor_matmul(t, mha->wk);
	mha->V = tensor_matmul(t, mha->wv);
	mha->num_heads = heads;
	mha->dk = emb_dim / heads;


	int rows = t->shape[0];
	int cols = t->shape[1];

	int common_shape[2] = {seq_len, mha->dk};
	Tensor *out = tensor_create(ndim, t->shape);

	for (int k = 0; k < heads; k++) {
		Tensor *Q_k = tensor_create(ndim, common_shape);
		Tensor *K_k = tensor_create(ndim, common_shape);
		Tensor *V_k = tensor_create(ndim, common_shape);

		float *base_Q = mha->Q->data + (k * mha->dk);
		float *base_K = mha->K->data + (k * mha->dk);
		float *base_V = mha->V->data + (k * mha->dk);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < mha->dk; j++) {
				int src = i * cols + j;
				int dest = (i * mha->dk) + j;
				Q_k->data[dest] = *(base_Q + src);
				K_k->data[dest] = *(base_K + src);
				V_k->data[dest] = *(base_V + src);
			}
		}

		arr[k] = scaled_dot_product_attention(Q_k, K_k, V_k, mha->dk);
	}

	for (int k = 0; k < heads; k++) {
		Tensor *head = arr[k];

		float *base = out->data + (k * mha->dk);

		int h_rows = head->shape[0];
		int h_cols = head->shape[1];

		for (int i = 0; i < out->shape[0]; i++) {
			for (int j = 0; j < mha->dk; j++) {
				int src_idx = (i * mha->dk) + j;
				*(base + i * cols +j) = head->data[src_idx];
			}
		}
	}

	return out;
}	

Tensor *scaled_dot_product_attention(Tensor *Q, Tensor *K, Tensor *V, int dk) {
	Tensor *kt = tensor_transpose(K);
	Tensor *qkt = tensor_matmul(Q, kt);
	for (int i = 0; i < qkt->shape[0]; i++) {
		for (int j = 0; j < qkt->shape[1]; j++) {
			qkt->data[i * qkt->shape[1] + j] = qkt->data[i * qkt->shape[1] +j] / sqrtf(dk);;
		}
	}
	Tensor *qkt_soft = tensor_softmax(qkt); // RAND_FLOAT is random we'll calculate this later
	Tensor *ret = tensor_matmul(qkt_soft, V);
	return ret;
}

MHA *mha_create(int num_heads, int seq_len, int emb_dim) {
	MHA *mha = malloc(sizeof(MHA));
	int ndim = 2;
	int shape_weights[2] = {emb_dim, emb_dim};
	int shape_tokens[2] = {seq_len, emb_dim};
	mha->wq = tensor_create_weights(ndim, shape_weights);
	mha->wk = tensor_create_weights(ndim, shape_weights);
	mha->wv = tensor_create_weights(ndim, shape_weights);
	mha->wo = tensor_create_weights(ndim, shape_weights); // output weights
	
	// define the tensor
	mha->Q = tensor_create(ndim, shape_tokens);
	mha->K = tensor_create(ndim, shape_tokens);
	mha->V = tensor_create(ndim, shape_tokens);
	mha->out = tensor_create(ndim, shape_tokens);
	mha->num_heads= num_heads;
	return mha;
}

	

//int main() {
//	//int seed = 32;
//	//srand(seed);
//	int ndim = 2;
//
//	int shape_tokens[2] = {SEQ_LEN, EMB_DIM};
//	int shape_weights[2] = {EMB_DIM, EMB_DIM};
//
//	Tensor *tokens = tensor_create(ndim, shape_tokens);
//	
//	int heads = 8;
//	int HEAD_DIM = EMB_DIM / heads;
//	Tensor *multi_head = multihead_attention(tokens, heads, SEQ_LEN, EMB_DIM, ndim);
//	tensor_shape(multi_head);
//	tensor_get(multi_head);
//
//	return 0;
//}
