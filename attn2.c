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


//Tensor *mha_backward(MHA *m, Tensor *t) {
//	mha->dwq = tensor_create(m->ndim, t->shape[1]);
//	mha->dwk = tensor_create(m->ndim, t->shape[1]);
//	mha->dwv = tensor_create(m->ndim, t->shape[1]);
//
//	// How can I take their gradients now??
//}


Tensor *mha_forward(Tensor *t, MHA *mha) {
	// free the existing Q, K and V to replace with tensor_matmul operations
	tensor_free(mha->Q);
	tensor_free(mha->K);
	tensor_free(mha->V);

	mha->Q = tensor_matmul(t, mha->wq);
	mha->K = tensor_matmul(t, mha->wk);
	mha->V = tensor_matmul(t, mha->wv);

	// extract the required parameters
	int rows = t->shape[0];
	int cols = t->shape[1];
	int heads = mha->num_heads;
	int dk = mha->dk;


	tensor_free(mha->out);
	mha->out = tensor_create(2, t->shape);
	int common_shape[2] = {rows, dk};

	for (int k = 0; k < heads; k++) {
		// first of all I need scaled_dot_product_scores
		// for which I need slicing Q, K and V
		// slicing logic first
		Tensor *Q_h = tensor_create(2, common_shape);
		Tensor *K_h = tensor_create(2, common_shape);
		Tensor *V_h = tensor_create(2, common_shape);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				int src = i * cols + j + k * dk;
				int dest = i * dk + j;

				Q_h->data[dest] = mha->Q->data[src];
				K_h->data[dest] = mha->K->data[src];
				V_h->data[dest] = mha->V->data[src];

			}
		}

		Tensor *head_out = scaled_dot_product_attention(Q_h, K_h, V_h, dk);
		// Write this back to the output
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				int head_idx = i * dk + j;
				int out_idx = i * cols + j + k * dk;
				mha->out->data[out_idx] = head_out->data[head_idx];
			}
		}
		tensor_free(Q_h);
		tensor_free(K_h);
		tensor_free(V_h);
	}
	return mha->out;
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
//	MHA *mha = mha_create(heads, SEQ_LEN, EMB_DIM);
//	Tensor *multi_head = mha_forward(tokens, mha);
//	tensor_shape(multi_head);
//	tensor_get(multi_head);
//
//	return 0;
//}
