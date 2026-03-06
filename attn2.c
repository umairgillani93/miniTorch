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

Tensor *multihead_attention(Tensor *tokens, int heads, int seq_len, int emb_dim, int ndim) {
	
	// final pointer array storing Tensor pointers corresponding to each "head" tensor
	Tensor **arr= malloc(heads * sizeof(Tensor *));
	
	MHA *mha = mha_create(heads, seq_len, emb_dim);
	mha->Q = tensor_matmul(tokens, mha->wq);
	mha->K = tensor_matmul(tokens, mha->wk);
	mha->V = tensor_matmul(tokens, mha->wv);
	mha->num_heads = heads;
	mha->dk = emb_dim / mha->num_heads;

	for (int i = 0; i < heads; i++) {
		Tensor *score = scaled_dot_product_attention(mha->Q, mha->K, mha->V, mha->num_heads);
		arr[i] = score;
	}

	int res_shape[2] = {seq_len, emb_dim};
	int res_rows = res_shape[0];
	int res_cols = res_shape[1];
	Tensor *res = tensor_create(ndim, res_shape);

	int offset = 0;
	for (int k = 0; k < heads; k++) {
		Tensor *h = res[k];
		int rows = h->shape[0];
		int cols = h->shape[1];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				int src_idx = h->data[i * cols + j];
				int dest_idx = res->data[i * res->shape[1] + (offset + j)];
			}
		}	
		offset += cols;
	}

	printf("Success!\n");
	return res;
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

	

int main() {
	//int seed = 32;
	//srand(seed);
	int ndim = 2;

	int *shape_tokens = malloc(ndim * sizeof(int));
	int *shape_weights = malloc(ndim * sizeof(int));

	// define shape_tokens
	shape_tokens[0] = SEQ_LEN;
	shape_tokens[1] = EMB_DIM; // this is for token embeddings
	
	// weights shape
	//shape_weights[0] = EMB_DIM;
	//shape_weights[1] = EMB_DIM;
	

	Tensor *tokens = tensor_create(ndim, shape_tokens);
	int heads = 8;
	Tensor *multi_head = multihead_attention(tokens, heads, SEQ_LEN, EMB_DIM, ndim); 

	return 0;
}
