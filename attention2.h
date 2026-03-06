#ifndef ATTENTION_H
#define ATTENTION_H
#include "tensor.h"

typedef struct {
	Tensor *wq;
	Tensor *wk;
	Tensor *wv;
	Tensor *wo;

	Tensor *Q;
	Tensor *K;
	Tensor *V;
	Tensor *out;

	int num_heads;
	int dk;
} MHA;

MHA *mha_create(int heads, int seq_len, int emb_dim);
Tensor *scaled_dot_product_attention(Tensor *Q, Tensor *K, Tensor *V, int heads);
Tensor *multihead_attention(Tensor *tokens, int heads, int seq_len, int emb_dim, int ndim);

#endif

