#ifndef ATTENTION_H
#define ATTENTION_H
#include "tensor.h"

typedef struct {
	int num_heads;
	int dk;

	// weights needed for backward pass
	Tensor *wq;
	Tensor *wk;
	Tensor *wv;
	Tensor *wo;

	// cached values
	Tensor *Q;
	Tensor *K;
	Tensor *V;
	Tensor *out;

	// gradients for optimizers
	Tensor *dwq;
	Tensor *dwk;
	Tensor *dwv;
	
} MHA;

MHA *mha_create(int heads, int seq_len, int emb_dim);
Tensor *scaled_dot_product_attention(Tensor *Q, Tensor *K, Tensor *V, int heads);
Tensor *mha_forward(Tensor *t, MHA *m);
Tensor *mha_backward(MHA *m, Tensor *t);

#endif

