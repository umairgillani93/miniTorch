#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "attention2.h"

Tensor **tensor_slice_cols(Tensor *t, int dk)  {
	int heads = 8;
	Tensor **mha = malloc(heads * sizeof(Tensor *));
	int rows = t->shape[0];
	int cols = t->shape[1];
	int shape[2] = {10, 4};

	for (int k = 0; k < heads; k++) {
		Tensor *head = tensor_create(2, shape);
		float *base = t->data + (k * dk);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < dk; j++) {
				float *src = (base + (i * cols) + j);
				head->data[i * dk + j] = *src;
			}
		}
		mha[k] = head;
	}
	return mha;
}

//Tensor *mha_backward(Tensor *mha) {
//	Tensor *wq = mha->wq;
//	Tensor *wk = mha->wk;
//	Tensor *wv = mha->wv;
//	Tensor *wo = mha->wo;
//
//	tensor_shape(wq);
//	tensor_shape(wk);
//	tensor_shape(wv);
//
//	return NULL;
//}

int main() {

	int seq_len = 10;
	int emb_dim = 32;
	int shape[2] = {seq_len, emb_dim};
	int heads = 8;
	int ndim = 2;

	Tensor *t = tensor_create(2, shape);
	Tensor *mha = mha_forward(t, heads, seq_len, emb_dim, ndim);
	tensor_shape(mha->Q);
	//Tensor *mha_b = mha_backward(mha);

	return 0;
}

