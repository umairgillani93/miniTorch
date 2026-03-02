#ifndef FFN_H
#define FFN_H
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"

typedef struct {
	Tensor *w1;
	Tensor *w2;
} FFN;

Tensor *forward(FFN *f, Tensor *x);
FFN *ffn_create(int input_dim, int hidden_dim);


#endif


