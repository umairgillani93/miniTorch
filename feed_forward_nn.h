#ifndef FFN_H
#define FFN_H
#include "tensor.h"
#include <stdbool.h>
#include "attention2.h"
#include "layer_norm.h"

typedef struct {
	// create the required tensors
	Tensor *w1;
	Tensor *w2;

	// save weights
	Tensor *dw1;
	Tensor *dw2;
	Tensor *h1;
	Tensor *a1;
	
	// output tensor
	Tensor *out;
	Tensor *inputs;
	bool save_inputs;

} FFN;

Tensor *ffn_forward(Tensor *x, FFN *y);
FFN *ffn_create(int input_dim, int hidden_dim);
Tensor *relu(Tensor *x);


#endif


