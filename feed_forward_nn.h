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
	Tensor *h1;
	Tensor *a1;

	// save weights
	Tensor *dw1;
	Tensor *dw2;
	Tensor *da1;
	Tensor *dh1;

	
	// output tensor
	Tensor *out;
	Tensor *inputs;
	bool save_inputs;
	// dw2
	//

} FFN;

Tensor *ffn_forward(Tensor *x, FFN *y);
Tensor *ffn_backward(FFN *f, Tensor *x, Tensor *loss);
FFN *ffn_create(int input_dim, int hidden_dim);
Tensor *relu(Tensor *x);
void sgd_optimizer(Tensor *a, Tensor *b, float lr);


#endif


