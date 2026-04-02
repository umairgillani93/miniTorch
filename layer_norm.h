#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"
#include "attention2.h"

typedef struct {
	int features; // in out case it's d_model OR embedding dimension,
								// as layer_norm is calculated along last dimention
	Tensor *beta; // Learnable shift. shape: features OR emb_dim
	Tensor *gemma; // Learnable sclae. shape: features OR emb_dim
	Tensor *x_hat; // Normalized(x)
	float *var; // cached per row variance
} LayerNorm;

Tensor *layer_norm_forward(Tensor *t);

#endif
