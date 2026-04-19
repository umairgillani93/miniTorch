#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"
#include "attention2.h"
#include "arena.h"

typedef struct {
	int features; // in out case it's d_model OR embedding dimension,
								// as layer_norm is calculated along last dimention
	Tensor *beta; // Learnable shift. shape: features OR emb_dim
	Tensor *gamma; // Learnable sclae. shape: features OR emb_dim
	Tensor *d_beta;
	Tensor *d_gamma;
	Tensor *x_hat; // Normalized(x)
	float *var; // cached per row variance
} LayerNorm;

Tensor *layer_norm_forward(Arena *A, LayerNorm *ln, Tensor *t);
LayerNorm *layer_norm_create(int features);
LayerNorm *layer_norm_create_new(Arena *A, int features);
void layer_norm_backward(LayerNorm *ln, Tensor *x, Tensor *dy, Tensor *dx, float lr);
void layer_norm_init_params(LayerNorm *ln);

#endif
