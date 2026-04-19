#ifndef MODEL_H
#define MODEL_H
#include "tensor.h"
#include "config.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"


void tensor_randomize_weights(Tensor *x);
void tensor_randomize(Tensor *x);
void mha_init_params(MHA *m);
void layer_norm_init_params(LayerNorm *ln);
void ffn_init_params(FFN *f);

#endif
