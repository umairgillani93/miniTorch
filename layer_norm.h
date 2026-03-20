#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"
#include "attention2.h"

Tensor *layer_norm_forward(Tensor *t);

#endif
