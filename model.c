#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "tensor.h"


#define RAND_FLOAT (float) rand() / (float) RAND_MAX

void tensor_randomize_weights(Tensor *x) {
	size_t size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = RAND_FLOAT;
	}
}

void tensor_randomize(Tensor *x) {
	size_t size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = (rand() % 10) + 1.0f;
	}
}
