#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
	int *shape;
	int *stride;
	int ndim;
	float *data;
} Tensor;


// prototypes definition
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_create_weights(int ndim, int *shape);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_softmax(Tensor *a);
Tensor *tensor_transpose(Tensor *t);
Tensor *relu_backward(Tensor *x, Tensor *y);
Tensor *tensor_mse_loss(Tensor *pred, Tensor *target);
void tensor_free(Tensor *t);
void tensor_get(Tensor *t);
int tensor_size(Tensor *t);
void tensor_shape(Tensor *t);

#endif
