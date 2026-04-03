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
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_scaler_multiplication(Tensor *x, float a);
Tensor *tensor_scaler_addition(Tensor *x, float a);
void tensor_fill_zeros(Tensor *a);
void tensor_add_inplace(Tensor **a, Tensor **b);
void tensor_free(Tensor *t);
void tensor_get(Tensor *t);
int tensor_size(Tensor *t);
float loss_value(Tensor *a, Tensor *b);
void tensor_shape(Tensor *t);

#endif
