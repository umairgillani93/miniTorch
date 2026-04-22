#include <stdbool.h>
#ifndef TENSOR_H
#define TENSOR_H


typedef struct Arena Arena;
typedef struct Tensor Tensor;



typedef struct Tensor {
	int *shape;
	int *stride;
	int ndim;
	float *data;
} Tensor;


// prototypes definition
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_create_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights(int ndim, int *shape);
Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_softmax(Tensor *a);
Tensor *tensor_transpose(Tensor *t);
Tensor *relu_backward(Tensor *x, Tensor *y);
Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target);
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_scaler_multiplication(Tensor *x, float a);
Tensor *tensor_scaler_addition(Tensor *x, float a);
void tensor_fill_zeros(Tensor *a);
void tensor_add_inplace(Tensor **a, Tensor **b);
//void tensor_free(Tensor *t);
void tensor_get(Tensor *t);
void tensor_check(char *name, Tensor *x);
int tensor_size(Tensor *t);
float loss_value(Tensor *a, Tensor *b);
void tensor_shape(Tensor *t);
bool is_exploding(Tensor *x);
void clip_gradient(Tensor *x);


// new methods added for model struct
void tensor_randomize_weights(Tensor *x);
void tensor_randomize(Tensor *x);



// Arena tensor methods
Tensor *tensor_create_new(Arena *A, int ndim, int *shape);

#endif
