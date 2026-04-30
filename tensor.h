#include <stdbool.h>
#ifndef TENSOR_H
#define TENSOR_H


typedef struct Arena Arena;
typedef struct Tensor Tensor;

typedef struct Op {
	void(*backward)(struct Tensor *self);
} Op;

typedef struct Tensor {
	int *shape;
	int *stride;
	int ndim;
	float *data;
	
	// New parameters
	Tensor *grad;
	bool requires_grad;
	Op *operations;
	Tensor **parents;
	int num_parents;
} Tensor;


// prototypes definition

Tensor *tensor_fill_like(Arena *A, Tensor *a, double eps);
Tensor *tensor_row_sum(Arena *A, Tensor *x);
Tensor *tensor_scalling(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_square(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_div(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_create_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights(int ndim, int *shape);
Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_softmax(Arena *A, Tensor *a);
Tensor *tensor_transpose(Tensor *t);
Tensor *relu_backward(Tensor *x, Tensor *y);
Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target);
Tensor *tensor_scaler_multiplication(Tensor *x, float a);
Tensor *tensor_scaler_addition(Tensor *x, float a);
void tensor_fill_zeros(Tensor *a);
void tensor_add_inplace(Tensor **a, Tensor **b);
//void tensor_free(Tensor *t);
void tensor_get_2d(Tensor *t);
void tensor_check(char *name, Tensor *x);
int tensor_size(Tensor *t);
float loss_value(Tensor *a, Tensor *b);
void tensor_shape_2d(Tensor *t);
bool is_exploding(Tensor *x);
void clip_gradient(Tensor *x);
Tensor *tensor_sqrt(Arena *A, Tensor *x);
//Tensor tensor_add(int *row1, int *row2); 
//Tensor tensor_sub(int *row1, int *row2);

// new methods added for model struct
void tensor_randomize_weights(Tensor *x);
void tensor_randomize(Tensor *x);

// Arena tensor methods
Tensor *tensor_create_new(Arena *A, int ndim, int *shape);


// Autograd tensor methods
void tensor_matmul_backward(Tensor *x);
void tensor_mean_backward(Tensor *x);
void tensor_add_backward(Tensor *x);
void tensor_square_backward(Tensor *x);
void tensor_sqrt_backward(Tensor *x);
void tensor_expand_cols_backward(Tensor *x);
void tensor_softmax_backward(Tensor *x);
Tensor *tensor_mean(Arena *A, Tensor *x);
Tensor *tensor_expand_cols(Arena *A, Tensor *m, int out_shape);
Tensor *tensor_add(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_subtract(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_expand_rows(Arena *A, Tensor *a, int out_rows);


#endif
