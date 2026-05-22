
#include <stdio.h>
#include <stdlib.h>

typedef enum {
	ADD = 10,
	MUL,
	MATMUL,
	RELU,
	SUB,
	DIV,
	EXP,
	LOG
} OpName;

typedef struct Op {
	OpType type;
	const *char name;
	void(*backward)(Arena *A, struct Tensor *self);
} Op;

typedef struct Tensor {
	int id;
	int *shape;
	int *stride;
	int ndim;
	float *data;
	
	// New parameters
	Tensor *grad;
	bool requires_grad;
	Op *operations; // Added name, and type as well!
	Tensor **parents;
	int num_parents;
} Tensor;



Tensor *tensor_create_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	t->ndim = ndim;
	t->shape = arena_alloc(A, ndim * sizeof(int));
	t->stride = arena_alloc(A, ndim * sizeof(int));

	// define the shape of Tensor
	int total = 1;
	for (int i = ndim - 1; i >= 0; i--) {
		t->shape[i] = shape[i];
		t->stride[i] = total;
		total *= shape[i];
	}
	// For autograd
	t->data = arena_alloc(A, total * sizeof(float));
	t->parents = NULL;
	t->operations = NULL;
	t->grad = NULL;
	t->num_parents = 0;
	return t;
}




int main() {
	
	Tensor *x = tensor_create_new(
	return 0;
}
