#include <stdbool.h>
#ifndef TENSOR_H
#define TENSOR_H
#include <stddef.h>

extern size_t GLOBAL_TENSOR_ID;

typedef struct Arena Arena;
typedef struct Tensor Tensor;

typedef struct {
    int visited_nodes;
    int null_parents;
    int null_ops;
    int null_grad;
    int missing_links;
		int corrupt_nodes;
} GraphReport;

typedef enum {
	F,
	FILL_LIKE,
	LEAF,
	MSE,
	ADD, 
	MUL,
	MATMUL,
	RELU,
	SUB,
	DIV,
	LOG,
	TRANSPOSE,
	SLICE,
	CONCAT,
	EXPAND_COLS,
	EXPAND_ROWS,
	ROW_MAX,
	EXP,
	ROW_SUM,
	SCALLED,
	SOFTMAX,
	MEAN,
	SQUARE,
	SQRT,
	DIVISION,
	SCALER_DIV

} OpType;


typedef struct Op {
	OpType type;
	const char *name;
	void(*backward)(Arena *A, struct Tensor *self);
} Op;

typedef struct Tensor {
	int id;
	int *shape;
	int *stride;
	int ndim;
	float *data;
	
	// New parameters
	bool is_leaf;	
	Tensor *grad;
	bool requires_grad;
	Op *operations; // Added name, and type as well!
	Tensor **parents;
	int num_parents;
	

	// For concat columns
	int cols;
} Tensor;


// prototypes definition


bool *vislist();
void validate_tensor_graph(Arena *A, Tensor *root, bool *visited, GraphReport *rep, int max_tensors);
void run_graph_validation(Arena *A, Tensor *o, int max_nodes);
void traverse_graph(Tensor *root, bool *visited);
void tensor_metadata(Tensor *x);
void backward(Arena *A, Tensor *out);
void f_backward(Arena *A, Tensor *out);
void dfs(Arena *A, Tensor *root, bool *visited, Tensor **topo, int *size);
Tensor *tensor_f(Arena *A, Tensor *x);
Tensor *tensor_scaler_div(Arena *A, Tensor *x, float val);
Tensor *tensor_slice_cols(Arena *A, Tensor *a, int num_heads, int dk);
Tensor *tensor_fill_like(Arena *A, Tensor *a, double eps);
Tensor *tensor_concat(Arena *A, Tensor **heads, int k);
Tensor *tensor_relu(Arena *A, Tensor *x);
Tensor *tensor_fill_val(Arena *A, Tensor *a, int val);
Tensor *tensor_row_sum(Arena *A, Tensor *x);
Tensor *tensor_scalling(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_square(Arena *A, Tensor *a);
Tensor *tensor_div(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_create_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights_new(Arena *A, int ndim, int *shape);
Tensor *tensor_create_weights(int ndim, int *shape);
Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_softmax(Arena *A, Tensor *a);
Tensor *tensor_transpose(Arena *A, Tensor *t);
Tensor *relu_backward(Tensor *x, Tensor *y);
Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target);
Tensor *tensor_scaler_multiplication(Tensor *x, float a);
Tensor *tensor_scaler_addition(Arena *A, Tensor *x, float a);
void tensor_fill_zeros(Tensor *a);
void tensor_add_inplace(Tensor **a, Tensor **b);
void tensor_fill_ones(Tensor *x);
void tensor_accumulate(Tensor *a, Tensor *b);
void tensor_relu_backward( Tensor *out);
void tensor_subtract_backward(Arena *A, Tensor *x);
void tensor_fill_like_backward(Arena *A, Tensor *o);
void tensor_concat_backward(Tensor *out);
void tensor_div_backward(Arena *A, Tensor *out);
//void tensor_free(Tensor *t);
void tensor_get_2d(Tensor *t);
void tensor_check(char *name, Tensor *x);
int tensor_size(Tensor *t);
float loss_value(Tensor *a, Tensor *b);
bool tensor_equal(Tensor *x, Tensor *y);
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
void tensor_transpose_backward(Tensor *x);
void tensor_matmul_backward(Arena *A, Tensor *currNode);
void tensor_mean_backward(Arena *A, Tensor *x);
void tensor_add_backward(Arena *A, Tensor *o);
void tensor_square_backward(Arena *A, Tensor *o);
void tensor_sqrt_backward(Arena *A, Tensor *o);
void tensor_mse_backward(Tensor *x);
void tensor_expand_cols_backward(Arena *A, Tensor *out);
void tensor_expand_rows_backward(Arena *A, Tensor *o);
void tensor_softmax_backward(Tensor *x);
void tensor_relu_backward(Tensor *x);
void tensor_fill_with(Tensor *x, float v);
void tensor_slice_cols_backward(Tensor *x);
Tensor *tensor_mean(Arena *A, Tensor *x);
Tensor *tensor_expand_cols(Arena *A, Tensor *m, int out_shape);
Tensor *tensor_add(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_subtract(Arena *A, Tensor *a, Tensor *b);
Tensor *tensor_expand_rows(Arena *A, Tensor *a, int out_rows);


#endif
