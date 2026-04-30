#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "attention2.h"
#include "arena.h"
#include "layer_norm.h"
#include "config.h"

float mean(float *arr, int size) {
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		sum += arr[i];
	}
	return sum / (float) size;
}

//float variance(int x, int size, float mean) {
//	float var = 0.0f;
//	for (int i = 0; i < size; i++) {
//		var = (x - mean) * (x - mean);	
//	}	
//	return var;
//}	

void layer_norm_init_params(LayerNorm *ln) {
	tensor_randomize_weights(ln->beta);
	tensor_randomize_weights(ln->gamma);
	tensor_randomize_weights(ln->d_beta);
	tensor_randomize_weights(ln->d_gamma);
}

Tensor *layer_norm_forward(Arena *A, LayerNorm *ln, Tensor *x) {

	/* ln->var  set this
	 ln->x_hat set this as well 
	1. Instead of float maths we need Tensor graph maths here
	2. Each operation should have their own reletive tensor
	3. tensor_mean, tensor_variance, tensor_add, tensor_sub etc ROW WISE
	4. [[1,2,3,]
 			[4,5,6]]
			[7,8, 9]]	
	5. Create computational graph using Tensor
	- tensor_mean
	- tensor_variance
	- tensor_add
	- tensor_sub etc..
	*/

	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *y_shape = arena_alloc(A, ndim * sizeof(int));
	y_shape[0] = rows;
	y_shape[1] = cols ;

	Tensor *y = tensor_create_new(A, ndim, y_shape);
	Tensor *mean = tensor_mean(A, x);
	
	// Docs reference: pytorch -> LINK HERE
	Tensor *mean_exp = tensor_expand_cols(A, mean, x->shape[1]);
	Tensor *diff = tensor_subtract(A, x, mean_exp);
	Tensor *sq = tensor_square(A, diff, diff);
	Tensor *var = tensor_mean(A, sq);
	Tensor *var_exp = tensor_expand_cols(A, var, x->shape[1]);
	Tensor *eps = tensor_fill_like(A, var_exp, 1e-3);

	Tensor *var_eps = tensor_add(A, var_exp, eps);
	Tensor *std = tensor_sqrt(A, var_eps);
	Tensor *out = tensor_div(A, var_eps, std);

	ln->var = var;
	ln->x_hat = out;

	Tensor *gamma_exp = tensor_expand_rows(A, ln->gamma, rows);
	Tensor *beta_exp = tensor_expand_rows(A, ln->beta, rows);
	tensor_randomize_weights(gamma_exp);
	tensor_randomize_weights(beta_exp);
	Tensor *yhat = tensor_scalling(A, gamma_exp, out);

	y = tensor_add(A, yhat, beta_exp);

	return y;
}

// MHA -> LAYER NORM -> FFN -> LAYER NORM -> Loss
// Loss -> LAYER NORM -> FFN -> LAYER NORM -> MHA
// Loss(ln2, target) -> LAYER NORM(L2, ffn) -> FFN(ln1, F) -> LAYER NORM(L1, attn) -> input(X)

//void layer_norm_backward(LayerNorm *ln, Tensor *x, Tensor *dy, Tensor *dx, float lr) {
//    int rows = x->shape[0];
//    int cols = x->shape[1];
//
//    // 1. Calculate gradients for gamma and beta (accumulate over rows)
//    for (int i = 0; i < rows; i++) {
//        for (int c = 0; c < cols; c++) {
//            float dy_val = dy->data[i * cols + c];
//            float xh_val = ln->x_hat->data[i * cols + c];
//
//            ln->d_gamma->data[c] += dy_val * xh_val;
//            ln->d_beta->data[c]  += dy_val;
//        }
//    }
//
//    // 2. Calculate gradient for input x
//    for (int i = 0; i < rows; i++) {
//        float *dy_row = dy->data + i * cols;
//        float *xh_row = ln->x_hat->data + i * cols;
//        float *dx_row = dx->data + i * cols;
//        
//        float var = ln->var[i];
//        float inv_std = 1.0f / sqrtf(var + EPS);
//
//        // Intermediate terms for the simplified LayerNorm gradient formula
//        float sum_dy_gamma = 0.0f;
//        float sum_dy_gamma_xhat = 0.0f;
//
//        for (int c = 0; c < cols; c++) {
//            float dy_gamma = dy_row[c] * ln->gamma->data[c];
//            sum_dy_gamma += dy_gamma;
//            sum_dy_gamma_xhat += dy_gamma * xh_row[c];
//        }
//
//        // The "one-liner" derivative formula for LayerNorm:
//        // dx = (1/N) * inv_std * [N*dy_gamma - sum(dy_gamma) - x_hat*sum(dy_gamma*x_hat)]
//        for (int c = 0; c < cols; c++) {
//            float dy_gamma = dy_row[c] * ln->gamma->data[c];
//            dx_row[c] = (1.0f / cols) * inv_std * (
//                (cols * dy_gamma) - sum_dy_gamma - (xh_row[c] * sum_dy_gamma_xhat)
//            );
//        }
//    }
//}
//

//Tensor *layer_norm_forward(LayerNorm *ln, Tensor *t) {
//	int rows = t->shape[0];
//	int cols = t->shape[1];
//	if (ln->var != NULL) free(ln->var);
//	ln->var = malloc(rows * sizeof(float));
//
//	for (int i = 0; i < rows; i++) {
//		float *row = t->data + (i * cols);
//		
//		// row mean
//		float row_mean = mean(row, cols);
//
//		// row variance
//		float var_sum = 0.0f;
//		for (int k = 0; k < cols; k++) {
//			float diff = row[k] - row_mean;
//			var_sum += (diff * diff);
//		}
//		float var_mean = var_sum / (float) cols;
//		
//		ln->var[i] = var_mean;
//
//		for (int c = 0; c < cols; c++) {
//			row[c] = (row[c] - row_mean) / sqrtf(var_mean + EPS);
//		}
//	}
//
//	ln->x_hat = t;
//	t = tensor_scaler_multiplication(t, GEMMA);
//	t = tensor_scaler_addition(t, BETA);
//	return t;
//}

LayerNorm *layer_norm_create(int features) {
	LayerNorm *ln = malloc(sizeof(LayerNorm));
	if (!ln) {
		fprintf(stderr, "Allocation failed\n");
		exit(1);
	}
	ln->features = features;
	int ndim = 2;
	int shape[2] = {1, features};
	ln->beta = tensor_create_weights(ndim, shape);
	ln->gamma = tensor_create_weights(ndim, shape);
	ln->d_gamma = tensor_create_weights(ndim, shape);
	ln->d_beta = tensor_create_weights(ndim, shape);
	ln->x_hat = NULL;
	ln->var = NULL; // forward activations cache initially NULL
	
	return ln;
}

LayerNorm *layer_norm_create_new(Arena *A, int features) {
	LayerNorm *ln = arena_alloc(A, sizeof(LayerNorm));
	//if (!ln) {
	//	fprintf(stderr, "Allocation failed\n");
	//	exit(1);
	//}
	ln->features = features;
	int ndim = 2;
	int *shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = 1;
	shape[1] = features;

	ln->beta = tensor_create_weights_new(A, ndim, shape);
	ln->gamma = tensor_create_weights_new(A, ndim, shape);
	ln->d_gamma = tensor_create_weights_new(A, ndim, shape);
	ln->d_beta = tensor_create_weights_new(A, ndim, shape);
	ln->x_hat = tensor_create_weights_new(A, ndim, shape);
	ln->var = tensor_create_weights_new(A, ndim, shape);
	
	return ln;
}

int main() {
	Arena *A = malloc(sizeof(Arena));
	int SIZE = 1024 * 1024 * 1024;
	arena_init(A, SIZE);
	int ndim = 2;
	int shape[2] = {SEQ_LEN, EMB_DIM};
	Tensor *x = tensor_create_new(A, ndim, shape);
	tensor_randomize(x);
	int features = 32;
	LayerNorm *ln = layer_norm_create_new(A, features);

	Tensor *out = layer_norm_forward(A, ln, x); // x is out MHA output
	printf("shape out: \n");
	tensor_get_2d(out);
	tensor_shape_2d(out);
	return 0;
}

