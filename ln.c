#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "attention2.h"
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

Tensor *layer_norm_forward(LayerNorm *ln, Tensor *x) {

    int rows = x->shape[0];
    int cols = x->shape[1];

    if (ln->var) free(ln->var);
    ln->var = malloc(rows * sizeof(float));

    if (ln->x_hat) tensor_free(ln->x_hat);
    ln->x_hat = tensor_create(2, x->shape);

    // output tensor y
    Tensor *y = tensor_create(2, x->shape);

    for (int i = 0; i < rows; i++) {

        float *x_row   = x->data + i * cols;
        float *xh_row  = ln->x_hat->data + i * cols;
        float *y_row   = y->data + i * cols;

        float mu = mean(x_row, cols);

        float var_sum = 0.0f;
        for (int k = 0; k < cols; k++) {
            float diff = x_row[k] - mu;
            var_sum += diff * diff;
        }

        float var = var_sum / cols;
        ln->var[i] = var;
        float inv_std = 1.0f / sqrtf(var + EPS);

        for (int c = 0; c < cols; c++) {

            // x_hat = normalized value (cache!)
            float x_hat = (x_row[c] - mu) * inv_std;
            xh_row[c] = x_hat;

            // y = gamma * x_hat + beta
            float gamma = ln->gamma->data[c];
            float beta  = ln->beta->data[c];

            y_row[c] = gamma * x_hat + beta;
        }
    }

    return y;
}

// MHA -> LAYER NORM -> FFN -> LAYER NORM -> Loss
// Loss -> LAYER NORM -> FFN -> LAYER NORM -> MHA
// Loss(ln2, target) -> LAYER NORM(L2, ffn) -> FFN(ln1, F) -> LAYER NORM(L1, attn) -> input(X)

void layer_norm_backward(LayerNorm *ln, Tensor *x, Tensor *dy, Tensor *dx, float lr) {
    int rows = x->shape[0];
    int cols = x->shape[1];

    // 1. Calculate gradients for gamma and beta (accumulate over rows)
    for (int i = 0; i < rows; i++) {
        for (int c = 0; c < cols; c++) {
            float dy_val = dy->data[i * cols + c];
            float xh_val = ln->x_hat->data[i * cols + c];

            ln->d_gamma->data[c] += dy_val * xh_val;
            ln->d_beta->data[c]  += dy_val;
        }
    }

    // 2. Calculate gradient for input x
    for (int i = 0; i < rows; i++) {
        float *dy_row = dy->data + i * cols;
        float *xh_row = ln->x_hat->data + i * cols;
        float *dx_row = dx->data + i * cols;
        
        float var = ln->var[i];
        float inv_std = 1.0f / sqrtf(var + EPS);

        // Intermediate terms for the simplified LayerNorm gradient formula
        float sum_dy_gamma = 0.0f;
        float sum_dy_gamma_xhat = 0.0f;

        for (int c = 0; c < cols; c++) {
            float dy_gamma = dy_row[c] * ln->gamma->data[c];
            sum_dy_gamma += dy_gamma;
            sum_dy_gamma_xhat += dy_gamma * xh_row[c];
        }

        // The "one-liner" derivative formula for LayerNorm:
        // dx = (1/N) * inv_std * [N*dy_gamma - sum(dy_gamma) - x_hat*sum(dy_gamma*x_hat)]
        for (int c = 0; c < cols; c++) {
            float dy_gamma = dy_row[c] * ln->gamma->data[c];
            dx_row[c] = (1.0f / cols) * inv_std * (
                (cols * dy_gamma) - sum_dy_gamma - (xh_row[c] * sum_dy_gamma_xhat)
            );
        }
    }
}


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

	ln->x_hat = NULL;
	ln->var = NULL; // forward activations cache initially NULL
	
	return ln;
}

//int main() {
//	int ndim = 2;
//	int *shape_tokens = malloc(ndim * sizeof(int));
//	
//	shape_tokens[0] = SEQ_LEN;
//	shape_tokens[1] = EMB_DIM;
//
//	LayerNorm *ln = layer_norm_create(EMB_DIM);
//	printf("before:\n");
//	printf("%p\n", ln->x_hat);
//	
//	printf("created layernorm\n");
//	Tensor *tokens = tensor_create(ndim, shape_tokens);
//
//	int *shape_weights = malloc(ndim * sizeof(int));
//
//	shape_weights[0] = EMB_DIM;
//	shape_weights[1] = EMB_DIM;
//
//	int heads = 8;
//
//	MHA *mha = mha_create(heads, SEQ_LEN, EMB_DIM);
//	Tensor *score = mha_forward(tokens, mha);
//	Tensor *t = layer_norm_forward(ln, score);
//	printf("after:\n");
//	tensor_shape(ln->x_hat);
//	printf("success!\n");
//	tensor_shape(t);
//	tensor_get(t);
//	return 0;
//}

