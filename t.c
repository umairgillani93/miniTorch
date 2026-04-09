#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"
#include "config.h"


float mean_new(float *row, int size) {
	float row_sum = 0.0f;
	for (int i = 0; i < size; i++) {
		row_sum += row[i];
	}
	return row_sum / (float) size;
	
}

Tensor *forward(LayerNorm *ln, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	// Till now we have calucalated 
	// mean / row, variance / row, defined x_hat, defined output y, 
	// append variance against the entry in ln i.e ln->var[i] = vf
	// Now we need to populate actual xh_row_
	// xh_row = x_row[i] - mean * inv_std;

	Tensor *y = tensor_create(2, x->shape);
	if (ln->x_hat) tensor_free(ln->x_hat);
	ln->x_hat = tensor_create(2, x->shape);

	if (ln->var) free(ln->var);
	ln->var = malloc(rows * sizeof(float));


	for (int i = 0; i < rows; i++) {
		float *x_row = x->data + (i * cols);
		float *xh_row = ln->x_hat->data + (i * cols);
		float *y_row = y->data + (i * cols);

		// calcuate row mean first
		float mu = mean_new(x_row, cols);
		
		float var_sum = 0.0f;
		for (int k = 0; k < cols; k++)  {
			float v = x_row[k] - mu;
			var_sum += v * v;
		}
		float vf = var_sum / cols;
		// save this variance against the entry in ln->var which is off size rows
		ln->var[i] = vf;

		float inv_std = 1.0f / sqrtf(vf + EPS);
		// populate x_hat
		for (int c = 0; c < cols; c++) {
			float x_hat = (x_row[c] - mu) * inv_std;
			xh_row[c] = x_hat;


			// populate gamman and beta too
			float gamma = ln->gamma->data[c];
			float beta = ln->beta->data[c];

			y_row[c] = gamma * x_hat + beta;
		}
	}
	return y;
}

int main() {
	LayerNorm *ln = layer_norm_create(EMB_DIM);
	int shape[2] = {SEQ_LEN, EMB_DIM};
	Tensor *t = tensor_create(2, shape);
	Tensor *out = forward(ln, t);
	tensor_shape(out);
	return 0;

}

