#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "attention2.h"

int main() {

	int ndim = 2;
	int seq_len = 100;
	int emb_dim = 32;
	int shape_x[2] = {seq_len, emb_dim};
	int shape_y[2] = {emb_dim, seq_len};
	int stride = emb_dim * 2;

	Tensor *x = tensor_create(ndim, shape_x);
	Tensor *y = tensor_create(ndim, shape_x);

	int shape_z[2] = {seq_len, seq_len};
	Tensor *z = tensor_create(ndim, shape_z); // since matrices are square so shape
																					// of z will be same as x and y


	for (int r = 0; r < seq_len; r++) {
		for (int c = 0; c < seq_len; c++) {
			float sum = 0.0f;
			for (int k = 0; k < emb_dim; k++) {
				sum += (x->data[r * emb_dim + k] * y->data[k * seq_len + c]);
			}
			z->data[r * seq_len + c] = sum;
		}
	}

	// Slice rows
	int r_chunk = 10;

	for (int r = 0; r < r_chunk; r++) {
		int *row_ptr = z->data + (r * seq_len);
		for (int k = 0; k < seq_len; k++) {
			printf("%d ", row_ptr[k]);
		}
		printf("---------------------------\n");
	}

	// Slice columns
	int c_chunk = 10;
	for (int r = 0; r < seq_len; r++) {
		for (int c = 0; c < c_chunk; c++) {
			int *col_ptr = z->data + (r * c_chunk);
		}
	}
	return 0;
}

