#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "avx2.h"

void _avx2_matmul(float *x_data, float *y_data, float *out_data, 
						      int x_rows, int x_cols, int y_cols) {
	
	float *Aptr = x_data;
	float *BTptr = y_data;
	float *Cptr = out_data;

	int lda = 32;
	int ldb = 32;
	int ldc = 16;

	for (int i = 0; i < 16; i++) {
		const float *a_row = Aptr + i * lda;
		float *c_row = Cptr + i * ldc;

		for (int j = 0; j < 16; j++) {

				const float *a = a_row;
				const float *b = BTptr + j * ldb;

				__m256 acc = _mm256_setzero_ps();
				for (int k = 0; k < 32; k += 8) {

					__m256 va = _mm256_loadu_ps(a);
					__m256 vb = _mm256_loadu_ps(b);

					acc = _mm256_add_ps(
								acc,
								_mm256_mul_ps(va, vb));

					a += 8;
					b += 8;
				}
				float tmp[8];
				_mm256_storeu_ps(tmp, acc);
				float sum = 0.0f;
				for (int t = 0; t < 8; t++)
						sum += tmp[t];

				c_row[j] = sum;
		}
	}
}
