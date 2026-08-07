#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "avx2.h"

void _avx2_matmul(
    float *x_data,
    float *y_data,
    float *out_data,
    int x_rows,
    int x_cols,
    int y_rows,
    int y_cols
) {
    if (x_cols != y_rows) {
        printf("Dimension mismatch: A cols (%d) != B rows (%d)\n",
               x_cols, y_rows);
        return;
    }

    int lda = x_cols;
    int ldb = y_rows;   // y_data is B^T
    int ldc = y_cols;

    for (int i = 0; i < x_rows; i++) {

        const float *a_row = x_data + i * lda;
        float *c_row = out_data + i * ldc;

        for (int j = 0; j < y_cols; j++) {

            const float *b_row = y_data + j * ldb;

            __m256 acc = _mm256_setzero_ps();

            int k = 0;

            for (; k + 8 <= x_cols; k += 8) {

                __m256 va = _mm256_loadu_ps(a_row + k);
                __m256 vb = _mm256_loadu_ps(b_row + k);

                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(va, vb)
                );
            }

            float tmp[8];
            _mm256_storeu_ps(tmp, acc);

            float sum = 0.0f;

            for (int t = 0; t < 8; t++) {
                sum += tmp[t];
            }

            for (; k < x_cols; k++) {
                sum += a_row[k] * b_row[k];
            }

            c_row[j] = sum;
        }
    }
}

//void _avx2_matmul(float *x_data, float *y_data, float *out_data, 
//						      int x_rows, int x_cols, int y_rows, int y_cols) {
//	
//	float *Aptr = x_data;
//	float *BTptr = y_data;
//	float *Cptr = out_data;
//
//	int lda = x_cols;
//	int ldb = y_rows;
//	int ldc = x_rows;
//	
//	printf("x rows: %d\n", x_rows);
//	printf("x cols: %d\n", x_cols);
//	printf("y rows: %d\n", y_rows);
//	printf("y cols: %d\n", y_cols);
//
//	for (int i = 0; i < x_rows; i++) {
//		const float *a_row = Aptr + i * lda;
//		float *c_row = Cptr + i * ldc;
//
//		for (int j = 0; j < x_rows; j++) {
//
//				const float *a = a_row;
//				const float *b = BTptr + j * ldb;
//
//				__m256 acc = _mm256_setzero_ps();
//				for (int k = 0; k < x_cols; k += 8) {
//
//					__m256 va = _mm256_loadu_ps(a);
//					__m256 vb = _mm256_loadu_ps(b);
//
//					acc = _mm256_add_ps(
//								acc,
//								_mm256_mul_ps(va, vb));
//
//					a += 8;
//					b += 8;
//				}
//				float tmp[8];
//				_mm256_storeu_ps(tmp, acc);
//				float sum = 0.0f;
//				for (int t = 0; t < 8; t++)
//						sum += tmp[t];
//
//				c_row[j] = sum;
//		}
//	}
//}
