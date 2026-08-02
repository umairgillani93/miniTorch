#ifndef AVX2_H
#define AVX2_H


void _avx2_matmul(float *x_data, float *y_data, float *out_data,
									int x_rows, int x_cols, int y_cols);


#endif
