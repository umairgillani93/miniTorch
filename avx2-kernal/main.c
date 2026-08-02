#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <tensor.h>
#include <arena.h>
#include <config.h>

void _avx2_matmul(float *x_data, float *y_data, float *out_data, 
						      int x_rows, int x_cols, y_cols) {
	
	float *Aptr = x->data;
	float *BTptr = y->data;
	float *Cptr = out->data;

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

//int main() {
//	srand(time(NULL));
//	Arena *A = malloc(sizeof(Arena));
//	size_t S = 1024 * 1024 * 1024;
//	arena_init(A, S);
//	printf("Arena initialized ..\n");
//	int ndim  = 2;
//	int *shape_a = arena_alloc(A, ndim * sizeof(int));
//	int *shape_b = arena_alloc(A, ndim * sizeof(int));
//	shape_a[0] = 16;
//	shape_a[1] = 32;
//
//	// FIX: Changed shape_b to 32x32 to match your inner K loop bounds and 8-wide AVX step
//	shape_b[0] = 32;
//	shape_b[1] = 16;
//
//
//	// output tensor to store results
//	int *out_shape = arena_alloc(A, ndim * sizeof(int));
//	// FIX: Must define output matrix dimensions (16 rows, 32 columns)
//	out_shape[0] = 16;
//	out_shape[1] = 16;
//
//
//	Tensor *x = tensor_create_new(A,ndim,shape_a);
//	Tensor *y = tensor_create_new(A,ndim,shape_b);
//	tensor_randomize_weights(x);
//	tensor_randomize_weights(y);
//
//	Tensor *out= tensor_create_new(A, ndim, out_shape);
//	tensor_randomize(out);
//
//
//	//for (int r = 0; r < 16; r++) {
//	//	for (int c = 0; c < 32; c++) {
//	//		float sum = 0.0f;
//	//		// k til range 'a' cols
//	//		for (int k = 0; k < 32; k++) {
//	//			sum += (x->data[r * 32 + k]) * 
//	//				(y->data[k * 16 + c]);
//	//		}
//	//		out->data[r * 16 + c]  = sum;
//	//	}
//	//}
//
//	// initialize the avx2 registers with float zero values
//	
//	float *Aptr = x->data;
//	float *BTptr = y->data;
//	float *Cptr = out->data;
//
//	int lda = 32;
//	int ldb = 32;
//	int ldc = 16;
//
//	for (int i = 0; i < 16; i++) {
//		const float *a_row = Aptr + i * lda;
//		float *c_row = Cptr + i * ldc;
//
//		for (int j = 0; j < 16; j++) {
//
//				const float *a = a_row;
//				const float *b = BTptr + j * ldb;
//
//				__m256 acc = _mm256_setzero_ps();
//				for (int k = 0; k < 32; k += 8) {
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
//	tensor_get_2d(out);
//	tensor_shape_2d(out);
//	return 0;
//}
