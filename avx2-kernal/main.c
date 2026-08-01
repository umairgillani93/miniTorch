#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <tensor.h>
#include <arena.h>
#include <config.h>


int main() {
	srand(time(NULL));
	Arena *A = malloc(sizeof(Arena));
	size_t S = 1024 * 1024 * 1024;
	arena_init(A, S);
	printf("Arena initialized ..\n");
	int ndim  = 2;
	int *shape_a = arena_alloc(A, ndim * sizeof(int));
	int *shape_b = arena_alloc(A, ndim * sizeof(int));
	shape_a[0] = 16;
	shape_a[1] = 32;

	// FIX: Changed shape_b to 32x32 to match your inner K loop bounds and 8-wide AVX step
	shape_b[0] = 32;
	shape_b[1] = 16;


	// output tensor to store results
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	// FIX: Must define output matrix dimensions (16 rows, 32 columns)
	out_shape[0] = 16;
	out_shape[1] = 16;


	Tensor *x = tensor_create_new(A,ndim,shape_a);
	Tensor *y = tensor_create_new(A,ndim,shape_b);
	tensor_randomize_weights(x);
	tensor_randomize_weights(y);

	Tensor *out = tensor_create_new(A, ndim, out_shape);
	tensor_randomize(out);


	//for (int r = 0; r < 16; r++) {
	//	for (int c = 0; c < 32; c++) {
	//		float sum = 0.0f;
	//		// k til range 'a' cols
	//		for (int k = 0; k < 32; k++) {
	//			sum += (x->data[r * 32 + k]) * 
	//				(y->data[k * 16 + c]);
	//		}
	//		out->data[r * 16 + c]  = sum;
	//	}
	//}

	// initialize the avx2 registers with float zero values
	float *x_ptr = x->data;
	float *y_ptr = y->data;
	float *out_ptr = out->data;
	int range = 32/8;

	int offset = 0;
	for (int r = 0; r < range; r++) {
		__m256 a = _mm256_loadu_ps(x_ptr + offset);
		__m256 b = _mm256_loadu_ps(y_ptr + offset);
		__m256 temp = _mm256_add_ps(a, b);
		_mm256_storeu_ps(out_ptr + offset, temp);
		offset += 8;
	}


	for (int r = 0; r < 16; r++) {
		for (int c = 0; c < 16; c++) {
			printf("%0.2f ", out->data[r * 16 + c]); // FIX: Index formula & pointer reference
		}
		printf("\n");
	}
	tensor_shape_2d(out);
	return 0;
}
