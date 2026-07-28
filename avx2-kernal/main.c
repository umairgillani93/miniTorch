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
	shape_b[1] = 32;


	// output tensor to store results
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	// FIX: Must define output matrix dimensions (16 rows, 32 columns)
	out_shape[0] = 16;
	out_shape[1] = 32;
	Tensor *out = tensor_create_new(A, ndim, out_shape);


	Tensor *x = tensor_create_new(A,ndim,shape_a);
	Tensor *y = tensor_create_new(A,ndim,shape_b);
	tensor_randomize_weights(x);
	tensor_randomize_weights(y);


	// here in my case rows = 16, and colms = 32
	// so I'll loop through 32 / 8 number of times with an offset of 8
	// for (int i = 0; i < 32; i += 8) like so 
	// now I need to initialize register
	float *ptr_x = x->data;
	float *ptr_y = y->data;
	float *ptr_out = out->data;
	
	for (int r = 0; r < 16; r++) {
		for (int c = 0; c < 32; c += 8) {


			// initialize accumulator register for inner dimension 'k'
			__m256 acc = _mm256_setzero_ps(); // zero initialized // FIX: __m265 -> __m256
			// looping through inner dimension 
			for (int k = 0; k < 32; k++) {
				// 2. Broadcast a single scalar element from Matrix X: X[i][k]
				// This loads one float and clones it into all 8 slots of the YMM register
					__m256 reg_x_scalar = _mm256_set1_ps(ptr_x[r * 32 + k]); // FIX: i -> r

					// Load a contiguous 8-float block from Matrix Y: Y[k][j ... j+7]
					__m256 reg_y_rowchunk = _mm256_loadu_ps(&ptr_y[k * 32 + c]); // FIX: j -> c

					//  Multiply and Accumulate: acc = acc + (reg_x_scalar * reg_y_rowchunk)
					acc = _mm256_fmadd_ps(reg_x_scalar, reg_y_rowchunk, acc);
				}
			//  Store the final calculated 8-float block back to the output tensor
			_mm256_storeu_ps(&ptr_out[r * 32 + c], acc); // FIX: i -> r, j -> c
			}
		}


	for (int r = 0; r < 16; r++) {
		for (int c = 0; c < 32; c++) {
			printf("%f ", ptr_out[r * 32 + c]); // FIX: Index formula & pointer reference
		}
		printf("\n");
	}
	return 0;
}
