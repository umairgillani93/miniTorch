#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <tensor.h>
#include <arena.h>
#include <config.h>


int main() {
	int A[2][8] = {
		{1,2,3,4,5,6,7,8},
		{13,2,3,282,33,4,4,8} // first 8-bytes array
		};

	int B[2][8] = {
		{1,2,3,4,5,6,7,8},
		{13,2,3,282,33,4,4,8} // second 8-bytes array
		};
	
	int C[2][8]; // contains matmul result // load the arrays to avx2 registers
	for (int i = 0; i < 2; i++) {
		__m256i va = _mm256_loadu_si256((__m256i*)A[i]);
		__m256i vb = _mm256_loadu_si256((__m256i*)B[i]);
		
		__m256i vc = _mm256_mullo_epi32(va, vb);
		_mm256_storeu_si256((__m256i*)C[i], vc);
	}

	for (int r = 0; r < 2; r++) {
		for (int c = 0; c < 8; c++) {
			printf("%d ", C[r][c]);
		}
		printf("\n");
	}

	
	srand(time(NULL));
	Arena *A = malloc(sizeof(Arena));
	size_t S = 1024 * 1024;
	arean_init(A, S);
	int ndim  = 2;
	int *shape = arena_alloc(A, ndim * sizeof(int));
	shape[0] = 1024;
	shape[1] = 1024;


	Tensor *a = tensor_create_new(A,ndim,shape);
	tensor_get_2d(a);
	return 0;
}

