#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

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
	return 0;
}



