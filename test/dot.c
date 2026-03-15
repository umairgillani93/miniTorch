#include <stdio.h>
#include <stdlib.h>

// Matrix structure
struct Matrix {
	int rows;
	int cols;
	int size;
	int *data;
};


int dot(const int *a, int *b, const int size) {
	int sum = 0;
	for (int i = 0; i < size; i++) {
		sum += a[i] *b[i];
	}
	return sum;
}


struct Matrix *transpose(struct Matrix *matrix) {
	struct Matrix *res = malloc(sizeof(struct Matrix));
	res->rows = matrix->cols;
	res->cols = matrix->rows;
	res->size = res->rows * res->cols;
	res->data = malloc(res->size * sizeof(int));
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->cols; j++) {
			res->data[j * res->rows + i] = matrix->data[i * matrix->cols + j];
		}
	}
	return res;
}



int main() {
	struct Matrix *A = malloc(sizeof(struct Matrix));

	A->rows = 3;
	A->cols = 4;
	A->size = A->rows  * A->cols;
	A->data = malloc(A->size * sizeof(int));


	// Initilize the matrices
	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < A->cols; j++) {
			A->data[i * A->cols + j] = (i + 1) * A->cols + j;
		}
	}

	struct Matrix *final = transpose(A);

	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < A->cols; j++) {
			printf("%d ", A->data[i * A->cols + j]);
		}
		printf("\n");
	}

	printf("-------------------\n");

	for (int i = 0; i < final->rows; i++) {
		for (int j = 0; j < final->cols; j++) {
			printf("%d ", final->data[i * final->cols + j]);
		}
		printf("\n");
	}

	//for (int i = 0; i < A->rows; i++) {
	//	for (int j = 0; j < A->cols; j++) {
	//		printf("%d ", A->data[i * A->cols + j]);
	//	}
	//	printf("\n");
	//}
	
	return 0;
}

