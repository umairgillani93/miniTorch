#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define RAND_FLOAT (float) rand() / (float) RAND_MAX;

// Matrix structure
struct Matrix {
	int rows;
	int cols;
	int size;
	float *data;
};

struct Matrix *matmul(struct Matrix *matrix1, struct Matrix *matrix2) {


	// Resultant Matrix
	struct Matrix *result = malloc(sizeof(struct Matrix));
	result -> rows = matrix1 -> rows;
	result -> cols = matrix2 -> cols;
	result -> size = matrix1 -> rows * matrix2 -> cols;
	result -> data = malloc(result -> size * sizeof(float));

	// Multiply the matrices
	for (int i = 0; i < matrix1->rows; i++) {
		for (int j = 0; j < matrix2->cols; j++) {
			float sum = 0.0f;
			for (int k = 0; k < matrix1->cols; k++) {
				// Remember 'k' is shared dimention for multplication of matrix1 and matrix2
				sum += matrix1->data[i * matrix1->cols + k] * matrix2->data[k * matrix2->cols + j];
			}
			result -> data[i * result-> cols + j] = (float) sum;
		}
	}
	return result;
}


int main() {
	// initilize structure pointer to the Matrix and allocate it's memory
	struct Matrix *matrix1 = malloc(sizeof(struct Matrix));
	struct Matrix *matrix2 = malloc(sizeof(struct Matrix));
	
	// define matrix data
	matrix1->rows = 3;
	matrix1->cols = 1;
	matrix1->size = matrix1->rows * matrix1->cols;
	matrix1->data = malloc(matrix1->size * sizeof(float));

	matrix2->rows = 1;
	matrix2->cols = 4;
	matrix2->size = matrix2->rows * matrix2->cols;
	matrix2->data = malloc(matrix2->size * sizeof(float));
	
	// assign values to the matrices
	for (int i = 0; i < matrix1->rows; i++) {
		for (int j = 0; j < matrix1->cols; j++) {
			matrix1->data[i * matrix1->cols + j] = RAND_FLOAT;
		}
	}

	for (int i = 0; i < matrix2->rows; i++) {
		for (int j = 0; j < matrix2->cols; j++) {
			matrix2->data[i * matrix2->cols + j] = RAND_FLOAT;
		}
	}

	// dimentionality check
	assert(matrix1->cols == matrix2->rows);

	struct Matrix *res = matmul(matrix1, matrix2);
	for (int i = 0; i < res -> rows; i++) {
		for (int j = 0; j < res -> cols; j++) {
			printf("%f ",  res->data[i * res -> cols + j]);
		}
		printf("\n");
	}
	printf("All good\n");
	return 0;
}

