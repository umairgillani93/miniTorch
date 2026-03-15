#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

struct Matrix{
	int x;
	int y;
	int size;
	float *data;
};

// matirx multiplication function
struct Matrix matmul(struct Matrix *first, struct Matrix *second) {
	// define rows and columns
	int first_rows = first->x;
	int first_cols = first->y;
	int second_rows = second->x;
	int second_cols = second->y;
	
	struct Matrix final;
	final.x = first_rows;
	final.y = second_cols;
	final.size = first_rows * second_cols;
	final.data = malloc(final.size * sizeof(float));
	

	for (int i = 0; i < first_rows; i++) {
		for (int j = 0; j < second_cols; j++) {
			float sum = 0.0f;
			// Multiplication for shared index
			for (int k = 0; k < first_cols; k++) {
				sum += first->data[i * first_cols + k] * second->data[k * second_cols + j];
			}
			final.data[i * first_cols + j] = sum;
		}
	}
	return final;
	
}

int main() {
	int rows = 2;
	int cols = 2;
	// int size = 2;

	struct Matrix *matrix1 = malloc(sizeof(struct Matrix));
	struct Matrix *matrix2 = malloc(sizeof(struct Matrix));
	matrix1->x = rows;
	matrix1->y = cols;
	matrix1->size = rows * cols;
	matrix1->data = malloc(rows * cols * sizeof(float));

	matrix2->x = rows;
	matrix2->y = cols;
	matrix2->size = rows * cols;
	matrix2->data = malloc(rows * cols * sizeof(float));
	
	// Initialize the matrices
	for (int i = 0; i < matrix1->x; i++) {
		for (int j = 0; j < matrix2->y; j++) {
			matrix1->data[i * matrix1->y + j] = (float) i * 2;
		}
	}

	for (int i = 0; i < matrix2->x; i++) {
		for (int j = 0; j < matrix2->y; j++) {
			matrix2->data[i * matrix2->y + j] = (float) i * 2;
		}
	}


	struct Matrix result = matmul(matrix1, matrix2);
	for (int i = 0; i < result.x; i++) {
		for (int j = 0; j < result.y; j++) {
			printf("%f ", result.data[i * result.y + j]); 
		}
		printf("\n");
	}
	printf("All good\n");


	return 0;
}
