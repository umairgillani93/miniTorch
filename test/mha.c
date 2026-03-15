#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define RAND_FLOAT  (float) rand() / (float) RAND_MAX

// Matrix structure
struct Matrix {
	int rows;
	int cols;
	int size;
	float *data;
};

struct Tensor {
	int rows;
	int cols;
	int size;
	float *data;
	int dim; // dimention / rank of a Tensor
	char *dtype;
	char *device;
};

//struct Matrix *transpose(struct Matrix *matrix) {
//	struct Matrix *res = malloc(sizeof(struct Matrix));
//	res->rows = matrix->cols;
//	res->cols = matrix->rows;
//	res->size = res->rows * res->cols;
//	res->data = malloc(res->size * sizeof(int));
//	for (int i = 0; i < matrix->rows; i++) {
//		for (int j = 0; j < matrix->cols; j++) {
//			res->data[j * res->rows + i] = matrix->data[i * matrix->cols + j];
//		}
//	}
//	return res;
//}


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

void softmax(struct Matrix *m) {
	for (int i = 0; i < m->rows; i++) {
		float res = 0.0f;
		for (int j = 0; j < m->cols; j++) {
			m->data[i * m->cols + j] = expf(m->data[i * m->cols +j]);
			res += m->data[i * m->cols +j];
		}
		for (int k = 0; k < m->cols; k++) {
			m->data[i * m->cols +k] = m->data[i * m->cols +k] / res;
		}
	}
}

void dk_square_root(struct Matrix *m, float dk) {

	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->data[i * m->cols + j] = m->data[i * m->cols + j] / dk;
		}
	}
}

void shape(struct Matrix *m) {
	printf("(%d, %d)\n",  m->rows,  m->cols);
}

void display(struct Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%f ", m->data[i * m->cols + j]);
        }
				printf("\n");
    }
}

struct Matrix *multi_head_attention(struct Matrix **arr, int SIZE) {
    int total_cols = 0;
    for (int i = 0; i < SIZE; i++) {
        total_cols += arr[i]->cols;
    }
		//prinft("total cols: %d\n", total_cols);

    struct Matrix *res = malloc(sizeof(struct Matrix));
    struct Matrix *first = arr[0];
		int curr_col = first->cols;

    res->rows = first->cols;
    res->cols = total_cols;
    res->size = first->rows * total_cols;
    res->data = malloc(res->size * sizeof(float));

		// Glue first head manually here
		for (int i = 0; i < first->rows; i++) {
			for (int j = 0; j < first->cols; j++) {
				res->data[i * res->cols + j] = first->data[i * first->cols + j];
			}
		}

    for (int k = 1; k < SIZE; k++) {
        struct Matrix *b = arr[k];
        for (int i = 0; i < b->rows; i++) {
            for (int j = 0; j < b->cols; j++) {
                res->data[i * res->cols + (curr_col + j)] = b->data[i * b->cols + j];
            }
        }
        curr_col += b->cols; 
    }
    return res;
}

struct Matrix *self_attention(struct Matrix *Q, struct Matrix *K, struct Matrix *V, 
	 														float dk) {

	// Source: Attention is all you need paper from google -> 2017!
	struct Matrix *qk = matmul(Q, K);
	dk_square_root(qk, dk);
	softmax(qk);
	struct Matrix *result = matmul(V, qk);
	return result;
		
}


int main() {
	srand(time(NULL));
	float e = 2.718;

	int rows = 2;
	int cols = 2;
	int size = rows * cols;

	struct Matrix *Q1= malloc(sizeof(struct Matrix));
	struct Matrix *K1= malloc(sizeof(struct Matrix));

	Q1->rows = rows;
	Q1->cols = cols;
	Q1->size = size;
	Q1->data = malloc(Q1->size * sizeof(float));

	K1->rows = rows;
	K1->cols = cols;
	K1->size = size;
	K1->data = malloc(K1->size * sizeof(float));
	
	struct Matrix *Q2= malloc(sizeof(struct Matrix));
	struct Matrix *K2= malloc(sizeof(struct Matrix));

	Q2->rows = rows;
	Q2->cols = cols;
	Q2->size = size;
	Q2->data = malloc(Q2->size * sizeof(float));

	K2->rows = rows;
	K2->cols = cols;
	K2->size = size;
	K2->data = malloc(K2->size * sizeof(float));

	for (int i = 0; i < Q1->rows; i++) {
		for (int j = 0; j < Q1->cols; j++) {
			Q1->data[i * Q1->cols + j] = RAND_FLOAT;
		}
	}

	for (int i = 0; i < K1->rows; i++) {
		for (int j = 0; j < K1->cols; j++) {
			K1->data[i * K1->cols + j] = RAND_FLOAT;
		}
	}

	for (int i = 0; i < Q2->rows; i++) {
		for (int j = 0; j < Q2->cols; j++) {
			Q2->data[i * Q2->cols + j] = RAND_FLOAT;
		}
	}

	for (int i = 0; i < K2->rows; i++) {
		for (int j = 0; j < K2->cols; j++) {
			K2->data[i * K2->cols + j] = RAND_FLOAT;
		}
	}
	int num_heads = 8;
	float d_model = 768.0f;
	float dk = d_model / (float) num_heads;

	// declare the value matrix
	struct Matrix *V1 = malloc(sizeof(struct Matrix));
	V1->rows = rows;
	V1->cols = cols;
	V1->size = size;
	V1->data = malloc(V1->size * sizeof(float));


	// Initialize matrix V
	for (int i = 0; i < V1->rows; i++) {
		for (int j = 0; j < V1->cols; j++) {
			V1->data[i * V1->cols + j] = RAND_FLOAT;
		}
	}


	struct Matrix *V2 = malloc(sizeof(struct Matrix));
	V2->rows = rows;
	V2->cols = cols;
	V2->size = size;
	V2->data = malloc(V2->size * sizeof(float));


	// Initialize matrix V
	for (int i = 0; i < V2->rows; i++) {
		for (int j = 0; j < V2->cols; j++) {
			V2->data[i * V2->cols + j] = RAND_FLOAT;
		}
	}

	struct Matrix *head1 = self_attention(Q1, K1, V1, dk); 
	struct Matrix *head2 = self_attention(Q2, K2, V2, dk); 

	struct Matrix *arr[2];
	arr[0] = head1;
	arr[1] = head2;

	int SIZE = 2;
	struct Matrix *final = multi_head_attention(arr, SIZE);

	display(final);

	

	return 0;
}

//struct Matrix *concatenate_heads(struct Matrix **heads, int num_heads) {
//    int seq_len = heads[0]->rows;
//    int dk = heads[0]->cols;
//    int d_model = dk * num_heads;
//
//    struct Matrix *res = malloc(sizeof(struct Matrix));
//    res->rows = seq_len;
//    res->cols = d_model;
//    res->data = malloc(seq_len * d_model * sizeof(float));
//
//    for (int i = 0; i < seq_len; i++) { // For each word
//        for (int h = 0; h < num_heads; h++) { // For each head
//            for (int j = 0; j < dk; j++) { // For each feature in that head
//                // This logic "glues" the heads side-by-side
//                res->data[i * d_model + (h * dk + j)] = heads[h]->data[i * dk + j];
//            }
//        }
//    }
//    return res;
//}

