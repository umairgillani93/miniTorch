#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "attention2.h"


int main() {

	//int arr[10] = {1,2,3,4,5,6,7,8,9,10};
	//int size = 10;
	//int stride = 2;
	//for (int i = 0; i < size/stride; i++) {
	//	int *idx = arr + i * stride;
	//	for (int j = 0; j < stride; j++) {
	//		printf("%d ", idx[j]);
	//	}
	//	printf("\n");
	//}

	int ndim = 2;
	int seq_len = 100;
	int emb_dim = 32;
	int shape[2] = {seq_len, emb_dim};
	int stride = emb_dim * 2;

	Tensor *x = tensor_create(ndim, shape);
	Tensor *y = tensor_create(ndim, shape);

	sgd_optimizer(

	int size = tensor_size(t);

	return 0;
}

