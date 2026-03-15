// Linear Regression Algorithm in C langauge
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int ITERATIONS = 150;


float loss(float w, float b, int size, float train[][2]) {
	float res = 0.0f;
	for (int i = 0; i < size; i++) {
		float y = (float) train[i][1];
		float y_hat = w * train[i][0] + b;
		float d =  y_hat -  y;
		float loss = d * d;
		res += loss;
	}

	res /=  size;
	return res;
}

float grad_w(float w, float b, int size, float train[][2]) {
	float dw = 0.0f;
	for (int i = 0; i < size; i++) {
		float x = train[i][0];
		float y = train[i][1];
		float y_hat = w * x + b;
		// float d = y_hat - y;
		dw += (y_hat - y) * x;
	}
	return 2.0f * dw / size;
}

float grad_b(float w, float b, int size, float train[][2]) {
	float db = 0.0f;
	for (int i = 0; i < size; i++) {
		float x = train[i][0];
		float y = train[i][1];
		float y_hat = w * x + b;
		// float d = y_hat - y;
		db += (y_hat - y);
	}
	return 2.0f * db / size;
}



int main() {
	// we need to implement y = wx + b
	srand(time(NULL));
	float TRAIN[1000][2]; 
	for (int i = 0; i < 1000; i++) {
		TRAIN[i][0] = i;
		TRAIN[i][1] = (float) 2 * i + (float) rand() / (float) RAND_MAX;
	}
	int SIZE = sizeof(TRAIN) / sizeof(TRAIN[0]);

	float w = (float) rand() / (float) RAND_MAX;
	float b = (float) rand() / (float) RAND_MAX;
	// float lr  = (float) rand() / (float) RAND_MAX;
	float lr = 0.0000001;

	float r = loss(w, b, SIZE,TRAIN);
	float dw = grad_w(w, b, SIZE, TRAIN);
	float db = grad_b(w, b, SIZE, TRAIN);

	for (int i = 0; i < ITERATIONS; i++) {
		w -= lr * grad_w(w, b, SIZE, TRAIN);
		b -= lr * grad_b(w, b, SIZE, TRAIN);
		printf("iteration: %d, loss: %.4f, w: %f, b: %f\n", 
				i, loss(w, b, SIZE, TRAIN), w, b);
	}

	return 0;
}
