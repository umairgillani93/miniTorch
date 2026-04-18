#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"

#define SEQ_LEN 10
#define EMB_DIM 32
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void sgd_optimizer(Tensor *w, Tensor *dw, float lr) {
	// simple SGD Soptmizer 
	// w = w - lr * dw;
	//tensor_shape(w);
	//tensor_shape(dw);
	assert(w->shape != dw->shape);
	int size = tensor_size(w);
	for (int i = 0; i < size; i++) {
		w->data[i] = w->data[i] - lr * dw->data[i];
	}
}


//bool is_exploding(Tensor *x) {
//	int size = tensor_size(x);
//	for (int i = 0; i < size; i++) {
//		float v = x->data[i];
//		if (isnan(v) || isinf(v)) {
//			return true;
//		}
//	}
//	return false;
//}

//void clip_gradient(Tensor *x) {
//    int size = tensor_size(x);
//    float threshold = 1.0f;
//    float MX = 0.0f;
//    bool has_bad = false;
//
//    // Pass 1 — detect NaN/Inf and find max abs gradient
//    for (int i = 0; i < size; i++) {
//        float g = x->data[i];
//
//        if (!isfinite(g)) {
//            has_bad = true;
//            break;
//        }
//
//        float v = fabsf(g);
//        if (v > MX) MX = v;
//    }
//
//    // If NaN/Inf found → zero gradients and STOP
//    if (has_bad) {
//        for (int i = 0; i < size; i++)
//            x->data[i] = 0.0f;
//        return;
//    }
//
//    // Pass 2 — clip if too large
//    if (MX > threshold) {
//        float scale = threshold / MX;   // compute once
//        for (int i = 0; i < size; i++)
//            x->data[i] *= scale;
//    }
//}
	
Tensor *ffn_backward(Arena *A, FFN *f, Tensor *x, Tensor *dout) {

	f->dw2 = tensor_matmul(A, tensor_transpose(f->a1), dout);
	f->da1 = tensor_matmul(A, dout, tensor_transpose(f->w2));
	f->dh1 = relu_backward(f->da1, f->h1); 
	f->dw1 = tensor_matmul(A, tensor_transpose(x), f->dh1);
	Tensor *dx = tensor_matmul(A, f->dh1, tensor_transpose(f->w1));

	return dx;
}

FFN *ffn_create(Arena *A, int input_dim, int hidden_dim) {
	int ndim = 2;
	int *shape1 = arena_alloc(A, ndim * sizeof(int));
	int *shape2 = arena_alloc(A, ndim * sizeof(int));

	shape1[0] = input_dim;
	shape1[1] = hidden_dim;

	shape2[0] = hidden_dim;
	shape2[1] = input_dim;


	FFN *f = arena_alloc(A, sizeof(FFN));
	f->w1 = tensor_create_weights_new(A, ndim, shape1);
	f->w2 = tensor_create_weights_new(A, ndim, shape2);
	//f->inputs = tensor_create_new(A, ndim, shape1);

	return f;
}



Tensor *ffn_forward(Arena *A, Tensor *x, FFN *f) {
	assert(x->shape[1] == f->w1->shape[0]);
	if (f->save_inputs == true) {
		f->inputs = x;
	}
	Tensor *h1 = tensor_matmul(A, x, f->w1);
	f->h1 = h1;
	f->a1 = relu(f->h1);
	assert(f->a1->shape[1] == f->w2->shape[0]);
	f->out = tensor_matmul(A, f->a1, f->w2);
	

	return f->out;
}	


Tensor *relu(Tensor *x) {
	int size = x->shape[0] * x->shape[1];
	for (int i = 0; i < size; i++) {
		float val = MAX(0, (float) x->data[i]);
		x->data[i] = val;
	}
	return x;
}

Tensor *forward(Arena *A, Tensor *x) {
	int shape1[2] = {32, 128};
	int shape2[2] = {128, 32};
	Tensor *w1 = tensor_create_weights(2, shape1);
	Tensor *w2 = tensor_create_weights(2, shape2);
	Tensor *h1 = tensor_matmul(A, x, w1);
	Tensor *a1 = relu(h1);
	Tensor *out = tensor_matmul(A, a1, w2);

	return out;
}	

//int main() {
//	int BACH_SIZE = 10;
//	int EPOCHS = 2;
//	for (int i = 0; i < EPOCHS; i++) {
//		for (int j = 0; j < BACH_SIZE; j++) {
//			// .. Rest of logic comes here
//		}
//	}
//	int ndim = 2;
//	int *shape_tokens = malloc(ndim * sizeof(int));
//	int *shape_weights= malloc(ndim * sizeof(int));
//
//	shape_tokens[0] = SEQ_LEN;
//	shape_tokens[1] = EMB_DIM;
//
//	shape_weights[0] = EMB_DIM;
//	shape_weights[1] = EMB_DIM;
//
//	int num_heads = 8;
//
//	// define token tensors
//	Tensor *tokens = tensor_create(ndim, shape_tokens);
//
//
//	// define FFN weights
//	FFN *f = ffn_create(32, 128);
//	//ffn_backward(f);
//
//	int heads = 8;
//	MHA *mha = mha_create(heads, SEQ_LEN, EMB_DIM);
//	Tensor *score = mha_forward(tokens, mha);
//	Tensor *ln1 = layer_norm(score);
//	Tensor *res = ffn_forward(ln1, f);
//	Tensor *pred = layer_norm(res);
//	
//	// Backward pass functions start here
//	Tensor *target = tensor_create(ndim, shape_tokens);
//	Tensor *loss = tensor_mse_loss(pred, target);
//	Tensor *final = ffn_backward(f, tokens, loss);
//
//	tensor_shape(final);
//	tensor_shape(tokens);
//	tensor_shape(mha->out);
//
//	Tensor *mha_back = mha_backward(mha, final, tokens);
//	tensor_get(mha_back);
//
//	return 0;
//}
