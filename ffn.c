#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"
#include "arena.h"
#include "config.h"

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
	
void ffn_init_params(FFN *F) {
	tensor_randomize_weights(F->w1);
	tensor_randomize_weights(F->w2);
}

Tensor *ffn_backward(Arena *A, FFN *f, Tensor *x, Tensor *dout) {

	f->dw2 = tensor_matmul(A, tensor_transpose(A, f->a1), dout);
	f->da1 = tensor_matmul(A, dout, tensor_transpose(A, f->w2));
	f->dh1 = tensor_relu(A, f->da1); 
	f->dw1 = tensor_matmul(A, tensor_transpose(A, x), f->dh1);
	Tensor *dx = tensor_matmul(A, f->dh1, tensor_transpose(A, f->w1));

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
	f->a1 = tensor_relu(A, f->h1);
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
//
//	Arena *A = malloc(sizeof(Arena));
//	int SIZE = 1024 * 1024 * 1024;
//	arena_init(A, SIZE);
//	int ndim = 2;
//	int shape[2] = {SEQ_LEN, EMB_DIM};
//	Tensor *x = tensor_create_new(A, ndim, shape);
//	tensor_randomize_weights(x);
//	//x->requires_grad = true;
//	int features = 32;
//
//	int num_heads = 8;
//	MHA *mha = mha_create_new(A, num_heads, SEQ_LEN, EMB_DIM);
//	mha_init_params(mha);
//	Tensor *out = mha_forward(A, x, mha);
//	////printf("MHA DATA: \n");
//	////tensor_get_2d(out);
//
//	////printf("MAH visited Graph: \n");
//	////run_graph_validation(A, out, MAX_NODES);
//	////printf("----------------------------------------\n");
//	//
//	LayerNorm *ln1 = layer_norm_create_new(A, features);
//	printf("Iniitiazing parameters for Layer Norm:\n");
//	layer_norm_init_params(ln1);
//	Tensor *ln_out= layer_norm_forward(A, ln1, out); // x is out MHA output
//	
//	FFN *f = ffn_create(A, EMB_DIM, HIDDEN_DIM);
//	ffn_init_params(f);
//	Tensor *ffn_out = ffn_forward(A, ln_out, f);
//	tensor_get_2d(out);
//
//	LayerNorm *ln2 = layer_norm_create_new(A, features);
//	printf("Iniitiazing parameters for Layer Norm 2:\n");
//	layer_norm_init_params(ln2);
//	Tensor *ln2_out= layer_norm_forward(A, ln2, ffn_out); // x is out MHA output
//	
//	tensor_shape_2d(ln2_out);
//	//bool *visited  = vislist();
//	//export_and_visualize_graph(ffn_out, "graph_ffn.dot", "graph_ffn.png");
//
//	run_graph_validation(A, ffn_out, MAX_NODES);
//	//free(A);
//
//	return 0;
//}
