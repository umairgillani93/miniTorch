// Copyright (C) 2026 Umair Gillani
// https://www.linkedin.com/in/umairgillani93
// https://www.github.com/umairgillani93
//
// This file is part of 'miniTorch' and licensed under GPLv3.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "tensor.h"
#include "attention2.h"
#include "layer_norm.h"
#include "feed_forward_nn.h"
#include "config.h"
#include "arena.h"

void optimizer(MHA *m, FFN *f, LayerNorm *l1, LayerNorm *l2, float lr) {
    // 1. Safety check: Ensure the struct pointers themselves aren't NULL
    if (!m || !f || !l1 || !l2) return;

    // Helper macro to safely update a tensor's data using its grad
    // This checks if the tensor, its data, its grad, and its grad->data all exist.
    #define SAFE_UPDATE(t) \
        if ((t) && (t)->data && (t)->grad && (t)->grad->data) { \
            int _sz = tensor_size(t); \
            for (int i = 0; i < _sz; i++) { \
                (t)->data[i] -= lr * (t)->grad->data[i]; \
            } \
        } else { \
            printf("Optimizer Warning: Skipped update for " #t " (NULL detected)\n"); \
        }

    // --- MHA Parameters ---
    SAFE_UPDATE(m->wq);
    SAFE_UPDATE(m->wk);
    SAFE_UPDATE(m->wv);
    SAFE_UPDATE(m->wo);

    // --- FFN Parameters ---
    SAFE_UPDATE(f->w1);
    SAFE_UPDATE(f->w2);
    SAFE_UPDATE(f->h1); // Bias 1
    SAFE_UPDATE(f->a1); // Bias 2

    // --- LayerNorm 1 Parameters ---
    SAFE_UPDATE(l1->beta);
    SAFE_UPDATE(l1->gamma);

    // --- LayerNorm 2 Parameters ---
    SAFE_UPDATE(l2->beta);
    SAFE_UPDATE(l2->gamma);

    #undef SAFE_UPDATE
}

//void optimizer(MHA *m, FFN *f, LayerNorm *l1, LayerNorm *l2, float lr) {
//    // Assuming wq, wk, wv, wo are all (EMB_DIM * EMB_DIM)
//    int m_size = tensor_size(m->wq); 
//    for (int i = 0; i < m_size; i++) {
//        m->wq->data[i] -= lr * m->wq->grad->data[i];
//        m->wk->data[i] -= lr * m->wk->grad->data[i];
//        m->wv->data[i] -= lr * m->wv->grad->data[i];
//        m->wo->data[i] -= lr * m->wo->grad->data[i];
//    }
//    
//    // FFN Weights 
//		int w1_size = f->w1->shape[0] * f->w1->shape[1];
//		int w2_size = f->w2->shape[0] * f->w2->shape[1];
//		int h1_size = f->h1->shape[0] * f->h1->shape[1];
//		int a1_size = f->a1->shape[0] * f->a1->shape[1];
//    for (int i = 0; i < w1_size; i++) f->w1->data[i] -= lr * f->w1->grad->data[i];
//    for (int i = 0; i < w2_size; i++) f->w2->data[i] -= lr * f->w2->grad->data[i];
//    
//    // FFN Biases 
//    for (int i = 0; i < h1_size; i++) f->h1->data[i] -= lr * f->h1->grad->data[i];
//    for (int i = 0; i < a1_size; i++) f->a1->data[i] -= lr * f->a1->grad->data[i];
//
//    // LayerNorm
//    int ln_size = l1->gamma->shape[0] * l1->gamma->shape[1];
//    for (int i = 0; i < ln_size; i++) {
//        l1->beta->data[i]  -= lr * l1->beta->grad->data[i];
//        l1->gamma->data[i] -= lr * l1->gamma->grad->data[i];
//        l2->beta->data[i]  -= lr * l2->beta->grad->data[i];
//        l2->gamma->data[i] -= lr * l2->gamma->grad->data[i];
//    }
//}

int main() {

	srand(time(NULL));

	Arena *A = malloc(sizeof(Arena));
	arena_init(A, ARENA_SIZE);
	printf("Arena initilized\n");

	int ndim = 2;
	int *shape_input = arena_alloc(A, ndim * sizeof(int));
	shape_input[0] = SEQ_LEN;
	shape_input[1] = EMB_DIM;
	
	// Creating actual data tensor
	// this should work
	Tensor *T = tensor_create_new(A, 2, shape_input);
	tensor_randomize_weights(T); 
	T->requires_grad = true;
	//tensor_get_2d(T);

	int size = tensor_size(T);

	// Create global MHA
	// we are here now....
	//MHA *M = mha_create_new(A, HEADS, SEQ_LEN, EMB_DIM);

	// Target tensor to compare the output against
	int *shape_target = arena_alloc(A, ndim * sizeof(int));
	shape_target[0] = SEQ_LEN;
	shape_target[1] = EMB_DIM;

	Tensor *target = tensor_create_new(A, ndim, shape_target);
	tensor_randomize(target);

	// define batches for Actual tensor
	int num_chunks = SEQ_LEN / BATCH_SIZE;

	// CREATED THESET COMPONENETS OUTSIDE FOR TESTING!!!
	MHA *m_batch = mha_create_new(A, HEADS, BATCH_SIZE, EMB_DIM);
	
	// Initialize parameters for MHA
	mha_init_params(A, m_batch);

	LayerNorm *L1 = layer_norm_create_new(A, EMB_DIM);
	layer_norm_init_params(A, L1);

	FFN *f = ffn_create(A, EMB_DIM, 128);
	ffn_init_params(f);

	LayerNorm *L2 = layer_norm_create_new(A, EMB_DIM);
	layer_norm_init_params(A, L2);

	// Log file
	FILE *logf = fopen("loss.csv", "w");
	fprintf(logf, "step,loss\n");
	int global_step = 0;

	for (int e = 1; e <= EPOCHS; e++) {
		for (int b = 0; b < num_chunks; b++) {

			//arena_reset(A);
			float *batch_ptr = T->data + b * BATCH_SIZE * EMB_DIM;
			float *target_ptr = target->data + b * BATCH_SIZE * EMB_DIM;

			int *shape_local = arena_alloc(A, ndim * sizeof(int));
			shape_local[0] = BATCH_SIZE;
			shape_local[1] = EMB_DIM;

			Tensor *batch_tensor = tensor_create_new(A, 2, shape_local);
			Tensor *target_batch = tensor_create_new(A, 2, shape_local);
			
			memcpy(batch_tensor->data, batch_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));
			memcpy(target_batch->data, target_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));

			batch_tensor->requires_grad = true;
			Tensor *attn_score = mha_forward(A, batch_tensor, m_batch);
			printf("attn score: \n");
			//tensor_get_2d(attn_score);

			clip_gradient(attn_score);
			tensor_check("attn_score_forward", attn_score);

			// Apply layer_norm
			Tensor *ln1 = layer_norm_forward(A, L1, attn_score);
			tensor_check("ln1_forward", ln1);
			//printf("LayerNorm #1 ran successfully!\n");

			// Create FFN feed-forward NN and run ffn_forward pass
			Tensor *ffn_ln = ffn_forward(A, ln1, f);
			tensor_check("ffn_ln_forward", ffn_ln);

			// Apply layer_norm
			Tensor *ln2 = layer_norm_forward(A, L2, ffn_ln);
			tensor_check("ln2_forward", ln2);
			printf("shape ln2.\n");
			tensor_shape_2d(ln2);

			Tensor *loss = tensor_mse_loss(A, ln2, target_batch);
			tensor_get_2d(loss);
			loss->grad->data[0] = 1.0f; // set the entry point for backward 
																	//
			run_graph_validation(A, loss, MAX_NODES);
			
			backward(A, loss);
			//backward(A, loss);
			//backward(A, loss);
			//backward(A, loss);
			//backward(A, loss);
			//export_and_visualize_graph_new(loss, "graph_main.dot", "graph_main.svg");
			optimizer(m_batch, f, L1, L2, alpha);
		}
	}
	free(A);
	printf("Traning finished!\n");
	//tensor_shape(T);

	return 0;
}
