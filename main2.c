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

void zero_grad(MHA *m, FFN *f, LayerNorm *l1, LayerNorm *l2) {
    if (!m || !f || !l1 || !l2) return;

    // Helper macro to safely zero out a tensor's gradient
    #define SAFE_ZERO(t) \
        if ((t) && (t)->grad && (t)->grad->data) { \
						int sz = tensor_size(t); \
            memset((t)->grad->data, 0, sz * sizeof(float)); \
        }

    // Zero MHA
    SAFE_ZERO(m->wq); SAFE_ZERO(m->wk); SAFE_ZERO(m->wv); SAFE_ZERO(m->wo);

    // Zero FFN
    SAFE_ZERO(f->w1); SAFE_ZERO(f->w2);
    SAFE_ZERO(f->h1); SAFE_ZERO(f->a1);

    // Zero LayerNorms
    SAFE_ZERO(l1->beta); SAFE_ZERO(l1->gamma);
    SAFE_ZERO(l2->beta); SAFE_ZERO(l2->gamma);

    #undef SAFE_ZERO
}

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
int main() {
    srand(time(NULL));

    // Master Arena for permanent parameters and dataset
    Arena *A = malloc(sizeof(Arena));
    arena_init(A, ARENA_SIZE);
    printf("Arena initialized\n");

    // Pure scratch workspace for the autograd graph
    Arena *Sandbox = malloc(sizeof(Arena));
    arena_init(Sandbox, ARENA_SIZE);
    printf("Sandbox initialized\n");

		int global_steps = 0;
    int ndim = 2;
    int *shape_input = arena_alloc(A, ndim * sizeof(int));
    shape_input[0] = SEQ_LEN;
    shape_input[1] = EMB_DIM;
    
    Tensor *T = tensor_create_new(A, 2, shape_input);
    tensor_randomize_weights(T); 
    T->requires_grad = true;

    int *shape_target = arena_alloc(A, ndim * sizeof(int));
    shape_target[0] = SEQ_LEN;
    shape_target[1] = EMB_DIM;

    Tensor *target = tensor_create_new(A, ndim, shape_target);
    tensor_randomize(target);

    int num_chunks = SEQ_LEN / BATCH_SIZE;

    // --- MASTER PARAMETERS (Stored safely in A) ---
    MHA *m_master = mha_create_new(A, HEADS, BATCH_SIZE, EMB_DIM);
    mha_init_params(A, m_master);

    LayerNorm *L1_master = layer_norm_create_new(A, EMB_DIM);
    layer_norm_init_params(A, L1_master);

    FFN *f_master = ffn_create(A, EMB_DIM, 128);
    ffn_init_params(f_master);

    LayerNorm *L2_master = layer_norm_create_new(A, EMB_DIM);
    layer_norm_init_params(A, L2_master);

    FILE *logf = fopen("loss.csv", "w");

    // Helper macro to sync data buffer arrays between two tensors
    #define SYNC_DATA(dest, src) \
        if ((dest) && (src) && (dest)->data && (src)->data) { \
            memcpy((dest)->data, (src)->data, tensor_size(src) * sizeof(float)); \
        }

    for (int e = 1; e <= EPOCHS; e++) {
        for (int b = 0; b < num_chunks; b++) {

            // 1. FRESH CANVAS: Clear everything from the previous batch completely
            arena_reset(Sandbox);

            // 2. Instantiate clean batch layers inside Sandbox
            MHA *m_batch = mha_create_new(Sandbox, HEADS, BATCH_SIZE, EMB_DIM);
            mha_init_params(Sandbox, m_batch);

            LayerNorm *L1 = layer_norm_create_new(Sandbox, EMB_DIM);
            layer_norm_init_params(Sandbox, L1);

            FFN *f = ffn_create(Sandbox, EMB_DIM, 128);
            ffn_init_params(f);

            LayerNorm *L2 = layer_norm_create_new(Sandbox, EMB_DIM);
            layer_norm_init_params(Sandbox, L2);

            // 3. Copy weights from Master (A) -> Batch (Sandbox)
            SYNC_DATA(m_batch->wq, m_master->wq); SYNC_DATA(m_batch->wk, m_master->wk);
            SYNC_DATA(m_batch->wv, m_master->wv); SYNC_DATA(m_batch->wo, m_master->wo);
            SYNC_DATA(f->w1, f_master->w1);       SYNC_DATA(f->w2, f_master->w2);
            SYNC_DATA(f->h1, f_master->h1);       SYNC_DATA(f->a1, f_master->a1);
            SYNC_DATA(L1->gamma, L1_master->gamma); SYNC_DATA(L1->beta, L1_master->beta);
            SYNC_DATA(L2->gamma, L2_master->gamma); SYNC_DATA(L2->beta, L2_master->beta);

            zero_grad(m_batch, f, L1, L2);

            float *batch_ptr = T->data + b * BATCH_SIZE * EMB_DIM;
            float *target_ptr = target->data + b * BATCH_SIZE * EMB_DIM;

            int *shape_local = arena_alloc(Sandbox, ndim * sizeof(int));
            shape_local[0] = BATCH_SIZE;
            shape_local[1] = EMB_DIM;

            Tensor *batch_tensor = tensor_create_new(Sandbox, 2, shape_local);
            Tensor *target_batch = tensor_create_new(Sandbox, 2, shape_local);
            
            memcpy(batch_tensor->data, batch_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));
            memcpy(target_batch->data, target_ptr, BATCH_SIZE * EMB_DIM * sizeof(float));

            batch_tensor->requires_grad = true;

            // 4. Run execution graph completely inside Sandbox
            Tensor *attn_score = mha_forward(Sandbox, batch_tensor, m_batch);
            Tensor *ln1 = layer_norm_forward(Sandbox, L1, attn_score);
            Tensor *ffn_ln = ffn_forward(Sandbox, ln1, f);
            Tensor *ln2 = layer_norm_forward(Sandbox, L2, ffn_ln);
            Tensor *loss = tensor_mse_loss(Sandbox, ln2, target_batch);
            
            loss->grad->data[0] = 1.0f; 
            
            backward(Sandbox, loss);

            // 5. Clip gradients inside Sandbox
            clip_gradient(m_batch->wq); clip_gradient(m_batch->wk);
            clip_gradient(m_batch->wv); clip_gradient(m_batch->wo);
            clip_gradient(f->w1);       clip_gradient(f->w2);
            clip_gradient(L1->gamma);   clip_gradient(L1->beta);
            clip_gradient(L2->gamma);   clip_gradient(L2->beta);

            // 6. Step optimizer inside Sandbox 
            optimizer(m_batch, f, L1, L2, alpha);

            // 7. Copy updated weights back from Batch (Sandbox) -> Master (A)
            SYNC_DATA(m_master->wq, m_batch->wq); SYNC_DATA(m_master->wk, m_batch->wk);
            SYNC_DATA(m_master->wv, m_batch->wv); SYNC_DATA(m_master->wo, m_batch->wo);
            SYNC_DATA(f_master->w1, f->w1);       SYNC_DATA(f_master->w2, f->w2);
            SYNC_DATA(f_master->h1, f->h1);       SYNC_DATA(f_master->a1, f->a1);
            SYNC_DATA(L1_master->gamma, L1->gamma); SYNC_DATA(L1_master->beta, L1->beta);
            SYNC_DATA(L2_master->gamma, L2->gamma); SYNC_DATA(L2_master->beta, L2->beta);

					if (b % 10 == 0) {
						printf("loss:%f after epoch: %d\n", loss->data[0], e);
						fprintf(logf, "%d,%f\n", global_steps++, loss->data[0]);
					}
        }
    }

    #undef SYNC_DATA
    free(A);
    free(Sandbox);
    return 0;
}
