#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include "tensor.h"
#include "attention2.h"
#include "feed_forward_nn.h"
#include "layer_norm.h"
#include "arena.h"
#include "graph_viz.h"
#include "config.h"

#include <stddef.h>

#define RAND_FLOAT  (float) rand() / (float) RAND_MAX
#define EPS 1e-5
#define MAX_NODES 10000

size_t GLOBAL_TENSOR_ID = 0;

// Backpropagation Intuition:
 	// so can we say like
	// if we have x @ y = z
	// and z + f = c
	// f(c) = L (loss)
	// now dL/dc = grad_c
	// dL/df = dL/dc * dc/df = grad_c * dc/df => dL/df = grad_f
	// dL/dz = dL/dc * dc/dz = grad_c * dc/dz => dL/dz = grad_z
	// dL/dx = dL/dc * dc/dz * dz/dx => grad_z * dz/dx
	// dL/dy = dL/dc * dc/dz * dz/dy => grad_z * dz/dy
	//
	//


// Some backward functions
// We want to take local gradiens depending upons the operations for this backward funntion (PyTorch style)
//

Tensor *ensure_grad(Arena *A, Tensor *t) {
    if (!t->grad) {
        t->grad = tensor_create_new(A, t->ndim, t->shape);
    }
    return t->grad;
}

void build_topology(Tensor *root, bool *visited, Tensor **topo, int *idx) {
    if (root == NULL) return;

    // If already visited, we skip it to prevent duplicating or processing early
    //if (visited[root->id]) return;
    
    // Temporarily mark as visited to prevent infinite cycles in the recursion
    visited[root->id] = true;

    // Recursively visit all parents first
    if (root->parents != NULL) { 
        for (int p = 0; p < root->num_parents; p++) {
            build_topology(root->parents[p], visited, topo, idx);
        }
    }
    
    // Add ourselves to the topology list AFTER all our parents are added
    topo[(*idx)++] = root;
}

void backward(Arena *A, Tensor *root) {
    Tensor *topo[MAX_NODES] = {0}; 
    bool visited[MAX_NODES] = {0}; 
    int index = 0;

    // 1. Build the forward topological order (Inputs -> ... -> Loss)
    build_topology(root, visited, topo, &index);
    
    // 2. Because of how shared nodes filter through the DFS, we must ensure 
    // we zero out gradients before running backprop, so accumulation works perfectly.
    // (Ensure root has an initial upstream gradient of 1.0 if it's the Loss)
    if (root->grad && root->grad->data) {
        root->grad->data[0] = 1.0f; 
    }

    // 3. Iterate backward through the list (Loss -> ... -> Inputs)
    // This perfectly reverses the dependency chain!
    for (int i = index - 1; i >= 0; i--) {
        Tensor *x = topo[i]; 
        if (!x) continue;
        
        // Execute the backward operation
        if (x->operations && x->operations->backward) {
            x->operations->backward(A, x);
        }
    }
}


//void build_topology(Tensor *root, Tensor **topo, int *idx, bool *added) {
//    if (!root) return;
//
//    if (added[root->id]) return;
//
//    // DO NOT mark visited before recursion
//    if (root->parents) {
//        for (int i = 0; i < root->num_parents; i++) {
//            build_topology(root->parents[i], topo, idx, added);
//        }
//    }
//
//    topo[(*idx)++] = root;
//    added[root->id] = true;
//}
//
//void backward(Arena *A, Tensor *root) {
//
//    Tensor *topo[MAX_NODES] = {0};
//    bool added[MAX_NODES] = {0};
//    int index = 0;
//
//    build_topology(root, topo, &index, added);
//
//    // initialize loss grad safely
//    if (!root->grad) {
//        root->grad = tensor_create_new(A, root->ndim, root->shape);
//    }
//
//    //root->grad->data[0] = 1.0f;
//
//    for (int i = index - 1; i >= 0; i--) {
//        Tensor *x = topo[i];
//
//        if (!x || !x->operations || !x->operations->backward)
//            continue;
//
//        x->operations->backward(A, x);
//    }
//}

bool *vislist() {
	bool *visited = (bool *)malloc(MAX_NODES *sizeof(bool));
	for (int i = 0; i < MAX_NODES; i++) {
		visited[i] = false;
	}
	return visited;
}

void validate_tensor_graph(Arena *A, Tensor *root, bool *visited, GraphReport *rep, int max_tensors)
{
    if (!root) {
        rep->missing_links++;
        return;
    }

    // --------------------------------------------------
    // 1. SAFE ID CHECK (CRITICAL FIX)
    // --------------------------------------------------
    if (root->id < 0 || root->id >= max_tensors) {
        printf("[FATAL] Invalid tensor id=%d\n", root->id);
        rep->corrupt_nodes++;
        return;
    }

    if (visited[root->id]) return;
    visited[root->id] = true;

    rep->visited_nodes++;

    // --------------------------------------------------
    // 2. SAFE PARENTS COUNT CHECK (CRITICAL FIX)
    // --------------------------------------------------
    if (root->num_parents < 0 || root->num_parents > MAX_PARENTS) {
        printf("[FATAL] Tensor %d has invalid num_parents=%d\n",
               root->id, root->num_parents);
        rep->corrupt_nodes++;
        return;
    }

    // --------------------------------------------------
    // 3. SAFE NULL / STRUCT VALIDATION
    // --------------------------------------------------

    if (root->num_parents > 0 && root->parents == NULL) {
        printf("[ERROR] Tensor %d: num_parents=%d but parents=NULL\n",
               root->id, root->num_parents);
        rep->null_parents++;
    }

    if (root->requires_grad && root->num_parents > 0 && root->operations == NULL) {
        printf("[WARNING] Tensor %d missing operations (non-leaf requires_grad tensor)\n",
               root->id);
        rep->null_ops++;
    }

    if (root->requires_grad && root->grad == NULL) {
        printf("[ERROR] Tensor %d requires_grad=1 but grad=NULL\n",
               root->id);
        rep->null_grad++;
    }

    // --------------------------------------------------
    // 4. DEBUG PRINT
    // --------------------------------------------------
    printf("[NODE] id=%d | req=%d | parents=%d | ops=%s | is_leaf=%d | grad=%p\n",
           root->id,
           root->requires_grad,
           root->num_parents,
					 (root->operations && root->operations->name) ? 
					 root->operations->name : "LEAF",
					 root->is_leaf,
					 (void*)root->grad);

    // --------------------------------------------------
    // 5. SAFE DFS TRAVERSAL (CRITICAL FIX)
    // --------------------------------------------------
    if (!root->parents) return;

    for (int i = 0; i < root->num_parents; i++) {

        Tensor *p = root->parents[i];

        if (!p) {
            printf("[ERROR] Tensor %d has NULL parent at index %d\n",
                   root->id, i);
            rep->missing_links++;
            continue;
        }

        validate_tensor_graph(A, p, visited, rep, max_tensors);
    }
}

//void validate_tensor_graph(Arena *A, Tensor *root, bool *visited, GraphReport *rep) {
//    if (!root) {
//        rep->missing_links++;
//        return;
//    }
//
//    if (visited[root->id]) return;
//    visited[root->id] = true;
//
//    rep->visited_nodes++;
//
//    // ---- BASIC VALIDATION ----
//    if (root->parents == NULL && root->num_parents > 0) {
//        printf("[ERROR] Tensor %d: num_parents=%d but parents=NULL\n",
//               root->id, root->num_parents);
//        rep->null_parents++;
//    }
//
//    if (root->requires_grad && root->operations == NULL) {
//        printf("[WARNING] Tensor %d requires_grad=1 but operations=NULL\n",
//               root->id);
//        rep->null_ops++;
//    }
//
//    if (root->requires_grad && root->grad == NULL) {
//        printf("[ERROR] Tensor %d requires_grad=1 but grad=NULL\n",
//               root->id);
//        rep->null_grad++;
//    }
//
//    // ---- PRINT DEBUG INFO ----
//    printf("[NODE] id=%d | req=%d | parents=%d | ops=%p | grad=%p\n",
//           root->id,
//           root->requires_grad,
//           root->num_parents,
//           (void*)root->operations,
//           (void*)root->grad);
//
//    // ---- DFS TRAVERSAL ----
//    for (int i = 0; i < root->num_parents; i++) {
//        if (!root->parents || !root->parents[i]) {
//            printf("[ERROR] Tensor %d has invalid parent at index %d\n",
//                   root->id, i);
//            rep->missing_links++;
//            continue;
//        }
//
//        validate_tensor_graph(A, root->parents[i], visited, rep);
//    }
//}

void run_graph_validation(Arena *A, Tensor *output, int max_nodes) {
    bool *visited = arena_alloc(A, max_nodes * sizeof(bool));
    memset(visited, 0, max_nodes * sizeof(bool));

    GraphReport rep = {0};

    printf("\n========== GRAPH VALIDATION START ==========\n");

    validate_tensor_graph(A, output, visited, &rep, max_nodes);

    printf("\n========== GRAPH REPORT ==========\n");
    printf("Visited nodes     : %d\n", rep.visited_nodes);
    printf("Null parents      : %d\n", rep.null_parents);
    printf("Null ops          : %d\n", rep.null_ops);
    printf("Null grad         : %d\n", rep.null_grad);
    printf("Missing links     : %d\n", rep.missing_links);
}



void traverse_graph(Tensor *root, bool *visited) {
	if (root == NULL) return;
	if (visited[root->id]) return;
	visited[root->id] = true;

	printf("----------------------------------------\n");
	tensor_metadata(root);
	if (root->num_parents > 0) {
		int parents = root->num_parents;
		for (int i = 0; i < root->num_parents; i++) {
			traverse_graph(root->parents[i], visited);
		}
	}
}


void dfs(Arena *A, Tensor *root, bool *visited, Tensor **topo, int *size) {
	// Rather than 'dfs' I need 'topological search' over here
	// which strictly keeps track to parents first rather than
	// visiting all the depth first

	Tensor arr[MAX_NODES];
	if (root == NULL) return;
	if (visited[root->id]) return;

	// make the Node visited, if it's not already
	visited[root->id] = true;

	tensor_metadata(root);
	if (root->parents) {
		int p = root->num_parents;
		for (int i = 0; i < p; i++) {
			dfs(A, root->parents[i], visited, topo, size);
		}
	}

	topo[(*size)++] = root;

	//printf("------------------------\n");
	//printf("\n");
	////tensor_metadata(root);
	//// check operation: root->operations->type:
	//// calculate the gradients of tensor parents using that operation type
	//if (root->operations != NULL) {
	//	switch (root->operations->type) {
	//		case ADD: 
	//			tensor_add_backward(A, root); 
	//			tensor_metadata(root);
	//			tensor_shape_2d(root->grad);

	//			break;


	//		//case MATMUL: tensor_matmul_backward(A, root); break;
	//	}
	//}
}


//void backward(arena *a, tensor *loss, int max_nodes) {
//	bool visited[MAX_NODES] = {0};
//	Tensor *topo[MAX_NODES];
//	int size = 0;
//
//	dfs(A, loss, visited, topo, &size);
//
//	// backward pass
//	for (int i = size - 1; i > 0; i--) {
//		Tensor *node = topo[i];
//		printf("got the Node from topo list: \n");
//		tensor_metadata(node);
//
//		//if (node->operations) {
//		//	int type = node->operations-type;
//		//	printf("Operation Type: %d\n", type);
//		//	switch (type) {
//		//		case ADD:
//		//				printf("Type :%d\n", type);
//		//				tensor_add_backward(A, node);
//		//				break;
//
//		//		case MATMUL:
//		//				printf("Type :%d\n", type);
//		//				tensor_matmul_backward(A, node);
//		//				break;
//		//	}
//		//}
//	}
//}
//

void tensor_metadata(Tensor *x) {
	// prints tensor shape
	printf("Tensor Id: %d\n", x->id);
	printf("shape: \n");
	tensor_shape_2d(x);

	printf("stride: \n");
	//tensor_stride_2d(x);
	printf("tensor dimension: %d\n", x->ndim);
	printf("Requires gradient: %d\n", x->requires_grad);

	if (x->operations != NULL) {
		printf("Created by: %s\n", x->operations->name);
		printf("Backward Function Type: %d\n", x->operations->type);
		printf("Backward Function Pointer: %p\n", x->operations->backward);
		printf("Num Parents: %d\n", x->num_parents);
	}

	else {
		printf("Leaf Node\n");
	}
	
	//else {
	//	fprintf(stderr, "<tensor_metadata> Error: Requires grad = false, so no operations to show\n");
	//}

}
	


void tensor_matmul_backward(Arena *A, Tensor *o) {
    if (!o) {
        fprintf(stderr, "[Error] <tensor_matmul_backward> Input is NULL\n");
        return;
    }

    if (!o->grad) {
        printf("[Warning] <tensor_matmul_backward> Gradient is NULL. Initializing..\n");
        o->grad = tensor_create_new(A, o->ndim, o->shape);
        int grad_size = o->shape[0] * o->shape[1];
        memset(o->grad->data, 0, grad_size * sizeof(float));
    }

    if (!o->parents) {
        fprintf(stderr, "[Error] <tensor_matmul_backward> Invalid parents in subtract\n");
        return;
    }

    Tensor *x = o->parents[0]; // first parent
    Tensor *y = o->parents[1]; // second parent

    if (!x->grad) {
        x->grad = tensor_create_new(A, x->ndim, x->shape);
        int size = x->shape[0] * x->shape[1];
        memset(x->grad->data, 0, size * sizeof(float));
    }

    if (!y->grad) {
        y->grad = tensor_create_new(A, y->ndim, y->shape);
        int size = y->shape[0] * y->shape[1];
        memset(y->grad->data, 0, size * sizeof(float));
    }
		
		int x_rows = x->shape[0];
		int x_cols = x->shape[1];

		int y_rows = y->shape[0];
		int y_cols = y->shape[1];

		int o_cols = o->shape[1];

		/*
		 * For matmul the derivative is as follows:
		 * z = x @ y
		 * dz/dx = up_stream_grad * y.T
		 * dz/dy = x.T * up_stream_grad 
		 */
		
		// dz/dz
		for (int r = 0; r < x_rows; r++) {
			for (int c = 0; c < x_cols; c++) {
				float sum = 0.0f;
				for (int k = 0; k < o_cols; k++) {
					sum +=  (o->grad->data[r * o_cols + k] * y->data[c * o_cols + k]);
				}
				x->grad->data[r * x_cols + c] += sum;
			}
		}
		
		// dy/dz
		for (int r = 0; r < y_rows; r++) {
			for (int c = 0; c < y_cols; c++) {
				float sum = 0.0f;
				for (int k = 0; k < o_cols; k++) {
					sum += (y->data[k * x_cols + r] * o->grad->data[k * o_cols + c]);
			}
			y->grad->data[r * y_cols + c] += sum;
		}

		printf("x->grad: \n");
		tensor_get_2d(x->grad);
		printf("y->grad: \n");
		tensor_get_2d(y->grad);
	}
}

Tensor *tensor_relu(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;
	
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_relu_backward;
		op->type = RELU;
		op->name = "OP_RELU";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if (x->data[r * cols + c] > 0) {
				out->data[r * cols + c] = x->data[r * cols + c];
			}
			else {
				out->data[r * cols + c] = 0.0f;
			}
		}
	}
	return out;
}



Tensor *tensor_fill_val(Arena *A, Tensor *x, int v) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;
	
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	// No computational graph for this
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = v;
		}
	}
	return out;
}

void tensor_fill_ones(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = 1.0f;
	}
}

void tensor_fill_zeros(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = 0.0f;
	}
}

void tensor_fill_with(Tensor *x, float v) {
	tensor_shape_2d(x->grad);
	if (!x || !x->grad) {
		fprintf(stderr, "x OR x->grad is NULL\n");
		return;
	}
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->grad->data[i] = v;
	}
}

void clip_gradient(Tensor *x) {
    int size = tensor_size(x);
    float threshold = 1.0f;
    float MX = 0.0f;
    bool has_bad = false;

    // Pass 1 — detect NaN/Inf and find max abs gradient
    for (int i = 0; i < size; i++) {
        float g = x->data[i];

        if (!isfinite(g)) {
            has_bad = true;
            break;
        }

        float v = fabsf(g);
        if (v > MX) MX = v;
    }

    // If NaN/Inf found → zero gradients and STOP
    if (has_bad) {
        for (int i = 0; i < size; i++)
            x->data[i] = 0.0f;
        return;
    }

    // Pass 2 — clip if too large
    if (MX > threshold) {
        float scale = threshold / MX;   // compute once
        for (int i = 0; i < size; i++)
            x->data[i] *= scale;
    }
}

bool is_exploding(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		float v = x->data[i];
		if (isnan(v) || isinf(v)) {
			return true;
		}
	}
	return false;
}

float max_element(float *row, int cols) {
	float mx = row[0];
	for (int i = 1; i < cols; i++) {
		if (row[i] > row[i - 1]) {
			mx = row[i];
		}
	}
	return mx;
}

Tensor *tensor_row_max(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1]; // row sum shrinks cols dimention
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	// Create output tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_row_max_backward;
		op->type = ROW_MAX;
		op->name = "OP_ROW_MAX";
		out->operations = op;
		
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	for (int r = 0; r < rows; r++) {
		float *row = x->data + r * cols;
		float MX = max_element(row, cols);
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = MX;
		}
	}
	return out;
}

Tensor *tensor_exp(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1]; // row sum shrinks cols dimention
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	// Create output tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->type = EXP;
		op->name = "OP_EXPONENT";
		op->backward = tensor_exp_backward;
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = expf(x->data[r * cols + c]);
		}
	}
	return out;
}

Tensor *tensor_row_sum(Arena *A, Tensor *x) {
	int rows = x->shape[0];
	int cols = x->shape[1]; // row sum shrinks cols dimention
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, rows * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = 1;

	// Create output tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->type = ROW_SUM;
		op->name = "OP_ROW_SUM";
		op->backward = tensor_row_sum_backward;
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	for (int r = 0; r < rows; r++) {
		float sum = 0.0f;
		for (int c = 0; c < cols; c++) {
			sum += x->data[r * cols + c];
		}
		out->data[r] = sum;
	}
	return out;
}

Tensor *tensor_fill_like(Arena *A, Tensor *x, double eps) {
	// No grad = true for this
	// Hence this is not the part of Computational graph
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = x->shape[0]; // rows of out
	out_shape[1] = x->shape[1]; // cols of out
															//
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_fill_like_backward;
		op->type = FILL_LIKE;
		op->name = "OP_FILL_LIKE";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = x->data[r *cols + c] + eps;
		}
	}
	return out;
}

void tensor_randomize_weights(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = RAND_FLOAT;
	}
}

void tensor_randomize(Tensor *x) {
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i] = (rand() % 10) + 1.0f;
	}
}

void tensor_accumulate(Tensor *x, Tensor *grad) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			x->grad->data[r * cols + c] += grad->data[r * cols + c];
		}
	}
}
		

void tensor_add_inplace(Tensor **a, Tensor **b) {
	assert((*a)->shape != (*b)->shape);
	int rows = (*a)->shape[0];
	int cols = (*a)->shape[1];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * cols + j;
			(*a)->data[idx] = (*b)->data[i];
		}
	}
}



void tensor_check(char *name, Tensor *x) {
	if (is_exploding(x)) {
		printf("NaN/Inf detected in: %s\n", name);
		exit(1);
	}
}


// for rows expansion
Tensor *tensor_expand_rows(Arena *A, Tensor *m, int out_rows) {
	assert(m->shape[0] = 1); // has only single row
	int cols = m->shape[1];
	int ndim = m->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = out_rows;
	out_shape[1] = cols;
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (m->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, sizeof(Tensor *));
		out->parents[0] = m;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_expand_rows_backward;
		op->type = EXPAND_ROWS;
		op->name = "OP_EXPAND_ROWS";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
		//out->cols = out_cols;
	}


	// computing logic
	for (int r = 0; r < out_rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = m->data[c];
		}
	}
	return out;
}

Tensor *tensor_element_wise_product(Arena *A, Tensor *a, Tensor *b) {
	printf("a shape: \n");
	tensor_shape_2d(a);
	
	printf("b shape: \n");
	tensor_shape_2d(b);
	assert((a->shape[0] == b->shape[0]) && (a->shape[1] == b->shape[1]));
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		if (a == NULL || b == NULL) {
			fprintf(stderr, "Found parents NULL, graph broken!\n");
		}
		out->parents[0] = a;
		out->parents[1] = b;
		Op *op = arena_alloc(A, sizeof(Op));
		op->type= ELEMENT_WISE_PRODUCT;
		op->name = "OP_ELEMENT_WISE_PRODUCT";
		op->backward = tensor_element_wise_product_backward;
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride; // col_offset = c * col_strid; // index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] * b->data[r * cols + c];
		}
	}
	return out;
}

Tensor *tensor_scaler_div(Arena *A, Tensor *x, float val) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = x->data[r * cols + c] / val;
		}
	}

	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->type= SCALER_DIV;
		op->name = "OP_SCALER_DIV";
		op->backward = tensor_softmax_backward;
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}
	return out;

}

Tensor *tensor_scaler_multiplication(Tensor *x, float val) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int size = tensor_size(x);
	for (int i = 0; i < size; i++) {
		x->data[i * cols + rows] = val * x->data[i * cols + rows];
	}
	return x;
}

Tensor *tensor_scaler_addition(Arena *A, Tensor *x, float val) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = x->data[r * cols + c] + val;
		}
	}
	return out;
}

Tensor *tensor_add(Arena *A, Tensor *a, Tensor *b) {
	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
	int ndim = a->ndim;
	int rows = a->shape[0];
	int cols = a->shape[1];
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;

		// out parents
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		if (a == NULL || b == NULL) {
			fprintf(stderr, "Found parents NULL, graph broken!\n");
		}
		out->parents[0] = a;
		out->parents[1] = b;

		// Operations
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_add_backward;
		op->type = ADD; // helps visualizing the computational graph
		op->name = "OP_ADD"; // helps with logs and monitoring
		out->operations = op;

		// gradients
		out->grad = tensor_create_new(A, ndim, out_shape);

	}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] + b->data[r * cols + c];
		}
	}
	return out;
}
	

//Tensor *tensor_add(Arena *A, Tensor *a, Tensor *b) {
//	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
//	int ndim = a->ndim;
//	int rows = a->shape[0];
//	int cols = a->shape[1];
//	int *out_shape = arena_alloc(A, ndim * sizeof(int));
//	out_shape[0] = rows;
//	out_shape[1] = cols;
//
//	Tensor *out = tensor_create_new(A, ndim, out_shape);
//
//	if (a->requires_grad || b->requires_grad) {
//		out->requires_grad = true;
//
//		// out parents
//		out->num_parents = 2;
//		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
//		out->parents[0] = a;
//		out->parents[1] = b;
//
//		// Operations
//		Op *op = arena_alloc(A, sizeof(Op));
//		op->backward = tensor_add_backward;
//		out->operations = op;
//
//		// gradients
//		out->grad = arena_alloc(A, rows * cols * sizeof(float));
//
//	}
//
//	for (int r = 0; r < rows; r++) {
//		for (int c = 0; c < cols; c++) {
//			out->data[r * cols + c] = a->data[r * cols + c] + b->data[r * cols + c];
//		}
//	}
//	return out;
//}

Tensor *tensor_subtract(Arena *A, Tensor *a, Tensor *b) {
	assert((a->shape[0] == b->shape[0]) && (a->shape[1] == b->shape[1]));
	int ndim = a->ndim;
	int rows = a->shape[0];
	int cols = a->shape[1];
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;

		// out parents
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		
		if (a == NULL || b == NULL) {
			fprintf(stderr, "Found parents NULL, graph broken!\n");
		}
		out->parents[0] = a;
		out->parents[1] = b;

		// Operations
		Op *op = arena_alloc(A, sizeof(Op));
		
		op->backward = tensor_subtract_backward;
		op->type = SUB;
		op->name = "OP_SUBTRACT";
		
		out->operations = op;
		

		// gradients
		out->grad = tensor_create_new(A, ndim, out_shape);

	}

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = (a->data[r * cols + c] - b->data[r * cols + c]);
		}
	}
	return out;
}

void tensor_relu_backward(Arena *A, Tensor *o) {

 if (!o || !o->parents || !o->grad) {
		fprintf(stderr, "[Error] relu_backward invalid input\n");
		return;
	}

	Tensor *p = o->parents[0]; // considering relu has only one parent
	if (!p->grad) {
		printf("[Warning] <tensor_relu_backward> Parent Gradient is NULL. Initializing ..\n");
		p->grad = tensor_create_new(A, p->ndim, p->shape);
		int grad_size = p->shape[0] * p->shape[1];
		memset(o->grad->data, 0, grad_size * sizeof(float));
	}

	int rows = o->shape[0];
	int cols = o->shape[1];

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int idx = r * cols + c;
			float prev = o->grad->data[idx]; // grab up_stream gradient
			float x = p->data[idx];

			float mask = (x > 0) ? 1.0f : 0.0f;
			p->grad->data[idx] += (mask * prev);
		}
	}
	printf("[OK] <tensor_relu_backward> done\n");
}

float loss_value(Tensor *pred, Tensor *target) {
	float squared_err = 0.0f;
	int size = tensor_size(pred);
	
	for (int i = 0; i < size; i++) {
		float diff =  (pred->data[i] - target->data[i]);
		squared_err += (diff * diff);
	}
	return squared_err / size;
}

Tensor *tensor_mse_loss(Arena *A, Tensor *pred, Tensor *target) {
	int rows = pred->shape[0];
	int cols = pred->shape[1];
	int ndim = pred->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *sub = tensor_subtract(A, pred, target);
	Tensor *sq = tensor_square(A, sub); 
	Tensor *mu = tensor_mean(A, sq);
	Tensor *exp = tensor_expand_cols(A, mu, cols);
	Tensor *out = tensor_f(A, exp);

	return out;

}

Tensor *tensor_create(int ndim, int *shape) {
	Tensor *t = malloc(sizeof(Tensor));
	if (!t) {
		fprintf(stderr, "something's wrong with memory allocation\n-> aborting..");
		return NULL;
	}
	t->shape = malloc(ndim * sizeof(int));
	t->stride = malloc(ndim * sizeof(int));
	t->ndim = ndim;


	// define the shape of Tensor
	for (int i = 0; i < ndim; i++) {
		t->shape[i] = shape[i];
	}
	// calcuate size of tensor in self-contained fashion
	int size = 1;
	for (int i = 0; i < ndim; i++) {
		size *= shape[i];
	}
	//printf("Size of tensor: %d\n", size);
	//ndim - 1 > is always 1, fastest changing dimension
	// for next ones wer reveser loop and assign
	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
	t->stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; i--) {
		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
	}
	//printf("Stride: %d, %d, %d\n", t->stride[0], t->stride[1], t->stride[2]);
	// define the data now
	t->data = malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		t->data[i] = (rand() % 10) + 1.0f;
		// printf("%f ", t->data[i]);
	}	

	return t;
}


Tensor *tensor_create_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	memset(t, 0, sizeof(Tensor));
	t->id = GLOBAL_TENSOR_ID++;
	//printf("GLOABL TENSOR ID: %ld\n", GLOBAL_TENSOR_ID);
	t->ndim = ndim;
	t->shape = arena_alloc(A, ndim * sizeof(int));
	t->stride = arena_alloc(A, ndim * sizeof(int));

	// define the shape of Tensor
	int total = 1;
	for (int i = ndim - 1; i >= 0; i--) {
		t->shape[i] = shape[i];
		t->stride[i] = total;
		total *= shape[i];
	}
	// For autograd
	t->data = arena_alloc(A, total * sizeof(float));
	memset(t->data, 0, total * sizeof(float));
	t->parents = NULL;
	t->operations = NULL;
	t->is_leaf = true;
	//Op *op = arena_alloc(A, sizeof(Op));
	//memset(op, 0, sizeof(Op));
	t->operations = NULL;
	t->grad = NULL;
	t->num_parents = 0;
	t->requires_grad = false;

	return t;
}

//Tensor *tensor_create(int ndim, int *shape) {
//	Tensor *t = malloc(sizeof(Tensor));
//	if (!t) {
//		fprintf(stderr, "something's wrong with memory allocation\n-> aborting..");
//		return NULL;
//	}
//	t->shape = malloc(ndim * sizeof(int));
//	t->stride = malloc(ndim * sizeof(int));
//	t->ndim = ndim;
//
//
//	// define the shape of Tensor
//	for (int i = 0; i < ndim; i++) {
//		t->shape[i] = shape[i];
//	}
//	// calcuate size of tensor in self-contained fashion
//	int size = 1;
//	for (int i = 0; i < ndim; i++) {
//		size *= shape[i];
//	}
//	//printf("Size of tensor: %d\n", size);
//	//ndim - 1 > is always 1, fastest changing dimension
//	// for next ones wer reveser loop and assign
//	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
//	t->stride[ndim - 1] = 1;
//	for (int i = ndim - 2; i >= 0; i--) {
//		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
//	}
//	//printf("Stride: %d, %d, %d\n", t->stride[0], t->stride[1], t->stride[2]);
//	// define the data now
//	t->data = malloc(size * sizeof(float));
//	for (int i = 0; i < size; i++) {
//		t->data[i] = (rand() % 10) + 1.0f;
//		// printf("%f ", t->data[i]);
//	}	
//
//	return t;
//}

Tensor *tensor_create_weights(int ndim, int *shape) {
	Tensor *t = malloc(sizeof(Tensor));
	if (!t) {
		fprintf(stderr, "something's wrong with memory allocation\n-> aborting..");
		return NULL;
	}
	t->shape = malloc(ndim * sizeof(int));
	t->stride = malloc(ndim * sizeof(int));
	t->ndim = ndim;


	// define the shape of Tensor
	for (int i = 0; i < ndim; i++) {
		t->shape[i] = shape[i];
	}
	// calcuate size of tensor in self-contained fashion
	int size = 1;
	for (int i = 0; i < ndim; i++) {
		size *= shape[i];
	}
	//printf("Size of tensor: %d\n", size);
	//ndim - 1 > is always 1, fastest changing dimension
	// for next ones wer reveser loop and assign
	// stride[i] = t->stride[i + 1] * t->shape[i + 1]
	t->stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; i--) {
		t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
	}
	//printf("Stride: %d, %d, %d\n", t->stride[0], t->stride[1], t->stride[2]);
	// define the data now
	t->data = malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		t->data[i] = RAND_FLOAT;
		// printf("%f ", t->data[i]);
	}	

	return t;
}


Tensor *tensor_create_weights_new(Arena *A, int ndim, int *shape) {
	Tensor *t = arena_alloc(A, sizeof(Tensor));
	t->shape = arena_alloc(A, ndim * sizeof(int));
	t->stride = arena_alloc(A, ndim * sizeof(int));
	t->ndim = ndim;

	int total = 1;
	for (int i = ndim - 1; i >= 0; i--) {
		t->shape[i] = shape[i];
		t->stride[i] = total;
		total *= shape[i];
	}

	t->data = arena_alloc(A, total * sizeof(float));
	// For autograd
	t->data = arena_alloc(A, total * sizeof(float));
	t->parents = NULL;
	t->operations = NULL;
	t->grad = NULL;
	t->num_parents = 0;

	return t;
}

Tensor *tensor_f(Arena *A, Tensor *x) {
	// sums all the elements of Tensor 'x' and returns 
	// scaler Tensor of shape (1,1)
	
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = 2;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = 1;
	out_shape[1] = 1;

	Tensor *out = tensor_create_new(A, ndim, out_shape);
	
	float sum = 0.0f;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			sum += x->data[r * cols + c];
		}
	}
	out->data[0] = sum;

	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false; // not leaf anymore
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = f_backward;
		op->type = F;
		op->name = "OP_F";
		out->operations = op;

		out->grad = tensor_create_new(A, ndim, out_shape);
	}
	return out;
}

Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b) {
	// let's say bot tensors are off same shape
	assert(a->shape[1] == b->shape[0]);
	int a_rows = a->shape[0];
	int a_cols = a->shape[1];
	int b_rows = b->shape[0];
	int b_cols = b->shape[1];

	

	int *out_shape = arena_alloc(A, a->ndim * sizeof(int));
	out_shape[0] = a_rows;
	out_shape[1] = b_cols;

	Tensor *out = tensor_create_new(A, a->ndim, out_shape);

	int ndim = a->ndim;
	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		// 1. NEED TO SAVE THE PARENTS
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		out->parents[1] = b;
		
		// 2. NEED TO POPULATE THE grad
		//int out_size = a->shape[0] * b->shape[1];
		
		out->grad = tensor_create_new(A, ndim, out_shape);
		
		// 3. Need to SAVE THE OPERATIONS for computation graph
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_matmul_backward;
		op->type = MATMUL;
		op->name = "OP_TIMES";
		out->operations = op;
		out->shared_dim = a_cols;
	}

	for (int r = 0; r < a_rows; r++) {
		for (int c = 0; c < b_cols; c++) {
			float sum = 0.0f;
			for (int k = 0; k < a_cols; k++) {
				sum += (a->data[(r * a_cols + k)] *
					 	b->data[(k * b_cols + c)]);
			}
			out->data[r * b_cols + c] = sum;
		}
	}
	return out;
}

//Tensor *tensor_matmul(Arena *A, Tensor *a, Tensor *b) {
//	// let's say bot tensors are off same shape
//	assert(a->shape[1] == b->shape[0]);
//	int a_rows = a->shape[0];
//	int a_cols = a->shape[1];
//	int b_rows = b->shape[0];
//	int b_cols = b->shape[1];
//
//	int *out_shape = arena_alloc(A, a->ndim * sizeof(int));
//	out_shape[0] = a_rows;
//	out_shape[1] = b_cols;
//
//	Tensor *out = tensor_create_new(A, a->ndim, out_shape);
//
//	if (a->requires_grad || b->requires_grad) {
//		out->requires_grad = true;
//		// 1. NEED TO SAVE THE PARENTS
//		out->num_parents = 2;
//		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
//		out->parents[0] = a;
//		out->parents[1] = b;
//		out->requires_grad = true;
//		
//		// 2. NEED TO POPULATE THE grad
//		int out_size = a->shape[0] * b->shape[1];
//		out->grad = arena_alloc(A, out_size * sizeof(float));
//		
//		// 3. Need to SAVE THE OPERATIONS for computation graph
//		Op *op = arena_alloc(A, sizeof(Op));
//		op->backward = tensor_matmul_backward;
//		out->operations = op;
//
//	}
//
//	for (int r = 0; r < a_rows; r++) {
//		for (int c = 0; c < b_cols; c++) {
//			float sum = 0.0f;
//			for (int k = 0; k < a_cols; k++) {
//				sum += (a->data[(r * a_cols + k)] *
//					 	b->data[(k * b_cols + c)]);
//			}
//			out->data[r * b_cols + c] = sum;
//		}
//	}
//	return out;
//}

//Tensor *tensor_matmul_forward(Arena *A, Tensor *a, Tensor *b) {
//	int rows_a = a->shape[0];
//	int cols_a = a->shape[1];
//
//	int rows_b = b->shape[0];
//	int cols_b = b->shape[1];
//
//	// resultant tensor having shape (rows_a, cols_b);
//	int ndim_r = 2;
//	int *shape_r = arena_alloc(A, ndim_r * sizeof(int));
//	shape_r[0] = a->shape[0];
//	shape_r[1] = b->shape[1];
//
//	Tensor *r = tensor_create_new(A, ndim_r, shape_r);
//	//printf("Created resultant tensor\n");
//
//	for (int i = 0; i < rows_a; i++) {
//		for (int j = 0; j < cols_b; j++) {
//			float sum = 0.0f;
//			for (int k = 0; k < cols_a; k++) {
//				sum += (a->data[i * a->stride[0] + k * a->stride[1]]  * b->data[k * b->stride[0] + j * b->stride[1]]);
//			}
//			r->data[i * r->stride[0] + j * r->stride[1]] = sum;
//		}
//	}
//	return r;
//}

Tensor *tensor_softmax(Arena *A, Tensor *x) {
	/*
	 * Tensor *max = tensor_row_max(A, Tensor *x)
	 * Tensor *max_exp = tensor_expand_cols(A, Tensor *x)
	 * Tensor *shifted = tensor_sub(A, max_exp, num_cols);
	 */
		int rows = x->shape[0];
    int cols = x->shape[1];
		int ndim = x->ndim;

    Tensor *row_max = tensor_row_max(A, x);
    //Tensor *row_max_expanded = tensor_expand_cols(A, row_max, cols);
    Tensor *shifted = tensor_subtract(A, x, row_max);
    Tensor *exp = tensor_exp(A, shifted);

    Tensor *row_sum = tensor_row_sum(A, exp);

    Tensor *row_sum_expanded = tensor_expand_cols(A, row_sum, cols);
    Tensor *out = tensor_div(A, exp, row_sum_expanded);

		//if (x->requires_grad) {
		//	out->requires_grad;
		//	out->num_parents = 1;
		//	out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		//	out->parents[0] = x;
		//	Op *op = arena_alloc(A, sizeof(Op));
		//	op->backward = tensor_softmax_backward;
		//	op->type = SOFTMAX;
		//	op->name = "OP_SOFTMAX";
		//	out->operations = op;
		//	out->grad = tensor_create_new(A, ndim, x->shape);
		//}

    return out;
}

	//int rows = x->shape[0];
	//int cols = x->shape[1];
	//int ndim = x->ndim;
	//int *out_shape = arena_alloc(A, ndim * sizeof(int));
	//out_shape[0] = rows;
	//out_shape[1] = cols;

	//Tensor *out = tensor_create_new(A, ndim, out_shape);

	//if (x->requires_grad) {
	//	out->requires_grad = true;
	//	out->num_parents = 1;
	//	out->parents = arena_alloc(A, sizeof(Tensor *));
	//	out->parents[0] = x;
	//	Op *op = arena_alloc(A, sizeof(Op));
	//	op->backward = tensor_softmax_backward;
	//	out->operations = op;
	//	out->grad = arena_alloc(A, rows * cols * sizeof(float));
	//}

	//// softmax(x) = e(x[i])/sum(row); // for each row
	//
	//Tensor *exp = tensor_exp(A, x);
	//Tensor *row_sum = tensor_row_sum(A, exp);
	//Tensor *row_sum_exp = tensor_expand_cols(A, row_sum, cols);
	//out = tensor_div(A, exp, row_sum_exp);

	//return out;
//}

//void tensor_free(Tensor *t) {
//	if (!t) return;
//	free(t->data);
//	free(t->stride);
//	free(t->shape);
//	free(t);
//	//printf("Freed successfully!\n");
//}

void tensor_get_2d(Tensor *t) {
	if (!t) return;
	int rows = t->shape[0];
	int cols = t->shape[1];
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			printf("%0.2f ", t->data[r * cols + c]);
		}
		printf("\n");
	}
}		

Tensor *tensor_transpose(Arena *A, Tensor *a) {
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = 2;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = cols;
	out_shape[1] = rows;
	Tensor *t = tensor_create_new(A, ndim, out_shape);
	//t->id = GLOBAL_TENSOR_ID++;

	if (a->requires_grad) {
		t->requires_grad = true;
		t->num_parents = 1;
		t->parents = arena_alloc(A, t->num_parents * sizeof(Tensor *));
		t->parents[0] = a;
		t->grad = tensor_create_new(A, ndim, out_shape);
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_transpose_backward;
		op->type = TRANSPOSE;
		op->name = "OP_Transpose";
		t->operations = op;
	}
	
	int rows_a = a->shape[0];
	int cols_a = a->shape[1];
	for (int i = 0; i < rows_a; i++) {
		for (int j = 0; j < cols_a; j++) {
			t->data[j * t->stride[0] + i * t->stride[1]] = a->data[i * a->stride[0] + j * a->stride[1]];
		}
	}

	return t;
}

int tensor_size(Tensor *t) {
	int size = 1;
	for (int i = 0; i < t->ndim; i++) {
		size *= t->shape[i];
	}
	//printf("Tensor size: %d\n", size);
	return size;
}	

void tensor_shape_2d(Tensor *t) {
	printf("(%d, %d)\n", t->shape[0], t->shape[1]);
}

void tensor_stride_2d(Tensor *t) {
	printf("(%d, %d)\n", t->stride[0], t->stride[1]);
}



// Auto-grad methods
Tensor *tensor_mean(Arena *A, Tensor *a) {
	// computes row-wise mean 
	// out_shape = (rows, 1)
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = 1;
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad) {
		// Build computation graph
		// define requires_grad = true for out;
		out->requires_grad = true;
		out->is_leaf = false;

		// define number of parents
		out->num_parents = 1;
		out->parents = arena_alloc(A, sizeof(Tensor *));;
		out->parents[0] = a;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_mean_backward;
		op->type = MEAN;
		op->name = "OP_MEAN";
		out->operations = op;
		
		// define gradients matrix
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	for (int r = 0; r < rows; r++) {
		float *row = a->data + r * cols;
		float row_sum = 0.0f;
		float row_mean = 0.0f;
		for (int c = 0; c < cols; c++) {
			row_sum += row[c];
		}
		row_mean = row_sum / cols;
		out->data[r] = row_mean; // cols = 1, c = 0 so r * 1 + 0 = r
	}
	return out;
}

Tensor *tensor_expand_cols(Arena *A, Tensor *m, int out_cols) {
	// Takes the mean value for each row 
	// and expands it column number of times to match the other tensor
	// reduce_mean = [m1],
	//               [m2],
	//               [m3]
	// expand_mean = [m1, m1, m1]
	//               [m2, m2, m2]
	//               [m3, m3, m3]
	
	//printf("shape m: \n");
	//tensor_shape_2d(m);
	assert(m->ndim == 2);
	assert(m->shape[1] == 1);
	int rows = m->shape[0];
	int ndim = 2;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = out_cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	//int *grad_shape= arena_alloc(A, ndim * sizeof(int));
	//grad_shape[0] = rows;
	//grad_shape[1] = 1;
	
	if (m->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, sizeof(Tensor *));
		out->parents[0] = m;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_expand_cols_backward;
		op->type = EXPAND_COLS;
		op->name = "OP_EXPAND_COLS";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
		out->cols = out_cols;
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		float v = m->data[r];
		for (int c = 0; c < out_cols; c++) {
			out->data[r * out_cols + c] = v;
		}
	}
	return out;
}

Tensor *tensor_square(Arena *A, Tensor *a) {
	//assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_square_backward;
		op->type = SQUARE;
		op->name = "OP_SQUARE";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = (a->data[r * cols + c] * a->data[r * cols + c]);
		}
	}
	return out;

}

Tensor *tensor_sqrt(Arena *A, Tensor *a) {
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = 2;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_sqrt_backward;
		op->type = SQRT;
		op->name = "OP_SQRT";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = sqrt(a->data[r * cols + c] + EPS); // Fixed: backward = 1/2 * (x + EPS) ^ 1/2
		}
	}
	return out;
}

Tensor *tensor_div(Arena *A, Tensor *a, Tensor *b) {
	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		out->parents[0] = a;
		out->parents[1] = b;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_square_backward;
		op->type = DIVISION;
		op->name = "OP_DIVISION";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] / b->data[r * cols + c];
		}
	}

	return out;

}


// back methods

void tensor_row_sum_backward(Arena *A, Tensor *o) {
    Tensor *x = o->parents[0];

    int rows = x->shape[0];
    int cols = x->shape[1];

    if (!x->grad) {
			x->grad = tensor_create_new(A, x->ndim, x->shape);
			int grad_size = x->shape[0] * x->shape[1];
			memset(x->grad->data, 0, grad_size * sizeof(float)); 
    }

    for (int r = 0; r < rows; r++) {
        float grad = o->grad->data[r];

        for (int c = 0; c < cols; c++) {
            x->grad->data[r * cols + c] += grad;
        }
    }
}

void tensor_exp_backward(Arena *A, Tensor *o) {
    Tensor *x = o->parents[0];

    int rows = x->shape[0];
    int cols = x->shape[1];

    if (!x->grad) {
			x->grad = tensor_create_new(A, x->ndim, x->shape);
			int grad_size = x->shape[0] * x->shape[1];
			memset(x->grad->data, 0, grad_size * sizeof(float)); 
    }

    for (int i = 0; i < rows * cols; i++) {
        x->grad->data[i] += o->grad->data[i] * o->data[i];
    }
}

void tensor_row_max_backward(Arena *A, Tensor *o) {
	Tensor *x = o->parents[0];

	int rows = x->shape[0];
	int cols = x->shape[1];

	if (!x->grad) {
			x->grad = tensor_create_new(A, x->ndim, x->shape);
			int grad_size = x->shape[0] * x->shape[1];
			memset(x->grad->data, 0, grad_size * sizeof(float)); 
	}

	for (int r = 0; r < rows; r++) {
			float *x_row = x->data + r * cols;
			int max_idx = 0;
			float mx = x_row[0];

			// Find the max item, only this will get gradient, 
			// all others will be zero
			for (int c = 1; c < cols; c++) {
					if (x_row[c] > mx) {
							mx = x_row[c];
							max_idx = c;
					}
			}

			float grad_sum = 0.0f;
			for (int c = 0; c < cols; c++) {
					grad_sum += o->grad->data[r * cols + c];
			}

			x->grad->data[r * cols + max_idx] += grad_sum;
	}
}

void tensor_fill_like_backward(Arena *A, Tensor *o) {
	
	if (!o || !o->grad) {
		fprintf(stderr, "out OR out->grad is NULL.\n");
		return;
	}

	if (!o->parents) {
		fprintf(stderr, "[Error] <tensor_fill_like_backward>! Parents is NULL.\n");
		printf("Quitting.. \n");
		return;
	}

	Tensor *p = o->parents[0];
	int ndim = o->ndim;

	if (!p->grad) {
		p->grad = tensor_create_new(A, p->ndim, p->shape);
		//tensor_fill_ones(p->grad); // added for safety

	}

	int rows = p->shape[0];
	int cols = p->shape[1];

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			p->grad->data[r * cols + c] += (o->grad->data[r * cols + c]);
		}
	}
	tensor_get_2d(p->grad);
	printf("[OK] <tensor_fill_like_backward> done!\n");
}

void f_backward(Arena *A, Tensor *o) {
	// Takes the foward output and computes it's parent gradient
	
	//printf("backward function pointer: %p\n", o->operations->backward);
	//printf("o parents: %d\n", o->num_parents);
	if (!o || !o->parents) {
		fprintf(stderr, "[Error] Input Parents not found..!\n");
		return;
	}

	Tensor *p = o->parents[0];
	if (!p->grad) {
		p->grad = tensor_create_new(A, p->ndim, p->shape);
	}


	float prev = o->grad->data[0];
	int p_rows = p->shape[0];
	int p_cols = p->shape[1];

	for (int r = 0; r < p_rows; r++) {
		for (int c = 0; c < p_cols; c++) {
			//float prev = o->grad->data[r];
			p->grad->data[r * p_cols + c] += prev;
		}
	}
	tensor_get_2d(p->grad);
	printf("[OK] <f_backward> done!\n");
}

void tensor_sqrt_backward(Arena *A, Tensor *o) {

		// FIXED:
		if (!o) {
			fprintf(stderr, "[Error] <tensor_sqrt_backward> input is NULL.\n");
			return;
		}


    if (!o->grad) {
				printf("[Warning] <tensor_sqrt_backward>. Input Gradient is NULL, Initializing..\n");
        o->grad = tensor_create_new(A, o->ndim, o->shape);
				int grad_size = o->shape[0] * o->shape[1];
				memset(o->grad->data, 0, grad_size * sizeof(float));
    }
		
		// check if parents exists
		if (!o->parents || !o->parents[0]) {
			fprintf(stderr, "[Error] <tensor_sqrt_backward> Parents is NULL\n");
			return;
		}
    
    Tensor *p = o->parents[0];

    if (!p->grad) {
				printf("[Warning] <tensor_sqrt_backward>. Parent Gradient is NULL, Initializing..\n");
        p->grad = tensor_create_new(A, p->ndim, p->shape);
				int grad_size = p->shape[0] * p->shape[1];
				memset(p->grad->data, 0, grad_size * sizeof(float));
    }

		/* Local Gradient of y = x^1/2 => y' =  1/2*x^2 
		 * and Global gradient would be dL/ds = prev * y'
		 */

		int rows = p->shape[0];
		int cols = p->shape[1]; // Here  'o' and it's parent will have same shapes  (row,  cols)


		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				int idx = r * cols + c;
				float prev = o->grad->data[idx];
				float curr = 1.0f / sqrt(p->data[idx]);
				printf("sqrt prev: %f\n", prev);
				printf("sqrt curr: %f\n", curr);
				p->grad->data[idx] += (curr * prev);
				printf("sqrt grad: %f\n", p->grad->data[idx]);
			}
		}

		tensor_get_2d(p->grad);
		printf("[OK] <tensor_sqrt_backward> done!\n");
}

void tensor_mse_backward(Tensor *x) {
	// Will be implemented later IA
}

void tensor_expand_rows_backward(Arena *A, Tensor *o) {

    if (!o || !o->grad || !o->parents[0]) return;
    Tensor *m = o->parents[0];
    
    if (!m->grad) {
        m->grad = tensor_create_new(A, m->ndim, m->shape);
    }

    int out_rows = o->shape[0];
    int cols = o->shape[1];

    for (int r = 0; r < out_rows; r++) {
        for (int c = 0; c < cols; c++) {
            m->grad->data[c] += o->grad->data[r * cols + c];
        }
    }
		tensor_get_2d(m->grad);
		//printf("EXPAND_ROWS backward: tensor %zu grad=%p parent=%p\n",
    //   o->id, o->grad, m);
		printf("[OK] <tensor_expand_rows> done!\n");
}

void tensor_square_backward(Arena *A, Tensor *o) {

	if (!o || !o->grad) {
		fprintf(stderr, "out OR out->grad is NULL.\n");
		return;
	}

	if (!o->parents) {
		fprintf(stderr, "[Error] <tensor_square_backward>! Parents is NULL.\n");
		return;
	}

	Tensor *p = o->parents[0];
	if (!p->grad) {
		p->grad = tensor_create_new(A, p->ndim, p->shape);
	}

	int rows = p->shape[0];
	int cols = p->shape[1];

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int idx = r * cols + c;
			float prev = o->grad->data[idx];
			float curr = 2.0f * p->data[idx];
			p->grad->data[r * cols + c] += (curr * prev);
		}
	}
	tensor_get_2d(p->grad);
	printf("\n");
	printf("[OK] <tensor_square_backward> done!\n");

}

void tensor_expand_cols_backward(Arena *A, Tensor *o) {

    if (!o) {
        fprintf(stderr, "[Error] <tensor_expand_cols_backward> output is NULL\n");
        return;
    }

    if (!o->grad) {
        printf("[Warning] output grad is NULL. Initializing\n");
        o->grad = tensor_create_new(A, o->ndim, o->shape);
        int size = o->shape[0] * o->shape[1];
        memset(o->grad->data, 0, size * sizeof(float));
    }

    if (!o->parents || !o->parents[0]) {
        fprintf(stderr, "[Error] parent missing\n");
        return;
    }

    Tensor *p = o->parents[0];

    if (!p->grad) {
        printf("[Warning] parent grad is NULL. Initializing\n");
        p->grad = tensor_create_new(A, p->ndim, p->shape);
        int size = p->shape[0] * 1; // mean forward pass has shape (rows, 1);
        memset(p->grad->data, 0, size * sizeof(float));
    }

    int o_rows = o->shape[0];
    int o_cols = o->shape[1];
		int p_cols = p->shape[1];

    for (int r = 0; r < o_rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < o_cols; c++) {
					//printf("o->grad->data: %f\n", o->grad->data[r * o_cols + c]);
					sum += o->grad->data[r * o_cols + c];
        }
				//printf("sum: %f\n",  sum);
				p->grad->data[r] += sum;
    }

		tensor_get_2d(p->grad);
    printf("[OK] <tensor_expand_cols_backward> done!\n");
}


//void tensor_expand_cols_backward(Tensor *x) {
//	// we have expanded the original tensor in columns direction
//	// example: x = [[r0c0, r0c1, r0c2...], 
//	//               [r1c0, r1c1, r1c2..]
//	//               [r2c0, r2c1, r2c2]]
//	//
//	//          y =    [[r0]
//	//                [r1]
//	// 							  [r2]]
//	//
//	// and the gradient y' = [[sum(r0)]
//	//                       [sum(r1)] 
//	//                       [sum(r2)]]
//	//
//	
//	Tensor *p = x->parents[0];
//
//	if (!p) {
//		fprintf(stderr, "Parents not found, exiting..\n");
//		return;
//		}
//
//	if (!p->requires_grad) return;
//
//	int rows = x->shape[0];
//	int cols = x->shape[1]; // we'll be shrinking along axis = 1;
//
//	if (!x->grad) {
//		fprintf(stderr, "x->grad is NULL, no up stream gradients found\n");
//		return;
//	}
//
//	for (int r = 0; r < rows; r++) {
//		float sum = 0.0f;
//		for (int c = 0; c < cols; c++) {
//			sum += x->grad->data[r * cols + c];
//			printf("sum: %f\n", sum);
//		}
//		p->grad->data[r] += sum;
//	}
//	/*
//	 * d(mean)/dL = d(expanded)/dL * prev_grad(dL/dL)
//	 * d(mean)/dL = d(expanded)/dL * 1 
//	 */
//	printf("[Success] Gradient for <TENSOR_EXPAND_COLS_BACKWARD> saved!\n");
//}

void tensor_mean_backward(Arena *A, Tensor *o) {

    if (!o) {
        fprintf(stderr, "[Error] Input is NULL\n");
        return;
    }

    if (!o->parents || !o->parents[0]) {
        fprintf(stderr, "[Error] <tensor_mean_backward> parent missing\n");
        return;
    }

    Tensor *p = o->parents[0];

    // Ensure parent gradient exists
    if (!p->grad) {
        p->grad = tensor_create_new(A, p->ndim, p->shape);
        int p_size = p->shape[0] * p->shape[1];
        memset(p->grad->data, 0, p_size * sizeof(float));
    }

    int p_rows = p->shape[0];
    int p_cols = p->shape[1];

    int o_rows = o->shape[0];
    int o_cols = o->shape[1];

    for (int r = 0; r < p_rows; r++) {
			float prev = o->grad->data[r];
			//printf("prev: %f\n", prev);
			for (int c = 0; c < p_cols; c++) {
				p->grad->data[r * p_cols + c] += (prev * (1.0f / (float)p_cols));
				//printf("p_cols : %f\n", (float) p_cols);
				//printf("o->grad->data: %f\n", prev);
				//printf("p->grad->data: %f\n", p->grad->data[r * p_cols + c]);
			}
    }
		tensor_get_2d(p->grad);

    printf("[OK] <tensor_mean_backward> done!\n");
}


//void tensor_matmul_backward(Arena *A, Tensor *x, Tensor *y, Tensor *grad_prev) {
//	// Take the gradients of tensor X and tensor Y, w.r.t. Loss using chain rule
//	// assusming that Node output is obtained using x @ y. It calculates local gradients
//	// dx/dL = grad_prev * y; 
//	// dy/dL = grad_prev * x; 
//	int x_rows = x->shape[0];
//	int x_cols = x->shape[1];
//	int y_rows = y->shape[0];
//	int y_cols = y->shape[1];
//
//	int ndim = 2;
//	x_shape = arena_alloc(A, ndim * sizeof(int));
//	y_shape = arena_alloc(A, ndim * sizeof(int));
//	
//	// create x and y gradients
//	x->grad = tensor_create_new(A, ndim, x_shape);
//	y->grad = tensor_create_new(A, ndim, y_shape);
//
//	x->grad = tensor_matmul(A, grad_prev, y);
//	y->grad = tensor_matmul(A, x, grad_prev);
//
//}

void tensor_add_backward(Arena *A, Tensor *o) {
	// by adding up the Tensors, whatever change happens in 
	// the Tensor will have linear impact on the currNode;
	// i.e if we raise Tensor (a) or Tensor (b) by samll about
	// that same change will be reflacted in currNode
	// dz/da = 1;
	// dz/db = 1;
	
	if (!o || !o->grad) {
		fprintf(stderr, "out OR out->grad is NULL.\n");
		return;
	}

	if (!o->parents) {
		fprintf(stderr, "[Error] <tensor_subtract_backward>! Parents is NULL.\n");
		printf("Quitting.. \n");
		return;
	}
	printf("o->parents: %d\n", o->num_parents);

	assert(o->num_parents == 2);

	Tensor *x = o->parents[0];
	Tensor *y = o->parents[1];
	int ndim = x->ndim;

	//if (!x->grad || !y->grad) {
	//	fprintf(stderr, "[Error] <tensor_add_backward>! Parents is NULL.\n");
	//	printf("Quitting.. \n");
	//	return;
	//}

	if (!x->grad) {
		x->grad = tensor_create_new(A, ndim, o->shape);
	}

	if (!y->grad) {
		y->grad = tensor_create_new(A, ndim, o->shape);
	}

	int rows = x->shape[0];
	int cols = x->shape[1];

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			x->grad->data[r * cols + c] += (o->grad->data[r * cols + c]);
			y->grad->data[r * cols + c] += (o->grad->data[r * cols + c]);
		}
	}
	tensor_get_2d(x->grad);
	printf("\n");
	tensor_get_2d(y->grad);
	printf("[OK] <tensor_add_backward> done!\n");
}


void tensor_element_wise_product_backward(Arena *A, Tensor *o) {
	
	if (!o || !o->grad) {
		fprintf(stderr, "out OR out->grad is NULL.\n");
		return;
	}

	if (!o->parents) {
		fprintf(stderr, "[Error] <tensor_subtract_backward>! Parents is NULL.\n");
		printf("Quitting.. \n");
		return;
	}

	assert(o->num_parents == 2);

	Tensor *x = o->parents[0];
	Tensor *y = o->parents[1];
	int ndim = x->ndim;


	if (!x->grad) {
		x->grad = tensor_create_new(A, ndim, o->shape);
	}

	if (!y->grad) {
		y->grad = tensor_create_new(A, ndim, o->shape);
	}

	int rows = x->shape[0];
	int cols = x->shape[1];

	int size = tensor_size(x);

	for (int i = 0; i < size; i++) {
		float val = o->grad->data[i];
		x->grad->data[i] += val * y->data[i];
		y->grad->data[i] += val * x->data[i];
	}

	tensor_get_2d(x->grad);
	printf("\n");
	tensor_get_2d(y->grad);
	printf("[OK] <tensor_element_wise_product_backward> done!\n");
}


void tensor_div_backward(Arena *A, Tensor *o) {
	// if A/B = Z, then
	// dZ/dA = up_stream_gradient (dL/dA) *  1/B;
	// dZ/dB = up_stream_gradient (dL/dA) * (- A/B^2)
	// this is what this function computes
	

    if (!o) {
        fprintf(stderr, "[Error] <tensor_div_backward> Input is NULL\n");
        return;
    }

    if (!o->grad) {
        printf("[Warning] <tensor_div_backward> Gradient is NULL. Initializing..\n");
        o->grad = tensor_create_new(A, o->ndim, o->shape);
        int grad_size = o->shape[0] * o->shape[1];
        memset(o->grad->data, 0, grad_size * sizeof(float));
    }

    if (!o->parents) {
        fprintf(stderr, "[Error] <tensor_div_backward> Invalid parents in subtract\n");
        return;
    }

    Tensor *x = o->parents[0];
    Tensor *y = o->parents[1];

    if (!x->grad) {
        x->grad = tensor_create_new(A, x->ndim, x->shape);
        int size = x->shape[0] * x->shape[1];
        memset(x->grad->data, 0, size * sizeof(float));
    }

    if (!y->grad) {
        y->grad = tensor_create_new(A, y->ndim, y->shape);
        int size = y->shape[0] * y->shape[1];
        memset(y->grad->data, 0, size * sizeof(float));
    }

		int rows = x->shape[0];
		int cols = x->shape[1];
		
		// for dZ/dB
		for (int r = 0; r < rows;  r++) {
			for (int c = 0; c < cols; c++) {
				int idx = r * cols + c;
				float prev = o->grad->data[idx];
				float x_val = x->data[idx];
				float y_val = y->data[idx];
				printf("<tensor_div_back> prev: %f\n", prev);

				x->grad->data[idx] += (prev * (1.0f / y_val));
				y->grad->data[idx] += (prev * (-x_val / (y_val * y_val)));
			}
		}
		tensor_get_2d(x->grad);
		printf("\n");
		tensor_get_2d(y->grad);
		printf("[OK] <tensor_div_backward> done!\n");
}


void tensor_softmax_backward(Tensor *x) {
	// Will be implemented later. IA
}

void tensor_subtract_backward(Arena *A, Tensor *o) {

    if (!o) {
        fprintf(stderr, "[Error] <tensor_subtract_backward> Input is NULL\n");
        return;
    }

    if (!o->grad) {
        printf("[Warning] <tensor_subtract_backward> Gradient is NULL. Initializing..\n");
        o->grad = tensor_create_new(A, o->ndim, o->shape);
        int grad_size = o->shape[0] * o->shape[1];
        memset(o->grad->data, 0, grad_size * sizeof(float));
    }

    if (!o->parents) {
        fprintf(stderr, "[Error] <tensor_subtract_backward> Invalid parents in subtract\n");
        return;
    }

    Tensor *x = o->parents[0];
    Tensor *y = o->parents[1];

    if (!x->grad) {
        x->grad = tensor_create_new(A, x->ndim, x->shape);
        int size = x->shape[0] * x->shape[1];
        memset(x->grad->data, 0, size * sizeof(float));
    }

    if (!y->grad) {
        y->grad = tensor_create_new(A, y->ndim, y->shape);
        int size = y->shape[0] * y->shape[1];
        memset(y->grad->data, 0, size * sizeof(float));
    }

    int rows = x->shape[0];
    int cols = x->shape[1];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {

            float prev = o->grad->data[r * cols + c];
						//printf("prev: %f\n", prev);
            x->grad->data[r * cols + c] += prev;
            y->grad->data[r * cols + c] -= prev;
        }
    }

		tensor_get_2d(x->grad);
		printf("\n");
		tensor_get_2d(y->grad);
    printf("[OK] <tensor_subtract_backward> done!\n");
}

void tensor_slice_cols_backward(Arena *A, Tensor *o) {
	Tensor *x = o->parents[0];

	int rows = x->shape[0];
	int cols = x->shape[1];

	int dk = o->shape[1];

	int k = o->shared_dim; // recovered from metadata

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < dk; c++) {
			int out_idx = r * dk + c;
			int x_idx = r * cols + k * dk + c;

			x->grad->data[x_idx] += o->grad->data[out_idx];
		}
	}
}

void tensor_transpose_backward(Arena *A, Tensor *o) {
	Tensor *x = o->parents[0];

	int rows = x->shape[0];
	int cols = x->shape[1];

	for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {

					x->grad->data[r * cols + c] +=
							o->grad->data[c * rows + r];  // just undo the indexs swaps to take the transpose again.
																						// i.e dL/dt += transpose(o->grad)
			}
	}
}

void tensor_concat_backward(Arena *A, Tensor *o) {
	int rows = o->shape[0];
	int out_cols = o->shape[1];

	int num_heads = o->num_parents;
	int cols = out_cols / num_heads;

	for (int k = 0; k < num_heads; k++) {
			Tensor *head = o->parents[k];

			for (int r = 0; r < rows; r++) {
					for (int c = 0; c < cols; c++) {

							int out_idx  = r * out_cols + k * cols + c;
							int head_idx = r * cols + c;

							head->grad->data[head_idx] +=
									o->grad->data[out_idx];
					}
			}
	}
}

bool tensor_equal(Tensor *x, Tensor *y) {
	int rows = x->shape[0];
	int cols = x->shape[1];

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if (x->data[r * cols + c] != y->data[r * cols + c]) {
				return false;
			}
		}
	}
	return true;
}

Tensor *tensor_concat(Arena *A, Tensor **heads, int num_heads) {
	int rows = heads[0]->shape[0];
	int cols = heads[0]->shape[1];
	int ndim = heads[0]->ndim;
	int out_cols = cols * num_heads;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = out_cols; // 8 * 4 = 32;
	
	// create out tensor
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (heads[0]->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents= HEADS;
		out->parents = arena_alloc(A, HEADS * sizeof(Tensor *));
		for (int i = 0; i < HEADS; i++) {
			out->parents[i] = heads[i];
		}

		Op *op = arena_alloc(A, sizeof(Op));
		op->type = CONCAT;
		op->name = "OP_CONCAT";
		op->backward = tensor_concat_backward;
		out->operations = op;

		out->grad = tensor_create_new(A, ndim, out_shape);
		
		// Extra parameter, which we are setting only for concat method
		out->cols = out_cols;

	}
	// core logic
	for (int k = 0; k < num_heads; k++) {
		Tensor *head = heads[k]; // head chunk
	
		for (int r= 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out->data[r * out_cols + k * cols + c] =
				 	head->data[r * cols + c];
			}
		}
	}
	return out;
}


Tensor *tensor_slice_cols(Arena *A, Tensor *x, int k, int dk) {
	int rows = x->shape[0];
	int cols = x->shape[1];
	int ndim = x->ndim;
	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = dk;
	// If actual tensor 'x' has shape (16, 32)
	// out tensor will be sliced version of it 
	// and will have shape (16, dk);
	
	// create output tensor now
	Tensor *out = tensor_create_new(A, ndim, out_shape);

	// build computational graph
	if (x->requires_grad) {
		out->requires_grad = true;
		out->is_leaf = false;
		out->num_parents = 1;
		out->parents = arena_alloc(A, sizeof(Tensor *));
		out->parents[0] = x;
		Op *op = arena_alloc(A, sizeof(Op));
		op->backward = tensor_slice_cols_backward;
		op->type = SLICE;
		op->name = "OP_SLICE";
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	// core logic here
	// slices the main tensor 'x' and populates tensor 'out'
	// [[1,2,3,4,5,6,7,8],
	//	  [1,2,3,4,5,6,7,8],
	//		 [1,2,3,4,5,6,7,8],
	//	  [1,2,3,4,5,6,7,8]]
	
	// using pointer arithematic
	//float *start = x->data + k * rows * dk;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < dk; c++) {
			out->data[r * dk + c] = x->data[r * cols + k * dk + c];
		}
	}
	return out;
}

Tensor *tensor_scalling(Arena *A, Tensor *a, Tensor *b) {
	assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]);
	int rows = a->shape[0];
	int cols = a->shape[1];
	int ndim = a->ndim;

	int *out_shape = arena_alloc(A, ndim * sizeof(int));
	out_shape[0] = rows;
	out_shape[1] = cols;

	Tensor *out = tensor_create_new(A, ndim, out_shape);

	if (a->requires_grad || b->requires_grad) {
		out->requires_grad = true;
		out->num_parents = 2;
		out->parents = arena_alloc(A, out->num_parents * sizeof(Tensor *));
		if (a == NULL || b == NULL) {
			fprintf(stderr, "Found parents NULL, graph broken!\n");
		}
		out->parents[0] = a;
		out->parents[1] = b;
		Op *op = arena_alloc(A, sizeof(Op));
		op->type= SCALLED;
		op->name = "OP_SCALED";
		op->backward = tensor_square_backward;
		out->operations = op;
		out->grad = tensor_create_new(A, ndim, out_shape);
	}

	// IMPORTANT!!!
	// row_offset = r * row_stride;
	// col_offset = c * col_strid;
	// index = row_offset + col_offset;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			out->data[r * cols + c] = a->data[r * cols + c] * b->data[r * cols + c];
		}
	}
	return out;
}



//int main() {
//
//	/*
//	 * Tensor as nodes
//	 * Tensor operations as Nodes
//	 * X(Node)----|some opeartion| (NOde)----> Y(Node)
//	 * shape, stride, id, operations, name, backward function...
//	 */
//
//	/*Loss -> getting computed -> using some opeerations
//
//	Loss -> mean_squared_error ->  mean((pred - targer) ^ 2) / SIZE
//
//	1. diff -> tensor_subtract(pred - target)
//	2. Square -> tensor_square(diff) | square shape: (16, 32)
//	3.  Mean -> tensor_mean(square) -> mean | shape mean: (16, 32) -> (16, 1)
//	4. Expand_cols -> tensor_expand_cols(Mean) -> shape: (16, 32) 
//	6. Loss = output(Expand_cols_method)
//
//	// we need to start backpropagation from the end 
//	// Loss -> shape = (seq_len, emb_dim) -> my case -> (16,32)
//	//
//	// dLoss/Loss = 1 -> starting from Gradient tensor ->shape [16, 32] all 1s
//	// Tensor_expand_cols_backward;
//	// for tensor_expand_cols_forward  we have shape = (16, 32) this returns us a tensor with memeber called grad having shape equals (16,32)
//	// tensor_expand_cols_backward() this will give us shape (16, 1);
//	//
//	// If we expand the cols of some tensor
//	// the gradient is simply the accumulation OR sum acorss the dimentions
//	//
//	// forward returns T =  [1, 1, ,1]  -> this stores grad memeber dimension as (16, 32)
//	//                      [1, 1,  1]
//	//
//	//                 T' = [3] -> this has actual dimention after taking gradient as (16, 1)
//	//                      [3]
//	/*/
//
//	// so can we say like
//	// if we have x @ y = z
//	// and z + f = c 
//	// f(c) = L (loss)
//	// now dL/dc = grad_c
//	// dL/df = dL/dc * dc/df = grad_c * dc/df => dL/df = grad_f
//	// dL/dz = dL/dc * dc/dz = grad_c * dc/dz => dL/dz = grad_z
//	// dL/dx = dL/dc * dc/dz * dz/dx => grad_z * dz/dx
//	// dL/dy = dL/dc * dc/dz * dz/dy => grad_z * dz/dy
//	srand(time(NULL));
//	Arena *A = malloc(sizeof(Arena));
//	size_t SIZE = 1024 * 1024;
//	arena_init(A, ARENA_SIZE);
//	printf("Arena allocated\n");
//	int ndim = 2;
//	int *shape_x = arena_alloc(A, ndim * sizeof(int));
//	int *shape_y = arena_alloc(A, ndim * sizeof(int));
//	shape_x[0] = 4;
//	shape_x[1] = 2;
//
//	shape_y[0] = 2;
//	shape_y[1] = 4;
//
//	Tensor *y = tensor_create_new(A, ndim, shape_y);
//	tensor_randomize_weights(y);
//	y->requires_grad = true;
//	y->grad = tensor_create_new(A, y->ndim, y->shape);
//	tensor_fill_zeros(y->grad);
//
//	Tensor *x = tensor_create_new(A, ndim, shape_x);
//	tensor_randomize_weights(x);
//	x->requires_grad = true;
//	x->grad = tensor_create_new(A, x->ndim, x->shape);
//	tensor_fill_zeros(x->grad);
//
//	MHA *m = mha_create_new(A, HEADS, SEQ_LEN, EMB_DIM);
//	mha_init_params(m);
//
//	Tensor *out = mha_forward(A, x, m);
//	tensor_get_2d(out);
//	//Tensor *loss = tensor_f(A, out); // (1,1)
//	//loss->grad->data[0] = 1.0f;
//
//	//run_graph_validation(A, loss, MAX_NODES);
//
//	//backward(A, loss);
//	//backward(A, loss);
//	//backward(A, loss);
//	//backward(A, loss);
//	//backward(A, loss);
//	//export_and_visualize_graph_new(loss, "graph_att.dot", "graph_att.png");
//
//	//mha->out = tensor_concat(A, heads_arr, mha->num_heads);
//
//	return 0;
//}
