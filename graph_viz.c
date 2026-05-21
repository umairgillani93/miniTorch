#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tensor.h"


#define MAX_TENSORS 100000

static bool visited[MAX_TENSORS];

static bool is_valid_tensor(Tensor *t) {
    return t != NULL && t->id >= 0 && t->id < MAX_TENSORS;
}

void export_and_visualize_graph(Tensor *root,
                                const char *dot_file,
                                const char *png_file)
{
    if (!root) {
        printf("[GRAPH] root is NULL\n");
        return;
    }

    FILE *f = fopen(dot_file, "w");
    if (!f) {
        printf("[GRAPH] failed to open dot file\n");
        return;
    }

    fprintf(f, "digraph TensorGraph {\n");
    fprintf(f, "rankdir=LR;\n");
    fprintf(f, "node [shape=box, style=rounded];\n");

    // reset visited
    for (int i = 0; i < MAX_TENSORS; i++)
        visited[i] = false;

    // DFS stack
    Tensor *stack[MAX_TENSORS];
    int sp = 0;

    stack[sp++] = root;

    while (sp > 0) {
        Tensor *t = stack[--sp];

        if (!is_valid_tensor(t)) continue;
        if (visited[t->id]) continue;
        visited[t->id] = true;

        // -------- OP NAME SAFE ACCESS --------
        const char *op_name = "NULL";

        if (t->operations && t->operations->name) {
            op_name = t->operations->name;
        }

        fprintf(f,
            "n%d [label=\"id:%d\\nop:%s\\nndim:%d\"];\n",
            t->id,
            t->id,
            op_name,
            t->ndim
        );

        for (int i = 0; i < t->num_parents; i++) {
            Tensor *p = t->parents[i];
            if (!is_valid_tensor(p)) continue;

            fprintf(f, "n%d -> n%d;\n", p->id, t->id);

            if (!visited[p->id]) {
                stack[sp++] = p;
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);

    printf("[GRAPH] DOT written to %s\n", dot_file);

    if (png_file) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "dot -Tpng %s -o %s",
                 dot_file, png_file);

        int ret = system(cmd);
        if (ret == 0) {
            printf("[GRAPH] PNG generated: %s\n", png_file);
        } else {
            printf("[GRAPH] failed to run graphviz (dot)\n");
        }
    }
}
