#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tensor.h"


#define MAX_TENSORS 100000

static bool visited[MAX_TENSORS];

static bool is_valid_tensor(Tensor *t) {
    return t != NULL && t->id >= 0 && t->id < MAX_TENSORS;
}

void export_and_visualize_graph_new(Tensor *root,
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

    // --- GRAPH HEADER ---
    fprintf(f, "digraph TensorGraph {\n");
    fprintf(f, "  rankdir=LR;\n"); // Left to Right flow
    fprintf(f, "  nodesep=0.5;\n"); // Vertical spacing
    fprintf(f, "  ranksep=0.8;\n"); // Horizontal spacing
    fprintf(f, "  splines=ortho;\n"); // Cleaner straight lines
    fprintf(f, "  node [shape=record, style=filled, fontname=\"Verdana\", fontsize=10];\n");

    // Reset visited array
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

        // Get Op Name safely
        const char *op_name = (t->operations && t->operations->name) ? t->operations->name : "DATA/NULL";

        // --- GRADIENT & COLOR LOGIC ---
        float g_val = 0.0f;
        const char *border_color = "#444444";
        const char *fill_color = "#ffffff";
        
        if (t->grad && t->grad->data) {
            g_val = t->grad->data[0]; // Representative: first element
            if (g_val != 0.0f) {
                border_color = "#e74c3c"; // Red border for active gradients
                fill_color = "#fdedec";   // Light red fill
            }
            if (g_val > 1.0f || g_val < -1.0f) {
                fill_color = "#fadbd8";   // Darker red for high gradients
            }
        }

        // --- RENDER NODE ---
        // Label format: { Top (Meta) | Bottom (Grad) }
        fprintf(f,
            "  n%d [label=\"{ID: %d | Op: %s | Shape: [%d,%d]} | {Grad: %.6f}\", color=\"%s\", fillcolor=\"%s\"];\n",
            t->id, t->id, op_name, t->shape[0], t->shape[1], g_val, border_color, fill_color
        );

        // --- RENDER EDGES ---
        for (int i = 0; i < t->num_parents; i++) {
            Tensor *p = t->parents[i];
            if (!is_valid_tensor(p)) continue;

            // Thick arrows for better visibility
            fprintf(f, "  n%d -> n%d [penwidth=1.5, color=\"#555555\"];\n", p->id, t->id);

            if (!visited[p->id]) {
                stack[sp++] = p;
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);

    printf("[GRAPH] DOT written to %s\n", dot_file);

    // --- GENERATE OUTPUTS ---
    if (png_file) {
        char cmd[1024];
        
        // 1. Generate PNG
        snprintf(cmd, sizeof(cmd), "dot -Tpng %s -o %s", dot_file, png_file);
        if (system(cmd) == 0) {
            printf("[GRAPH] PNG ready: %s\n", png_file);
        }

        // 2. Generate SVG (For Interactive Zooming/Searching)
        char svg_path[1024];
        strncpy(svg_path, png_file, 1024);
        char *ext = strrchr(svg_path, '.');
        if (ext) strcpy(ext, ".svg");

        snprintf(cmd, sizeof(cmd), "dot -Tsvg %s -o %s", dot_file, svg_path);
        if (system(cmd) == 0) {
            printf("[GRAPH] INTERACTIVE SVG ready: %s\n", svg_path);
            printf(">> Open the .svg file in your web browser to zoom and search! <<\n");
        }
    }
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
