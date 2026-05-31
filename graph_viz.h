#ifndef GRAPH_VIZ_H
#define GRAPH_VIZ_H

#include <stdbool.h>
#include "tensor.h"

typedef struct Tensor Tensor;

void export_and_visualize_graph(Tensor *root,
                                const char *dot_file,
                                const char *png_file);


void export_and_visualize_graph_new(Tensor *root,
                                const char *dot_file,
                                const char *png_file);
#endif
