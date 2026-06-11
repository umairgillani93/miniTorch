#ifndef CONFIG_H
#define CONFIG_H
#include <stddef.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define RAND_FLOAT (float) rand() / (float) RAND_MAX
#define HIDDEN_DIM 128
#define BATCH_SIZE 16
#define SEQ_LEN    4
#define EMB_DIM    4
#define LR         1e-3
#define HEADS      8
#define HEAD_DIM EMB_DIM / HEADS
#define EPOCHS     1
#define BETA      1e-9
#define GEMMA     14-3
#define EPS       1e-5
#define ARENA_SIZE 1024 * 1024 * 1024
#define MAX_NODES 10000000
#define MAX_PARENTS 10000000

#endif
