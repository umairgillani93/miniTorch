#ifndef CONFIG_H
#define CONFIG_H

#define HIDDEN_DIM 128
#define BATCH_SIZE 16
#define SEQ_LEN    16000
#define EMB_DIM    32
#define LR         1e-3
#define HEADS      8
#define HEAD_DIM EMB_DIM / HEADS
#define EPOCHS     3
#define BETA      1e-9
#define GEMMA     14-3
#define EPS       1e-5

#endif
