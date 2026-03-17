#ifndef CONFIG_H
#define CONFIG_H

#define BATCH_SIZE 16
#define SEQ_LEN    160
#define EMB_DIM    32
#define EPOCHS     100
#define LEARNING_RATE 0.001f
#define HEADS      8
#define HEAD_DIM EMB_DIM / HEADS

#endif
