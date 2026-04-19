#gcc main.c ffn.c ln.c attn2.c tensor.c arena.c -lm && ./a.out
gcc arena.c model.c tensor.c -lm && ./a.out
