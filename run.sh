#gcc main.c model.c ffn.c ln.c attn2.c tensor.c arena.c -lm && ./a.out
gcc -g -O0 -fsanitize=address -fno-omit-frame-pointer ln.c attn2.c tensor.c arena.c -lm && ./a.out 
#gcc  tensor.c arena.c -lm && ./a.out

