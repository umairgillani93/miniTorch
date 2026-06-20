#gcc main.c model.c ffn.c ln.c attn2.c tensor.c arena.c -lm && ./a.out
gcc -g -O0 -fsanitize=address -fno-omit-frame-pointer  main2.c ffn.c ln.c attn2.c tensor.c arena.c graph_viz.c -lm && ./a.out 
#gcc -g -O0 -fsanitize=address -fno-omit-frame-pointer tensor.c ffn.c ln.c attn2.c arena.c graph_viz.c -lm && ./a.out 
#gcc  tensor.c arena.c -lm && ./a.out
