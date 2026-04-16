#include <stdio.h>
#include <stdlib.h>

int main() {
	int *arr = malloc(2 * sizeof(int));
	arr[0] = 1;
	arr[1] = 2;

	for (int i = 0; i < 2; i++) {
		printf("%d\n", arr[i]);
	}
	return 0;
}

