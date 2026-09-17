#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()  {
	char *name[2] = {
		"umair", "someafricanname"
	};

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < strlen(name[i]); j++) {
			printf("%c ", name[i][j]);
		}
		printf("\n");
	}
 	return 0;
}

