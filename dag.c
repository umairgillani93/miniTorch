#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


typedef struct Node {
	int data;
	struct Node *nextNode;
} Node;

typedef struct {
	int num_nodes;
	Node **adj_list;
} Graph;

Node *node_create(int data) {
	Node *n = (Node *)malloc(sizeof(Node));
	n->data = data;
	n->nextNode = NULL;
	return n;
}

Graph *graph_create(int nodes) {
	Graph *g = (Graph *)malloc(sizeof(Graph));
	g->num_nodes = nodes;
	g->adj_list = malloc(nodes * sizeof(Node *));

	for (int n = 0; n < nodes; ++n) {
		g->adj_list[n] = NULL;
	}
	return g;
}


void graph_edge(Graph *g, int A, int B) {
	Node *n = node_create(B);
	n->nextNode = g->adj_list[A];
	g->adj_list[A] = n;
}

// Traverse the graph using
void dfs(Node *node, bool *visited) {
	if (!node) {
		return;
	}
	int v = node->data;
	if (visited[v]) return;
	visited[v] = true;
	printf("%d\n", v);
	Node *temp = node->nextNode;
	while(temp!= NULL) {
		dfs(temp, visited);
		temp = temp->nextNode;
	}
}


int main() {
	/*
	 * DFS INTUITION:
	 * Pick one Node (any one)
	 * check all it's neighbors recursively 
	 * add all the Nodes that are visisted in some list *visited
	 * Do it for all the Nodes, and remember ONCE A NODE IS VISITED YOU WON'T BE VISITING THAT AGAIN
	 */
	Graph *g = graph_create(6);
	graph_edge(g, 1, 4);
	graph_edge(g, 2, 4);
	graph_edge(g, 3, 4);
	graph_edge(g, 1, 5);
	graph_edge(g, 2, 5);
	graph_edge(g, 3, 5);

	bool visited[g->num_nodes];
	for (int i = 0; i < g->num_nodes; i++) {
		visited[i] = false;
	}

	for (int i = 0; i < g->num_nodes; i++) {
		if (!visited[i]) {
			dfs(g->adj_list[i], visited);
		}
	}
	return 0;
}

