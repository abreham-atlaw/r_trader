


if __name__ == "__main__":
	import networkx as nx
	import matplotlib.pyplot as plt

	graph = nx.Graph()
	graph.add_edge("A", "B")
	graph.add_edge("A", "C")
	nx.draw(graph)
	plt.show()
