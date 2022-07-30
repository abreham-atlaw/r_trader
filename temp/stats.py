import networkx as nx
import matplotlib.pyplot as plt
import random

durations = {
	phase: 0
	for phase in [
		'market_state_copy',
		'agent_state_copy',
		'open_trades_copy',

		'prediction',
		'state_action_to_model_input',

		'promising_action_node',
		'random_state_node',

		'expected_transition_probability',
		'random_choice',
		'expected_transition_probability_norm',

		'select',
		'expand',
		'simulate',
		'backpropagate'

	]
}

iterations = {
	phase: 0
	for phase in [
		'open_trades_copy', "main_loop", 'prediction', 'cached_prediction', 'expected_transition_probability_nodes'
	]
}

possible_state_visits = []
valid_actions = []
prediction_inputs = []


def get_percentages(total):
	percentages = {}
	for phase, duration in durations.items():
		percentages[phase] = duration*100/total
	return percentages


def get_nodes(parent_node, depth=None):
	children = parent_node.get_children()
	if len(children) == 0:
		return []

	if depth == 0:
		return children
	elif depth is not None:
		depth -= 1
	nodes = []
	for node in children:
		nodes += get_nodes(node, depth)
		nodes.append(node)
	return nodes


def get_max_depth(node):

	children = node.get_children()
	if len(children) == 0:
		return 1
	return max(
		[
			get_max_depth(child)
			for child in children
		]
	) + 1


def remove_duplicate(values):
	filtered_values = []
	for value in values:
		if value not in filtered_values:
			filtered_values.append(value)
	return filtered_values


def draw_test_graph():
	graph = nx.Graph()
	graph.add_edge("A", "B")
	graph.add_edge("A", "C")
	nx.draw(graph)
	plt.show()

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

	'''
	From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
	Licensed under Creative Commons Attribution-Share Alike
	If the graph is a tree this will return the positions to plot this in a
	hierarchical layout.
	G: the graph (must be a tree)
	root: the root node of current branch
	- if the tree is directed and this is not given,
	  the root will be found and used
	- if the tree is directed and this is given, then
	  the positions will be just for the descendants of this node.
	- if the tree is undirected and not given,
	  then a random choice will be used.
	width: horizontal space allocated for this branch - avoids overlap with other branches
	vert_gap: gap between levels of hierarchy
	vert_loc: vertical location of root
	xcenter: horizontal location of root
	'''
	if not nx.is_tree(G):
		raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

	if root is None:
		if isinstance(G, nx.DiGraph):
			root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
		else:
			root = random.choice(list(G.nodes))

	def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

		if pos is None:
			pos = {root:(xcenter,vert_loc)}
		else:
			pos[root] = (xcenter, vert_loc)
		children = list(G.neighbors(root))
		if not isinstance(G, nx.DiGraph) and parent is not None:
			children.remove(parent)
		if len(children)!=0:
			dx = width/len(children)
			nextx = xcenter - width/2 - dx/2
			for child in children:
				nextx += dx
				pos = _hierarchy_pos(
					G,
					child,
					width = dx,
					vert_gap = vert_gap,
					vert_loc = vert_loc-vert_gap, xcenter=nextx,
					pos=pos, parent = root
				)
		return pos


	return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def draw_graph(root_node, depth=None):
	def get_node_label(node):
		if node.parent is None:
			return "Root"

		if node.node_type == 0:
			total_value = f"\n{node.get_total_value(): .4f}"
			if node.parent.get_children().index(node) % 2 == 1:
				return "Inc"+total_value
			return "Dec"+total_value

		action = node.action
		label = ""
		if action is None:
			label = "None"
		elif action.action == 0:
			label = "Sell"
		elif action.action == 1:
			label = "Buy"
		elif action.action == 2:
			label = "Close"

		return f"{label}\n{node.total_value: .4f}"

	def get_edge_label(node):
		if node.node_type == 0:
			return f"{node.weight: .5f}"
		return ""

	def get_node_color(node):
		if node.node_type == 0:
			return "darkred"
		return "navy"
	# TODO: YOU ARE HERE: ANALYZE WHERE THE TOTAL VALUE OF NONE ACTIONS COME FROM. BECAUSE THEIR DIRECT CHILDREN SEEM
	#  TO HAVE AN INSTANT VALUE OF 0.

	if depth is None:
		depth = get_max_depth(root_node)

	graph = nx.Graph()
	nodes = get_nodes(root_node, depth=depth) + [root_node]
	print(f"About to draw {len(nodes)}")
	for i, node in enumerate(nodes):
		if node is not root_node:
			graph.add_edge(i, nodes.index(node.parent))
	positions = hierarchy_pos(graph, nodes.index(root_node))
	nx.draw_networkx_labels(
		graph,
		pos=positions,
		labels={i: str(f"{get_node_label(node)}") for i, node in enumerate(nodes) if node is not root_node},
		font_size=7,
		font_color="whitesmoke"
	)
	nx.draw_networkx_edge_labels(
		graph,
		pos=positions,
		edge_labels={(i, nodes.index(node.parent)): get_edge_label(node) for i, node in enumerate(nodes) if node is not root_node},
		font_size=7,
		font_color="black",
	)
	nx.draw(graph, pos=positions, node_size=1200, node_color="navy", edge_color="silver")
	plt.show()