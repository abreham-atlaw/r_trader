
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


def get_nodes(parent_node):
	children = parent_node.get_children()
	if len(children) == 0:
		return []
	nodes = []
	for node in children:
		nodes += get_nodes(node)
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


def draw_graph(root_node):
	graph = nx.Graph()
	nodes = get_nodes(root_node) + [root_node]
	print(f"About to draw {len(nodes)}")
	for i, node in enumerate(nodes):
		if node.parent is not None:
			graph.add_edge(str(i), str(nodes.index(node.parent)))
	nx.draw(graph)
