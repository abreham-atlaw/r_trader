from abc import ABC, abstractmethod

import numpy as np

import gc
import psutil

from lib.utils.logger import Logger
from .mba import ModelBasedAgent
from temp import stats

class MonteCarloAgent(ModelBasedAgent, ABC):

	class Node:

		def __init__(self, parent, state, action, weight: float = 1.0, instant_value=0):
			self.children = []
			self.visits = 0
			self.total_value = 0
			self.instant_value = instant_value
			self.parent: MonteCarloAgent.Node = parent
			self.state = state
			self.action = action
			self.weight = weight

		def increment_visits(self):
			self.visits += 1

		def add_value(self, value):
			self.total_value += value

		def add_child(self, child):
			self.children.append(child)

		def detach_state(self):
			self.state = None

		def detach_action(self):
			self.action = None

		def is_visited(self) -> bool:
			return self.get_visits() != 0

		def has_children(self):
			return len(self.get_children()) != 0

		def get_children(self):
			return self.children

		def get_visits(self):
			return self.visits

		def get_total_value(self):
			return self.total_value

		def calc_uct(self):
			if self.visits == 0:
				return np.inf

			if self.get_visits() == 0:
				return 0

			return (self.total_value / self.get_visits()) + np.sqrt(
				np.log(self.parent.get_visits()) / self.get_visits()
			)

	def __init__(self, *args, min_free_memory_percent=10, **kwargs):
		super(MonteCarloAgent, self).__init__(*args, **kwargs)
		self.__min_free_memory = min_free_memory_percent

	@abstractmethod
	def _init_resources(self) -> object:
		pass

	@abstractmethod
	def _has_resource(self, resources) -> bool:
		pass

	def _get_state_value(self, state):
		return self._get_environment().get_reward(state)

	def __get_random_state_node(self, action_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':
		state_nodes = action_node.get_children()

		if len(state_nodes) == 1:
			return state_nodes[0]

		probabilities = np.array([
			state_node.weight
			for state_node in state_nodes
		])
		probabilities /= probabilities.sum()

		choice = np.random.choice(
			state_nodes,
			1,
			p=probabilities
		)[0]

		return choice

	def __clean_node(self, node, clean_action=False):
		node.detach_state()
		if clean_action:
			node.detach_action()

	def __manage_resources(self):
		if psutil.virtual_memory().percent > (100 - self.__min_free_memory):
			Logger.info("Releasing memory")
			gc.collect()

	def __select(self, parent_state_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':

		promising_action_node: MonteCarloAgent.Node = max(
														parent_state_node.get_children(),
														key=lambda node: node.calc_uct()
													)

		chosen_state_node: MonteCarloAgent.Node = self.__get_random_state_node(promising_action_node)

		if len(chosen_state_node.get_children()) == 0:
			return chosen_state_node

		return self.__select(chosen_state_node)

	def __expand(self, state_node: 'MonteCarloAgent.Node'):
		if self._get_environment().is_episode_over(state_node.state):
			return

		for action in self._get_available_actions(state_node.state):
			action_node = MonteCarloAgent.Node(state_node, state_node.state, action)
			for possible_state in self._get_possible_states(state_node.state, action):
				weight = self._get_expected_transition_probability(state_node.state, action, possible_state)
				value = self._get_environment().get_reward(possible_state)
				possible_state_node = MonteCarloAgent.Node(action_node, possible_state, None, weight=weight, instant_value=value)
				action_node.add_child(possible_state_node)
			state_node.add_child(action_node)
			self.__clean_node(action_node)

		self.__clean_node(state_node)

	def __simulate(self, state_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':

		if not state_node.has_children():
			return state_node

		action_node: MonteCarloAgent.Node = np.random.choice(state_node.get_children(), 1)[0]

		return self.__get_random_state_node(action_node)

	def __backpropagate(self, state_node: 'MonteCarloAgent.Node', previous_node_value) -> None:

		if state_node.parent is None:
			state_node.increment_visits()
			return

		state_node_value = 0
		if not state_node.is_visited():
			state_node_value = state_node.instant_value
		state_node_value += self._discount_factor * previous_node_value
		state_node.add_value(state_node_value)
		state_node.increment_visits()

		action_node = state_node.parent
		action_node_value = state_node_value * state_node.weight
		action_node.add_value(action_node_value)
		action_node.increment_visits()

		self.__backpropagate(action_node.parent, action_node_value)

	def __monte_carlo_tree_search(self, state) -> object:
		root_node = MonteCarloAgent.Node(None, state, None)
		self.__expand(root_node)

		resources = self._init_resources()

		stats.iterations["main_loop"] = 0
		while self._has_resource(resources):
			leaf_node = self.__select(root_node)
			self.__expand(leaf_node)
			final_node = self.__simulate(leaf_node)
			self.__backpropagate(final_node, 0)
			self.__manage_resources()
			stats.iterations["main_loop"] += 1

		Logger.info(f"Simulations Done: Iterations: {stats.iterations['main_loop']}, Depth: {stats.get_max_depth(root_node)}, Nodes: {len(stats.get_nodes(root_node))}")

		return max(root_node.get_children(), key=lambda node: node.get_total_value() / node.get_visits()).action

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		pass

	def _get_optimal_action(self, state, **kwargs):
		return self.__monte_carlo_tree_search(state)
