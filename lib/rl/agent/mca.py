from typing import *
from abc import ABC, abstractmethod

import numpy as np

import gc
import psutil
import uuid

from lib.utils.logger import Logger
from .mba import ModelBasedAgent
from lib.utils.math import sigmoid
from lib.utils.staterepository import StateRepository, SectionalDictStateRepository
from temp import stats


class MonteCarloAgent(ModelBasedAgent, ABC):

	class Node:

		class NodeType:
			STATE = 0
			ACTION = 1

		def __init__(self, parent, action, node_type, weight: float = 1.0, instant_value: float = 0.0, id=None):
			self.children = []
			self.visits = 0
			self.total_value = 0
			self.instant_value = instant_value
			self.parent: MonteCarloAgent.Node = parent
			self.action = action
			self.weight = weight
			self.node_type = node_type
			self.id = id
			if id is None:
				self.id = self.__generate_id()

		def increment_visits(self):
			self.visits += 1

		def add_value(self, value):
			self.total_value += value

		def set_total_value(self, value):
			self.total_value = value

		def add_child(self, child):
			self.children.append(child)
			child.parent = self

		def detach_action(self):
			self.action = None

		def is_visited(self) -> bool:
			return self.get_visits() != 0

		def has_children(self):
			return len(self.get_children()) != 0

		def get_children(self) -> List['MonteCarloAgent.Node']:
			return self.children

		def get_visits(self):
			return self.visits

		def get_total_value(self):
			return self.total_value

		def __generate_id(self):
			return uuid.uuid4().hex

		def find_node_by_id(self, id) -> Union['MonteCarloAgent.Node', None]:
			if self.id == id:
				return self
			for child in self.get_children():
				result = child.find_node_by_id(id)
				if result is not None:
					return result
			return None

		def __eq__(self, other):
			return isinstance(other, MonteCarloAgent.Node) and self.id == other.id

	def __init__(
			self,
			*args,
			min_free_memory_percent=10,
			logical=False,
			uct_exploration_weight=1,
			state_repository: StateRepository = None,
			**kwargs
	):

		super(MonteCarloAgent, self).__init__(*args, **kwargs)
		self.__min_free_memory = min_free_memory_percent
		self.__logical = logical
		self.__uct_exploration_weight = uct_exploration_weight
		self.__set_mode(logical)
		self._state_repository = state_repository
		if state_repository is None:
			self._state_repository = SectionalDictStateRepository(2, 15)

	def __set_mode(self, logical: bool):
		if logical:
			self._backpropagate, self._uct = self.__logical_backpropagate, self.__logical_uct
		else:
			self._backpropagate, self._uct = self.__legacy_backpropagate, self.__legacy_uct

	@abstractmethod
	def _init_resources(self) -> object:
		pass

	@abstractmethod
	def _has_resource(self, resources) -> bool:
		pass

	def _get_state_value(self, state):
		return self._get_environment().get_reward(state)

	def _get_random_state_node(self, action_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':
		state_nodes = action_node.get_children()

		if len(state_nodes) == 1:
			return state_nodes[0]

		probabilities = np.array([
			state_node.weight
			for state_node in state_nodes
		]).astype('float64')
		probabilities /= probabilities.sum()

		choice = np.random.choice(
			state_nodes,
			1,
			p=probabilities
		)[0]

		return choice

	def __manage_resources(self):
		if psutil.virtual_memory().percent > (100 - self.__min_free_memory):
			Logger.info("Realising Memory. Calling gc.collect")
			gc.collect()

	def __legacy_uct(self, node: 'MonteCarloAgent.Node') -> float:
		if not node.is_visited():
			return np.inf

		return (node.get_total_value()/node.get_visits()) + np.sqrt(np.log(node.parent.get_visits())/node.get_visits())

	def __logical_uct(self, node: 'MonteCarloAgent.Node') -> float:
		if not node.is_visited():
			return np.inf

		return sigmoid(node.get_total_value()) + np.sqrt(np.log(node.parent.get_visits())/node.get_visits()) * self.__uct_exploration_weight

	def __check_stm(self, node) -> 'MonteCarloAgent.Node':
		if self.__short_term_memory is None:
			return node

		memory: Optional[MonteCarloAgent.Node] = self.__short_term_memory.recall(node)
		if memory is None:
			return node
		return memory

	def _select(self, parent_state_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':

		promising_action_node: MonteCarloAgent.Node = max(
													parent_state_node.get_children(),
													key=lambda node: self._uct(node)
												)

		chosen_state_node: MonteCarloAgent.Node = self._get_random_state_node(promising_action_node)

		if len(chosen_state_node.get_children()) == 0:
			return chosen_state_node

		return self._select(chosen_state_node)

	def _expand(self, state_node: 'MonteCarloAgent.Node'):
		if self._get_environment().is_episode_over(self._state_repository.retrieve(state_node.id)):
			return

		for action in self._get_available_actions(self._state_repository.retrieve(state_node.id)):
			action_node = MonteCarloAgent.Node(state_node, action, MonteCarloAgent.Node.NodeType.ACTION)
			for possible_state in self._get_possible_states(self._state_repository.retrieve(state_node.id), action):
				weight = self._get_expected_transition_probability(self._state_repository.retrieve(state_node.id), action, possible_state)
				value = self._get_environment().get_reward(possible_state)
				possible_state_node = MonteCarloAgent.Node(
					action_node,
					None,
					MonteCarloAgent.Node.NodeType.STATE,
					weight=weight,
					instant_value=value
				)
				self._state_repository.store(possible_state_node.id, possible_state)
				action_node.add_child(possible_state_node)
			state_node.add_child(action_node)

	def __simulate(self, state_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':

		if not state_node.has_children():
			return state_node

		action_node: MonteCarloAgent.Node = np.random.choice(state_node.get_children(), 1)[0]

		return self._get_random_state_node(action_node)

	def __legacy_backpropagate(self, node: 'MonteCarloAgent.Node', reward=None) -> None:
		node.increment_visits()
		if node.parent is None:
			return

		if reward is None:
			reward = node.instant_value
		node.add_value(self._discount_factor * reward)

		if node.node_type == MonteCarloAgent.Node.NodeType.ACTION:
			reward = reward * self._discount_factor

		self.__legacy_backpropagate(node.parent, reward)

	def __logical_backpropagate(self, node: 'MonteCarloAgent.Node') -> None:
		node.increment_visits()

		if node.node_type == MonteCarloAgent.Node.NodeType.STATE:
			node.set_total_value(node.instant_value)
			if node.has_children():
				node.add_value(self._discount_factor * max([action_node.get_total_value() for action_node in node.get_children()]))

		else:
			total_weight = np.sum([state_node.weight for state_node in node.get_children()])
			node.set_total_value(
				np.sum([
					state_node.get_total_value()*state_node.weight/total_weight
					for state_node in node.get_children()
				])
			)

		if node.parent is None:
			return

		self.__logical_backpropagate(node.parent)

	def _monte_carlo_simulation(self, root_node: 'MonteCarloAgent.Node'):
		self._expand(root_node)

		resources = self._init_resources()

		stats.iterations["main_loop"] = 0
		while self._has_resource(resources):
			leaf_node = self._select(root_node)
			self._expand(leaf_node)
			final_node = self.__simulate(leaf_node)
			self._backpropagate(final_node)
			self.__manage_resources()
			stats.iterations["main_loop"] += 1

	def _monte_carlo_tree_search(self, state) -> object:
		root_node = MonteCarloAgent.Node(None, None, MonteCarloAgent.Node.NodeType.STATE)
		self._state_repository.store(root_node.id, state)

		self._monte_carlo_simulation(root_node)

		Logger.info(
			f"Simulations Done: Iterations: {stats.iterations['main_loop']}, "
			f"Depth: {stats.get_max_depth(root_node)}, "
			f"Nodes: {len(stats.get_nodes(root_node))}"
		)
		optimal_action = max(root_node.get_children(), key=lambda node: node.get_total_value()).action
		Logger.info(f"Best Action {optimal_action}")
		self._state_repository.clear()
		return optimal_action

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		pass

	def _get_optimal_action(self, state, **kwargs):
		return self._monte_carlo_tree_search(state)
