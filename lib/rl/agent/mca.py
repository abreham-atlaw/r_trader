import json
import os
import random
import time
import typing
from datetime import datetime
from typing import *
from abc import ABC, abstractmethod

import numpy as np

import gc
import psutil
import uuid
from dataclasses import dataclass

from lib.utils.logger import Logger
from .mba import ModelBasedAgent
from lib.utils.math import sigmoid
from lib.utils.staterepository import StateRepository, SectionalDictStateRepository
from lib.utils.stm import ShortTermMemory, CueMemoryMatcher, ExactCueMemoryMatcher
from temp import stats
from ..environment import ModelBasedState
from ...network.rest_interface import Serializer


class MonteCarloAgent(ModelBasedAgent, ABC):

	class Node:

		class NodeType:
			STATE = 0
			ACTION = 1

		def __init__(self, parent, action, node_type, depth: int = 0, weight: float = 1.0, instant_value: float = 0.0, id=None):
			self.children = []
			self.visits = 0
			self.total_value = 0
			self.instant_value = instant_value
			self.parent: MonteCarloAgent.Node = parent
			self.action = action
			self.weight = weight
			self.node_type = node_type
			self.depth = depth
			self.id = id
			if id is None:
				self.id = self.__generate_id()
			self.predicted_value = None

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

	@dataclass
	class NodeMemory:
		node: 'MonteCarloAgent.Node'

	class NodeShortTermMemory(ShortTermMemory):

		def _import_memory(self, memory: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.NodeMemory':
			return MonteCarloAgent.NodeMemory(memory)

		def _export_memory(self, memory: 'MonteCarloAgent.NodeMemory') -> object:
			return memory.node

		def set_matcher(self, matcher: 'MonteCarloAgent.NodeMemoryMatcher'):
			self._matcher = matcher

		def get_matcher(self) -> 'MonteCarloAgent.NodeMemoryMatcher':
			return self._matcher

	class NodeMemoryMatcher(CueMemoryMatcher):

		def __init__(self, state_matcher: Optional[CueMemoryMatcher] = None, repository: Optional[StateRepository] = None):
			self.__repository = repository
			self.__state_matcher: CueMemoryMatcher = state_matcher
			if self.__state_matcher is None:
				self.__state_matcher = ExactCueMemoryMatcher()

		def set_repository(self, repository: StateRepository):
			self.__repository = repository

		def get_repository(self) -> StateRepository:
			if self.__repository is None:
				raise Exception("Repository Not set yet")
			return self.__repository

		def is_match(self, cue: 'MonteCarloAgent.Node', memory: 'MonteCarloAgent.NodeMemory') -> bool:
			return self.__state_matcher.is_match(
				self.get_repository().retrieve(cue.id),
				self.get_repository().retrieve(memory.node.id)
			)

	def __init__(
			self,
			*args,
			min_free_memory_percent=10,
			logical=False,
			uct_exploration_weight=1,
			state_repository: StateRepository = None,
			use_stm: bool = True,
			short_term_memory: 'MonteCarloAgent.NodeShortTermMemory' = None,
			probability_correction: bool = False,
			min_probability: typing.Optional[float] = None,
			top_k_nodes: typing.Optional[int] = None,
			dynamic_k_threshold: float = None,
			dump_nodes: bool = False,
			dump_path: str = "./",
			dump_visited_only: bool = False,
			node_serializer: typing.Optional['NodeSerializer'] = None,
			state_serializer: typing.Optional[Serializer] = None,
			squash_epsilon: float = 1e-9,
			**kwargs
	):

		super(MonteCarloAgent, self).__init__(*args, **kwargs)
		self.__min_free_memory = min_free_memory_percent
		self.__logical = logical
		self.__uct_exploration_weight = uct_exploration_weight
		self.__set_mode(logical)
		self.__min_probability = min_probability
		self.__use_stm: bool = use_stm
		self.__short_term_memory = short_term_memory
		if short_term_memory is None and use_stm:
			self.__short_term_memory = MonteCarloAgent.NodeShortTermMemory(
				size=10,  # TODO: GET SIZE FROM FIRST INSTANCE OF RUNNING
				matcher=MonteCarloAgent.NodeMemoryMatcher()
			)
		self._state_repository = state_repository
		if state_repository is None:
			self._state_repository = SectionalDictStateRepository(2, 15, serializer=state_serializer)
		self.__short_term_memory.get_matcher().set_repository(self._state_repository)
		self.__probability_correction = probability_correction
		self.__current_graph_: Optional[MonteCarloAgent.Node] = None
		self.__top_k_nodes = top_k_nodes
		self.__dump_nodes = dump_nodes
		self.__dump_path = dump_path
		self.__dump_visited_only = dump_visited_only
		self.__serializer = node_serializer
		self.__squash_epsilon = squash_epsilon
		self.__dynamic_k_threshold = dynamic_k_threshold

	@property
	def trim_mode(self) -> bool:
		return self.__top_k_nodes is not None or self.__dynamic_k_threshold is not None

	def set_repository(self, repository: StateRepository):
		self._state_repository = repository

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

	def __init_current_graph(self, graph: 'MonteCarloAgent.Node'):
		self.__current_graph = graph

	def __get_current_graph(self) -> 'MonteCarloAgent.Node':
		if self.__current_graph is None:
			raise ValueError("Graph not set. Make sure __init_current_graph was called.")
		return self.__current_graph

	def _get_state_value(self, state):
		return self._get_environment().get_reward(state)

	def _get_random_state_node(self, action_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':
		state_nodes: List[MonteCarloAgent.Node] = action_node.get_children()

		if len(state_nodes) == 1:
			return state_nodes[0]

		probabilities = np.array([
			state_node.weight
			for state_node in state_nodes
		]).astype('float64')
		counts = np.array([
			state_node.visits
			for state_node in state_nodes
		]).astype('float64')

		if self.__probability_correction:
			probabilities = self.__correct_probabilities(probabilities, counts)
		else:
			probabilities = self.__squash_probabilities(probabilities)

		if self.__min_probability is not None:
			probabilities[probabilities < self.__min_probability] = 0.0
			probabilities = self.__squash_probabilities(probabilities)

		choice = np.random.choice(
			state_nodes,
			1,
			p=probabilities
		)[0]

		return choice

	def __squash_probabilities(self, probs: np.ndarray) -> np.ndarray:
		probs[probs <= 0] = self.__squash_epsilon
		return probs / probs.sum()

	def __correct_probabilities(self, expected: np.ndarray, counts: np.ndarray):
		total_counts = counts.sum()

		if total_counts == 0:
			return expected

		frequency_probs = counts / counts.sum()

		corrected = self.__squash_probabilities(
			(2 * expected) - frequency_probs
		)
		return corrected

	def __manage_resources(self, end=False):
		if psutil.virtual_memory().percent > (100 - self.__min_free_memory) or end:
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

	def __check_stm(self, node: 'MonteCarloAgent.Node') -> Optional['MonteCarloAgent.Node']:
		if self.__short_term_memory is None:
			return None

		memory: Optional[MonteCarloAgent.Node] = self.__short_term_memory.recall(node)
		if memory is None:
			return None
		return memory

	def __move_node(self, source: 'MonteCarloAgent.Node', destination: 'MonteCarloAgent.Node'):
		destination.children = source.children
		destination.visits = source.visits
		destination.total_value = source.total_value
		destination.weight = source.weight
		for child in source.children:
			child.parent = destination

	def __expand_from_stm(self, node: 'MonteCarloAgent.Node') -> bool:

		memorized = self.__check_stm(node)
		if memorized is None or not memorized.has_children():
			return False
		self.__move_node(memorized, node)

		return True

	def __store_to_stm(self, root: 'MonteCarloAgent.Node'):

		state_nodes = []
		for action_node in root.get_children():
			state_nodes.extend(action_node.get_children())

		for node in sorted(state_nodes, key=lambda n: n.weight, reverse=True)[:self.__short_term_memory.size]:
			self.__short_term_memory.memorize(node)

	def __get_children_states(self, state_node: 'MonteCarloAgent.Node') -> Dict[str, object]:
		if not state_node.has_children():
			return {state_node.id: self._state_repository.retrieve(state_node.id)}
		states = {}
		for action_node in state_node.get_children():
			for child in action_node.get_children():
				states.update(self.__get_children_states(child))

		return states

	def __backup_states(self) -> Dict[str, object]:
		states = {}
		for memory in self.__short_term_memory:
			states[memory.id] = self._state_repository.retrieve(memory.id)
			states.update(self.__get_children_states(memory))

		return states

	def __restore_backups(self, backup: Dict[str, object]):
		for key, state in backup.items():
			self._state_repository.store(key, state)

	def __backup_and_clear_repository(self):
		backup = self.__backup_states()
		self._state_repository.clear()
		self.__restore_backups(backup)

	def __dump_node(self, node):
		dump_path = os.path.abspath(os.path.join(self.__dump_path, f"{datetime.now().timestamp()}"))
		os.mkdir(dump_path)

		json_ = self.__serializer.serialize(node)
		with open(os.path.join(dump_path, "graph.json"), "w") as file:
			json.dump(json_, file)

		self._state_repository.dump(
			os.path.join(dump_path, "states.json"),
			keys=[node.id for node in stats.get_nodes(node, visited=True, include_root=True) if node.node_type == MonteCarloAgent.Node.NodeType.STATE]
		)

	def __finalize_step(self, root: 'MonteCarloAgent.Node'):

		if self.__dump_nodes:
			self.__dump_node(root)

		if self.__use_stm:
			self.__store_to_stm(root)
			self.__backup_and_clear_repository()
		else:
			self._state_repository.clear()
		self.__manage_resources(end=True)

	def _select(self, parent_state_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':

		promising_action_node: MonteCarloAgent.Node = max(
													parent_state_node.get_children(),
													key=lambda node: self._uct(node)
												)

		chosen_state_node: MonteCarloAgent.Node = self._get_random_state_node(promising_action_node)

		if len(chosen_state_node.get_children()) == 0:
			return chosen_state_node

		return self._select(chosen_state_node)

	def __collect_transition_inputs(self, state_node: 'MonteCarloAgent.Node') -> Tuple[List[ModelBasedState], List[Any], List[ModelBasedState], List['MonteCarloAgent.Node']]:
		states_inputs = []
		actions_inputs = []
		final_states_inputs = []
		state_nodes = []
		state: ModelBasedState = self._state_repository.retrieve(state_node.id)
		for action in self._get_available_actions(state):
			action_node = MonteCarloAgent.Node(state_node, action, MonteCarloAgent.Node.NodeType.ACTION)
			for final_state in self._get_possible_states(state, action):
				states_inputs.append(state)
				actions_inputs.append(action)
				final_states_inputs.append(final_state)

				final_state_node = MonteCarloAgent.Node(
					action_node,
					None,
					MonteCarloAgent.Node.NodeType.STATE,
				)
				action_node.add_child(final_state_node)
				state_nodes.append(final_state_node)
			state_node.add_child(action_node)

		return states_inputs, actions_inputs, final_states_inputs, state_nodes

	def __get_k(self, node: 'MonteCarloAgent.Node') -> int:
		if not self.__dynamic_k_threshold:
			return self.__top_k_nodes
		weights = np.array([child.weight for child in node.get_children()])
		weights = weights / np.sum(weights)
		importance = weights / np.max(weights)
		return np.sum(importance > self.__dynamic_k_threshold)

	def __trim_node(self, node: 'MonteCarloAgent.Node'):
		k = self.__get_k(node)
		node.children = sorted(node.children, key=lambda n: n.weight, reverse=True)[:k]

	def _expand(self, state_node: 'MonteCarloAgent.Node', stm=True):
		if self._get_environment().is_episode_over(self._state_repository.retrieve(state_node.id)):
			return

		if self.__use_stm and stm and self.__expand_from_stm(state_node):
			Logger.info("Recalled from STM")
			Logger.info(f"Recall Depth: {stats.get_max_depth(state_node)}")
			return

		states, actions, final_states, possible_state_nodes = self.__collect_transition_inputs(state_node)

		start_time = datetime.now()
		probability_distribution = self._get_expected_transition_probability_distribution(states, actions, final_states)
		stats.durations["get_expected_transition_probability_distribution"] += (datetime.now() - start_time).total_seconds()

		start_time = datetime.now()
		for i in range(len(probability_distribution)):
			possible_state_nodes[i].instant_value = self._get_expected_instant_reward(final_states[i])
			possible_state_nodes[i].weight = probability_distribution[i]
			possible_state_nodes[i].depth = state_node.depth + 1
			self._state_repository.store(possible_state_nodes[i].id, final_states[i])
			final_states[i].set_depth(possible_state_nodes[i].depth)
		stats.durations["for i in range(len(probability_distribution))"] += (
						datetime.now() - start_time).total_seconds()

		start_time = datetime.now()
		if self.trim_mode:
			for action_node in state_node.children:
				self.__trim_node(action_node)
		stats.durations["trim"] += (datetime.now() - start_time).total_seconds()

	def __simulate(self, state_node: 'MonteCarloAgent.Node') -> 'MonteCarloAgent.Node':

		if not state_node.has_children():
			return state_node

		action_node: MonteCarloAgent.Node = np.random.choice(state_node.get_children(), 1)[0]

		return self._get_random_state_node(action_node)

	def _get_action_node_value(self, node: 'MonteCarloAgent.Node'):
		total_weight = np.sum([state_node.weight for state_node in node.get_children()])
		return np.sum([
			state_node.get_total_value() * state_node.weight / total_weight
			for state_node in node.get_children()
		])

	def __legacy_backpropagate(self, node: 'MonteCarloAgent.Node', reward=None) -> None:
		node.increment_visits()
		if node.parent is None:
			return

		if reward is None:
			reward = node.instant_value
		node.add_value(self._get_discount_factor(node.depth) * reward)

		if node.node_type == MonteCarloAgent.Node.NodeType.ACTION:
			reward = reward * self._get_discount_factor(node.depth)

		self.__legacy_backpropagate(node.parent, reward)

	def __logical_backpropagate(self, node: 'MonteCarloAgent.Node') -> None:
		node.increment_visits()

		if node.node_type == MonteCarloAgent.Node.NodeType.STATE:
			node.set_total_value(node.instant_value)
			if node.has_children():
				node.add_value(self._get_discount_factor(node.depth) * max([action_node.get_total_value() for action_node in node.get_children()]))

		else:
			node.set_total_value(
				self._get_action_node_value(node)
			)

		if node.parent is None:
			return

		self.__logical_backpropagate(node.parent)

	def _monte_carlo_simulation(self, root_node: 'MonteCarloAgent.Node'):
		self._expand(root_node)

		resources = self._init_resources()

		stats.iterations["main_loop"] = 0
		while self._has_resource(resources):
			start_time = datetime.now()
			leaf_node = self._select(root_node)
			stats.durations["select"] += (datetime.now() - start_time).total_seconds()

			start_time = datetime.now()
			self._expand(leaf_node, stm=False)
			stats.durations["expand"] += (datetime.now() - start_time).total_seconds()

			start_time = datetime.now()
			final_node = self.__simulate(leaf_node)
			stats.durations["simulate"] += (datetime.now() - start_time).total_seconds()

			self._backpropagate(final_node)

			self.__manage_resources()
			if random.random() < 0.1:
				stats.draw_graph_live(root_node, visited=True, state_repository=self._state_repository, uct_fn=self._uct)
			stats.iterations["main_loop"] += 1

	def _monte_carlo_tree_search(self, state) -> None:
		root_node = MonteCarloAgent.Node(None, None, MonteCarloAgent.Node.NodeType.STATE)
		self._state_repository.store(root_node.id, state)
		self.__init_current_graph(root_node)

		self._monte_carlo_simulation(root_node)

		Logger.info(
			f"Simulations Done: Iterations: {stats.iterations['main_loop']}, "
			f"Depth: {stats.get_max_depth(root_node)}, "
			f"Nodes: {len(stats.get_nodes(root_node))}"
		)
		optimal_action = max(root_node.get_children(), key=lambda node: node.get_total_value()).action
		Logger.info(f"Best Action {optimal_action}")
		self.__finalize_step(root_node)

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		for action_node in self.__get_current_graph().get_children():
			if action_node.action == action:
				return action_node.get_total_value()

		raise Exception(f"Action Not Found in Graph. Action={action}")

	def _get_optimal_action(self, state, **kwargs):
		self._monte_carlo_tree_search(state)
		return super()._get_optimal_action(state, **kwargs)

	def _explore(self, state):
		resource = self._init_resources()
		while self._has_resource(resource):
			time.sleep(0.5)
		return super()._explore(state)