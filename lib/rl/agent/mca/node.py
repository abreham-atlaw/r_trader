import typing

import uuid


class Node:

	class NodeType:
		STATE = 0
		ACTION = 1

	def __init__(
			self,
			parent,
			action,
			node_type,
			depth: int = 0,
			weight: float = 1.0,
			instant_value: float = 0.0,
			id=None
	):
		self.children = []
		self.visits = 0
		self.total_value = 0
		self.instant_value = instant_value
		self.parent: Node = parent
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

	def get_children(self) -> typing.List['Node']:
		return self.children

	def get_visits(self):
		return self.visits

	def get_total_value(self):
		return self.total_value

	@staticmethod
	def __generate_id():
		return uuid.uuid4().hex

	def find_node_by_id(self, id) -> typing.Union['Node', None]:
		if self.id == id:
			return self
		for child in self.get_children():
			result = child.find_node_by_id(id)
			if result is not None:
				return result
		return None

	def __eq__(self, other):
		return isinstance(other, Node) and self.id == other.id
