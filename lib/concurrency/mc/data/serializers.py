from typing import *
from abc import ABC, abstractmethod

from lib.network.rest_interface.serializers import Serializer
from lib.rl.agent import MonteCarloAgent


class NodeSerializer(Serializer, ABC):
	
	def __init__(self, action_serializer=None):
		super(NodeSerializer, self).__init__(MonteCarloAgent)
		self.__action_serializer = self._init_action_serializer()

	@abstractmethod
	def _init_action_serializer(self) -> Serializer:
		pass

	def serialize(self, node: MonteCarloAgent.Node) -> Dict:
		json = node.__dict__.copy()
		json.pop("parent")
		json["children"] = [
			self.serialize(child)
			for child in node.get_children()
		]
		json["action"] = self.__action_serializer.serialize(node.action)
		return json

	def deserialize(self, json_: Dict, parent=None) -> MonteCarloAgent.Node:
		node = MonteCarloAgent.Node(None, None, None, None)
		node.__dict__ = json_.copy()
		node.parent = parent
		node.children = [
			self.deserialize(child_json, parent=node)
			for child_json in json_["children"]
		]
		node.action = self.__action_serializer.deserialize(json_["action"])
		return node
