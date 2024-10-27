from lib.utils.stm import ShortTermMemory

from .node_memory import NodeMemory
from .node_memory_matcher import NodeMemoryMatcher
from ..node import Node


class NodeShortTermMemory(ShortTermMemory):

	def _import_memory(self, memory: Node) -> NodeMemory:
		return NodeMemory(memory)

	def _export_memory(self, memory: NodeMemory) -> object:
		return memory.node

	def set_matcher(self, matcher: NodeMemoryMatcher):
		self._matcher = matcher

	def get_matcher(self) -> NodeMemoryMatcher:
		return self._matcher
