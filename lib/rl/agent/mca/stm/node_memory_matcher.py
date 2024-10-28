import typing

from lib.rl.agent.mca.node import Node
from lib.rl.agent.mca.stm.node_memory import NodeMemory
from lib.utils.staterepository import StateRepository
from lib.utils.stm import CueMemoryMatcher, ExactCueMemoryMatcher


class NodeMemoryMatcher(CueMemoryMatcher):

	def __init__(
			self,
			state_matcher: typing.Optional[CueMemoryMatcher] = None,
			repository: typing.Optional[StateRepository] = None
	):
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

	def is_match(self, cue: Node, memory: NodeMemory) -> bool:
		return self.__state_matcher.is_match(
			self.get_repository().retrieve(cue.id),
			self.get_repository().retrieve(memory.node.id)
		)
