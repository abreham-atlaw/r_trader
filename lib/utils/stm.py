from typing import *
from abc import ABC, abstractmethod


class CueMemoryMatcher(ABC):

	@abstractmethod
	def is_match(self, cue: object, memory: object) -> bool:
		pass


class ExactCueMemoryMatcher(CueMemoryMatcher):

	def is_match(self, cue: object, memory: object) -> bool:
		return cue == memory


class StochasticCueMemoryMatcher(CueMemoryMatcher, ABC):

	def __init__(self, threshold: float):
		self._threshold = threshold

	@abstractmethod
	def _evaluate(self, cue: object, memory: object) -> float:
		pass

	def is_match(self, cue: object, memory: object) -> bool:
		return self._evaluate(cue, memory) > self._threshold


class ShortTermMemory:

	class EmptyMemory:
		pass

	def __init__(self, size, matcher: CueMemoryMatcher):
		self.size = size
		self.__memories: List[Optional[object]] = None
		self._matcher = matcher
		self.clear()

	def _export_memory(self, memory: object) -> object:
		return memory

	def _import_memory(self, memory: object) -> object:
		return memory

	def __sort_memories(self, memories: List[object], recall_index=None) -> List[object]:
		return self._sort_memory(
			[memory for memory in memories if not isinstance(memory, ShortTermMemory.EmptyMemory)],
			recall_index=recall_index
		) + [memory for memory in memories if isinstance(memory, ShortTermMemory.EmptyMemory)]

	def _sort_memory(self, memories: List[object], recall_index=None) -> List[object]:
		return memories

	def recall(self, cue) -> Optional[object]:

		for i, memory in enumerate(self.__memories):
			if (not isinstance(memory, ShortTermMemory.EmptyMemory)) and self._matcher.is_match(cue, memory):
				self.__memories = self.__sort_memories(self.__memories, i)
				return self._export_memory(memory)

		return None

	def memorize(self, memory):
		self.__memories.pop()
		self.__memories.insert(
			0,
			self._import_memory(memory)
		)
		self.__memories = self.__sort_memories(self.__memories)

	def clear(self):
		self.__memories = [ShortTermMemory.EmptyMemory() for _ in range(self.size)]

	def __iter__(self):
		for memory in self.__memories:
			yield self._export_memory(memory)

	def __getitem__(self, idx) -> object:
		return self._export_memory(self.__memories[idx])
