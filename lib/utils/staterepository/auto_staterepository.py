import typing

from .staterepository import StateRepository, StateNotFoundException
from .file_staterepository import PickleStateRepository
from .dict_staterepository import SectionalDictStateRepository


class AutoStateRepository(StateRepository):

	def __init__(
			self,
			memory_size: int,
			in_memory_repository: StateRepository = None,
			disk_repository: StateRepository = None,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		if in_memory_repository is None:
			in_memory_repository = SectionalDictStateRepository(memory_size, serializer=self._serializer)
		if disk_repository is None:
			disk_repository = PickleStateRepository(serializer=self._serializer)

		self._in_memory_repository = in_memory_repository
		self._disk_repository = disk_repository
		self._memory_size = memory_size
		self._queue = []

	def __move_to_disk(self):
		key = self._queue.pop(0)
		self._disk_repository.store(key, self._in_memory_repository.retrieve(key))
		self._in_memory_repository.remove(key)

	def store(self, key: str, state):
		if len(self._queue) >= self._memory_size:
			self.__move_to_disk()

		self._in_memory_repository.store(key, state)
		self._queue.append(key)

	def retrieve(self, key: str) -> object:
		try:
			return self._in_memory_repository.retrieve(key)
		except StateNotFoundException:
			return self._disk_repository.retrieve(key)

	def exists(self, key: str) -> bool:
		return self._in_memory_repository.exists(key) or self._disk_repository.exists(key)

	def clear(self):
		[
			repo.clear()
			for repo in [self._in_memory_repository, self._disk_repository]
		]
		self._queue = []

	def destroy(self):
		[
			repo.destroy()
			for repo in [self._in_memory_repository, self._disk_repository]
		]
		self._queue = []

	def get_keys(self) -> typing.List[str]:
		return self._in_memory_repository.get_keys() + self._disk_repository.get_keys()
