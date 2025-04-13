from core import Config
from lib.network.rest_interface import Serializer
from lib.utils.staterepository import StateRepository, AutoStateRepository, SectionalDictStateRepository, \
	PickleStateRepository


class AgentUtilsProvider:

	@staticmethod
	def provide_in_memory_state_repository() -> StateRepository:
		return SectionalDictStateRepository(2, 15)

	@staticmethod
	def provide_disk_state_repository() -> StateRepository:
		return PickleStateRepository()

	@staticmethod
	def provide_state_repository() -> StateRepository:
		if Config.AGENT_USE_AUTO_STATE_REPOSITORY:
			return AutoStateRepository(
				Config.AGENT_AUTO_STATE_REPOSITORY_MEMORY_SIZE,
				in_memory_repository=AgentUtilsProvider.provide_in_memory_state_repository(),
				disk_repository=AgentUtilsProvider.provide_disk_state_repository()
			)
		return AgentUtilsProvider.provide_in_memory_state_repository()
