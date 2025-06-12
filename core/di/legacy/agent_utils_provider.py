from core import Config
from lib.network.rest_interface import Serializer
from lib.rl.agent.mca.resource_manager import MCResourceManager, TimeMCResourceManager
from lib.utils.logger import Logger
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
			Logger.info("Using auto state repository...")
			return AutoStateRepository(
				Config.AGENT_AUTO_STATE_REPOSITORY_MEMORY_SIZE,
				in_memory_repository=AgentUtilsProvider.provide_in_memory_state_repository(),
				disk_repository=AgentUtilsProvider.provide_disk_state_repository()
			)
		Logger.info("Using in-memory state repository...")
		return AgentUtilsProvider.provide_in_memory_state_repository()

	@staticmethod
	def provide_resource_manager() -> MCResourceManager:
		from core.agent.agents.montecarlo_agent.trader_resource_manager import TraderMCResourceManager
		from .environment_utils_provider import EnvironmentUtilsProvider

		if Config.AGENT_USE_CUSTOM_RESOURCE_MANAGER:
			manager = TraderMCResourceManager(
				trader=EnvironmentUtilsProvider.provide_trader(),
				granularity=Config.MARKET_STATE_GRANULARITY,
				instrument=Config.AGENT_STATIC_INSTRUMENTS[0],
				delta_multiplier=Config.OANDA_SIM_DELTA_MULTIPLIER
			)
		else:
			manager = TimeMCResourceManager(
				step_time=Config.AGENT_STEP_TIME
			)
		Logger.info(f"Using Resource Manager: {manager.__class__.__name__}")
		return manager
	