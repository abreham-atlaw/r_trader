import typing

from .resource_manager import ResourcesManager, ResourceUnavailableException
from .sessions_manager import SessionsManager


class FusedManager(SessionsManager):

	def __init__(self, resources_manager: ResourcesManager, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__resources_manager = resources_manager

	def start_session(
			self,
			kernel: str,
			meta_data: typing.Dict[str, typing.Any],
			gpu=False,
			raise_exception=False,
			sync_notebooks=True
	):
		if sync_notebooks:
			self.sync_notebooks()
		try:
			account = self.__resources_manager.allocate_notebook(gpu)
		except ResourceUnavailableException:
			if not gpu:
				if raise_exception:
					raise ResourceUnavailableException()
				return
			gpu = False
			try:
				account = self.__resources_manager.allocate_notebook(gpu)
			except ResourceUnavailableException:
				if raise_exception:
					raise ResourceUnavailableException()
				print("[-]Resource Unavailable. Exiting...")
				return

		super().start_session(kernel, account, meta_data, gpu, sync_notebooks=False)
