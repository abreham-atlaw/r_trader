import typing

from .data.models import Resources
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
			device: int = Resources.Devices.CPU,
			raise_exception=False,
			sync_notebooks=True
	):
		if sync_notebooks:
			self.sync_notebooks()


		try:
			account = self.__resources_manager.allocate_notebook(device=device)
		except ResourceUnavailableException:
			if device == Resources.Devices.CPU:
				if raise_exception:
					raise ResourceUnavailableException()
				print("[-]Resource Unavailable. Exiting...")
				return
			device = {
				Resources.Devices.GPU: Resources.Devices.TPU,
				Resources.Devices.TPU: Resources.Devices.CPU
			}.get(device)
			return self.start_session(kernel, meta_data, device, raise_exception, sync_notebooks=False)

		super().start_session(kernel, account, meta_data, device=device, sync_notebooks=False)
