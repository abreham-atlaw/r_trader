import time
import typing

from lib.utils.logger import Logger
from .data.models import Resources
from .resource_manager import ResourcesManager, ResourceUnavailableException
from .sessions_manager import SessionsManager


class FusedManager(SessionsManager):

	def __init__(self, resources_manager: ResourcesManager, *args, confirm_timeout=30, **kwargs):
		super().__init__(*args, **kwargs)
		self.__resources_manager = resources_manager
		self.__downgrade_map = {
			Resources.Devices.GPU: Resources.Devices.TPU,
			Resources.Devices.TPU: Resources.Devices.CPU
		}
		self.__confirm_timeout = confirm_timeout

	def __downgrade_device(self, device, allowed_devices: typing.Optional[typing.List[int]]):
		if allowed_devices is None:
			allowed_devices = list(self.__downgrade_map.values())
		return {
			key: value
			for key, value in self.__downgrade_map.items()
			if value in allowed_devices
		}.get(device)

	def start_session(
			self,
			kernel: str,
			meta_data: typing.Dict[str, typing.Any],
			device: int = Resources.Devices.CPU,
			raise_exception=False,
			sync_notebooks=True,
			allowed_devices: typing.Optional[typing.List[int]] = None,
			confirm: bool = True
	):
		if sync_notebooks:
			self.sync_notebooks()
		try:
			account = self.__resources_manager.allocate_notebook(device=device)
		except ResourceUnavailableException:
			device = self.__downgrade_device(device, allowed_devices)
			if device is None:
				if raise_exception:
					raise ResourceUnavailableException()
				print("[-]Resource Unavailable. Exiting...")
				return
			return self.start_session(
				kernel=kernel,
				meta_data=meta_data,
				device=device,
				raise_exception=raise_exception,
				sync_notebooks=False,
				allowed_devices=allowed_devices
			)

		super().start_session(
			kernel=kernel,
			account=account,
			meta_data=meta_data,
			device=device,
			sync_notebooks=False
		)

		if confirm:
			time.sleep(self.__confirm_timeout)
			Logger.info(f"Confirming run...")
			if not self.is_notebook_running(kernel):
				Logger.warning(f"Notebook {kernel} is not running. Retrying...")
				return self.start_session(
					kernel=kernel,
					meta_data=meta_data,
					device=device,
					raise_exception=raise_exception,
					sync_notebooks=False,
					allowed_devices=allowed_devices
				)
			Logger.success(f"Notebook {kernel} run successful on {account.username}(device={device})")
