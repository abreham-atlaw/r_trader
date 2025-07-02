import typing

import shutil

from lib.utils.logger import Logger
from .resource_manager import MCResourceManager


class DiskResourceManager(MCResourceManager):

	def __init__(self, min_remaining_space=0.1):
		super().__init__()
		Logger.info(f"Initializing Disk Resource Manager with min_remaining_space={min_remaining_space}")
		self.__min_remaining_space = min_remaining_space

	@staticmethod
	def __get_disk_space() -> float:
		_, __, free = shutil.disk_usage(".")
		return free/(2**30)

	def init_resource(self) -> float:
		resource = self.__get_disk_space()
		Logger.info(f"Initial Disk Space: {resource: .2f} GiB")
		return resource

	def has_resource(self, resource: float) -> bool:
		space = self.__get_disk_space()
		percentage = space/resource
		available = percentage > self.__min_remaining_space
		if not available:
			Logger.warning(f"Disk space too low. Only {space: .2f} GiB({percentage*100: .2f}% of {resource: .2f} GiB) remaining.")
		return available
