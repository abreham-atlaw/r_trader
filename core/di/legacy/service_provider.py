from core import Config
from lib.utils.file_storage import FileStorage, PCloudClient


class ServiceProvider:

	@staticmethod
	def provide_file_storage() -> FileStorage:
		return PCloudClient(
			token=Config.PCLOUD_API_TOKEN,
			folder=Config.PCLOUD_FOLDER
		)
