from core import Config
from lib.utils.file_storage import FileStorage, PCloudCombinedFileStorage


class ServiceProvider:

	@staticmethod
	def provide_file_storage() -> FileStorage:
		return PCloudCombinedFileStorage(
			tokens=Config.PCLOUD_TOKENS,
			base_path=Config.PCLOUD_FOLDER
		)
