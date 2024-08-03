from core import Config
from lib.utils.file_storage import FileStorage, PCloudCombinedFileStorage


class ServiceProvider:

	@staticmethod
	def provide_file_storage(path=None) -> FileStorage:
		if path is None:
			path = Config.PCLOUD_FOLDER
		return PCloudCombinedFileStorage(
			tokens=Config.PCLOUD_TOKENS,
			base_path=path
		)
