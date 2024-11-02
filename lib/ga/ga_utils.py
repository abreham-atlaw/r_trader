import os.path
import typing

from lib.ga import Species
from lib.ga.serializers import PopulationSerializer
from lib.network.rest_interface import Serializer
from lib.utils.file_storage import FileStorage
from lib.utils.fileio import PickleFileIO, SerializerFileIO


class GAUtils:

	@staticmethod
	def load_population(path: str, serializer: Serializer = None) -> typing.List[Species]:

		file_io = PickleFileIO()
		if serializer is not None:
			file_io = SerializerFileIO(PopulationSerializer(serializer))
		return file_io.loads(path)

	@staticmethod
	def load_from_fs(path: str, fs: FileStorage, serializer: Serializer = None) -> typing.List[Species]:
		download_path = os.path.basename(path)
		fs.download(path, download_path)
		return GAUtils.load_population(download_path, serializer)
