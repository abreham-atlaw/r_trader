from core.utils.research.data.load.dataloader.dataloader import SpinozaDataLoader
from core.utils.research.data.load.flm.file_loader import FileLoader


class FileDataLoader(SpinozaDataLoader):

	def __init__(self, file_loader: FileLoader):
		self.file_loader = file_loader

	def __iter__(self):
		for i in range(len(self.file_loader)):
			yield self.file_loader[i]

	def __len__(self):
		return len(self.file_loader)
