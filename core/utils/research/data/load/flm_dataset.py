from torch.utils.data import Dataset

from core.utils.research.data.load.flm import FileLoadManager


class FLMDataset(Dataset):

	def __init__(
			self,
			flm: FileLoadManager,
	):
		self.__file_size = flm.fileloader.file_size
		self.__flm = flm
		self.__flm.start()

	def __getitem__(self, idx):
		file_idx = idx // self.__file_size
		data_idx = idx % self.__file_size

		X, y = self.__flm[file_idx]

		return tuple([dp[data_idx] for dp in [X, y]])

	def __len__(self):
		return len(self.__flm.fileloader) * self.__file_size
