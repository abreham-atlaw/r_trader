from abc import ABC, abstractmethod


class SpinozaDataLoader(ABC):

	@abstractmethod
	def __iter__(self):
		pass

	@abstractmethod
	def __len__(self):
		pass

	@abstractmethod
	def shuffle(self):
		pass
