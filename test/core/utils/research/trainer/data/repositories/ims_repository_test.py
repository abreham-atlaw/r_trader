import unittest

import numpy as np
from matplotlib import pyplot as plt

from core.di import ResearchProvider
from core.utils.research.training.data.repositories.ims_repository import IMSRepository
from core.utils.research.training.trackers.stats_tracker import Keys
from lib.utils.torch_utils.model_handler import ModelHandler


class IMSRepositoryTest(unittest.TestCase):

	def setUp(self):
		model = ModelHandler.load(
			"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-110-cum-0-it-4-tot.zip"
		)
		self.model_name = ModelHandler.generate_signature(model)
		self.labels = IMSRepository.retrieve_labels(self.model_name)
		self.label = self.labels[1]
		self.repository: IMSRepository = ResearchProvider.provide_ims_repository(
			model_name=self.model_name,
			label=f"gradients_"
		)
		print(f"Using model_name: {self.model_name}")
		print(f"Using labels: {', '.join(self.labels)}")
		print(f"Using label: {self.label}")


	def test_store(self):
		for i in range(10):
			self.repository.store(
				value=np.random.random((10, 100)),
				epoch=0,
				batch=1
			)
		self.repository.sync()

	def test_retrieve(self):
		df = self.repository.retrieve()
		print(df)

	def test_retrieve_X(self):
		repository: IMSRepository = ResearchProvider.provide_ims_repository(
			model_name=self.model_name,
			label="X"
		)

		df = repository.retrieve()
		print(df)

	def test_retrieve_from_signature(self):

		repository: IMSRepository = ResearchProvider.provide_ims_repository(self.model_name, self.label)

		df = repository.retrieve()
		print(df)

		self.assertTrue(df.shape[0] > 0)

		for col in df.columns:
			plt.figure()
			plt.title(col)
			plt.plot(df[col].to_numpy())

		plt.show()

	def test_get_labels(self):

		labels = IMSRepository.retrieve_labels(self.model_name)
		print(f"Labels: {', '.join(labels)}")

	def test_plot_gradients_stat(self):
		stat = "mean_value"

		for label in filter(lambda l: l.startswith("gradients_"), self.labels):
			repository = ResearchProvider.provide_ims_repository(self.model_name, label)
			df = repository.retrieve()

			plt.figure()
			plt.title(label)
			plt.plot(df[stat].to_numpy())
		plt.show()
