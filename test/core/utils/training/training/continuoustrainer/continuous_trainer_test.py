import typing

import numpy as np
from tensorflow import keras

import unittest
from unittest.mock import MagicMock

from core.utils.training.training import Trainer
from lib.utils.file_storage import PCloudClient
from core.utils.training.datapreparation.dataprocessor import DataProcessor
from core.utils.training.training.continuoustrainer import ContinuousTrainer
from core.utils.training.training.continuoustrainer.callbacks import PCloudContinuousTrainerCheckpointCallback
from core.utils.training.training.continuoustrainer.repository import PCloudTrainerRepository


class ContinuousTrainerTest(unittest.TestCase):

	__ID = "5"
	__EPOCH = 24
	__STATE = Trainer.State(__EPOCH, 0, 0, 0)

	def setUp(self) -> None:
		return
		# self.__create_checkpoint()

	def __create_checkpoint(self) -> typing.Tuple[str, str]:
		core_model, delta_model = self.__generate_model(), self.__generate_model(True)
		repository = PCloudTrainerRepository("/Apps/RTrader")

		callback = PCloudContinuousTrainerCheckpointCallback(base_path="/Apps/RTrader")
		callback.init(self.__ID, repository)
		callback.on_epoch_end(core_model, delta_model, self.__STATE)

	def __generate_model(self, delta=False):
		model = keras.Sequential()
		model.add(keras.layers.InputLayer(101 + int(delta)))
		model.add(keras.layers.Dense(10))
		model.add(keras.layers.Dense(1))
		model.compile(optimizer="adam", loss="mse")
		return model

	def test_functionality(self):

		generator = MagicMock()
		generator.__getitem__ = MagicMock(return_value=np.sin(np.arange(1100)).reshape((10, 110)))
		generator.__len__ = MagicMock(return_value=30)
		core_model, delta_model = self.__generate_model(), self.__generate_model(True)
		processor = DataProcessor(generator, core_model, delta_model, 32, 32)
		trainer = ContinuousTrainer(
			repository=PCloudTrainerRepository("/Apps/RTrader")
		)
		trainer.fit(
			id=self.__ID,
			core_model=core_model,
			delta_model=delta_model,
			processor=processor,
			depth=1,
			epochs=50,
			callbacks=[
				PCloudContinuousTrainerCheckpointCallback(base_path="/Apps/RTrader", batch_end=False)
			],
			timeout=2*60
		)
