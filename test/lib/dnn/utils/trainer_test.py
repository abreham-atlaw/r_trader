from typing import *

import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense

import os

from lib.dnn.utils import KerasTrainer


class KerasTrainerTest(unittest.TestCase):

	class TestKerasTrainer(KerasTrainer):

		def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
			data = data.dropna()
			return data.drop(columns=["quality"]).to_numpy(), data["quality"].to_numpy()

		def _create_model(self) -> keras.Model:
			model = Sequential()
			model.add(Input(shape=(11,)))
			model.add(Dense(32, activation="relu"))
			model.add(Dense(16, activation="relu"))
			model.add(Dense(4, activation="relu"))
			model.add(Dense(1, activation="relu"))

			return model

		def _compile_model(self, model: keras.Model):
			model.compile(
				optimizer=keras.optimizers.adam_v2.Adam(),
				loss=keras.losses.mse
			)
		
	DATA_PATH = "test/lib/dnn/utils/data/wine_quality.csv"
	EXPORT_PATH = "test/lib/dnn/utils/model.h5"
	EPOCHES = 120
	BATCH_SIZE = 2
	FIT_PARAMS = {"verbose": 1, "shuffle": True}


	@patch("pandas.read_csv")
	def test_correct_calls(self, read_csv_mock):
		
		trainer = KerasTrainerTest.TestKerasTrainer(
			KerasTrainerTest.DATA_PATH,
			KerasTrainerTest.EXPORT_PATH,
			epochs=KerasTrainerTest.EPOCHES,
			batch_size=KerasTrainerTest.BATCH_SIZE,
			fit_params=KerasTrainerTest.FIT_PARAMS
			)
		data = read_csv_mock.return_value = mock.Mock()
		trainer._prepare_data = mock.Mock()
		prepared_data = trainer._prepare_data.return_value = mock.Mock(), mock.Mock()
		trainer._split_data = mock.Mock()
		split_data = trainer._split_data.return_value = mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
		trainer._create_model = mock.Mock()
		model = trainer._create_model.return_value = mock.Mock()
		trainer.start()
		trainer._prepare_data.assert_called_once_with(data)
		trainer._split_data.assert_called_once_with(prepared_data[0], prepared_data[1])
		trainer._create_model.assert_called_once()
		model.fit.assert_called_once_with(
			split_data[0],
			split_data[2],
			epochs=KerasTrainerTest.EPOCHES,
			batch_size=KerasTrainerTest.BATCH_SIZE,
			**KerasTrainerTest.FIT_PARAMS
		)
		model.evaluate.assert_called_once_with(
			split_data[1],
			split_data[3]
		)
		model.save.assert_called_once_with(
			KerasTrainerTest.EXPORT_PATH
		)

	def test_practical_run(self):
		trainer = KerasTrainerTest.TestKerasTrainer(
			KerasTrainerTest.DATA_PATH,
			KerasTrainerTest.EXPORT_PATH,
			epochs=KerasTrainerTest.EPOCHES,
			batch_size=KerasTrainerTest.BATCH_SIZE,
		)

		history, test_history = trainer.start()
		self.assertIsNotNone(history)
		self.assertIsNotNone(test_history)
		print(f"history, test_history: {history}, {test_history}")
		self.assertTrue(os.path.exists(KerasTrainerTest.EXPORT_PATH))

if __name__ == "__main__":
	unittest.main()