import unittest
from threading import Thread
from multiprocessing import Process

from core.utils.research.eval.live_prediction.live_predictor import LivePredictor


class LivePredictorProcess(Process):

	def __init__(self, model_path: str, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__predictor = LivePredictor(model_path)

	def run(self):
		self.__predictor.start()

class LivePredictorTest(unittest.TestCase):

	def test_functionality(self):
		predictor = LivePredictor(
			model_path="/home/abrehamatlaw/Downloads/Compressed/albertcamus0-rtrader-training-cnn-111-cum-0-it-0-tot_1.zip",
			sleep_time=1*60,
			delay_mode=True,
			prediction_window=1
		)

		predictor.start()

	def test_multiple_models(self):

		MODELS = [
			"/home/abrehamatlaw/Downloads/Compressed/abrehamatlaw0-drmca-cnn-91-cum-0.zip",
			"/home/abrehamatlaw/Downloads/abrehamatlaw0-drmca-cnn-111-tot-cum-0.zip",
			# "/home/abrehamatlaw/Downloads/Compressed/abrehamatlaw0-drmca-cnn-125-cum-0.zip"
		]

		processes = []
		for model in MODELS:
			process = LivePredictorProcess(model)
			process.start()
			processes.append(process)

		for process in processes:
			process.join()
