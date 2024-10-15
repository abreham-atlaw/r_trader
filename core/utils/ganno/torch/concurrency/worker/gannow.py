from core.utils.ganno.torch.concurrency.data.serializers import CNNConfigSerializer, LinearConfigSerializer
from core.utils.ganno.torch.optimizer import CNNOptimizer, LinearOptimizer
from lib.concurrency.ga.queen import GAQueen
from lib.concurrency.ga.worker import GAWorker
from lib.network.rest_interface import Serializer


class CNNOptimizerWorker(GAWorker, CNNOptimizer):

	def _init_serializer(self) -> Serializer:
		return CNNConfigSerializer()


class LinearOptimizerWorker(GAWorker, LinearOptimizer):

	def _init_serializer(self) -> Serializer:
		return LinearConfigSerializer()
