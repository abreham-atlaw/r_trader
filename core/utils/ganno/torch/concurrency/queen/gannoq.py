from core.utils.ganno.torch.concurrency.data.serializers import CNNConfigSerializer
from core.utils.ganno.torch.optimizer import CNNOptimizer
from lib.concurrency.ga.queen import GAQueen
from lib.network.rest_interface import Serializer


class CNNOptimizerQueen(GAQueen, CNNOptimizer):

	def _init_serializer(self) -> Serializer:
		return CNNConfigSerializer()

