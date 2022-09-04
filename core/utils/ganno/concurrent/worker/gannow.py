from typing import *

from lib.concurrency.ga.worker import GAWorker
from lib.ga import Species
from lib.network.rest_interface import Serializer
from core.utils.ganno import NNGeneticAlgorithm
from core.utils.ganno.nnconfig import NNConfig
from core.utils.ganno.concurrent.data.serializers import NNConfigSerializer


class GannoWorker(GAWorker, NNGeneticAlgorithm):

	def _init_serializer(self) -> Serializer:
		return NNConfigSerializer()
