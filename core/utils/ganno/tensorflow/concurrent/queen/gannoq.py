from abc import ABC
from typing import *

from lib.concurrency.ga.queen import GAQueen
from lib.network.rest_interface import Serializer
from core.utils.ganno import NNGeneticAlgorithm
from core.utils.ganno.concurrent.data.serializers import NNConfigSerializer


class GannoQueen(GAQueen, NNGeneticAlgorithm, ABC):

	def __init__(self, *args, **kwargs):
		super().__init__(
			*args,
			trainer=None,
			**kwargs
		)

	def _init_serializer(self) -> Serializer:
		return NNConfigSerializer()
