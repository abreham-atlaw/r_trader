from typing import *

from tensorflow.keras import losses, optimizers, activations

from lib.network.rest_interface import Serializer
from core.utils.ganno.nnconfig import NNConfig, ModelConfig, ConvPoolLayer


class ConvPoolLayerSerializer(Serializer):

	def __init__(self):
		super().__init__(ConvPoolLayer)

	def serialize(self, data: ConvPoolLayer) -> Dict:
		return data.__dict__.copy()

	def deserialize(self, json_: Dict) -> ConvPoolLayer:
		obj = ConvPoolLayer(*[None for _ in range(3)])
		obj.__dict__ = json_.copy()
		return obj


class ModelConfigSerializer(Serializer):

	def __init__(self):
		super().__init__(ModelConfig)
		self.__conv_serializer = ConvPoolLayerSerializer()

	def serialize(self, data: ModelConfig) -> Dict:
		json = data.__dict__.copy()
		json["ff_conv_pool_layers"] = [
			self.__conv_serializer.serialize(layer)
			for layer in data.ff_conv_pool_layers
		]
		json["loss"] = losses.serialize(data.loss)
		json["optimizer"] = optimizers.serialize(data.optimizer)
		json["dense_activation"] = activations.serialize(data.dense_activation)
		json["conv_activation"] = activations.serialize(data.conv_activation)

		return json

	def deserialize(self, json_: Dict) -> ModelConfig:
		config = ModelConfig(*[None for _ in range(15)])
		config.__dict__ = json_.copy()
		config.ff_conv_pool_layers = [
			self.__conv_serializer.deserialize(layer)
			for layer in config.ff_conv_pool_layers
		]
		config.loss = losses.deserialize(config.loss)
		config.optimizer = optimizers.deserialize(config.optimizer)
		config.dense_activation = activations.deserialize(config.dense_activation)
		config.conv_activation = activations.deserialize(config.conv_activation)

		return config


class NNConfigSerializer(Serializer):

	def __init__(self):
		super().__init__(NNConfig)
		self.__mc_serializer = ModelConfigSerializer()

	def serialize(self, data: NNConfig) -> Dict:
		return {
			"core_config": self.__mc_serializer.serialize(data.core_config),
			"delta_config": self.__mc_serializer.serialize(data.delta_config)
		}

	def deserialize(self, json_: Dict) -> NNConfig:
		return NNConfig(
			core_config=self.__mc_serializer.deserialize(json_["core_config"]),
			delta_config=self.__mc_serializer.deserialize(json_["delta_config"])
		)
