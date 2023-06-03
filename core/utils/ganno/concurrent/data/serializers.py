from typing import *

from tensorflow.keras import losses, optimizers, activations

from lib.network.rest_interface import Serializer
from core.utils.ganno.nnconfig import NNConfig, ModelConfig, ConvPoolLayer, KalmanFiltersConfig, TransformerConfig


class ConvPoolLayerSerializer(Serializer):

	def __init__(self):
		super().__init__(ConvPoolLayer)

	def serialize(self, data: ConvPoolLayer) -> Dict:
		return data.__dict__.copy()

	def deserialize(self, json_: Dict) -> ConvPoolLayer:
		obj = ConvPoolLayer(*[None for _ in range(4)])
		obj.__dict__ = json_.copy()
		return obj


class KalmanFilterConfigSerializer(Serializer):

	def __init__(self):
		super().__init__(KalmanFiltersConfig)

	def serialize(self, data: object) -> Dict:
		return data.__dict__.copy()

	def deserialize(self, json_: Dict) -> object:
		obj = KalmanFiltersConfig(None, None)
		obj.__dict__ = json_.copy()
		return obj


class TransformerConfigSerializer(Serializer):

	def __init__(self):
		super().__init__(TransformerConfig)

	def serialize(self, data: TransformerConfig) -> Dict:
		json_ = data.__dict__.copy()
		json_["dense_activation"] = activations.serialize(data.dense_activation)
		return json_

	def deserialize(self, json_: Dict) -> TransformerConfig:
		obj = TransformerConfig(None, None, None, None, None)
		obj.__dict__ = json_.copy()
		obj.dense_activation = activations.deserialize(obj.dense_activation)
		return obj


class ModelConfigSerializer(Serializer):

	def __init__(self):
		super().__init__(ModelConfig)
		self.__conv_serializer = ConvPoolLayerSerializer()
		self.__kalmanconfigserializer = KalmanFilterConfigSerializer()
		self.__transformer_serializer = TransformerConfigSerializer()

	def serialize(self, data: ModelConfig) -> Dict:
		json = data.__dict__.copy()
		json["ff_conv_pool_layers"] = [
			self.__conv_serializer.serialize(layer)
			for layer in data.ff_conv_pool_layers
		]
		json["kalman_filters"] = self.__kalmanconfigserializer.serialize(data.kalman_filters)
		if data.transformer_config is not None:
			json["transformer_config"] = self.__transformer_serializer.serialize(data.transformer_config)
		json["loss"] = losses.serialize(data.loss)
		json["optimizer"] = optimizers.serialize(data.optimizer)
		json["dense_activation"] = activations.serialize(data.dense_activation)
		json["conv_activation"] = activations.serialize(data.conv_activation)

		return json

	def deserialize(self, json_: Dict) -> ModelConfig:
		config = ModelConfig(*[None for _ in range(18)])
		config.__dict__ = json_.copy()
		config.ff_conv_pool_layers = [
			self.__conv_serializer.deserialize(layer)
			for layer in config.ff_conv_pool_layers
		]
		config.kalman_filters = self.__kalmanconfigserializer.deserialize(config.kalman_filters)
		if config.transformer_config is not None:
			config.transformer_config = self.__transformer_serializer.deserialize(config.transformer_config)
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
