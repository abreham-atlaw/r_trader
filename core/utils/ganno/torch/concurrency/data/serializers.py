import json

import typing

from core.utils.ganno.torch.nnconfig import ConvLayer, CNNConfig, LinearConfig
from lib.network.rest_interface import Serializer


class LinearConfigSerializer(Serializer):

	def __init__(self):
		super().__init__(LinearConfig)

	def serialize(self, data: LinearConfig) -> typing.Dict:
		return data.__dict__.copy()

	def serialize_json(self, data: LinearConfig):
		return json.dumps(self.serialize(data))

	def deserialize(self, json_: typing.Dict) -> LinearConfig:
		return LinearConfig(**json_)

	def deserialize_json(self, json_: str) -> LinearConfig:
		return self.deserialize(json.loads(json_))


class ConvLayerSerializer(Serializer):

	def __init__(self):
		super().__init__(ConvLayer)

	def serialize(self, data: ConvLayer) -> typing.Dict:
		return data.__dict__

	def serialize_json(self, data: ConvLayer):
		return json.dumps(self.serialize(data))

	def deserialize(self, json_: typing.Dict) -> ConvLayer:
		return ConvLayer(**json_)

	def deserialize_json(self, json_: str) -> ConvLayer:
		return self.deserialize(json.loads(json_))


class CNNConfigSerializer(Serializer):

	def __init__(self):
		super().__init__(CNNConfig)
		self.__conv_layer_serializer = ConvLayerSerializer()
		self.__linear_block_serializer = LinearConfigSerializer()

	def serialize(self, data: CNNConfig) -> typing.Dict:
		data_dict = data.__dict__.copy()
		data_dict['layers'] = [self.__conv_layer_serializer.serialize(layer) for layer in data.layers]
		data_dict["ff_block"] = self.__linear_block_serializer.serialize(data.ff_block)
		return data_dict

	def serialize_json(self, data: CNNConfig):
		return json.dumps(self.serialize(data))

	def deserialize(self, json_: typing.Dict) -> CNNConfig:
		json_['layers'] = [self.__conv_layer_serializer.deserialize(layer) for layer in json_['layers']]
		json_["ff_block"] = self.__linear_block_serializer.deserialize(json_["ff_block"])
		return CNNConfig(**json_)

	def deserialize_json(self, json_: str) -> CNNConfig:
		return self.deserialize(json.loads(json_))
