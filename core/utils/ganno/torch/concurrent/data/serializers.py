import json

import typing

from core.utils.ganno.torch.nnconfig import ConvLayer, CNNConfig
from lib.network.rest_interface import Serializer


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

	def serialize(self, data: CNNConfig) -> typing.Dict:
		data_dict = data.__dict__.copy()  # create a copy of data's dict
		data_dict['layers'] = [self.__conv_layer_serializer.serialize(layer) for layer in data.layers]
		return data_dict

	def serialize_json(self, data: CNNConfig):
		return json.dumps(self.serialize(data))

	def deserialize(self, json_: typing.Dict) -> CNNConfig:
		json_['layers'] = [self.__conv_layer_serializer.deserialize(layer) for layer in json_['layers']]
		return CNNConfig(**json_)

	def deserialize_json(self, json_: str) -> CNNConfig:
		return self.deserialize(json.loads(json_))
