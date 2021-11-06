from typing import *

from core.utils.optimization_cacher import connection, read_cursor

from dataclasses import dataclass


class Utils:

	@staticmethod
	def deserialize(response, object_class):
		if response is None:
			return None
		if type(response) == list:
			return [object_class(*value) for value in response]
		return object_class(*response)


@dataclass
class Config:

	TABLE_NAME = "config"
	COLUMNS = ["id", "seq_len", "loss", "optimizer", "hidden_activation", "delta", "average_window", "value"]

	id: Union[int, None]
	seq_len: int
	loss: str
	optimizer: str
	hidden_activation: str
	delta: bool
	average_window: float
	value: Union[float, None] = None
	hidden_layers: List = None

	def get_hidden_layers(self) -> List[int]:
		if self.hidden_layers is None:
			self.hidden_layers = HiddenLayer.get_by_config_id(self.id)
		return HiddenLayer.to_plain(self.hidden_layers)

	def set_hidden_layers(self, layers: List[int]):
		self.hidden_layers = HiddenLayer.from_plain(layers, self.id)

	def save(self, commit=False):
		if self.hidden_layers is None:
			raise Exception("Hidden Layers not set.")
		cursor = connection.cursor()
		cursor.execute(
			f"INSERT INTO {Config.TABLE_NAME}({', '.join(self.COLUMNS[1:])}) values(%s, %s, %s, %s, %s, %s, %s) RETURNING id",
			(self.seq_len, self.loss, self.optimizer, self.hidden_activation, self.delta, self.average_window, self.value)
		)
		self.id = cursor.fetchone()
		for layer in self.hidden_layers:
			layer.config_id = self.id
			layer.save()

		if commit:
			connection.commit()

	def get_value(self) -> Union[float, None]:
		if self.value is None:
			if self.hidden_layers is None:
				raise Exception("Hidden Layers not set.")
			read_cursor.execute(
				f"SELECT {', '.join(Config.COLUMNS)} FROM {Config.TABLE_NAME} WHERE "
				f"average_window = %s AND "
				f"delta = %s AND "
				f"hidden_activation = %s AND "
				f"loss = %s AND "
				f"optimizer = %s AND "
				f"seq_len = %s",
				(self.average_window, self.delta, self.hidden_activation, self.loss, self.optimizer, self.seq_len)
			)
			response = read_cursor.fetchall()
			configs = Utils.deserialize(response, Config)
			for config in configs:
				if config.get_hidden_layers() == self.get_hidden_layers():
					self.value = config.value
		return self.value


@dataclass
class HiddenLayer:

	TABLE_NAME = "hidden_layer"
	COLUMNS = ["id", "units", "depth", "config_id"]

	id: Union[int, None]
	units: int
	depth: int
	config_id: int

	def save(self, commit=False):
		cursor = connection.cursor()
		cursor.execute(
			f"INSERT INTO {HiddenLayer.TABLE_NAME}({', '.join(HiddenLayer.COLUMNS[1:])}) values(%s, %s, %s) RETURNING id",
			(self.units, self.depth, self.config_id)
		)
		self.id = cursor.fetchone()
		if commit:
			connection.commit()

	@staticmethod
	def get_by_config_id(config_id):
		read_cursor.execute(
			f"SELECT {', '.join(HiddenLayer.COLUMNS)} FROM {HiddenLayer.TABLE_NAME} WHERE config_id = %s",
			(config_id,)
		)
		response = read_cursor.fetchall()
		return Utils.deserialize(response, HiddenLayer)

	@staticmethod
	def to_plain(layers) -> List[int]:
		return [
			layer.units
			for layer in sorted(layers, key=lambda l: l.depth)
		]

	@staticmethod
	def from_plain(layers: List[int], config_id):
		return [
			HiddenLayer(None, unit, i, config_id)
			for i, unit in enumerate(layers)
		]
