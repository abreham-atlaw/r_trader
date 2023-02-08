import typing

from abc import ABC, abstractmethod

from tensorflow.keras import models


class DNNManager:

	def __init__(
			self,
			default_load=True,
			custom_objects: typing.Optional[typing.Dict[str, typing.Any]] = None,
			path: typing.Optional[str] = None
	):
		self.__default_load = default_load
		self.__custom_objects = custom_objects
		if custom_objects is None:
			self.__custom_objects = {}
		self.__path = path

	@abstractmethod
	def create(self) -> models.Model:
		pass

	def load(self) -> models.Model:
		if self.__path is None:
			raise NotImplementedError("Either path should be passed or load should be implemented")
		return models.load_model(self.__path, custom_objects=self.__custom_objects)

	def save(self, model: models.Model):
		if self.__path is None:
			raise NotImplementedError("Either path should be passed or save should be implemented")
		model.save(self.__path)

	def get_model(self) -> models.Model:
		print("[+]Getting Model...")
		functions = [self.create, self.load]
		if self.__default_load:
			functions.reverse()
		for func in functions:
			try:
				print(f"[+]Trying {func.__name__}...")
				return func()
			except Exception as ex:
				print(f"[-]Function {func.__name__} failed with exception {ex}")
		raise Exception("Loading and Creating Failed")
