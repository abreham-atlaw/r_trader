import typing
from abc import ABC, abstractmethod

import random
from datetime import datetime

from core.utils.resman.data.models import Resource


class ResourceRepository(ABC):

	def __init__(self, category: str):
		self._category = category

	def _choose(self, resources: typing.List[Resource]) -> Resource:
		return random.choice(resources)

	def _select(self) -> Resource:
		valid_resources = [
			resource
			for resource in self._get_all()
			if not self.is_locked(resource)
		]
		if len(valid_resources) == 0:
			raise ResourceUnavailableException()
		return self._choose(valid_resources)

	def _lock(self, resource: Resource):
		resource.lock_datetime = datetime.now()
		self._update(resource)

	def _unlock(self, resource: Resource):
		resource.lock_datetime = None
		self._update(resource)

	def _get_by_id(self, id: str) -> Resource:
		for resource in self._get_all():
			if resource.id == id:
				return resource
		raise ResourceNotFoundException()

	def is_locked(self, resource: Resource) -> bool:
		return resource.locked

	def is_locked_by_id(self, id: str) -> bool:
		try:
			stat = self._get_by_id(id)
			return self.is_locked(stat)
		except ResourceNotFoundException:
			return False

	def lock_by_id(self, id: str, create: bool = True) -> Resource:
		try:
			resource = self._get_by_id(id)
		except ResourceNotFoundException:
			if create:
				resource = self.create(id)
			else:
				raise
		self._lock(resource)
		return resource

	def unlock_by_id(self, id: str):
		resource = self._get_by_id(id)
		self._unlock(resource)

	def allocate(self) -> Resource:
		resource = self._select()
		self._lock(resource)
		return resource

	def release(self, resource: Resource):
		self._unlock(resource)

	def release_by_id(self, id: str):
		resource = self._get_by_id(id)
		self._unlock(resource)

	def create(self, id: str) -> Resource:
		print(f"Creating {id}")
		resource = Resource(id, lock_datetime=None)
		self._create(resource)
		return resource

	def get_locked(self):
		return [
			resource
			for resource in self._get_all()
			if self.is_locked(resource)
		]

	@abstractmethod
	def _get_all(self) -> typing.List[Resource]:
		pass

	@abstractmethod
	def _update(self, resource: Resource):
		pass

	@abstractmethod
	def _create(self, resource: Resource):
		pass


class ResourceNotFoundException(Exception):
	pass


class ResourceUnavailableException(Exception):
	pass
