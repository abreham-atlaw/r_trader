import typing

from dependency_injector import containers, providers

from pymongo import MongoClient

from core import Config


class ServicesContainer(containers.DeclarativeContainer):

	config = providers.Configuration()

	mongo_client = providers.Singleton(
		MongoClient,
		Config.MONGODB_URL
	)
