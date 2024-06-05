import typing

from dependency_injector import containers, providers

from core.utils.kaggle.data.repositories import MongoAccountsRepository, MongoSessionsRepository, MongoResourcesRepository
from core.utils.kaggle import ResourcesManager, SessionsManager, FusedManager


class KaggleContainer(containers.DeclarativeContainer):

	config = providers.Configuration()
	mongo_client = providers.Dependency()

	accounts_repository = providers.Singleton(
		MongoAccountsRepository,
		mongo_client
	)

	sessions_repository = providers.Singleton(
		MongoSessionsRepository,
		accounts_repository,
		mongo_client
	)

	resources_repository = providers.Singleton(
		MongoResourcesRepository,
		sessions_repository,
		mongo_client
	)

	resources_manager = providers.Singleton(
		ResourcesManager,
		accounts_repository,
		resources_repository
	)

	sessions_manager = providers.Singleton(
		SessionsManager,
		sessions_repository,
		accounts_repository
	)

	fused_manager = providers.Singleton(
		FusedManager,
		resources_manager,
		sessions_repository,
		accounts_repository
	)
