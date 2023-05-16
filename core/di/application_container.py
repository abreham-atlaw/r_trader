import typing

from dependency_injector import containers, providers

from core.utils.kaggle.di.containers import KaggleContainer
from .services_container import ServicesContainer


class ApplicationContainer(containers.DeclarativeContainer):

	config = providers.Configuration()
	# wiring_config = containers.WiringConfiguration(
	# 	modules=["core.utils.training.training.continuoustrainer.callbacks"],
	# 	auto_wire=True
	# )

	services = providers.Container(
		ServicesContainer,
		config=config.services
	)

	kaggle = providers.Container(
		KaggleContainer,
		mongo_client=services.mongo_client,
		config=config.kaggle
	)
