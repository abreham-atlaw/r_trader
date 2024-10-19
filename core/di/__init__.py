from .application_container import ApplicationContainer

from .legacy import *


def init_di():
	container = ApplicationContainer()
	print("Wiring")
	container.wire(
		modules=["core.utils.training.training.continuoustrainer.callbacks"]
	)
