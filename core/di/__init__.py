from .application_container import ApplicationContainer


def init_di():
	container = ApplicationContainer()
	print("Wiring")
	container.wire(
		modules=["core.utils.training.training.continuoustrainer.callbacks"]
	)
