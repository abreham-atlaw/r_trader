from core.utils.research.training.callbacks import Callback
from core.utils.research.training.data.metric import Metric
from core.utils.research.training.data.repositories.metric_repository import MetricRepository


class MetricCallback(Callback):

	def __init__(self, repository: MetricRepository):
		super().__init__()
		self.__repository = repository

	def on_epoch_end(self, model, epoch, losses, logs=None):
		for i, loss in enumerate(losses):
			self.__repository.write_metric(
				Metric(
					source=i,
					model=0,
					epoch=epoch,
					depth=0,
					value=(loss,)
				)
			)
