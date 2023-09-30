from tensorflow import keras

from .trainer import Trainer


class SingleModelTrainer(Trainer):

	def fit(
			self,
			core_model: keras.Model,
			*args,
			**kwargs
	) -> 'Trainer.MetricsContainer':
		return super().fit(
			core_model,
			None,
			*args,
			delta_training=False,
			**kwargs
		)
