import torch


class ModelHandler:

	@staticmethod
	def load(path: str) -> torch.nn.Module:
		model = torch.jit.load(path)
		return model

	@staticmethod
	def save(model: torch.nn.Module, path: str):
		model_scripted = torch.jit.script(model)
		model_scripted.save(path)
