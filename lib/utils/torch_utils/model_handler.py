import torch


class ModelHandler:

	@staticmethod
	def load(path: str, device=None) -> torch.nn.Module:
		model = torch.jit.load(path, map_location=device)
		return model

	@staticmethod
	def save(model: torch.nn.Module, path: str):
		model_scripted = torch.jit.script(model)
		model_scripted.save(path)
