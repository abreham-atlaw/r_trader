import hashlib
import shutil
from uuid import uuid4

import torch
import json
import zipfile
import os
import importlib

from core.utils.research.model.model.savable import SpinozaModule


class ModelHandler:

	__MODEL_PREFIX = "__model__"

	@staticmethod
	def save(model, path, to_cpu=True):
		if to_cpu:
			original_device = next(model.parameters()).device
			model = model.to(torch.device('cpu'))
		# Export model config
		model_config = model.export_config()
		# Add the class name to the config
		model_config['class_name'] = model.__class__.__name__
		model_config['module_name'] = model.__class__.__module__

		model_config_copy = {}
		for key, value in model_config.items():
			if isinstance(value, SpinozaModule):
				filename = f"{key}.zip"
				ModelHandler.save(value, filename)
				model_config_copy[f"{ModelHandler.__MODEL_PREFIX}{key}"] = filename
			else:
				model_config_copy[key] = value

		model_config = model_config_copy

		with open('model_config.json', 'w') as f:
			json.dump(model_config, f)

		# Save model state dict
		torch.save(model.state_dict(), 'model_state.pth')

		# Zip the two files together
		with zipfile.ZipFile(path, 'w') as zipf:
			zipf.write('model_config.json')
			zipf.write('model_state.pth')
			for key, value in model_config.items():
				if key.startswith(ModelHandler.__MODEL_PREFIX):
					zipf.write(value)

		# Remove the temporary files
		os.remove('model_config.json')
		os.remove('model_state.pth')
		for key, value in model_config.items():
			if key.startswith(ModelHandler.__MODEL_PREFIX):
				os.remove(value)

		if to_cpu:
			model.to(original_device)

	@staticmethod
	def load(path, dtype=torch.float32):
		dirname = f"{os.path.basename(path).replace('.', '_')} - {uuid4()}"

		try:
			os.makedirs(dirname)
		except FileExistsError:
			pass
		# Unzip the file
		with zipfile.ZipFile(path, 'r') as zipf:
			zipf.extractall(dirname, )

		# Load the model config
		with open(os.path.join(dirname, 'model_config.json'), 'r') as f:
			model_config = json.load(f)

		# Get the class from the class name
		module = importlib.import_module(model_config['module_name'])
		ModelClass = getattr(module, model_config['class_name'])

		# Remove class_name and module_name from model_config
		model_config.pop('class_name')
		model_config.pop('module_name')

		model_config_copy = {}
		for key, value in model_config.items():
			if key.startswith(ModelHandler.__MODEL_PREFIX):
				model_config_copy[key[len(ModelHandler.__MODEL_PREFIX):]] = ModelHandler.load(os.path.join(dirname, value))
				os.remove(os.path.join(dirname, value))
			else:
				model_config_copy[key] = value
		model_config = model_config_copy
		# Use the import_config method to deserialize the config
		model_config = ModelClass.import_config(model_config)

		# Create the model
		model: SpinozaModule = ModelClass(**model_config)

		# Load the state dict
		model.load_state_dict_lazy(torch.load(os.path.join(dirname, 'model_state.pth'), map_location=torch.device('cpu')))

		shutil.rmtree(dirname, ignore_errors=True)

		return model.type(dtype)

	@staticmethod
	def generate_signature(model: SpinozaModule) -> str:
		config = model.export_config()
		return hashlib.sha256(
			f"{model.__class__.__name__}-{model.__class__.__module__}-{str(config)}".encode("utf-8")
		).hexdigest()
