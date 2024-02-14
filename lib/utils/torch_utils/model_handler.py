import torch
import json
import zipfile
import os
import importlib


class ModelHandler:

    @staticmethod
    def save(model, path):
        # Export model config
        model_config = model.export_config()
        # Add the class name to the config
        model_config['class_name'] = model.__class__.__name__
        model_config['module_name'] = model.__class__.__module__
        with open('model_config.json', 'w') as f:
            json.dump(model_config, f)

        # Save model state dict
        torch.save(model.state_dict(), 'model_state.pth')

        # Zip the two files together
        with zipfile.ZipFile(path, 'w') as zipf:
            zipf.write('model_config.json')
            zipf.write('model_state.pth')

        # Remove the temporary files
        os.remove('model_config.json')
        os.remove('model_state.pth')

    @staticmethod
    def load(path):
        # Unzip the file
        with zipfile.ZipFile(path, 'r') as zipf:
            zipf.extractall()

        # Load the model config
        with open('model_config.json', 'r') as f:
            model_config = json.load(f)

        # Get the class from the class name
        module = importlib.import_module(model_config['module_name'])
        ModelClass = getattr(module, model_config['class_name'])

        # Remove class_name and module_name from model_config
        model_config.pop('class_name')
        model_config.pop('module_name')

        # Use the import_config method to deserialize the config
        model_config = ModelClass.import_config(model_config)

        # Create the model
        model = ModelClass(**model_config)

        # Load the state dict
        model.load_state_dict(torch.load('model_state.pth'), map_location=torch.device('cpu'))

        # Remove the temporary files
        os.remove('model_config.json')
        os.remove('model_state.pth')

        return model
