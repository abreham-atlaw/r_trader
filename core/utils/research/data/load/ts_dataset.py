import torch
from torch.utils.data import Dataset
import os


class TSDataset(Dataset):
	def __init__(self, root_path: str):
		self.root_path = root_path
		self.batch_files = [f for f in os.listdir(root_path) if f.endswith('.pt')]
		self.batch_files.sort()

	def __len__(self):
		return len(self.batch_files)

	def __getitem__(self, idx):
		batch_path = os.path.join(self.root_path, self.batch_files[idx])
		batch = torch.jit.load(batch_path)

		# Access X and y as attributes directly
		X, y = batch.X, batch.y

		return X, y
