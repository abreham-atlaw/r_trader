
from pymongo import MongoClient
from dataclasses import asdict

from core.Config import MONGODB_URL
from core.utils.research.training.data.state import TrainingState


class TrainingStateRepository:

	def __init__(self, db_name: str = "rtrader", collection_name: str = "training_states", uri: str = MONGODB_URL):
		self.client = MongoClient(uri)
		self.db = self.client[db_name]
		self.collection = self.db[collection_name]

	def save(self, state: TrainingState):
		state_dict = asdict(state)
		filter = {'id': state.id}
		update = {'$set': state_dict}
		self.collection.update_one(filter, update, upsert=True)

	def get(self, id: str):
		document = self.collection.find_one({'id': id})
		if document is not None:
			document.pop("_id")
			return TrainingState(**document)
		return None
