import base64
import pickle
from typing import Dict

from .serializers import Serializer


class PickleSerializer(Serializer):

	def __init__(self):
		super().__init__(output_class=object)

	def serialize(self, data: object) -> Dict:
		return base64.b64encode(pickle.dumps(data)).decode('utf-8')

	def deserialize(self, json_: Dict) -> object:
		return pickle.loads(base64.b64decode(json_))
