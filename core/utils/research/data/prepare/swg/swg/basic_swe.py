import os.path

from core.utils.research.data.prepare.swg.abstract_swe import SampleWeightExporter
from .basic_swg import BasicSampleWeightGenerator


class BasicSampleWeightExporter(SampleWeightExporter):

	def __init__(
			self,
			data_path: str,
			export_path: str,
			generator: BasicSampleWeightGenerator,
			X_dir: str = "X",
			y_dir: str = "y",
	):
		super().__init__(
			input_paths=[
				os.path.join(data_path, axis)
				for axis in [X_dir, y_dir]
			],
			export_path=export_path,
			generator=generator,
		)
