import os.path

from core.utils.research.data.prepare.swg.abstract_swe import SampleWeightExporter

from .basic_xswg import BasicXSampleWeightGenerator


class BasicXSampleWeightExporter(SampleWeightExporter):

	def __init__(
			self,
			data_path: str,
			export_path: str,
			generator: BasicXSampleWeightGenerator,
			X_dir: str = "X",
	):
		super().__init__(
			input_paths=[
				os.path.join(data_path, axis)
				for axis in [X_dir]
			],
			export_path=export_path,
			generator=generator,
		)
