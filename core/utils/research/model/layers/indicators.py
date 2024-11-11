import typing
import torch
import torch.nn as nn
from core.utils.research.model.layers import *
from core.utils.research.model.model.savable import SpinozaModule


class Indicators(SpinozaModule):
	def __init__(
		self,
		delta: bool = False,
		ksf: typing.Optional[typing.List[int]] = None,
		mma: typing.Optional[typing.List[int]] = None,
		msa: typing.Optional[typing.List[int]] = None,
		msd: typing.Optional[typing.List[int]] = None,
		rsi: typing.Optional[typing.List[int]] = None,
		so: typing.Optional[typing.List[int]] = None
	):
		super().__init__()
		self.__args = {
			"delta": delta,
			"ksf": ksf,
			"mma": mma,
			"msa": msa,
			"msd": msd,
			"rsi": rsi,
			"so": so
		}
		self.delta = Delta() if delta else None
		self.ksf = [KalmanStaticFilter(alpha, beta) for alpha, beta in ksf] if ksf else None
		self.mma = MultipleMovingAverages(mma) if mma else None
		self.msa = [MovingAverage(size) for size in msa] if msa else None
		self.msd = [MovingStandardDeviation(size) for size in msd] if msd else None
		self.rsi = [RelativeStrengthIndex(size) for size in rsi] if rsi else None
		self.so = [StochasticOscillator(size) for size in so] if so else None
		self.combiner = OverlaysCombiner()

	@property
	def indicators_len(self):
		count = 1
		if self.delta:
			count += 1
		if self.ksf:
			count += len(self.ksf)
		if self.mma:
			count += 1
		if self.msa:
			count += len(self.msa)
		if self.msd:
			count += len(self.msd)
		if self.rsi:
			count += len(self.rsi)
		if self.so:
			count += len(self.so)
		return count

	def call(self, inputs: torch.Tensor) -> torch.Tensor:
		# Annotate outputs explicitly to satisfy TorchScript's strict typing
		outputs: typing.List[torch.Tensor] = [inputs]

		if self.delta is not None:
			outputs.append(self.delta(inputs))
		if self.ksf is not None:
			for filter in self.ksf:
				outputs.append(filter(inputs))
		if self.mma is not None:
			outputs.append(self.mma(inputs))
		if self.msa is not None:
			for ma in self.msa:
				outputs.append(ma(inputs))
		if self.msd is not None:
			for msd in self.msd:
				outputs.append(msd(inputs))
		if self.rsi is not None:
			for rsi in self.rsi:
				outputs.append(rsi(inputs))
		if self.so is not None:
			for so in self.so:
				outputs.append(so(inputs))

		return self.combiner(outputs)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.__args
