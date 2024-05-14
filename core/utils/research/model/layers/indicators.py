import typing

import torch.nn as nn

from core.utils.research.model.layers import *
from core.utils.research.model.model.savable import SavableModule


class Indicators(SavableModule):
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

    def forward(self, inputs):
        outputs = []
        if self.delta:
            outputs.append(self.delta(inputs))
        if self.ksf:
            for filter in self.ksf:
                outputs.append(filter(inputs))
        if self.mma:
            outputs.append(self.mma(inputs))
        if self.msa:
            for ma in self.msa:
                outputs.append(ma(inputs))
        if self.msd:
            for msd in self.msd:
                outputs.append(msd(inputs))
        if self.rsi:
            for rsi in self.rsi:
                outputs.append(rsi(inputs))
        if self.so:
            for so in self.so:
                outputs.append(so(inputs))
        return self.combiner(outputs)
