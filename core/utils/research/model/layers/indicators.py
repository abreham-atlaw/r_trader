import typing

import torch.nn as nn

from core.utils.research.model.layers import Delta, KalmanStaticFilter, MultipleMovingAverages, MovingAverage, \
    MovingStandardDeviation, RelativeStrengthIndex, StochasticOscillator, OverlaysCombiner
from core.utils.research.model.model.savable import SpinozaModule


class Indicators(SpinozaModule):
    def __init__(
            self,
            delta: typing.Union[typing.List[int], typing.Union[bool, int]] = 0,
            ksf: typing.Optional[typing.List[int]] = None,
            mma: typing.Optional[typing.List[int]] = None,
            msa: typing.Optional[typing.List[int]] = None,
            msd: typing.Optional[typing.List[int]] = None,
            rsi: typing.Optional[typing.List[int]] = None,
            so: typing.Optional[typing.List[int]] = None,
            identities: int = 0
    ):
        super().__init__()
        self.__args = {
            "delta": delta,
            "ksf": ksf,
            "mma": mma,
            "msa": msa,
            "msd": msd,
            "rsi": rsi,
            "so": so,
            "identities": identities
        }
        self.delta = self.__prepare_arg_delta(delta)
        self.ksf = [KalmanStaticFilter(alpha, beta) for alpha, beta in ksf] if ksf else None
        self.mma = MultipleMovingAverages(mma) if mma else None
        self.msa = [MovingAverage(size) for size in msa] if msa else None
        self.msd = [MovingStandardDeviation(size) for size in msd] if msd else None
        self.rsi = [RelativeStrengthIndex(size) for size in rsi] if rsi else None
        self.so = [StochasticOscillator(size) for size in so] if so else None
        self.identities = [nn.Identity() for _ in range(identities)]
        self.combiner = OverlaysCombiner()

    @staticmethod
    def __prepare_arg_delta(delta: typing.Union[typing.List[int], typing.Union[bool, int]]):
        if isinstance(delta, bool):
            delta = 1 if delta else 0
        if isinstance(delta, int):
            delta = [delta]
        return [Delta(n=n) for n in delta]

    @property
    def indicators_len(self):
        count = 1
        if self.delta:
            count += len(self.delta)
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
        count += len(self.identities)
        return count

    def call(self, inputs):
        outputs = [inputs]
        if self.delta:
            for delta in self.delta:
                outputs.append(delta(inputs))
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

        outputs.extend([
            identity(inputs)
            for identity in self.identities
        ])

        return self.combiner(outputs)

    def export_config(self) -> typing.Dict[str, typing.Any]:
        return self.__args
