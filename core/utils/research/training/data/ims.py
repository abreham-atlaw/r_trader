import typing
import numpy as np
from dataclasses import dataclass


@dataclass
class IMS:
    name: str
    compute: typing.Callable


class IMSs:
    min_value = IMS(
        "min_value",
        lambda values: np.min(values)
    )

    max_value = IMS(
        "max_value",
        lambda values: np.max(values)
    )

    mean_value = IMS(
        "mean_value",
        lambda values: np.mean(values)
    )

    standard_variance = IMS(
        "standard_variance",
        lambda values: np.var(values)
    )

    l2_norm = IMS(
        "l2_norm",
        lambda values: np.linalg.norm(values, ord=2)
    )

    # Individual percentiles
    percentile_25 = IMS(
        "percentile_25",
        lambda values: np.percentile(values, 25)
    )

    percentile_50 = IMS(
        "percentile_50",
        lambda values: np.percentile(values, 50)
    )

    percentile_75 = IMS(
        "percentile_75",
        lambda values: np.percentile(values, 75)
    )

    all = [
        min_value,
        max_value,
        mean_value,
        standard_variance,
        l2_norm,
        percentile_25,
        percentile_50,
        percentile_75
    ]
