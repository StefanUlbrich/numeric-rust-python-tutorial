import timeit
from dataclasses import dataclass
from typing import NewType

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs

Likelihood = NewType("Likelihod", NDArray[np.float64])


@dataclass
class GaussianMixtureModel:
    """A mixture model"""
    means: NDArray[np.float64]
    covs: NDArray[np.float64]
    weights: NDArray[np.float64]


def initialize(
    data: NDArray[np.float64], n_components: int, alpha: float = 1.0
) -> GaussianMixtureModel:
    """Generate random responsibilities for intialization"""
    return GaussianMixtureModel(np.zeros(0), np.zeros(0), np.zeros(0))


def expect(gmm: GaussianMixtureModel, data: NDArray[np.float64]) -> Likelihood:
    """Expectation step"""
    return np.zeros(0)


def maximize(
    gmm: GaussianMixtureModel, responsibilities: Likelihood, data: NDArray[np.float64]
) -> None:
    """Maximization step"""


def run():
    """Run a simple training"""

    data, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
    print(data.shape)
    model = initialize(data, 3)

    for _ in range(10):
        r = expect(model, data)
        maximize(model, r, data)

    print(model)


def bench():
    "Simple benchmarks"

    benchmark = """
from gmm import make_blobs, initialize, expect, maximize
data, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
print(data.shape)
model = initialize(data, 3)

r = expect(model, data)
    """
    t = timeit.Timer("maximize(model, r, data)", setup=benchmark)
    print(f"{t.timeit()} seconds")