'''A very basic, yet agruably elegant implementation of Gaussian mixture models'''

import timeit
from dataclasses import dataclass
from typing import NewType

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

from gmm import gmm as _gmm


Likelihood = NewType("Likelihod", NDArray[np.float64])


@dataclass
class GaussianMixtureModel:
    """A mixture model"""

    means: NDArray[np.float64]
    covs: NDArray[np.float64]
    weights: NDArray[np.float64]


def initialize(data: NDArray[np.float64], n_components: int, alpha: float = 1.0) -> GaussianMixtureModel:
    """Generate random responsibilities for intialization"""

    responsibilities = np.random.dirichlet(n_components * [alpha], data.shape[0]).T

    gmm = GaussianMixtureModel(np.zeros(0), np.zeros(0), np.zeros(0))
    maximize(gmm, responsibilities, data)
    return gmm


def expect(gmm: GaussianMixtureModel, data: NDArray[np.float64]) -> Likelihood:
    """Expectation step"""
    res = np.array([w * mvn.pdf(data, mean=m, cov=c) for m, c, w in zip(gmm.means, gmm.covs, gmm.weights)])
    res /= res.sum(axis=0)
    return res


def maximize_v0(gmm: GaussianMixtureModel, responsibilities: Likelihood, data: NDArray[np.float64]) -> None:
    """Maximization step. With loops"""

    sum_responsibilities = responsibilities.sum(axis=1)

    gmm.means = (
        np.sum(data[np.newaxis, :, :] * responsibilities[:, :, np.newaxis], axis=1)
        / sum_responsibilities[:, np.newaxis]
    )

    data = data[np.newaxis, :, :] - gmm.means[:, np.newaxis, :]
    gmm.covs = np.array([(d.T * r[np.newaxis, :] @ d) / r.sum() for r, d in zip(responsibilities, data)])

    gmm.weights = sum_responsibilities
    gmm.weights /= gmm.weights.sum()


def maximize_v2(gmm: GaussianMixtureModel, responsibilities: Likelihood, data: NDArray[np.float64]) -> None:

    means, covs, weights = _gmm.maximize(data, responsibilities.T)
    gmm.means = means
    gmm.covs = covs
    gmm.weights = weights

def maximize(gmm: GaussianMixtureModel, responsibilities: Likelihood, data: NDArray[np.float64]) -> None:
    """Maximization step. Uses einstein sum notation to avoid loops"""

    sum_responsibilities = responsibilities.sum(axis=1)
    gmm.means = (
        np.sum(data[np.newaxis, :, :] * responsibilities[:, :, np.newaxis], axis=1)
        / sum_responsibilities[:, np.newaxis]
    )
    data = data[:, np.newaxis, :] - gmm.means[np.newaxis, :, :]

    gmm.covs = (
        np.einsum("nkd, kn,  nke -> kde", data, responsibilities, data)
        / sum_responsibilities[:, np.newaxis, np.newaxis]
    )

    gmm.weights = sum_responsibilities / sum_responsibilities.sum()


def run():
    """Run a simple training"""

    data, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)  # pylint: disable=W0632
    model = initialize(data, 3)

    for _ in range(10):
        r = expect(model, data)
        maximize_v0(model, r, data)

    plt.scatter(data[:50, 0], data[:50, 1])
    plt.scatter(model.means[:, 0], model.means[:, 1])
    plt.show()


def bench():
    "Simple benchmarks"

    benchmark = """
from gmm import make_blobs, initialize, expect, maximize, maximize_v0, maximize_v2
data, _ = make_blobs(n_samples=10000, centers=40, n_features=2, random_state=0)
model = initialize(data, 40)

r = expect(model, data)
    """
    n = 1000
    repeat = 7
    res = timeit.repeat("maximize(model, r, data)", setup=benchmark, number=n, repeat=repeat)
    print(f"With einsum — fastest: {min(res)/n}, slowest: {max(res)/n}, mean: {sum(res)/repeat/n} ")

    res = timeit.repeat("maximize_v0(model, r, data)", setup=benchmark, number=n, repeat=repeat)
    print(f"With loops — fastest: {min(res)/n}, slowest: {max(res)/n}, mean: {sum(res)/repeat/n} ")

    res = timeit.repeat("maximize_v2(model, r, data)", setup=benchmark, number=n, repeat=repeat)
    print(f"With loops — fastest: {min(res)/n}, slowest: {max(res)/n}, mean: {sum(res)/repeat/n} ")


def make_data():
    '''Generate data for the rust implementation'''
    data, _ = make_blobs(n_samples=10000, centers=10, n_features=2, random_state=7)  # pylint: disable=W0632
    responsibilities = np.random.dirichlet(10 * [1.0], data.shape[0])
    gmm = GaussianMixtureModel(np.zeros(0), np.zeros(0), np.zeros(0))
    maximize(gmm, responsibilities.T, data)

    np.save("data.npy", data)
    np.save("responsibilities.npy", responsibilities.T)
    np.save("means.npy", gmm.means)
