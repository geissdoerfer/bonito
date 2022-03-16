import pytest
from scipy.stats import expon
import numpy as np

from neslab.bonito import NormalDistribution
from neslab.bonito import ExponentialDistribution
from neslab.bonito import GaussianMixtureModel

from neslab.bonito import bonito


@pytest.fixture
def charging_times():
    return expon().rvs((10000, 2))


@pytest.mark.parametrize("dist_cls", [NormalDistribution, ExponentialDistribution, GaussianMixtureModel])
def test_learning(dist_cls, charging_times):
    dist_model = dist_cls()
    for c in charging_times[:, 0]:
        dist_model.sgd_update(c)


@pytest.mark.parametrize("target_probability", [0.5, 0.75, 0.9, 0.99])
def test_bonito(target_probability, charging_times):
    n_success = 0
    for ci, success in bonito((charging_times[:, 0], charging_times[:, 1]), (ExponentialDistribution, ExponentialDistribution), target_probability):
        n_success += success
    assert abs((n_success / charging_times.shape[0]) - target_probability) < 0.1
