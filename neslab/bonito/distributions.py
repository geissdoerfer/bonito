import numpy as np
from scipy.stats import expon
from scipy.stats import norm
import warnings
from .bisection import bisection
from .bisection import crit_lt, crit_abs, crit_gt


class ProbabilityDistribution(object):
    def __init__(self, model_parameters, eta: float):
        self._mp = model_parameters
        self._eta = eta

    @property
    def nparams(self):
        return self._mp.size

    def cdf(self, x):
        """Cumulative density function."""
        raise NotImplementedError()

    def ppf(self, p, **kwargs):
        raise NotImplementedError()

    def dll(self, p, **kwargs):
        """Derivative of the log likelihood."""
        raise NotImplementedError()

    def sgd_update(self, x):
        """Update model parameters using stochastic gradient descent."""
        self._mp = self._mp - self._eta * -self.dll(x)


class ExponentialDistribution(ProbabilityDistribution):
    def __init__(self, model_parameters: np.ndarray = np.array([10.0]), eta: float = 0.01):
        super().__init__(model_parameters, eta)

    def cdf(self, x):
        return expon.cdf(x, scale=1.0 / self._mp[0])

    def ppf(self, p, **kwargs):
        return expon.ppf(p, scale=1.0 / self._mp[0])

    def dll(self, x):
        """Natural gradient descent. Multiply original gradient by -1/F^-1 (Fisher inf matrix).

        - https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/
        - https://arxiv.org/pdf/1807.04489.pdf
        """

        d = self._mp - np.power(self._mp, 2) * x
        if (self._mp + self._eta * d) < 1e-9:
            # warnings.warn("exponential hit singularity")
            return (-self._mp + np.random.uniform(1e-9, self._mp)) / self._eta
        return d


class NormalDistribution(ProbabilityDistribution):
    def __init__(self, model_parameters: np.ndarray = np.array([1.0, 1e-4]), eta: float = 0.01):
        super().__init__(model_parameters, eta)

    def cdf(self, x):
        return norm.cdf(x, loc=self._mp[0], scale=np.sqrt(self._mp[1]))

    def ppf(self, p, **kwargs):
        return norm.ppf(p, loc=self._mp[0], scale=np.sqrt(self._mp[1]))

    def dll(self, x):
        """Derivative of MLE of normal distribution according to Titterington"""
        d = np.empty_like(self._mp)
        d[0] = x - self._mp[0]
        d[1] = np.power(x - self._mp[0], 2) - self._mp[1]
        return d


class GaussianMixtureModel(ProbabilityDistribution):
    def __init__(self, model_parameters: np.ndarray = np.array([[0.95, 0.15, 1e-4], [0.05, 0.3, 1e-4]]), eta: float = 0.0001):
        super().__init__(model_parameters, eta)

    def cdf(self, x):
        parts = [self._mp[k][0] * norm.cdf(x, loc=self._mp[k][1], scale=np.sqrt(self._mp[k][2])) for k in range(self._mp.shape[0])]
        if hasattr(x, "__len__"):
            return np.sum(parts, axis=0)
        else:
            return np.sum(parts)

    def ppf(self, p, bound_type: str = None):
        def fn_obj(x):
            return self.cdf(x) - p

        a = min(
            norm.ppf(p, loc=self._mp[0][1], scale=np.sqrt(self._mp[0][2])),
            norm.ppf(p, loc=self._mp[1][1], scale=np.sqrt(self._mp[1][2])),
        )

        b = max(
            norm.ppf(p, loc=self._mp[0][1], scale=np.sqrt(self._mp[0][2])),
            norm.ppf(p, loc=self._mp[1][1], scale=np.sqrt(self._mp[1][2])),
        )

        if bound_type is None:
            crit = crit_abs
        else:
            if bound_type == "lower":
                crit = crit_lt
            else:
                crit = crit_gt

        try:
            _, res = bisection(fn_obj, a, b, criterion=crit)
        except ValueError:
            return b
        except RuntimeError:
            warnings.warn("Bisection did not converge")
            return b
        return res

    def responsibilites(self, x):
        """'Responsibilities' of individual components."""
        K = self._mp.shape[0]
        covs = [max(1e-6, self._mp[k, 2]) for k in range(K)]
        scaled_pdfs = [self._mp[k, 0] * norm.pdf(x, loc=self._mp[k, 1], scale=np.sqrt(covs[k])) for k in range(K)]
        div = np.sum(scaled_pdfs)
        if div <= 0:
            # warnings.warn("sample outside of distribution")
            return np.zeros((K,))
        return np.array([scaled_pdfs[k] / div for k in range(K)])

    def dll(self, x):
        """Derivative of log likelihood according to Titterington et. al. (1984)"""
        K = self._mp.shape[0]
        d = np.empty_like(self._mp)
        resps = self.responsibilites(x)
        for k in range(K):
            d[k, 0] = resps[k] - self._mp[k, 0]
            d[k, 1] = 1 / self._mp[k, 0] * resps[k] * (x - self._mp[k, 1])
            d[k, 2] = 1 / self._mp[k, 0] * resps[k] * (np.power(x - self._mp[k, 1], 2) - self._mp[k, 2])
        return d


def inverse_joint_cdf(dists: tuple, p: float = 0.99):
    """Computes the inverse joint cdf of two independent probability distributions using bisection method.

    Args:
        dists: two probability distributions
        p: target probability
    """

    def objective_function(x):
        cdfs = [dists[i].cdf(x) for i in range(2)]
        return cdfs[0] * cdfs[1] - p

    # Lower search interval bracket
    a = max([dists[i].ppf(p, bound_type="lower") for i in range(2)])

    q = np.sqrt(p)
    # Upper search interval bracket
    b = max([dists[i].ppf(q, bound_type="upper") for i in range(2)])

    try:
        _, res = bisection(objective_function, a, b)
    except ValueError:
        return b
    except RuntimeError:
        warnings.warn("Bisection did not converge")
        return b
    return res
