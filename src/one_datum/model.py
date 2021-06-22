# -*- coding: utf-8 -*-

__all__ = ["RVErrorModel", "DR2RVErrorModel"]

from typing import Any, Callable, Dict, Optional, Tuple, Union

import kepler
import numpy as np
import scipy.stats

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any


Sampler = Callable[[np.random.Generator, int], ArrayLike]


class RVErrorModel:
    def __init__(
        self,
        num_transits: int,
        num_samples: int,
        seed: Optional[int] = None,
    ):
        self.num_transits = num_transits
        self.num_samples = num_samples

        self.random = np.random.default_rng(seed)
        self.time_samples = self.sample_times(
            self.random, num_samples, num_transits
        )
        self.parameter_samples, self.fiducial_model = self.sample_parameters(
            self.random, self.time_samples
        )
        self.rate_parameter = np.sum(
            (
                self.fiducial_model
                - np.mean(self.fiducial_model, axis=0)[None, :]
            )
            ** 2,
            axis=0,
        )

    def sample_parameters(
        self, random: np.random.Generator, times: np.ndarray
    ) -> Tuple[Dict[str, ArrayLike], ArrayLike]:
        raise NotImplementedError("Must be implemented by subclasses")

    @staticmethod
    def sample_times(
        random: np.random.Generator,
        num_samples: int,
        num_transits: int,
    ) -> ArrayLike:
        raise NotImplementedError("Must be implemented by subclasses")

    @staticmethod
    def compute_fiducial_model(
        times: ArrayLike,
        *,
        semiamp: ArrayLike,
        period: ArrayLike,
        phase: ArrayLike,
        ecc: Optional[ArrayLike] = None,
        omega: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        mean_anom = 2 * np.pi * times / period[None, ...] + phase[None, ...]

        if ecc is None:
            assert omega is None
            return semiamp * np.cos(mean_anom)

        assert omega is not None
        cosw = np.cos(omega)
        sinw = np.sin(omega)
        _, cosf, sinf = kepler.kepler(
            mean_anom, ecc[None, ...] + np.zeros_like(mean_anom)
        )
        return semiamp * (
            cosw[None, ...] * (ecc[None, ...] + cosf) - sinw[None, ...] * sinf
        )

    def compute_logprob(
        self,
        sample_variance: float,
        log_sigma: float,
    ) -> np.ndarray:
        sigma = np.exp(log_sigma)
        ivar = 1.0 / sigma ** 2
        rate = self.rate_parameter * ivar
        ncx2 = scipy.stats.ncx2(df=self.num_transits, nc=rate)
        return ncx2.logpdf((self.num_transits - 1) * sample_variance * ivar)

    def sample(
        self,
        random: np.random.Generator,
        sample_variance: float,
        log_sigma: float,
    ) -> ArrayLike:
        logp = self.compute_logprob(sample_variance, log_sigma)
        p = np.exp(logp - np.logaddexp.reduce(logp))
        n = len(p)
        inds = random.choice(n, size=n, p=p)
        return {k: v[inds] for k, v in self.parameter_samples.items()}


class DR2RVErrorModel(RVErrorModel):
    def __init__(
        self,
        num_transits: int,
        num_samples: int,
        log_semiamp: Union[Sampler, float],
        log_period: Union[Sampler, float],
        ecc: Optional[Union[Sampler, float]] = None,
        seed: Optional[int] = None,
    ):
        self.log_period = sampler_or_constant(log_period)
        self.log_semiamp = sampler_or_constant(log_semiamp)
        self.ecc = sampler_or_constant(ecc)
        super().__init__(num_transits, num_samples, seed)

    @staticmethod
    def sample_times(
        random: np.random.Generator,
        num_samples: int,
        num_transits: int,
    ) -> np.ndarray:
        return random.uniform(0, 668.0, (num_transits, num_samples))

    def sample_parameters(
        self, random: np.random.Generator, times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = times.shape[1]

        log_period = self.log_period(random, num_samples)
        log_semiamp = self.log_semiamp(random, num_samples)
        phase = random.uniform(-np.pi, np.pi, num_samples)

        if self.ecc is None:
            params = {
                "log_semiamp": log_semiamp,
                "log_period": log_period,
                "phase": phase,
            }
            mod = self.compute_fiducial_model(
                times,
                semiamp=np.exp(log_semiamp),
                period=np.exp(log_period),
                phase=phase,
            )
        else:
            ecc = self.ecc(random, num_samples)
            omega = random.uniform(-np.pi, np.pi, num_samples)
            params = {
                "log_semiamp": log_semiamp,
                "log_period": log_period,
                "phase": phase,
                "ecc": ecc,
                "omega": omega,
            }
            mod = self.compute_fiducial_model(
                times,
                semiamp=np.exp(log_semiamp),
                period=np.exp(log_period),
                phase=phase,
                ecc=ecc,
                omega=omega,
            )

        return params, mod


def sampler_or_constant(
    func: Optional[Union[Sampler, float]]
) -> Optional[Sampler]:
    if func is None:
        return None
    if callable(func):
        return func
    value = func
    return lambda _, size: np.broadcast_to(value, size)
