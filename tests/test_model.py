# -*- coding: utf-8 -*-

import numpy as np

from one_datum.model import DR2RVErrorModel


def test_resample() -> None:
    # Generate simulated data
    random = np.random.default_rng(1234)
    num_transits = 5
    num_samples = 20000
    true_period = 15.123
    true_semiamp = 5.0
    true_phase = -1.2
    true_ecc = 0.1
    true_omega = 2.4
    true_sigma = 0.2
    true_times = DR2RVErrorModel.sample_times(random, num_transits, 1)
    true_model = (
        DR2RVErrorModel.compute_fiducial_model(
            true_times,
            semiamp=np.array([true_semiamp]),
            period=np.array([true_period]),
            phase=np.array([true_phase]),
            ecc=np.array([true_ecc]),
            omega=np.array([true_omega]),
        )
        + random.normal(0.0, true_sigma, num_transits)
    )
    sample_variance = np.var(true_model, ddof=1)

    # Set up the inference model
    model = DR2RVErrorModel(
        num_transits,
        num_samples,
        log_semiamp=lambda random, size: random.uniform(
            np.log(1.0), np.log(50.0), size
        ),
        log_period=lambda random, size: random.uniform(
            np.log(1.0), np.log(100.0), size
        ),
        ecc=lambda random, size: random.uniform(0, 1, size),
        seed=5678,
    )

    # Manually compute the quantiles using a weighted search
    logp = model.compute_logprob(sample_variance, np.log(true_sigma))
    weights = np.exp(logp - np.max(logp))
    log_semiamp_samps = model.parameter_samples["log_semiamp"]
    inds = np.argsort(log_semiamp_samps)
    cdf = np.cumsum(weights[inds])
    q0 = log_semiamp_samps[inds][
        np.clip(
            np.searchsorted(cdf, np.array([0.2, 0.5, 0.8]) * cdf[-1]) - 1,
            0,
            len(cdf) - 1,
        )
    ]

    # Compute the quantiles using resampling
    samples = model.sample(random, sample_variance, np.log(true_sigma))
    q = np.percentile(samples["log_semiamp"], [20.0, 50.0, 80.0])

    # Check that the quantiles are reasonably consistent
    np.testing.assert_allclose(q, q0, atol=2e-2)

    # Check that the truth is within the 20-80th percentiles; this will catch
    # some disastrous bugs and it is expected to work with this random seed
    assert (q[0] < np.log(true_semiamp)) and (np.log(true_semiamp) < q[-1])
