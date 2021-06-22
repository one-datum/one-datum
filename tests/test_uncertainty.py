# -*- coding: utf-8 -*-

import numpy as np

from one_datum import uncertainty


def test_estimate_on_grid() -> None:
    model = uncertainty.get_uncertainty_model()
    np.testing.assert_allclose(
        uncertainty.estimate_uncertainty(*np.meshgrid(*model.grid)),
        model.values.T,
    )
