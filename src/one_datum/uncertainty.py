# -*- coding: utf-8 -*-

__all__ = ["get_filename", "get_uncertainty_model", "estimate_uncertainty"]

from typing import Any, Optional

import numpy as np
import pkg_resources
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any


def get_filename() -> str:
    return pkg_resources.resource_filename(__name__, "data/noise-model.fits")


def get_uncertainty_model(
    *,
    bounds_error: bool = False,
    fill_value: Optional[float] = None,
    filename: Optional[str] = None,
) -> RegularGridInterpolator:
    """
    Get a callable interpolator to estimate the per-transit radial velocity
    uncertainty of a Gaia EDR3 source as a function of apparent G-magnitude
    and observed BP-RP color.

    Args:
        filename (str, optional): The path to the FITS file with the model. By
            default, this will be the model bundled with the code.
        bounds_error (bool, optional): If ``True``, when interpolated values
            are requested outside of the domain of the input data, a
            ``ValueError`` is raised. If ``False``, then ``fill_value`` is
            used. Defaults to ``False``.
        fill_value (Optional[float], optional): If provided, the value to use
            for points outside of the interpolation domain. If ``None``,
            values outside the domain are extrapolated. Defaults to ``None``.

    Returns:
        RegularGridInterpolator: A callable object which takes an apparent
            G-magnitude and observed BP-RP color, and returns the natural log
            of the estimated per-transit radial velocity uncertainty. This can
            also accept arrays as input.
    """
    if not filename:
        filename = get_filename()

    with fits.open(filename) as f:
        hdr = f[0].header
        mu = f[1].data

    color_bins = np.linspace(
        hdr["MIN_COL"], hdr["MAX_COL"], hdr["NUM_COL"] + 1
    )
    mag_bins = np.linspace(hdr["MIN_MAG"], hdr["MAX_MAG"], hdr["NUM_MAG"] + 1)
    return RegularGridInterpolator(
        [
            0.5 * (mag_bins[1:] + mag_bins[:-1]),
            0.5 * (color_bins[1:] + color_bins[:-1]),
        ],
        mu,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )


def estimate_uncertainty(
    gmag: ArrayLike, bp_rp: ArrayLike, *, filename: Optional[str] = None
) -> ArrayLike:
    """
    Estimate the per-transit radial velocity uncertainty of a Gaia EDR3 source
    given its apparent G-magnitude and observed BP-RP color.

    Args:
        gmag (array): The apparent Gaia G-band magnitude of the source.
        bp_rp (array): The observed BP-RP color of the source.
        filename (str, optional): The path to the FITS file with the model. By
            default, this will be the model bundled with the code.

    Returns:
        array: An estimate of Gaia's per-transit radial velocity uncertainty
            for this source (or array of sources).
    """
    model = get_uncertainty_model(filename=filename)
    return model(np.stack(np.broadcast_arrays(gmag, bp_rp), axis=-1))
