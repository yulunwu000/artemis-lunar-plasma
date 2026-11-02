"""SVM crustal magnetic field model of Tsunakawa et al. (2015).

Evaluates the vacuum crustal field from their degree-450 spherical-harmonic coefficients (LunarSVM_SHC_v01.dat) with pyshtools / SHTOOLS
(Wieczorek & Meschede, 2018).

Reference: Tsunakawa, H., et al. (2015), J. Geophys. Res. Planets, 120, 1160-1185, doi:10.1002/2014JE004785.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyshtools as sh

R_M = 1737.4 #lunar reference radius used by the SVM / km
LMAX = 450 # highest degree of SVM coefficients in spherical harmonic expansion

DEFAULT_SHC = (Path(__file__).resolve().parents[2] / 'data' / 'globalSVM20150511' / 'LunarSVM_SHC_v01.dat')

@lru_cache(maxsize=None) #cache data

def load_svm(path=DEFAULT_SHC):
    """
    Read the SVM coefficients and return a pyshtools SHMagCoeffs object.
    File columns: n, m, g, h (nT).
    Conventions: Schmidt semi-normalised, no Condon-Shortley phase, reference radius R_M.
    """
    c = np.loadtxt(path, skiprows=10)
    n, m = c[:, 0].astype(int), c[:, 1].astype(int)
    coeffs = np.zeros((2, LMAX + 1, LMAX + 1))
    coeffs[0, n, m] = c[:, 2] # g_nm (cosine terms)
    coeffs[1, n, m] = c[:, 3] # h_nm (sine terms)
    return sh.SHMagCoeffs.from_array(coeffs, r0=R_M, normalization='schmidt', csphase=1)


def svm_field(lon, lat, alt):
    """SVM crustal field at given positions.

    Parameters
    ----------
    lon, lat: longitude (deg E) and latitude (deg N)
    alt: altitude above R_M (km)

    Returns
    -------
    Be, Bn, Br: local east, north, up components (nT)
    """
    lon, lat, alt = np.broadcast_arrays(np.atleast_1d(lon).astype(float), np.atleast_1d(lat).astype(float), np.atleast_1d(alt).astype(float))
    out = np.atleast_2d(load_svm().expand(lon=lon, lat=lat, a=R_M + alt))
    Br, Btheta, Bphi = out.T
    return Bphi, -Btheta, Br


def svm_map(alt=0.0, lmax=899):
    """Global grid of the SVM field at altitude alt (km).
    Grid spacing is 180 / (2 (lmax + 1)) degrees. lmax must be >= 450, or pyshtools drops the highest-degree coefficients.

    Returns
    -------
    lats: 1-D array (deg N), from 90 down to -90
    lons: 1-D array (deg E), from 0 up to 360
    Be, Bn, Br, Bt: 2-D arrays (nT), shape (len(lats), len(lons))
    """
    if lmax < LMAX:
        raise ValueError(f'lmax must be >= {LMAX}, got {lmax}')

    g = load_svm().expand(a=R_M + alt, lmax=lmax)
    return (g.rad.lats(), g.rad.lons(), g.phi.data, -g.theta.data, g.rad.data, g.total.data)