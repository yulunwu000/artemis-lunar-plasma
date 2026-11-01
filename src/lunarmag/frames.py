"""Coordinate transforms for comparing spacecraft magnetic field data with the SVM.

Frames
------
GSE : Earth-centred; X towards the Sun, Z ecliptic north.
SSE : Moon-centred; X Moon-to-Sun, Z close to ecliptic north.
SEL : Moon-centred, Moon-fixed (selenographic); X to (0 N, 0 E), Z along the spin axis.
ENR : local east, north, up (radial) at a point on or above the Moon, the components used by the SVM (Tsunakawa et al., 2015).
"""
import numpy as np


def _enr_axes(lon, lat):
    """Unit vectors east, north, up at (lon, lat), written in SEL components.

    Returns three arrays of shape (N, 3).
    """
    phi, lam = np.radians(np.atleast_1d(lon)), np.radians(np.atleast_1d(lat))
    e = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    n = np.stack([-np.sin(lam) * np.cos(phi), -np.sin(lam) * np.sin(phi), np.cos(lam)], axis=-1)
    r = np.stack([ np.cos(lam) * np.cos(phi),  np.cos(lam) * np.sin(phi), np.sin(lam)], axis=-1)
    return e, n, r


def sel2enr(B, lon, lat):
    """Rotate vectors from SEL Cartesian into local east, north, up.

    Parameters
    ----------
    B: array (N, 3)   vectors in SEL (e.g. nT)
    lon, lat: arrays (N,)    position of each vector (deg E, deg)

    Returns
    -------
    Be, Bn, Br : arrays (N,)  projections of B onto the local east, north, up directions
    """
    B = np.atleast_2d(B)
    e, n, r = _enr_axes(lon, lat)
    return np.sum(B * e, axis=1), np.sum(B * n, axis=1), np.sum(B * r, axis=1)


def enr2sel(Be, Bn, Br, lon, lat):
    """Inverse of sel2enr: local east, north, up components back to an (N, 3) SEL array."""
    e, n, r = _enr_axes(lon, lat)
    return (np.atleast_1d(Be)[:, None] * e + np.atleast_1d(Bn)[:, None] * n + np.atleast_1d(Br)[:, None] * r)


def gse2sel(times, B_gse, name='lunarmag_tmp'):
    """Rotate field vectors from GSE to SEL with pyspedas (rotation only, no translation).

    Uses the ARTEMIS lunar ephemeris (SLP) support data, which pyspedas downloads
    automatically, so this needs network access the first time for a given day.

    Parameters
    ----------
    times : array (N,)     Unix times (s)
    B_gse : array (N, 3)   vectors in GSE

    Returns
    -------
    B_sel : array (N, 3)
    """
    from pyspedas import store_data, set_coords, get_data, del_data
    from pyspedas.projects.themis import gse2sse, sse2sel

    store_data(name + '_gse', data={'x': np.asarray(times), 'y': np.asarray(B_gse)})
    set_coords(name + '_gse', 'GSE')
    if not gse2sse(name + '_gse', name + '_sse'):
        raise RuntimeError('gse2sse failed; see the pyspedas log above')
    if not sse2sel(name + '_sse', name + '_sel'):
        raise RuntimeError('sse2sel failed; see the pyspedas log above')
    B_sel = get_data(name + '_sel').y
    del_data(name + '_*')
    return B_sel