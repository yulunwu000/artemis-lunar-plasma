"""Spacecraft position along a pass: lon/lat/alt, solar zenith angle and lunar shadow."""
import numpy as np
from scipy.interpolate import CubicSpline

from .svm import R_M


def xyz_to_lonlatalt(xyz):
    """Moon-centred Cartesian position (km, SEL) -> lon (deg E, 0-360), lat (deg), alt (km above R_M)."""
    xyz = np.atleast_2d(xyz)
    r = np.linalg.norm(xyz, axis=1)
    lat = np.degrees(np.arcsin(xyz[:, 2] / r))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360
    return lon, lat, r - R_M


def solar_zenith(xyz_sse):
    """Solar zenith angle (deg): angle between the position vector and the Sun direction (+X in SSE)."""
    xyz_sse = np.atleast_2d(xyz_sse)
    return np.degrees(np.arccos(xyz_sse[:, 0] / np.linalg.norm(xyz_sse, axis=1)))


def in_shadow(xyz_sse):
    """True where the spacecraft is in the Moon's shadow: behind the Moon (x < 0) and within one lunar radius of the Sun-Moon line (cylinder approximation)."""
    xyz_sse = np.atleast_2d(xyz_sse)
    return (xyz_sse[:, 0] < 0) & (np.hypot(xyz_sse[:, 1], xyz_sse[:, 2]) < R_M)


def shadow_intervals(t, shadow, pad=0.0):
    """List of (start, end) times of each continuous stretch where shadow is True."""
    s = np.concatenate([[False], np.asarray(shadow, bool), [False]])
    starts = np.flatnonzero(~s[:-1] & s[1:]) # False -> True
    ends = np.flatnonzero(s[:-1] & ~s[1:]) - 1 # last True before True -> False
    return [(t[a] - pad, t[b] + pad) for a, b in zip(starts, ends)]


def in_intervals(t, intervals):
    """True where t falls inside any of the (start, end) intervals."""
    t = np.asarray(t, dtype=float)
    mask = np.zeros(len(t), dtype=bool)
    for a, b in intervals:
        mask |= (t >= a) & (t <= b)
    return mask

def load_track(probe, trange, t_out=None):
    """Load an ARTEMIS probe's position and derive everything the pipeline needs.

    Parameters
    ----------
    probe  : 'b' or 'c'
    trange : [start, end] time strings
    t_out  : optional array of Unix times; positions are cubic-splined onto them (e.g. the FGM timestamps). Must lie inside the loaded time range.

    Returns
    -------
    dict with keys
        t        Unix times (s)
        lon, lat, alt      deg E, deg, km above R_M
        xyz_sel, xyz_sse   positions (km), shape (N, 3)
        sza      solar zenith angle (deg)
        shadow   True where in the Moon's shadow
    """
    import pyspedas
    from pyspedas import get_data

    names = [f'th{probe}_pos_sel', f'th{probe}_pos_sse']
    pyspedas.projects.themis.state(probe=probe, trange=trange, varnames=names, time_clip=True)
    sel, sse = get_data(names[0]), get_data(names[1])
    t, xyz_sel, xyz_sse = sel.times, sel.y, sse.y

    if t_out is not None:
        t_out = np.asarray(t_out, dtype=float)
        if t_out.min() < t[0] or t_out.max() > t[-1]:
            raise ValueError('t_out extends beyond the loaded positions; widen trange (splines are only reliable between known samples)')
        xyz_sel = CubicSpline(t, xyz_sel, axis=0)(t_out)
        xyz_sse = CubicSpline(sse.times, xyz_sse, axis=0)(t_out)
        t = t_out

    lon, lat, alt = xyz_to_lonlatalt(xyz_sel)
    return {'t': t, 'lon': lon, 'lat': lat, 'alt': alt,
            'xyz_sel': xyz_sel, 'xyz_sse': xyz_sse,
            'sza': solar_zenith(xyz_sse), 'shadow': in_shadow(xyz_sse)}