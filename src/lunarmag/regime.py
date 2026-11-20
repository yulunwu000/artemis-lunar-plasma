"""Plasma regime of a pass (by geometry only)."""
import numpy as np

V_EARTH = 30.0 #Earth orbital speed (km/s)


def tail_angle(pos_gse, v_sw=400.0):
    """Angle (deg) between Earth-centred GSE positions (N, 3) and the aberrated tail axis.

    0 deg = straight down the tail (full Moon); 180 deg = sunward of Earth (new Moon).
    """
    pos = np.atleast_2d(np.asarray(pos_gse, dtype=float))
    a = np.arctan(V_EARTH / v_sw) # aberration angle (wind sock)
    axis = np.array([-np.cos(a), np.sin(a), 0.0]) #downwind direction in GSE
    r = np.linalg.norm(pos, axis=1)
    return np.degrees(np.arccos(np.clip(pos @ axis / r, -1, 1)))


def regime_geo(angle_deg, cut=75.0):
    """Regime label: 'solar_wind' if well away from the magnetosphere, else 'uncertain'."""
    return 'solar_wind' if angle_deg > cut else 'uncertain'