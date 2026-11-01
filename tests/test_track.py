"""Checks of the position geometry."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from lunarmag import R_M, xyz_to_lonlatalt, solar_zenith, in_shadow, shadow_intervals   # noqa: E402


def test_lonlatalt_known_points():
    """Points on the axes map to the expected lon, lat, alt."""
    xyz = np.array([[R_M + 10, 0, 0],      # over (0 N, 0 E), 10 km up
                    [0, R_M, 0],           # (0 N, 90 E), on the surface
                    [0, -R_M, 0],          # (0 N, 270 E)
                    [0, 0, R_M + 100]])    # north pole, 100 km up
    lon, lat, alt = xyz_to_lonlatalt(xyz)
    assert np.allclose(lon[:3], [0, 90, 270])
    assert np.allclose(lat, [0, 0, 0, 90])
    assert np.allclose(alt, [10, 0, 0, 100])


def test_solar_zenith():
    """Sunward = 0 deg, sideways = 90 deg, anti-sunward = 180 deg."""
    sza = solar_zenith(np.array([[2000, 0, 0], [0, 2000, 0], [-2000, 0, 0]]))
    assert np.allclose(sza, [0, 90, 180])


def test_shadow_cylinder():
    """Only behind-and-inside-the-cylinder counts as shadow."""
    pts = np.array([[-3000, 500, 400],     # behind, rho < R_M  -> shadow
                    [-3000, 1500, 1000],   # behind, rho > R_M  -> lit
                    [3000, 100, 0]])       # sunward            -> lit
    assert list(in_shadow(pts)) == [True, False, False]


def test_shadow_intervals():
    t = np.arange(10.0)
    s = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1], bool)
    assert shadow_intervals(t, s) == [(1, 2), (5, 7), (9, 9)]