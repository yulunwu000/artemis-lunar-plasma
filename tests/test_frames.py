"""Checks of the SEL <-> local east/north/up transforms."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from lunarmag import sel2enr, enr2sel # noqa: E402

BCAL = REPO / 'data' / 'globalSVM20150511' / 'B_Positions_supplied.rst'


def test_axes_at_sub_earth_point():
    """At (0 N, 0 E): east = +Y, north = +Z, up = +X."""
    out = np.column_stack(sel2enr(np.eye(3), np.zeros(3), np.zeros(3)))
    assert np.allclose(out, [[0, 0, 1],      # X -> up
                             [1, 0, 0],      # Y -> east
                             [0, 1, 0]])     # Z -> north


def test_matches_bcal():
    """Converting Bcal's Bx, By, Bz reproduces Bcal's own Be, Bn, Br (Tsunakawa's code)."""
    b = np.loadtxt(BCAL, skiprows=1)
    Be, Bn, Br = sel2enr(b[:, 3:6], b[:, 0], b[:, 1])
    for ours, col in ((Be, 6), (Bn, 7), (Br, 8)):
        assert np.abs(ours - b[:, col]).max() < 0.005


def test_round_trip_and_length():
    """enr2sel undoes sel2enr, and the rotation preserves vector length."""
    rng = np.random.default_rng(1)
    B = rng.normal(size=(500, 3))
    lon, lat = rng.uniform(0, 360, 500), rng.uniform(-90, 90, 500)
    Be, Bn, Br = sel2enr(B, lon, lat)
    assert np.allclose(enr2sel(Be, Bn, Br, lon, lat), B)
    assert np.allclose(np.sqrt(Be**2 + Bn**2 + Br**2), np.linalg.norm(B, axis=1))