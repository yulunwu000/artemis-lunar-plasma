"""Checks that the SVM coefficient model reproduces Tsunakawa et al.'s published 30 km map."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from lunarmag import svm_field, svm_map # noqa: E402

MAP30 = REPO / 'data' / 'globalSVM20150511' / 'LunarSVM_030_05_v01.dat'


def test_svm_field_matches_30km_map():
    """svm_field at 2000 random map points agrees with the file to within rounding."""
    d = np.loadtxt(MAP30, skiprows=12)
    s = d[np.random.default_rng(0).choice(len(d), 2000, replace=False)]
    Be, Bn, Br = svm_field(s[:, 0], s[:, 1], 30.0)
    for model, col in ((Be, 2), (Bn, 3), (Br, 4)):
        assert np.abs(model - s[:, col]).max() < 0.05 #file is rounded to 0.01 nT


def test_svm_map_matches_30km_map():
    """The global grid agrees with the file at every 0.5 deg point."""
    d = np.loadtxt(MAP30, skiprows=12)
    lats, lons, Be, Bn, Br, Bt = svm_map(30.0, lmax=719)
    step = lats[0] - lats[1]
    i = np.rint((90 - d[:, 1]) / step).astype(int)
    j = np.rint(d[:, 0] / step).astype(int)
    assert np.corrcoef(Bt[i, j], d[:, 5])[0, 1] > 0.9999
    assert np.sqrt(np.mean((Bt[i, j] - d[:, 5]) ** 2)) < 0.01


def test_sawyer_periapsis_value():
    """Regression check: the field below the 4 Dec 2013 THB periapsis."""
    Be, Bn, Br = svm_field(85.109, -0.259, 14.79)
    assert np.allclose([Be[0], Bn[0], Br[0]], [-1.002, 0.450, 0.144], atol=0.002)