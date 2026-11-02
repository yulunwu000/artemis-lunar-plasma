"""Checks of the background fit on synthetic data with a known delay, offset and noise."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from lunarmag import on_grid, fit_background, background_at   # noqa: E402

T0 = 1386115200.0 # 2013-12-04 00:00 UT


def synthetic_pair(delay, offset, noise=0.2, seed=0, smooth=40):
    """A smooth random 'solar wind' seen by a monitor, and by a target `delay` s later."""
    rng = np.random.default_rng(seed)
    tt = np.arange(0, 1800, 0.5)
    sw = np.column_stack([np.convolve(rng.normal(size=tt.size), np.ones(smooth) / smooth, 'same') * 8 + c for c in (-4, -5, 1)])
    t_m = T0 + np.arange(250, 1800, 4.0)
    t_t = T0 + np.arange(210, 1800, 4.0)
    B_m = on_grid(T0 + tt, sw, t_m) + rng.normal(0, noise, (t_m.size, 3))
    B_t = on_grid(T0 + tt, sw, t_t - delay) + np.asarray(offset) + rng.normal(0, noise, (t_t.size, 3))
    return t_t, B_t, t_m, B_m


def test_recovers_delay_offset_noise():
    t_t, B_t, t_m, B_m = synthetic_pair(delay=40, offset=[-0.3, -0.1, -0.6])
    fit = fit_background(t_t, B_t, t_m, B_m, window=(T0 + 300, T0 + 600))
    assert abs(fit['shift'] - 40) <= 1
    assert fit['r'] > 0.9 and not fit['at_edge']
    assert np.allclose(fit['offset'], [-0.3, -0.1, -0.6], atol=0.1)
    assert np.allclose(fit['noise'], 0.2 * np.sqrt(2), atol=0.1)   # both probes add 0.2 nT noise


def test_negative_delay():
    """Monitor downstream of the target: the shift comes out negative."""
    t_t, B_t, t_m, B_m = synthetic_pair(delay=-25, offset=[0, 0, 0])
    fit = fit_background(t_t, B_t, t_m, B_m, window=(T0 + 400, T0 + 700))
    assert abs(fit['shift'] + 25) <= 1


def test_background_reproduces_target():
    """With no noise and a smooth field, target - background is ~0 in the fitted window.
    (Residuals that remain come from linear interpolation between 4 s samples.)"""
    t_t, B_t, t_m, B_m = synthetic_pair(delay=14, offset=[0.5, 0, 0], noise=0.0, smooth=800)
    fit = fit_background(t_t, B_t, t_m, B_m, window=(T0 + 300, T0 + 600))
    sel = (t_t > T0 + 300) & (t_t < T0 + 600)
    dB = B_t[sel] - background_at(t_t[sel], t_m, B_m, fit)
    assert np.nanmax(np.abs(dB)) < 0.2


#choose quiet window via maximum correlation:
def test_quiet_window_avoids_disturbance():
    """With a 'lunar' disturbance added to the target, the chosen window avoids it."""
    from lunarmag import choose_quiet_window
    t_t, B_t, t_m, B_m = synthetic_pair(delay=14, offset=[-0.3, -0.1, -0.6])
    B_t = B_t.copy()
    hit = (t_t > T0 + 600) & (t_t < T0 + 900)
    B_t[hit] += np.random.default_rng(5).normal(0, 3, (hit.sum(), 3)) # simulate strong local disturbance
    fit = choose_quiet_window(t_t, B_t, t_m, B_m, span=(T0 + 300, T0 + 1200))
    start, end = fit['window']
    assert end <= T0 + 600 or start >= T0 + 900 # no overlap with the disturbance
    assert abs(fit['shift'] - 14) <= 1 and fit['ok']


def test_quiet_window_respects_exclusions():
    from lunarmag import choose_quiet_window
    t_t, B_t, t_m, B_m = synthetic_pair(delay=14, offset=[0, 0, 0])
    excl = [(T0 + 250, T0 + 800)]
    fit = choose_quiet_window(t_t, B_t, t_m, B_m, span=(T0 + 300, T0 + 1200), exclude=excl)
    assert fit['window'][0] >= T0 + 800