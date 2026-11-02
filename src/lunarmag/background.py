"""Solar-wind background at the low-altitude probe, from an upstream monitor probe.

The monitor (e.g. THC far upstream) sees the solar wind before it reaches the target (e.g. THB near the surface). We find the delay between them, the steady
offset, and the residual scatter (noise floor) during a quiet interval when both are in undisturbed solar wind; the background at the target is then
    B_bg(t) = B_monitor(t - shift) + offset
"""

import numpy as np


def on_grid(t, B, t_new):
    """Linearly interpolate vector data B (N, 3) at times t onto times t_new.

    All times are Unix seconds. Times outside the data return NaN rather than copies of the first or last sample.
    """
    t_new = np.asarray(t_new, dtype=float)
    return np.column_stack([np.interp(t_new, t, B[:, i], left=np.nan, right=np.nan) for i in range(3)])


def fit_background(t_target, B_target, t_monitor, B_monitor, window, max_lag=120.0, dt=1.0, min_points=60):
    """Fit the monitor -> target delay, offset and noise floor over a quiet window.

    Parameters
    ----------
    t_target, B_target   : target probe times (Unix s) and field (N, 3), e.g. THB
    t_monitor, B_monitor : monitor probe times and field, e.g. THC
    window     : (start, end) Unix times when BOTH probes are in undisturbed solar wind
    max_lag    : largest delay tried, in either direction (s)
    dt         : grid spacing and lag step (s)
    min_points : minimum overlapping samples for a lag to count

    Returns
    -------
    dict with
        shift      delay (s): B_target(t) ~ B_monitor(t - shift); positive = monitor upstream
        r          mean correlation of the three components at that shift
        at_edge    True if the best shift is at +-max_lag (then not trustworthy)
        offset     mean target - monitor difference (nT), shape (3,)
        noise      standard deviation of that difference per component (nT), shape (3,)
        sigma_mag  noise floor for |dB|: sqrt(sum(noise**2)) (n
        n_points   number of samples used
        lags, scores   the full correlation curve, for plotting
    """
    grid = np.arange(window[0], window[1], dt)
    Bt = on_grid(t_target, B_target, grid)
    lags = np.arange(-max_lag, max_lag + dt, dt)
    scores = np.full(len(lags), np.nan)

    for k, L in enumerate(lags):
        Bm = on_grid(t_monitor, B_monitor, grid - L)
        ok = np.all(np.isfinite(Bt), axis=1) & np.all(np.isfinite(Bm), axis=1)
        if ok.sum() >= min_points:
            scores[k] = np.mean([np.corrcoef(Bt[ok, i], Bm[ok, i])[0, 1] for i in range(3)])

    if np.all(np.isnan(scores)):
        raise ValueError('no lag gave enough overlapping data; check the window and data coverage')

    k = np.nanargmax(scores)
    shift = lags[k]
    diff = Bt - on_grid(t_monitor, B_monitor, grid - shift)
    ok = np.all(np.isfinite(diff), axis=1)
    offset, noise = diff[ok].mean(axis=0), diff[ok].std(axis=0)

    return {'shift': shift, 'r': scores[k], 'at_edge': abs(shift) >= max_lag,
            'offset': offset, 'noise': noise, 'sigma_mag': np.sqrt(np.sum(noise**2)),
            'n_points': int(ok.sum()), 'window': tuple(window),
            'lags': lags, 'scores': scores}


def background_at(t, t_monitor, B_monitor, fit):
    """Solar-wind field expected at the target at times t: shifted monitor + offset."""
    return on_grid(t_monitor, B_monitor, np.asarray(t, dtype=float) - fit['shift']) + fit['offset']


#choose quiet window via maximum correlation:
def choose_quietswindow(t_target, B_target, t_monitor, B_monitor, span, length=180.0, step=30.0, exclude=(), r_min=0.8, **fit_kw):
    """Find window of maximum correlation between two probes and fit the background there.

    Parameters
    ----------
    span    : (start, end) Unix times to search, e.g. the half-hour before periapsis
    exclude : list of (start, end) intervals windows must not overlap (e.g. shadow)
    r_min   : minimum r for the result to count as a usable background
    fit_kw  : passed on to fit_background (max_lag, dt, min_points)

    Returns
    -------
    The best fit dict (as from fit_background) with an extra key
        ok   True if r >= r_min and the shift is not at the edge of the search
    or None if no window had enough data.
    """
    best = None
    for start in np.arange(span[0], span[1] - length + 1e-9, step):
        w = (start, start + length)
        if any(w[0] < b and w[1] > a for a, b in exclude): # overlaps an excluded interval
            continue
        try:
            fit = fit_background(t_target, B_target, t_monitor, B_monitor, w, **fit_kw)
        except ValueError: # not enough data in this window
            continue
        if best is None or fit['r'] > best['r']:
            best = fit
    if best is not None:
        best['ok'] = bool(best['r'] >= r_min and not best['at_edge'])
    return best