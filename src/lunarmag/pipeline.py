"""For analysing measured lunar effect versus the SVM crustal field for a single pass.

For one low-altitude pass of an ARTEMIS probe:
    dB  = B_measured - B_background (the Moon's total effect: crust + interaction field)
    dBi = dB - B_SVM (the interaction field)
both in local east, north, up.
"""
import numpy as np

from .svm import R_M, svm_field
from .frames import sel2enr, gse2sel
from .track import load_track, shadow_intervals, in_intervals
from .background import background_at, choose_quiet_window, MAX_LAG
from .regime import tail_angle, regime_geo

POST_SEARCH = 1200.0 # default reach (s) of post-pass background check

def _angle(a, b):
    """Angle (deg) between two vectors."""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def analyse_arrays(t_peri, t_b, B_b, t_c, B_c, trk, trk_raw, trk_mon, half_window=1200.0, search=(1800.0, 120.0), bg_alt_min=0.0, alt_max=300.0, rotate=gse2sel, shadow_pad=120.0, post_search = POST_SEARCH, **window_kw):
    """The calculation, given data already loaded (so it can be tested with fake data).

    Parameters
    ----------
    t_peri        : Unix time of periapsis
    t_b, B_b      : target (e.g. THB) FGM times and GSE field, (N, 3)
    t_c, B_c      : monitor (e.g. THC) FGM times and GSE field
    trk           : load_track result for the target on its FGM times (splined) within the analysis window
    trk_raw       : load_track result for the target at 1-min resolution over the whole load range
    trk_mon       : load_track result for the monitor at 1-min resolution
    half_window   : analysis window is t_peri +- half_window (s)
    search        : quiet-window search spans: (t_peri - search[0], t_peri - search[1]) before and (t_peri + search[1], t_peri + half_window) after
    alt_max       : only samples below this altitude (km) enter the summary numbers
    rotate        : function (times, B_gse) -> B_sel; gse2sel by default
    post_search   : the post-pass check searches (t_peri + search[1], t_peri + post_search); pass a larger value (e.g. 3600) to look beyond a long shadow
    window_kw     : passed to choose_quiet_window (length, step, sigma_max, r_min, ...)
    bg_alt_min    : background windows must have the target above this altitude (km); default 0.
    """

    #background: quiet window before and after pass
    shadows = shadow_intervals(trk_raw['t'], trk_raw['shadow']) #list of (start, end) times for all shadow periods
    near_shadow = shadow_intervals(trk_raw['t'], trk_raw['shadow'], pad=shadow_pad, outer=True) #add buffer

    near_moon = shadow_intervals(trk_raw['t'], trk_raw['alt'] < bg_alt_min, outer=True)
    bg_exclude = near_shadow + near_moon

    #find best quiet window before periapsis
    fit = choose_quiet_window(t_b, B_b, t_c, B_c, span=(t_peri - search[0], t_peri - search[1]), exclude=bg_exclude, **window_kw)
    if fit is None:
        raise RuntimeError('no quiet window with enough data before periapsis')

    #find best quiet window after periapsis
    fit_post = choose_quiet_window(t_b, B_b, t_c, B_c, span=(t_peri + search[1], t_peri + post_search), exclude=bg_exclude, **window_kw)

    #measured lunar effect on target time series
    t = trk['t'] #analysis times
    B_meas = B_b[np.isin(t_b, t)] #target field for given time series
    B_bg = background_at(t, t_c, B_c, fit) #predicted background
    dB_gse = B_meas - B_bg #element-wise subtraction to obtain total effect due to Moon

    #rotate into local east, north, up at the spacecraft
    dB = np.column_stack(sel2enr(rotate(t, dB_gse), trk['lon'], trk['lat'])) #GSE to SEL to local
    bg = np.column_stack(sel2enr(rotate(t, B_bg), trk['lon'], trk['lat']))
    svm = np.column_stack(svm_field(trk['lon'], trk['lat'], trk['alt'])) 
    dBi = dB - svm #rotated interaction field

    #summary over sunlit, low-altitude, valid samples
    use = (~trk['shadow']) & (~in_intervals(t, near_shadow)) & (trk['alt'] < alt_max) & np.all(np.isfinite(dB), axis=1) #mask requiring not in shadow, low altitide, and all 3 components of dB exist
    sig = fit['sigma_mag'] #noise floor
    k_peri = int(np.argmin(trk['alt'])) #index of periapsis

    # post-pass check: did the background stay stable across periapsis?
    if fit_post is None:
        post_status = 'none' #no window with enough data
    elif not fit_post['ok']:
        post_status = 'disturbed' #a window exists but fails the quality test
    else:
        post_status = 'ok'
    post_good = (post_status == 'ok') and fit['ok']

    #dictionary
    s = {
        't_peri': t_peri,
        'min_alt_km': float(trk['alt'][k_peri]),
        'lon_peri': float(trk['lon'][k_peri]), 'lat_peri': float(trk['lat'][k_peri]),
        'sza_peri': float(trk['sza'][k_peri]),
        'shadow_s': float(sum(b - a for a, b in shadows)),
        'monitor_upstream': bool(np.median(trk_mon['xyz_sse'][:, 0]) > 0 and np.min(np.linalg.norm(trk_mon['xyz_sse'], axis=1)) > 3 * R_M),
        'monitor_in_shadow': bool(trk_mon['shadow'].any()),
        'bg_window': fit['window'], 'bg_shift': fit['shift'], 'bg_r': fit['r'],
        'bg_sigma': sig, 'bg_ok': fit['ok'],
        'post_shift': np.nan if fit_post is None else fit_post['shift'],
        'post_sigma': np.nan if fit_post is None else fit_post['sigma_mag'],
        'post_window': None if fit_post is None else fit_post['window'],
        'post_status': post_status,
        'shift_drift_s': fit_post['shift'] - fit['shift'] if post_good else np.nan,
        'offset_drift_nT': float(np.linalg.norm(fit_post['offset'] - fit['offset'])) if post_good else np.nan,
        'n_used': int(use.sum()),
        'shadow_pad': shadow_pad,
        'bg_alt_min': bg_alt_min,
    }
    if use.any():
        mag = lambda v: np.linalg.norm(v[use], axis=1)
        svm_u, dBi_u = svm[use], dBi[use]
        strong = np.linalg.norm(svm_u, axis=1) > 1.0 #where the crust is non-negligible
        k = int(np.argmax(np.linalg.norm(svm_u, axis=1)))
        s.update({
            'peak_svm': float(mag(svm).max()),
            'peak_dB': float(mag(dB).max()),
            'peak_dBi': float(mag(dBi).max()),
            'snr': float(mag(dB).max() / sig),
            'frac_above_3sigma': float(np.mean(mag(dB) > 3 * sig)),
            # + : interaction field along the crustal field (compression); - : against it (cancellation)
            'compression_nT': float(np.median(np.sum(dBi_u[strong] * svm_u[strong], axis=1) / np.linalg.norm(svm_u[strong], axis=1))) if strong.any() else np.nan,
            'n_strong': int(strong.sum()),
            'peak_dBi_crust': float(np.linalg.norm(dBi_u[strong], axis=1).max()) if strong.any() else np.nan,
            't_peak_dB': float(t[use][np.argmax(mag(dB))] - t_peri),
            # angle between the upstream IMF and the crustal field where the crust is strongest
            'shear_deg': _angle(bg[use][k], svm_u[k]),
        })

    return {'summary': s, 't': t, 'use': use, 'alt': trk['alt'], 'lon': trk['lon'], 'lat': trk['lat'],
            'sza': trk['sza'], 'shadow': trk['shadow'], 'B_meas_gse': B_meas, 'B_bg_gse': B_bg,
            'dB_enr': dB, 'svm_enr': svm, 'dBi_enr': dBi, 'bg_enr': bg,
            'fit': fit, 'fit_post': fit_post, 'post_window': None if fit_post is None else fit_post['window']}


def analyse_pass(t_peri, probe='b', monitor='c', half_window=1200.0, search=(1800.0, 120.0), post_search=POST_SEARCH, eclipse_corrections=True, no_update=False, **kw):
    """Load everything for one pass and run analyse_arrays.

    t_peri : periapsis time, as a string ('2013-12-04/00:12') or Unix seconds
    """
    import pyspedas
    from pyspedas import get_data, time_double, time_string

    t_peri = time_double(t_peri)
    pad = kw.get('max_lag', MAX_LAG) + 10.0
    t0, t1 = t_peri - max(half_window, search[0]) - pad, t_peri + post_search + pad
    trange = [time_string(t0), time_string(t1)]

    pyspedas.projects.themis.fgm(probe=[probe, monitor], trange=trange, level='l2', time_clip=True, apply_eclipse_corrections=eclipse_corrections, no_update=no_update)
    b, c = get_data(f'th{probe}_fgs_gse'), get_data(f'th{monitor}_fgs_gse')
    for name, d in ((f'th{probe}', b), (f'th{monitor}', c)):
        if d is None:
            raise RuntimeError(f'{name}: no FGM data loaded')
        if not (d.times[0] <= t_peri <= d.times[-1]):
            raise RuntimeError(f'{name}: FGM data do not contain periapsis (stale or missing data)')

    trk_raw = load_track(probe, trange, trange, no_update=no_update)
    trk_mon = load_track(monitor, trange, trange, no_update=no_update)
    sel = ((b.times >= max(t_peri - half_window, trk_raw['t'][0])) & (b.times <= min(t_peri + half_window, trk_raw['t'][-1])))
    trk = load_track(probe, trange, t_out=b.times[sel], no_update=no_update)

    r = analyse_arrays(t_peri, b.times, b.y, c.times, c.y, trk, trk_raw, trk_mon, half_window=half_window, search=search, post_search=post_search, **kw)

    pos = get_data(f'th{probe}_pos_gse')
    if pos is None:
        pyspedas.projects.themis.state(probe=probe, trange=trange, time_clip=True, no_update=no_update)
        pos = get_data(f'th{probe}_pos_gse')
    near = np.abs(pos.times - t_peri) <= half_window
    ang = float(np.median(tail_angle(pos.y[near])))
    r['summary'].update({'tail_angle_deg': ang, 'regime_geo': regime_geo(ang)})
    return r

def run_passes(t_peris, ids=None, **kw):
    """Run analyse_pass on each periapsis time.

    t_peris : list of periapsis times
    ids     : time labels (optional, e.g. '2013-13'); defaults to 0, 1, 2, ...

    Returns (results, failures): one summary row per successful pass, and the
    time and error message of every pass that failed.
    """
    import pandas as pd
    ids = list(range(len(t_peris))) if ids is None else list(ids)
    rows, failures = [], []
    for i, (pid, tp) in enumerate(zip(ids, t_peris)):
        try:
            s = analyse_pass(tp, **kw)['summary']
            s['pass_id'] = pid
            rows.append(s)
        except Exception as e:
           failures.append({'pass_id': pid, 't_peri': tp, 'error': f'{type(e).__name__}: {e}'})
        print(f'\r{i + 1}/{len(t_peris)} done, {len(failures)} failed', end='')
    return pd.DataFrame(rows), pd.DataFrame(failures)