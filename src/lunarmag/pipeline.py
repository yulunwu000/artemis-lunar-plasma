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
from .background import background_at, choose_quiet_window


def _angle(a, b):
    """Angle (deg) between two vectors."""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def analyse_arrays(t_peri, t_b, B_b, t_c, B_c, trk, trk_raw, trk_mon, half_window=1200.0, search=(1800.0, 120.0), alt_max=300.0, rotate=gse2sel, shadow_pad=120.0, post_search = 3600.00, **window_kw):
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
    window_kw     : passed to choose_quiet_window (length, step, sigma_max, r_min, ...)
    """

    #background: quiet window before and after pass
    shadows = shadow_intervals(trk_raw['t'], trk_raw['shadow']) #list of (start, end) times for all shadow periods
    near_shadow = shadow_intervals(trk_raw['t'], trk_raw['shadow'], pad=shadow_pad, outer=True) #add buffer

    #find best quiet window before periapsis
    fit = choose_quiet_window(t_b, B_b, t_c, B_c, span=(t_peri - search[0], t_peri - search[1]), exclude=near_shadow, **window_kw)
    if fit is None:
        raise RuntimeError('no quiet window with enough data before periapsis')

    #find best quiet window after periapsis
    fit_post = choose_quiet_window(t_b, B_b, t_c, B_c, span=(t_peri + search[1], t_peri + post_search), exclude=near_shadow, **window_kw)

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
        'n_used': int(use.sum()),
        'shadow_pad': shadow_pad,
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
            # angle between the upstream IMF and the crustal field where the crust is strongest
            'shear_deg': _angle(bg[use][k], svm_u[k]),
        })

    return {'summary': s, 't': t, 'use': use, 'alt': trk['alt'], 'lon': trk['lon'], 'lat': trk['lat'],
            'sza': trk['sza'], 'shadow': trk['shadow'], 'B_meas_gse': B_meas, 'B_bg_gse': B_bg,
            'dB_enr': dB, 'svm_enr': svm, 'dBi_enr': dBi, 'bg_enr': bg,
            'fit': fit, 'fit_post': fit_post, 'post_window': None if fit_post is None else fit_post['window']}


def analyse_pass(t_peri, probe='b', monitor='c', half_window=1200.0, search=(1800.0, 120.0), post_search=3600.0, eclipse_corrections=True, **kw):
    """Load everything for one pass and run analyse_arrays.

    t_peri : periapsis time, as a string ('2013-12-04/00:12') or Unix seconds
    """
    import pyspedas
    from pyspedas import get_data, time_double, time_string

    t_peri = time_double(t_peri)
    pad = 120.0
    t0, t1 = t_peri - max(half_window, search[0]) - pad, t_peri + post_search + pad
    trange = [time_string(t0), time_string(t1)]

    pyspedas.projects.themis.fgm(probe=[probe, monitor], trange=trange, level='l2', time_clip=True, apply_eclipse_corrections=eclipse_corrections)
    b, c = get_data(f'th{probe}_fgs_gse'), get_data(f'th{monitor}_fgs_gse')
    if b is None or c is None:
        raise RuntimeError('FGM data missing for target or monitor')

    trk_raw = load_track(probe, trange)
    trk_mon = load_track(monitor, trange)
    sel = ((b.times >= max(t_peri - half_window, trk_raw['t'][0])) & (b.times <= min(t_peri + half_window, trk_raw['t'][-1])))
    trk = load_track(probe, trange, t_out=b.times[sel])

    return analyse_arrays(t_peri, b.times, b.y, c.times, c.y, trk, trk_raw, trk_mon, half_window=half_window, post_search=post_search, **kw)