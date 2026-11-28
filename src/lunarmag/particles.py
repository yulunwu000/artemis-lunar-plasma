"""ESA velocity distributions (THEMIS/ARTEMIS) and electron pitch-angle distributions."""
import numpy as np


def read_esa_distribution(trange, probe='b', datatype='peeb', no_update=False):
    """Read the 3D ESA distributions (L2) overlapping a time range.

    datatype : 'peeb' / 'peer' / 'peef' (electrons: burst / reduced / full), 'peib' etc. for ions

    Returns a dict with one entry per spin (N spins, nE energies, nA angle bins):
        t0, t1      : start and end of each spin (Unix s)
        eflux       : (N, nE, nA) differential energy flux; NaN where the bin is invalid
        energy      : (N, nE, nA) energy of each bin (eV)
        theta, phi  : (N, nE, nA) direction of each bin (deg), DSL frame.
     These are the directions the particles TRAVEL (see beam_direction).
    """
    import cdflib
    from pyspedas import time_double
    from pyspedas.projects import themis

    paths = themis.esd(trange=trange, probe=probe, datatype=datatype, downloadonly=True, no_update=no_update)
    t_lo, t_hi = time_double(trange[0]), time_double(trange[1])
    parts = [_read_one(cdflib.CDF(p), t_lo, t_hi) for p in sorted(paths)]
    parts = [p for p in parts if len(p['t0'])]
    if not parts:
        raise RuntimeError(f'no {datatype} distributions in {trange}')
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _read_one(cdf, t_lo, t_hi):
    """Read one ESA L2 file, keeping the spins that overlap (t_lo, t_hi)."""
    t0 = np.asarray(cdf.varget('time'), float)
    t1 = np.asarray(cdf.varget('end_time'), float)
    keep = (t1 >= t_lo) & (t0 <= t_hi)

    flux = np.asarray(cdf.varget('eflux'), float)[keep]
    good = np.asarray(cdf.varget('bins')) > 0 
    good = good[keep] if good.ndim == 3 else np.broadcast_to(good, flux.shape)
    en_ind = np.asarray(cdf.varget('en_ind')).astype(int)[keep]
    an_ind = np.asarray(cdf.varget('an_ind')).astype(int)[keep]
    dphi = (np.asarray(cdf.varget('phi_offset'), float) + np.asarray(cdf.varget('eclipse_dphi'), float))[keep]

    energy = np.asarray(cdf.varget('energy'), float)
    theta = np.asarray(cdf.varget('theta'), float)
    phi = np.asarray(cdf.varget('phi'), float)
    if energy.ndim == 3: 
        energy, theta, phi = energy[0], theta[0], phi[0]

    n_a = flux.shape[2]
    en = np.repeat(energy[:, en_ind].T[:, :, None], n_a, axis=2)  
    th = theta[:, :, an_ind].transpose(2, 0, 1)      
    ph = phi[:, :, an_ind].transpose(2, 0, 1) + dphi[:, None, None]
    flux = np.where(good & (flux > 0) & (flux < 1e30), flux, np.nan)
    return {'t0': t0[keep], 't1': t1[keep], 'eflux': flux, 'energy': en, 'theta': th, 'phi': ph}


def particle_directions(theta, phi):
    """Unit vectors (..., 3) from latitude theta and longitude phi (deg)."""
    th, ph = np.radians(theta), np.radians(phi)
    return np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)], axis=-1)


def pitch_angles(vhat, bhat):
    """Pitch angle (deg) between unit velocity directions (..., 3) and the field direction bhat (3,)."""
    return np.degrees(np.arccos(np.clip(vhat @ bhat, -1, 1)))


def beam_direction(dist, mask):
    """Direction of the brightest bin, averaged over the spins in mask.

    For ions in the solar wind this should point anti-sunward (x close to -1) if theta/phi
    are travel directions, and sunward (x close to +1) if they are look directions.
    """
    f = np.nanmean(dist['eflux'][mask], axis=0)
    k = np.unravel_index(np.nanargmax(f), f.shape)
    i0 = np.flatnonzero(mask)[0]
    return particle_directions(dist['theta'][i0][k], dist['phi'][i0][k])


def pitch_angle_distribution(dist, b_times, b_vec, emin=20.0, emax=300.0, pa_step=22.5, reverse=False):
    """Mean flux in pitch-angle bins, one row per spin.

    b_times, b_vec : magnetic field times and vectors, in the SAME frame as the distribution (DSL)
    emin, emax     : energy range averaged over (eV); keep emin above the spacecraft potential
    pa_step        : pitch-angle bin width (deg); ~22.5 matches the instrument's angular resolution
    reverse        : True if theta/phi turn out to be look directions rather than travel directions

    Returns dict: t (mid-spin times), pa (bin centres), pad (N, n_bins) mean eflux, NaN if empty.
    """
    edges = np.arange(0, 180 + pa_step, pa_step)
    mids = edges[:-1] + pa_step / 2
    n = len(dist['t0'])
    pad = np.full((n, len(mids)), np.nan)

    for i in range(n):
        m = (b_times >= dist['t0'][i]) & (b_times < dist['t1'][i]) 
        if not m.any():
            continue
        bhat = b_vec[m].mean(axis=0)
        bhat = bhat / np.linalg.norm(bhat)

        v = particle_directions(dist['theta'][i], dist['phi'][i])
        if reverse:
            v = -v
        alpha = pitch_angles(v, bhat)                                  

        f, e = dist['eflux'][i], dist['energy'][i]
        use = (e >= emin) & (e <= emax) & np.isfinite(f)
        k = np.clip(np.digitize(alpha[use], edges) - 1, 0, len(mids) - 1)
        total = np.bincount(k, weights=f[use], minlength=len(mids))
        count = np.bincount(k, minlength=len(mids))
        with np.errstate(invalid='ignore', divide='ignore'):
            pad[i] = total / count                                     

    return {'t': (dist['t0'] + dist['t1']) / 2, 'pa': mids, 'pad': pad}


def load_pad(trange, probe='b', datatype='peeb', emin=20.0, emax=300.0, pa_step=22.5, no_update=False):
    """Load the distribution and field for trange and return its pitch-angle distribution."""
    from pyspedas import get_data
    from pyspedas.projects import themis

    dist = read_esa_distribution(trange, probe, datatype, no_update)
    for res in ('fgl', 'fgs'):                                          
        themis.fgm(trange=trange, probe=probe, varnames=[f'th{probe}_{res}_dsl'], time_clip=True, no_update=no_update)
        b = get_data(f'th{probe}_{res}_dsl')
        if b is not None:
            break
    else:
        raise RuntimeError('no DSL magnetic field for this interval')
    return pitch_angle_distribution(dist, b.times, b.y, emin, emax, pa_step)