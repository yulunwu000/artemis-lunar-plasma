"""Map projections and plotting helpers for crustal-field maps."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .frames import _enr_axes

def project(lon, lat, lon0, lat0, lambert=False):
    """Page coords (x, y) and depth = cos c of points (lon, lat), viewed centred on (lon0, lat0)."""
    lam, phi = np.radians(lat), np.radians(lon) #convert to radians
    p = np.stack([np.cos(lam)*np.cos(phi), np.cos(lam)*np.sin(phi), np.sin(lam)], axis=-1) #unit position vector in SEL
    e, n, r = (a[0] for a in _enr_axes(lon0, lat0))
    x, y, depth = p @ e, p @ n, p @ r #orthographic: length sin c
    if lambert:
        k = np.sqrt(2 / (1 + np.clip(depth, -1 + 1e-12, 1)))   # rescale to 2 sin(c/2)
        x, y = k * x, k * y
    return x, y, depth

def globe(ax, lats, lons, field, lon0, lat0, lambert=False, dlon=30, lat_lines=(-60, -30, 0, 30, 60), grid_color='w', **kw):
    LON, LAT = np.meshgrid(lons, lats)
    x, y, depth = project(LON, LAT, lon0, lat0, lambert)
    pc = ax.pcolormesh(x, y, np.ma.masked_where(depth <= 0, field), shading='gouraud', rasterized=True, **kw)
    lines = [(np.full(181, lo), np.linspace(-90, 90, 181)) for lo in range(0, 360, dlon)]
    lines += [(np.linspace(0, 360, 361), np.full(361, la)) for la in lat_lines]
    for lo, la in lines:
        x, y, d = project(lo, la, lon0, lat0, lambert)
        ax.plot(np.where(d > 0, x, np.nan), np.where(d > 0, y, np.nan), color=grid_color, lw=0.3, alpha=0.6)
    R = np.sqrt(2) if lambert else 1
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color='k', lw=0.8))
    ax.set_xlim(-1.02*R, 1.02*R); ax.set_ylim(-1.02*R, 1.02*R)    # stop hidden hemisphere setting the limits
    ax.set_aspect('equal'); ax.axis('off')
    return pc

#colour scheme from Tsunakawa 
tsu = LinearSegmentedColormap.from_list('tsunakawa', [
    (0.00, '#1a2a80'), (0.15, '#2f5fb5'), (0.30, '#20b0b0'), (0.40, '#7fbf3f'),
    (0.46, '#e4f0c0'), (0.50, '#ffffff'), (0.54, '#fff4c0'), (0.62, '#f5c020'),
    (0.75, '#f08020'), (0.88, '#e02020'), (1.00, '#9a1020')])
tsu.set_under('#101848'); tsu.set_over('#300808')