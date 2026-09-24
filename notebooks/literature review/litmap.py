#!/usr/bin/env python3
"""
Literature map: solar-wind interaction with lunar crustal magnetic anomalies.

Plots every reviewed paper by publication year (x) and method (y), coloured by
which physical picture it supports. Reads papers.csv next to this script and
writes literature_map.png (300 dpi) and literature_map.svg.

Usage
-----
    python litmap.py                 # outputs next to the script
    python litmap.py --out figures/  # outputs into figures/

papers.csv columns
------------------
    key     unique id (not plotted)
    label   text shown next to the marker, e.g. "Halekas 2014"
    year    publication year; values < 1990 are drawn left of the axis break
    row     method row: one of the keys in ROWS below
    interp  picture supported: one of the keys in INTERP below
    dx, dy  label offset from the marker (dx in years, dy in rows)
    ha      label alignment: left | center | right  (use dy = 0 for side labels)

Requires: matplotlib >= 3.5
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "svg.fonttype": "none",        # keep text editable in the SVG
})

# ---- style -----------------------------------------------------------------
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f1f0ec"

# colour, marker, legend label -- names follow the deck's section titles.
# The five colours pass an all-pairs colour-vision check only together with
# the distinct marker shapes and direct labels, so keep all three.
INTERP = {
    "fluid":     ("#2a78d6", "o", "Fluid mini-magnetosphere"),
    "whistler":  ("#1baf7a", "D", "Wave–particle (whistler vs shock)"),
    "ambipolar": ("#eda100", "^", "Charge-separated field"),
    "kinetic":   ("#e34948", "s", "Partial shielding (reflected ions)"),
    "recon":     ("#4a3aa7", "*", "Electron-only reconnection"),
    "context":   ("#b5b4ae", "o", "Inputs / baseline"),
}

# method rows, top to bottom: key, label, group
ROWS = [
    ("surface",     "Surface (rover)",           "Observations"),
    ("ena",         "Orbit: ENA imaging",        "Observations"),
    ("orb_ions",    "Orbit: ion spectra",        "Observations"),
    ("orb_fields",  "Orbit: fields & electrons", "Observations"),
    ("maps",        "Crustal field maps",        "Observations"),
    ("fluid_sim",   "MHD / Hall-MHD",            "Models"),
    ("kinetic_sim", "Hybrid / PIC",              "Models"),
    ("lab",         "Laboratory",                "Models"),
    ("theory",      "Theory & analogue",         "Theory"),
]

# crustal field maps are drawn as one grey span (inputs, not interpretations)
FIELD_MAP_YEARS = [2001, 2008, 2015]   # Hood; Mitchell / Richmond & Hood / Purucker; Tsunakawa
FIELD_MAP_LABEL = r"Hood 2001 → Tsunakawa 2015 (model inputs: $B_\mathrm{surf}$, $d$, $L$)"

FOOTNOTE = ("Colour = our reading of each paper's interpretation.  Not shown: "
            "regolith-scattering baselines (Saito 2008; Wieser 2009; Schaufelberger 2011).")

# x-axis: 1971 is drawn at X_EARLY, with a break mark between it and 2000
X_EARLY, X_BREAK = 1993.0, 1995.0
X0, X1 = 1991.2, 2027.4
RECENT_FROM = 2020


def xpos(year: float) -> float:
    return X_EARLY if year < 1990 else year


def load_papers(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        papers = list(csv.DictReader(f))
    row_keys = {k for k, _, _ in ROWS}
    for p in papers:
        if p["row"] not in row_keys:
            raise ValueError(f"{p['key']}: unknown row '{p['row']}'")
        if p["interp"] not in INTERP:
            raise ValueError(f"{p['key']}: unknown interp '{p['interp']}'")
    return papers


def draw(papers: list[dict]) -> plt.Figure:
    y_of = {k: len(ROWS) - 1 - i for i, (k, _, _) in enumerate(ROWS)}
    mixed = len({p["found"].strip() for p in papers}) > 1

    fig, ax = plt.subplots(figsize=(13.33, 5.9), dpi=100)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.83, bottom=0.13)

    # recency band
    ax.axvspan(RECENT_FROM, X1, color=BAND, zorder=0, lw=0)
    ax.text((RECENT_FROM + X1) / 2, len(ROWS) - 0.35, f"since {RECENT_FROM}",
            ha="center", va="bottom", color=INK2, fontsize=11)

    # row guides, row labels, group separators and group labels
    for key, label, _ in ROWS:
        ax.axhline(y_of[key], color=GRID, lw=0.8, zorder=1)
        ax.text(X0 - 0.4, y_of[key], label, ha="right", va="center", color=INK)
    groups = []
    for _, _, g in ROWS:
        if g not in groups:
            groups.append(g)
    for g in groups[1:]:
        first = next(k for k, _, gg in ROWS if gg == g)
        ax.axhline(y_of[first] + 0.5, color=MUTED, lw=1.0, zorder=1)
    for g in groups:
        ys = [y_of[k] for k, _, gg in ROWS if gg == g]
        ax.text(X0 - 7.9, (max(ys) + min(ys)) / 2, g, rotation=90, ha="center",
                va="center", color=INK2, fontweight="bold")

    # field maps span
    ym = y_of["maps"]
    ax.plot([FIELD_MAP_YEARS[0], FIELD_MAP_YEARS[-1]], [ym, ym], color=INTERP["context"][0],
            lw=2, zorder=2, solid_capstyle="round")
    for yr in FIELD_MAP_YEARS:
        ax.plot(yr, ym, "o", ms=8, color=INTERP["context"][0], mec="white", mew=2, zorder=3)
    ax.text(sum(FIELD_MAP_YEARS) / len(FIELD_MAP_YEARS), ym + 0.30, FIELD_MAP_LABEL,
            ha="center", va="bottom", color=INK2, fontsize=11)

    # papers
    for p in papers:
        colour, marker, _ = INTERP[p["interp"]]
        x, y = xpos(float(p["year"])), y_of[p["row"]]
        ms = 15 if marker == "*" else 10
        if mixed and p["found"].strip() == "us":
            ax.plot(x, y, "o", ms=ms + 8, mfc="none", mec=INK, mew=1.6, zorder=3)
        ax.plot(x, y, marker, ms=ms, color=colour, mec="white", mew=1.5, zorder=4)
        dy = float(p["dy"])
        ax.text(x + float(p["dx"]), y + dy, p["label"], ha=p["ha"],
                va="center" if dy == 0 else ("bottom" if dy > 0 else "top"),
                fontsize=11, color=INK2 if p["interp"] == "context" else INK, zorder=5)

    # axes, ticks and the axis break
    ax.set_xlim(X0, X1)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([X_EARLY, 2000, 2005, 2010, 2015, 2020, 2025],
                  ["1971", "2000", "2005", "2010", "2015", "2020", "2025"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelcolor=INK2, length=4)
    ax.plot([X_BREAK - 0.2, X_BREAK + 0.2], [-0.6, -0.6], color="white", lw=4,
            clip_on=False, zorder=5)
    for off in (-0.25, 0.25):
        ax.plot([X_BREAK + off - 0.2, X_BREAK + off + 0.2], [-0.72, -0.48],
                color=MUTED, lw=1.2, clip_on=False, zorder=6)

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", type=Path, default=HERE / "papers.csv")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    fig = draw(load_papers(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        path = args.out / f"literature_map.{ext}"
        fig.savefig(path, **kw)
        print("wrote", path)


if __name__ == "__main__":
    main()
