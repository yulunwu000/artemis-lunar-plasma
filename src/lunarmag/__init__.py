"""lunarmag: tools for comparing ARTEMIS magnetic field data with lunar crustal field models."""
from .svm import R_M, load_svm, svm_field, svm_map
from .frames import sel2enr, enr2sel, gse2sel
from .track import load_track, xyz_to_lonlatalt, solar_zenith, in_shadow, shadow_intervals
from .background import on_grid, fit_background, background_at, choose_quiet_window
from .pipeline import analyse_arrays, analyse_pass