"""
lunarmag: tools for comparing ARTEMIS magnetic field data with lunar crustal field models.
"""

from .svm import R_M, load_svm, svm_field, svm_map
from .frames import sel2enr, enr2sel, gse2sel