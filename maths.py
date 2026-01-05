from modules.framework.ops.advanced_math import DiracOps, RelativisticOps, LogicOps
from modules.framework.complex import ComplexTensor, fft, ifft

# Expose these under more user-friendly names
def bra_ket(bra, ket): return DiracOps.bra_ket(bra, ket)
def ket_bra(ket, bra): return DiracOps.ket_bra(ket, bra)

def minkowski_distance(v): return RelativisticOps.minkowski_norm(v)
def lorentz_boost(v, speed): return RelativisticOps.lorentz_boost(v, speed)

def fuzzy_and(a, b): return LogicOps.AND(a, b)
def fuzzy_or(a, b): return LogicOps.OR(a, b)
def fuzzy_not(a): return LogicOps.NOT(a)

__all__ = ["bra_ket", "ket_bra", "minkowski_distance", "lorentz_boost", "fuzzy_and", "fuzzy_or", "ComplexTensor", "fft"]
