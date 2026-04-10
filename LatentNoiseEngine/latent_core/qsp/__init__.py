"""Quantum Signal Processing: polynomial construction, phase solver, transform."""
from .polynomial import build_polynomial, eval_polynomial, chebyshev_basis, lowpass_coeffs
from .phase_solver import solve_phases, qsp_unitary, qsp_real_part, validate_phases
from .transform import qsp_effective_hamiltonian, apply_polynomial_to_operator
from .circuit import QSPCircuit
