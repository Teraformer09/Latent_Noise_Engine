"""Physics layer: Hamiltonian construction, latent state dynamics."""
from .hamiltonian import build_hamiltonian, hamiltonian_from_latent
from .coupling import latent_step, build_stable_A
