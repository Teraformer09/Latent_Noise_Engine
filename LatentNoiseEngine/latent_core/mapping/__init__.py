"""Latent-to-channel mappings: softmax, Pauli channel, CPTP evolution."""
from .softmax import softmax, theta_to_probs, init_softmax_weights
from .pauli_channel import apply_pauli_channel, kraus_operators, evolve_step, check_channel
