"""JAX engine: JIT ops, vmap vectorization, functional random keys, autodiff."""
from .random import make_key, split_key, split_keys, make_trajectory_keys, make_batch_keys
from .jit_ops import jit_softmax, jit_apply_pauli_channel, jit_fidelity, jit_trace_distance
from .vectorization import vmap_trajectories, vmap_metrics
