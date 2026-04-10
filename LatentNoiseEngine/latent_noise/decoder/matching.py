import numpy as np
import pymatching

def get_rotated_surface_code_hx(d):
    num_qubits = d * d
    num_checks = (d * d - 1) // 2
    H = np.zeros((num_checks, num_qubits), dtype=int)
    check_idx = 0
    for r in range(d + 1):
        for c in range(d + 1):
            if (r + c) % 2 == 1:
                qubits = []
                for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d:
                        qubits.append(nr * d + nc)
                if len(qubits) >= 2 and check_idx < num_checks:
                    H[check_idx, qubits] = 1
                    check_idx += 1
    return H

def get_logical_z_op(d):
    op = np.zeros(d * d, dtype=int)
    # Logical Z for the rotated surface code: a string of Z gates along the
    # top row (column 0 to d-1), connecting the two smooth (Z) boundaries.
    for c in range(d):
        op[c] = 1  # top row
    return op

def build_decoder_for_distance(d):
    H = get_rotated_surface_code_hx(d)
    return pymatching.Matching(H)
