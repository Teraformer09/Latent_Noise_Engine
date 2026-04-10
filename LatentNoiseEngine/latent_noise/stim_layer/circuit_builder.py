# NOTE: This module is not used by the current pipeline. It is kept for
# future Qiskit-based circuit export. Qiskit is not listed in requirements.txt;
# the import is guarded so the rest of the package loads without it installed.
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    _QISKIT_OK = True
except ImportError:
    _QISKIT_OK = False

import numpy as np

def build_surface_code_circuit(p):
    if not _QISKIT_OK:
        raise ImportError("qiskit is required for build_surface_code_circuit. Run: pip install qiskit")
    # p is a dict: {'px': float, 'py': float, 'pz': float}
    
    # 9 data qubits, 4 ancilla qubits for X-checks
    data = QuantumRegister(9, 'data')
    ancilla = QuantumRegister(4, 'ancilla')
    cr = ClassicalRegister(4, 'syndrome')
    qc = QuantumCircuit(data, ancilla, cr)
    
    # Normally we would initialize to |0>
    
    # Inject noise (Pauli Z errors on data qubits for X-checks)
    # Note: In a real simulation, we'd use Aer's noise model, 
    # but to be explicit and flexible, we can apply rotations or Pauli's.
    # Since we are using the 'clifford' simulator, we can just use 
    # random Pauli Z with probability pz.
    
    # For now, let's keep the circuit structure and use Aer's noise model 
    # in the sampler for the noise injection.
    
    # X-Checks (detect Z-errors)
    checks = [
        [0, 1, 3, 4], # C0
        [1, 2, 4, 5], # C1
        [3, 4, 6, 7], # C2
        [4, 5, 7, 8]  # C3
    ]
    
    for i, qubits in enumerate(checks):
        qc.h(ancilla[i])
        for q in qubits:
            qc.cx(ancilla[i], data[q])
        qc.h(ancilla[i])
        qc.measure(ancilla[i], cr[i])
        
    return qc

def get_logical_z_circuit():
    # To measure logical Z, we can just measure data qubits 0, 3, 6
    data = QuantumRegister(9, 'data')
    cr = ClassicalRegister(3, 'logical')
    qc = QuantumCircuit(data, cr)
    qc.measure(data[0], cr[0])
    qc.measure(data[3], cr[1])
    qc.measure(data[6], cr[2])
    return qc