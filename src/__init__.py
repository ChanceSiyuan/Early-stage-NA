"""Early-stage-NA: M=2 Virtual Distillation for Quantum Error Mitigation.

Public API
----------

Circuit construction & simulation:
    generate_target_circuit  -- build near-Clifford or local-shallow ansatze
    build_m2_circuit         -- construct the 2N-qubit M=2 protocol circuit
    build_single_copy_circuit-- wrap a target circuit with Z-basis measurement
    transpile_for_method     -- transpile to backend-appropriate basis gates
    run_simulation           -- end-to-end unmitigated + mitigated workflow

Building blocks:
    apply_b2_gate            -- append B^(2) beamsplitter gate to a circuit
    get_pauli_noise_model    -- Pauli-only noise model for trajectory simulation
    compute_unmitigated_expval -- <Z_k> from single-copy counts
    compute_mitigated_expval   -- mitigated <Z_k> from M=2 counts

Example
-------
>>> from src import run_simulation, generate_target_circuit, get_pauli_noise_model
>>> circuit = generate_target_circuit(5, "local_shallow", seed=42)
>>> noise = get_pauli_noise_model(0.01, 0.02)
>>> result = run_simulation(circuit, noise, "matrix_product_state", shots=10000)
>>> print(f"Unmitigated: {result['unmitigated_z']:.4f}")
>>> print(f"Mitigated:   {result['mitigated_z']:.4f}")
"""

from src.utils import (
    apply_b2_gate,
    compute_mitigated_expval,
    compute_unmitigated_expval,
    get_pauli_noise_model,
)
from src.simulation import (
    build_m2_circuit,
    build_single_copy_circuit,
    generate_target_circuit,
    run_simulation,
    transpile_for_method,
)

__all__ = [
    "apply_b2_gate",
    "build_m2_circuit",
    "build_single_copy_circuit",
    "compute_mitigated_expval",
    "compute_unmitigated_expval",
    "generate_target_circuit",
    "get_pauli_noise_model",
    "run_simulation",
    "transpile_for_method",
]
