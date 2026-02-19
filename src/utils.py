"""Low-level building blocks for the M=2 Virtual Distillation protocol.

This module provides:
- B^(2) beamsplitter gate decomposition into native Qiskit gates
- Pauli-only noise model construction for trajectory-based simulation
- Post-processing functions for mitigated/unmitigated expectation values

The B^(2) gate jointly diagonalizes per-pair SWAP operators, enabling
estimation of Tr(O rho^2) / Tr(rho^2) from computational-basis measurements
on two copies of a noisy quantum state.

References:
    Huggins et al., "Virtual Distillation for Quantum Error Mitigation" (2021)
    Koczor, "Error Suppression by Derangement" (2020)
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error


# ---------------------------------------------------------------------------
# B^(2) beamsplitter gate
# ---------------------------------------------------------------------------

def apply_b2_gate(circuit: QuantumCircuit, q_copy1: int, q_copy2: int) -> None:
    """Append the M=2 beamsplitter gate B^(2) between a qubit pair.

    The B^(2) gate diagonalizes the 2-qubit SWAP operator in the
    computational basis.  Its unitary is::

        B^(2) = |00><00|
              + (|01> + |10>)(<01| / sqrt2)
              + (-|01> + |10>)(<10| / sqrt2)
              + |11><11|

    Decomposition into native gates (4 CX + 2 Ry)::

        CNOT(q1 -> q2)  -->  CRy(-pi/2, ctrl=q2, tgt=q1)  -->  CNOT(q1 -> q2)

    Each B^(2) contributes approximately 2 non-Clifford gates when transpiled
    to the Clifford+T basis (relevant for the extended_stabilizer backend).

    CRITICAL: the CRy angle must be **-pi/2** (negative). Using +pi/2
    transposes the off-diagonal signs and breaks SWAP diagonalization.

    Args:
        circuit: Quantum circuit to modify in-place.
        q_copy1: Qubit index belonging to copy 1.
        q_copy2: Qubit index belonging to copy 2.
    """
    circuit.cx(q_copy1, q_copy2)
    # CRy(-pi/2) decomposes internally into 2 CX + 2 Ry(+/-pi/4)
    circuit.cry(-np.pi / 2, q_copy2, q_copy1)
    circuit.cx(q_copy1, q_copy2)


# ---------------------------------------------------------------------------
# Pauli noise model factory
# ---------------------------------------------------------------------------

# Gate names that may appear after transpilation for any of the three methods.
_SINGLE_QUBIT_GATES = [
    "h", "s", "sdg", "t", "tdg",
    "x", "y", "z",
    "rx", "ry", "rz",
    "u", "u1", "u2", "u3",
    "id", "sx", "sxdg",
]
_TWO_QUBIT_GATES = ["cx", "cz", "cy", "swap", "cry"]


def get_pauli_noise_model(
    p_1q: float,
    p_2q: float,
    noise_type: str = "depolarizing",
) -> NoiseModel:
    """Build a Pauli-only noise model for trajectory-based simulation.

    All error channels are constructed exclusively from Pauli operators,
    ensuring compatibility with the ``extended_stabilizer`` and
    ``matrix_product_state`` simulation methods in Qiskit Aer.  These
    backends handle Pauli noise via **Monte Carlo trajectory sampling**:
    each shot stochastically applies a single Pauli error after every gate,
    keeping the state representation pure throughout the simulation and
    avoiding the exponential memory cost of a full density matrix.

    Supported ``noise_type`` values:

    ============== ================================ ================================
    noise_type     1-qubit channel                  2-qubit channel
    ============== ================================ ================================
    depolarizing   Depolarizing(p_1q)               Depolarizing(p_2q)
    bit_flip       X with prob p_1q                 X*X with prob p_2q
    phase_flip     Z with prob p_1q                 Z*Z with prob p_2q
    bit_phase_flip Y with prob p_1q                 Y*Y with prob p_2q
    ============== ================================ ================================

    To swap in a different Pauli channel in the future, add a new branch
    that constructs a ``pauli_error`` with the desired Pauli-string
    probabilities.  As long as every error is a probabilistic mixture of
    Pauli operators, trajectory-based simulation will remain valid.

    Args:
        p_1q: Error probability for single-qubit gates (must be in [0, 1]).
        p_2q: Error probability for two-qubit gates (must be in [0, 1]).
        noise_type: One of ``'depolarizing'``, ``'bit_flip'``,
            ``'phase_flip'``, ``'bit_phase_flip'``.

    Returns:
        A :class:`~qiskit_aer.noise.NoiseModel` ready to pass to
        ``AerSimulator``.

    Raises:
        ValueError: If probabilities are out of range or noise_type is
            unknown.
    """
    if not (0 <= p_1q <= 1):
        raise ValueError(f"p_1q must be in [0, 1], got {p_1q}")
    if not (0 <= p_2q <= 1):
        raise ValueError(f"p_2q must be in [0, 1], got {p_2q}")

    if noise_type == "depolarizing":
        error_1q = depolarizing_error(p_1q, 1)
        error_2q = depolarizing_error(p_2q, 2)
    elif noise_type == "bit_flip":
        error_1q = pauli_error([("X", p_1q), ("I", 1 - p_1q)])
        error_2q = pauli_error([("XX", p_2q), ("II", 1 - p_2q)])
    elif noise_type == "phase_flip":
        error_1q = pauli_error([("Z", p_1q), ("I", 1 - p_1q)])
        error_2q = pauli_error([("ZZ", p_2q), ("II", 1 - p_2q)])
    elif noise_type == "bit_phase_flip":
        error_1q = pauli_error([("Y", p_1q), ("I", 1 - p_1q)])
        error_2q = pauli_error([("YY", p_2q), ("II", 1 - p_2q)])
    else:
        raise ValueError(
            f"Unknown noise_type: '{noise_type}'. "
            "Supported: 'depolarizing', 'bit_flip', 'phase_flip', 'bit_phase_flip'"
        )

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1q, _SINGLE_QUBIT_GATES)
    noise_model.add_all_qubit_quantum_error(error_2q, _TWO_QUBIT_GATES)
    return noise_model


# ---------------------------------------------------------------------------
# Post-processing: expectation values
# ---------------------------------------------------------------------------

def compute_unmitigated_expval(
    counts: dict[str, int],
    observable_qubit: int = 0,
) -> float:
    """Compute the unmitigated <Z_k> from single-copy measurement counts.

    Args:
        counts: Qiskit counts dictionary from an N-qubit circuit.
        observable_qubit: Logical qubit index *k* for the Z observable.

    Returns:
        The raw expectation value ``<Z_k>``.
    """
    total = 0
    expval = 0.0
    for bitstring, count in counts.items():
        n_qubits = len(bitstring)
        # Qiskit big-endian: bitstring[0] = qubit N-1, bitstring[-1] = qubit 0
        bit_k = int(bitstring[n_qubits - 1 - observable_qubit])
        z_k = 1 - 2 * bit_k  # 0 -> +1, 1 -> -1
        expval += count * z_k
        total += count
    return expval / total


def compute_mitigated_expval(
    counts: dict[str, int],
    observable_qubit: int = 0,
    n_qubits: int | None = None,
) -> tuple[float, float, float]:
    """Compute mitigated <Z_k> via M=2 Virtual Distillation post-processing.

    Circuit layout assumed:
        Qubits  0 .. N-1    : copy 1
        Qubits  N .. 2N-1   : copy 2

    For each measurement shot the function evaluates the diagonalized form
    of ``Tr(Z_k rho^2) / Tr(rho^2)`` derived from the B^(2) beamsplitter
    transformation (see Huggins et al., Eq. 11-14).

    Per-shot formulas (using bit values b1[j], b2[j] in {0, 1})::

        SWAP eigenvalue for pair j:
            s_j = -1   if b1[j] == 0 and b2[j] == 1
            s_j = +1   otherwise

        Denominator (estimates Tr(rho^2)):
            D_shot = prod_j  s_j          (always +/- 1)

        Observable factor for Z_k:
            obs_k = 1 - b1[k] - b2[k]     (+1 if both 0, -1 if both 1, 0 if disagree)

        Numerator (estimates Tr(Z_k rho^2)):
            E_shot = obs_k * D_shot / s_k

    The mitigated expectation value is ``mean(E) / mean(D)``.

    Args:
        counts: Qiskit counts dictionary from the 2N-qubit M=2 circuit.
        observable_qubit: Logical qubit index *k* for the Z observable.
        n_qubits: Number of logical qubits *N*. Inferred from the bit string
            length (``len / 2``) when not provided.

    Returns:
        ``(mitigated_value, mean_numerator, mean_denominator)`` where the
        last two values are diagnostic (the denominator estimates
        ``Tr(rho^2)``).  Returns ``(nan, ...)`` if the denominator is
        effectively zero.
    """
    numerator_sum = 0.0
    denominator_sum = 0.0
    total_shots = 0

    for bitstring, count in counts.items():
        n = n_qubits if n_qubits is not None else len(bitstring) // 2

        # ------------------------------------------------------------------
        # Bit extraction  (Qiskit big-endian convention)
        #   bitstring has length 2N.
        #   bitstring[2N - 1 - j]  ->  qubit j       (copy 1, j in 0..N-1)
        #   bitstring[N  - 1 - j]  ->  qubit N+j     (copy 2, j in 0..N-1)
        # ------------------------------------------------------------------
        d_shot = 1
        s_values = [0] * n
        for j in range(n):
            b1_j = int(bitstring[2 * n - 1 - j])
            b2_j = int(bitstring[n - 1 - j])
            s_j = -1 if (b1_j == 0 and b2_j == 1) else 1
            s_values[j] = s_j
            d_shot *= s_j

        # Observable factor for Z_k
        k = observable_qubit
        b1_k = int(bitstring[2 * n - 1 - k])
        b2_k = int(bitstring[n - 1 - k])
        obs_k = 1 - b1_k - b2_k  # +1 / 0 / -1

        # Numerator: obs_k * prod_{j != k} s_j  =  obs_k * D_shot / s_k
        # When obs_k == 0 the result is 0 regardless of s_k.
        if obs_k == 0:
            e_shot = 0.0
        else:
            e_shot = obs_k * d_shot / s_values[k]

        numerator_sum += e_shot * count
        denominator_sum += d_shot * count
        total_shots += count

    mean_num = numerator_sum / total_shots
    mean_den = denominator_sum / total_shots

    if abs(mean_den) < 1e-15:
        # Tr(rho^2) too small to estimate -- state is nearly maximally mixed.
        return float("nan"), mean_num, mean_den

    return mean_num / mean_den, mean_num, mean_den
