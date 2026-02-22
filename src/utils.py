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
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error


# ---------------------------------------------------------------------------
# B^(2) beamsplitter gate
# ---------------------------------------------------------------------------

def apply_b2_gate(circuit: QuantumCircuit, q_copy1: int, q_copy2: int) -> None:
    """Append the M=2 beamsplitter gate B^(2)† between a qubit pair.

    The VD protocol requires B^(2)† (the *inverse* of the SWAP-diagonalizing
    unitary) so that symmetric input states map to the SWAP +1 eigenspace
    in the measurement basis.  The forward transform B^(2) diagonalizes
    SWAP as ``B†·SWAP·B = diag(1, 1, -1, 1)``; applying B† in the circuit
    ensures ``E[D] = Tr(ρ²)`` for any state ρ.

    Decomposition into native gates (4 CX + 2 Ry)::

        CNOT(q1 -> q2)  -->  CRy(+pi/2, ctrl=q2, tgt=q1)  -->  CNOT(q1 -> q2)

    Each B^(2)† contributes approximately 2 non-Clifford gates when transpiled
    to the Clifford+T basis (relevant for the extended_stabilizer backend).

    Args:
        circuit: Quantum circuit to modify in-place.
        q_copy1: Qubit index belonging to copy 1.
        q_copy2: Qubit index belonging to copy 2.
    """
    circuit.cx(q_copy1, q_copy2)
    circuit.cry(np.pi / 2, q_copy2, q_copy1)
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
    p_readout: float = 0.0,
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
        p_readout: Symmetric readout (SPAM) error probability. When non-zero,
            a symmetric bit-flip readout error is added to every qubit:
            P(0|1) = P(1|0) = p_readout.

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
    if not (0 <= p_readout <= 0.5):
        raise ValueError(f"p_readout must be in [0, 0.5], got {p_readout}")

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

    if p_readout > 0:
        ro_probs = [[1 - p_readout, p_readout],
                     [p_readout, 1 - p_readout]]
        noise_model.add_all_qubit_readout_error(ReadoutError(ro_probs))

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


def compute_unmitigated_string_expval(
    counts: dict[str, int],
    observable_qubits: list[int],
) -> float:
    """Compute the unmitigated expectation value of a Z-string operator.

    Evaluates ``<W> = <prod_{k in S} Z_k>`` from single-copy measurement
    counts.  Each Z_k contributes +1 if qubit k measured 0, or -1 if
    qubit k measured 1.  The product over the set S gives the per-shot
    contribution.

    Args:
        counts: Qiskit counts dictionary from an N-qubit circuit.
        observable_qubits: List of logical qubit indices defining the
            Z-string operator ``W = prod_{k in observable_qubits} Z_k``.

    Returns:
        The raw (unmitigated) expectation value ``<W>``.
    """
    if not observable_qubits:
        return 1.0  # empty product = identity operator

    total = 0
    expval = 0.0
    for bitstring, count in counts.items():
        n_qubits = len(bitstring)
        w_shot = 1
        for k in observable_qubits:
            bit_k = int(bitstring[n_qubits - 1 - k])
            w_shot *= 1 - 2 * bit_k  # 0 -> +1, 1 -> -1
        expval += count * w_shot
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


def compute_mitigated_string_expval(
    counts: dict[str, int],
    observable_qubits: list[int],
    n_qubits: int | None = None,
) -> tuple[float, float, float]:
    """Compute mitigated expectation value of a Z-string operator via M=2 VD.

    For string operator ``W = prod_{k in S} Z_k``, the per-shot estimator
    generalises the single-qubit formula (see
    :func:`compute_mitigated_expval`).

    Per-shot formulas for the multi-qubit case::

        SWAP eigenvalue for pair j:
            s_j = -1   if b1[j] == 0 and b2[j] == 1
            s_j = +1   otherwise

        Denominator (estimates Tr(rho^2)):
            D_shot = prod_j  s_j

        Observable factor for each k in S:
            obs_k = 1 - b1[k] - b2[k]   (+1 / 0 / -1)

        Numerator (estimates Tr(W rho^2)):
            E_shot = (prod_{k in S} obs_k) * D_shot / (prod_{k in S} s_k)

    When any ``obs_k == 0`` the shot contributes nothing to the numerator
    (the two copies disagree on qubit k).

    Args:
        counts: Qiskit counts dictionary from the 2N-qubit M=2 circuit.
        observable_qubits: List of logical qubit indices defining the
            Z-string operator ``W = prod_k Z_k``.
        n_qubits: Number of logical qubits *N*.  Inferred from the bit
            string length (``len / 2``) when not provided.

    Returns:
        ``(mitigated_value, mean_numerator, mean_denominator)``.  Returns
        ``(nan, ...)`` if the denominator is effectively zero.
    """
    if not observable_qubits:
        # Empty product → identity operator: Tr(I * rho^2) / Tr(rho^2) = 1
        return 1.0, 1.0, 1.0

    numerator_sum = 0.0
    denominator_sum = 0.0
    total_shots = 0

    for bitstring, count in counts.items():
        n = n_qubits if n_qubits is not None else len(bitstring) // 2

        # Bit extraction (Qiskit big-endian convention)
        d_shot = 1
        s_values = [0] * n
        for j in range(n):
            b1_j = int(bitstring[2 * n - 1 - j])
            b2_j = int(bitstring[n - 1 - j])
            s_j = -1 if (b1_j == 0 and b2_j == 1) else 1
            s_values[j] = s_j
            d_shot *= s_j

        # Observable factors for each qubit in the string
        obs_product = 1
        s_product = 1
        zero_obs = False
        for k in observable_qubits:
            b1_k = int(bitstring[2 * n - 1 - k])
            b2_k = int(bitstring[n - 1 - k])
            obs_k = 1 - b1_k - b2_k
            if obs_k == 0:
                zero_obs = True
                break
            obs_product *= obs_k
            s_product *= s_values[k]

        if zero_obs:
            e_shot = 0.0
        else:
            e_shot = obs_product * d_shot / s_product

        numerator_sum += e_shot * count
        denominator_sum += d_shot * count
        total_shots += count

    mean_num = numerator_sum / total_shots
    mean_den = denominator_sum / total_shots

    if abs(mean_den) < 1e-15:
        return float("nan"), mean_num, mean_den

    return mean_num / mean_den, mean_num, mean_den


# ---------------------------------------------------------------------------
# Readout error mitigation
# ---------------------------------------------------------------------------

def mitigate_readout_counts(
    counts: dict[str, int | float],
    p_readout: float,
) -> dict[str, float]:
    """Correct measurement counts for symmetric readout errors.

    Applies per-qubit inverse confusion matrix to undo symmetric bit-flip
    readout noise.  For symmetric error probability *p*, each qubit's 2x2
    confusion matrix and its inverse are::

        C   = [[1-p,  p ],      C^{-1} = (1/(1-2p)) * [[1-p, -p ],
               [ p, 1-p]]                               [-p,  1-p]]

    The full N-qubit correction is the tensor product
    ``C_1^{-1} x ... x C_N^{-1}``, applied one qubit at a time by pairing
    bitstrings that differ only at that position.

    Returns a quasi-probability dict (values may be negative) suitable for
    feeding into ``compute_mitigated_expval()`` or
    ``compute_unmitigated_expval()``.

    Args:
        counts: Qiskit counts dictionary (or quasi-probability dict from a
            previous correction step).
        p_readout: The symmetric readout error probability that was used
            during simulation.  Must be in [0, 0.5).

    Returns:
        A dictionary mapping bitstrings to corrected quasi-probabilities.
    """
    if p_readout <= 0:
        return dict(counts)

    gamma = 1.0 / (1 - 2 * p_readout)
    a = gamma * (1 - p_readout)   # diagonal element of C^{-1}
    b = gamma * (-p_readout)       # off-diagonal element of C^{-1}

    # Determine bit width from first key
    first_key = next(iter(counts))
    n_bits = len(first_key)

    corrected: dict[str, float] = dict(counts)

    for bit_pos in range(n_bits):
        new_corrected: dict[str, float] = {}
        visited: set[str] = set()

        for bs in corrected:
            if bs in visited:
                continue

            # Build partner bitstring (flipped at bit_pos)
            bs_list = list(bs)
            bs_list[bit_pos] = '1' if bs_list[bit_pos] == '0' else '0'
            partner = ''.join(bs_list)
            visited.add(bs)
            visited.add(partner)

            val_bs = corrected.get(bs, 0.0)
            val_partner = corrected.get(partner, 0.0)

            new_corrected[bs] = a * val_bs + b * val_partner
            new_corrected[partner] = b * val_bs + a * val_partner

        corrected = new_corrected

    return corrected


# ---------------------------------------------------------------------------
# Subsystem purity estimation
# ---------------------------------------------------------------------------

def compute_subsystem_purity(
    counts: dict[str, int],
    subsystem: list[int],
    n_qubits: int,
) -> float:
    """Estimate Tr(ρ_A²) from M=2 counts with B^(2) applied only to subsystem A.

    When the B^(2) beamsplitter is applied only to qubit pairs (j, N+j) for
    j ∈ A, the per-shot SWAP eigenvalue product restricted to A gives an
    unbiased estimator of the subsystem purity::

        d_shot = prod_{j in A} s_j

    where ``s_j = -1`` if ``(b1[j]==0 and b2[j]==1)`` else ``+1``.

    Args:
        counts: Qiskit counts dict from a 2N-qubit circuit.
        subsystem: Qubit indices defining subsystem A.
        n_qubits: Number of logical qubits N.

    Returns:
        Estimated Tr(ρ_A²).
    """
    denominator_sum = 0.0
    total_shots = 0

    for bitstring, count in counts.items():
        n = n_qubits
        d_shot = 1
        for j in subsystem:
            b1_j = int(bitstring[2 * n - 1 - j])
            b2_j = int(bitstring[n - 1 - j])
            s_j = -1 if (b1_j == 0 and b2_j == 1) else 1
            d_shot *= s_j

        denominator_sum += d_shot * count
        total_shots += count

    return denominator_sum / total_shots


def compute_renyi_entropy(
    counts: dict[str, int],
    subsystem: list[int],
    n_qubits: int,
) -> tuple[float, float]:
    """Compute S_2(A) = -log₂(Tr(ρ_A²)) from M=2 counts.

    Args:
        counts: Qiskit counts dict from a 2N-qubit circuit.
        subsystem: Qubit indices defining subsystem A.
        n_qubits: Number of logical qubits N.

    Returns:
        ``(s2, tr_rho_a_sq)`` tuple.
    """
    tr_rho_a_sq = compute_subsystem_purity(counts, subsystem, n_qubits)
    if tr_rho_a_sq <= 0:
        return float("nan"), tr_rho_a_sq
    s2 = -np.log2(tr_rho_a_sq)
    return s2, tr_rho_a_sq
