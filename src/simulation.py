"""M=2 Virtual Distillation simulation for quantum error mitigation.

This module provides the high-level simulation workflow:
- Target circuit generation (near-Clifford and local-shallow ansatze)
- M=2 protocol circuit construction (two copies + B^(2) measurement layer)
- Backend-aware transpilation and execution via Qiskit Aer
- Unified ``run_simulation`` entry point returning both unmitigated
  and mitigated expectation values

All noisy simulations use **Pauli-only noise models** processed through
Monte Carlo trajectory sampling, avoiding density-matrix memory walls
even at large qubit counts (2N = 40 for the M=2 circuit at N = 20).

References:
    Huggins et al., "Virtual Distillation for Quantum Error Mitigation" (2021)
    Koczor, "Error Suppression by Derangement" (2020)
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.utils import (
    apply_b2_gate,
    compute_mitigated_expval,
    compute_unmitigated_expval,
)


# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------

def generate_target_circuit(
    n_qubits: int,
    circuit_type: str,
    *,
    max_t_gates: int = 4,
    n_layers: int = 3,
    seed: int | None = None,
) -> QuantumCircuit:
    """Generate an N-qubit target circuit tailored for a simulation backend.

    Two circuit families are supported:

    ``'near_clifford'``
        Deep Clifford backbone (random H/S gates and nearest-neighbour CNOT
        ladders) with a *bounded* number of T gates sprinkled on randomly
        chosen qubits.  Designed for the ``extended_stabilizer`` method whose
        runtime scales exponentially only in the non-Clifford gate count.

    ``'local_shallow'``
        Hardware-efficient ansatz with Ry/Rz rotation layers interleaved with
        nearest-neighbour CX entangling layers.  Shallow depth (controlled by
        ``n_layers``) and 1-D connectivity make it ideal for the
        ``matrix_product_state`` method, whose efficiency relies on bounded
        entanglement / bond dimension.

    Args:
        n_qubits: Number of qubits *N*.
        circuit_type: ``'near_clifford'`` or ``'local_shallow'``.
        max_t_gates: Maximum T gates for near_clifford circuits (default 4).
        n_layers: Number of rotation + entanglement layers (default 3).
        seed: Random seed for reproducibility.

    Returns:
        A :class:`QuantumCircuit` with *N* qubits and no classical bits
        (measurement is added later).
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name=f"target_{circuit_type}")

    if circuit_type == "near_clifford":
        # ---- Deep Clifford backbone ----
        for layer_idx in range(n_layers):
            # Random single-qubit Clifford gates
            for q in range(n_qubits):
                gate = rng.choice(["h", "s", "id"])
                if gate == "h":
                    qc.h(q)
                elif gate == "s":
                    qc.s(q)
                # "id" -> no gate (identity)

            # Nearest-neighbour CNOT ladder (alternating even/odd pairs)
            start = layer_idx % 2
            for q in range(start, n_qubits - 1, 2):
                qc.cx(q, q + 1)

        # ---- Bounded T-gate injection ----
        n_t = min(max_t_gates, n_qubits)
        t_qubits = rng.choice(n_qubits, size=n_t, replace=False)
        for q in t_qubits:
            qc.t(int(q))

    elif circuit_type == "local_shallow":
        # ---- Hardware-efficient ansatz ----
        for layer_idx in range(n_layers):
            # Parameterised rotation layer
            for q in range(n_qubits):
                theta_y = rng.uniform(0, 2 * np.pi)
                theta_z = rng.uniform(0, 2 * np.pi)
                qc.ry(theta_y, q)
                qc.rz(theta_z, q)

            # 1-D nearest-neighbour entangling layer
            start = layer_idx % 2
            for q in range(start, n_qubits - 1, 2):
                qc.cx(q, q + 1)

    else:
        raise ValueError(
            f"Unknown circuit_type: '{circuit_type}'. "
            "Supported: 'near_clifford', 'local_shallow'"
        )

    return qc


# ---------------------------------------------------------------------------
# M=2 protocol circuit construction
# ---------------------------------------------------------------------------

def build_single_copy_circuit(target_circuit: QuantumCircuit) -> QuantumCircuit:
    """Wrap a target circuit with computational-basis measurement.

    Used for computing the **unmitigated** expectation value <Z_k> as a
    comparison baseline.

    Args:
        target_circuit: N-qubit state-preparation circuit (no measurements).

    Returns:
        An N-qubit circuit with measurements on all qubits.
    """
    n = target_circuit.num_qubits
    qc = QuantumCircuit(n, n, name="single_copy")
    qc.compose(target_circuit, qubits=list(range(n)), inplace=True)
    qc.measure(list(range(n)), list(range(n)))
    return qc


def build_m2_circuit(target_circuit: QuantumCircuit) -> QuantumCircuit:
    """Construct the 2N-qubit M=2 Virtual Distillation circuit.

    Layout::

        Qubits  0  .. N-1   : copy 1  (target circuit applied)
        Qubits  N  .. 2N-1  : copy 2  (target circuit applied)

    After both copies are independently prepared, a layer of B^(2)
    beamsplitter gates is applied to every pair ``(i, N + i)`` for
    ``i in 0 .. N-1``.  This jointly diagonalises the per-pair SWAP
    operators so that ``Tr(O rho^2) / Tr(rho^2)`` can be estimated from
    computational-basis measurements via classical post-processing
    (see :func:`~src.utils.compute_mitigated_expval`).

    ``compose()`` is used instead of ``append(to_instruction())`` to flatten
    the sub-circuit gates.  This is essential because Qiskit Aer's noise
    model attaches errors to individual gates -- opaque instructions would
    hide the gates and skip noise injection.

    Args:
        target_circuit: N-qubit state-preparation circuit (no measurements).

    Returns:
        A 2N-qubit circuit with measurements on all qubits.
    """
    n = target_circuit.num_qubits
    qc = QuantumCircuit(2 * n, 2 * n, name="m2_protocol")

    # Prepare copy 1 on qubits 0 .. N-1
    qc.compose(target_circuit, qubits=list(range(n)), inplace=True)

    # Prepare copy 2 on qubits N .. 2N-1
    qc.compose(target_circuit, qubits=list(range(n, 2 * n)), inplace=True)

    qc.barrier()  # cosmetic separator

    # B^(2) measurement layer -- one beamsplitter per qubit pair
    for i in range(n):
        apply_b2_gate(qc, i, n + i)

    qc.barrier()

    # Measure all 2N qubits in the computational basis
    qc.measure(list(range(2 * n)), list(range(2 * n)))
    return qc


# ---------------------------------------------------------------------------
# Transpilation
# ---------------------------------------------------------------------------

def transpile_for_method(
    circuit: QuantumCircuit,
    method_name: str,
) -> QuantumCircuit:
    """Transpile a circuit to basis gates compatible with *method_name*.

    ======================== ==================================================
    Method                   Basis gates
    ======================== ==================================================
    ``statevector``          No transpilation (supports all gates natively)
    ``extended_stabilizer``  Clifford + T: cx, h, s, sdg, t, tdg, x, y, z
    ``matrix_product_state`` Standard + continuous: cx, h, s, sdg, rx, ry, rz,
                             x, y, z, sx, sxdg, u1, u2, u3
    ======================== ==================================================

    Args:
        circuit: The quantum circuit to transpile.
        method_name: Aer simulation method name.

    Returns:
        The transpiled circuit.
    """
    if method_name == "statevector":
        return circuit  # statevector handles arbitrary unitaries

    if method_name == "extended_stabilizer":
        basis = ["cx", "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"]
    elif method_name == "matrix_product_state":
        basis = [
            "cx", "id", "x", "y", "z", "h", "s", "sdg",
            "sx", "sxdg", "rx", "ry", "rz", "u1", "u2", "u3",
        ]
    else:
        raise ValueError(f"Unknown method: '{method_name}'")

    return transpile(circuit, basis_gates=basis, optimization_level=2)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    target_circuit: QuantumCircuit,
    noise_model: NoiseModel | None,
    method_name: str,
    shots: int = 10_000,
    observable_qubit: int = 0,
    seed: int | None = None,
) -> dict:
    """Execute the M=2 Virtual Distillation workflow end-to-end.

    This function:

    1. Builds a **single-copy** circuit from *target_circuit*, runs it on
       ``AerSimulator(method=method_name)`` with the given *noise_model*, and
       computes the **unmitigated** ``<Z_k>``.
    2. Builds the **M=2 (2N-qubit)** protocol circuit, runs it on the same
       backend, and post-processes the counts to obtain the **mitigated**
       ``<Z_k> = Tr(Z_k rho^2) / Tr(rho^2)``.

    How Pauli noise is handled via Monte Carlo trajectories
    -------------------------------------------------------
    When a Pauli-only :class:`NoiseModel` is supplied to ``AerSimulator``
    with ``method='extended_stabilizer'`` or ``method='matrix_product_state'``,
    each shot is simulated as a **pure-state trajectory**:

    * The simulator maintains a compact pure-state representation --
      a stabiliser tableau (extended stabiliser) or a tensor train (MPS).
    * After every noisy gate, **one** Pauli error is sampled from the
      channel's probability distribution (e.g. X with prob p/3, Y with
      p/3, Z with p/3 for depolarising noise at rate *p*).
    * The sampled Pauli is applied as a Clifford update (stabiliser) or a
      trivial local tensor update (MPS), preserving the efficient
      representation.
    * Each shot independently samples a different error trajectory,
      collectively Monte-Carlo-averaging over the noise ensemble.

    This avoids constructing the full 2^{2N} x 2^{2N} density matrix,
    keeping memory at O(N * 2^t) for extended_stabiliser (t = non-Clifford
    gate count) or O(N * chi^2) for MPS (chi = bond dimension).

    Args:
        target_circuit: N-qubit state-preparation circuit.
        noise_model: A Pauli-based :class:`NoiseModel`, or ``None`` for
            noiseless simulation.
        method_name: ``'statevector'``, ``'extended_stabilizer'``, or
            ``'matrix_product_state'``.
        shots: Number of measurement shots.
        observable_qubit: Logical qubit for the Z expectation value.
        seed: Random seed (controls both trajectory sampling and measurement).

    Returns:
        A dictionary::

            {
                'method':          str,
                'n_qubits':        int,    # logical qubit count N
                'unmitigated_z':   float,
                'mitigated_z':     float,
                'tr_rho_sq':       float,  # estimated Tr(rho^2)
                'shots':           int,
            }
    """
    n_qubits = target_circuit.num_qubits

    # ---- configure backend ------------------------------------------------
    backend_kwargs: dict = {"method": method_name}
    if noise_model is not None:
        backend_kwargs["noise_model"] = noise_model
    backend = AerSimulator(**backend_kwargs)

    # ---- unmitigated (single-copy) ----------------------------------------
    single_circ = build_single_copy_circuit(target_circuit)
    single_circ = transpile_for_method(single_circ, method_name)

    job_single = backend.run(single_circ, shots=shots, seed_simulator=seed)
    counts_single = job_single.result().get_counts()
    unmitigated_z = compute_unmitigated_expval(counts_single, observable_qubit)

    # ---- mitigated (M=2 protocol) -----------------------------------------
    m2_circ = build_m2_circuit(target_circuit)
    m2_circ = transpile_for_method(m2_circ, method_name)

    job_m2 = backend.run(m2_circ, shots=shots, seed_simulator=seed)
    counts_m2 = job_m2.result().get_counts()
    mitigated_z, _mean_num, mean_den = compute_mitigated_expval(
        counts_m2, observable_qubit, n_qubits
    )

    return {
        "method": method_name,
        "n_qubits": n_qubits,
        "unmitigated_z": unmitigated_z,
        "mitigated_z": mitigated_z,
        "tr_rho_sq": mean_den,
        "shots": shots,
    }
