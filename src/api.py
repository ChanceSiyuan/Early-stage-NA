"""User-friendly public API for the Early-stage-NA VD framework.

Provides a small set of high-level functions that cover the four main
research workflows with sensible defaults.  Each function wraps the
lower-level per-module pipelines so that a new user can get results
in 2-3 lines of code.

Example
-------
>>> from src.api import mitigate, topological_order, entanglement_entropy, kibble_zurek
>>> result = mitigate(n_qubits=5, noise=(0.01, 0.02))
>>> topo   = topological_order()
>>> ent    = entanglement_entropy(n_atoms=8)
>>> kz     = kibble_zurek(n_atoms=6)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.analog import AnalogEvolutionResult
from src.entanglement import (
    EntanglementEntropyResult,
    run_entanglement_entropy,
)
from src.kibble_zurek import KibbleZurekResult, run_kibble_zurek
from src.simulation import generate_target_circuit, run_simulation
from src.topological import (
    TopologicalBenchmarkResult,
    run_topological_benchmark,
    run_topological_vd,
)
from src.utils import get_pauli_noise_model


# ---------------------------------------------------------------------------
# 1. Gate-based VD mitigation
# ---------------------------------------------------------------------------

def mitigate(
    n_qubits: int = 5,
    circuit_type: Literal["near_clifford", "local_shallow"] = "local_shallow",
    noise: tuple[float, float] | None = (0.01, 0.02),
    backend: Literal["statevector", "matrix_product_state"] = "matrix_product_state",
    observable_qubit: int = 0,
    shots: int = 10_000,
    seed: int | None = 42,
) -> dict:
    """Run M=2 Virtual Distillation on a gate-based circuit.

    Generates a target circuit, applies Pauli noise, and compares
    unmitigated vs VD-mitigated ``<Z_k>``.

    Args:
        n_qubits: Number of logical qubits.
        circuit_type: ``'near_clifford'`` or ``'local_shallow'``.
        noise: ``(p_1q, p_2q)`` error probabilities, or ``None`` for noiseless.
        backend: Aer simulation method.
        observable_qubit: Which qubit's ``<Z>`` to estimate.
        shots: Measurement shots.
        seed: Random seed.

    Returns:
        Dict with ``'unmitigated_z'``, ``'mitigated_z'``, ``'tr_rho_sq'``,
        ``'n_qubits'``, ``'method'``, ``'shots'``.

    Example::

        >>> from src.api import mitigate
        >>> r = mitigate(n_qubits=5, noise=(0.01, 0.02))
        >>> print(f"Unmitigated: {r['unmitigated_z']:+.4f}")
        >>> print(f"Mitigated:   {r['mitigated_z']:+.4f}")
    """
    circuit = generate_target_circuit(n_qubits, circuit_type, seed=seed)
    noise_model = None
    if noise is not None:
        noise_model = get_pauli_noise_model(noise[0], noise[1])
    return run_simulation(
        circuit, noise_model, backend,
        shots=shots, observable_qubit=observable_qubit, seed=seed,
    )


# ---------------------------------------------------------------------------
# 2. Topological order detection
# ---------------------------------------------------------------------------

def topological_order(
    lattice_size: tuple[int, int] = (2, 2),
    plaquette: int = 0,
    noise_sweep: list[float] | None = None,
    shots: int = 10_000,
    seed: int = 42,
) -> TopologicalBenchmarkResult | dict:
    """Detect topological order on a Kagome lattice via VD.

    With ``noise_sweep``, runs a benchmark across multiple noise levels
    and returns a :class:`TopologicalBenchmarkResult`.  Without it,
    runs a single noiseless evaluation and returns a dict.

    Args:
        lattice_size: ``(n_rows, n_cols)`` of Kagome unit cells.
        plaquette: Which hexagonal plaquette to evaluate.
        noise_sweep: List of ``p_2q`` values to sweep.  ``None`` for
            a single noiseless run.
        shots: Measurement shots.
        seed: Random seed.

    Returns:
        :class:`TopologicalBenchmarkResult` if sweeping, else dict.

    Example::

        >>> from src.api import topological_order
        >>> r = topological_order()
        >>> print(f"<W_hex> = {r['mitigated_w']:+.4f}")
    """
    n_rows, n_cols = lattice_size
    if noise_sweep is not None:
        return run_topological_benchmark(
            n_rows=n_rows, n_cols=n_cols,
            plaquette_index=plaquette,
            noise_levels=np.array(noise_sweep),
            shots=shots, seed=seed,
        )
    return run_topological_vd(
        n_rows=n_rows, n_cols=n_cols,
        plaquette_index=plaquette,
        shots=shots, seed=seed,
    )


# ---------------------------------------------------------------------------
# 3. Entanglement entropy
# ---------------------------------------------------------------------------

def entanglement_entropy(
    n_atoms: int = 8,
    shots: int = 10_000,
    seed: int | None = 42,
) -> EntanglementEntropyResult:
    """Measure subsystem Renyi-2 entanglement entropy via VD.

    Prepares a 1D Rydberg chain near the Z2 phase transition, sweeps
    contiguous subsystem sizes, and fits CFT logarithmic scaling to
    extract the central charge.

    Args:
        n_atoms: Chain length.
        shots: Measurement shots per subsystem size.
        seed: Random seed.

    Returns:
        :class:`EntanglementEntropyResult` with ``s2_entropy``,
        ``fitted_central_charge``, etc.

    Example::

        >>> from src.api import entanglement_entropy
        >>> r = entanglement_entropy(n_atoms=8)
        >>> print(f"Central charge c = {r.fitted_central_charge:.3f}")
    """
    return run_entanglement_entropy(n_atoms=n_atoms, shots=shots, seed=seed)


# ---------------------------------------------------------------------------
# 4. Kibble-Zurek scaling
# ---------------------------------------------------------------------------

def kibble_zurek(
    n_atoms: int = 8,
    quench_durations: list[int] | None = None,
    shots: int = 10_000,
    seed: int | None = 42,
) -> KibbleZurekResult:
    """Measure Kibble-Zurek scaling of correlation length via VD.

    Sweeps quench rates on a 1D Rydberg chain, computes spatial
    correlation functions C(r), extracts correlation lengths, and
    fits the KZ scaling exponent.

    Args:
        n_atoms: Chain length.
        quench_durations: Sweep durations in ns.  Defaults to
            ``[500, 1000, 1500, 2000, 3000, 4000]``.
        shots: Measurement shots per correlation evaluation.
        seed: Random seed.

    Returns:
        :class:`KibbleZurekResult` with ``kz_exponent_mitigated``,
        ``kz_prediction``, etc.

    Example::

        >>> from src.api import kibble_zurek
        >>> r = kibble_zurek(n_atoms=6, quench_durations=[500, 1000, 2000])
        >>> print(f"KZ exponent = {r.kz_exponent_mitigated:.3f}")
    """
    return run_kibble_zurek(
        n_atoms=n_atoms,
        quench_durations_ns=quench_durations,
        shots=shots, seed=seed,
    )
