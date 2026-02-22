"""Subsystem 2nd Rényi entanglement entropy via Virtual Distillation.

Implements Phase 3 of the analog-digital VD research plan:

  Task 3.1  1D chain state preparation at the quantum phase transition
  Task 3.2  Subsystem VD measurement (partial B^(2) layer)
  Task 3.3  Purity extraction and S_2(A) computation
  Task 3.4  CFT verification: S_2(A) ~ (c/3) log|A| scaling

References:
    Calabrese & Cardy, "Entanglement entropy and quantum field theory" (2004)
    Islam et al., "Measuring entanglement entropy in a quantum many-body
        system" (Nature, 2015)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.analog import (
    AnalogEvolutionResult,
    build_rydberg_sequence,
    create_1d_register,
    simulate_pulser,
)
from src.simulation import build_subsystem_m2_circuit_from_statevector
from src.utils import compute_renyi_entropy


# ---------------------------------------------------------------------------
# 1D chain state preparation  (Task 3.1)
# ---------------------------------------------------------------------------

def prepare_1d_chain_state(
    n_atoms: int,
    spacing_um: float = 7.0,
    omega_max: float = 2.0,
    delta_initial: float = -5.0,
    delta_final: float = 5.0,
    duration_ns: int = 2000,
    noise_model=None,
) -> AnalogEvolutionResult:
    """Prepare a 1D Rydberg chain state via adiabatic detuning sweep.

    The sweep crosses the Z2 ordered phase boundary.  Near the critical
    point the entanglement structure exhibits logarithmic scaling
    characteristic of a CFT (Ising universality class, c = 1/2).

    Args:
        n_atoms: Number of atoms in the chain.
        spacing_um: Inter-atom spacing in um.
        omega_max: Maximum Rabi frequency in rad/us.
        delta_initial: Initial detuning in rad/us.
        delta_final: Final detuning in rad/us.
        duration_ns: Sweep duration in ns.
        noise_model: Optional Pulser NoiseModel.

    Returns:
        AnalogEvolutionResult with statevector in Qiskit convention.
    """
    register = create_1d_register(n_atoms, spacing_um)
    seq = build_rydberg_sequence(
        register,
        protocol="sweep",
        omega_max=omega_max,
        delta_initial=delta_initial,
        delta_final=delta_final,
        duration_ns=duration_ns,
    )
    result = simulate_pulser(seq, noise_model=noise_model)
    result.metadata.update({
        "n_atoms": n_atoms,
        "spacing_um": spacing_um,
        "omega_max": omega_max,
        "delta_final": delta_final,
        "phase": "1d_chain",
    })
    return result


# ---------------------------------------------------------------------------
# Subsystem purity evaluation  (Task 3.2 + 3.3)
# ---------------------------------------------------------------------------

def evaluate_subsystem_purity(
    analog_result: AnalogEvolutionResult,
    subsystem: list[int],
    noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Evaluate Tr(ρ_A²) and S_2(A) for a subsystem via M=2 VD.

    Builds a subsystem M=2 circuit (B^(2) only on subsystem pairs),
    runs on AerSimulator, and post-processes to extract the subsystem
    purity and 2nd Rényi entropy.

    Args:
        analog_result: AnalogEvolutionResult (must have statevector).
        subsystem: Qubit indices defining subsystem A.
        noise_model: Qiskit NoiseModel for digital layer.
        shots: Measurement shots.
        seed: Random seed.

    Returns:
        Dict with keys: ``'subsystem'``, ``'n_qubits'``,
        ``'subsystem_size'``, ``'tr_rho_a_sq'``, ``'s2_entropy'``,
        ``'shots'``, ``'source'``.
    """
    if analog_result.statevector is None:
        raise ValueError("evaluate_subsystem_purity requires a pure-state result.")

    sv = analog_result.statevector
    n_qubits = analog_result.n_qubits
    sv = sv / np.linalg.norm(sv)

    backend_kwargs: dict = {"method": "statevector"}
    if noise_model is not None:
        backend_kwargs["noise_model"] = noise_model
    backend = AerSimulator(**backend_kwargs)

    m2_circ = build_subsystem_m2_circuit_from_statevector(sv, n_qubits, subsystem)
    job = backend.run(m2_circ, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()

    s2, tr_rho_a_sq = compute_renyi_entropy(counts, subsystem, n_qubits)

    return {
        "subsystem": list(subsystem),
        "n_qubits": n_qubits,
        "subsystem_size": len(subsystem),
        "tr_rho_a_sq": tr_rho_a_sq,
        "s2_entropy": s2,
        "shots": shots,
        "source": "entanglement",
    }


# ---------------------------------------------------------------------------
# Subsystem size sweep  (Task 3.4)
# ---------------------------------------------------------------------------

@dataclass
class EntanglementEntropyResult:
    """Results from a subsystem entropy sweep.

    Attributes:
        n_qubits: Total chain length L.
        subsystem_sizes: Array of subsystem sizes swept.
        tr_rho_a_sq: Array of Tr(ρ_A²) at each size.
        s2_entropy: Array of S_2(A) at each size.
        cft_prediction: CFT-predicted S_2 values (after fitting).
        fitted_central_charge: Fitted central charge c.
        metadata: Additional info.
    """

    n_qubits: int
    subsystem_sizes: np.ndarray
    tr_rho_a_sq: np.ndarray
    s2_entropy: np.ndarray
    cft_prediction: np.ndarray | None = None
    fitted_central_charge: float | None = None
    metadata: dict = field(default_factory=dict)


def sweep_subsystem_size(
    analog_result: AnalogEvolutionResult,
    max_subsystem_size: int | None = None,
    noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
) -> EntanglementEntropyResult:
    """Sweep contiguous subsystem sizes and compute S_2(A) for each.

    For a chain of length L, sweeps subsystems A = {0, 1, ..., l-1}
    for l in [1, max_subsystem_size].  Default max is L // 2.

    Args:
        analog_result: AnalogEvolutionResult with statevector.
        max_subsystem_size: Maximum subsystem size (default L // 2).
        noise_model: Qiskit NoiseModel for digital layer.
        shots: Measurement shots per subsystem size.
        seed: Random seed.

    Returns:
        EntanglementEntropyResult with arrays of S_2 values.
    """
    n = analog_result.n_qubits
    max_l = max_subsystem_size if max_subsystem_size is not None else n // 2
    sizes = np.arange(1, max_l + 1)

    tr_arr = np.zeros(len(sizes))
    s2_arr = np.zeros(len(sizes))

    rng = np.random.default_rng(seed)

    for idx, l in enumerate(sizes):
        subsystem = list(range(l))
        level_seed = int(rng.integers(0, 2**31))
        result = evaluate_subsystem_purity(
            analog_result, subsystem,
            noise_model=noise_model, shots=shots, seed=level_seed,
        )
        tr_arr[idx] = result["tr_rho_a_sq"]
        s2_arr[idx] = result["s2_entropy"]

    return EntanglementEntropyResult(
        n_qubits=n,
        subsystem_sizes=sizes,
        tr_rho_a_sq=tr_arr,
        s2_entropy=s2_arr,
        metadata=dict(analog_result.metadata),
    )


# ---------------------------------------------------------------------------
# CFT scaling fit
# ---------------------------------------------------------------------------

def fit_cft_scaling(
    result: EntanglementEntropyResult,
    boundary: str = "open",
) -> EntanglementEntropyResult:
    """Fit CFT logarithmic scaling to S_2(A) data.

    For open boundary conditions::

        S_2(l) = (c/6) * log((2L/π) * sin(πl/L)) + const

    For periodic boundary conditions::

        S_2(l) = (c/3) * log((L/π) * sin(πl/L)) + const

    Args:
        result: EntanglementEntropyResult from sweep_subsystem_size.
        boundary: ``'open'`` or ``'periodic'``.

    Returns:
        Updated result with ``cft_prediction`` and
        ``fitted_central_charge`` populated.
    """
    L = result.n_qubits
    sizes = result.subsystem_sizes
    s2 = result.s2_entropy

    # Chord-length variable
    if boundary == "open":
        x = np.log((2 * L / np.pi) * np.sin(np.pi * sizes / L))
        prefactor = 6.0  # c = slope * 6
    else:
        x = np.log((L / np.pi) * np.sin(np.pi * sizes / L))
        prefactor = 3.0  # c = slope * 3

    # Filter out NaN entries
    valid = np.isfinite(s2) & np.isfinite(x)
    if valid.sum() < 2:
        return result

    coeffs = np.polyfit(x[valid], s2[valid], 1)
    slope, intercept = coeffs

    result.fitted_central_charge = slope * prefactor
    result.cft_prediction = slope * x + intercept
    return result


# ---------------------------------------------------------------------------
# End-to-end convenience pipeline
# ---------------------------------------------------------------------------

def run_entanglement_entropy(
    n_atoms: int = 8,
    spacing_um: float = 7.0,
    omega_max: float = 2.0,
    delta_initial: float = -5.0,
    delta_final: float = 5.0,
    duration_ns: int = 2000,
    noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
) -> EntanglementEntropyResult:
    """Run the full entanglement entropy pipeline.

    1. Prepare 1D chain state via adiabatic sweep.
    2. Sweep subsystem sizes.
    3. Fit CFT scaling.

    Args:
        n_atoms: Number of atoms in the chain.
        spacing_um: Inter-atom spacing in um.
        omega_max: Maximum Rabi frequency in rad/us.
        delta_initial: Initial detuning in rad/us.
        delta_final: Final detuning in rad/us.
        duration_ns: Sweep duration in ns.
        noise_model: Qiskit NoiseModel for digital layer.
        shots: Measurement shots per subsystem size.
        seed: Random seed.

    Returns:
        EntanglementEntropyResult with fitted central charge.
    """
    analog_result = prepare_1d_chain_state(
        n_atoms, spacing_um, omega_max,
        delta_initial, delta_final, duration_ns,
    )
    result = sweep_subsystem_size(
        analog_result, noise_model=noise_model, shots=shots, seed=seed,
    )
    return fit_cft_scaling(result)


def print_entropy_results(result: EntanglementEntropyResult) -> None:
    """Pretty-print an EntanglementEntropyResult table."""
    L = result.n_qubits
    print(
        f"\n{'=' * 60}\n"
        f"Subsystem Entanglement Entropy:  {L}-atom 1D chain\n"
        f"{'=' * 60}"
    )
    header = f"{'|A|':>5s}  {'Tr(rho_A^2)':>12s}  {'S_2(A)':>10s}"
    if result.cft_prediction is not None:
        header += f"  {'CFT pred':>10s}"
    print(header)
    print("-" * 60)

    for i, l in enumerate(result.subsystem_sizes):
        line = f"{l:5d}  {result.tr_rho_a_sq[i]:12.4f}  {result.s2_entropy[i]:10.4f}"
        if result.cft_prediction is not None:
            line += f"  {result.cft_prediction[i]:10.4f}"
        print(line)

    if result.fitted_central_charge is not None:
        print(f"\nFitted central charge c = {result.fitted_central_charge:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Phase 3 entanglement entropy pipeline."""
    print("Phase 3: Subsystem 2nd Renyi Entanglement Entropy via VD")
    print("Preparing 1D chain (8 atoms)...")

    result = run_entanglement_entropy(n_atoms=8, shots=10_000, seed=42)
    print_entropy_results(result)


if __name__ == "__main__":
    main()
