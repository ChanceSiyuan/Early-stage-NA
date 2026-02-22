"""Topological order detection via analog-digital Virtual Distillation.

Provides the Kagome lattice geometry, Z2 Quantum Spin Liquid (QSL)
adiabatic preparation via Pulser, hexagonal plaquette string operators,
and benchmark sweeps comparing unmitigated vs VD-mitigated string operator
expectation values across noise levels.

This module implements Phase 2 of the analog-digital VD research plan:

  Task 2.1  Adiabatic Z2 QSL preparation on a Kagome lattice
  Task 2.2  Closed-loop string operators W_l = prod_{i in l} Z_i
  Task 2.3  M=2 digital VD for multi-qubit string operators
  Task 2.4  Benchmark sweep across noise levels

References:
    Semeghini et al., "Probing topological spin liquids on a programmable
        quantum simulator" (2021), Science 374, 1242-1247
    Samajdar et al., "Quantum Phases of Rydberg Atoms on a Kagome Lattice"
        (2021), PNAS 118, e2015785118
    Verresen et al., "Prediction of toric code topological order from
        Rydberg blockade" (2021), arXiv:2112.00020
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pulser
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.analog import (
    AnalogEvolutionResult,
    build_rydberg_sequence,
    simulate_pulser,
)
from src.neutral_atom_noise import NeutralAtomNoiseParams, compose_noise_models
from src.simulation import (
    build_m2_circuit_from_statevector,
    build_single_copy_circuit_from_statevector,
)
from src.utils import (
    compute_mitigated_string_expval,
    compute_unmitigated_string_expval,
    get_pauli_noise_model,
)


# ---------------------------------------------------------------------------
# Kagome lattice geometry
# ---------------------------------------------------------------------------

@dataclass
class KagomeLattice:
    """Kagome lattice geometry and atom-to-qubit mapping.

    Attributes:
        n_rows: Number of unit cell rows.
        n_cols: Number of unit cell columns.
        spacing_um: Nearest-neighbour distance in micrometres.
        positions: List of (x, y) coordinates in um, one per atom.
        atom_labels: List of ``(row, col, sublattice)`` tuples identifying
            each atom.  Sublattice is ``'A'``, ``'B'``, or ``'C'``.
        n_atoms: Total number of atoms (``3 * n_rows * n_cols``).
        plaquettes: List of hexagonal plaquettes, each a list of 6 atom
            indices ordered around the hexagon.
    """

    n_rows: int
    n_cols: int
    spacing_um: float
    positions: list[tuple[float, float]]
    atom_labels: list[tuple[int, int, str]]
    n_atoms: int
    plaquettes: list[list[int]]


def create_kagome_lattice(
    n_rows: int = 2,
    n_cols: int = 2,
    spacing_um: float = 5.5,
) -> KagomeLattice:
    """Create a Kagome lattice with atom positions and plaquette enumeration.

    The Kagome lattice has 3 atoms per unit cell arranged on the bonds of a
    triangular lattice.  The lattice vectors are::

        a1 = (2a, 0)
        a2 = (a, a * sqrt(3))

    where ``a = spacing_um`` (nearest-neighbour distance).

    Sublattice positions within unit cell ``(i, j)``::

        A = i * a1 + j * a2
        B = i * a1 + j * a2 + (a/2, a * sqrt(3) / 2)
        C = i * a1 + j * a2 + (a, 0)

    The three sublattice sites form an equilateral triangle of side ``a``.

    Hexagonal plaquettes are identified by tracing around hexagons in the
    Kagome lattice.  Each hexagon involves atoms from 3 neighbouring unit
    cells.  For an ``n_rows x n_cols`` grid, plaquettes centred at ``(i, j)``
    exist when ``i + 1 < n_cols`` and ``j + 1 < n_rows``.

    Examples:
        2x2 -> 12 atoms, 1 plaquette.
        3x2 -> 18 atoms, 2 plaquettes.
        3x3 -> 27 atoms, 4 plaquettes.

    Args:
        n_rows: Number of unit cell rows.
        n_cols: Number of unit cell columns.
        spacing_um: Nearest-neighbour distance in micrometres.  Default
            5.5 um is chosen so that R_b/a ~ 1.8 at typical Omega values,
            placing the system in the Rydberg blockade regime appropriate
            for the Z2 spin liquid phase.

    Returns:
        A :class:`KagomeLattice` with positions, labels, and plaquette indices.
    """
    a = spacing_um
    sqrt3 = math.sqrt(3)

    # Lattice vectors
    a1 = np.array([2 * a, 0.0])
    a2 = np.array([a, a * sqrt3])

    # Sublattice offsets within each unit cell
    offsets = {
        "A": np.array([0.0, 0.0]),
        "B": np.array([a / 2, a * sqrt3 / 2]),
        "C": np.array([a, 0.0]),
    }
    sublattice_names = ["A", "B", "C"]

    positions: list[tuple[float, float]] = []
    labels: list[tuple[int, int, str]] = []

    # Atom index = 3 * (i * n_cols + j) + sublattice_index
    # where sublattice_index: A=0, B=1, C=2
    for i in range(n_rows):
        for j in range(n_cols):
            origin = i * a2 + j * a1  # row = a2 direction, col = a1 direction
            for s_idx, s_name in enumerate(sublattice_names):
                pos = origin + offsets[s_name]
                positions.append((float(pos[0]), float(pos[1])))
                labels.append((i, j, s_name))

    n_atoms = 3 * n_rows * n_cols

    def _atom_index(row: int, col: int, sublattice: int) -> int:
        """Map (row, col, sublattice) to flat atom index."""
        return 3 * (row * n_cols + col) + sublattice

    # Hexagonal plaquette enumeration
    # A hexagonal plaquette centred between cells (i,j), (i,j+1), (i+1,j)
    # has 6 vertices (traced around the hexagon):
    #   B(i,j), C(i,j), A(i,j+1), B(i,j+1), C(i+1,j), A(i+1,j)
    # This requires i+1 < n_rows and j+1 < n_cols.
    plaquettes: list[list[int]] = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            hex_indices = [
                _atom_index(i, j, 1),      # B(i, j)
                _atom_index(i, j, 2),      # C(i, j)
                _atom_index(i, j + 1, 0),  # A(i, j+1)
                _atom_index(i, j + 1, 1),  # B(i, j+1)
                _atom_index(i + 1, j, 2),  # C(i+1, j)
                _atom_index(i + 1, j, 0),  # A(i+1, j)
            ]
            plaquettes.append(hex_indices)

    return KagomeLattice(
        n_rows=n_rows,
        n_cols=n_cols,
        spacing_um=spacing_um,
        positions=positions,
        atom_labels=labels,
        n_atoms=n_atoms,
        plaquettes=plaquettes,
    )


def create_kagome_register(lattice: KagomeLattice) -> pulser.Register:
    """Create a Pulser Register from a Kagome lattice.

    Args:
        lattice: A :class:`KagomeLattice` instance.

    Returns:
        A Pulser ``Register`` with atom positions matching the Kagome
        geometry.
    """
    return pulser.Register.from_coordinates(lattice.positions, prefix="q")


# ---------------------------------------------------------------------------
# Z2 QSL adiabatic preparation  (Task 2.1)
# ---------------------------------------------------------------------------

def prepare_z2_qsl(
    lattice: KagomeLattice,
    omega_max: float = 1.8,
    delta_initial: float = -8.0,
    delta_final: float = 2.5,
    duration_ns: int = 4000,
    noise_model=None,
) -> AnalogEvolutionResult:
    """Prepare a Z2 Quantum Spin Liquid state via adiabatic detuning sweep.

    Uses the Pulser backend to simulate adiabatic evolution of Rydberg atoms
    on a Kagome lattice.  The protocol sweeps detuning from a large negative
    value (all atoms in ground state) to a value near the Z2 QSL phase
    boundary.

    The Z2 QSL phase on the Kagome lattice occurs when the Rydberg blockade
    radius satisfies ``R_b / a`` in ``[1.5, 2.5]`` and the final detuning
    ``delta_final / Omega`` is near 1-2.

    With the default parameters (``a = 5.5 um``, ``Omega = 1.8 rad/us``),
    ``R_b / a ~ 1.8``, placing the system in the QSL regime.

    Args:
        lattice: :class:`KagomeLattice` defining the atom geometry.
        omega_max: Maximum Rabi frequency in rad/us.
        delta_initial: Initial detuning in rad/us (large negative).
        delta_final: Final detuning in rad/us.
        duration_ns: Sweep duration in nanoseconds.
        noise_model: Optional Pulser ``NoiseModel`` for Lindblad simulation.

    Returns:
        An :class:`~src.analog.AnalogEvolutionResult` with ``statevector``
        (coherent) or ``density_matrix`` (noisy), in Qiskit convention.
    """
    register = create_kagome_register(lattice)
    seq = build_rydberg_sequence(
        register,
        protocol="sweep",
        omega_max=omega_max,
        delta_initial=delta_initial,
        delta_final=delta_final,
        duration_ns=duration_ns,
    )
    result = simulate_pulser(seq, noise_model=noise_model)

    # Enrich metadata with lattice and QSL parameters
    result.metadata.update({
        "lattice_rows": lattice.n_rows,
        "lattice_cols": lattice.n_cols,
        "spacing_um": lattice.spacing_um,
        "omega_max": omega_max,
        "delta_final": delta_final,
        "phase": "z2_qsl",
    })
    return result


# ---------------------------------------------------------------------------
# String operators  (Task 2.2)
# ---------------------------------------------------------------------------

def get_plaquette_qubits(
    lattice: KagomeLattice,
    plaquette_index: int = 0,
) -> list[int]:
    """Get the qubit indices forming a hexagonal plaquette string operator.

    Returns the 6 atom/qubit indices that define the closed-loop string
    operator ``W_hex = prod_{i in hex} Z_i`` for the specified plaquette.

    Args:
        lattice: :class:`KagomeLattice` with enumerated plaquettes.
        plaquette_index: Index into ``lattice.plaquettes``.

    Returns:
        List of 6 qubit indices forming the hexagonal plaquette.

    Raises:
        IndexError: If *plaquette_index* is out of range.
    """
    if plaquette_index < 0 or plaquette_index >= len(lattice.plaquettes):
        raise IndexError(
            f"plaquette_index {plaquette_index} out of range "
            f"(lattice has {len(lattice.plaquettes)} plaquettes)"
        )
    return list(lattice.plaquettes[plaquette_index])


# ---------------------------------------------------------------------------
# String operator evaluation via VD  (Task 2.3)
# ---------------------------------------------------------------------------

def evaluate_string_operator(
    analog_result: AnalogEvolutionResult,
    observable_qubits: list[int],
    noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Evaluate a Z-string operator using both unmitigated and M=2 VD methods.

    Performs the full analog-digital pipeline for a multi-qubit Z-string
    operator ``W = prod_{k in observable_qubits} Z_k``:

    1. Build single-copy circuit from statevector, run with noise, compute
       unmitigated ``<W>``.
    2. Build M=2 circuit from statevector, run with noise, compute mitigated
       ``<W> = Tr(W rho^2) / Tr(rho^2)``.

    Args:
        analog_result: :class:`~src.analog.AnalogEvolutionResult` from QSL
            preparation.
        observable_qubits: Qubit indices defining the string operator.
        noise_model: Qiskit ``NoiseModel`` for the digital B^(2) layer.
        shots: Number of measurement shots.
        seed: Random seed.

    Returns:
        Dictionary with keys ``'observable_qubits'``, ``'n_qubits'``,
        ``'unmitigated_w'``, ``'mitigated_w'``, ``'tr_rho_sq'``,
        ``'shots'``, ``'source'``.
    """
    if analog_result.statevector is None:
        raise ValueError(
            "evaluate_string_operator currently requires a pure-state "
            "AnalogEvolutionResult (statevector must be non-None)."
        )

    sv = analog_result.statevector
    n_qubits = analog_result.n_qubits

    # Pulser simulations may produce statevectors with norm slightly != 1
    # due to numerical integration; renormalize so Qiskit initialize() accepts it.
    sv = sv / np.linalg.norm(sv)

    backend_kwargs: dict = {"method": "statevector"}
    if noise_model is not None:
        backend_kwargs["noise_model"] = noise_model
    backend = AerSimulator(**backend_kwargs)

    # --- Unmitigated (single-copy) ---
    single_circ = build_single_copy_circuit_from_statevector(sv, n_qubits)
    job_single = backend.run(single_circ, shots=shots, seed_simulator=seed)
    counts_single = job_single.result().get_counts()
    unmitigated_w = compute_unmitigated_string_expval(
        counts_single, observable_qubits,
    )

    # --- Mitigated (M=2 protocol) ---
    m2_circ = build_m2_circuit_from_statevector(sv, n_qubits)
    job_m2 = backend.run(m2_circ, shots=shots, seed_simulator=seed)
    counts_m2 = job_m2.result().get_counts()
    mitigated_w, _mean_num, mean_den = compute_mitigated_string_expval(
        counts_m2, observable_qubits, n_qubits,
    )

    return {
        "observable_qubits": list(observable_qubits),
        "n_qubits": n_qubits,
        "unmitigated_w": unmitigated_w,
        "mitigated_w": mitigated_w,
        "tr_rho_sq": mean_den,
        "shots": shots,
        "source": "topological",
    }


# ---------------------------------------------------------------------------
# End-to-end convenience pipeline
# ---------------------------------------------------------------------------

def run_topological_vd(
    n_rows: int = 2,
    n_cols: int = 2,
    spacing_um: float = 5.5,
    plaquette_index: int = 0,
    omega_max: float = 1.8,
    delta_initial: float = -8.0,
    delta_final: float = 2.5,
    duration_ns: int = 4000,
    noise_params: NeutralAtomNoiseParams | None = None,
    digital_noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Run the full topological VD pipeline: Kagome QSL + string operator.

    End-to-end convenience function that:

    1. Creates the Kagome lattice.
    2. Prepares the Z2 QSL state via adiabatic evolution (optionally noisy).
    3. Evaluates the hexagonal plaquette string operator using both
       unmitigated and M=2 VD methods.

    When *noise_params* is provided it drives both analog (Pulser Lindblad)
    and digital (Qiskit depolarizing) noise.  When *digital_noise_model* is
    provided directly, it overrides the digital noise derived from
    *noise_params*.

    Args:
        n_rows: Kagome lattice unit cell rows.
        n_cols: Kagome lattice unit cell columns.
        spacing_um: Nearest-neighbour distance in um.
        plaquette_index: Which hexagonal plaquette to evaluate.
        omega_max: Rabi frequency for QSL preparation.
        delta_initial: Initial detuning for sweep.
        delta_final: Final detuning for sweep.
        duration_ns: Sweep duration in ns.
        noise_params: Physical noise parameters.  ``None`` for noiseless.
        digital_noise_model: Override for digital-layer noise model.
        shots: Measurement shots.
        seed: Random seed.

    Returns:
        Dictionary with evaluation results and lattice metadata.
    """
    lattice = create_kagome_lattice(n_rows, n_cols, spacing_um)

    # Resolve noise models
    pulser_noise = None
    qiskit_noise = None
    if noise_params is not None:
        pulser_noise, qiskit_noise = compose_noise_models(noise_params)
    if digital_noise_model is not None:
        qiskit_noise = digital_noise_model

    analog_result = prepare_z2_qsl(
        lattice, omega_max, delta_initial, delta_final, duration_ns,
        noise_model=pulser_noise,
    )

    obs_qubits = get_plaquette_qubits(lattice, plaquette_index)
    result = evaluate_string_operator(
        analog_result, obs_qubits,
        noise_model=qiskit_noise, shots=shots, seed=seed,
    )

    result["lattice_rows"] = n_rows
    result["lattice_cols"] = n_cols
    result["lattice_atoms"] = lattice.n_atoms
    result["plaquette_index"] = plaquette_index
    return result


# ---------------------------------------------------------------------------
# Benchmark sweep  (Task 2.4)
# ---------------------------------------------------------------------------

@dataclass
class TopologicalBenchmarkResult:
    """Results from a topological VD benchmark sweep.

    Attributes:
        noise_levels: Array of two-qubit gate error probabilities swept.
        unmitigated_w: Array of unmitigated ``<W_hex>`` at each noise level.
        mitigated_w: Array of mitigated ``<W_hex>`` at each noise level.
        tr_rho_sq: Array of ``Tr(rho^2)`` estimates at each noise level.
        noiseless_w: The noiseless baseline ``<W_hex>`` value.
        lattice_info: Dict with lattice geometry summary.
        plaquette_qubits: The qubit indices used for the string operator.
    """

    noise_levels: np.ndarray
    unmitigated_w: np.ndarray
    mitigated_w: np.ndarray
    tr_rho_sq: np.ndarray
    noiseless_w: float
    lattice_info: dict
    plaquette_qubits: list[int]


_DEFAULT_NOISE_LEVELS = np.array(
    [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
)


def run_topological_benchmark(
    n_rows: int = 2,
    n_cols: int = 2,
    spacing_um: float = 5.5,
    plaquette_index: int = 0,
    noise_levels: np.ndarray | None = None,
    shots: int = 10_000,
    seed: int = 42,
    omega_max: float = 1.8,
    delta_initial: float = -8.0,
    delta_final: float = 2.5,
    duration_ns: int = 4000,
) -> TopologicalBenchmarkResult:
    """Benchmark unmitigated vs VD-mitigated string operators across noise.

    Sweeps digital-layer gate error probability while keeping the analog
    preparation noiseless.  This isolates the impact of digital measurement
    noise on topological order detection.

    The strategy:

    1. Prepare the Z2 QSL state **once** (analog phase, noiseless).
    2. For each noise level, apply scaled digital noise to the B^(2) layer
       and evaluate the hexagonal plaquette string operator.
    3. Compare unmitigated ``<W>`` vs mitigated
       ``Tr(W rho^2) / Tr(rho^2)``.

    Args:
        n_rows: Kagome lattice rows.
        n_cols: Kagome lattice columns.
        spacing_um: Nearest-neighbour distance in um.
        plaquette_index: Plaquette to evaluate.
        noise_levels: Array of ``p_2q`` values to sweep.  Defaults to
            ``[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]``.
            Single-qubit error is set to ``p_2q / 2``.
        shots: Measurement shots per noise level.
        seed: Random seed.
        omega_max: Rabi frequency for QSL preparation.
        delta_initial: Initial detuning for sweep.
        delta_final: Final detuning for sweep.
        duration_ns: Sweep duration in ns.

    Returns:
        A :class:`TopologicalBenchmarkResult` with arrays of results.
    """
    if noise_levels is None:
        noise_levels = _DEFAULT_NOISE_LEVELS.copy()

    # --- Step 1: Prepare QSL state once ---
    lattice = create_kagome_lattice(n_rows, n_cols, spacing_um)
    analog_result = prepare_z2_qsl(
        lattice, omega_max, delta_initial, delta_final, duration_ns,
    )
    obs_qubits = get_plaquette_qubits(lattice, plaquette_index)

    n_levels = len(noise_levels)
    unmit_arr = np.zeros(n_levels)
    mit_arr = np.zeros(n_levels)
    purity_arr = np.zeros(n_levels)

    rng = np.random.default_rng(seed)

    # --- Step 2: Sweep digital noise ---
    for idx, p_2q in enumerate(noise_levels):
        level_seed = int(rng.integers(0, 2**31))

        if p_2q <= 0:
            digital_noise = None
        else:
            p_1q = p_2q / 2
            digital_noise = get_pauli_noise_model(p_1q, p_2q)

        result = evaluate_string_operator(
            analog_result, obs_qubits,
            noise_model=digital_noise, shots=shots, seed=level_seed,
        )

        unmit_arr[idx] = result["unmitigated_w"]
        mit_arr[idx] = result["mitigated_w"]
        purity_arr[idx] = result["tr_rho_sq"]

    # Noiseless baseline is the first entry (p_2q = 0)
    noiseless_w = unmit_arr[0] if noise_levels[0] == 0 else float("nan")

    lattice_info = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_atoms": lattice.n_atoms,
        "n_plaquettes": len(lattice.plaquettes),
        "spacing_um": spacing_um,
    }

    return TopologicalBenchmarkResult(
        noise_levels=noise_levels,
        unmitigated_w=unmit_arr,
        mitigated_w=mit_arr,
        tr_rho_sq=purity_arr,
        noiseless_w=noiseless_w,
        lattice_info=lattice_info,
        plaquette_qubits=obs_qubits,
    )


def print_benchmark_results(result: TopologicalBenchmarkResult) -> None:
    """Pretty-print a :class:`TopologicalBenchmarkResult` table."""
    info = result.lattice_info
    print(
        f"\n{'=' * 72}\n"
        f"Topological VD Benchmark:  {info['n_atoms']}-atom Kagome lattice  "
        f"({info['n_rows']}x{info['n_cols']} unit cells)\n"
        f"Plaquette qubits: {result.plaquette_qubits}\n"
        f"Noiseless <W_hex>: {result.noiseless_w:+.4f}\n"
        f"{'=' * 72}"
    )
    print(
        f"{'p_2q':>8s}  {'Unmit <W>':>12s}  {'Mit <W>':>12s}  "
        f"{'Tr(rho^2)':>10s}  {'Unmit err':>10s}  {'Mit err':>10s}"
    )
    print("-" * 72)

    for i, p in enumerate(result.noise_levels):
        u = result.unmitigated_w[i]
        m = result.mitigated_w[i]
        tr = result.tr_rho_sq[i]
        u_err = abs(u - result.noiseless_w)
        m_err = abs(m - result.noiseless_w)
        print(
            f"{p:8.4f}  {u:+12.4f}  {m:+12.4f}  "
            f"{tr:10.4f}  {u_err:10.4f}  {m_err:10.4f}"
        )

    print("=" * 72)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Phase 2 topological VD benchmark."""
    print("Phase 2: Topological Order & String Operators via Analog-Digital VD")
    print("Preparing Z2 QSL on 2x2 Kagome lattice (12 atoms)...")

    result = run_topological_benchmark(
        n_rows=2,
        n_cols=2,
        shots=10_000,
        seed=42,
    )
    print_benchmark_results(result)


if __name__ == "__main__":
    main()
