"""Analog evolution backends for neutral-atom Rydberg Hamiltonian simulation.

Provides a common interface wrapping Pulser and Bloqade for analog state
preparation, producing statevectors or density matrices that feed into the
digital B^(2) measurement layer in ``src.simulation``.

Pulser backend
    Uses ``QutipEmulator`` for time evolution of Rydberg Hamiltonians on
    arbitrary atom registers.  Supports both coherent (pure-state) and
    noisy (Lindblad master equation → density matrix) simulation.

Bloqade backend
    Uses the ``bloqade-analog`` Python SDK with its native local emulator.
    The high-level API returns shot-based counts; statevector access uses
    the internal emulator IR.

Basis convention
    Pulser: ``|r>`` = index 0, ``|g>`` = index 1 per qubit.
    Qiskit: ``|0>`` = ``|g>`` = index 0, ``|1>`` = ``|r>`` = index 1.
    All outputs are converted to Qiskit convention before returning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Pulser imports (deferred to function scope where needed for optional deps)
# ---------------------------------------------------------------------------
import pulser
from pulser import Pulse, Sequence
from pulser.devices import MockDevice
from pulser.waveforms import RampWaveform
from pulser_simulation import QutipEmulator

# ---------------------------------------------------------------------------
# Bloqade imports
# ---------------------------------------------------------------------------
from bloqade.analog import start as bloqade_start


# ---------------------------------------------------------------------------
# Common result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalogEvolutionResult:
    """Result of an analog Rydberg Hamiltonian evolution.

    Exactly one of ``statevector`` or ``density_matrix`` will be non-None.

    Attributes:
        statevector: Complex amplitudes in Qiskit convention, shape ``(2^N,)``.
        density_matrix: Density matrix in Qiskit convention, shape ``(2^N, 2^N)``.
        n_qubits: Number of atoms / logical qubits.
        backend_name: ``"pulser"`` or ``"bloqade"``.
        metadata: Timing, parameters, and other backend-specific info.
    """

    statevector: np.ndarray | None = None
    density_matrix: np.ndarray | None = None
    n_qubits: int = 0
    backend_name: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Basis conversion helpers
# ---------------------------------------------------------------------------

def _flip_index(index: int, n_qubits: int) -> int:
    """Flip all bits in *index* (XOR with 2^N - 1).

    Maps between Pulser ordering (|r>=0, |g>=1) and Qiskit (|g>=0, |r>=1).
    """
    return index ^ ((1 << n_qubits) - 1)


def _pulser_to_qiskit_sv(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert a Pulser statevector to Qiskit basis ordering."""
    dim = 1 << n_qubits
    out = np.empty(dim, dtype=complex)
    for i in range(dim):
        out[_flip_index(i, n_qubits)] = sv[i]
    return out


def _pulser_to_qiskit_dm(dm: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert a Pulser density matrix to Qiskit basis ordering."""
    dim = 1 << n_qubits
    out = np.empty((dim, dim), dtype=complex)
    for i in range(dim):
        fi = _flip_index(i, n_qubits)
        for j in range(dim):
            out[fi, _flip_index(j, n_qubits)] = dm[i, j]
    return out


# ---------------------------------------------------------------------------
# Pulser backend
# ---------------------------------------------------------------------------

def create_1d_register(n_atoms: int, spacing_um: float = 7.0) -> pulser.Register:
    """Create a 1-D chain of atoms with uniform spacing.

    Args:
        n_atoms: Number of atoms.
        spacing_um: Inter-atom distance in micrometres.

    Returns:
        A Pulser ``Register``.
    """
    coords = [[i * spacing_um, 0.0] for i in range(n_atoms)]
    return pulser.Register.from_coordinates(coords, prefix="q")


def create_2d_register(
    n_atoms: int,
    layout: str = "square",
    spacing_um: float = 7.0,
) -> pulser.Register:
    """Create a 2-D atom register.

    Args:
        n_atoms: Desired number of atoms (actual count may be rounded up
            to fill the lattice).
        layout: ``"square"`` or ``"triangular"``.
        spacing_um: Lattice spacing in micrometres.

    Returns:
        A Pulser ``Register``.
    """
    if layout == "square":
        side = math.ceil(math.sqrt(n_atoms))
        reg = pulser.Register.square(side, spacing=spacing_um, prefix="q")
    elif layout == "triangular":
        side = math.ceil(math.sqrt(n_atoms))
        reg = pulser.Register.triangular_lattice(
            rows=side, atoms_per_row=side, spacing=spacing_um, prefix="q",
        )
    else:
        raise ValueError(f"Unknown layout '{layout}'. Use 'square' or 'triangular'.")
    return reg


def build_rydberg_sequence(
    register: pulser.Register,
    protocol: str = "quench",
    omega_max: float = 2.0,
    delta_initial: float = -5.0,
    delta_final: float = 5.0,
    duration_ns: int = 1000,
    device=None,
) -> Sequence:
    """Build a Pulser ``Sequence`` for Rydberg Hamiltonian evolution.

    Args:
        register: Atom register.
        protocol: ``"quench"`` (constant drive) or ``"sweep"`` (detuning ramp).
        omega_max: Maximum Rabi frequency in rad/µs.
        delta_initial: Initial detuning in rad/µs (sweep only).
        delta_final: Final detuning in rad/µs (sweep only).
        duration_ns: Pulse duration in nanoseconds.
        device: Pulser device (defaults to ``MockDevice``).

    Returns:
        A Pulser ``Sequence`` ready for simulation.
    """
    if device is None:
        device = MockDevice
    seq = Sequence(register, device)
    seq.declare_channel("ryd", "rydberg_global")

    if protocol == "quench":
        seq.add(
            Pulse.ConstantPulse(duration_ns, amplitude=omega_max, detuning=0.0, phase=0.0),
            "ryd",
        )
    elif protocol == "sweep":
        seq.add(
            Pulse.ConstantAmplitude(
                omega_max,
                RampWaveform(duration_ns, delta_initial, delta_final),
                0.0,
            ),
            "ryd",
        )
    else:
        raise ValueError(f"Unknown protocol '{protocol}'. Use 'quench' or 'sweep'.")

    seq.measure("ground-rydberg")
    return seq


def simulate_pulser(
    sequence: Sequence,
    noise_model=None,
    n_trajectories: int = 1,
) -> AnalogEvolutionResult:
    """Run a Pulser sequence through QutipEmulator.

    Args:
        sequence: A measured Pulser ``Sequence``.
        noise_model: A ``pulser.noise_model.NoiseModel`` for Lindblad
            simulation, or ``None`` for coherent evolution.
        n_trajectories: Number of noise trajectories (ignored when
            ``noise_model`` is ``None``).

    Returns:
        An ``AnalogEvolutionResult`` with either ``statevector`` (coherent)
        or ``density_matrix`` (noisy) populated, in Qiskit convention.
    """
    n_qubits = len(sequence.register.qubit_ids)

    if noise_model is not None:
        from pulser_simulation import QutipConfig

        config = QutipConfig(noise_model=noise_model)
        sim = QutipEmulator.from_sequence(sequence, config=config)
    else:
        sim = QutipEmulator.from_sequence(sequence)

    results = sim.run()
    final_state = results.get_final_state()

    metadata = {
        "duration_ns": sequence.get_duration(),
        "n_atoms": n_qubits,
        "noisy": noise_model is not None,
    }

    if final_state.isket:
        sv_pulser = final_state.full().flatten()
        sv_qiskit = _pulser_to_qiskit_sv(sv_pulser, n_qubits)
        return AnalogEvolutionResult(
            statevector=sv_qiskit,
            n_qubits=n_qubits,
            backend_name="pulser",
            metadata=metadata,
        )
    else:
        dm_pulser = final_state.full()
        dm_qiskit = _pulser_to_qiskit_dm(dm_pulser, n_qubits)
        return AnalogEvolutionResult(
            density_matrix=dm_qiskit,
            n_qubits=n_qubits,
            backend_name="pulser",
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Bloqade backend
# ---------------------------------------------------------------------------

def build_bloqade_program(
    n_atoms: int,
    spacing_um: float = 7.0,
    protocol: str = "quench",
    omega_max: float = 2.0,
    delta_initial: float = -5.0,
    delta_final: float = 5.0,
    duration_us: float = 1.0,
):
    """Build a Bloqade analog program for Rydberg evolution.

    Args:
        n_atoms: Number of atoms in a 1-D chain.
        spacing_um: Inter-atom spacing in micrometres.
        protocol: ``"quench"`` or ``"sweep"``.
        omega_max: Maximum Rabi frequency in rad/µs.
        delta_initial: Initial detuning in rad/µs (sweep only).
        delta_final: Final detuning in rad/µs (sweep only).
        duration_us: Evolution time in microseconds.

    Returns:
        A Bloqade program object ready for ``.bloqade.python().run()``.
    """
    prog = bloqade_start
    for i in range(n_atoms):
        prog = prog.add_position((i * spacing_um, 0))

    if protocol == "quench":
        prog = (
            prog
            .rydberg.rabi.amplitude.uniform
            .constant(duration_us, omega_max)
            .detuning.uniform
            .constant(duration_us, 0.0)
        )
    elif protocol == "sweep":
        prog = (
            prog
            .rydberg.rabi.amplitude.uniform
            .constant(duration_us, omega_max)
            .detuning.uniform
            .linear(duration_us, delta_initial, delta_final)
        )
    else:
        raise ValueError(f"Unknown protocol '{protocol}'. Use 'quench' or 'sweep'.")

    return prog


def simulate_bloqade(
    program,
    shots: int = 10_000,
) -> AnalogEvolutionResult:
    """Run a Bloqade program on the native Python emulator.

    The high-level Bloqade API returns shot-based measurement counts.
    We reconstruct an approximate probability vector from the counts and
    build a statevector with uniform phases (amplitudes only).

    For exact statevector access, use the Pulser backend instead.

    Args:
        program: A Bloqade program (from ``build_bloqade_program``).
        shots: Number of measurement shots.

    Returns:
        An ``AnalogEvolutionResult`` with ``statevector`` populated.
    """
    result = program.bloqade.python().run(shots)
    report = result.report()
    counts_list = report.counts()
    counts = counts_list[0]  # first (and only) batch element

    # Determine n_qubits from the first bitstring key
    first_key = next(iter(counts))
    n_qubits = len(first_key)
    dim = 1 << n_qubits

    # Build probability vector from shot counts
    probs = np.zeros(dim)
    for bitstring, count in counts.items():
        # Bloqade bitstrings: '0' = ground, '1' = Rydberg (same as Qiskit)
        index = int(bitstring, 2)
        probs[index] = count / shots

    # Approximate statevector (real, positive amplitudes from probabilities)
    sv = np.sqrt(np.maximum(probs, 0.0))
    norm = np.linalg.norm(sv)
    if norm > 1e-15:
        sv /= norm

    metadata = {
        "shots": shots,
        "n_atoms": n_qubits,
        "approximate": True,
    }

    return AnalogEvolutionResult(
        statevector=sv,
        n_qubits=n_qubits,
        backend_name="bloqade",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_analog_evolution(
    n_qubits: int,
    backend: str = "pulser",
    protocol: str = "quench",
    spacing_um: float = 7.0,
    duration_ns: int = 1000,
    noise_model=None,
    n_trajectories: int = 1,
    **kwargs,
) -> AnalogEvolutionResult:
    """Run analog Rydberg evolution on the chosen backend.

    Single entry point that dispatches to Pulser or Bloqade.

    Args:
        n_qubits: Number of atoms.
        backend: ``"pulser"`` or ``"bloqade"``.
        protocol: ``"quench"`` or ``"sweep"``.
        spacing_um: Inter-atom spacing in micrometres.
        duration_ns: Evolution time in nanoseconds (Pulser) or converted
            to microseconds for Bloqade.
        noise_model: Pulser ``NoiseModel`` (Pulser backend only).
        n_trajectories: Noise trajectories (Pulser backend only).
        **kwargs: Extra keyword arguments forwarded to the backend builder
            (e.g. ``omega_max``, ``delta_initial``, ``delta_final``).

    Returns:
        An ``AnalogEvolutionResult``.
    """
    omega_max = kwargs.get("omega_max", 2.0)
    delta_initial = kwargs.get("delta_initial", -5.0)
    delta_final = kwargs.get("delta_final", 5.0)

    if backend == "pulser":
        reg = create_1d_register(n_qubits, spacing_um)
        seq = build_rydberg_sequence(
            reg, protocol, omega_max, delta_initial, delta_final, duration_ns,
        )
        return simulate_pulser(seq, noise_model, n_trajectories)

    elif backend == "bloqade":
        duration_us = duration_ns / 1000.0
        shots = kwargs.get("shots", 10_000)
        prog = build_bloqade_program(
            n_qubits, spacing_um, protocol,
            omega_max, delta_initial, delta_final, duration_us,
        )
        return simulate_bloqade(prog, shots)

    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'pulser' or 'bloqade'.")
