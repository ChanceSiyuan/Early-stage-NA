"""Neutral-atom noise channels for analog-digital Virtual Distillation.

Provides realistic noise models for both the analog (Rydberg evolution) and
digital (B^(2) measurement layer) phases of the hybrid protocol on neutral
atom arrays.

Analog noise
    Mapped to Pulser's ``NoiseModel`` for Lindblad master-equation simulation:
    spontaneous emission (relaxation), dephasing, and Doppler broadening.

Digital noise
    Mapped to a Qiskit Aer ``NoiseModel`` with depolarizing gate errors and
    symmetric readout errors, matching the structure of
    :func:`src.utils.get_pauli_noise_model` but parameterized from physical
    neutral-atom error rates.

Atom loss
    Classical preprocessing step: Bernoulli-sample lost atoms and project
    their qubits to ``|0>`` in the statevector before the digital layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pulser.noise_model import NoiseModel as PulserNoiseModel
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from src.utils import _SINGLE_QUBIT_GATES, _TWO_QUBIT_GATES


@dataclass
class NeutralAtomNoiseParams:
    """Physical noise parameters for a neutral-atom quantum processor.

    Default values are representative of current Cs/Rb platforms.

    Attributes:
        spontaneous_emission_rate: Rydberg decay rate in 1/µs.
        dephasing_rate: Rydberg dephasing rate in 1/µs.
        doppler_temperature_uK: Atom temperature in µK (Doppler broadening).
        p_atom_loss: Probability of losing each atom per sequence.
        p_1q_digital: Depolarizing error probability for single-qubit gates.
        p_2q_digital: Depolarizing error probability for two-qubit gates.
        p_readout: Symmetric readout error probability.
        evolution_time_us: Total analog evolution time in µs.
    """

    spontaneous_emission_rate: float = 0.0077   # 1/µs  (Cs: ~1/130 µs)
    dephasing_rate: float = 0.005               # 1/µs
    doppler_temperature_uK: float = 20.0        # µK
    p_atom_loss: float = 0.02                   # per atom per sequence
    p_1q_digital: float = 0.005
    p_2q_digital: float = 0.02
    p_readout: float = 0.02
    evolution_time_us: float = 1.0


# ---------------------------------------------------------------------------
# Analog-phase noise (Pulser)
# ---------------------------------------------------------------------------

def get_pulser_noise_model(params: NeutralAtomNoiseParams) -> PulserNoiseModel:
    """Build a Pulser ``NoiseModel`` from neutral-atom noise parameters.

    Maps physical rates to Pulser's Lindblad master-equation channels:
    relaxation (spontaneous emission), dephasing, and Doppler temperature.

    Args:
        params: Neutral-atom noise parameters.

    Returns:
        A ``pulser.noise_model.NoiseModel``.
    """
    return PulserNoiseModel(
        relaxation_rate=params.spontaneous_emission_rate,
        dephasing_rate=params.dephasing_rate,
        temperature=params.doppler_temperature_uK,
        p_false_pos=params.p_readout / 2,
        p_false_neg=params.p_readout / 2,
    )


# ---------------------------------------------------------------------------
# Digital-phase noise (Qiskit Aer)
# ---------------------------------------------------------------------------

def get_neutral_atom_digital_noise(params: NeutralAtomNoiseParams) -> NoiseModel:
    """Build a Qiskit Aer noise model for the digital B^(2) layer.

    Applies depolarizing errors to single- and two-qubit gates and symmetric
    readout errors, using the same gate lists as
    :func:`src.utils.get_pauli_noise_model`.

    Args:
        params: Neutral-atom noise parameters.

    Returns:
        A Qiskit Aer ``NoiseModel``.
    """
    noise_model = NoiseModel()

    if params.p_1q_digital > 0:
        error_1q = depolarizing_error(params.p_1q_digital, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, _SINGLE_QUBIT_GATES)

    if params.p_2q_digital > 0:
        error_2q = depolarizing_error(params.p_2q_digital, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, _TWO_QUBIT_GATES)

    if params.p_readout > 0:
        p = params.p_readout
        ro_probs = [[1 - p, p], [p, 1 - p]]
        noise_model.add_all_qubit_readout_error(ReadoutError(ro_probs))

    return noise_model


# ---------------------------------------------------------------------------
# Atom loss model
# ---------------------------------------------------------------------------

def apply_atom_loss(
    statevector: np.ndarray,
    n_qubits: int,
    p_loss: float,
    seed: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Simulate atom loss by projecting lost qubits to ``|0>``.

    For each atom, independently sample whether it is lost (Bernoulli with
    probability *p_loss*).  Lost atoms are projected onto ``|0>`` and the
    statevector is renormalized.

    Args:
        statevector: Complex amplitude vector of length ``2^N``.
        n_qubits: Number of qubits *N*.
        p_loss: Per-atom loss probability.
        seed: Random seed.

    Returns:
        ``(modified_statevector, lost_qubit_indices)`` where the statevector
        has been projected and renormalized.
    """
    rng = np.random.default_rng(seed)
    lost = [q for q in range(n_qubits) if rng.random() < p_loss]

    if not lost:
        return statevector.copy(), lost

    sv = statevector.copy()
    dim = 1 << n_qubits

    for q in lost:
        # Zero out all amplitudes where qubit q is in state |1>.
        # In Qiskit convention, qubit q corresponds to bit position q
        # (from the right / LSB side).
        bit_mask = 1 << q
        for i in range(dim):
            if i & bit_mask:
                sv[i] = 0.0

    norm = np.linalg.norm(sv)
    if norm > 1e-15:
        sv /= norm

    return sv, lost


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_noise_models(
    params: NeutralAtomNoiseParams,
) -> tuple[PulserNoiseModel, NoiseModel]:
    """Build both analog and digital noise models from a single parameter set.

    Args:
        params: Neutral-atom noise parameters.

    Returns:
        ``(pulser_noise, qiskit_noise)`` tuple for the full analog-digital
        pipeline.
    """
    return get_pulser_noise_model(params), get_neutral_atom_digital_noise(params)
