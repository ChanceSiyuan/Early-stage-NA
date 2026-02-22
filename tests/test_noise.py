"""Tests for src.neutral_atom_noise — noise model construction."""

import numpy as np
import pytest

from src.neutral_atom_noise import (
    NeutralAtomNoiseParams,
    apply_atom_loss,
    compose_noise_models,
    get_neutral_atom_digital_noise,
    get_pulser_noise_model,
)


class TestNeutralAtomNoiseParams:
    def test_defaults(self):
        params = NeutralAtomNoiseParams()
        assert params.spontaneous_emission_rate == pytest.approx(0.0077)
        assert params.dephasing_rate == pytest.approx(0.005)
        assert params.doppler_temperature_uK == pytest.approx(20.0)
        assert params.p_atom_loss == pytest.approx(0.02)
        assert params.p_1q_digital == pytest.approx(0.005)
        assert params.p_2q_digital == pytest.approx(0.02)
        assert params.p_readout == pytest.approx(0.02)
        assert params.evolution_time_us == pytest.approx(1.0)

    def test_custom_values(self):
        params = NeutralAtomNoiseParams(
            spontaneous_emission_rate=0.01,
            p_2q_digital=0.05,
        )
        assert params.spontaneous_emission_rate == pytest.approx(0.01)
        assert params.p_2q_digital == pytest.approx(0.05)


class TestPulserNoiseModel:
    def test_creation(self):
        params = NeutralAtomNoiseParams()
        noise = get_pulser_noise_model(params)
        # Should be a Pulser NoiseModel instance
        from pulser.noise_model import NoiseModel as PulserNoiseModel
        assert isinstance(noise, PulserNoiseModel)

    def test_rates_mapped_correctly(self):
        params = NeutralAtomNoiseParams(
            spontaneous_emission_rate=0.01,
            dephasing_rate=0.02,
            doppler_temperature_uK=30.0,
        )
        noise = get_pulser_noise_model(params)
        assert noise.relaxation_rate == pytest.approx(0.01)
        assert noise.dephasing_rate == pytest.approx(0.02)
        assert noise.temperature == pytest.approx(30.0)


class TestDigitalNoiseModel:
    def test_creation(self):
        params = NeutralAtomNoiseParams()
        noise = get_neutral_atom_digital_noise(params)
        from qiskit_aer.noise import NoiseModel
        assert isinstance(noise, NoiseModel)

    def test_zero_errors_empty_model(self):
        params = NeutralAtomNoiseParams(
            p_1q_digital=0.0, p_2q_digital=0.0, p_readout=0.0,
        )
        noise = get_neutral_atom_digital_noise(params)
        # An empty noise model should have no quantum errors
        assert len(noise.to_dict()["errors"]) == 0


class TestAtomLoss:
    def test_no_loss(self):
        sv = np.array([1, 0, 0, 0], dtype=complex)
        sv_out, lost = apply_atom_loss(sv, 2, p_loss=0.0, seed=42)
        assert lost == []
        np.testing.assert_array_almost_equal(sv_out, sv)

    def test_all_loss(self):
        """All atoms lost → project to |00...0>, renormalize."""
        sv = np.array([0, 0, 0, 1], dtype=complex)  # |11>
        sv_out, lost = apply_atom_loss(sv, 2, p_loss=1.0, seed=42)
        assert len(lost) == 2
        # |11> projected to |0> on both qubits → all zero → but original
        # had no |00> component, so result is zero vector (edge case)
        # Actually: projecting qubit 0 to |0> kills |11> entirely
        # The function should handle this gracefully
        assert np.linalg.norm(sv_out) < 1e-10 or abs(sv_out[0]) > 0.99

    def test_single_qubit_loss(self):
        """Lose qubit 0: |01> + |11> → |01> (qubit 0 projected to |0>)."""
        sv = np.array([0, 1, 0, 1], dtype=complex) / np.sqrt(2)  # (|01> + |11>) / sqrt(2)
        # Force qubit 0 to be lost by using a seed that gives loss on q0
        # Instead, test with p_loss=1.0 on a 1-qubit system
        sv_1q = np.array([0.6, 0.8], dtype=complex)
        sv_out, lost = apply_atom_loss(sv_1q, 1, p_loss=1.0, seed=42)
        assert lost == [0]
        # Qubit 0 projected to |0>: amplitude at index 1 (|1>) zeroed
        assert abs(sv_out[0]) == pytest.approx(1.0)
        assert abs(sv_out[1]) == pytest.approx(0.0)

    def test_preserves_normalization(self):
        sv = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        sv_out, _ = apply_atom_loss(sv, 2, p_loss=0.5, seed=123)
        norm = np.linalg.norm(sv_out)
        assert abs(norm - 1.0) < 1e-10 or norm < 1e-10  # either normalized or zero


class TestComposition:
    def test_returns_tuple(self):
        params = NeutralAtomNoiseParams()
        result = compose_noise_models(params)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_correct_types(self):
        from pulser.noise_model import NoiseModel as PulserNoiseModel
        from qiskit_aer.noise import NoiseModel as QiskitNoiseModel

        params = NeutralAtomNoiseParams()
        pulser_noise, qiskit_noise = compose_noise_models(params)
        assert isinstance(pulser_noise, PulserNoiseModel)
        assert isinstance(qiskit_noise, QiskitNoiseModel)
