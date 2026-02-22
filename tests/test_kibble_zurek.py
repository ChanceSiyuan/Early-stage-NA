"""Tests for src.kibble_zurek — Kibble-Zurek dynamic phase transitions via VD."""

import numpy as np
import pytest

from src.analog import AnalogEvolutionResult
from src.kibble_zurek import (
    KibbleZurekResult,
    compute_correlation_function,
    fit_correlation_length,
    fit_kz_scaling,
    run_kibble_zurek,
    sweep_quench_rates,
)


SEED = 42
SHOTS = 10_000


def _make_synthetic(n_qubits: int, seed: int = 42) -> AnalogEvolutionResult:
    """Create a synthetic AnalogEvolutionResult with a random statevector."""
    rng = np.random.default_rng(seed)
    sv = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    sv /= np.linalg.norm(sv)
    return AnalogEvolutionResult(
        statevector=sv,
        density_matrix=None,
        n_qubits=n_qubits,
        metadata={"source": "synthetic"},
    )


# ---------------------------------------------------------------------------
# Correlation function tests
# ---------------------------------------------------------------------------

class TestComputeCorrelationFunction:
    @pytest.fixture(scope="class")
    def corr_result(self):
        analog = _make_synthetic(4, seed=SEED)
        return compute_correlation_function(analog, shots=SHOTS, seed=SEED)

    def test_returns_valid_dict(self, corr_result):
        expected = {
            "n_qubits", "distances", "unmitigated_corr",
            "mitigated_corr", "unmitigated_zi", "mitigated_zi",
        }
        assert expected.issubset(set(corr_result.keys()))

    def test_distances_shape(self, corr_result):
        # N=4 -> distances [1, 2, 3]
        assert len(corr_result["distances"]) == 3
        np.testing.assert_array_equal(corr_result["distances"], [1, 2, 3])

    def test_corr_shape(self, corr_result):
        assert len(corr_result["unmitigated_corr"]) == 3
        assert len(corr_result["mitigated_corr"]) == 3

    def test_zi_shape(self, corr_result):
        assert len(corr_result["unmitigated_zi"]) == 4
        assert len(corr_result["mitigated_zi"]) == 4

    def test_ground_state_zero_correlation(self):
        """|0000>: all <Z_i>=+1, <Z_iZ_j>=+1, so C(r)=0."""
        sv = np.zeros(2**4, dtype=complex)
        sv[0] = 1.0
        analog = AnalogEvolutionResult(
            statevector=sv, density_matrix=None, n_qubits=4, metadata={},
        )
        result = compute_correlation_function(analog, shots=SHOTS, seed=SEED)
        for c in result["unmitigated_corr"]:
            assert abs(c) < 0.1

    def test_random_state_finite_correlations(self):
        """Random state should have some nonzero correlations."""
        analog = _make_synthetic(4, seed=99)
        result = compute_correlation_function(analog, shots=SHOTS, seed=SEED)
        # At least one distance should have |C(r)| > 0.01
        assert any(abs(c) > 0.01 for c in result["unmitigated_corr"])


# ---------------------------------------------------------------------------
# Correlation length fitting
# ---------------------------------------------------------------------------

class TestFitCorrelationLength:
    def test_known_exponential(self):
        """Synthetic C(r) = exp(-r/2) should give ξ ≈ 2."""
        distances = np.arange(1, 8)
        corr = np.exp(-distances / 2.0)
        xi = fit_correlation_length(distances, corr)
        assert abs(xi - 2.0) < 0.1

    def test_short_range_exponential(self):
        """C(r) = exp(-r/1.5) should give ξ ≈ 1.5."""
        distances = np.arange(1, 6)
        corr = np.exp(-distances / 1.5)
        xi = fit_correlation_length(distances, corr)
        assert abs(xi - 1.5) < 0.1

    def test_zero_correlation_returns_nan(self):
        """All-zero C(r) should return nan."""
        distances = np.arange(1, 5)
        corr = np.zeros(4)
        xi = fit_correlation_length(distances, corr)
        assert np.isnan(xi)

    def test_single_point_returns_nan(self):
        """Only one valid point — insufficient for fit."""
        distances = np.array([1])
        corr = np.array([0.5])
        xi = fit_correlation_length(distances, corr)
        assert np.isnan(xi)

    def test_positive_slope_returns_nan(self):
        """Growing correlations (unphysical) should return nan."""
        distances = np.arange(1, 5)
        corr = np.exp(distances / 2.0)  # growing
        xi = fit_correlation_length(distances, corr)
        assert np.isnan(xi)


# ---------------------------------------------------------------------------
# KZ scaling fit
# ---------------------------------------------------------------------------

class TestFitKZScaling:
    def test_synthetic_kz_data(self):
        """Synthetic ξ ~ τ_Q^{0.5} data should recover exponent ≈ 0.5."""
        durations = np.array([500, 1000, 2000, 4000], dtype=float)
        distances = np.arange(1, 6)

        # Build synthetic C(r, τ_Q) = exp(-r / ξ(τ_Q)) with ξ = 0.1 * τ_Q^0.5
        xi_true = 0.1 * durations**0.5
        unmit_corr = np.zeros((len(durations), len(distances)))
        for idx, xi in enumerate(xi_true):
            unmit_corr[idx] = np.exp(-distances / xi)

        result = KibbleZurekResult(
            n_qubits=6,
            quench_durations_ns=durations,
            distances=distances,
            unmitigated_corr=unmit_corr,
            mitigated_corr=unmit_corr.copy(),
        )
        fitted = fit_kz_scaling(result, nu=1.0, z=1.0)
        assert fitted.kz_prediction == pytest.approx(0.5)
        assert abs(fitted.kz_exponent_unmitigated - 0.5) < 0.15

    def test_populates_xi_arrays(self):
        """After fitting, xi arrays should be populated."""
        durations = np.array([1000, 2000], dtype=float)
        distances = np.arange(1, 4)
        corr = np.array([[0.8, 0.5, 0.3], [0.9, 0.7, 0.5]])
        result = KibbleZurekResult(
            n_qubits=4,
            quench_durations_ns=durations,
            distances=distances,
            unmitigated_corr=corr,
            mitigated_corr=corr.copy(),
        )
        fitted = fit_kz_scaling(result)
        assert fitted.unmitigated_xi is not None
        assert fitted.mitigated_xi is not None
        assert len(fitted.unmitigated_xi) == 2

    def test_prediction_value(self):
        """kz_prediction should be ν/(1+zν)."""
        durations = np.array([1000, 2000], dtype=float)
        distances = np.arange(1, 4)
        corr = np.array([[0.8, 0.5, 0.3], [0.9, 0.7, 0.5]])
        result = KibbleZurekResult(
            n_qubits=4,
            quench_durations_ns=durations,
            distances=distances,
            unmitigated_corr=corr,
            mitigated_corr=corr.copy(),
        )
        fitted = fit_kz_scaling(result, nu=1.0, z=1.0)
        assert fitted.kz_prediction == pytest.approx(0.5)

        fitted2 = fit_kz_scaling(result, nu=0.5, z=2.0)
        assert fitted2.kz_prediction == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Quench rate sweep (integration)
# ---------------------------------------------------------------------------

class TestSweepQuenchRates:
    def test_smoke_small_chain(self):
        """sweep_quench_rates completes on a 4-atom chain with 2 durations."""
        result = sweep_quench_rates(
            n_atoms=4,
            quench_durations_ns=[500, 1000],
            shots=1000,
            seed=SEED,
        )
        assert isinstance(result, KibbleZurekResult)
        assert result.n_qubits == 4
        assert len(result.quench_durations_ns) == 2
        assert result.unmitigated_corr.shape == (2, 3)
        assert result.mitigated_corr.shape == (2, 3)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestRunKibbleZurek:
    def test_end_to_end_smoke(self):
        """run_kibble_zurek completes on a 4-atom chain."""
        result = run_kibble_zurek(
            n_atoms=4,
            quench_durations_ns=[500, 1000],
            shots=1000,
            seed=SEED,
        )
        assert isinstance(result, KibbleZurekResult)
        assert result.kz_prediction is not None
        assert result.unmitigated_xi is not None
        assert result.mitigated_xi is not None
        assert len(result.unmitigated_xi) == 2
