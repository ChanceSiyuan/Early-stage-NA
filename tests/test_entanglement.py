"""Tests for src.entanglement — subsystem Rényi entropy via VD."""

import numpy as np
import pytest

from src.analog import AnalogEvolutionResult
from src.simulation import (
    build_m2_circuit_from_statevector,
    build_subsystem_m2_circuit_from_statevector,
)
from src.entanglement import (
    EntanglementEntropyResult,
    evaluate_subsystem_purity,
    fit_cft_scaling,
    prepare_1d_chain_state,
    run_entanglement_entropy,
    sweep_subsystem_size,
)
from src.utils import compute_subsystem_purity, compute_renyi_entropy
from qiskit_aer import AerSimulator


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


def _run_subsystem_m2(sv, n_qubits, subsystem, shots=SHOTS, seed=SEED):
    """Helper: run subsystem M=2 circuit, return counts."""
    circ = build_subsystem_m2_circuit_from_statevector(sv, n_qubits, subsystem)
    backend = AerSimulator(method="statevector")
    job = backend.run(circ, shots=shots, seed_simulator=seed)
    return job.result().get_counts()


# ---------------------------------------------------------------------------
# Circuit builder tests
# ---------------------------------------------------------------------------

class TestBuildSubsystemM2Circuit:
    def test_circuit_shape(self):
        sv = np.array([1, 0, 0, 0], dtype=complex)
        circ = build_subsystem_m2_circuit_from_statevector(sv, 2, [0])
        assert circ.num_qubits == 4
        assert circ.num_clbits == 4

    def test_none_subsystem_matches_full(self):
        """subsystem=None should produce same gate count as full M=2."""
        sv = np.array([1, 0, 0, 0], dtype=complex)
        full = build_m2_circuit_from_statevector(sv, 2)
        sub_none = build_subsystem_m2_circuit_from_statevector(sv, 2, None)
        # Same number of operations (initialize + barrier + B^(2) + barrier + measure)
        assert full.size() == sub_none.size()

    def test_partial_subsystem_fewer_gates(self):
        """Partial subsystem should have fewer gates than full."""
        n = 4
        sv = np.zeros(2**n, dtype=complex)
        sv[0] = 1.0
        full = build_subsystem_m2_circuit_from_statevector(sv, n, list(range(n)))
        partial = build_subsystem_m2_circuit_from_statevector(sv, n, [0])
        assert partial.size() < full.size()


# ---------------------------------------------------------------------------
# Subsystem purity post-processing
# ---------------------------------------------------------------------------

class TestComputeSubsystemPurity:
    def test_product_state_full_purity(self):
        """|00> product state: Tr(rho_A^2) = 1.0 for any subsystem."""
        sv = np.array([1, 0, 0, 0], dtype=complex)
        counts = _run_subsystem_m2(sv, 2, [0])
        purity = compute_subsystem_purity(counts, [0], 2)
        assert abs(purity - 1.0) < 0.1

    def test_bell_state_half_purity(self):
        """Bell state (|00>+|11>)/sqrt(2): Tr(rho_A^2) = 0.5 for A={0}."""
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        counts = _run_subsystem_m2(sv, 2, [0], shots=50_000)
        purity = compute_subsystem_purity(counts, [0], 2)
        assert abs(purity - 0.5) < 0.1

    def test_product_state_subsystem_pure(self):
        """|+0>: subsystem {0} is pure, Tr(rho_A^2) = 1.0."""
        sv = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        counts = _run_subsystem_m2(sv, 2, [0])
        purity = compute_subsystem_purity(counts, [0], 2)
        assert abs(purity - 1.0) < 0.1

    def test_ghz_state_half_purity(self):
        """3-qubit GHZ: any single-qubit subsystem has Tr(rho_A^2) = 0.5."""
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0 / np.sqrt(2)  # |000>
        sv[7] = 1.0 / np.sqrt(2)  # |111>
        counts = _run_subsystem_m2(sv, 3, [0], shots=50_000)
        purity = compute_subsystem_purity(counts, [0], 3)
        assert abs(purity - 0.5) < 0.1


class TestComputeRenyiEntropy:
    def test_product_state_zero_entropy(self):
        """|00>: S_2 = 0 for any subsystem."""
        sv = np.array([1, 0, 0, 0], dtype=complex)
        counts = _run_subsystem_m2(sv, 2, [0])
        s2, purity = compute_renyi_entropy(counts, [0], 2)
        assert abs(s2) < 0.15

    def test_bell_state_one_ebit(self):
        """Bell pair: S_2 = 1.0 for subsystem {0}."""
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        counts = _run_subsystem_m2(sv, 2, [0], shots=50_000)
        s2, purity = compute_renyi_entropy(counts, [0], 2)
        assert abs(s2 - 1.0) < 0.2

    def test_ghz_entropy(self):
        """3-qubit GHZ: S_2({0}) = 1.0."""
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0 / np.sqrt(2)
        sv[7] = 1.0 / np.sqrt(2)
        counts = _run_subsystem_m2(sv, 3, [0], shots=50_000)
        s2, _ = compute_renyi_entropy(counts, [0], 3)
        assert abs(s2 - 1.0) < 0.2


# ---------------------------------------------------------------------------
# evaluate_subsystem_purity integration
# ---------------------------------------------------------------------------

class TestEvaluateSubsystemPurity:
    def test_returns_valid_dict(self):
        analog = _make_synthetic(4)
        result = evaluate_subsystem_purity(analog, [0, 1], shots=1000, seed=SEED)
        expected_keys = {
            "subsystem", "n_qubits", "subsystem_size",
            "tr_rho_a_sq", "s2_entropy", "shots", "source",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_source_key(self):
        analog = _make_synthetic(4)
        result = evaluate_subsystem_purity(analog, [0], shots=1000, seed=SEED)
        assert result["source"] == "entanglement"

    def test_noiseless_product_state(self):
        """Product state: Tr(rho_A^2) ~ 1.0."""
        sv = np.zeros(2**4, dtype=complex)
        sv[0] = 1.0
        analog = AnalogEvolutionResult(
            statevector=sv, density_matrix=None, n_qubits=4, metadata={},
        )
        result = evaluate_subsystem_purity(analog, [0, 1], shots=SHOTS, seed=SEED)
        assert abs(result["tr_rho_a_sq"] - 1.0) < 0.1

    def test_entangled_state_purity_below_one(self):
        """Random entangled state: Tr(rho_A^2) < 1.0 for a proper subsystem."""
        analog = _make_synthetic(4, seed=123)
        result = evaluate_subsystem_purity(analog, [0, 1], shots=SHOTS, seed=SEED)
        assert result["tr_rho_a_sq"] < 0.95


# ---------------------------------------------------------------------------
# Sweep and CFT fit
# ---------------------------------------------------------------------------

class TestSweepSubsystemSize:
    def test_sweep_returns_correct_sizes(self):
        analog = _make_synthetic(6)
        result = sweep_subsystem_size(analog, shots=1000, seed=SEED)
        np.testing.assert_array_equal(result.subsystem_sizes, [1, 2, 3])

    def test_sweep_product_state_low_entropy(self):
        """Product state: all S_2 values should be near 0."""
        sv = np.zeros(2**4, dtype=complex)
        sv[0] = 1.0
        analog = AnalogEvolutionResult(
            statevector=sv, density_matrix=None, n_qubits=4, metadata={},
        )
        result = sweep_subsystem_size(analog, shots=SHOTS, seed=SEED)
        assert all(s < 0.15 for s in result.s2_entropy)

    def test_sweep_entangled_state_nonzero(self):
        """Random state: at least some S_2 > 0."""
        analog = _make_synthetic(4, seed=99)
        result = sweep_subsystem_size(analog, shots=SHOTS, seed=SEED)
        assert any(s > 0.1 for s in result.s2_entropy)


class TestFitCFTScaling:
    def test_fit_populates_fields(self):
        """After fitting, fitted_central_charge and cft_prediction should be set."""
        analog = _make_synthetic(6)
        result = sweep_subsystem_size(analog, shots=SHOTS, seed=SEED)
        fitted = fit_cft_scaling(result)
        assert fitted.fitted_central_charge is not None
        assert fitted.cft_prediction is not None
        assert len(fitted.cft_prediction) == len(fitted.subsystem_sizes)

    def test_synthetic_log_scaling(self):
        """Synthetic S_2 data with known c should be recovered by the fit."""
        L = 20
        sizes = np.arange(1, L // 2 + 1)
        c_true = 0.5
        x = np.log((2 * L / np.pi) * np.sin(np.pi * sizes / L))
        s2 = (c_true / 6.0) * x + 0.3  # OBC formula

        result = EntanglementEntropyResult(
            n_qubits=L,
            subsystem_sizes=sizes,
            tr_rho_a_sq=2.0 ** (-s2),
            s2_entropy=s2,
        )
        fitted = fit_cft_scaling(result, boundary="open")
        assert abs(fitted.fitted_central_charge - c_true) < 0.01


# ---------------------------------------------------------------------------
# 1D chain preparation
# ---------------------------------------------------------------------------

class TestPrepare1DChainState:
    @pytest.fixture(scope="class")
    def chain_result(self):
        return prepare_1d_chain_state(4, duration_ns=1000)

    def test_returns_analog_result(self, chain_result):
        assert isinstance(chain_result, AnalogEvolutionResult)

    def test_correct_n_qubits(self, chain_result):
        assert chain_result.n_qubits == 4

    def test_normalized_sv(self, chain_result):
        norm = np.linalg.norm(chain_result.statevector)
        assert abs(norm - 1.0) < 1e-4

    def test_not_trivial(self, chain_result):
        p_ground = abs(chain_result.statevector[0]) ** 2
        assert p_ground < 0.99


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_run_entanglement_entropy_smoke(self):
        """run_entanglement_entropy completes on a small chain."""
        result = run_entanglement_entropy(
            n_atoms=4, duration_ns=1000, shots=1000, seed=SEED,
        )
        assert isinstance(result, EntanglementEntropyResult)
        assert result.n_qubits == 4
        assert result.fitted_central_charge is not None
        assert len(result.s2_entropy) == 2  # sizes 1, 2 for L=4
