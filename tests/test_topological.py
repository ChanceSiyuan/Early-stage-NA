"""Tests for src.topological — Kagome lattice, string operators, and Z2 QSL."""

import math

import numpy as np
import pytest

from src.analog import AnalogEvolutionResult
from src.simulation import (
    build_m2_circuit_from_statevector,
    build_single_copy_circuit_from_statevector,
)
from src.topological import (
    KagomeLattice,
    create_kagome_lattice,
    create_kagome_register,
    evaluate_string_operator,
    get_plaquette_qubits,
    prepare_z2_qsl,
    run_topological_vd,
)
from src.utils import (
    compute_mitigated_expval,
    compute_mitigated_string_expval,
    compute_unmitigated_expval,
    compute_unmitigated_string_expval,
    get_pauli_noise_model,
)
from qiskit_aer import AerSimulator


SEED = 42
SHOTS = 10_000


# ---------------------------------------------------------------------------
# Kagome lattice geometry
# ---------------------------------------------------------------------------

class TestKagomeLattice:
    def test_2x2_atom_count(self):
        lat = create_kagome_lattice(2, 2)
        assert lat.n_atoms == 12

    def test_3x2_atom_count(self):
        lat = create_kagome_lattice(3, 2)
        assert lat.n_atoms == 18

    def test_3x3_atom_count(self):
        lat = create_kagome_lattice(3, 3)
        assert lat.n_atoms == 27

    def test_positions_length(self):
        lat = create_kagome_lattice(2, 2)
        assert len(lat.positions) == lat.n_atoms

    def test_labels_length(self):
        lat = create_kagome_lattice(2, 2)
        assert len(lat.atom_labels) == lat.n_atoms

    def test_2x2_has_one_plaquette(self):
        lat = create_kagome_lattice(2, 2)
        assert len(lat.plaquettes) == 1

    def test_3x2_has_two_plaquettes(self):
        lat = create_kagome_lattice(3, 2)
        assert len(lat.plaquettes) == 2

    def test_3x3_has_four_plaquettes(self):
        lat = create_kagome_lattice(3, 3)
        assert len(lat.plaquettes) == 4

    def test_plaquette_has_six_atoms(self):
        lat = create_kagome_lattice(2, 2)
        for plaq in lat.plaquettes:
            assert len(plaq) == 6

    def test_plaquette_indices_valid(self):
        lat = create_kagome_lattice(2, 2)
        for plaq in lat.plaquettes:
            for idx in plaq:
                assert 0 <= idx < lat.n_atoms

    def test_plaquette_indices_distinct(self):
        lat = create_kagome_lattice(2, 2)
        for plaq in lat.plaquettes:
            assert len(set(plaq)) == 6

    def test_nearest_neighbor_distance(self):
        """Minimum inter-atom distance should equal spacing_um."""
        a = 5.5
        lat = create_kagome_lattice(2, 2, spacing_um=a)
        positions = np.array(lat.positions)
        min_dist = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        assert abs(min_dist - a) < 1e-10

    def test_sublattice_labels(self):
        lat = create_kagome_lattice(2, 2)
        sublattices = [label[2] for label in lat.atom_labels]
        assert sublattices.count("A") == 4  # 2*2
        assert sublattices.count("B") == 4
        assert sublattices.count("C") == 4


class TestKagomeRegister:
    def test_register_atom_count(self):
        lat = create_kagome_lattice(2, 2)
        reg = create_kagome_register(lat)
        assert len(reg.qubit_ids) == 12

    def test_register_positions_match(self):
        """Register positions should match lattice positions (up to centering)."""
        lat = create_kagome_lattice(2, 2, spacing_um=6.0)
        reg = create_kagome_register(lat)
        reg_coords = [reg.qubits[qid] for qid in reg.qubit_ids]
        # Pulser centers coordinates by subtracting the centroid
        lat_arr = np.array(lat.positions)
        centroid = lat_arr.mean(axis=0)
        for (lx, ly), rc in zip(lat.positions, reg_coords):
            assert abs((lx - centroid[0]) - rc[0]) < 1e-10
            assert abs((ly - centroid[1]) - rc[1]) < 1e-10


# ---------------------------------------------------------------------------
# Multi-qubit Z-string VD post-processing
# ---------------------------------------------------------------------------

class TestStringOperatorPostProcessing:
    def _run_m2_counts(self, sv, n_qubits, shots=SHOTS, seed=SEED):
        """Helper: run M=2 circuit on a pure state, return counts."""
        m2_circ = build_m2_circuit_from_statevector(sv, n_qubits)
        backend = AerSimulator(method="statevector")
        job = backend.run(m2_circ, shots=shots, seed_simulator=seed)
        return job.result().get_counts()

    def _run_single_counts(self, sv, n_qubits, shots=SHOTS, seed=SEED):
        """Helper: run single-copy circuit, return counts."""
        circ = build_single_copy_circuit_from_statevector(sv, n_qubits)
        backend = AerSimulator(method="statevector")
        job = backend.run(circ, shots=shots, seed_simulator=seed)
        return job.result().get_counts()

    def test_mitigated_single_qubit_equivalence(self):
        """compute_mitigated_string_expval([k]) matches compute_mitigated_expval(k)."""
        n = 3
        # Random-ish state
        rng = np.random.default_rng(SEED)
        sv = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        sv /= np.linalg.norm(sv)

        counts = self._run_m2_counts(sv, n)

        for k in range(n):
            single_val, _, _ = compute_mitigated_expval(counts, k, n)
            string_val, _, _ = compute_mitigated_string_expval(counts, [k], n)
            assert abs(single_val - string_val) < 1e-12, (
                f"Mismatch at qubit {k}: single={single_val}, string={string_val}"
            )

    def test_unmitigated_single_qubit_equivalence(self):
        """compute_unmitigated_string_expval([k]) matches compute_unmitigated_expval(k)."""
        n = 3
        rng = np.random.default_rng(SEED)
        sv = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        sv /= np.linalg.norm(sv)

        counts = self._run_single_counts(sv, n)

        for k in range(n):
            single_val = compute_unmitigated_expval(counts, k)
            string_val = compute_unmitigated_string_expval(counts, [k])
            assert abs(single_val - string_val) < 1e-12

    def test_ground_state_unmitigated_string(self):
        """For |000>, W = Z0*Z1*Z2 should give +1."""
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0
        counts = self._run_single_counts(sv, 3)
        w = compute_unmitigated_string_expval(counts, [0, 1, 2])
        assert abs(w - 1.0) < 1e-10

    def test_excited_state_unmitigated_string(self):
        """For |111>, W = Z0*Z1*Z2 should give (-1)^3 = -1."""
        sv = np.zeros(8, dtype=complex)
        sv[7] = 1.0  # |111>
        counts = self._run_single_counts(sv, 3)
        w = compute_unmitigated_string_expval(counts, [0, 1, 2])
        assert abs(w - (-1.0)) < 1e-10

    def test_noiseless_m2_string_consistent(self):
        """Noiseless M=2 mitigated W should closely match unmitigated W."""
        n = 3
        rng = np.random.default_rng(123)
        sv = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        sv /= np.linalg.norm(sv)

        single_counts = self._run_single_counts(sv, n, shots=50_000, seed=99)
        m2_counts = self._run_m2_counts(sv, n, shots=50_000, seed=99)

        unmit = compute_unmitigated_string_expval(single_counts, [0, 1])
        mit, _, _ = compute_mitigated_string_expval(m2_counts, [0, 1], n)
        # Noiseless: mitigated ≈ unmitigated (both estimate same quantity)
        assert abs(mit - unmit) < 0.15

    def test_empty_observable_qubits(self):
        """Empty observable list = identity operator → <W> = 1."""
        counts = {"000": 500, "111": 500}
        w = compute_unmitigated_string_expval(counts, [])
        assert abs(w - 1.0) < 1e-12

        # M=2 version: 6-bit strings for n=3
        m2_counts = {"000000": 500, "111111": 500}
        mit, _, _ = compute_mitigated_string_expval(m2_counts, [], 3)
        assert abs(mit - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Z2 QSL preparation
# ---------------------------------------------------------------------------

class TestQSLPreparation:
    @pytest.fixture(scope="class")
    def qsl_result(self):
        """Prepare QSL once for all tests in this class."""
        lat = create_kagome_lattice(2, 2, spacing_um=5.5)
        return prepare_z2_qsl(lat, duration_ns=2000)

    def test_returns_analog_result(self, qsl_result):
        assert isinstance(qsl_result, AnalogEvolutionResult)

    def test_has_statevector(self, qsl_result):
        assert qsl_result.statevector is not None

    def test_correct_n_qubits(self, qsl_result):
        assert qsl_result.n_qubits == 12

    def test_normalized_sv(self, qsl_result):
        norm = np.linalg.norm(qsl_result.statevector)
        assert abs(norm - 1.0) < 1e-4

    def test_not_trivial_state(self, qsl_result):
        """The prepared state should not be the all-ground-state."""
        # |000...0> has all amplitude on index 0
        p_ground = abs(qsl_result.statevector[0]) ** 2
        # A non-trivial QSL state should have population spread across basis states
        assert p_ground < 0.99, "State appears to be trivially ground state"


# ---------------------------------------------------------------------------
# String operator evaluation (using synthetic small states for speed)
# ---------------------------------------------------------------------------

def _make_synthetic_analog_result(n_qubits: int, seed: int = 42) -> AnalogEvolutionResult:
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


class TestEvaluateStringOperator:
    """Test evaluate_string_operator with a small synthetic state (6 qubits)."""

    @pytest.fixture(scope="class")
    def noiseless_eval(self):
        """Run noiseless string operator evaluation on a synthetic 6-qubit state."""
        analog_res = _make_synthetic_analog_result(6, seed=SEED)
        obs_qubits = [0, 1, 2, 3, 4, 5]
        return evaluate_string_operator(
            analog_res, obs_qubits, shots=SHOTS, seed=SEED,
        )

    def test_returns_valid_dict(self, noiseless_eval):
        expected_keys = {
            "observable_qubits", "n_qubits", "unmitigated_w",
            "mitigated_w", "tr_rho_sq", "shots", "source",
        }
        assert expected_keys.issubset(set(noiseless_eval.keys()))

    def test_noiseless_tr_rho2_near_one(self, noiseless_eval):
        assert abs(noiseless_eval["tr_rho_sq"] - 1.0) < 0.15

    def test_source_is_topological(self, noiseless_eval):
        assert noiseless_eval["source"] == "topological"

    def test_n_qubits_correct(self, noiseless_eval):
        assert noiseless_eval["n_qubits"] == 6

    def test_noiseless_mitigated_matches_unmitigated(self, noiseless_eval):
        """Without noise, mitigated and unmitigated should be close."""
        assert abs(noiseless_eval["mitigated_w"] - noiseless_eval["unmitigated_w"]) < 0.2


class TestEvaluateStringOperatorNoisy:
    def test_noise_degrades_string_value(self):
        """With digital noise, |<W>| should decrease."""
        analog_res = _make_synthetic_analog_result(6, seed=SEED)
        obs_qubits = [0, 1, 2]

        noiseless = evaluate_string_operator(
            analog_res, obs_qubits, shots=SHOTS, seed=SEED,
        )
        noise = get_pauli_noise_model(0.02, 0.05)
        noisy = evaluate_string_operator(
            analog_res, obs_qubits, noise_model=noise, shots=SHOTS, seed=SEED,
        )

        # Noise should push |<W>| toward zero
        assert abs(noisy["unmitigated_w"]) <= abs(noiseless["unmitigated_w"]) + 0.2


# ---------------------------------------------------------------------------
# End-to-end smoke test (synthetic state, no Pulser)
# ---------------------------------------------------------------------------

class TestTopologicalVDSynthetic:
    def test_evaluate_string_operator_smoke(self):
        """evaluate_string_operator completes on a small synthetic state."""
        analog_res = _make_synthetic_analog_result(4, seed=SEED)
        result = evaluate_string_operator(
            analog_res, [0, 1, 2, 3], shots=1000, seed=SEED,
        )
        assert "mitigated_w" in result
        assert "unmitigated_w" in result
        assert result["n_qubits"] == 4


class TestGetPlaquetteQubits:
    def test_valid_index(self):
        lat = create_kagome_lattice(2, 2)
        qubits = get_plaquette_qubits(lat, 0)
        assert len(qubits) == 6

    def test_invalid_index_raises(self):
        lat = create_kagome_lattice(2, 2)
        with pytest.raises(IndexError):
            get_plaquette_qubits(lat, 5)

    def test_negative_index_raises(self):
        lat = create_kagome_lattice(2, 2)
        with pytest.raises(IndexError):
            get_plaquette_qubits(lat, -1)
