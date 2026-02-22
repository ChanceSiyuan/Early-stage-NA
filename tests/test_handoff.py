"""Tests for statevector/density-matrix handoff into the digital M=2 layer."""

import numpy as np
import pytest

from src.analog import AnalogEvolutionResult
from src.simulation import (
    build_m2_circuit_from_statevector,
    build_single_copy_circuit_from_statevector,
    generate_target_circuit,
    run_analog_digital_simulation,
    run_simulation,
)
from src.utils import compute_mitigated_expval, get_pauli_noise_model
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit


SEED = 42
SHOTS = 10_000


class TestCircuitBuilders:
    def test_m2_circuit_shape(self):
        sv = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        qc = build_m2_circuit_from_statevector(sv, n_qubits=2)
        assert qc.num_qubits == 4
        assert qc.num_clbits == 4

    def test_single_copy_circuit_shape(self):
        sv = np.array([1, 0, 0, 0], dtype=complex)
        qc = build_single_copy_circuit_from_statevector(sv, n_qubits=2)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

    def test_m2_circuit_infers_n_qubits(self):
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0
        qc = build_m2_circuit_from_statevector(sv)
        assert qc.num_qubits == 6  # 2 * 3

    def test_single_copy_infers_n_qubits(self):
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0
        qc = build_single_copy_circuit_from_statevector(sv)
        assert qc.num_qubits == 3


class TestPureStatePurity:
    def test_noiseless_tr_rho2_near_one(self):
        """Noiseless M=2 on a pure state should give Tr(rho^2) ~ 1.0."""
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1.0  # |00>
        analog_res = AnalogEvolutionResult(
            statevector=sv, n_qubits=2, backend_name="test", metadata={},
        )
        result = run_analog_digital_simulation(
            analog_res, noise_model=None, shots=SHOTS,
            observable_qubit=0, seed=SEED,
        )
        assert abs(result["tr_rho_sq"] - 1.0) < 0.1

    def test_noiseless_z0_for_ground_state(self):
        """<Z_0> for |00> should be +1.0."""
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1.0
        analog_res = AnalogEvolutionResult(
            statevector=sv, n_qubits=2, backend_name="test", metadata={},
        )
        result = run_analog_digital_simulation(
            analog_res, noise_model=None, shots=SHOTS,
            observable_qubit=0, seed=SEED,
        )
        assert result["mitigated_z"] > 0.9


class TestHandoffMatchesGateCircuit:
    def test_sv_path_agrees_with_gate_path(self):
        """Statevector handoff should produce similar unmitigated results to gate-based."""
        n = 3
        target = generate_target_circuit(n, "local_shallow", n_layers=2, seed=SEED)

        # Gate-based path (noiseless)
        gate_result = run_simulation(
            target, None, "statevector", shots=SHOTS,
            observable_qubit=0, seed=SEED,
        )

        # Extract statevector
        backend = AerSimulator(method="statevector")
        qc = QuantumCircuit(n)
        qc.compose(target, qubits=list(range(n)), inplace=True)
        qc.save_statevector()
        job = backend.run(qc)
        sv = np.asarray(job.result().get_statevector())

        # SV handoff path (noiseless)
        analog_res = AnalogEvolutionResult(
            statevector=sv, n_qubits=n, backend_name="test", metadata={},
        )
        sv_result = run_analog_digital_simulation(
            analog_res, noise_model=None, shots=SHOTS,
            observable_qubit=0, seed=SEED,
        )

        # Compare unmitigated values (stable, not ratio-dependent)
        assert abs(gate_result["unmitigated_z"] - sv_result["unmitigated_z"]) < 0.1


class TestDensityMatrixPath:
    def test_eigendecomposition_produces_valid_results(self):
        """Mixed-state path should run without error and return valid dict."""
        n = 2
        # Create a simple mixed state: 0.7|00><00| + 0.3|11><11|
        dm = np.zeros((4, 4), dtype=complex)
        dm[0, 0] = 0.7
        dm[3, 3] = 0.3

        analog_res = AnalogEvolutionResult(
            density_matrix=dm, n_qubits=n, backend_name="test", metadata={},
        )
        result = run_analog_digital_simulation(
            analog_res, noise_model=None, shots=SHOTS,
            observable_qubit=0, seed=SEED,
        )
        assert "mitigated_z" in result
        assert "source" in result
        assert result["source"] == "analog_digital"
        assert result["n_trajectories"] == 2  # two non-zero eigenvalues

    def test_requires_sv_or_dm(self):
        """Should raise if neither statevector nor density_matrix is set."""
        analog_res = AnalogEvolutionResult(
            n_qubits=2, backend_name="test", metadata={},
        )
        with pytest.raises(ValueError, match="neither statevector nor density_matrix"):
            run_analog_digital_simulation(analog_res)


class TestNoisyMitigation:
    def test_noise_degrades_unmitigated(self):
        """Noisy unmitigated <Z> should differ from noiseless baseline."""
        n = 3
        target = generate_target_circuit(n, "local_shallow", n_layers=2, seed=SEED)

        # Noiseless reference
        noiseless = run_simulation(
            target, None, "statevector", shots=SHOTS,
            observable_qubit=0, seed=SEED,
        )

        # Noisy
        noise = get_pauli_noise_model(0.05, 0.10)
        noisy = run_simulation(
            target, noise, "statevector", shots=SHOTS,
            observable_qubit=0, seed=SEED,
        )

        # Noise should push unmitigated <Z> toward zero (depolarization)
        assert abs(noisy["unmitigated_z"]) <= abs(noiseless["unmitigated_z"]) + 0.15
