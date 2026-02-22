"""Tests for src.analog — Pulser and Bloqade analog evolution backends."""

import numpy as np
import pytest

from src.analog import (
    AnalogEvolutionResult,
    build_bloqade_program,
    build_rydberg_sequence,
    create_1d_register,
    create_2d_register,
    run_analog_evolution,
    simulate_bloqade,
    simulate_pulser,
)


class TestRegisterCreation:
    def test_create_1d_register_atom_count(self):
        reg = create_1d_register(5, spacing_um=7.0)
        assert len(reg.qubit_ids) == 5

    def test_create_1d_register_spacing(self):
        reg = create_1d_register(3, spacing_um=10.0)
        coords = list(reg.qubits.values())
        # Adjacent atoms should be 10µm apart (Pulser centers the register)
        dx = float(coords[1][0] - coords[0][0])
        assert abs(dx - 10.0) < 1e-10

    def test_create_2d_register_square(self):
        reg = create_2d_register(4, layout="square", spacing_um=5.0)
        # 4 atoms → 2x2 square
        assert len(reg.qubit_ids) == 4

    def test_create_2d_register_triangular(self):
        reg = create_2d_register(4, layout="triangular", spacing_um=5.0)
        assert len(reg.qubit_ids) >= 4

    def test_create_2d_register_invalid_layout(self):
        with pytest.raises(ValueError, match="Unknown layout"):
            create_2d_register(4, layout="hexagonal")


class TestSequenceBuilding:
    def test_build_quench_sequence(self):
        reg = create_1d_register(3)
        seq = build_rydberg_sequence(reg, protocol="quench", duration_ns=500)
        assert seq.get_duration() >= 500

    def test_build_sweep_sequence(self):
        reg = create_1d_register(3)
        seq = build_rydberg_sequence(
            reg, protocol="sweep", omega_max=2.0,
            delta_initial=-5.0, delta_final=5.0, duration_ns=1000,
        )
        assert seq.get_duration() >= 1000

    def test_invalid_protocol(self):
        reg = create_1d_register(3)
        with pytest.raises(ValueError, match="Unknown protocol"):
            build_rydberg_sequence(reg, protocol="adiabatic")


class TestPulserSimulation:
    def test_ground_state_conversion(self):
        """Zero-amplitude quench → all ground → |000> in Qiskit convention."""
        reg = create_1d_register(3, spacing_um=7.0)
        seq = build_rydberg_sequence(
            reg, protocol="quench", omega_max=0.0, duration_ns=100,
        )
        result = simulate_pulser(seq)
        assert result.statevector is not None
        assert result.n_qubits == 3
        assert result.backend_name == "pulser"
        # All amplitude on |000> (index 0)
        assert abs(result.statevector[0]) ** 2 > 0.999

    def test_statevector_normalized(self):
        reg = create_1d_register(3, spacing_um=7.0)
        seq = build_rydberg_sequence(
            reg, protocol="quench", omega_max=2.0, duration_ns=500,
        )
        result = simulate_pulser(seq)
        norm = np.linalg.norm(result.statevector)
        assert abs(norm - 1.0) < 1e-6

    def test_basis_conversion_single_qubit(self):
        """Single atom with zero drive stays in ground = |0> in Qiskit."""
        reg = create_1d_register(1, spacing_um=7.0)
        seq = build_rydberg_sequence(
            reg, protocol="quench", omega_max=0.0, duration_ns=100,
        )
        result = simulate_pulser(seq)
        # |g> = |0> should have all amplitude
        assert abs(result.statevector[0]) ** 2 > 0.999
        assert abs(result.statevector[1]) ** 2 < 0.001


class TestBloqadeBackend:
    def test_build_program_quench(self):
        prog = build_bloqade_program(3, protocol="quench", duration_us=1.0)
        # Should not raise
        assert prog is not None

    def test_build_program_sweep(self):
        prog = build_bloqade_program(3, protocol="sweep", duration_us=1.0)
        assert prog is not None

    def test_build_program_invalid_protocol(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            build_bloqade_program(3, protocol="adiabatic")

    def test_simulate_bloqade_returns_sv(self):
        prog = build_bloqade_program(3, protocol="quench", duration_us=0.5)
        result = simulate_bloqade(prog, shots=1000)
        assert result.statevector is not None
        assert result.n_qubits == 3
        assert result.backend_name == "bloqade"
        assert result.metadata["approximate"] is True

    def test_simulate_bloqade_normalized(self):
        prog = build_bloqade_program(2, protocol="quench", duration_us=0.5)
        result = simulate_bloqade(prog, shots=5000)
        norm = np.linalg.norm(result.statevector)
        assert abs(norm - 1.0) < 1e-10


class TestConvenienceWrapper:
    def test_dispatch_pulser(self):
        result = run_analog_evolution(
            3, backend="pulser", protocol="quench",
            duration_ns=200, omega_max=0.0,
        )
        assert result.backend_name == "pulser"

    def test_dispatch_bloqade(self):
        result = run_analog_evolution(
            3, backend="bloqade", protocol="quench",
            duration_ns=500, omega_max=1.0, shots=1000,
        )
        assert result.backend_name == "bloqade"

    def test_dispatch_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            run_analog_evolution(3, backend="cirq")
