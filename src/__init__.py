"""Early-stage-NA: M=2 Virtual Distillation for Quantum Error Mitigation.

High-level API (recommended)
----------------------------
Four user-friendly entry points covering the main research workflows::

    from src.api import mitigate, topological_order, entanglement_entropy, kibble_zurek

    mitigate(n_qubits=5, noise=(0.01, 0.02))
    topological_order(lattice_size=(2, 2))
    entanglement_entropy(n_atoms=8)
    kibble_zurek(n_atoms=6)

Per-module API
--------------

Circuit construction & simulation:
    generate_target_circuit, build_m2_circuit, build_single_copy_circuit,
    transpile_for_method, run_simulation

Analog-digital hybrid:
    AnalogEvolutionResult, run_analog_evolution,
    build_m2_circuit_from_statevector, run_analog_digital_simulation

Topological order (Phase 2):
    KagomeLattice, create_kagome_lattice, prepare_z2_qsl,
    evaluate_string_operator, run_topological_vd, run_topological_benchmark

Entanglement entropy (Phase 3):
    EntanglementEntropyResult, evaluate_subsystem_purity,
    sweep_subsystem_size, fit_cft_scaling, run_entanglement_entropy

Kibble-Zurek dynamics (Phase 4):
    KibbleZurekResult, compute_correlation_function,
    fit_correlation_length, fit_kz_scaling, run_kibble_zurek

Noise models:
    get_pauli_noise_model, NeutralAtomNoiseParams, get_pulser_noise_model,
    get_neutral_atom_digital_noise, compose_noise_models

Building blocks:
    apply_b2_gate, compute_unmitigated_expval, compute_mitigated_expval,
    compute_mitigated_string_expval, compute_subsystem_purity,
    compute_renyi_entropy, mitigate_readout_counts
"""

from src.utils import (
    apply_b2_gate,
    compute_mitigated_expval,
    compute_mitigated_string_expval,
    compute_renyi_entropy,
    compute_subsystem_purity,
    compute_unmitigated_expval,
    compute_unmitigated_string_expval,
    get_pauli_noise_model,
    mitigate_readout_counts,
)
from src.simulation import (
    build_m2_circuit,
    build_m2_circuit_from_statevector,
    build_single_copy_circuit,
    build_subsystem_m2_circuit_from_statevector,
    generate_target_circuit,
    run_analog_digital_simulation,
    run_simulation,
    transpile_for_method,
)
from src.analog import (
    AnalogEvolutionResult,
    run_analog_evolution,
)
from src.neutral_atom_noise import (
    NeutralAtomNoiseParams,
    compose_noise_models,
    get_neutral_atom_digital_noise,
    get_pulser_noise_model,
)
from src.topological import (
    KagomeLattice,
    TopologicalBenchmarkResult,
    create_kagome_lattice,
    create_kagome_register,
    evaluate_string_operator,
    get_plaquette_qubits,
    prepare_z2_qsl,
    run_topological_benchmark,
    run_topological_vd,
)
from src.entanglement import (
    EntanglementEntropyResult,
    evaluate_subsystem_purity,
    fit_cft_scaling,
    prepare_1d_chain_state,
    run_entanglement_entropy,
    sweep_subsystem_size,
)
from src.kibble_zurek import (
    KibbleZurekResult,
    compute_correlation_function,
    fit_correlation_length,
    fit_kz_scaling,
    run_kibble_zurek,
    sweep_quench_rates,
)
from src.api import (
    entanglement_entropy,
    kibble_zurek,
    mitigate,
    topological_order,
)

__all__ = [
    "AnalogEvolutionResult",
    "EntanglementEntropyResult",
    "KagomeLattice",
    "KibbleZurekResult",
    "NeutralAtomNoiseParams",
    "TopologicalBenchmarkResult",
    "apply_b2_gate",
    "build_m2_circuit",
    "build_m2_circuit_from_statevector",
    "build_single_copy_circuit",
    "build_subsystem_m2_circuit_from_statevector",
    "compose_noise_models",
    "compute_correlation_function",
    "compute_mitigated_expval",
    "compute_mitigated_string_expval",
    "compute_renyi_entropy",
    "compute_subsystem_purity",
    "compute_unmitigated_expval",
    "compute_unmitigated_string_expval",
    "create_kagome_lattice",
    "create_kagome_register",
    "entanglement_entropy",
    "evaluate_string_operator",
    "evaluate_subsystem_purity",
    "fit_cft_scaling",
    "fit_correlation_length",
    "fit_kz_scaling",
    "generate_target_circuit",
    "get_neutral_atom_digital_noise",
    "get_pauli_noise_model",
    "get_plaquette_qubits",
    "get_pulser_noise_model",
    "kibble_zurek",
    "mitigate",
    "mitigate_readout_counts",
    "prepare_1d_chain_state",
    "prepare_z2_qsl",
    "run_analog_digital_simulation",
    "run_analog_evolution",
    "run_entanglement_entropy",
    "run_kibble_zurek",
    "run_simulation",
    "run_topological_benchmark",
    "run_topological_vd",
    "sweep_quench_rates",
    "sweep_subsystem_size",
    "topological_order",
    "transpile_for_method",
]
