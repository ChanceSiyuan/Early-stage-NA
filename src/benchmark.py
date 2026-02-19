#!/usr/bin/env python3
"""Benchmarking script for M=2 Virtual Distillation error mitigation.

Runs two benchmarks:
  1. N=5  (10 physical qubits) -- statevector & MPS for consistency,
     extended_stabilizer with reduced shots (expensive due to T-gate scaling)
  2. N=20 (40 physical qubits) -- MPS full M=2 protocol (demonstrating
     OOM-free scalability), extended_stabilizer single-copy only (the M=2
     circuit's B^(2) layer adds ~2N non-Clifford gates, making runtime
     exponential in N for this backend)

Usage:
    uv run python src/benchmark.py
"""

from __future__ import annotations

import time

from qiskit.circuit import QuantumCircuit
from qiskit import transpile

from src.simulation import (
    build_m2_circuit,
    build_single_copy_circuit,
    generate_target_circuit,
    run_simulation,
    transpile_for_method,
)
from src.utils import (
    compute_unmitigated_expval,
    get_pauli_noise_model,
)
from qiskit_aer import AerSimulator


# ===================================================================
# Configuration
# ===================================================================
P_1Q = 0.01          # 1 % single-qubit depolarising error rate
P_2Q = 0.02          # 2 % two-qubit depolarising error rate
NOISE_TYPE = "depolarizing"
SHOTS = 10_000
SEED = 42


def _header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _print_result(result: dict, elapsed: float, circuit_type: str) -> None:
    print(f"  Method          : {result['method']}")
    print(f"  Circuit type    : {circuit_type}")
    print(f"  N (logical)     : {result['n_qubits']}")
    print(f"  Shots           : {result['shots']}")
    print(f"  Unmitigated <Z0>: {result['unmitigated_z']:+.6f}")
    print(f"  Mitigated   <Z0>: {result['mitigated_z']:+.6f}")
    print(f"  Tr(rho^2)       : {result['tr_rho_sq']:.6f}")
    print(f"  Wall time       : {elapsed:.1f} s")
    print()


def _count_non_clifford(circuit: QuantumCircuit) -> int:
    """Count T + Tdg gates in a circuit (proxy for non-Clifford cost)."""
    ops = circuit.count_ops()
    return ops.get("t", 0) + ops.get("tdg", 0)


def benchmark_small(noise_model) -> None:
    """Benchmark 1: N=5 (10 physical qubits), all three methods.

    - statevector and MPS run at full shots (fast).
    - extended_stabilizer uses fewer shots because each shot is expensive:
      the M=2 circuit transpiles to ~18 T-gates for N=5, and the
      extended_stabilizer runtime scales as O(2^{alpha*t}) per shot.
    """
    _header("BENCHMARK 1: N=5 (10 physical qubits) -- All three methods")

    # Show the T-gate cost for extended_stabilizer
    target_nc = generate_target_circuit(5, "near_clifford", max_t_gates=4, seed=SEED)
    m2_nc = build_m2_circuit(target_nc)
    m2_nc_t = transpile_for_method(m2_nc, "extended_stabilizer")
    t_count = _count_non_clifford(m2_nc_t)
    print(f"\n  [info] M=2 circuit (N=5, near_clifford) transpiles to {t_count} T+Tdg gates")
    print(f"         extended_stabilizer runtime ~ O(2^{{0.5*{t_count}}}) per shot\n")

    configs = [
        # (method,                  circuit_type,   shots)
        ("statevector",             "local_shallow", SHOTS),
        ("matrix_product_state",    "local_shallow", SHOTS),
        ("extended_stabilizer",     "near_clifford", 500),
    ]

    for method, circuit_type, method_shots in configs:
        target = generate_target_circuit(
            5, circuit_type, max_t_gates=4, n_layers=3, seed=SEED,
        )

        t0 = time.perf_counter()
        result = run_simulation(
            target, noise_model, method,
            shots=method_shots, observable_qubit=0, seed=SEED,
        )
        elapsed = time.perf_counter() - t0

        _print_result(result, elapsed, circuit_type)


def benchmark_large(noise_model) -> None:
    """Benchmark 2: N=20 (40 physical qubits), scalable methods.

    Statevector is excluded: it would need 2^40 complex amplitudes (~8 TB).

    matrix_product_state
        Runs the full M=2 protocol (40 physical qubits) with 10k shots.
        MPS handles this efficiently because the shallow 1-D circuit keeps
        the bond dimension bounded.

    extended_stabilizer
        Runs only the single-copy (unmitigated) circuit at N=20.  The M=2
        circuit at N=20 would have ~40 non-Clifford gates from the B^(2)
        measurement layer alone, making per-shot cost O(2^20) -- hours per
        shot.  This is a fundamental limitation: the B^(2) beamsplitter
        introduces O(N) non-Clifford gates, making extended_stabilizer
        unsuitable for the M=2 protocol at large N.  However, the
        single-copy circuit (with 2-4 T gates) runs efficiently and
        demonstrates that extended_stabilizer avoids OOM at 20 qubits.
    """
    _header("BENCHMARK 2: N=20 (40 physical qubits) -- Scalable methods")

    # --- MPS: full M=2 protocol ---
    print("  --- matrix_product_state: full M=2 protocol ---\n")
    target_mps = generate_target_circuit(
        20, "local_shallow", n_layers=2, seed=SEED,
    )

    t0 = time.perf_counter()
    result = run_simulation(
        target_mps, noise_model, "matrix_product_state",
        shots=SHOTS, observable_qubit=0, seed=SEED,
    )
    elapsed = time.perf_counter() - t0
    _print_result(result, elapsed, "local_shallow")

    # --- Extended Stabilizer: single-copy only (M=2 too expensive) ---
    print("  --- extended_stabilizer: single-copy only (OOM avoidance demo) ---\n")
    target_es = generate_target_circuit(
        20, "near_clifford", max_t_gates=2, n_layers=5, seed=SEED,
    )

    single_circ = build_single_copy_circuit(target_es)
    single_circ_t = transpile_for_method(single_circ, "extended_stabilizer")
    t_count_single = _count_non_clifford(single_circ_t)
    print(f"  [info] Single-copy circuit (N=20): {t_count_single} T+Tdg gates")

    # Also show what the M=2 circuit would cost
    m2_circ = build_m2_circuit(target_es)
    m2_circ_t = transpile_for_method(m2_circ, "extended_stabilizer")
    t_count_m2 = _count_non_clifford(m2_circ_t)
    print(f"  [info] M=2 circuit (N=20, 40 qubits): {t_count_m2} T+Tdg gates")
    print(f"         (skipped -- per-shot cost O(2^{{{t_count_m2 // 2}}}) is impractical)\n")

    backend = AerSimulator(method="extended_stabilizer", noise_model=noise_model)
    t0 = time.perf_counter()
    try:
        job = backend.run(single_circ_t, shots=SHOTS, seed_simulator=SEED)
        counts = job.result().get_counts()
        unmit = compute_unmitigated_expval(counts, observable_qubit=0)
        elapsed = time.perf_counter() - t0

        print(f"  Method          : extended_stabilizer (single-copy)")
        print(f"  Circuit type    : near_clifford")
        print(f"  N (logical)     : 20")
        print(f"  Shots           : {SHOTS}")
        print(f"  Unmitigated <Z0>: {unmit:+.6f}")
        print(f"  (M=2 mitigated skipped -- see [info] above)")
        print(f"  Wall time       : {elapsed:.1f} s")
        print()
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  extended_stabilizer FAILED after {elapsed:.1f} s")
        print(f"  Error: {exc}")
        print()


def main() -> None:
    """Entry point."""
    print("M=2 Virtual Distillation -- Benchmark Suite")
    print(f"Noise: {NOISE_TYPE}  p_1q={P_1Q}  p_2q={P_2Q}  shots={SHOTS}")

    noise_model = get_pauli_noise_model(P_1Q, P_2Q, NOISE_TYPE)

    benchmark_small(noise_model)
    benchmark_large(noise_model)

    _header("Benchmarking complete")


if __name__ == "__main__":
    main()
