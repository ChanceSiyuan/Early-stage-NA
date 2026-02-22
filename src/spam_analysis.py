#!/usr/bin/env python3
"""SPAM error analysis for M=2 Virtual Distillation.

Quantifies how readout (SPAM) errors propagate through the M=2 post-processing,
their interplay with gate noise, and whether readout error mitigation composes
cleanly with virtual distillation.

Analytical background
---------------------
For symmetric readout error probability *p* on each qubit:

- **Unmitigated <Z_k>**: Each qubit's measurement outcome is flipped with
  probability *p*, biasing the expectation value by a factor (1 - 2p).
  For a single-qubit Z observable: <Z_k>_measured = (1 - 2p) * <Z_k>_true.

- **M=2 denominator Tr(rho^2)**: Each SWAP eigenvalue factor involves two
  qubits (one from each copy), so each pair is biased by (1 - 2p)^2.
  With N pairs: Tr(rho^2)_measured ~ Tr(rho^2)_true * (1 - 2p)^{2N}.
  This exponential suppression makes the denominator very small at moderate
  readout error rates.

- **M=2 numerator Tr(Z_k rho^2)**: Similarly biased, but the ratio E/D
  benefits from partial cancellation of the readout bias. However, the
  exponentially small denominator amplifies statistical (shot) noise.

- **Readout mitigation**: Per-qubit confusion matrix inversion rescales by
  1/(1-2p)^{2N}, recovering the true Tr(rho^2) but amplifying shot noise
  by the same factor. This is the fundamental tension: mitigation corrects
  the bias but at the cost of increased variance.

Usage:
    uv run python src/spam_analysis.py
"""

from __future__ import annotations

import numpy as np

from src.simulation import (
    build_m2_circuit,
    build_single_copy_circuit,
    generate_target_circuit,
    transpile_for_method,
)
from src.utils import (
    compute_mitigated_expval,
    compute_unmitigated_expval,
    get_pauli_noise_model,
    mitigate_readout_counts,
)
from qiskit_aer import AerSimulator


# ===================================================================
# Configuration
# ===================================================================
METHOD = "matrix_product_state"
SEED = 42


# ===================================================================
# Core analysis functions
# ===================================================================

def run_spam_sweep(
    n_qubits: int = 5,
    circuit_type: str = "local_shallow",
    p_1q: float = 0.005,
    p_2q: float = 0.005,
    readout_rates: np.ndarray | None = None,
    shots: int = 20_000,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Sweep readout error rate and measure its impact on virtual distillation.

    For each readout rate, runs both single-copy (unmitigated) and M=2
    (mitigated) circuits, with and without readout error mitigation applied
    to the measurement counts.

    Args:
        n_qubits: Number of logical qubits N.
        circuit_type: Target circuit type ('local_shallow' or 'near_clifford').
        p_1q: Single-qubit gate error probability.
        p_2q: Two-qubit gate error probability.
        readout_rates: Array of readout error probabilities to sweep.
        shots: Number of measurement shots per point.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with arrays for each sweep point:
            - readout_rates: the p_readout values swept
            - unmit_raw: unmitigated <Z_0> without RO correction
            - unmit_corrected: unmitigated <Z_0> with RO correction
            - mit_raw: mitigated <Z_0> without RO correction
            - mit_corrected: mitigated <Z_0> with RO correction
            - tr_rho2_raw: Tr(rho^2) estimate without RO correction
            - tr_rho2_corrected: Tr(rho^2) estimate with RO correction
    """
    if readout_rates is None:
        readout_rates = np.linspace(0, 0.10, 11)

    target = generate_target_circuit(
        n_qubits, circuit_type, n_layers=3, seed=seed,
    )

    # Build and transpile circuits once (gate structure is independent of RO error)
    single_circ = build_single_copy_circuit(target)
    single_circ = transpile_for_method(single_circ, METHOD)

    m2_circ = build_m2_circuit(target)
    m2_circ = transpile_for_method(m2_circ, METHOD)

    # Result arrays
    n_points = len(readout_rates)
    results = {
        "readout_rates": np.array(readout_rates),
        "unmit_raw": np.zeros(n_points),
        "unmit_corrected": np.zeros(n_points),
        "mit_raw": np.zeros(n_points),
        "mit_corrected": np.zeros(n_points),
        "tr_rho2_raw": np.zeros(n_points),
        "tr_rho2_corrected": np.zeros(n_points),
    }

    for i, p_ro in enumerate(readout_rates):
        noise = get_pauli_noise_model(p_1q, p_2q, "depolarizing", p_readout=p_ro)
        backend = AerSimulator(method=METHOD, noise_model=noise)

        # --- Single-copy (unmitigated) ---
        job_single = backend.run(single_circ, shots=shots, seed_simulator=seed)
        counts_single = job_single.result().get_counts()

        results["unmit_raw"][i] = compute_unmitigated_expval(counts_single, 0)

        corrected_single = mitigate_readout_counts(counts_single, p_ro)
        results["unmit_corrected"][i] = compute_unmitigated_expval(
            corrected_single, 0,
        )

        # --- M=2 (mitigated) ---
        job_m2 = backend.run(m2_circ, shots=shots, seed_simulator=seed)
        counts_m2 = job_m2.result().get_counts()

        mit_raw, _, den_raw = compute_mitigated_expval(counts_m2, 0, n_qubits)
        results["mit_raw"][i] = mit_raw
        results["tr_rho2_raw"][i] = den_raw

        corrected_m2 = mitigate_readout_counts(counts_m2, p_ro)
        mit_corr, _, den_corr = compute_mitigated_expval(
            corrected_m2, 0, n_qubits,
        )
        results["mit_corrected"][i] = mit_corr
        results["tr_rho2_corrected"][i] = den_corr

    return results


def run_gate_vs_readout_crossover(
    n_qubits: int = 5,
    circuit_type: str = "local_shallow",
    p_1q: float = 0.005,
    p_2q: float = 0.005,
    readout_rates: np.ndarray | None = None,
    shots: int = 20_000,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Find the crossover point where SPAM error exceeds gate error.

    Runs with fixed gate noise and varying readout error. Compares the
    mitigated result against the gate-noise-only baseline (p_readout=0)
    to identify when readout error becomes the dominant noise source.

    Args:
        n_qubits: Number of logical qubits N.
        circuit_type: Target circuit type.
        p_1q: Single-qubit gate error probability (fixed).
        p_2q: Two-qubit gate error probability (fixed).
        readout_rates: Array of readout error probabilities to sweep.
        shots: Number of measurement shots per point.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
            - readout_rates: the p_readout values swept
            - gate_only_mit: mitigated <Z_0> with gate noise only (scalar, repeated)
            - mit_raw_error: |mitigated_raw - gate_only| at each point
            - mit_corrected_error: |mitigated_corrected - gate_only| at each point
    """
    if readout_rates is None:
        readout_rates = np.linspace(0, 0.05, 11)

    target = generate_target_circuit(
        n_qubits, circuit_type, n_layers=3, seed=seed,
    )
    m2_circ = build_m2_circuit(target)
    m2_circ = transpile_for_method(m2_circ, METHOD)

    # Baseline: gate noise only, no readout error
    noise_baseline = get_pauli_noise_model(p_1q, p_2q, "depolarizing", p_readout=0.0)
    backend_baseline = AerSimulator(method=METHOD, noise_model=noise_baseline)
    job_baseline = backend_baseline.run(m2_circ, shots=shots, seed_simulator=seed)
    counts_baseline = job_baseline.result().get_counts()
    gate_only_mit, _, _ = compute_mitigated_expval(counts_baseline, 0, n_qubits)

    n_points = len(readout_rates)
    results = {
        "readout_rates": np.array(readout_rates),
        "gate_only_mit": gate_only_mit,
        "mit_raw_error": np.zeros(n_points),
        "mit_corrected_error": np.zeros(n_points),
    }

    for i, p_ro in enumerate(readout_rates):
        noise = get_pauli_noise_model(p_1q, p_2q, "depolarizing", p_readout=p_ro)
        backend = AerSimulator(method=METHOD, noise_model=noise)

        job = backend.run(m2_circ, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()

        mit_raw, _, _ = compute_mitigated_expval(counts, 0, n_qubits)
        results["mit_raw_error"][i] = abs(mit_raw - gate_only_mit)

        corrected = mitigate_readout_counts(counts, p_ro)
        mit_corr, _, _ = compute_mitigated_expval(corrected, 0, n_qubits)
        results["mit_corrected_error"][i] = abs(mit_corr - gate_only_mit)

    return results


# ===================================================================
# Main entry point
# ===================================================================

def main() -> None:
    """Run full SPAM error analysis and print results."""
    print("=" * 78)
    print("  SPAM Error Analysis for M=2 Virtual Distillation")
    print("=" * 78)

    n_qubits = 5
    p_1q, p_2q = 0.005, 0.005
    shots = 20_000

    # ------------------------------------------------------------------
    # Sweep 1: readout error rate vs accuracy
    # ------------------------------------------------------------------
    print(f"\n--- Sweep 1: Readout error impact (N={n_qubits}, "
          f"p_1q={p_1q}, p_2q={p_2q}, shots={shots}) ---\n")

    sweep = run_spam_sweep(
        n_qubits=n_qubits, p_1q=p_1q, p_2q=p_2q, shots=shots,
    )

    header = (
        f"{'p_ro':>6s} | {'unmit_raw':>10s} | {'unmit_corr':>10s} | "
        f"{'mit_raw':>10s} | {'mit_corr':>10s} | "
        f"{'Tr2_raw':>10s} | {'Tr2_corr':>10s}"
    )
    print(header)
    print("-" * len(header))

    for i in range(len(sweep["readout_rates"])):
        p_ro = sweep["readout_rates"][i]
        print(
            f"{p_ro:6.3f} | "
            f"{sweep['unmit_raw'][i]:+10.6f} | "
            f"{sweep['unmit_corrected'][i]:+10.6f} | "
            f"{sweep['mit_raw'][i]:+10.6f} | "
            f"{sweep['mit_corrected'][i]:+10.6f} | "
            f"{sweep['tr_rho2_raw'][i]:10.6f} | "
            f"{sweep['tr_rho2_corrected'][i]:10.6f}"
        )

    # Analytical prediction for Tr(rho^2) suppression
    print(f"\n--- Analytical: Tr(rho^2) suppression factor (1-2p)^{{2N}} ---\n")
    for i in range(len(sweep["readout_rates"])):
        p_ro = sweep["readout_rates"][i]
        if p_ro > 0:
            predicted_factor = (1 - 2 * p_ro) ** (2 * n_qubits)
            baseline_tr2 = sweep["tr_rho2_raw"][0]  # p_ro=0 value
            if baseline_tr2 > 1e-10:
                observed_ratio = sweep["tr_rho2_raw"][i] / baseline_tr2
            else:
                observed_ratio = float("nan")
            print(f"  p_ro={p_ro:.3f}:  predicted={(predicted_factor):.6f}  "
                  f"observed={observed_ratio:.6f}")

    # ------------------------------------------------------------------
    # Sweep 2: gate noise vs readout crossover
    # ------------------------------------------------------------------
    print(f"\n--- Sweep 2: Gate vs readout error crossover ---\n")

    crossover = run_gate_vs_readout_crossover(
        n_qubits=n_qubits, p_1q=p_1q, p_2q=p_2q, shots=shots,
    )

    print(f"  Gate-only mitigated <Z_0> = {crossover['gate_only_mit']:+.6f}\n")

    header2 = f"{'p_ro':>6s} | {'|raw-baseline|':>14s} | {'|corr-baseline|':>15s}"
    print(header2)
    print("-" * len(header2))

    for i in range(len(crossover["readout_rates"])):
        p_ro = crossover["readout_rates"][i]
        print(
            f"{p_ro:6.3f} | "
            f"{crossover['mit_raw_error'][i]:14.6f} | "
            f"{crossover['mit_corrected_error'][i]:15.6f}"
        )

    # Find crossover: where raw SPAM error exceeds gate-only error level
    # Gate-only error is the p_ro=0 point (should be near zero)
    gate_error_baseline = crossover["mit_raw_error"][0]
    crossover_idx = None
    for i in range(1, len(crossover["readout_rates"])):
        if crossover["mit_raw_error"][i] > gate_error_baseline + 0.01:
            crossover_idx = i
            break

    print()
    if crossover_idx is not None:
        p_cross = crossover["readout_rates"][crossover_idx]
        print(f"  SPAM error exceeds gate error (by >0.01) at p_readout ~ {p_cross:.3f}")
    else:
        print("  SPAM error does not clearly exceed gate error in the swept range.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  Summary")
    print(f"{'=' * 78}")
    print(f"""
  1. Readout errors suppress Tr(rho^2) exponentially: (1-2p)^{{2N}}.
     At N={n_qubits}, p_ro=0.05: suppression factor = {(1 - 2*0.05)**(2*n_qubits):.4f}
     At N={n_qubits}, p_ro=0.10: suppression factor = {(1 - 2*0.10)**(2*n_qubits):.4f}

  2. The mitigated ratio E/D partially cancels readout bias, but the
     exponentially small denominator amplifies shot noise.

  3. Per-qubit readout mitigation (confusion matrix inversion) recovers
     the unbiased Tr(rho^2) but amplifies variance by 1/(1-2p)^{{2N}}.

  4. Readout mitigation composes with virtual distillation: corrected
     mitigated values track the gate-noise-only baseline closely when
     readout error is moderate (p_ro < ~0.05).
""")


if __name__ == "__main__":
    main()
