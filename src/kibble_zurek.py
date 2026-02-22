"""Kibble-Zurek dynamic phase transitions via Virtual Distillation.

Implements Phase 4 of the analog-digital VD research plan:

  Task 4.1  Quench simulation — linear detuning sweeps at varying rates
  Task 4.2  Spatial correlation function C(r) from single-copy measurements
  Task 4.3  Mitigated correlations via M=2 VD
  Task 4.4  Critical exponent extraction — ξ from C(r), KZ scaling ξ ~ τ_Q^{ν/(1+zν)}

References:
    Kibble, "Topology of cosmic domains and strings" (J. Phys. A, 1976)
    Zurek, "Cosmological experiments in superfluid helium?" (Nature, 1985)
    Keesling et al., "Quantum Kibble-Zurek mechanism and critical dynamics
        on a programmable Rydberg simulator" (Nature, 2019)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit_aer.noise import NoiseModel

from src.analog import AnalogEvolutionResult
from src.entanglement import prepare_1d_chain_state
from src.topological import evaluate_string_operator


# ---------------------------------------------------------------------------
# Task 4.2 + 4.3: Spatial correlation function
# ---------------------------------------------------------------------------

def compute_correlation_function(
    analog_result: AnalogEvolutionResult,
    noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Compute the connected spatial correlation function C(r).

    For a 1D chain of N atoms, evaluates the connected correlator::

        C(r) = <Z_i Z_{i+r}> - <Z_i><Z_{i+r}>

    averaged over all valid site pairs (i, i+r) for each distance
    r in [1, N-1].  Both unmitigated and M=2 VD-mitigated values
    are computed via :func:`~src.topological.evaluate_string_operator`.

    Args:
        analog_result: AnalogEvolutionResult with statevector.
        noise_model: Qiskit NoiseModel for digital B^(2) layer.
        shots: Measurement shots per operator evaluation.
        seed: Random seed.

    Returns:
        Dict with keys ``'n_qubits'``, ``'distances'``,
        ``'unmitigated_corr'``, ``'mitigated_corr'``,
        ``'unmitigated_zi'``, ``'mitigated_zi'``.
    """
    n = analog_result.n_qubits
    rng = np.random.default_rng(seed)

    # --- Single-site <Z_i> ---
    unmit_zi = np.zeros(n)
    mit_zi = np.zeros(n)
    for i in range(n):
        level_seed = int(rng.integers(0, 2**31))
        res = evaluate_string_operator(
            analog_result, [i], noise_model=noise_model,
            shots=shots, seed=level_seed,
        )
        unmit_zi[i] = res["unmitigated_w"]
        mit_zi[i] = res["mitigated_w"]

    # --- Two-site <Z_i Z_{i+r}> for each distance r ---
    distances = np.arange(1, n)
    unmit_corr = np.zeros(len(distances))
    mit_corr = np.zeros(len(distances))

    for idx, r in enumerate(distances):
        n_pairs = n - r
        unmit_zz_sum = 0.0
        mit_zz_sum = 0.0
        unmit_disc_sum = 0.0
        mit_disc_sum = 0.0

        for i in range(n_pairs):
            level_seed = int(rng.integers(0, 2**31))
            res = evaluate_string_operator(
                analog_result, [i, i + r], noise_model=noise_model,
                shots=shots, seed=level_seed,
            )
            unmit_zz_sum += res["unmitigated_w"]
            mit_zz_sum += res["mitigated_w"]
            unmit_disc_sum += unmit_zi[i] * unmit_zi[i + r]
            mit_disc_sum += mit_zi[i] * mit_zi[i + r]

        unmit_corr[idx] = unmit_zz_sum / n_pairs - unmit_disc_sum / n_pairs
        mit_corr[idx] = mit_zz_sum / n_pairs - mit_disc_sum / n_pairs

    return {
        "n_qubits": n,
        "distances": distances,
        "unmitigated_corr": unmit_corr,
        "mitigated_corr": mit_corr,
        "unmitigated_zi": unmit_zi,
        "mitigated_zi": mit_zi,
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KibbleZurekResult:
    """Results from a Kibble-Zurek quench rate sweep.

    Attributes:
        n_qubits: Chain length.
        quench_durations_ns: Array of sweep durations (τ_Q).
        distances: Array of spatial distances r.
        unmitigated_corr: 2D array [n_durations, n_distances] of C(r).
        mitigated_corr: 2D array [n_durations, n_distances] of C(r).
        unmitigated_xi: Fitted correlation lengths (unmitigated).
        mitigated_xi: Fitted correlation lengths (mitigated).
        kz_exponent_unmitigated: Fitted KZ exponent (unmitigated).
        kz_exponent_mitigated: Fitted KZ exponent (mitigated).
        kz_prediction: Theoretical KZ exponent ν/(1+zν).
        metadata: Additional info.
    """

    n_qubits: int
    quench_durations_ns: np.ndarray
    distances: np.ndarray
    unmitigated_corr: np.ndarray
    mitigated_corr: np.ndarray
    unmitigated_xi: np.ndarray | None = None
    mitigated_xi: np.ndarray | None = None
    kz_exponent_unmitigated: float | None = None
    kz_exponent_mitigated: float | None = None
    kz_prediction: float | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task 4.4: Correlation length extraction
# ---------------------------------------------------------------------------

def fit_correlation_length(
    distances: np.ndarray,
    corr: np.ndarray,
    min_r: int = 1,
) -> float:
    """Extract correlation length ξ from C(r) ~ A exp(-r/ξ).

    Fits ``log|C(r)| = log|A| - r/ξ`` via linear regression.

    Args:
        distances: Array of spatial distances.
        corr: Array of C(r) values.
        min_r: Minimum distance to include in fit.

    Returns:
        Fitted correlation length ξ.  Returns ``nan`` if the fit fails.
    """
    mask = (distances >= min_r) & (np.abs(corr) > 0)
    if mask.sum() < 2:
        return float("nan")

    x = distances[mask].astype(float)
    y = np.log(np.abs(corr[mask]))

    try:
        slope, _ = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")

    if slope >= 0:
        return float("nan")

    return -1.0 / slope


# ---------------------------------------------------------------------------
# Task 4.1: Quench rate sweep
# ---------------------------------------------------------------------------

DEFAULT_QUENCH_DURATIONS_NS = [500, 1000, 1500, 2000, 3000, 4000]


def sweep_quench_rates(
    n_atoms: int = 8,
    spacing_um: float = 7.0,
    omega_max: float = 2.0,
    delta_initial: float = -5.0,
    delta_final: float = 5.0,
    quench_durations_ns: list[int] | np.ndarray | None = None,
    noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
) -> KibbleZurekResult:
    """Sweep quench rates and compute correlation functions at each rate.

    For each duration in *quench_durations_ns*:

    1. Prepare 1D chain state via adiabatic detuning sweep.
    2. Compute full correlation function C(r).

    Args:
        n_atoms: Number of atoms in the chain.
        spacing_um: Inter-atom spacing in μm.
        omega_max: Maximum Rabi frequency in rad/μs.
        delta_initial: Initial detuning in rad/μs.
        delta_final: Final detuning in rad/μs.
        quench_durations_ns: Sweep durations to scan.
        noise_model: Qiskit NoiseModel for digital layer.
        shots: Measurement shots per correlation evaluation.
        seed: Random seed.

    Returns:
        KibbleZurekResult with correlation data at each quench rate.
    """
    if quench_durations_ns is None:
        quench_durations_ns = DEFAULT_QUENCH_DURATIONS_NS
    durations = np.asarray(quench_durations_ns, dtype=float)

    rng = np.random.default_rng(seed)
    distances = np.arange(1, n_atoms)

    unmit_corr = np.zeros((len(durations), len(distances)))
    mit_corr = np.zeros((len(durations), len(distances)))

    for idx, dur in enumerate(durations):
        level_seed = int(rng.integers(0, 2**31))
        analog_result = prepare_1d_chain_state(
            n_atoms, spacing_um, omega_max,
            delta_initial, delta_final, duration_ns=int(dur),
        )
        corr = compute_correlation_function(
            analog_result, noise_model=noise_model,
            shots=shots, seed=level_seed,
        )
        unmit_corr[idx] = corr["unmitigated_corr"]
        mit_corr[idx] = corr["mitigated_corr"]

    return KibbleZurekResult(
        n_qubits=n_atoms,
        quench_durations_ns=durations,
        distances=distances,
        unmitigated_corr=unmit_corr,
        mitigated_corr=mit_corr,
        metadata={
            "spacing_um": spacing_um,
            "omega_max": omega_max,
            "delta_initial": delta_initial,
            "delta_final": delta_final,
        },
    )


# ---------------------------------------------------------------------------
# Task 4.4: KZ scaling fit
# ---------------------------------------------------------------------------

def fit_kz_scaling(
    result: KibbleZurekResult,
    nu: float = 1.0,
    z: float = 1.0,
) -> KibbleZurekResult:
    """Fit Kibble-Zurek scaling ξ ~ τ_Q^{ν/(1+zν)}.

    1. Extract ξ from C(r) at each quench rate.
    2. Fit log(ξ) vs log(τ_Q) via linear regression.
    3. Compare fitted exponent to theoretical prediction.

    For 1D Ising universality (ν=1, z=1): predicted exponent = 1/2.

    Args:
        result: KibbleZurekResult from :func:`sweep_quench_rates`.
        nu: Critical exponent ν (default 1.0 for Ising).
        z: Dynamic critical exponent z (default 1.0 for Ising).

    Returns:
        Updated result with ξ arrays and KZ exponents populated.
    """
    n_dur = len(result.quench_durations_ns)
    unmit_xi = np.full(n_dur, np.nan)
    mit_xi = np.full(n_dur, np.nan)

    for idx in range(n_dur):
        unmit_xi[idx] = fit_correlation_length(
            result.distances, result.unmitigated_corr[idx],
        )
        mit_xi[idx] = fit_correlation_length(
            result.distances, result.mitigated_corr[idx],
        )

    result.unmitigated_xi = unmit_xi
    result.mitigated_xi = mit_xi
    result.kz_prediction = nu / (1.0 + z * nu)

    # Fit KZ exponent: log(ξ) = α * log(τ_Q) + const
    durations = result.quench_durations_ns

    for label, xi_arr, attr in [
        ("unmitigated", unmit_xi, "kz_exponent_unmitigated"),
        ("mitigated", mit_xi, "kz_exponent_mitigated"),
    ]:
        valid = np.isfinite(xi_arr) & (xi_arr > 0)
        if valid.sum() >= 2:
            try:
                slope, _ = np.polyfit(
                    np.log(durations[valid]), np.log(xi_arr[valid]), 1,
                )
                setattr(result, attr, slope)
            except (np.linalg.LinAlgError, ValueError):
                pass

    return result


# ---------------------------------------------------------------------------
# End-to-end convenience pipeline
# ---------------------------------------------------------------------------

def run_kibble_zurek(
    n_atoms: int = 8,
    spacing_um: float = 7.0,
    omega_max: float = 2.0,
    delta_initial: float = -5.0,
    delta_final: float = 5.0,
    quench_durations_ns: list[int] | np.ndarray | None = None,
    noise_model: NoiseModel | None = None,
    shots: int = 10_000,
    seed: int | None = None,
    nu: float = 1.0,
    z: float = 1.0,
) -> KibbleZurekResult:
    """Run the full Kibble-Zurek pipeline.

    1. Sweep quench rates, computing C(r) at each.
    2. Fit correlation lengths and KZ scaling.

    Args:
        n_atoms: Number of atoms in the chain.
        spacing_um: Inter-atom spacing in μm.
        omega_max: Maximum Rabi frequency in rad/μs.
        delta_initial: Initial detuning in rad/μs.
        delta_final: Final detuning in rad/μs.
        quench_durations_ns: Sweep durations to scan.
        noise_model: Qiskit NoiseModel for digital layer.
        shots: Measurement shots per correlation evaluation.
        seed: Random seed.
        nu: Critical exponent ν.
        z: Dynamic critical exponent z.

    Returns:
        KibbleZurekResult with fitted KZ exponents.
    """
    result = sweep_quench_rates(
        n_atoms, spacing_um, omega_max,
        delta_initial, delta_final, quench_durations_ns,
        noise_model, shots, seed,
    )
    return fit_kz_scaling(result, nu, z)


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_kz_results(result: KibbleZurekResult) -> None:
    """Pretty-print a KibbleZurekResult."""
    L = result.n_qubits
    print(
        f"\n{'=' * 70}\n"
        f"Kibble-Zurek Quench Sweep:  {L}-atom 1D chain\n"
        f"{'=' * 70}"
    )

    # Correlation length table
    if result.unmitigated_xi is not None:
        print(f"\n{'tau_Q (ns)':>12s}  {'xi_unmit':>10s}  {'xi_mit':>10s}")
        print("-" * 40)
        for i, dur in enumerate(result.quench_durations_ns):
            u_xi = result.unmitigated_xi[i]
            m_xi = result.mitigated_xi[i]
            u_str = f"{u_xi:10.4f}" if np.isfinite(u_xi) else "       nan"
            m_str = f"{m_xi:10.4f}" if np.isfinite(m_xi) else "       nan"
            print(f"{dur:12.0f}  {u_str}  {m_str}")

    # KZ exponents
    print()
    pred = result.kz_prediction
    if pred is not None:
        print(f"KZ prediction (nu/(1+z*nu)):  {pred:.4f}")
    if result.kz_exponent_unmitigated is not None:
        print(f"Fitted exponent (unmitigated): {result.kz_exponent_unmitigated:.4f}")
    if result.kz_exponent_mitigated is not None:
        print(f"Fitted exponent (mitigated):   {result.kz_exponent_mitigated:.4f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Phase 4 Kibble-Zurek pipeline."""
    print("Phase 4: Kibble-Zurek Dynamic Phase Transitions via VD")
    print("Preparing 1D chain sweeps (8 atoms)...")

    result = run_kibble_zurek(n_atoms=8, shots=10_000, seed=42)
    print_kz_results(result)


if __name__ == "__main__":
    main()
