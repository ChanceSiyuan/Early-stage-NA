# Early-stage-NA

Numerical simulation of **M=2 Virtual Distillation** (Huggins et al., 2021) for quantum error mitigation on neutral-atom Rydberg platforms. The framework spans four research phases — from gate-based VD infrastructure through topological order detection, entanglement entropy measurement, and Kibble-Zurek dynamic scaling — all built on an analog-digital hybrid architecture using Pulser/Bloqade for state preparation and Qiskit Aer for the digital measurement layer.

The M=2 protocol estimates error-mitigated expectation values via

$$\langle O \rangle_{\text{mit}} = \frac{\text{Tr}(O\,\rho^2)}{\text{Tr}(\rho^2)}$$

by performing joint measurements on two copies of a noisy quantum state.

## Setup

Requires Python >= 3.10. Uses [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync
```

## Quick Start

Four high-level functions cover the main workflows:

```python
from src.api import mitigate, topological_order, entanglement_entropy, kibble_zurek
```

### 1. Gate-based VD mitigation

```python
result = mitigate(n_qubits=5, noise=(0.01, 0.02))
print(f"Unmitigated <Z_0>: {result['unmitigated_z']:+.4f}")
print(f"Mitigated   <Z_0>: {result['mitigated_z']:+.4f}")
print(f"Tr(rho^2):         {result['tr_rho_sq']:.4f}")
```

### 2. Topological order on Kagome lattice

```python
# Single noiseless evaluation
result = topological_order(lattice_size=(2, 2))
print(f"<W_hex> mitigated: {result['mitigated_w']:+.4f}")

# Noise sweep benchmark
bench = topological_order(noise_sweep=[0.0, 0.01, 0.05, 0.10])
```

### 3. Entanglement entropy

```python
result = entanglement_entropy(n_atoms=8)
print(f"Central charge c = {result.fitted_central_charge:.3f}")
```

### 4. Kibble-Zurek scaling

```python
result = kibble_zurek(n_atoms=6, quench_durations=[500, 1000, 2000])
print(f"KZ exponent (mitigated): {result.kz_exponent_mitigated:.3f}")
print(f"KZ prediction (Ising):   {result.kz_prediction:.3f}")
```

## Per-Module API

For finer control, import directly from the submodules. All public symbols are re-exported from `src`.

### Circuit Construction & Simulation (`src.simulation`)

| Function | Description |
|---|---|
| `generate_target_circuit(n, type, ...)` | Build `near_clifford` or `local_shallow` ansatz |
| `build_m2_circuit(target)` | Construct 2N-qubit M=2 protocol circuit |
| `build_single_copy_circuit(target)` | Wrap target with Z-basis measurement |
| `build_m2_circuit_from_statevector(sv)` | M=2 circuit initialized from a statevector |
| `build_subsystem_m2_circuit_from_statevector(sv, subsystem=...)` | Partial B^(2) for subsystem purity |
| `transpile_for_method(circuit, method)` | Transpile to backend-appropriate basis gates |
| `run_simulation(target, noise, method, ...)` | End-to-end gate-based VD workflow |
| `run_analog_digital_simulation(analog_result, ...)` | M=2 from analog-evolved state (pure or mixed) |

### Analog Backends (`src.analog`)

| Symbol | Description |
|---|---|
| `AnalogEvolutionResult` | Dataclass: statevector/density_matrix + metadata |
| `create_1d_register(n, spacing)` | 1D Pulser register |
| `create_2d_register(n, layout, spacing)` | 2D square/triangular register |
| `build_rydberg_sequence(register, protocol, ...)` | Pulser sequence (`'quench'` or `'sweep'`) |
| `simulate_pulser(sequence, ...)` | Run QutipEmulator |
| `build_bloqade_program(n, ...)` | Bloqade program builder |
| `simulate_bloqade(program, ...)` | Run Bloqade emulator |
| `run_analog_evolution(n, backend, ...)` | Unified dispatch to Pulser or Bloqade |

### Noise Models (`src.neutral_atom_noise`, `src.utils`)

| Symbol | Description |
|---|---|
| `NeutralAtomNoiseParams` | Dataclass: physical noise parameters (emission, dephasing, loss, gate errors) |
| `get_pauli_noise_model(p_1q, p_2q, type, p_readout)` | Pauli-only noise for trajectory simulation |
| `get_pulser_noise_model(params)` | Lindblad noise for analog phase |
| `get_neutral_atom_digital_noise(params)` | Qiskit noise for digital B^(2) layer |
| `compose_noise_models(params)` | Both analog + digital noise as a tuple |
| `apply_atom_loss(sv, n, p_loss)` | Stochastic atom loss channel |
| `mitigate_readout_counts(counts, p_readout)` | Confusion matrix inversion for SPAM errors |

### Post-Processing (`src.utils`)

| Function | Description |
|---|---|
| `apply_b2_gate(circuit, q1, q2)` | Append B^(2) beamsplitter gate (CX-CRy(+π/2)-CX) |
| `compute_unmitigated_expval(counts, qubit)` | Raw `<Z_k>` from single-copy counts |
| `compute_mitigated_expval(counts, qubit, n)` | Mitigated `<Z_k>` via VD post-processing |
| `compute_unmitigated_string_expval(counts, qubits)` | Raw `<∏Z_k>` for multi-qubit strings |
| `compute_mitigated_string_expval(counts, qubits, n)` | Mitigated `<∏Z_k>` via VD |
| `compute_subsystem_purity(counts, subsystem, n)` | Tr(ρ_A²) from partial B^(2) |
| `compute_renyi_entropy(counts, subsystem, n)` | S_2(A) = -log₂(Tr(ρ_A²)) |

### Topological Order — Phase 2 (`src.topological`)

| Symbol | Description |
|---|---|
| `KagomeLattice` | Dataclass: positions, labels, plaquettes |
| `TopologicalBenchmarkResult` | Dataclass: noise sweep results |
| `create_kagome_lattice(rows, cols, spacing)` | Kagome geometry with plaquette enumeration |
| `create_kagome_register(lattice)` | Convert to Pulser register |
| `prepare_z2_qsl(lattice, ...)` | Adiabatic Z2 QSL preparation |
| `get_plaquette_qubits(lattice, index)` | 6 qubit indices for hexagonal plaquette |
| `evaluate_string_operator(analog_result, qubits, ...)` | Unmitigated + M=2 VD for Z-string |
| `run_topological_vd(...)` | End-to-end Kagome QSL + string operator |
| `run_topological_benchmark(...)` | Sweep digital noise levels |

### Entanglement Entropy — Phase 3 (`src.entanglement`)

| Symbol | Description |
|---|---|
| `EntanglementEntropyResult` | Dataclass: S_2 arrays, fitted central charge |
| `prepare_1d_chain_state(n, ...)` | 1D Rydberg chain at phase transition |
| `evaluate_subsystem_purity(analog_result, subsystem, ...)` | Tr(ρ_A²) and S_2(A) via M=2 VD |
| `sweep_subsystem_size(analog_result, ...)` | Sweep contiguous subsystem sizes |
| `fit_cft_scaling(result, boundary)` | Fit CFT log scaling, extract central charge |
| `run_entanglement_entropy(n_atoms, ...)` | Full pipeline with CFT fit |

### Kibble-Zurek Dynamics — Phase 4 (`src.kibble_zurek`)

| Symbol | Description |
|---|---|
| `KibbleZurekResult` | Dataclass: correlation data, fitted KZ exponents |
| `compute_correlation_function(analog_result, ...)` | Connected correlator C(r) = ⟨Z_iZ_{i+r}⟩ - ⟨Z_i⟩⟨Z_{i+r}⟩ |
| `fit_correlation_length(distances, corr)` | Extract ξ from C(r) ~ exp(-r/ξ) |
| `sweep_quench_rates(n_atoms, ...)` | Sweep quench durations, compute C(r) at each |
| `fit_kz_scaling(result, nu, z)` | Fit ξ ~ τ_Q^{ν/(1+zν)} |
| `run_kibble_zurek(n_atoms, ...)` | Full pipeline: sweep + fit |

## Benchmarks

```bash
uv run python src/benchmark.py
```

| Benchmark | Qubits (physical) | Backends | Typical runtime |
|---|---|---|---|
| N=5 | 10 | statevector, extended_stabilizer, MPS | seconds (SV/MPS), minutes (ES) |
| N=20 | 40 | MPS (full M=2), extended_stabilizer (single-copy) | ~36s (MPS) |

## Tests

```bash
uv run pytest tests/ -v
```

117 tests across all modules, ~17 seconds total.

## Project Structure

```
src/
  api.py               # High-level user-friendly API (mitigate, topological_order, ...)
  __init__.py           # Public API exports (45+ functions, 6 dataclasses)
  utils.py              # B^(2) gate, noise models, post-processing
  simulation.py         # Circuit builders, transpilation, simulation runners
  analog.py             # Pulser/Bloqade integration
  neutral_atom_noise.py # Physical noise parameters for neutral atoms
  topological.py        # Kagome lattice, Z2 QSL, string operators
  entanglement.py       # Subsystem purity, Renyi entropy, CFT fitting
  kibble_zurek.py       # Correlation functions, KZ scaling
  spam_analysis.py      # SPAM error analysis scripts
  validation.py         # Cross-validation checks
  benchmark.py          # N=5, N=20 benchmarks
tests/
  test_analog.py        # Pulser/Bloqade backend tests
  test_handoff.py       # Analog-digital handoff tests
  test_noise.py         # Noise model tests
  test_topological.py   # Kagome lattice & string operator tests
  test_entanglement.py  # Subsystem entropy tests
  test_kibble_zurek.py  # Correlation & KZ scaling tests
```

## Key Implementation Details

- **B^(2) beamsplitter gate**: Decomposes into `CX → CRy(+π/2) → CX` (the inverse transform B^(2)†). Each gate contributes ~2 non-Clifford gates in Clifford+T basis.
- **Pauli noise only**: All error channels use Pauli operators exclusively, enabling Monte Carlo trajectory sampling on MPS and extended_stabilizer backends without density-matrix fallback.
- **Qiskit bit string convention**: Big-endian — `bitstring[0]` = highest qubit. Copy 1 bits: `b1[j] = bitstring[2N-1-j]`, Copy 2 bits: `b2[j] = bitstring[N-1-j]`.
- **SWAP eigenvalue**: `s_j = -1` if `(b1[j]==0, b2[j]==1)`, else `+1`. Mitigated value is ratio of sample means.
- **Extended stabilizer limitation**: B^(2) layer adds O(N) non-Clifford gates (~2N T-gates). At N=20: ~44 T-gates, impractical for extended_stabilizer. Use MPS instead.
- **Pulser normalization**: Statevectors from Pulser numerical integration may have norm ≠ 1.0; always renormalized before `initialize()`.

## References

- W. J. Huggins *et al.*, "Virtual Distillation for Quantum Error Mitigation," *Phys. Rev. X* **11**, 041036 (2021).
- B. Koczor, "Exponential Error Suppression for Near-Term Quantum Devices," *Phys. Rev. X* **11**, 031057 (2021).
- Semeghini *et al.*, "Probing topological spin liquids on a programmable quantum simulator," *Science* **374**, 1242 (2021).
- Calabrese & Cardy, "Entanglement entropy and quantum field theory," *JSTAT* P06002 (2004).
- Keesling *et al.*, "Quantum Kibble-Zurek mechanism and critical dynamics on a programmable Rydberg simulator," *Nature* **568**, 207 (2019).
