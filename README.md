# Early-stage-NA

Numerical simulation of the **M=2 Virtual Distillation** (Huggins et al., 2021) and **Error Suppression by Derangement** (Koczor, 2020) protocols for quantum error mitigation, targeting neutral atom array architectures.

The M=2 protocol estimates error-mitigated expectation values via

$$\langle O \rangle_{\text{mit}} = \frac{\text{Tr}(O\,\rho^2)}{\text{Tr}(\rho^2)}$$

by performing joint measurements on two copies of a noisy quantum state. This repository provides a modular Qiskit + Qiskit Aer simulation framework that compares three backends — `statevector`, `extended_stabilizer`, and `matrix_product_state` — with Pauli noise driven by Monte Carlo trajectory sampling to avoid density-matrix memory walls.

## Setup

Requires Python >= 3.10. Uses [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync
```

## Quick Start

```python
from src import run_simulation, generate_target_circuit, get_pauli_noise_model

# Generate a 5-qubit shallow ansatz circuit
circuit = generate_target_circuit(5, "local_shallow", seed=42)

# 1% single-qubit, 2% two-qubit depolarising noise
noise = get_pauli_noise_model(0.01, 0.02)

# Run M=2 protocol on MPS backend
result = run_simulation(circuit, noise, "matrix_product_state", shots=10000)

print(f"Unmitigated <Z_0>: {result['unmitigated_z']:+.4f}")
print(f"Mitigated   <Z_0>: {result['mitigated_z']:+.4f}")
print(f"Tr(rho^2):         {result['tr_rho_sq']:.4f}")
```

## Benchmarks

Run the full benchmark suite (N=5 all backends, N=20 scalable backends):

```bash
uv run python src/benchmark.py
```

| Benchmark | Qubits (physical) | Backends | Typical runtime |
|---|---|---|---|
| N=5 | 10 | statevector, extended_stabilizer, MPS | seconds (SV/MPS), minutes (ES) |
| N=20 | 40 | MPS (full M=2), extended_stabilizer (single-copy) | ~36s (MPS) |

The `matrix_product_state` backend handles the full 40-qubit M=2 protocol without OOM. The `extended_stabilizer` backend is limited at large N because the B^(2) measurement layer introduces O(N) non-Clifford gates.

## API Reference

### Circuit Construction

| Function | Description |
|---|---|
| `generate_target_circuit(n, type, ...)` | Build `near_clifford` or `local_shallow` ansatz |
| `build_m2_circuit(target)` | Construct 2N-qubit M=2 protocol circuit |
| `build_single_copy_circuit(target)` | Wrap target with Z-basis measurement |
| `transpile_for_method(circuit, method)` | Transpile to backend-appropriate basis gates |

### Noise & Post-Processing

| Function | Description |
|---|---|
| `get_pauli_noise_model(p_1q, p_2q, type)` | Pauli-only noise model (depolarizing, bit_flip, phase_flip, bit_phase_flip) |
| `compute_unmitigated_expval(counts, qubit)` | Raw `<Z_k>` from single-copy counts |
| `compute_mitigated_expval(counts, qubit, n)` | Mitigated `<Z_k>` via Virtual Distillation post-processing |

### Simulation

| Function | Description |
|---|---|
| `run_simulation(target, noise, method, ...)` | End-to-end: returns unmitigated + mitigated `<Z_0>` and `Tr(rho^2)` |

## Project Structure

```
src/
  __init__.py      # Public API exports
  utils.py         # B^(2) gate, noise model, post-processing math
  simulation.py    # Circuit generation, M=2 construction, simulation runner
  benchmark.py     # N=5 and N=20 benchmarks
paper/
  main.tex         # Research paper (RevTeX)
  manuscript.tex   # Extended manuscript
  refs.bib         # Bibliography
```

## Key Implementation Details

- **B^(2) beamsplitter gate**: Decomposes into `CX → CRy(-π/2) → CX` (4 CX + 2 Ry). Each gate contributes ~2 non-Clifford gates in Clifford+T basis.
- **Pauli noise only**: All error channels use Pauli operators exclusively, enabling Monte Carlo trajectory sampling on MPS and extended_stabilizer backends without density-matrix fallback.
- **Post-processing**: Per-shot SWAP eigenvalues computed via `s_j = -1 if (b1=0, b2=1) else +1`, with mitigated value as ratio of sample means.

## References

- W. J. Huggins *et al.*, "Virtual Distillation for Quantum Error Mitigation," *Phys. Rev. X* **11**, 041036 (2021).
- B. Koczor, "Exponential Error Suppression for Near-Term Quantum Devices," *Phys. Rev. X* **11**, 031057 (2021).
