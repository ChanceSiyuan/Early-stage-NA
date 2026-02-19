You are an expert researcher and engineer in Quantum Information and Quantum Computing. I am conducting LARGE-SCALE numerical simulations to reproduce the core results of two seminal quantum error mitigation papers:
1. "Virtual Distillation for Quantum Error Mitigation" (Huggins et al., 2020)
2. "Error Suppression by Derangement" (Koczor, 2020)

My goal is to evaluate the $M=2$ protocol (calculating $\text{Tr}(O \rho^2) / \text{Tr}(\rho^2)$ via joint measurements) on large quantum circuits. Because the $M=2$ protocol doubles the qubit count to $2N$, standard statevector/density-matrix simulators quickly hit a memory wall. 

Therefore, I need you to write a highly modular, production-ready Python script (using Qiskit and Qiskit Aer) that implements the $M=2$ protocol and compares its execution across THREE different simulation methods:
1. `statevector` (as a baseline for small scales)
2. `extended_stabilizer` (for large scales with few non-Clifford gates)
3. `matrix_product_state` (for large scales with low depth / 1D connectivity)

### CRITICAL REQUIREMENT: Scalable Pauli Noise & Quantum Trajectories
To maintain scalability with noise, we MUST absolutely avoid full density matrix simulations. The noise simulation must be driven by Quantum Trajectories (Monte Carlo pure-state sampling).
- You must build a flexible noise interface that strictly injects **Pauli-based noise** (e.g., single- and two-qubit depolarizing, bit-flip, phase-flip).
- You must configure the `AerSimulator` so that when noise is applied to the `extended_stabilizer` or `matrix_product_state` backends, it forces stochastic pure-state evolution (Monte Carlo shots) instead of crashing or falling back to a density matrix.

### Task Requirements:

1. **Tailored Circuit Generation**: 
   Write a function to generate an $N$-qubit target circuit with a `circuit_type` flag:
   - `near_clifford`: Deep circuit dominated by Clifford gates with a strictly bounded number of T-gates (optimized for Extended Stabilizer).
   - `local_shallow`: Shallow hardware-efficient ansatz with 1D nearest-neighbor entanglement (optimized for MPS).

2. **Extensible Pauli Noise Interface**:
   Implement a function `get_pauli_noise_model(p_1q, p_2q, noise_type='depolarizing')` returning a `NoiseModel`. Ensure it only uses operations compatible with efficient Pauli tracking (e.g., `pauli_error`, `depolarizing_error`). 

3. **The M=2 Protocol**:
   Construct the $2N$-qubit composite circuit preparing $\rho \otimes \rho$. Implement the joint measurement layer for the $M=2$ case (either Virtual Distillation transversal diagonalizing measurement or ESD CSWAP test). Ensure the measurement setup avoids breaking the simulation advantages.

4. **Unified Simulation Execution (Monte Carlo)**:
   Write `run_simulation(circuit, noise_model, method_name, shots)`. 
   - Explicitly configure the backend: `AerSimulator(method=method_name, noise_model=noise_model)`. 
   - Add comments explaining how Qiskit Aer handles this via Monte Carlo trajectories for MPS/Extended Stabilizer.
   - Compute both UNMITIGATED and MITIGATED expectation values of $Z_0$.

5. **Benchmarking Script**:
   - Small system ($N=5$, 10 physical qubits): Compare all three methods with a non-zero Pauli noise rate to prove consistency.
   - Scaled-up system ($N=20$, 40 physical qubits): Run ONLY `extended_stabilizer` and `matrix_product_state` under the Pauli noise model, demonstrating that the trajectory-based simulation successfully avoids Out-of-Memory (OOM) errors.

Please provide the complete, heavily commented code. Specifically, explain your design choices for the Pauli noise interface and how to swap in different Pauli channels in the future.