"""
D-ND Omega Kernel Demonstration
"The Physics of Thought"
"""

import os
import sys

# Add the parent directory to sys.path to allow importing dnd_kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp
from dnd_kernel.omega import OmegaKernel


def main():
    print("========================================")
    print("   D-ND OMEGA KERNEL: INITIALIZATION    ")
    print("========================================")

    # 1. Initialize the Kernel
    # Size 100 represents a small "Cognitive Field" of 100 concepts/qubits
    kernel = OmegaKernel(size=100, seed=1337)
    print(f"Kernel Initialized. Field Size: {kernel.size}")

    # 2. Define the Intent (The Perturbation)
    intent = "Harmonize the duality between Chaos and Order to find Equilibrium."
    print(f'\n[Input Intent]: "{intent}"')

    # 3. Phase 1: Perturbation (Non-Dual Expansion)
    print("\n--- PHASE 1: PERTURBATION (Non-Dual) ---")
    h_bias = kernel.perturb(intent)
    print(f"Bias Field Generated. Mean Intensity: {jnp.mean(jnp.abs(h_bias)):.4f}")

    # 4. Phase 2: Focus (Dual Contraction)
    print("\n--- PHASE 2: FOCUS (Dual) ---")
    # We apply a logic density of 0.1 (sparse connections)
    J_coupling = kernel.focus(logic_density=0.1)
    print(f"Logical Topology Generated. Connections: {jnp.count_nonzero(J_coupling) // 2}")

    # 5. Phase 3: Crystallization (Manifestation)
    print("\n--- PHASE 3: CRYSTALLIZATION (Resultant) ---")
    results = kernel.crystallize(steps=2000)

    # 6. Analysis of the Resultant
    R = results["R"]
    coherence = results["coherence"]
    energy = results["energy"]

    print("\n========================================")
    print("          COGNITIVE RESULTANT           ")
    print("========================================")
    print(f"Final Energy (Hamiltonian): {energy:.4f}")
    print(f"Coherence (Order Parameter): {coherence:.4f}")

    # Interpret the state
    # +1 represents "Assertion/Order", -1 represents "Negation/Chaos"
    positive_nodes = jnp.sum(R == 1)
    negative_nodes = jnp.sum(R == -1)

    print(f"State Distribution: {positive_nodes} (+) / {negative_nodes} (-)")

    if coherence > 0.8:
        print("Status: HIGHLY COHERENT (Strong Conclusion)")
    elif coherence > 0.3:
        print("Status: METASTABLE (Nuanced Conclusion)")
    else:
        print("Status: DISORDERED (Cognitive Dissonance)")

    print("\n[Omega Cycle Complete]")


if __name__ == "__main__":
    main()
