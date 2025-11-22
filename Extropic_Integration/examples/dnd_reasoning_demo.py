"""
D-ND Omega Kernel Demonstration (Autological)
"The Physics of Thought" - Self-Improving Cycle
"""

import os
import sys

# Add the parent directory to sys.path to allow importing dnd_kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dnd_kernel.omega import OmegaKernel


def main():
    print("========================================")
    print("   D-ND OMEGA KERNEL: AUTOLOGICAL DEMO  ")
    print("========================================")

    # 1. Initialize the Kernel
    kernel = OmegaKernel(size=100, seed=1337)
    print(f"Kernel Initialized. Field Size: {kernel.size}")
    print(f"Initial Logic Density: {kernel.logic_density}")

    # Define a sequence of intents to test adaptation
    intents = [
        "Generate absolute chaos and noise",  # Should have low coherence -> Increase Density
        "Establish perfect order and structure",  # Should have high coherence -> Decrease Density
        "Generate absolute chaos and noise",  # Should be more constrained now?
    ]

    for i, intent in enumerate(intents):
        print(f"\n\n--- CYCLE {i + 1}: '{intent}' ---")

        # Run the full Autological Cycle
        result = kernel.process_intent(intent, steps=2000)

        # Analysis
        coherence = result["coherence"]
        energy = result["energy"]

        print(f"Resultant Coherence: {coherence:.4f}")
        print(f"Resultant Energy:    {energy:.4f}")

        # Show Adaptation
        print(f"New Logic Density:   {kernel.logic_density:.2f}")

        if coherence > 0.8:
            print("Status: CRYSTALLIZED (High Order)")
        elif coherence < 0.2:
            print("Status: FLUID (High Entropy)")
        else:
            print("Status: METASTABLE (Complex)")

    print("\n========================================")
    print("          EXPERIENCE LOG                ")
    print("========================================")
    print(f"History of Coherence: {kernel.experience}")
    print("[Omega Cycle Complete]")


if __name__ == "__main__":
    main()
