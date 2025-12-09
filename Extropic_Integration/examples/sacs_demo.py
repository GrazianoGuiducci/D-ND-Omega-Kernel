"""
SACS Demo: Verifying the Cognitive Architecture.

Runs the SACS Orchestrator on a sequence of intents to demonstrate:
1. Dipole Detection (Sonar)
2. Metric Warping (Telaio)
3. Thermodynamic Collapse (Omega)
4. Manifesto Generation (Cristallizzatore)
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Extropic_Integration.architect.sacs import SACS


def run_sacs_demo():
    print("Initializing SACS (System Architecture for Cognitive Synthesis)...")
    sacs = SACS(size=100)

    intents = [
        "Generate absolute chaos and entropy",
        "Establish perfect order and gravity",
        "Find the balance between void and matter",
    ]

    # Enable verbose logging for Sculptor
    # Note: Sculptor logs to stdout, so we just run it.

    for intent in intents:
        manifesto = sacs.process(intent, steps=500)
        print(manifesto)
        print("\n" + "=" * 30 + "\n")

    # Verify Memory
    print("Verifying Archivist Memory...")
    if os.path.exists("system_memory.json"):
        import json

        with open("system_memory.json", "r") as f:
            memory = json.load(f)
            print(f"Memory contains {len(memory['cycles'])} cycles.")
            print(f"Taxonomy: {json.dumps(memory['taxonomy'], indent=2)}")
    else:
        print("Error: system_memory.json not found!")


if __name__ == "__main__":
    run_sacs_demo()
