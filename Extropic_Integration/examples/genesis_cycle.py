"""
Genesis Cycle: The Autological Evolution Simulation.

This script runs the SACS Architecture in a continuous loop to demonstrate
the emergence of "Taxonomic Order" from initial noise.

Process:
1. Initialize SACS with empty memory (or load existing).
2. Generate a sequence of evolutionary intents (Chaos -> Order -> Complexity).
3. Allow the system to:
    - Perceive (Sonar)
    - Construct (Telaio)
    - Sculpt (Scultore)
    - Collapse (Omega)
    - Learn (Archivista)
4. Display the evolved Taxonomy at the end.
"""

import os
import random
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Extropic_Integration.architect.sacs import SACS


def generate_intent(cycle_num: int, previous_manifesto: str) -> str:
    """
    Generates an evolutionary intent based on the cycle number.
    In a real system, this would be the 'Seeker' (vE_Sonar) generating queries.
    """
    base_concepts = ["chaos", "order", "entropy", "gravity", "void", "matter", "energy", "structure"]

    # Phase 1: Primordial Soup (Random)
    if cycle_num < 3:
        c1 = random.choice(base_concepts)
        c2 = random.choice(base_concepts)
        return f"Explore the interaction between {c1} and {c2}"

    # Phase 2: Directed Evolution (Based on previous result)
    # Simple logic: If previous was chaotic, seek order. If ordered, seek complexity.
    if "CHAOTIC" in previous_manifesto:
        return "Stabilize the field with high gravity and order"
    elif "DOGMATIC" in previous_manifesto:
        return "Introduce entropy to allow new growth"
    else:
        return "Synthesize a complex structure from balanced forces"


def run_genesis_cycle(cycles: int = 5):
    print(f"Initializing Genesis Cycle ({cycles} Epochs)...")
    sacs = SACS(size=100)

    # Clear memory for a fresh start (optional, but good for demo)
    if os.path.exists("system_memory.json"):
        os.remove("system_memory.json")
        # Re-init archivist to reload empty memory
        sacs.archivista.memory = {"cycles": [], "taxonomy": {}}

    previous_manifesto = "NEUTRAL"

    for i in range(cycles):
        print(f"\n--- EPOCH {i + 1}/{cycles} ---")

        # 1. Generate Intent
        intent = generate_intent(i, previous_manifesto)
        print(f"Intent: '{intent}'")

        # 2. Process
        previous_manifesto = sacs.process(intent, steps=300)

        # 3. Brief pause for dramatic effect (and file I/O safety)
        time.sleep(0.5)

    print("\n" + "=" * 30)
    print("GENESIS COMPLETE. EVOLVED TAXONOMY:")
    print("=" * 30)

    # Display the learned Taxonomy
    taxonomy = sacs.archivista.memory["taxonomy"]
    for concept, stats in taxonomy.items():
        print(f"Concept: {concept.upper():<15} | Count: {stats['count']:<3} | Avg Charge: {stats['avg_charge']:.2f}")


if __name__ == "__main__":
    run_genesis_cycle(cycles=6)
