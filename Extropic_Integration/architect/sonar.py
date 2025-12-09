"""
vE_Sonar: The Semantic Dipole Detector.

Role:
- Scans the input for "Semantic Dipoles" (Polarizations).
- Detects the "Charge" of concepts (Positive/Negative sentiment or active/passive).
- Maps the "Spacetime Coordinates" of the intent.

Reference: 'gravità come Modulatore della Metrica Spaziotemporale.txt'
"""

from typing import List, Tuple


class vE_Sonar:
    """
    Virtual Entity: Sonar.
    Detects latent structures in the noise.
    """

    def __init__(self):
        # A simple lexicon of "Charged" concepts for the prototype.
        # In a full system, this would be a Vector Database or LLM.
        self.polarities = {
            "chaos": -1.0,
            "order": 1.0,
            "entropy": -1.0,
            "gravity": 1.0,
            "noise": -1.0,
            "signal": 1.0,
            "void": 0.0,
            "matter": 1.0,
            "expansion": -0.5,
            "contraction": 0.5,
        }

    def scan(self, intent: str) -> List[Tuple[str, float]]:
        """
        Scans the intent for charged concepts (Dipoles).
        Returns a list of (concept, charge).
        """
        words = intent.lower().split()
        detected_dipoles = []

        print(f"[vE_Sonar] Scanning intent: '{intent}'")

        for word in words:
            # Simple keyword matching for prototype
            # Remove punctuation
            clean_word = word.strip(".,;!?")
            if clean_word in self.polarities:
                charge = self.polarities[clean_word]
                detected_dipoles.append((clean_word, charge))
                print(f"  -> Detected Dipole: {clean_word} (Charge: {charge})")

        if not detected_dipoles:
            print("  -> No strong dipoles detected. Assuming Neutral Field.")

        return detected_dipoles

    def analyze_polarization(self, dipoles: List[Tuple[str, float]]) -> float:
        """
        Calculates the net polarization of the intent.
        """
        if not dipoles:
            return 0.0

        charges = [d[1] for d in dipoles]
        return sum(charges) / len(charges)
