"""
Unit tests for D-ND Hybrid Simulation.

Verifies the PhiTransform logic and its integration.
"""

import unittest

from Extropic_Integration.hybrid.phi import PhiTransform


class TestHybridSimulation(unittest.TestCase):
    def test_phi_transform_trigger(self):
        """Verify Phi triggers only when Coherence AND Tension are high."""
        phi = PhiTransform(coherence_threshold=0.8, tension_threshold=0.6)

        # Case 1: Low Coherence, Low Tension (Noise) -> No Trigger
        is_phi, _ = phi.evaluate(coherence=0.2, tension=0.2)
        self.assertFalse(is_phi)

        # Case 2: High Coherence, Low Tension (Stable Crystal) -> No Trigger
        is_phi, _ = phi.evaluate(coherence=0.9, tension=0.1)
        self.assertFalse(is_phi)

        # Case 3: Low Coherence, High Tension (Frustration) -> No Trigger
        is_phi, _ = phi.evaluate(coherence=0.3, tension=0.9)
        self.assertFalse(is_phi)

        # Case 4: High Coherence, High Tension (Criticality) -> TRIGGER
        is_phi, coeff = phi.evaluate(coherence=0.9, tension=0.8)
        self.assertTrue(is_phi)
        self.assertGreater(coeff, 0.0)

    def test_phi_transform_application(self):
        """Verify Phi lowers density (increases entropy)."""
        phi = PhiTransform()
        current_density = 0.8  # High structure

        # Apply mild transform
        new_density = phi.apply_transform(current_density, coefficient=0.5)
        self.assertLess(new_density, current_density)

        # Apply strong transform
        new_density_strong = phi.apply_transform(current_density, coefficient=1.0)
        self.assertLess(new_density_strong, new_density)


if __name__ == "__main__":
    unittest.main()
