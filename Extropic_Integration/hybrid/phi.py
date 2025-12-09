"""
Phi Transform: Phase Transition Logic for D-ND Hybrid Simulation.

This module implements the logic for "Critical Transitions" (Phi) based on
the interplay between Coherence (Order) and Tension (Energy/Entropy).

Scientific Rigor:
- Coherence -> Order Parameter (Magnetization)
- Tension -> Free Energy / Frustration
- Phi Transform -> Phase Transition / Criticality
"""

import jax.numpy as jnp
from typing import Tuple

class PhiTransform:
    """
    Manages the Phase Transition logic (Phi).
    
    The system monitors Coherence (C) and Tension (T).
    - Low C, Low T: Noise (Seek Order)
    - High C, Low T: Crystallization (Stable)
    - High C, High T: Criticality (Phi Trigger -> Re-opening)
    """
    
    def __init__(self, coherence_threshold: float = 0.8, tension_threshold: float = 0.6):
        self.coherence_threshold = coherence_threshold
        self.tension_threshold = tension_threshold
        self.transition_active = False
        
    def evaluate(self, coherence: float, tension: float) -> Tuple[bool, float]:
        """
        Evaluates the system state and determines if a Phi Transform is needed.
        
        Args:
            coherence: Measure of order (0.0 to 1.0).
            tension: Measure of internal conflict/energy (0.0 to 1.0).
            
        Returns:
            (is_transition, transition_coefficient)
            - is_transition: True if Phi Transform is triggered.
            - transition_coefficient: 0.0 (No change) to 1.0 (Full reset).
        """
        # Logic: If we are highly coherent but also highly tense, 
        # it means we are forcing a fit that doesn't belong (Overfitting/Dogma).
        # We need to "melt" the system to allow new structures to emerge.
        
        if coherence > self.coherence_threshold and tension > self.tension_threshold:
            self.transition_active = True
            # The coefficient scales with how far we are above the thresholds
            excess_c = coherence - self.coherence_threshold
            excess_t = tension - self.tension_threshold
            coeff = min(1.0, (excess_c + excess_t) * 2.0)
            return True, coeff
            
        self.transition_active = False
        return False, 0.0

    def apply_transform(self, current_density: float, coefficient: float) -> float:
        """
        Applies the Phi Transform to the Logic Density.
        
        If a transition is triggered, we lower the density (increase entropy)
        to allow for re-organization.
        """
        # Lower density proportional to the coefficient
        # If coeff is 1.0 (Full Reset), density drops significantly
        new_density = max(0.05, current_density - (0.3 * coefficient))
        return new_density
