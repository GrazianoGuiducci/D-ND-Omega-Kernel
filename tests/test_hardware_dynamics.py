"""
Unit tests for Extropic Hardware Dynamics.

Verifies the scientific rigor of the Combinatorial Mechanics implementation.
"""

import unittest
import jax
import jax.numpy as jnp
from Extropic_Integration.hardware_dynamics.combinatorial import (
    MetricTensor, 
    nulla_tutto_potential, 
    transfer_function
)
from Extropic_Integration.hardware_dynamics.metrics import (
    curvature_index, 
    cycle_stability
)

class TestHardwareDynamics(unittest.TestCase):
    
    def test_metric_tensor_warping(self):
        """Verify MetricTensor creates symmetric warped space."""
        size = 10
        mt = MetricTensor(size)
        mt.warp_space(0, 1, 0.8) # Strong gravity (contraction)
        mt.warp_space(2, 3, -0.5) # Expansion
        
        metric = mt.get_metric()
        
        self.assertEqual(metric[0, 1], 0.8)
        self.assertEqual(metric[1, 0], 0.8) # Symmetry
        self.assertEqual(metric[2, 3], -0.5)
        self.assertEqual(metric[0, 0], 1.0) # Identity preserved
        
    def test_transfer_function_collapse(self):
        """Verify the collapse from potential to state."""
        size = 5
        potential = nulla_tutto_potential(size, scale=0.1)
        metric = jnp.eye(size) # Flat space
        # Add some warping
        metric = metric.at[0, 1].set(0.5)
        metric = metric.at[1, 0].set(0.5)
        
        state = jnp.ones(size)
        
        new_state = transfer_function(state, potential, metric)
        
        # Expectation: tanh(0.1 + metric @ 1.0)
        # For node 0: 0.1 + (1.0*1 + 0.5*1) = 1.6 -> tanh(1.6) > 0
        self.assertTrue(jnp.all(new_state > 0))
        
    def test_curvature_metric(self):
        """Verify curvature increases with assonance."""
        size = 2
        state = jnp.array([1.0, 1.0])
        
        # Case 1: Dissonance (Anti-ferromagnetic)
        # Energy = -0.5 * (1 * -1 * 1) = 0.5 (High Energy)
        topology_bad = jnp.array([[0, -1.0], [-1.0, 0]])
        curv_bad = curvature_index(state, topology_bad)
        
        # Case 2: Assonance (Ferromagnetic)
        # Energy = -0.5 * (1 * 1 * 1) = -0.5 (Low Energy)
        topology_good = jnp.array([[0, 1.0], [1.0, 0]])
        curv_good = curvature_index(state, topology_good)
        
        # Curvature should be higher (more positive) for Assonance
        # Note: Our metric maps Low Energy -> High Curvature (1.0)
        # High Energy -> Low Curvature (-1.0)
        self.assertGreater(curv_good, curv_bad)
        
    def test_cycle_stability_convergence(self):
        """Verify stability theorem logic."""
        # Converging sequence
        self.assertTrue(cycle_stability(1.0001, 1.0000, epsilon=1e-3))
        
        # Diverging sequence
        self.assertFalse(cycle_stability(1.5, 1.0, epsilon=1e-3))

if __name__ == '__main__':
    unittest.main()
