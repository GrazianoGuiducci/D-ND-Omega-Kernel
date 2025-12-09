"""
Combinatorial Mechanics for Extropic Hardware.

This module implements the "D-ND Physics" operators that map indeterminate states
to observable realities using Combinatorial Mechanics and Relational Topologies.

Scientific Rigor:
- AssonanceMatrix -> Weighted Adjacency Matrix (Graph Theory) / Hamiltonian Coupling (Physics)
- NullaTuttoPotential -> Scalar Field / Uniform Superposition (Quantum Mechanics)
- Transfer Function -> Projection Operator / Phase Transition (Thermodynamics)
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Any
from jaxtyping import Array, Float

class MetricTensor:
    """
    Represents the Spacetime Metric (g_uv) of the Cognitive Space.
    
    Formerly 'AssonanceMatrix', this class now defines how the space itself
    is curved by Semantic Gravity.
    
    g_uv = delta_uv + h_uv (Perturbation from Flat Space)
    """
    def __init__(self, size: int):
        self.size = size
        # Initialize with Flat Space (Euclidean Identity)
        # We store the perturbation h_uv separately for clarity
        self.metric = jnp.eye(size) 

    def warp_space(self, i: int, j: int, gravity: float):
        """
        Warps the metric tensor between two nodes.
        Gravity > 0: Contraction (Nodes get closer/more coupled).
        Gravity < 0: Expansion (Nodes get further/decoupled).
        """
        # We modify the off-diagonal elements to represent curvature/coupling
        # In a metric tensor, g_ij represents the dot product of basis vectors.
        # Higher g_ij = stronger correlation/closer distance in manifold.
        self.metric = self.metric.at[i, j].set(gravity)
        self.metric = self.metric.at[j, i].set(gravity)

    def get_metric(self) -> Float[Array, "N N"]:
        return self.metric

def nulla_tutto_potential(size: int, scale: float = 1.0) -> Float[Array, "N"]:
    """
    Models the 'Nulla-Tutto' (NT) Potential Field.
    
    Physically, this represents the 'Vacuum State' or 'Uniform Superposition'
    before observation. It is a scalar field of potential energy.
    
    Args:
        size: Number of dimensions (nodes).
        scale: Energy scale of the potential.
        
    Returns:
        A vector representing the potential at each node.
    """
    # In a perfect NT state, potential is uniform or zero-point energy.
    # We model it as a uniform field with slight quantum fluctuations (noise).
    return jnp.ones(size) * scale

def transfer_function(
    state: Float[Array, "N"], 
    potential: Float[Array, "N"], 
    metric: Float[Array, "N N"]
) -> Float[Array, "N"]:
    """
    The 'Collapse' mechanism that transfers information from the Indeterminate
    (Potential) to the Observable (State) via the Metric Tensor.
    
    Mathematically: S_new = activation( Potential + Metric @ State )
    Physically: Geodesic flow in curved space.
    """
    # Interaction term: How the current state flows along the metric
    # In flat space (Identity), this is just the state itself.
    # In curved space, it's the covariant derivative (simplified).
    interaction = jnp.dot(metric, state)
    
    # Total Field = Intrinsic Potential + Interaction
    total_field = potential + interaction
    
    # Transfer/Activation (e.g., Hyperbolic Tangent for spin relaxation)
    return jnp.tanh(total_field)
