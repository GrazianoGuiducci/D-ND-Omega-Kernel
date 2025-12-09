"""
Metrics for Extropic Hardware Dynamics.

This module implements the measurement operators to validate the stability
and curvature of the cognitive process.

Scientific Rigor:
- Curvature Index -> Fisher Information Metric / Hessian Eigenvalues
- Cycle Stability -> Lyapunov Stability / Convergence Criterion
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

def curvature_index(
    state: Float[Array, "N"], 
    topology: Float[Array, "N N"]
) -> float:
    """
    Measures the 'Informational Curvature' of the system state.
    
    In Information Geometry, curvature relates to how distinguishable probability
    distributions are. In D-ND, it relates to the 'gravity' of the semantic state.
    
    We approximate this using the energy landscape curvature (Hessian trace)
    induced by the topology.
    
    H = - sum J_ij s_i s_j
    Curvature ~ sum |J_ij| * correlation(s_i, s_j)
    
    For Metric Tensor (g_uv), we use the perturbation h_uv = g_uv - I.
    We automatically zero out the diagonal to focus on interactions.
    """
    # Zero out diagonal (Self-Interaction / Identity)
    # This makes the function work for both Adjacency Matrix (0 diag) and Metric (1 diag)
    h_uv = topology - jnp.diag(jnp.diag(topology))
    
    # Simple approximation: The energy density relative to the max possible energy
    # High curvature = Deep energy well (Strong Assonance)
    energy = -0.5 * jnp.dot(state, jnp.dot(h_uv, state))
    max_energy = 0.5 * jnp.sum(jnp.abs(h_uv))
    
    # Normalized curvature [-1, 1]
    # 1.0 = Perfect Assonance (Ground State, Low Energy)
    # -1.0 = Perfect Dissonance (High Energy)
    # We invert energy because Energy is negative for Assonance.
    return float(-energy / (max_energy + 1e-9))

def cycle_stability(
    omega_current: float, 
    omega_prev: float, 
    epsilon: float = 1e-3
) -> bool:
    """
    Implements the D-ND Stability Theorem:
    | Omega(n+1) / Omega(n) - 1 | < epsilon
    
    This checks if the 'Nulla-Tutto' measure (Omega) is converging.
    Omega here represents the 'Curvature' or 'Coherence' of the state.
    
    Args:
        omega_current: Metric value at step n+1
        omega_prev: Metric value at step n
        epsilon: Stability threshold
        
    Returns:
        True if the cycle is stable (converged), False otherwise.
    """
    if omega_prev == 0:
        return False
        
    ratio = jnp.abs((omega_current / omega_prev) - 1.0)
    return bool(ratio < epsilon)
