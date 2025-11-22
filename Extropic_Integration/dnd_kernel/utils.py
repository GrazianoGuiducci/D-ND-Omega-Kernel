"""
D-ND Kernel Utilities
Helper functions for Semantic Resonance and Topological Coupling.
"""

import jax.numpy as jnp
import jax.random as jrandom
import hashlib
import numpy as np

def semantic_resonance(text: str, size: int, seed: int = 42) -> jnp.ndarray:
    """
    Mock Semantic Resonance.
    Maps a text string (Intent) to a bias vector (h) using deterministic hashing.
    
    In a full implementation, this would use an LLM embedding model.
    """
    # Create a deterministic seed from the text
    hash_object = hashlib.sha256(text.encode())
    hex_dig = hash_object.hexdigest()
    text_seed = int(hex_dig[:8], 16)
    
    # Combine with base seed
    key = jrandom.PRNGKey(seed + text_seed)
    
    # Generate random vector in [-1, 1]
    # Represents the "pull" of the intent on each node
    resonance_vector = jrandom.uniform(key, shape=(size,), minval=-1.0, maxval=1.0)
    
    return resonance_vector

def topological_coupling(size: int, density: float = 0.2, seed: int = 42) -> jnp.ndarray:
    """
    Mock Topological Coupling.
    Generates a sparse symmetric matrix (J) representing logical constraints.
    
    Args:
        size: Number of nodes (concepts).
        density: Probability of connection between two nodes.
        seed: Random seed.
        
    Returns:
        J_matrix: Symmetric matrix with zero diagonal.
    """
    np.random.seed(seed)
    
    # Generate random matrix
    mask = np.random.rand(size, size) < density
    weights = np.random.randn(size, size)
    
    J = weights * mask
    
    # Make symmetric
    J = (J + J.T) / 2.0
    
    # Zero diagonal (no self-interaction in standard Ising)
    np.fill_diagonal(J, 0.0)
    
    return jnp.array(J)

def matrix_to_edges(J_matrix: jnp.ndarray):
    """
    Converts a J matrix to a list of edges (u, v, weight) for THRML.
    """
    size = J_matrix.shape[0]
    edges = []
    # Use numpy for iteration as it's easier for graph construction
    J_np = np.array(J_matrix)
    
    for i in range(size):
        for j in range(i + 1, size):
            w = J_np[i, j]
            if w != 0:
                edges.append((i, j, w))
                
    return edges
