"""
D-ND Kernel Utilities
Helper functions for Semantic Resonance and Topological Coupling.
"""

import hashlib

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def semantic_resonance(text: str, size: int, seed: int = 42) -> jnp.ndarray:
    """
    Semantic Resonance (Enhanced).
    Maps a text string (Intent) to a bias vector (h) using a Concept Dictionary.
    This simulates "Concept Activation" rather than just random hashing.
    """
    # Simple Concept Dictionary (Ontology)
    # Maps keywords to specific "regions" (seeds) of the latent space.
    concepts = {
        "order": 100,
        "logic": 101,
        "structure": 102,
        "focus": 103,
        "chaos": 200,
        "entropy": 201,
        "void": 202,
        "noise": 203,
        "evolution": 300,
        "growth": 301,
        "change": 302,
        "dynamic": 303,
        "stasis": 400,
        "fixed": 401,
        "invariant": 402,
        "static": 403,
        "duality": 500,
        "binary": 501,
        "split": 502,
        "unity": 600,
        "one": 601,
        "whole": 602,
    }

    text_lower = text.lower()
    active_seeds = []

    # Find active concepts
    for word, val in concepts.items():
        if word in text_lower:
            active_seeds.append(val)

    # If no concepts found, fall back to hash of the text
    if not active_seeds:
        hash_object = hashlib.sha256(text.encode())
        hex_dig = hash_object.hexdigest()
        base_seed = int(hex_dig[:8], 16)
    else:
        # Combine active concept seeds
        base_seed = sum(active_seeds)

    # Combine with master seed
    final_seed = seed + base_seed
    key = jrandom.PRNGKey(final_seed)

    # Generate resonance vector
    # We use the seed to determine the "direction" of the bias
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
