"""
D-ND Kernel Genesis Module
Handles the emergence from the Continuum Nulla-Tutto (NT) and the f_MIR Imprinting.
"""

from typing import Tuple

import jax.numpy as jnp
import jax.random as jrandom


class Genesis:
    """
    The Genesis Operator.
    Manages the transition from Non-Dual (NT) to Dual (Manifested) states.
    """

    def __init__(self, key_seed: int = 42):
        self.key = jrandom.PRNGKey(key_seed)

    def create_void(self, size: int) -> jnp.ndarray:
        """
        Creates the Continuum Nulla-Tutto (NT) state.
        Represented as a superposition of 0s (or balanced probabilities).
        """
        # In a spin system, the void could be represented as 0 magnetization
        return jnp.zeros(size)

    def f_MIR_imprint(self, intent_vector: jnp.ndarray, logic_matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        f_MIR (Meta-Intent Resonance) Imprint.
        Transforms the Semantic Intent and Logical Constraints into Physical Parameters.

        Mapping:
        - Intent Vector -> External Bias Field (h)
        - Logic Matrix -> Coupling Matrix (J)

        Args:
            intent_vector: Vector representing the semantic intent.
            logic_matrix: Matrix representing the logical topology.

        Returns:
            h_bias: The external magnetic field.
            J_coupling: The interaction matrix.
        """
        # In this isomorphism, the Intent IS the Bias Field.
        h_bias = intent_vector

        # The Logic IS the Coupling.
        # We ensure J is symmetric and has zero diagonal (standard Ising).
        J_coupling = logic_matrix
        J_coupling = (J_coupling + J_coupling.T) / 2.0
        J_coupling = J_coupling.at[jnp.diag_indices_from(J_coupling)].set(0.0)

        return h_bias, J_coupling

    def perturb_void(self, size: int, perturbation_strength: float = 0.1) -> jnp.ndarray:
        """
        Introduces a perturbation into the Void to initiate the dialectical process.
        """
        self.key, subkey = jrandom.split(self.key)
        noise = jrandom.normal(subkey, shape=(size,)) * perturbation_strength
        return noise
