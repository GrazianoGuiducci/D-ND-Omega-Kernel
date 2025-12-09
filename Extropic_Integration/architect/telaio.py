"""
vE_Telaio: The Metric Builder.

Role:
- Translates Semantic Dipoles into Spacetime Metric (g_uv).
- Constructs the "Geometry of Thought".
- Warps the MetricTensor based on the "Gravity" of concepts.

Reference: 'gravità come Modulatore della Metrica Spaziotemporale.txt'
"""

from typing import List, Tuple

from Extropic_Integration.hardware_dynamics.combinatorial import MetricTensor


class vE_Telaio:
    """
    Virtual Entity: Telaio (The Loom/Frame).
    Weaves the Metric Tensor from Semantic Threads.
    """

    def __init__(self, size: int):
        self.size = size

    def weave_metric(self, dipoles: List[Tuple[str, float]]) -> MetricTensor:
        """
        Constructs a MetricTensor based on the detected dipoles.
        """
        mt = MetricTensor(self.size)

        print(f"[vE_Telaio] Weaving Metric for {len(dipoles)} dipoles...")

        if not dipoles:
            print("  -> No dipoles. Returning Flat Space.")
            return mt

        # Logic:
        # 1. Positive Charge (Order/Gravity) -> Creates Contraction (Attractor).
        # 2. Negative Charge (Chaos/Entropy) -> Creates Expansion (Repulsor).
        # We distribute these effects across the matrix.

        # For prototype, we map dipoles to specific regions (blocks) of the matrix.
        # In a full system, this would use a Semantic Map (Embedding Space).

        block_size = self.size // (len(dipoles) + 1)

        for idx, (concept, charge) in enumerate(dipoles):
            start = idx * block_size
            end = start + block_size

            # Gravity Strength depends on charge magnitude
            # Positive Charge (>0) -> Attractive Gravity (Increases coupling)
            # Negative Charge (<0) -> Repulsive Gravity (Decreases coupling)
            gravity = charge * 0.8  # Scale factor

            gravity_type = "Attractive" if gravity > 0 else "Repulsive"
            print(f"  -> Weaving '{concept}' ({gravity_type}, g={gravity:.2f}) into region [{start}:{end}]")

            # Apply warping to this block
            for i in range(start, end):
                for j in range(start, end):
                    if i != j:
                        # In MetricTensor, positive warp increases the value (stronger connection)
                        # Negative warp decreases it (weaker connection/repulsion)
                        mt.warp_space(i, j, gravity)

        return mt
