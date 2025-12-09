"""
SACS Orchestrator: The Cognitive Architecture.

Integrates the Virtual Entities (Sonar, Telaio, Cristallizzatore) with the
Omega Kernel to form a complete Autological System.

Flow:
1. Intent -> Sonar (Dipole Detection)
2. Dipoles -> Telaio (Metric Construction)
3. Metric -> Omega (Thermodynamic Cycle)
4. Result -> Cristallizzatore (Manifesto Generation)
"""

from typing import Any, Dict

from Extropic_Integration.architect.archivista import vE_Archivista
from Extropic_Integration.architect.cristallizzatore import vE_Cristallizzatore
from Extropic_Integration.architect.scultore import vE_Scultore
from Extropic_Integration.architect.sonar import vE_Sonar
from Extropic_Integration.architect.telaio import vE_Telaio
from Extropic_Integration.dnd_kernel.omega import OmegaKernel


class SACS:
    """
    System Architecture for Cognitive Synthesis (SACS).
    The Mind that drives the Omega Engine.
    """

    def __init__(self, size: int = 100):
        self.size = size

        # The Virtual Entities
        self.sonar = vE_Sonar()
        self.telaio = vE_Telaio(size)
        self.cristallizzatore = vE_Cristallizzatore()
        self.scultore = vE_Scultore()
        self.archivista = vE_Archivista()

        # The Engine
        self.omega = OmegaKernel(size)

    def process(self, intent: str, steps: int = 1000) -> Dict[str, Any]:
        """
        Executes the full SACS Cognitive Cycle.
        """
        print(f"\n=== SACS CYCLE START: '{intent}' ===")

        # 1. Perception (Sonar)
        dipoles = self.sonar.scan(intent)

        # 2. Construction (Telaio)
        metric_tensor = self.telaio.weave_metric(dipoles)

        # 3. Integration (Inject Metric into Omega)
        # We override the default random coupling with our constructed metric
        self.omega.metric_tensor = metric_tensor.get_metric()

        # 4. Processing (Omega Cycle)
        # vE_Scultore: Apply Dynamic Gravity based on intent
        gravity_info = self.omega.apply_dynamic_gravity(intent)

        self.omega.perturb(intent)
        self.omega.focus(self.omega.logic_density)  # Use current density
        result = self.omega.crystallize(steps, sculptor=self.scultore)

        # 5. Expression (Cristallizzatore)
        manifesto = self.cristallizzatore.manifest(intent, result, dipoles)

        # 6. Memory (Archivista)
        self.archivista.archive_cycle(intent, result, dipoles, manifesto)

        # vE_Archivista: Retroactive Consolidation (MMS Phase 5)
        memory_info = self.omega.consolidate_memory(intent, result)

        # Build detailed timeline for Didactic Layer using Omega's D-ND DSL logic
        dsl_trace = self.omega._generate_dsl_trace(intent, gravity_info, result)
        rosetta_stone = self.omega._generate_rosetta_stone()

        # Add Didactic Info with Timeline
        lattice_data = self.omega._generate_lattice_data(result["R"])
        tensor_field = self.omega.metric_tensor.tolist()

        result["didactic"] = {
            "timeline": dsl_trace,
            "gravity_info": gravity_info,
            "memory_info": memory_info,
            "lagrangian_distance": abs(result["tension"] - 0.5),
            "rosetta_stone": rosetta_stone,
            "lattice": lattice_data,
            "tensor_field": tensor_field,
            "entropy": float(self.omega.entropy),
            "gravity": float(gravity_info["curvature"]),
        }

        print("=== SACS CYCLE END ===\n")
        return {"manifesto": manifesto, "result": result}
