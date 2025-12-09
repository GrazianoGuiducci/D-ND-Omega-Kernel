"""
vE_Archivista: The Retroactive Learner.

Role:
- Implements "Taxonomic Order General/Atomic Retroactive".
- Stores the "Manifesto" and "Metrics" of each cycle.
- Updates the System's Knowledge Base (Long-Term Memory).
- Provides "Precedents" for future cycles (Case Law / Memory).

Reference: 'MMS_Kernel_Dev_to_MMS_omega_kernel'
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List


class vE_Archivista:
    """
    Virtual Entity: Archivista (The Archivist).
    The Keeper of the System's Evolution.
    """

    def __init__(self, memory_path: str = "system_memory.json"):
        self.memory_path = memory_path
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """Loads the persistent memory from disk."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[vE_Archivista] Error loading memory: {e}")
                return {"cycles": [], "taxonomy": {}}
        return {"cycles": [], "taxonomy": {}}

    def archive_cycle(self, intent: str, result: Dict[str, Any], dipoles: List[tuple], manifesto: str):
        """
        Archives the results of a SACS cycle.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "intent": intent,
            "metrics": {
                "coherence": float(result["coherence"]),
                "tension": float(result["tension"]),
                "energy": float(result["energy"]),
            },
            "dipoles": dipoles,
            "manifesto": manifesto,
        }

        self.memory["cycles"].append(entry)
        self._update_taxonomy(intent, dipoles, result["coherence"])
        self._save_memory()

        print(f"[vE_Archivista] Cycle archived. Memory size: {len(self.memory['cycles'])} entries.")

    def _update_taxonomy(self, intent: str, dipoles: List[tuple], coherence: float):
        """
        Updates the internal Taxonomy based on success (Coherence).
        Retroactive Learning: If a cycle was coherent, reinforce the concepts.
        """
        # Lower threshold for prototype evolution (was 0.5)
        if coherence > 0.1:
            # Successful thought! Reinforce the dipoles.
            for concept, charge in dipoles:
                if concept not in self.memory["taxonomy"]:
                    self.memory["taxonomy"][concept] = {"count": 0, "avg_charge": 0.0}

                # Update stats
                stats = self.memory["taxonomy"][concept]
                stats["count"] += 1
                # Running average of charge (could evolve)
                stats["avg_charge"] = (stats["avg_charge"] * (stats["count"] - 1) + charge) / stats["count"]

            print(f"[vE_Archivista] Taxonomy updated for coherent intent: '{intent}'")

    def _save_memory(self):
        """Saves memory to disk."""
        try:
            with open(self.memory_path, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"[vE_Archivista] Error saving memory: {e}")

    def retrieve_precedent(self, intent: str) -> Dict[str, Any]:
        """
        Searches for a similar past intent (Case Law).
        """
        # Simple keyword match for prototype
        for entry in reversed(self.memory["cycles"]):
            if entry["intent"] == intent:
                print(f"[vE_Archivista] Precedent found for '{intent}'.")
                return entry
        return None
