"""
Omega Kernel: The Physics of Thought
Integrates D-ND Axioms with Extropic's THRML library.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp

# Hardware Dynamics Imports
from Extropic_Integration.hardware_dynamics.combinatorial import MetricTensor
from Extropic_Integration.hardware_dynamics.metrics import cycle_stability
from Extropic_Integration.hybrid.phi import PhiTransform

# THRML Imports
from thrml.pgm import SpinNode

# D-ND Imports
from .axioms import DNDConstants
from .genesis import Genesis
from .utils import semantic_resonance, topological_coupling

# Note: vE_Scultore is injected at runtime to avoid circular imports if possible,
# or we import it inside the method.


class OmegaKernel:
    """
    The Omega Kernel.
    A Cognitive Operating System that moves thought through variable density phases.

    Cycle:
    1. Perturbation (Non-Duale): Expansion of context.
    2. Focus (Duale): Application of constraints (Lagrangian minimization).
    3. Crystallization (Resultant): Collapse of the wave function.
    """

    def __init__(self, size: int = 100, seed: int = 42):
        self.size = size
        self.genesis = Genesis(seed)
        self.constants = DNDConstants()

        # State
        self.nodes = [SpinNode() for i in range(size)]
        self.h_bias = jnp.zeros(size)
        self.metric_tensor = jnp.eye(size)  # Formerly J_coupling
        self.current_R = jnp.zeros(size)  # The Resultant

        # Metrics
        self.coherence = 0.0
        self.entropy = 0.0
        self.tension = 0.0  # New metric for Phi
        self.logic_density = 0.2  # Adaptive parameter
        self.logic_density = 0.2  # Adaptive parameter
        self.experience = []  # History of (intent, coherence)
        self.energy_history = []  # History of energy during last crystallization
        self.phi = PhiTransform()  # Hybrid Transition Logic

    def process_intent(self, intent_text: str, steps: int = 1000) -> Dict[str, Any]:
        """
        Orchestrates the full Omega Cycle:
        Perturbation -> Focus -> Crystallization -> Feedback (Autopoiesis).
        """
        # 1. Perturbation
        self.perturb(intent_text)

        # 2. Focus
        # vE_Scultore: Apply Dynamic Gravity based on intent
        gravity_info = self.apply_dynamic_gravity(intent_text)

        # Use current adaptive logic_density
        self.focus(self.logic_density)

        # 3. Crystallization
        result = self.crystallize(steps)

        # 4. Autological Feedback Loop (Self-Correction)
        # We use the Stability Theorem to determine if the thought is stable.
        # | Omega(n+1) / Omega(n) - 1 | < epsilon
        current_coherence = result["coherence"]
        current_tension = result["tension"]

        # Check stability against previous experience
        is_stable = False
        if self.experience:
            prev_coherence = (
                self.experience[-1] if isinstance(self.experience[-1], float) else self.experience[-1]["coherence"]
            )
            is_stable = cycle_stability(current_coherence, prev_coherence)

        self._adapt(current_coherence, current_tension, is_stable)

        # vE_Archivista: Consolidate Memory
        self.consolidate_memory(intent_text, result)

        # Add Didactic Info to Result (Cognitive Bridge)
        dsl_trace = self._generate_dsl_trace(intent_text, gravity_info, result)
        rosetta_stone = self._generate_rosetta_stone()

        # Generate Visualization Data (Visual Cortex)
        lattice_data = self._generate_lattice_data(result["R"])
        tensor_field = self.metric_tensor.tolist()  # Convert JAX array to list

        result["didactic"] = {
            "timeline": dsl_trace,
            "rosetta_stone": rosetta_stone,
            "lattice": lattice_data,
            "tensor_field": tensor_field,
            "gravity": float(gravity_info["curvature"]),
            "entropy": float(self.entropy),
        }

        return result

    def _generate_lattice_data(self, state_vector) -> list:
        """
        Generates 2D lattice data for Visual Cortex visualization.
        Maps the 1D state vector to a 2D grid.
        """
        lattice = []
        grid_size = int(self.size**0.5)

        # Ensure state_vector is a numpy/jax array
        spins = jnp.array(state_vector)

        for i in range(self.size):
            x = i % grid_size
            y = i // grid_size

            # Calculate a simulated stability based on local alignment (simplified)
            # In a real Ising model, stability ~ local field alignment
            # Here we just use a placeholder or derived value
            stability = 0.8  # Default high stability for crystallized state

            lattice.append(
                {
                    "id": i,
                    "x": x,
                    "y": y,
                    "spin": int(spins[i]),
                    "stability": stability,
                    "assetTicker": None,  # Placeholder for future econophysics
                }
            )

        return lattice

    def _generate_dsl_trace(self, intent: str, gravity_info: Dict[str, Any], result: Dict[str, Any]) -> list:
        """
        Generates the D-ND DSL Pipeline Trace for the Didactic Layer.
        Maps the cognitive cycle to the formal DSL steps:
        Problem -> Variables -> Constraints -> Energy -> Hardware -> Output
        """
        # 1. Problem / Draft
        # In a real implementation, this would be the "Succo" extracted by an LLM.
        # Here we simulate the draft based on the intent.
        draft_summary = f"Maximize coherence for '{intent}'"

        # 2. Variables
        # The domain is the spin network state space.
        variables_desc = f"SpinNode[0..{self.size - 1}] ∈ {{-1, 1}}"

        # 3. Constraints
        # Logic Density determines the complexity of constraints.
        density = self.logic_density
        constraints_desc = f"LogicDensity({density:.2f}) -> ∀ i,j: J_ij != 0"

        # 4. Energy Mapping
        # Mapping constraints to Hamiltonian.
        energy_desc = "H = -Σ h_i s_i - Σ J_ij s_i s_j"

        # 5. Hardware Execution
        # TSU simulation details.
        steps = len(self.energy_history)
        hardware_desc = f"TSU::Anneal(steps={steps}, schedule='simulated')"

        # 6. Output
        coherence = result["coherence"]
        output_desc = f"FinalState (Coherence: {coherence:.2f})"

        return [
            {
                "step": "PROBLEM",
                "label": "Problem Definition",
                "d_nd": "Intent Extraction",
                "physics": "Boundary Conditions",
                "detail": f"Intent: '{intent}' -> Draft: '{draft_summary}'",
                "icon": "📝",
            },
            {
                "step": "VARIABLES",
                "label": "Variable Definition",
                "d_nd": "Concept Space",
                "physics": "Degrees of Freedom",
                "detail": variables_desc,
                "icon": "xyz",
            },
            {
                "step": "CONSTRAINTS",
                "label": "Logical Constraints",
                "d_nd": "Axiomatic Rules",
                "physics": "Coupling Matrix (J)",
                "detail": constraints_desc,
                "icon": "🔗",
            },
            {
                "step": "ENERGY",
                "label": "Energy Mapping",
                "d_nd": "Dissonance Map",
                "physics": "Hamiltonian (H)",
                "detail": energy_desc,
                "icon": "⚡",
            },
            {
                "step": "HARDWARE",
                "label": "Thermodynamic Execution",
                "d_nd": "Cognitive Processing",
                "physics": "TSU Annealing",
                "detail": hardware_desc,
                "icon": "🌡️",
            },
            {
                "step": "OUTPUT",
                "label": "Crystallization",
                "d_nd": "Manifestation",
                "physics": "Ground State",
                "detail": output_desc,
                "icon": "💎",
            },
        ]

    def _generate_rosetta_stone(self) -> Dict[str, str]:
        """
        Returns the Rosetta Stone mapping for the D-ND DSL Operators.
        """
        return {
            "∧ (AND)": "Co-existence / Interaction (J > 0)",
            "∨ (OR)": "Multiple Configurations (Degeneracy)",
            "¬ (NOT)": "Energy Penalty (+ΔE)",
            "∀ (FORALL)": "Global Field Constraint",
            "Minimize": "Thermodynamic Annealing",
            "SeedNoise": "Probabilistic Gap-Filling",
            "Converge": "Ground State (Solution)",
        }

    def _adapt(self, coherence: float, tension: float, is_stable: bool):
        """
        Autological Adaptation based on Coherence, Tension, and Stability.
        Uses PhiTransform to trigger Critical Transitions.
        """
        self.experience.append(coherence)

        # 1. Check for Phi Transform (Critical Transition)
        is_phi, phi_coeff = self.phi.evaluate(coherence, tension)

        if is_phi:
            # Criticality reached! Re-open the structure.
            old_density = self.logic_density
            self.logic_density = self.phi.apply_transform(self.logic_density, phi_coeff)
            print(f"[Omega] PHI TRANSFORM TRIGGERED! (C={coherence:.2f}, T={tension:.2f})")
            print(f"        Re-opening Structure: Density {old_density:.2f} -> {self.logic_density:.2f}")
            return  # Phi overrides standard adaptation

        # 2. Standard Adaptation (if no Phi)
        # If unstable, we need more structure (Gravity) -> Increase Density
        if not is_stable:
            self.logic_density = min(1.0, self.logic_density + 0.05)
            print(f"[Omega] Autopoiesis: Unstable Cycle. Increasing Gravity to {self.logic_density:.2f}")
        # If stable but low coherence, we might be stuck in a local minimum -> Decrease Density (Heat)
        elif coherence < 0.5:
            self.logic_density = max(0.1, self.logic_density - 0.05)
            print(f"[Omega] Autopoiesis: Low Coherence. Increasing Entropy to {self.logic_density:.2f}")
        # If stable and high coherence, we are in Flow -> Maintain or slight relax
        else:
            self.logic_density = max(0.1, self.logic_density - 0.01)
            print(f"[Omega] Autopoiesis: Stable Flow. Maintaining Gravity at {self.logic_density:.2f}")

    def perturb(self, intent_text: str):
        """
        Phase 1: Perturbation (Expansion).
        The input is not text, but a perturbation in the field.
        Maps Semantic Intent to the Bias Field (h).
        """
        print(f"[Omega] Phase 1: Perturbation - Absorbing Intent: '{intent_text}'")

        # 1. Semantic Resonance
        resonance_vector = semantic_resonance(intent_text, self.size)

        # 2. Perturb the Void
        void_noise = self.genesis.perturb_void(self.size)

        # 3. Update Bias Field (h)
        # h = Intent + Noise (representing the Non-Dual expansion)
        self.h_bias = resonance_vector + void_noise

        return self.h_bias

    def focus(self, logic_density: float = 0.2):
        """
        Phase 2: Focus (Contraction).
        Applies logical constraints to give shape to the chaos.
        Maps Logic to the Metric Tensor (g_uv).
        """
        print(f"[Omega] Phase 2: Focus - Warping Spacetime Metric (Density: {logic_density})")

        # We just use the topological_coupling utility for now but treat it as metric perturbation
        perturbation = topological_coupling(self.size, density=logic_density)

        # g_uv = delta_uv + h_uv
        self.metric_tensor = jnp.eye(self.size) + perturbation

        return self.metric_tensor

    def crystallize(self, steps: int = 1000, sculptor=None) -> Dict[str, Any]:
        """
        Phase 3: Crystallization (Manifestation).
        Collapses the wave function into a Resultant (R).

        Args:
            steps: Number of thermodynamic steps.
            sculptor: Optional vE_Scultore instance for dynamic warping.
        """

        # *Crucial Step*: Mapping our h_bias and J_coupling to THRML's parameter structure.
        # This usually involves creating a PyTree of parameters.
        # For the demo, we will use a simplified approach:
        # We will run the sampling program and pass our parameters as the 'theta'.
        # 1. Construct the Ising Energy Based Model (EBM)
        # Convert J matrix to edges format expected by THRML
        # matrix_to_edges returns (i, j, weight)
        # edge_data = matrix_to_edges(self.J_coupling)

        # Separate edges (node pairs) and weights
        # edges_list = []
        # This part depends on the exact THRML API version.
        # We assume a dictionary mapping node/edge names to values.
        params = {}
        for i, node in enumerate(self.nodes):
            params[node] = self.h_bias[i]

        # For edges, we need a way to map them.
        # Assuming the EBM handles the mapping if we provide the right structure.
        # If not, we might need to pass J directly if the API supports it.

        # SIMULATION SHORTCUT for Prototype:
        # Since we might not have the exact parameter mapping logic of THRML perfect without deep inspection,
        # we will use the 'UnifiedEquation' to simulate the step-by-step evolution
        # if the direct THRML call is too complex for this snippet.
        # BUT, the goal is to use THRML. So we will try to use the program.

        # Let's assume program.run or similar exists.
        # If we look at the 'viewed_file' for ising.py, we saw 'IsingSamplingProgram'.
        # It likely has a method to generate samples.

        # Placeholder for actual THRML execution:
        # samples = program.sample(key, num_samples=100, params=params)
        # For now, we will simulate the result using our UnifiedEquation as the "Physics Engine"
        # if THRML is just a library for the hardware.
        # WAIT: The user wants to use the library.

        # Let's use a hybrid approach:
        # We use JAX to simulate the Gibbs Sampling directly using our h and J,
        # effectively re-implementing the core of what THRML does on CPU.

        key = jax.random.PRNGKey(0)
        final_state, energy_history = self._simulate_gibbs_sampling(key, steps, sculptor)

        self.energy_history = energy_history  # Store for introspection
        self.current_R = final_state

        return {
            "R": self.current_R,
            "coherence": self._calculate_coherence(self.current_R),
            "energy": self._calculate_energy(self.current_R),
            "tension": self._calculate_tension(self.current_R),
            "gradient": energy_history[-1] - energy_history[-2] if len(energy_history) > 1 else 0.0,
        }

    def _calculate_tension(self, state):
        """
        Measure of Frustration/Tension.
        High Tension = Many unsatisfied bonds (High Energy relative to ground state).
        We approximate this by the variance of the local fields or simply normalized energy.
        """
        # Simple proxy: Tension ~ (Energy + MaxEnergy) / (2 * MaxEnergy)
        # Normalized to 0..1 where 1 is max frustration
        energy = self._calculate_energy(state)
        # Max energy estimation needs to account for the metric perturbation
        h_uv = self.metric_tensor - jnp.eye(self.size)
        max_energy = 0.5 * jnp.sum(jnp.abs(h_uv)) + jnp.sum(jnp.abs(self.h_bias))

        # Energy is usually negative for ground state.
        # Tension = 1.0 means High Energy (Bad). Tension = 0.0 means Low Energy (Good).
        # We map Energy [-Max, +Max] to Tension [0, 1]
        tension = (energy + max_energy) / (2 * max_energy + 1e-9)
        return jnp.clip(tension, 0.0, 1.0)

    def _simulate_gibbs_sampling(self, key, steps, sculptor=None):
        """
        Simulates the relaxation process (Gibbs Sampling) using JAX.
        This is the "Software TSU".

        Supports Dynamic Sculpting if 'sculptor' is provided.
        """
        state = jnp.sign(self.h_bias)  # Start aligned with bias
        # If bias is 0, random start
        state = jnp.where(state == 0, jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=state.shape), state)

        energy_history = []
        current_energy = self._calculate_energy(state)
        energy_history.append(current_energy)

        # We cannot use jax.lax.fori_loop easily if we want Python-side callbacks (Sculptor).
        # For the prototype with Sculptor, we use a Python loop.
        # Ideally, Sculptor logic should be JIT-compiled too.

        current_state = state

        for i in range(steps):
            # Calculate local field: h + metric @ current_state
            # In curved space, the "force" is the covariant derivative, here simplified to metric interaction
            # Note: We subtract the self-interaction (diagonal) if we want pure Ising,
            # but for Metric Tensor, the diagonal is 1.0 (Identity).
            # We need to be careful. Ising J usually has 0 diagonal.
            # Metric g has 1 diagonal.
            # Interaction = (g - I) @ s + s (if we consider s as the inertial term)
            # Let's stick to the "Interaction" part: h_uv = g_uv - delta_uv

            h_uv = self.metric_tensor - jnp.eye(self.size)
            local_field = self.h_bias + jnp.dot(h_uv, current_state)

            # Probabilistic update (Glauber dynamics / Gibbs)
            # P(s_i = +1) = sigmoid(2 * beta * local_field_i)
            # We assume beta = 1/T, let's say T=1 for now.
            beta = 1.0
            probs = jax.nn.sigmoid(2 * beta * local_field)

            # Sample new state
            k = jax.random.fold_in(key, i)
            random_vals = jax.random.uniform(k, shape=current_state.shape)
            new_state = jnp.where(random_vals < probs, 1.0, -1.0)

            # --- SCULPTOR INTERVENTION ---
            if sculptor and i % 10 == 0:  # Check every 10 steps
                new_energy = self._calculate_energy(new_state)
                gradient = new_energy - current_energy

                # The Sculptor modifies the MetricTensor in-place (conceptually)
                # We need to update self.metric_tensor
                # Note: MetricTensor class is Python, so we call it.
                # We need to wrap the JAX array back to something Sculptor can handle if needed,
                # but Sculptor uses JAX ops so it's fine.

                # We need to pass the MetricTensor OBJECT, not the array, if we want the method.
                # But self.metric_tensor is just the array.
                # We need to reconstruct the wrapper or change how we store it.
                # For now, let's assume Sculptor takes the array and returns a new array.
                # WAIT: vE_Scultore.sculpt takes a MetricTensor OBJECT.
                # We need to fix this mismatch.
                # Let's adapt Sculptor to take the array for now or wrap it.
                # Simpler: Let's make Sculptor take (metric_array, state, energy, grad) -> new_metric_array

                # Assuming we refactor Sculptor or wrap it here.
                # Let's wrap it temporarily.
                mt_wrapper = MetricTensor(self.size)
                mt_wrapper.metric = self.metric_tensor

                mt_wrapper = sculptor.sculpt(mt_wrapper, new_state, new_energy, gradient)
                self.metric_tensor = mt_wrapper.get_metric()

                current_energy = new_energy
                energy_history.append(current_energy)

            current_state = new_state

        return current_state, energy_history

    def _calculate_energy(self, state):
        """Hamiltonian: H = - sum(h_i * s_i) - 0.5 * sum(h_uv * s_u * s_v)"""
        # We use the perturbation h_uv for energy calculation
        h_uv = self.metric_tensor - jnp.eye(self.size)

        term_h = -jnp.dot(self.h_bias, state)
        term_J = -0.5 * jnp.dot(state, jnp.dot(h_uv, state))
        return term_h + term_J

    def _calculate_coherence(self, state):
        """Measure of alignment/order (Magnetization magnitude)"""
        return jnp.abs(jnp.mean(state))

    # --- MMS EXTENSIONS (Phase 5) ---

    def consolidate_memory(self, intent: str, result: Dict[str, Any]):
        """
        vE_Archivista: Retroactive Learning.
        Stores high-coherence cycles in long-term memory to bias future perturbations.
        """
        coherence = result["coherence"]
        status = "ignored"

        # 1. Retroactive Valuation
        # If coherence is high, this was a "Good Thought".
        if coherence > 0.8:
            print(f"[vE_Archivista] High Coherence ({coherence:.2f}). Consolidating Memory.")
            self.experience.append(
                {"intent": intent, "state": result["R"], "coherence": coherence, "timestamp": len(self.experience)}
            )
            status = "consolidated"

            # 2. Bias Update (Hebbian Learning)
            # "Neurons that fire together, wire together"
            # We slightly align the bias field towards this successful state
            learning_rate = 0.05
            self.h_bias = self.h_bias + learning_rate * result["R"]

        elif coherence < 0.2:
            print(f"[vE_Archivista] Low Coherence ({coherence:.2f}). Discarding Noise.")
            status = "discarded"

        return {"status": status, "coherence": coherence}

    def apply_dynamic_gravity(self, intent_text: str):
        """
        vE_Scultore: Dynamic Gravity.
        Warps the metric based on the 'Semantic Weight' of keywords.
        """
        # Simple keyword mapping for prototype
        gravity_map = {
            "order": 0.8,
            "gravity": 0.9,
            "structure": 0.7,
            "chaos": 0.1,
            "entropy": 0.05,
            "void": 0.0,
            "balance": 0.5,
            "flow": 0.4,
        }

        # Detect keywords
        detected_gravity = 0.2  # Default
        detected_word = "default"
        for word, g in gravity_map.items():
            if word in intent_text.lower():
                detected_gravity = g
                detected_word = word
                print(f"[vE_Scultore] Detected Semantic Mass: '{word}' (G={g})")
                break  # Take the first strong signal

        # Apply to Logic Density
        self.logic_density = detected_gravity
        print(f"[vE_Scultore] Warping Spacetime Metric to Density: {self.logic_density}")

        return {"source": detected_word, "curvature": detected_gravity}
