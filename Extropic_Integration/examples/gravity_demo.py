"""
Gravity Demo: Visualizing Extropic Hardware Dynamics.

This script demonstrates the "Physics of Thought" by visualizing:
1. The Nulla-Tutto Potential (The Indeterminate Void).
2. The Assonance Matrix (The Semantic Gravity/Topology).
3. The Collapse (The Emergence of a Stable Thought).

Scientific Rigor:
- Uses Matplotlib to plot Phase Space and Energy Landscapes.
- Visualizes the 'Transfer Function' from Potential to Resultant.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Extropic_Integration.hardware_dynamics.combinatorial import (
    MetricTensor, 
    nulla_tutto_potential, 
    transfer_function
)
from Extropic_Integration.hardware_dynamics.metrics import curvature_index, cycle_stability
from Extropic_Integration.dnd_kernel.omega import OmegaKernel

def visualize_potential(size: int, ax):
    """Plots the Nulla-Tutto Potential (White Noise / Vacuum Energy)."""
    potential = nulla_tutto_potential(size, scale=0.5)
    # Add some random noise to simulate quantum fluctuations
    noise = np.random.normal(0, 0.1, size)
    field = potential + noise
    
    ax.bar(range(size), field, color='lightgray', alpha=0.5, label='Vacuum Potential')
    ax.plot(range(size), field, color='gray', linestyle='--', alpha=0.7)
    ax.set_title("Phase 1: Nulla-Tutto Potential (Indeterminate)")
    ax.set_ylim(-1.5, 1.5)
    ax.legend()

def visualize_curvature(metric, ax):
    """Plots the Metric Perturbation (h_uv) as a Heatmap (Gravity Well)."""
    # We subtract Identity to show only the warping (Gravity)
    h_uv = metric - np.eye(metric.shape[0])
    cax = ax.imshow(h_uv, cmap='coolwarm', interpolation='nearest')
    ax.set_title("Phase 2: Spacetime Metric (Gravity)")
    return cax

def visualize_collapse(states_history, ax):
    """Plots the evolution of the state vector (The Thought)."""
    # states_history is (steps, size)
    steps = len(states_history)
    size = states_history[0].shape[0]
    
    # We plot the 'Magnetization' (Average State) over time
    magnetization = [np.mean(s) for s in states_history]
    
    ax.plot(range(steps), magnetization, color='purple', linewidth=2, label='Coherence (Order)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title("Phase 3: Collapse (Emergence of Resultant)")
    ax.set_xlabel("Thermodynamic Steps")
    ax.set_ylabel("Coherence")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()

def run_demo():
    print("Initializing Gravity Demo (Metric Tensor Edition)...")
    size = 50
    kernel = OmegaKernel(size=size)
    
    # 1. Define Intent (Perturbation)
    intent = "Create Order from Chaos"
    print(f"Intent: {intent}")
    kernel.perturb(intent)
    
    # 2. Define Topology (Focus)
    # We manually inject some strong gravity to ensure a nice plot
    mt = MetricTensor(size)
    # Create a "Black Hole" (Attractor) block in the middle
    for i in range(20, 30):
        for j in range(20, 30):
            if i != j:
                mt.warp_space(i, j, 0.5)
    
    # Inject this metric into the kernel
    kernel.metric_tensor = mt.get_metric()
    
    # 3. Crystallize (Collapse)
    # We run the cycle step-by-step to capture history
    steps = 50
    history = []
    
    # Initial state (Random / High Temp)
    current_state = np.random.choice([-1.0, 1.0], size=size)
    history.append(current_state)
    
    print("Collapsing Field...")
    for t in range(steps):
        # Physics update (using our combinatorial transfer function for demo)
        potential = kernel.h_bias
        metric = kernel.metric_tensor
        
        # S_new = tanh(Potential + Metric @ S_old)
        # We add a 'temperature' factor to simulate annealing
        temp = max(0.1, 1.0 - (t / steps)) # Cooling down
        
        # JAX to Numpy for plotting
        pot_np = np.array(potential)
        met_np = np.array(metric)
        
        # Calculate field (Covariant flow)
        # Interaction = Metric @ State
        # Note: Metric includes Identity, so this includes self-persistence
        field = pot_np + met_np @ current_state
        
        # Update state (Soft spin)
        current_state = np.tanh(field / temp)
        history.append(current_state)
        
    final_coherence = np.abs(np.mean(current_state))
    print(f"Final Coherence: {final_coherence:.4f}")
    
    # --- Visualization ---
    print("Generating Visualization...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Potential
    visualize_potential(size, ax1)
    
    # Plot 2: Metric (Perturbation)
    cax = visualize_curvature(kernel.metric_tensor, ax2)
    fig.colorbar(cax, ax=ax2)
    
    # Plot 3: Collapse
    visualize_collapse(history, ax3)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "gravity_plot.png")
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    
    # Also print ASCII art for the user
    print("\n--- ASCII Resultant ---")
    ascii_state = "".join(["#" if s > 0 else "." for s in current_state])
    print(f"[{ascii_state}]")

if __name__ == "__main__":
    run_demo()
