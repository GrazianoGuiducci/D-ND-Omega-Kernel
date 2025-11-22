"""
D-ND Omega Kernel: Performance Benchmark
Tests the scaling of the Autological Cycle on current hardware.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import jax
from dnd_kernel.omega import OmegaKernel

def run_benchmark(size, steps=1000):
    print(f"\n--- Benchmarking Size: {size} Nodes ---")
    
    # Initialize
    start_init = time.time()
    kernel = OmegaKernel(size=size, seed=42)
    end_init = time.time()
    print(f"Initialization Time: {end_init - start_init:.4f}s")
    
    # JIT Compilation Warmup (First run is always slower due to JAX compilation)
    print("Warming up JIT compilation...")
    kernel.process_intent("warmup", steps=10)
    
    # Actual Run
    start_run = time.time()
    kernel.process_intent("Benchmark Intent", steps=steps)
    end_run = time.time()
    
    duration = end_run - start_run
    print(f"Execution Time ({steps} steps): {duration:.4f}s")
    print(f"Steps per Second: {steps / duration:.2f}")
    
    return duration

def main():
    print("========================================")
    print("   D-ND OMEGA KERNEL: SCALING TEST      ")
    print("========================================")
    print(f"JAX Backend: {jax.devices()[0]}")
    
    sizes = [100, 500, 1000, 2000]
    results = {}
    
    for size in sizes:
        try:
            duration = run_benchmark(size)
            results[size] = duration
        except Exception as e:
            print(f"Failed at size {size}: {e}")
            break
            
    print("\n========================================")
    print("          SUMMARY RESULTS               ")
    print("========================================")
    for size, duration in results.items():
        print(f"Nodes: {size:<5} | Time: {duration:.4f}s")

if __name__ == "__main__":
    main()
