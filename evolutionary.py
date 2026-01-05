from modules.framework.module import Module
from modules.framework.tensor import Tensor
from modules.framework.device import device
from paradox.simulation import SimulationEnv
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy

class ThinkingModule(Module):
    """
    A Module that 'Thinks' (Simulates) before answering.
    It takes the latent encoding, runs it through a Physics Simulation
    to evolve the thought, and then returns the result.
    """
    def __init__(self, memory_engine, steps=5, dt=0.1):
        super().__init__()
        self.memory_engine = memory_engine
        self.sim = SimulationEnv(memory_engine)
        self.steps = steps
        self.dt = dt
        
    def thought_dynamics(self, vectors, dt, backend):
        """
        The 'Laws of Physics' for independent thought.
        Here: Thoughts tend to drift towards 'Attractor States' (stored memories).
        This models 'Associative Reasoning'.
        """
        # 1. Find nearest memory (Attractor)
        # This is expensive, so we do a simplified version using the engine's index
        # For differentiation, we'd need this to be differentiable.
        # Currently SimulationEnv logic is usually non-differentiable (numpy backend).
        # So this module is primarily for INFERENCE enhancement.
        
        # Simple Logic: Consolidate.
        # Move slightly towards average of batch (Groupthink)
        if len(vectors) > 1:
             center = np.mean(vectors, axis=0)
             # vector -> center
             diff = center - vectors
             return diff * 0.1 * dt
        return np.zeros_like(vectors)

    def forward(self, x):
        """
        x: (Batch, Dim) latent vector
        """
        # 1. Inject into Simulation
        # We need to temporarily 'load' x into the engine to simulate it
        # But engine usually stores persistent memory.
        # We can run a 'hypothetical' simulation on raw vectors without the engine state loop
        
        # Manually run simulation loop on input tensor
        current_state = x.data
        if hasattr(current_state, 'cpu'): current_state = current_state.cpu().numpy()
        
        for _ in range(self.steps):
            delta = self.thought_dynamics(current_state, self.dt, "numpy")
            current_state += delta
            
        # Return as Tensor
        return Tensor(current_state, device_type=x.device)
