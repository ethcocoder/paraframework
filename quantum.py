import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy
from modules.framework.tensor import Tensor
from modules.framework.device import device
from modules.framework.module import Parameter

class SuperpositionTensor:
    """
    Represents a quantum-like state: Sum(alpha_i * |state_i>).
    Allows the network to hold multiple conflicting hypotheses simultaneously.
    """
    def __init__(self, states, amplitudes=None):
        """
        states: List of Tensors (all same shape)
        amplitudes: List of floats (complex numbers supported ideally, but using floats for now).
                    Must sum to 1.0 (approximated).
        """
        self.states = states
        if amplitudes is None:
            # Equal superposition
            n = len(states)
            self.amplitudes = [1.0 / np.sqrt(n)] * n
        else:
            self.amplitudes = amplitudes

    def collapse(self):
        """
        'Measure' the state. Collapses to a single Tensor based on probability = amplitude^2.
        """
        xp = device.backend
        probs = [a**2 for a in self.amplitudes]
        
        # Normalize probs just in case
        total_p = sum(probs)
        probs = [p/total_p for p in probs]
        
        # Sample
        idx = np.random.choice(len(self.states), p=probs)
        return self.states[idx]

    def expected_value(self):
        """
        Returns the weighted average (Mean Field Theory).
        Useful for continuous inference without collapse.
        """
        result = self.states[0] * self.amplitudes[0]
        for i in range(1, len(self.states)):
            result = result + (self.states[i] * self.amplitudes[i])
        return result

    def apply(self, func):
        """
        Apply a function (Layer) to all states in superposition.
        Returns a new SuperpositionTensor.
        """
        new_states = [func(s) for s in self.states]
        return SuperpositionTensor(new_states, self.amplitudes)

class EntanglementManager:
    """
    Manages 'Spooky Action at a Distance' between network parameters.
    If Parameter A is entangled with Parameter B, gradients flowing into A
    will partially flow into B, even if B was not in the computation graph.
    """
    def __init__(self):
        # List of (param_a, param_b, strength)
        self.links = []

    def entangle(self, param_a, param_b, strength=0.1):
        """
        Links two parameters. Strength determins correlation (0.0 to 1.0).
        """
        self.links.append((param_a, param_b, strength))

    def sync_gradients(self):
        """
        Apply entanglement logic. Call this BEFORE optimizer.step().
        """
        for p_a, p_b, strength in self.links:
            if p_a.grad is not None and p_b.grad is not None:
                # Correlated updates: G_a += s * G_b
                # If they learned related concepts, they reinforce.
                
                # Copy data to avoid modification during iteration issues or loops
                ga = p_a.grad.data
                gb = p_b.grad.data
                
                # We add a fraction of the other's gradient
                # Using ._data to write directly
                p_a.grad._data += strength * gb
                p_b.grad._data += strength * ga

class QuantumLayer(Parameter): # Placeholder
    pass
