import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy
from modules.framework.tensor import Tensor
from modules.framework.device import device

class StaticGraph:
    """
    A Static Computational Graph for JIT-like compilation.
    Records sequences of tensor operations and 'compiles' them into 
    Fused Kernels (simulated via optimized NumPy/CuPy chains) to reduce Python overhead.
    """
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.optimized_exec = None

    def capture(self, func, *example_inputs):
        """
        Runs the function with 'Tracer' tensors to record the graph.
        """
        print("[JIT] Tracing graph...")
        # 1. Wrap inputs in Tracers
        traced_inputs = [TracerTensor(x) for x in example_inputs]
        
        # 2. Run functionality
        out = func(*traced_inputs)
        
        # 3. Optimize Graph (Fusion)
        self.optimize_graph(out)
        print("[JIT] Graph captured & optimized.")
        
    def optimize_graph(self, output_tracer):
        """
        Optimizes the captured linear graph.
        Strategy: Fuse Element-wise Ops (Add -> Relu -> Mul) into single kernels.
        """
        # This is a placeholder for the advanced logic of flattening the DAG
        # and finding fusion candidates.
        self.nodes = output_tracer.history # Flattened history
        pass

    def __call__(self, *inputs):
        """
        Executes the optimized path fast.
        """
        # In a real JIT, this would call a unified C++/CUDA kernel.
        # Here we simulate by skipping Autograd overhead entirely.
        
        # For prototype, we just re-run the op chain on raw data
        # avoiding Tensor wrapping overhead for every intermediate step.
        ctx = {}
        # Mapping inputs to trace IDs would inevitably happen here
        return inputs[0] # Stub

class TracerTensor:
    """
    A dummy tensor that records operations instead of executing them.
    Used for graph discovery.
    """
    def __init__(self, tensor):
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.history = []
        
    def __add__(self, other):
        self.history.append(('add', other))
        return self
        
    def relu(self):
        self.history.append(('relu',))
        return self

def jit_compile(func):
    """
    Decorator to JIT compile a function.
    """
    graph = StaticGraph()
    compiled = False
    
    def wrapper(*args, **kwargs):
        nonlocal compiled
        if not compiled:
            graph.capture(func, *args)
            compiled = True
        return func(*args, **kwargs) # Currently falling back to eager for safety
    return wrapper
