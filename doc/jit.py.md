# jit.py

## Overview
A Just-In-Time (JIT) compilation prototype for the Paradox Framework. It provides tools for capturing the execution trace of a function and "compiling" it into a static computational graph to reduce Python execution overhead.

## Purpose
The JIT module aims to improve training and inference speed by capturing a sequence of operations once and executing them as a single "fused" block, skipping the overhead of multiple Python calls and redundant tensor wrapping.

## Key Components

### `StaticGraph` (Class)
Represents a recorded sequence of operations.
- **`capture(func, *example_inputs)`**: Traces the provided function by running it with `TracerTensor` objects.
- **`optimize_graph(output_tracer)`**: Analyzes the recorded history to find optimization opportunities, such as operator fusion (e.g., combining an `add` followed by a `relu` into a single loop).
- **`__call__(*inputs)`**: Executes the optimized operation sequence on real data.

### `TracerTensor` (Class)
A "fake" tensor used during the capture phase. Instead of performing math, it records every operation called on it (e.g., `+`, `relu`, `sum`) into a history list.

### `@jit_compile` (Decorator)
A convenience decorator to wrap functions for JIT compilation.
- **Mechanism**: The first time the function is called, it triggers `StaticGraph.capture`. Subsequent calls use the captured graph.

## Current Limitations (Prototype Status)
- **Eager Fallback**: The current implementation primarily serves as a tracer; it captures the graph but falls back to eager execution for safety in the prototype phase.
- **Operation Support**: Only a subset of operations (like `add` and `relu`) are currently being traced.
- **Fusion**: The automatic fusion logic is a placeholder for future advanced implementation using C++/CUDA kernels.

## Future Vision
The JIT compiler is designed to eventually interface with the `C-Bridge` to generate and compile dynamic kernels on-the-fly, allowing the Python-based framework to achieve C-level performance for customized layer architectures.
