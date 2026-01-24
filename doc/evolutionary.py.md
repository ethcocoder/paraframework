# evolutionary.py

## Overview
An experimental module implementing "Thinking" or "Associative Reasoning" layers. It uses physics-inspired simulations to evolve latent representations towards stable "attractor states" before producing an output.

## Purpose
Classic neural networks are static (one pass forward). The `evolutionary.py` module introduces a dynamic dimension where the "thought" can evolve over multiple time steps, potentially allowing the AI to "think" its way out of noisy inputs or towards more coherent stored memories.

## Key Components

### `ThinkingModule` (Class)
A subclass of `Module` that integrates a `SimulationEnv` to process data.

- **`__init__(memory_engine, steps=5, dt=0.1)`**: 
  - `memory_engine`: The backend for retrieving attractor states.
  - `steps`: How many "brain cycles" or simulation steps to run per forward pass.
  - `dt`: The time delta for each physics step.

- **`thought_dynamics(vectors, dt, backend)`**: 
  - Represents the "Laws of Physics" for the AI's internal reasoning.
  - **Associative Reasoning**: Implements a simple attractor logic where thoughts in a batch tend to drift towards a common center (simulating "groupthink" or coherence enhancement).
  - **Drift Logic**: Calculates the difference from the mean and applies a restorative force.

- **`forward(x)`**:
  - Takes a standard latent tensor.
  - Unwraps the data and runs it through the simulation loop for `self.steps` iterations.
  - Re-wraps the "evolved" state back into a `Tensor`.

## Concept: Attractor States
In biological brains and Hopfield networks, memories are seen as low-energy "valleys" (attractors). If an AI's internal state is near a memory, the `ThinkingModule` effectively pushes the state into that valley over several iterations, improving clarity and recall.

## Integration
- **Paradma Connection**: Relies on `paradma` for underlying mathematical consistency.
- **Simulation Connection**: Uses `paradox.simulation` to manage the environment where thoughts exist.

## Performance Note
Because the simulation happens inside the forward pass, using a `ThinkingModule` increases inference time proportionally to the number of `steps`. It is intended for high-quality reasoning tasks where "fast-thinking" (immediate response) is less important than "slow-thinking" (deliberative reasoning).
