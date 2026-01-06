# Paradox Framework 🌌

A next-generation, high-performance deep learning framework built for the **Ultimate Paradox** ecosystem. It combines PyTorch-like flexibility with advanced cognitive and quantum-inspired capabilities.

## 🚀 Key Features

### 1. **Core Engine**
- **Autograd Ready**: Full automatic differentiation with a custom computation graph.
- **Paradma Backend**: Optimized for CPU performance via `Paradma`, with seamless `CuPy` support for GPU acceleration.
- **Tensor Axioms**: Mathematical operations designed for high-dimensional latent space manipulation.

### 2. **Cognitive & Evolutionary Learning**
- **Cognitive Training**: Integrated with `LatentMemoryEngine` for memory-augmented learning.
- **Dreaming & Sleep**: Autonomous "dream" cycles for latent space exploration and clustering dynamics during "sleep".
- **Emotional Regulation**: Training loops influenced by `AIEmotions` (Reflexion, Inceptio, etc.), dynamically adjusting learning rates based on model "state".
- **Evolutionary Optimization**: Built-in modules for genetic and evolutionary parameter tuning.

### 3. **Quantum-Inspired Neural Ops**
- **Superposition Tensors**: Hold multiple conflicting hypotheses simultaneously using `SuperpositionTensor`.
- **Spooky Action (Entanglement)**: Parameters can be linked via `EntanglementManager`, allowing gradients to flow across "entangled" layers even without direct graph connections.

### 4. **Modern DL Utilities**
- **NN Module**: PyTorch-style `Module`, `Parameter`, and functional layers.
- **Optimizers**: Advanced `AdamW` and `SGD` with momentum.
- **Mixed Precision (AMP)**: Support for float16 training to optimize memory and speed.
- **Quantization**: Tools for model compression and efficient inference.

---

## 🛠 Installation

The framework is part of the Paradox ecosystem. Ensure you have the `paradma` backend available for maximum performance.

```bash
# Ensure dependencies are met
pip install numpy cupy-cuda12x  # (Choose cupy version based on your CUDA)
```

---

## 💻 Quick Start

### Basic Tensor Operations
```python
from modules.framework.tensor import Tensor

# Create tensors
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0])

# Math operations
z = (x * y).sum()
z.backward()

print(x.grad) # Gradient calculated automatically
```

### Building a Module
```python
from modules.framework.module import Module
from modules.framework.nn.layers import Linear

class ParadoxModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(128, 64)
        self.fc2 = Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

model = ParadoxModel()
```

### Advanced: Superposition & Entanglement
```python
from modules.framework.quantum import SuperpositionTensor, EntanglementManager

# Create a state in superposition
states = [Tensor([1, 0]), Tensor([0, 1])]
q_tensor = SuperpositionTensor(states)

# Collapse to a single hypothesis
result = q_tensor.collapse()

# Entangle two parameters
manager = EntanglementManager()
manager.entangle(model.fc1.weight, model.fc2.weight, strength=0.2)

# Gradients will now sync between fc1 and fc2
manager.sync_gradients()
```

---

## 🧠 Cognitive Training Loop

Unlike standard frameworks, Paradox supports **Cognitive Training**:

```python
from modules.framework.cognitive import CognitiveTrainer

trainer = CognitiveTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    memory_engine=my_latent_memory
)

# Standard training step + Emotional feedback + Memory formation
trainer.train_step(inputs, targets)

# Run a dream cycle to consolidate knowledge
trainer.dream(num_dreams=10)
```

---

## 📁 Project Structure

- `tensor.py`: Core multidimensional array with autograd.
- `module.py`: Base class for neural network components.
- `nn/`: Common layers (`Linear`), loss functions, and activation functions.
- `optim.py`: Optimizers (`AdamW`, `SGD`).
- `quantum.py`: Quantum-inspired logic (Superposition, Entanglement).
- `cognitive.py`: High-level trainers with memory and dreaming.
- `device.py`: Backend management (CPU/GPU).
- `ops/`: Low-level mathematical operation kernels.

---

## 📜 License
Internal Paradox AI Framework. proprietary.
