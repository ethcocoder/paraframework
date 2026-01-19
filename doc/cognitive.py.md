# Cognitive Trainer - Line by Line Explanation

**File:** `modules/framework/cognitive.py`

This module defines the high-level **training loops and cognitive cycles** (Learning, Dreaming, Sleeping). It integrates the Model with the Memory Engine and Emotional System.

## Imports and Setup

```python
3: sys.path.append(os.path.join(os.getcwd(), "paradma"))
...
7: from paradox.engine import LatentMemoryEngine
...
10: from modules.self_awareness.ai_emotions import AIEmotions
```
**Lines 1-12:** Imports core components. It links the neural network world (`modules.framework`) with the agent world (`paradox.*`, `modules.self_awareness`).

## Class: CognitiveTrainer

```python
13: class CognitiveTrainer:
14:     def __init__(self, model, optimizer, ...):
...
23:         self.sim_env = SimulationEnv(self.memory)
...
```
**Lines 13-26:** Initialization. Initializes the "Environment", "Emotions", and "Curiosity" subsystems alongside standard Deep Learning components (model, optimizer).

### Training Step: The Core Loop
```python
41:     def train_step(self, inputs, targets, remember_prob=0.1):
...
63:         feedback_signal = min(loss_val, 1.0)
64:         self.emotions.update_from_feedback(feedback_signal)
```
**Lines 41-74:** Standard forward/backward pass calculated loss.
- **Lines 63-64:** **Emotional Feedback**. The loss value directly inputs into the emotion system. High loss -> Frustration/Fluxion. Low loss -> Equilibria.
- **Lines 69-72:** **Adaptive Learning Rate**. If `Reflexion` (meta-cognition) is high, the Learning Rate drops (processing slows down to "think harder").

### Memory Safeguard
```python
76:         # 3. Memory Formation (SAFEGUARDED)
80:         if np.random.rand() < adjusted_prob:
...
108:                 self.memory.add(vec_flat, ...)
```
**Lines 76-113:** Randomly converts training inputs into Long Term Memories. The probability is adjusted by `Inceptio` (openness to new ideas).

### Dreaming (Replay)
```python
116:     def dream(self, num_dreams=5):
...
148:             mixed_vec = ParadoxMixer.interpolate(vec_a, vec_b, ratio=ratio)
...
159:             outputs = self.model(inp)
```
**Lines 116-172:** "Dreaming".
- Picks two existing memories.
- **Mixes** them together (`ParadoxMixer.interpolate`).
- Trains the model on this synthetic data ("hallucination" as data augmentation).
- This consolidates memory and robustness.

### Sleep (Consolidation)
```python
193:     def sleep(self, duration=10):
194:         def clustering_dynamics(vectors, dt, backend):
195:             return -0.001 * vectors * dt
...
197:         self.sim_env.run(...)
```
**Lines 193-197:** "Sleep". Runs a physics simulation on the vector space. `clustering_dynamics` slightly shrinks vectors, organizing the manifold (simulating memory consolidation/pruning).
