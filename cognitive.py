import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

from modules.framework.tensor import Tensor
from modules.framework.device import device
from paradox.engine import LatentMemoryEngine
from paradox.mixer import ParadoxMixer
from paradox.simulation import SimulationEnv
from modules.self_awareness.ai_emotions import AIEmotions
import numpy as np # [PARADMA] Replacing Numpy

class CognitiveTrainer:
    def __init__(self, model, optimizer, criterion, memory_engine: LatentMemoryEngine, reconstruction=False):
        from modules.curiosity.question_generator import QuestionGenerator as CuriosityModule

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = memory_engine
        self.reconstruction = reconstruction
        
        self.sim_env = SimulationEnv(self.memory)
        self.emotions = AIEmotions(paradox_engine=self.memory)
        self.curiosity = CuriosityModule(latent_engine=self.memory)

    def _extract_logits(self, outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _pad_vector_to_memory(self, vec):
        vec_flat = vec.flatten()
        if vec_flat.shape[0] != self.memory.dimension:
            padded = np.zeros(self.memory.dimension, dtype=vec_flat.dtype)
            size = min(vec_flat.shape[0], self.memory.dimension)
            padded[:size] = vec_flat[:size]
            vec_flat = padded
        return vec_flat

    def train_step(self, inputs, targets, remember_prob=0.1):
        # 1. Forward
        outputs = self.model(inputs)
        logits = self._extract_logits(outputs)
        
        # Reshape for sequence loss if necessary
        if len(logits.data.shape) == 3:
            B, T, V = logits.data.shape
            logits = Tensor(logits.data.reshape(B*T, V))
            targets = Tensor(targets.data.reshape(B*T))
            
        logits_data = logits.data if isinstance(logits, Tensor) else logits
        targets_data = targets.data if isinstance(targets, Tensor) else targets
        
        loss = self.criterion(Tensor(logits_data), Tensor(targets_data))
        loss_val = float(loss.data) if hasattr(loss.data, 'item') else float(loss.data)
        
        # 2. Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Emotional Regulation
        feedback_signal = min(loss_val, 1.0)
        self.emotions.update_from_feedback(feedback_signal)
        
        state = self.emotions.get_state()
        reflexion = state['Reflexion']
        
        base_lr = self.optimizer.defaults.get('lr', 0.001)
        new_lr = base_lr * (1.0 - 0.5 * reflexion)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.optimizer.step()
        
        # 3. Memory Formation (SAFEGUARDED)
        inceptio = state['Inceptio']
        adjusted_prob = remember_prob * (1.0 + inceptio)
        
        if np.random.rand() < adjusted_prob:
            # Safe extraction of numpy data
            data_np = inputs.data
            if hasattr(data_np, 'get'): data_np = data_np.get() # Handle CuPy
            elif hasattr(data_np, 'cpu'): data_np = data_np.cpu().numpy() # Handle Torch
            
            target_np = targets.data
            if hasattr(target_np, 'get'): target_np = target_np.get()
            elif hasattr(target_np, 'cpu'): target_np = target_np.cpu().numpy()
            
            # Ensure we have data to sample
            if len(data_np) > 0:
                idx = np.random.randint(0, len(data_np))
                vec = data_np[idx]
                
                # Handle target extraction safely for scalars or arrays
                try:
                    if np.ndim(target_np) == 0:
                        label = int(target_np)
                    elif target_np.ndim == 1 and len(target_np) > idx:
                         # Likely a list of scalar labels
                         label = int(target_np[idx])
                    else:
                        label = 0 
                except:
                    label = 0

                vec_flat = self._pad_vector_to_memory(vec)
                self.memory.add(vec_flat, attributes={
                    "label": label,
                    "origin": "training",
                    "emotional_context": state
                })
            
        return loss.data

    def dream(self, num_dreams=5):
        # We check the ACTUAL length of the vector storage
        if hasattr(self.memory.vectors, 'shape'):
             real_count = self.memory.vectors.shape[0]
        else:
             real_count = len(self.memory.vectors)

        if real_count < 2:
            return 0.0
        
        state = self.emotions.get_state()
        inceptio = state['Inceptio']
        
        losses = []
        for _ in range(num_dreams):
            ids = np.random.choice(real_count, 2, replace=False)
            
            vec_a = self.memory.vectors[ids[0]]
            vec_b = self.memory.vectors[ids[1]]
            
            try:
                attr_a = self.memory.retrieve(int(ids[0]))
                attr_b = self.memory.retrieve(int(ids[1]))
                label = attr_a.get('label', 0) if np.random.rand() > 0.5 else attr_b.get('label', 0)
            except:
                label = 0
            
            if inceptio > 0.7:
                ratio = 0.5 + (np.random.rand() - 0.5) * 0.2
            else:
                ratio = 0.1
            
            mixed_vec = ParadoxMixer.interpolate(vec_a, vec_b, ratio=ratio)
            mixed_vec = self._pad_vector_to_memory(mixed_vec)
            
            # Create tensors on the correct device
            inp = Tensor(np.expand_dims(mixed_vec, 0), device_type=device.current_device)
            
            if self.reconstruction:
                tgt = inp
            else:
                tgt = Tensor(np.array([label]), device_type=device.current_device)
            
            outputs = self.model(inp)
            logits = self._extract_logits(outputs)
            loss = self.criterion(logits, tgt)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            l_val = float(loss.data)
            self.emotions.update_from_curiosity(l_val * 0.5)
            
            self.optimizer.step()
            losses.append(loss.data)
            
        return np.mean(losses) if losses else 0.0

    def latent_replay_step(self, batch_size=32):
        if self.memory.count < batch_size:
            return 0.0
        indices = np.random.choice(self.memory.count, batch_size, replace=False)
        vectors = self.memory.vectors[indices]
        targets = []
        for idx in indices:
            attrs = self.memory.retrieve(int(idx))
            targets.append(attrs.get("label", 0))
        input_tensor = Tensor(vectors, device_type=device.current_device)
        target_tensor = Tensor(np.array(targets), device_type=device.current_device)
        outputs = self.model(input_tensor)
        logits = self._extract_logits(outputs)
        loss = self.criterion(logits, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data

    def sleep(self, duration=10):
        def clustering_dynamics(vectors, dt, backend):
            return -0.001 * vectors * dt
        print(f"Paradox AI is sleeping... (State: {self.emotions.get_state()})")
        self.sim_env.run(steps=duration, dynamics_fn=clustering_dynamics, dt=1.0)