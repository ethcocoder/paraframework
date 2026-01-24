"""
Knowledge Distillation Module
==============================

Enables transfer of intelligence from large 'Teacher' models to 
efficient 'Student' models within the Paradox framework.
"""

import typing as t
from modules.framework.tensor import Tensor
from modules.framework.module import Module
from modules.framework.nn import functional as F

class Distiller:
    """
    Manages the distillation training process.
    """
    def __init__(self, teacher: Module, student: Module, temperature: float = 2.0):
        self.teacher = teacher
        self.student = student
        self.temp = temperature
        
    def compute_distillation_loss(self, inputs: Tensor, alpha: float = 0.5) -> Tensor:
        """
        Loss = alpha * KL(Teacher_soft, Student_soft) + (1-alpha) * CE(Student, Target)
        
        Args:
            inputs: Training batch input
            alpha: Weight of distillation loss vs hard labels
        """
        # 1. Get Teacher Logits (Forward pass in eval mode)
        with F.no_grad():
            teacher_logits = self.teacher.forward(inputs)
            
        # 2. Get Student Logits
        student_logits = self.student.forward(inputs)
        
        # 3. Compute Soft Targets (Temperature scaling)
        soft_targets = (teacher_logits / self.temp).softmax(axis=-1)
        soft_student = (student_logits / self.temp).log_softmax(axis=-1)
        
        # 4. KL Divergence (Approximate)
        # Loss = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
        # Here we use soft_targets * (temp^2) factor as per Hinton et al.
        distill_loss = -(soft_targets * soft_student).sum(axis=-1).mean() * (self.temp ** 2)
        
        return distill_loss

    def match_features_loss(self, teacher_layer: Tensor, student_layer: Tensor) -> Tensor:
        """
        Encourages student intermediate layers to match teacher representations.
        Useful for transfer learning of deep structures.
        """
        # If dimensions don't match, we might need a projection layer (not handled here yet)
        if teacher_layer.shape != student_layer.shape:
             return Tensor(0.0)
             
        # Mean Squared Error between feature maps
        diff = teacher_layer - student_layer
        return (diff * diff).mean()

def train_distillation_step(distiller: Distiller, inputs: Tensor, optimizer: t.Any):
    """Execution wrapper for a single distillation step."""
    optimizer.zero_grad()
    loss = distiller.compute_distillation_loss(inputs)
    loss.backward()
    optimizer.step()
    return float(loss.data)
