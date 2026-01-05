from modules.framework.module import Module
from modules.framework.nn import functional as F

class _Loss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

class NLLLoss(_Loss):
    def __init__(self, reduction='mean', ignore_index=-100):
        super().__init__(reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.nll_loss(input, target, reduction=self.reduction, ignore_index=self.ignore_index)

class CrossEntropyLoss(_Loss):
    def __init__(self, reduction='mean', ignore_index=-100):
        super().__init__(reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.cross_entropy(input, target, reduction=self.reduction, ignore_index=self.ignore_index)
