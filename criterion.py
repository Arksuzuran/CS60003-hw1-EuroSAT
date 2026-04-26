import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        self.labels = labels
        N = logits.shape[0]

        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        correct_probs = self.probs[np.arange(N), labels]
        correct_probs = np.clip(correct_probs, 1e-15, 1.0)
        
        loss = -np.sum(np.log(correct_probs)) / N
        return loss

    def backward(self):
        N = self.probs.shape[0]
        grad_logits = self.probs.copy()

        grad_logits[np.arange(N), self.labels] -= 1.0
        
        grad_logits /= N
        
        return grad_logits