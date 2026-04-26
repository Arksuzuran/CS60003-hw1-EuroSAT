import numpy as np

class SGD:
    def __init__(self, model, lr=0.01, weight_decay=0.0, lr_decay=1.0):
        self.model = model
        self.lr = lr
        self.initial_lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.epoch_count = 0

    def step(self):
        for p_dict in self.model.get_params_and_grads():
            param = p_dict['param']
            grad = p_dict['grad']
            
            if self.weight_decay > 0 and param.ndim > 1:
                grad_final = grad + self.weight_decay * param
            else:
                grad_final = grad
                
            param -= self.lr * grad_final

    def zero_grad(self):
        for p_dict in self.model.get_params_and_grads():
            if p_dict['grad'] is not None:
                p_dict['grad'].fill(0.0)

    def step_lr(self):
        self.epoch_count += 1
        self.lr = self.initial_lr * (self.lr_decay ** self.epoch_count)
        