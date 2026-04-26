import numpy as np

class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros((input_dim, output_dim))
        self.db = np.zeros((1, output_dim))
        self.x = None

    def forward(self, x):
        self.x = x
        return self.x @ self.W + self.b
    
    def backward(self, grad_output):
        self.dW[:] = self.x.T @ grad_output
        self.db[:] = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T
    
    def get_params_and_grads(self):
        return [
            {'param': self.W, 'grad': self.dW},
            {'param': self.b, 'grad': self.db},
        ]

class ReLU(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.x <= 0] = 0
        return grad_input

class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        x_clipped = np.clip(x, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.out
    
    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)

class Tanh(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)

class MLP(Layer):
    def __init__(self, input_dim, hidden_dim, num_classes, activation_type='relu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.activation_type = activation_type
        if activation_type == 'relu':
            ActLayer = ReLU
        elif activation_type == 'sigmoid':
            ActLayer = Sigmoid
        elif activation_type == 'tanh':
            ActLayer = Tanh
        else:
            raise ValueError(f"Activation type {activation_type} not supported")

        self.layers = [
            Linear(input_dim, hidden_dim),
            ActLayer(),
            Linear(hidden_dim, hidden_dim),
            ActLayer(),
            Linear(hidden_dim, num_classes),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def get_params_and_grads(self):
        params_and_grads = []
        for layer in self.layers:
            if hasattr(layer, 'get_params_and_grads'):
                params_and_grads.extend(layer.get_params_and_grads())
        return params_and_grads

