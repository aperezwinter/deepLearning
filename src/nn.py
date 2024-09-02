import random
from math import log
from src.scalar import Scalar

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        # nin: number of inputs
        super().__init__()
        self.weights = [Scalar(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Scalar(0)
        self.nonlin = nonlin

    def __call__(self, x):
        a = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        return a.logistic() if self.nonlin else a.linear()
        
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self) -> str:
        act = "Sigmoid" if self.nonlin else "Linear"
        return f"{act} Neuron({len(self.weights)})"
    
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        # nin: number of inputs
        # nout: number of outputs
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):

    def __init__(self, nin, nouts):
        # nin: number of inputs (input layer)
        # nout: number of outputs (hidden + out layers)
        # nonlin=True for all hidden layers, except for the last layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        self._loss = Scalar(1.0)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def reset(self):
        for p in self.parameters():
            p.value = random.uniform(-1,1)
            p.grad = 0
    
    def backward(self):
        self.zero_grad() # flush gradients (set to zero)
        self._loss.backward()

    def forward(self, X):
        return [self(x) for x in X]
    
    def maeLoss(self, y_gt, y_pred, out=True):
        N=len(y_gt); M=len(y_pred)
        assert N == M, f"Mismatched dimensions {N} != {M}."
        self._loss = sum(abs(yi - yj) for yi, yj in zip(y_gt, y_pred)) / N
        return self._loss if out else None
    
    def mseLoss(self, y_gt, y_pred, out=True):
        N=len(y_gt); M=len(y_pred)
        assert N == M, f"Mismatched dimensions {N} != {M}."
        self._loss = sum((yi - yj)**2 for yi, yj in zip(y_gt, y_pred)) / N
        return self._loss if out else None
    
    def crossEntropyLoss(self, y_gt, y_pred, out=True):
        N=len(y_gt); M=len(y_pred)
        assert N == M, f"Mismatched dimensions {N} != {M}."
        self._loss = sum(-yi * log(yj) for yi, yj in zip(y_gt, y_pred)) / N
        return self._loss if out else None

    def fit(self, X, y, loss='mseLoss', optimizer='gd', lr=1e-2, epochs=10000, verbose=False, out=True):
        i=0; epoch_loss = []
        e=int(epochs/20) if epochs > 20 else 1
        while i < epochs:
            try:
                loss_func = getattr(self, loss)
            except AttributeError:
                raise AttributeError(f"Loss function '{loss}' not found.")
            if optimizer == 'sgd':
                idx = list(range(len(y)))
                random.shuffle(idx)
                X = X[idx]; y = y[idx]
                end = int(len(y) * 0.2) if int(len(y) * 0.2) > 0 else 1 # 20% of the dataset
                X_train = X[:end]
                y_train = y[:end]
            else:
                X_train = X
                y_train = y
            loss_func(X_train, y_train, out=False) # forward pass
            self.backward() # backward pass
            for p in self.parameters(): # update
                p.value -= lr * p.grad # gradient descent
            if verbose and (i % e == 0):
                print(f"Epoch {i}: loss={self._loss}")
            if out and (i % 2 == 0):
                epoch_loss.append([i, self._loss.value])
            i += 1
        if verbose and (i % e != 0):
            print(f"Epoch {i}: loss={self._loss}")
        if out and (i % 2 != 0):    
            epoch_loss.append([i, self._loss.value])
        return epoch_loss