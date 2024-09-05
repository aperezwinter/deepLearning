import random
from math import log
from src.scalar import Scalar

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def zero_momentum(self):
        for p in self.parameters():
            p.momentum = 0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, nin, **kwargs):
        # nin: number of inputs
        super().__init__()
        # self.weights = [Scalar(random.gauss(0,1)) for _ in range(nin)]
        self.weights = [Scalar(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Scalar(0)
        self.act = kwargs.get('act') if 'act' in kwargs else 'linear'

    def __call__(self, x):
        z = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        if self.act == 'sigmoid':
            return z.logistic()
        elif self.act == 'tanh':
            return z.tanh()
        elif self.act == 'relu':
            return z.relu()
        elif self.act == 'leaky_relu':
            return z.leaky_relu()
        else:
            return z
    
    def __repr__(self) -> str:
        return f"{self.act} Neuron({len(self.weights)})"
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def reset(self):
        for wi in self.weights:
            #wi.value = random.gauss(0,1)
            wi.value = random.uniform(-1,1)
            wi.grad = 0
        self.bias.value = 0
        self.bias.grad = 0
    
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        # nin: number of inputs
        # nout: number of outputs
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def reset(self):
        for n in self.neurons:
            n.reset()
    
class MLP(Module):

    def __init__(self, nin, nouts, act='relu'):
        # nin: number of inputs (input layer)
        # nout: number of outputs (hidden + out layers)
        # act: activation function (default: ReLU 'relu')
        sz = [nin] + nouts
        act = act if isinstance(act, list) else [act] * len(nouts)
        self.layers = [Layer(sz[i], sz[i+1], act=act[i]) for i in range(len(nouts))]
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
        for layer in self.layers:
            layer.reset()
    
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
        
    def train_GD(self, X, y, loss='mseLoss', lr=1e-2, epochs=10000, out=True):
        i=0; loss_values = []
        loss_func = getattr(self, loss)     # get loss function
        while i < epochs:
            y_pred = self.forward(X)        # forward pass
            loss_func(y, y_pred, out=False) # compute loss
            self.backward()                 # backward pass
            for p in self.parameters():     # update weights
                p.value -= lr * p.grad
            if out:
                loss_values.append(self._loss.value)
            i += 1
        return loss_values if out else None
    
    def train_GD_momentum(self, X, y, loss='mseLoss', lr=1e-2, mu=0.9, epochs=10000, out=True):
        i=0; loss_values = []
        loss_func = getattr(self, loss)     # get loss function
        self.zero_momentum()
        while i < epochs:
            y_pred = self.forward(X)        # forward pass
            loss_func(y, y_pred, out=False) # compute loss
            self.backward()                 # backward pass
            for p in self.parameters():     # update weights
                p.momentum = mu * p.momentum + (1 - mu) * p.grad
                p.value -= lr * p.momentum
            if out:
                loss_values.append(self._loss.value)
            i += 1
        return loss_values if out else None
    
    def train_SGD(self, X, y, loss='mseLoss', lr=1e-2, epochs=1000, out=True):
        i=0; loss_values = []
        loss_func = getattr(self, loss)    # get loss function
        while i < epochs:
            idx = list(range(len(y)))
            random.shuffle(idx)
            X = X[idx]; y = y[idx]
            for Xi, yi in zip(X, y):
                Xi = [Xi]; yi = [yi]
                y_pred = self.forward(Xi)           # forward pass
                loss_func(yi, y_pred, out=False)    # compute loss
                self.backward()                     # backward pass
                for p in self.parameters():         # update weights
                    p.value -= lr * p.grad
            if out:
                loss_values.append(self._loss.value)
            i += 1
        return loss_values if out else None
    
    def train_miniBatch_SGD(self, X, y, loss='mseLoss', lr=1e-2, epochs=1000, batch_size=32, out=True):
        i=0; loss_values = []
        batch_size = min(batch_size, len(y))
        loss_func = getattr(self, loss)    # get loss function
        while i < epochs:
            idx = list(range(len(y)))
            random.shuffle(idx)
            X = X[idx]; y = y[idx]
            for j in range(0, len(y), batch_size):
                X_batch = X[j:j + batch_size]
                y_batch = y[j:j + batch_size]
                y_pred = self.forward(X_batch)          # forward pass
                loss_func(y_batch, y_pred, out=False)   # compute loss
                self.backward()                         # backward pass
                for p in self.parameters():             # update weights
                    p.value -= lr * p.grad
            if out:
                loss_values.append(self._loss.value)
            i += 1
        return loss_values if out else None

    def fit(self, X, y, loss='mseLoss', optimizer='gd', lr=1e-2, epochs=1000, verbose=False, out=True):
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