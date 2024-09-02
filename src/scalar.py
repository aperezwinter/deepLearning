from math import exp, erf, sqrt, pi

class Scalar:
    """ Tensor of rank 0: stores its value and gradient. """

    def __init__(self, value, _children=(), _op='', _label=''):
        self.value = value
        self.grad = 0
        # private variables for build autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # operation produced by this node, for graphviz, debugging, etc
        self._label = _label # scalar label, for graphviz, debugging, etc

    def __add__(self, other): # self + other
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(value=self.value + other.value, _children=(self, other), _op='+')
        
        def _backward(): # local backward
            # Let z = x + y, then dz/dx = dz/dy = 1
            # So, dz/dx = dz/dz * dz/dx = 1 * 1 = z.grad * 1 = z.grad
            # So, dz/dy = dz/dz * dz/dy = 1 * 1 = z.grad * 1 = z.grad
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward # set backward func from '+' operation
        
        return out
    
    def __sub__(self, other): # self - other
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(value=self.value - other.value, _children=(self, other), _op='-')
        
        def _backward(): # local backward
            # Let z = x - y, then dz/dx = 1, dz/dy = -1
            # So, dz/dx = dz/dz * dz/dx = 1 * 1 = z.grad * 1 = z.grad
            # So, dz/dy = dz/dz * dz/dy = 1 * 1 = z.grad * 1 = z.grad
            self.grad += out.grad
            other.grad += out.grad * -1
        out._backward = _backward # set backward func from '-' operation
        
        return out
    
    def __mul__(self, other): # self * other
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(value=self.value * other.value, _children=(self, other), _op='*')
        
        def _backward(): # local backward
            # Let z = x * y, then dz/dx = y, dz/dy = x
            # So, dz/dx = dz/dz * dz/dx = 1 * y = z.grad * y.value
            # So, dz/dy = dz/dz * dz/dy = 1 * x = z.grad * x.value
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward # set backward func from '*' operation
        
        return out
    
    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "Only supporting real powers"
        out = Scalar(value=self.value**other, _children=(self,), _op='**{:.2g}'.format(other))
        
        def _backward(): # local backward
            # Let z = x^y, then dz/dx = y * x^(y-1)
            # So, dz/dx = dz/dz * dz/dx = 1 * y * x^(y-1)
            # ... = z.grad * y * x.value^(y-1)
            self.grad += out.grad * (other * self.value**(other-1))
        out._backward = _backward

        return out

    def __truediv__(self, other): # self / other
        # Check zero division problem
        if isinstance(other, (int, float)):
            assert other != 0, "Division by zero!"
            other = Scalar(other)
        elif isinstance(other, Scalar):
            assert other.value != 0, "Division by zero!"
        else:
            raise TypeError(f"Variable to '/' with must be of type (int, float, Scalar) not {type(other)}.")
        out = Scalar(value=self.value/other.value, _children=(self, other), _op='/')
        
        def _backward(): # local backward
            # Let z = x / y, then dz/dx = 1/y, dz/dy = -x * y^(-2)
            # So, dz/dx = dz/dz * dz/dx = 1 * y^(-1) = out.grad * y.value^(-1)
            # So, dz/dy = dz/dz * dz/dy = 1 * -x * y^(-2) = ...
            # ... = out.grad * -x.value * y.value^(-2) 
            self.grad += out.grad * (other.value**-1)
            other.grad += out.grad * (-1 * self.value * other.value**-2)
        out._backward = _backward

        return out
    
    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rsub__(self, other): # other - self
        return other - self
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __rtruediv__(self, other): # other / self
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other / self
    
    def __str__(self):
        return "Scalar({}): value={:.3g} | grad={:.3g}".format(
            self._label, self.value, self.grad)
    
    def __repr__(self):
        return "Scalar(value={:.3g}, grad={:.3g})".format(self.value, self.grad)
    
    def backward(self):
        # Topological order all of the children in the graph
        topo, visited = [], set()
        def build_topo(scalar):
            if scalar not in visited:
                visited.add(scalar)
                for child in scalar._prev:
                    build_topo(child)
                topo.append(scalar)
        build_topo(self)

        # Set grad=1 and go one variable at a time and apply 
        # the chain rule to get its gradient
        self.grad = 1
        for scalar in reversed(topo):
            scalar._backward()
    
    def logistic(self):
        # Logistic function: f(x) = 1 / (1 + e^(-x)), f' = f*(1-f)
        value = 1 / (1 + exp(self.value)**-1)
        out = Scalar(value, _children=(self,), _op='logistic')
        
        def _backward(): # local backward
            # Let y = 1 / (1 + e^(-x)), then dy/dx = y * (1-y)
            # So, dy/dx = dy/dy * dy/dx = 1 * y * (1-y) = ...
            # ... = out.grad * y.value * (1 - y.value)
            self.grad += out.grad * out.value * (1 - out.value)
        out._backward = _backward

        return out
    
    def tanh(self):
        # Hyperbolic tangent function: f(x) = (e^(2x)-1)/(e^(2x)+1), f' = 1-f^2
        value = (exp(2*self.value)-1) / (exp(2*self.value)+1)
        out = Scalar(value, _children=(self,), _op='tanh')
        
        def _backward(): # local backward
            # Let y = tanh(x), then dy/dx = 1 - tanh(x)^2
            # So, dy/dx = dy/dy * dy/dx = 1 * (1 - y^2) = ...
            # ... = out.grad * (1 - out.value^2)
            self.grad += out.grad * (1 - out.value**2)
        out._backward = _backward

        return out
    
    def relu(self):
        # Rectified linear unit: f(x) = max(0,x), f' = 1(y>0) or 0(y<=0)
        out = Scalar(value=max(0,self.value), _children=(self,), _op='ReLU')
        
        def _backward(): # local backward
            # Let y = max(0,x), then dy/dx = 1(y>0) or 0 (y<=0)
            # So, dy/dx = dy/dy * dy/dx = 1 * {1,0} = out.grad * {1,0}
            self.grad += out.grad * (out.value > 0)
        out._backward = _backward

        return out
    
    def linear(self):
        # Linear unit: f(x) = x, f' = 1
        out = Scalar(value=self.value, _children=(self,), _op='LU')
        
        def _backward(): # local backward
            # Let y = x, then dy/dx = 1
            # So, dy/dx = dy/dy * dy/dx = 1 * 1 = out.grad
            self.grad += out.grad
        out._backward = _backward

        return out 
    
    def erf(self):
        # Error function: erf(x)= (2/√pi) * int_0^x e^(-t^2)dt
        out = Scalar(value=erf(self.value), _children=(self,), _op='erf')
        
        def _backward(): # local backward
            # Let y = erf(x), then dy/dx = (2/√pi) * e^(-x^2)
            # So, dy/dx = dy/dy * dy/dx = 1 * (2/√pi) * e^(-x^2) = ...
            # ... = out.grad * (2/√pi) * exp(-self.value^2)
            self.grad += out.grad * (2/sqrt(pi)) * exp(-self.value**2)
        out._backward = _backward

        return out
