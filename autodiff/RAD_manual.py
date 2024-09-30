class Variable:

    def __init__(self, value, adjoint=0.0):
        self.value = value
        self.adjoint = adjoint

    def backward(self, adjoint):
        self.adjoint += adjoint

    def __add__(self, other):
        variable = Variable(self.value + other.value)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * 1.0
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __sub__(self, other):
        variable = Variable(self.value - other.value)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * -1.0
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __mul__(self, other):
        variable = Variable(self.value * other.value)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * other.value
            other_adjoint = adjoint * self.value
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __truediv__(self, other):
        variable = Variable(self.value / other.value)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * (1.0 / other.value)
            other_adjoint = adjoint * (-1.0 * self.value / other.value**2)
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __repr__(self) -> str:
        return f"value: {self.value}, adjoint: {self.adjoint}"
    
    
# x = Variable(1)
# y = Variable(2)
# z = Variable(3)

# w1 = x + y
# w2 = x*y
# w3 = z*w1

# v = w3 + w2 # xy + z(x+y)

# v.backward(1) 

# print(x)
# print(y)
# print(z)

x = Variable(1)
y = Variable(2)
z = Variable(3)

import numpy as np

M1 = np.array([x, z])
M2 = np.array([y, x+y])

v = M1 @ M2.T

v.backward(1) 

print(x)
print(y)
print(z)

