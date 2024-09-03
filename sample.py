class ForwardAD:
    def __init__(self, val, derivative):
        """
        Initialize with a value and its derivative.
        val: the value of the function at a certain point
        derivative: the derivative of the function with respect to the variable
        """
        self.val = val
        self.derivative = derivative

    def __add__(self, other):
        """
        Define addition for ForwardAD.
        (f + g)' = f' + g'
        """
        return ForwardAD(self.val + other.val, self.derivative + other.derivative)

    def __mul__(self, other):
        """
        Define multiplication for ForwardAD.
        (f * g)' = f'g + fg'
        """
        return ForwardAD(self.val * other.val, self.val * other.derivative + self.derivative * other.val)

    def __repr__(self):
        """
        Return the string representation of the ForwardAD.
        """
        return f"ForwardAD(val={self.val}, derivative={self.derivative})"

def test_function(x, y, z):
    
    """This function maps the internal dependencies and retuns the derivative"""
    
    w1 = x * y
    w2 = x + y
    w3 = z * w2
    y = w1 + w3
    
    return y
    
x_val = 1
y_val = 1
z_val = 1
    

# calculate derivate wrt x
x = ForwardAD(x_val, 0)
y = ForwardAD(y_val, 0)
z = ForwardAD(z_val, 1)

y = test_function(x, y, z)
print(y)