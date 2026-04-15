import numpy as np
from gradient_descent import gradient_descent

# Example: minimize f(x) = (x - 3)^2
def f(x):
    return (x[0] - 3)**2

def grad_f(x):
    return np.array([2 * (x[0] - 3)])

x_opt, history = gradient_descent(f, grad_f, x0=[0])

print("Optimal x:", x_opt)
print("Final value:", f(x_opt))
