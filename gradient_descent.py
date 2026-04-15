import numpy as np

def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iters=1000, tol=1e-6):
    """
    Performs gradient descent optimization.

    Parameters:
    f           : objective function
    grad_f      : gradient of the function
    x0          : initial guess (numpy array)
    learning_rate : step size
    max_iters   : maximum iterations
    tol         : tolerance for convergence

    Returns:
    x           : optimal solution
    history     : list of function values
    """
    x = np.array(x0, dtype=float)
    history = []

    for i in range(max_iters):
        grad = grad_f(x)
        x_new = x - learning_rate * grad

        history.append(f(x))

        # Convergence check
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, history
