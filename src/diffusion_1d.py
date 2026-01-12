import numpy as np

def solve_diffusion_1d(L=10.0, N=50, D=1.0, Sigma_a=0.1, S=1.0):
    """
    Solve the 1D steady-state diffusion equation using finite differences.
    """

    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    A = np.zeros((N, N))
    b = np.zeros(N)

    # Interior nodes
    for i in range(1, N-1):
        A[i, i-1] = -D / dx**2
        A[i, i]   = 2*D / dx**2 + Sigma_a
        A[i, i+1] = -D / dx**2
        b[i] = S

    # Boundary conditions (vacuum)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    b[0] = 0.0
    b[-1] = 0.0

    phi = np.linalg.solve(A, b)
    return x, phi