import numpy as np
import scipy.linalg

# Advection coefficients, change to 0 for Poisson case
b1 = 10
b2 = 20

# Discretization parameters
M = 4  # M x M mesh size
N = 10  # number of time steps
h = 1/M  # spatial mesh step size
m = 1/N  # temporal step size

# List of approximations at each time step
U = []

# Initial condition (at t = 0)
U0 = np.ones((M-1)*(M-1))
U.append(U0)

# Construct the finite difference matrix A (block tridiagonal)
alpha = -4/h**2 - b1/h - b2/h
beta = 1/h**2 + b1/h
gamma = 1/h**2 + b2/h

B1 = np.zeros((M-1,))
B1[:2] = (alpha, gamma)
B2 = np.zeros((M-1,))
B2[:2] = (alpha, 1/h**2)

B = scipy.linalg.toeplitz(B1, B2)

I_gamma = np.identity(M-1) * 1/h**2
I_beta = np.identity(M-1) * beta

lower = scipy.linalg.toeplitz((0, 1, 0), (0, 0, 0))
upper = scipy.linalg.toeplitz((0, 0, 0), (0, 1, 0))

A = np.kron(lower, I_gamma) + np.kron(upper, I_beta) + np.kron(np.identity(M-1), B)

# At this point, we would precondition A, and then use GMRES
# For now, using standard solver for illustration
# System we want to solve is (I + mA)U^{n+1} = U^{n}
for n in range(1, N+1):
    Unp1 = scipy.linalg.solve((np.identity((M-1)**2) + m*A), U[n-1])
    U.append(Unp1)

# U now contains the spatial approximations at each time step
for i in range(N+1):
    print(U[i])
