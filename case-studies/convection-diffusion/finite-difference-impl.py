import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Advection coefficients, change to 0 for Poisson case
b1 = 10
b2 = 20

# Discretization parameters
M = 64  # M x M mesh size
N = 10  # number of time steps
h = 1/M  # spatial mesh step size
m = 1/N  # temporal step size

# List of approximations at each time step
U = []

# Initial condition (at t = 0)
U0 = np.zeros((M-1)*(M-1))
U0[:M-1] = 1.0
U.append(U0)

# Construct the finite difference matrix A (block tridiagonal)
alpha = 4/h**2 - b1/h - b2/h
delta = -1/h**2
beta = delta + b1/h
gamma = delta + b2/h

B1 = np.zeros((M-1,))
B1[:2] = (alpha, gamma)
B2 = np.zeros((M-1,))
B2[:2] = (alpha, delta)

B = scipy.linalg.toeplitz(B1, B2)

I_delta = np.identity(M-1) * delta
I_beta = np.identity(M-1) * beta

tri_vec = np.zeros(M-1)
tri_vec[1] = 1
lower = scipy.linalg.toeplitz(tri_vec, np.zeros(M-1))
upper = scipy.linalg.toeplitz(np.zeros(M-1), tri_vec)

A = np.kron(lower, I_beta) + np.kron(upper, I_delta) + np.kron(np.identity(M-1), B)

# At this point, we would precondition A, and then use GMRES
# For now, using standard solver for illustration
# System we want to solve is (I + mA)U^{n+1} = U^{n}
for n in range(N):
    Unp1 = scipy.linalg.solve((np.identity((M-1)**2) + m*A), U[n])
    U.append(Unp1)

# U now contains the spatial approximations at each time step, create some plots at various time steps
x = np.linspace(0, 1, num=M)
y = np.linspace(0, 1, num=M)

timestep = 8

u_approx = U[timestep].reshape((M-1, M-1))

plt.pcolormesh(x, y, u_approx)
plt.colorbar()
plt.show()
