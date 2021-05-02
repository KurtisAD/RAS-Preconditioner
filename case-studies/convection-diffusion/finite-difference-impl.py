import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

import precond

# Convection coefficients, change to 0 for Poisson case
b1 = 10
b2 = 20

# Discretization parameters
M = 33  # M x M mesh size
N = 5  # number of time steps
h = 1 / M  # spatial mesh step size
m = 1 / N  # temporal step size

# List of approximations at each time step
U = []

# Initial condition (at t = 0)
U0 = np.zeros((M - 1) * (M - 1))
U0[:M - 1] = 5000.0
U.append(U0)

# Construct the finite difference matrix A (block tridiagonal)
alpha = 4 / h ** 2 - b1 / h - b2 / h
delta = -1 / h ** 2
beta = delta + b1 / h
gamma = delta + b2 / h

B1 = np.zeros((M - 1,))
B1[:2] = (alpha, gamma)
B2 = np.zeros((M - 1,))
B2[:2] = (alpha, delta)

B = scipy.linalg.toeplitz(B1, B2)

I_delta = np.identity(M - 1) * delta
I_beta = np.identity(M - 1) * beta

tri_vec = np.zeros(M - 1)
tri_vec[1] = 1
lower = scipy.linalg.toeplitz(tri_vec, np.zeros(M - 1))
upper = scipy.linalg.toeplitz(np.zeros(M - 1), tri_vec)

A = np.kron(lower, I_beta) + np.kron(upper, I_delta) + np.kron(np.identity(M - 1), B)
coeffA = np.identity((M - 1) ** 2) + m * A
nn = len(coeffA)

# At this point, we would precondition A, and then use GMRES
# For now, using standard solver for illustration
# System we want to solve is (I + mA)U^{n+1} = U^{n}

# Precondition the coefficient matrix for given subd and delta
G = precond.graph(coeffA)
subd = 16
dd = 3
pc_as = precond.AS(coeffA, subd, precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, dd))  # AS
pc_ras = precond.RAS(coeffA, subd, precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, dd),
                     precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, 0))  # RAS

#M_u = lambda u: np.linalg.solve(pc_ras, u)
#M_ras = spla.LinearOperator(((M - 1) ** 2, (M - 1) ** 2), M_u)

for n in range(N):
    # Unp1 = scipy.linalg.solve(coeffA, U[n])
    Unp1, ec = spla.gmres(coeffA, U[n], restart=30, maxiter=10000, M=pc_as, callback=precond.counter)
    if n == np.floor(N / 2):
        print('Delta=', dd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0
    U.append(Unp1)

# U now contains the spatial approximations at each time step, create colormap plot given a timestep to verify
# results
'''
x = np.linspace(0, 1, num=M)
y = np.linspace(0, 1, num=M)

timestep = N
u_approx = U[timestep].reshape((M - 1, M - 1))

plt.pcolormesh(x, y, u_approx)
plt.colorbar()
plt.show()
'''
