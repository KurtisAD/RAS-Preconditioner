import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

import precond

# Convection coefficients, change to 0 for Poisson case
b1 = 10
b2 = 20

# Discretization parameters
M = 64  # M x M mesh size
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
B1[:2] = (alpha, gamma)  # column
B2 = np.zeros((M - 1,))
B2[:2] = (alpha, delta)  # row

B = scipy.linalg.toeplitz(B1, B2)  # construct toeplitz matrix

I_delta = np.identity(M - 1) * delta
I_beta = np.identity(M - 1) * beta

tri_vec = np.zeros(M - 1)
tri_vec[1] = 1
lower = scipy.linalg.toeplitz(tri_vec, np.zeros(M - 1))  # below B format
upper = scipy.linalg.toeplitz(np.zeros(M - 1), tri_vec)  # above B format

A = np.kron(lower, I_beta) + np.kron(upper, I_delta) + np.kron(np.identity(M - 1), B)

# Final coefficient matrix
coeffA = np.identity((M - 1) ** 2) + m * A

# Precondition the coefficient matrix for delta = 3, subd = 16 with RAS
# This is for solving the system at each timestep to create the plots
# Comparison of iterations is afterwards!
G = precond.graph(coeffA)
pc_ras = precond.RAS(coeffA, 16, precond.delta_overlap(G, precond.partition_graph(coeffA, 16), 16, 3),
                     precond.delta_overlap(G, precond.partition_graph(coeffA, 16), 16, 0))  # RAS

for n in range(N):
    Unp1, ec = spla.gmres(np.matmul(pc_ras, coeffA), np.matmul(pc_ras, U[n]), restart=30, maxiter=10000,
                          callback=precond.counter)
    U.append(Unp1)

# Re-run each preconditioner at t = 1 for reporting on the number of iterations for each delta
subd = 16
# AS
for ddd in range(1, 3+1):
    pc_as = precond.AS(coeffA, subd, precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, ddd))
    spla.gmres(coeffA, U[0], restart=30, maxiter=10000, callback=precond.counter)
    print('Without AS; delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0
    spla.gmres(np.matmul(pc_as, coeffA), np.matmul(pc_as, U[0]), restart=30, maxiter=10000, callback=precond.counter)
    print('With AS; delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0

# RAS
for ddd in range(1, 3 + 1):
    pc_ras = precond.RAS(coeffA, subd, precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, ddd),
                         precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, 0))
    spla.gmres(coeffA, U[0], restart=30, maxiter=10000, callback=precond.counter)
    print('Without RAS; delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0
    spla.gmres(np.matmul(pc_ras, coeffA), np.matmul(pc_ras, U[0]), restart=30, maxiter=10000, callback=precond.counter)
    print('With RAS; delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0

# ASH
for ddd in range(1, 3+1):
    pc_ash = precond.ASH(coeffA, subd, precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, ddd),
                         precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, 0))
    spla.gmres(coeffA, U[0], restart=30, maxiter=10000, callback=precond.counter)
    print('Without ASH; subd = ', subd, ', delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0
    spla.gmres(np.matmul(pc_ash, coeffA), np.matmul(pc_ash, U[0]), restart=30, maxiter=10000, callback=precond.counter)
    print('With ASH; subd = ', subd, ', delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0

# RASH
for ddd in range(1, 3 + 1):
    pc_rash = precond.RASH(coeffA, subd, precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, ddd),
                         precond.delta_overlap(G, precond.partition_graph(coeffA, subd), subd, 0))
    spla.gmres(coeffA, U[0], restart=30, maxiter=10000, callback=precond.counter)
    print('Without RASH; subd = ', subd, ', delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0
    spla.gmres(np.matmul(pc_rash, coeffA), np.matmul(pc_rash, U[0]), restart=30, maxiter=10000, callback=precond.counter)
    print('With RASH; subd = ', subd, ', delta =', ddd, ',', 'iterations =', precond.counter.niter)
    precond.counter.niter = 0

# Finally, colorplot a given timestep to validate results using the full U vector obtained prior
x = np.linspace(0, 1, num=M)
y = np.linspace(0, 1, num=M)

timestep = 5
u_approx = U[timestep].reshape((M - 1, M - 1))

plt.pcolormesh(x, y, u_approx)
plt.colorbar()
plt.show()
