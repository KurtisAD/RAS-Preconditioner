import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from itertools import chain
import networkx as nx
import scipy.sparse.linalg as spla


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1


def selection_sort(arr):
    for i in range(len(arr)):
        swap = i + np.argmin(arr[i:])
        (arr[i], arr[swap]) = (arr[swap], arr[i])
    return arr


def graph(A):
    len_A = len(A)
    G = nx.Graph()
    G.add_nodes_from(list(range(1,len_A + 1)))

    for i in range(len_A):
        for j in range(len_A):
            if A[i][j] != 0:
                G.add_edge(i+1, j+1)
    return G


def partition_graph(A, N):
    len_A = len(A)
    size = int(len_A/N)
    arr = np.zeros((N, size))
    counter = 1
    for i in range(N):
        for j in range(size):
            arr[i][j] = counter
            counter += 1
    return arr


def next_overlap(G, prev, N):
    arr = [0]*N
    for i in range(N):
        copy = prev[i][:]
        size = len(prev[i])
        for j in range(size):
            for edge in G.edges():
                e1, e2 = edge
                node = prev[i][j]
                if node in edge and edge != (node, node):
                    if node != e1 and e1 not in prev[i][:]:
                        copy = np.append(copy, e1)
                    elif node != e2 and e2 not in prev[i][:]:
                        copy = np.append(copy, e2)
        arr[i] = selection_sort(copy)

    return arr


def delta_overlap(G, prev, N, delta):
    for i in range(delta):
        prev = next_overlap(G, prev, N)
    return prev


def R(A, W):
    len_A = len(A)
    R = np.zeros((len_A, len_A))

    for i in range(len_A):
        if i + 1 in W:
            R[i][i] = 1

    return R



def restrict(A, W):
    len_A = len(A)
    n = len(W)
    copy_rows = np.zeros((n, len_A))
    copy = np.zeros((n, n))
    counter = 0
    for i in W:
        copy_rows[counter,:] = A[int(i-1),:]
        counter += 1
    counter = 0
    for i in W:
        copy[:,counter] = copy_rows[:,int(i-1)]
        counter += 1

    return copy


def enlargen(A, A_enlarge, N, W):
    len_A = len(A)
    matrix = np.zeros((len_A, len_A))
    start = int(W[0]-1)
    end = int(W[-1]-1)
    matrix[start:end+1, start:end+1] = A_enlarge

    return matrix


def AS(A, N, W):
    len_A = len(A)
    matrix = np.zeros((len_A, len_A))
    for i in range(N):
        R_i_delta = R(A, W[i])
        A_i = np.matmul(np.matmul(R_i_delta, A), R_i_delta)
        A_i_inv = np.linalg.inv(restrict(A_i, W[i]))
        A_i_inv = enlargen(A, A_i_inv, N, W[i])
        matrix = matrix + np.matmul(np.matmul(R_i_delta, A_i_inv), R_i_delta)

    return matrix


def RAS(A, N, W, W_0):
    matrix = np.zeros((128, 128))
    for i in range(N):
        R_i_delta = R(A, W[i])
        R_0 = R(A, W_0[i])
        A_i = np.matmul(np.matmul(R_i_delta, A), R_i_delta)
        A_i_inv = np.linalg.inv(restrict(A_i, W[i]))
        A_i_inv = enlargen(A_i_inv, N, W[i])
        matrix = matrix + np.matmul(np.matmul(R_0, A_i_inv), R_i_delta)

    return matrix




A = np.zeros((128, 128))
b = np.zeros((128, 1))
A[-1][-1] = 1

for i in range(127):
    A[i][i] = 6
    A[i][i+1] = -2
    A[i+1][i] = 7
    b[i] = 8



G = graph(A)
N = 16

counter = gmres_counter()


######## RA precond ########

# for delta in range(3 + 1):
#     precond = AS(A, N, delta_overlap(G, partition_graph(A, N), N, delta))
#     spla.gmres(A, b, restart=30, maxiter=10000, callback=counter)
#     print('Without AS; delta =', delta,',', 'iterations =', counter.niter)
#     counter.niter = 0
#     spla.gmres(np.matmul(precond, A), np.matmul(precond, b), restart=30, maxiter=10000, callback=counter)
#     print('With AS; delta =', delta,',', 'iterations =', counter.niter)
#     counter.niter = 0
#
#
# for delta in range(3 + 1):
#     precond = AS(A, N, delta_overlap(G, partition_graph(A, N), N, delta))
#     print(np.linalg.cond(A))
#     print(np.linalg.cond(np.matmul(precond, A)))


######## RAS precond ########

# for delta in range(3 + 1):
#     precond = RAS(A, N, delta_overlap(G, partition_graph(A, N), N, delta), delta_overlap(G, partition_graph(A, N), N, 0))
#     spla.gmres(A, b, restart=30, maxiter=10000, callback=counter)
#     print('Without AS; delta =', delta,',', 'iterations =', counter.niter)
#     counter.niter = 0
#     spla.gmres(np.matmul(precond, A), np.matmul(precond, b), restart=30, maxiter=10000, callback=counter)
#     print('With AS; delta =', delta,',', 'iterations =', counter.niter)
#     counter.niter = 0
#
#
# for delta in range(3 + 1):
#     precond = RAS(A, N, delta_overlap(G, partition_graph(A, N), N, delta), delta_overlap(G, partition_graph(A, N), N, 0))
#     print(np.linalg.cond(A))
#     print(np.linalg.cond(np.matmul(precond, A)))
