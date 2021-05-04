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

counter = gmres_counter()

# Selection sort which is used to order the nodes in some functions.

def selection_sort(arr):
    for i in range(len(arr)):
        swap = i + np.argmin(arr[i:])
        (arr[i], arr[swap]) = (arr[swap], arr[i])
    return arr


# The graph G which is associated to the matrix A.

def graph(A):
    len_A = len(A)
    G = nx.Graph()
    G.add_nodes_from(list(range(1,len_A + 1)))

    for i in range(len_A):
        for j in range(len_A):
            if A[i][j] != 0:
                G.add_edge(i+1, j+1)
    return G


# Canonical partitioning of the unknowns for the problem.

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


# Given a delta overlapping partition, this returns the next delta overlapping partition.

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
        arr[i] = selection_sort(list(set(copy)))

    return arr


# Generates the delta overlap partition.

def delta_overlap(G, prev, N, delta):
    for i in range(delta):
        prev = next_overlap(G, prev, N)
    return prev


# The matrices R of the article.

def R(A, W):
    len_A = len(A)
    R = np.zeros((len_A, len_A))

    for i in range(len_A):
        if i + 1 in W:
            R[i][i] = 1

    return R


# This returns the subblock of A_i which we wish to invert.

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


# This takes the inverted subblock of A_i obtained from the function restrict and replaces it where it originated.

def enlargen(A, A_enlarge, N, W):
    len_A = len(A)
    matrix = np.zeros((len_A, len_A))
    start = int(W[0]-1)
    end = start + len(W) - 1
    matrix[start:end+1, start:end+1] = A_enlarge

    return matrix


# AS precond

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


# RAS precond

def RAS(A, N, W, W_0):
    len_A = len(A)
    matrix = np.zeros((len_A, len_A))
    for i in range(N):
        R_i_delta = R(A, W[i])
        R_0 = R(A, W_0[i])
        A_i = np.matmul(np.matmul(R_i_delta, A), R_i_delta)
        A_i_inv = np.linalg.inv(restrict(A_i, W[i]))
        A_i_inv = enlargen(A, A_i_inv, N, W[i])
        matrix = matrix + np.matmul(np.matmul(R_0, A_i_inv), R_i_delta)

    return matrix


# ASH precond

def ASH(A, N, W, W_0):
    len_A = len(A)
    matrix = np.zeros((len_A, len_A))
    for i in range(N):
        R_i_delta = R(A, W[i])
        R_0 = R(A, W_0[i])
        A_i = np.matmul(np.matmul(R_i_delta, A), R_i_delta)
        A_i_inv = np.linalg.inv(restrict(A_i, W[i]))
        A_i_inv = enlargen(A, A_i_inv, N, W[i])
        matrix = matrix + np.matmul(np.matmul(R_i_delta, A_i_inv), R_0)

    return matrix


# RASH precond

def RASH(A, N, W, W_0):
    len_A = len(A)
    matrix = np.zeros((len_A, len_A))
    for i in range(N):
        R_i_delta = R(A, W[i])
        R_0 = R(A, W_0[i])
        A_i = np.matmul(np.matmul(R_i_delta, A), R_i_delta)
        A_i_inv = np.linalg.inv(restrict(A_i, W[i]))
        A_i_inv = enlargen(A, A_i_inv, N, W[i])
        matrix = matrix + np.matmul(np.matmul(R_0, A_i_inv), R_0)

    return matrix
