#date: 2023-03-02T16:58:43Z
#url: https://api.github.com/gists/616caf2b27863b0cba8476fa7d9b1619
#owner: https://api.github.com/users/skojaku

# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-02 11:43:18
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-02 11:58:51
import numpy as np
from scipy import sparse
import scipy.linalg


def LaplacianEigenMap(A, dim, return_basis_vector=False):
    """Laplacian EigenMap

    Example:
    >> import networkx as nx
    >> G = nx.karate_club_graph() # Get karate club net
    >> A = nx.adjacency_matrix(G) # To scipy sparse format
    >> emb = LaplacianEigenMap(A, dim=5) # Get the 5-dimensional embedding. Each column may have different norm.
    >> eig_emb = LaplacianEigenMap(A, dim=5, return_basis_vector=True) # Every column will be normalized to have a unit norm.

    :param A: Network
    :type A: scipy.sparse format
    :param dim: Dimension
    :type dim: int
    :return_basis_vector: Set True to obtain the eigenvectors of the Laplacian Matrix. Otherwise, return the projection of the (normalized) given network onto the space spanned by the Laplaian basis.
    :return: Embedding
    :rtype: numpy.ndarray of (num nodes, dim)
    """
    deg = np.array(A.sum(axis=1)).reshape(-1)
    Dinv = sparse.diags(1 / np.maximum(np.sqrt(deg), 1))
    L = Dinv @ A @ Dinv
    w, v = sparse.linalg.eigs(L, k=dim + 1)
    order = np.argsort(-w)
    v = np.real(v[:, order[1:]])
    w = np.real(w[order[1:]])

    if return_basis_vector:
        return v

    return L @ v