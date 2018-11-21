import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix, csr_matrix, diags
from regression.regressor import fit

def fit_link_prediction_function(graphA: nx.Graph, graphB: nx.Graph, function, k):
    """
    compute the spectral function that transforms the graphA into the graphB
    :param graphA:
    :param graphB:
    :param function: @see regression/functions.py for examples
    :param k: the amount of singular values to compute
    :return: the function with the fitted parameters
    """
    A = nx.adjacency_matrix(graphA).asfptype()
    B = nx.adjacency_matrix(graphB).asfptype()
    Ua, sigmaA, Vta = svds(A, k)
    Ub, sigmaB, Vtb = svds(B, k)
    return fit(sigmaA, sigmaB, function)


def predict_links(graph: nx.Graph, f, k):
    """
    perform link prediction on a graph using a fitted function
    :param graph:
    :param f:
    :param k: number of sigular value to keep
    :return: the graph with edges predicted
    """
    A = nx.adjacency_matrix(graph).asfptype()
    u, sigmaA, vt = svds(A, k, return_singular_vectors=True)
    sigmaB = f(sigmaA)
    B = csr_matrix(u) * diags(sigmaB) * csc_matrix(vt)
    return nx.from_scipy_sparse_matrix(B)
