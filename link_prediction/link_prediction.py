import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix, csr_matrix, diags
from regression.regressor import fit
import matplotlib.pyplot as plt


def fit_link_prediction_function(graphA: nx.Graph, graphB: nx.Graph, function, k, verbose=False, ax=None):
    """
    compute the spectral function that transforms the graphA into the graphB
    :param graphA:
    :param graphB:
    :param function: @see regression/functions.py for examples
    :param k: the amount of singular values to compute
    :param verbose: if true plot the sigmaA and sigmaB
    :return: the function with the fitted parameters
    """
    A = nx.adjacency_matrix(graphA).asfptype()
    B = A - nx.adjacency_matrix(graphB).asfptype()
    den = max(np.max(A), np.max(B))
    # A = A / den
    # B = B / den
    Ub, sigmaB, Vtb = svds(B, k, which='LM')
    Ua, sigmaA, Vta = svds(A, k, which='LM')
    coefs = np.polyfit(sigmaA, sigmaB,3)
    f = np.vectorize(lambda x: np.sum([x ** (len(coefs)-i-1) * coefs[i] for i in range(len(coefs))]))
    # f = fit(sigmaA, sigmaB, function)
    space = np.linspace(min(sigmaA), max(sigmaA))
    fx = np.vectorize(f)(space)
    if ax is None:
        ax = plt
    ax.plot(sigmaA, sigmaB, 'rx', label='eigen values')
    ax.plot(space, fx, 'b--', label='best fit')
    return f, coefs, plt


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
    return B
