import csv_interface
import link_prediction.link_prediction as lp
import regression.functions as funcs
import scipy as sp
import scipy.sparse.linalg as spar
import networkx as nx

graph_list = csv_interface.get_graphs_from_csv("sample.csv")

graphA = graph_list[0][1]
graphB = graph_list[1][1]
k = min(graphA.number_of_nodes(), graphA.number_of_edges())
f = lp.fit_link_prediction_function(graphA, graphB, funcs.hyperbolic_sine, k=k) # TODO: find a better way to set k
graphAbis = lp.predict_links(graphA, f, k=k)
dist = spar.norm(nx.adjacency_matrix(graphB)-nx.adjacency_matrix(graphAbis), ord='fro')
print("in sample error:", dist)
