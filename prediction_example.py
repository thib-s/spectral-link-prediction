import csv_interface
import link_prediction.link_prediction as lp
import regression.functions as funcs
import scipy as sp
import scipy.sparse.linalg as spar
import networkx as nx
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


infile = open('dataset1', 'rb')
graph_list = pickle.load(infile)
infile.close()
# graph_list = csv_interface.get_graphs_from_csv("sample.csv")

n_of_timesteps = len(graph_list)
delta_t_t1 = []
delta_t1_t2 = []
avg_t = []
avg_t1 = []
avg_t2 = []
in_sample_err = []
out_sample_err = []
coefs = []
fig, ax = plt.subplots(int((n_of_timesteps-1)/2), 2, True, True)
fig.set_size_inches(8, fig.get_figheight(), forward=True)
fig.set_size_inches(5, fig.get_figwidth(), forward=True)

for i in tqdm(range(n_of_timesteps - 2)):
    graph_t = graph_list[i][1]
    graph_t1 = graph_list[i + 1][1]
    graph_t2 = graph_list[i + 2][1]
    graph_shape = nx.adjacency_matrix(graph_t1).shape
    delta_t_t1.append(
        spar.norm(nx.adjacency_matrix(graph_t) - nx.adjacency_matrix(graph_t1), ord=1)
        / (graph_shape[0] * graph_shape[1])
    )
    delta_t1_t2.append(
        spar.norm(nx.adjacency_matrix(graph_t1) - nx.adjacency_matrix(graph_t2), ord=1)
        / (graph_shape[0] * graph_shape[1])
    )
    k = 100  # int(min(graphA.number_of_nodes(), graphA.number_of_edges()) * 0.10)
    f, coef, _ = lp.fit_link_prediction_function(graph_t, graph_t1, funcs.exponential, k=k,
                                                   verbose=True, ax=ax[int(i/2), i%2])  # TODO: find a better way to set k
    # ax[int(i / 2), i % 2].set(title='spectral transformation at t=' + str(i))
    ax[int(i / 2), i % 2].set(ylabel='t='+str(i))#, xlabel='graph at t='+str(i))
    # plt.savefig("transform_t+" + str(i) + '.png')
    # plt.clf()
    coefs.append(list(coef))
    B = lp.predict_links(graph_t, f, k=k)
    dist = spar.norm(nx.adjacency_matrix(graph_t1) - B, ord=1) / (B.shape[0] * B.shape[1])
    avg_t.append(np.mean(nx.adjacency_matrix(graph_t)))
    avg_t1.append(np.mean(nx.adjacency_matrix(graph_t1)))
    avg_t2.append(np.mean(nx.adjacency_matrix(graph_t2)))
    # print("std of holds", np.std(nx.adjacency_matrix(graphB)))
    in_sample_err.append(dist)
    C = lp.predict_links(graph_t1, f, k=k)
    dist = spar.norm(nx.adjacency_matrix(graph_t2) - C, ord=1) / (C.shape[0] * C.shape[1])
    out_sample_err.append(dist)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
plt.savefig("tranforms.png")
plt.show()
pd.DataFrame(np.vstack((np.array([avg_t, avg_t1, avg_t2, delta_t_t1, delta_t1_t2, in_sample_err, out_sample_err]), np.array(coefs).transpose())).transpose(),
             columns=['avg_t', ' avg_t1', ' avg_t2', ' delta_t_t1', ' delta_t1_t2', ' in_sample_err',
                      ' out_sample_err']+['alpha'+str(i) for i in range(len(np.array(coefs).transpose()))]).to_csv("results.csv")
