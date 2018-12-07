import pickle

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_graphs_from_csv(filename):
    """
    The main function of the file.
    Given a CSV file of the format (date, cusip, fundno, shares), extracts a graph for each distinct time.

    Parameters
    ----------
    filename : type
        Path to csv file.

    Returns
    -------
    list(tuple(date,networkx graph))
        A list of weighted bipartite graphs - one for each distinct date, along with the date.
    """
    data = pd.read_csv(filename)

    unique_dates = data['fdate'].unique()
    unique_fund_numbers = data['fundno'].unique() #One set of the bipartite graph
    unique_stocks = data['cusip'].unique()

    isolated_vertices = create_edgeless_graph(unique_fund_numbers,unique_stocks) #Now for each date, we add edges corresponding to the graph configuration on that date

    holding_graphs = []
    for date in unique_dates:
        holding_graph = isolated_vertices.copy()
        holdings_on_date = data[data['fdate']==date]
        holding_graphs.append((date,create_graph_for_date(holding_graph,holdings_on_date)))

    return holding_graphs

def save_graphs_to_pickle(graphs,pickle_filename):
    """
    Pickles the list of graphs.

    Parameters
    ----------
    graphs : list(tuple(date,networkx graph))

        A list of weighted bipartite graphs - one for each distinct date, along with the date.
    pickle_filename : string
        Output file path
    """

    outfile = open(pickle_filename,'wb')
    pickle.dump(graphs,outfile)
    outfile.close()

def create_graph_for_date(graph,holdings):
    """Adds edges to the graph corresponding to holdings on that particular day

    Parameters
    ----------
    graph : networkx graph
        All isolated vertices that edges may be drawn between
    holdings : pandas DataFrame
        Holdings corresponding to a particular date

    Returns
    -------
    networkx graph
        Holding graph for the given input data
    """
    edges = []
    for index, row in tqdm(holdings.iterrows()):
        edges.append((row['cusip'],row['fundno'],row['shares']))

    graph.add_weighted_edges_from(edges)
    return graph
def create_edgeless_graph(fund_numbers,stocks):
    """Each graph corresponds to holdings at a particular time.
    But each graph MUST have the same vertex set (i.e. their adjacency matrices must have the same dimensions).
    This function creates the vertex set of the entire graph, without edges.

    Parameters
    ----------
    fund_numbers : numpy array
        Unique funds in the csv file. One set of the bipartite graph
    stocks : numpy array
        Unique stocks held by funds in fund_numbers at some point in time. The other set of the bipartite graph

    Returns
    -------
    networkx graph
        A graph of isolated vertices, no edges

    """
    graph =nx.Graph()
    graph.add_nodes_from(stocks)
    graph.add_nodes_from(fund_numbers)
    return graph
def main():
    graphs = get_graphs_from_csv('ds1.csv')
    save_graphs_to_pickle(graphs, 'dataset1')
if __name__ == '__main__':
    main()
