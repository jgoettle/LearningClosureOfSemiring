
import time
from matplotlib import pyplot as plt
import numpy as np
from transitive_closure.transitive_closure import  transitive_closure_dag, transitive_reduction_binary, transitive_reduction_weighted, transitive_reduction_weighted_with_correction
import networkx as nx

def remove_cyclic_edges_by_weight(W):
    G = nx.from_numpy_array(W, create_using=nx.DiGraph) 

    sorted_edges = sorted(G.edges(data=True), key=lambda x: abs(x[2]["weight"]))
    num_rem = 0
    for u, v, data in sorted_edges:
        G.remove_edge(u, v)
        if not nx.has_path(G, v, u):
            G.add_edge(u, v, weight=data["weight"])
        else:
            num_rem = num_rem + 1
    print("{} edges were removed to make the graph a DAG".format(num_rem))
    return nx.to_numpy_array(G, weight="weight"), num_rem

def visualize_dag(W):
    """
    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG

    Returns:
        W_tc (np.ndarray): [d, d] weighted adj matrix of the transitive closure of a DAG
    """
    
    G = nx.from_numpy_array(W, create_using=nx.DiGraph) 
    
    # https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(G, subset_key="layer")

    plt.figure(figsize=(15,5))
    
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='grey', arrows=True)

    edge_labels = {edge: f'{weight:.2f}' for edge, weight in nx.get_edge_attributes(G, 'weight').items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='black')


    plt.show()

def plot_matrix_distribution(matrices):
    all_entries = np.concatenate([matrix.flatten() for matrix in matrices])
    all_entries = all_entries[all_entries!=0]

    min = np.percentile(all_entries, 2)
    max = np.percentile(all_entries, 98)

    num_bins = 100
    bins = np.linspace(min, max, num_bins)


    hist, bin_edges = np.histogram(all_entries, bins=bins)


    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0] 
    plt.bar(bin_centers, hist, width=bin_width, align='center',color='blue')

    plt.xlabel("Values")
    plt.ylabel("Number of entries")

    plt.tight_layout()

    plt.show()
    