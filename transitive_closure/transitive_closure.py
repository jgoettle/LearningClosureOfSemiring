from operator import add, mul, sub
from operator import mul
import numpy as np
import networkx as nx

# (sr_plus, sr_times, sr_zero, sr_id)
semiring_options = {
        'min-plus': (min, add, np.inf, 0),
        'min-times': (min, mul, np.inf, 1),
        'plus-times': (add, mul, 0, 1),
        'max-times': (max, mul, 0, 1),
        'max-plus': (max, add, 0, 0),
        'max-min': (max, min, 0, 1) #sr_id is 1 if the values are bound by 1
    }

def transitive_closure(W, semiring = 'max-times'):
    """computes the transitive closure with the Floyd-Warshall Algorithm.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        semiring (string): semiring type

    Returns:
        W_tc (np.ndarray): [d, d] weighted adj matrix of the transitive closure of a DAG
    """

    if semiring in semiring_options:
        sr_plus, sr_times, sr_zero, sr_id = semiring_options[semiring]
    else:
        raise ValueError('unknown semiring type')

    d = W.shape[0]
    W_tc = np.zeros((d+1,d,d))
    W_tc[0] = np.where((W == 0), sr_zero, W)
    for k in range(d):
        for i in range(d):
            for j in range(d):
                W_tc[k+1][i][j] = sr_plus(W_tc[k][i][j], sr_times(W_tc[k][i][k], W_tc[k][k][j]))
                #print (W_tc[i][j])
    return np.where(W_tc[d] == sr_zero, 0, W_tc[d])

def transitive_closure_dag(W, semiring = 'max-times'):
    """computes the transitive closure of a DAG over a semiring.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        semiring (string): semiring type

    Returns:
        W_tc (np.ndarray): [d, d] weighted adj matrix of the transitive closure of a DAG
    """

    if semiring in semiring_options:
        sr_plus, sr_times, sr_zero, sr_id = semiring_options[semiring]
    else:
        raise ValueError('unknown semiring type')
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    topo_order = list(nx.topological_sort(G))

    W_tc = np.where((W == 0), sr_zero, W)

    for source in topo_order:
        processing = False
        for u in topo_order:
            if u == source:
                processing = True  # Start processing from the source onward.
            if not processing:
                continue
            
            for v in G.predecessors(u):
                W_tc[source][u] = sr_plus(W_tc[source][u], sr_times(W_tc[source][v], W_tc[v][u]))
    
    return np.where(W_tc == sr_zero, 0, W_tc)


def transitive_closure_dag_optimized(W, semiring='max-times'):

    if semiring in semiring_options:
        sr_plus, sr_times, sr_zero, sr_id = semiring_options[semiring]
    else:
        raise ValueError('unknown semiring type')

    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    topo_order = list(nx.topological_sort(G))
    
    W_tc = np.where(W == 0, sr_zero, W)

    for u in topo_order:
        successors_u = list(G.successors(u))
        if not successors_u:
            continue
        

        reach_from = W_tc[:, u]  # (column "u" for all p)
        
        for v in successors_u:
            for p in range(W_tc.shape[0]):
                if reach_from[p] != sr_zero:
                    W_tc[p, v] = sr_plus(W_tc[p, v], sr_times(W_tc[p, u], W_tc[u, v]))

    W_tc = np.where(W_tc == sr_zero, 0, W_tc)
    return W_tc

def transitive_reduction_weighted(W_tc, semiring = 'max-times'):
    """computes the transitive reduction

    Args:
        W_tc (np.ndarray): [d, d] weighted adj matrix of transitive closure of a DAG
        semiring (string): semiring type

    Returns:
        W_tc (np.ndarray): [d, d] weighted adj matrix of the transitive reduction
    """
    if semiring in semiring_options:
        sr_plus, sr_times, sr_zero, sr_id = semiring_options[semiring]
        if semiring == "plus-times":
            I = np.eye(W_tc.shape[0])
            W_tr = I- np.linalg.inv(I+W_tc)
            return np.where(abs(W_tr)<1e-8, 0, W_tr)
    else:
        raise ValueError('unknown semiring type')

    W_tr = np.where((W_tc == 0), sr_zero, W_tc)

    G = nx.from_numpy_array(W_tc, create_using = nx.DiGraph)
    topo_order = list(nx.topological_sort(G))

    for u in topo_order:
        successors = list(G.successors(u))
        for v in successors:
            for w in successors:
                if w != v and W_tc[w][v]!= sr_zero:
                    w1 = sr_times(W_tc[u][w],W_tc[w][v])
                    if sr_plus(w1, W_tc[u][v]) == w1:
                        W_tr[u][v] = sr_zero
                        break

    return np.where(W_tr == sr_zero, 0, W_tr)


def transitive_reduction_weighted_with_correction(W_tc, semiring = 'max-times', correction = "plus", correction_val=0.1):
    """computes the transitive reduction with correction.

    Args:
        W_tc (np.ndarray): [d, d] weighted adj matrix of transitive closure of a DAG
        semiring (string): semiring type

    Returns:
        W_tc (np.ndarray): [d, d] weighted adj matrix of the transitive reduction
    """
    if semiring in semiring_options:
        sr_plus, sr_times, sr_zero, sr_id = semiring_options[semiring]
        # no correction when plus-times, use closed formula
        if semiring == "plus-times": 
            I = np.eye(W_tc.shape[0])
            W_tr = I- np.linalg.inv(I+W_tc)
            return np.where(abs(W_tr)<1e-8, 0, W_tr)
    else:
        raise ValueError('unknown semiring type')

    correction_f = add
    if correction == "none":
        correction_val = 0
    elif correction == "semiring-times" or correction == "sr-t":
        correction_f = sr_times
    elif correction == "minus":
        correction_f = sub
    elif correction != "plus":
        print(correction)
        raise ValueError('correction not known')

    W_tr = np.where((W_tc == 0), sr_zero, W_tc)

    G = nx.from_numpy_array(W_tc, create_using = nx.DiGraph)
    topo_order = list(nx.topological_sort(G))
    for u in topo_order:
        successors = list(G.successors(u))
        for v in successors:
            for w in successors:
                if w != v and W_tc[w][v]!= sr_zero:
                    w_comp = correction_f(sr_times(W_tc[u][w],W_tc[w][v]), correction_val)
                    if sr_plus(w_comp, W_tc[u][v]) == w_comp:
                        W_tr[u][v] = sr_zero
                        break
    return  np.where(W_tr == sr_zero, 0, W_tr)


def compute_linear_sem(W_tc):
    """
    computes the adj. matrix of the plus-times DAG, result is the same as
    I = np.eye(W_tc.shape[0])
    return I- np.linalg.inv(I+W_tc)

    Args:
        W_tc (np.ndarray): [d, d] weighted adj matrix of transitive closure of a DAG
        semiring (string): semiring type

    Returns:
        W_tc (np.ndarray): [d, d] weighted adj matrix
    """
    W_linear_sem = W_tc.copy()
    G = nx.from_numpy_array(W_tc, create_using=nx.DiGraph)
    topo_order = list(nx.topological_sort(G))
    
    
    for i in reversed(topo_order):
        for j in topo_order:
            for p in G.predecessors(j):
                    W_linear_sem[i][j] = W_linear_sem[i][j] - W_tc[i][p]*W_linear_sem[p][j]
                
    return W_linear_sem


def transitive_reduction_binary(W):
    G = nx.transitive_reduction(nx.from_numpy_array(W, create_using=nx.DiGraph))

    '''
    own implementation
    topo_order = list(nx.topological_sort(G))

    for u in topo_order:
        successors = list(G.successors(u))
        for v in successors:
            for w in successors:
                if w!=v and G.has_edge(w, v):
                    G.remove_edge(u,v)
                    break
    '''
    return nx.to_numpy_array(G, weight="weight")