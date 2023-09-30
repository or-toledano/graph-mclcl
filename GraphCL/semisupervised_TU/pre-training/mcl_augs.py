import networkx as nx
import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from tqdm import tqdm


def prune(A, t):
    P = A.toarray()
    P[P < t] = 0
    P = ss.csc_matrix(P)
    P.eliminate_zeros()
    row_indices = A.argmax(axis=0).reshape((A.shape[1],))
    col_indices = np.arange(A.shape[1])
    P[row_indices, col_indices] = A[row_indices, col_indices]
    return P


def converged(A):
    A_T = A.toarray().T
    for col in A_T:
        non_zero_element = col[np.nonzero(col)]
        if not np.allclose(non_zero_element, non_zero_element[0]):
            return False
    return True


def MCL_raw(G, nodes, r, t=1e-6, steps=20):
    A = nx.adjacency_matrix(G, nodelist=nodes)
    A = normalize(A, norm='l1', axis=0)
    A_i = A
    for _ in tqdm(range(steps), total=steps):
        # do expansion step
        A_i = A_i.T * A_i
        # do inflation step
        A_i = normalize(A_i.power(r), norm='l1', axis=0)
        A_i = prune(A_i, t)
        if converged(A_i):
            break
    return A_i


def MCL(G, nodes, r, t=1e-6, steps=20):
    A_i = MCL_raw(G, nodes, r=r, t=t, steps=steps)
    return get_clusters(A_i, nodes)


def get_clusters(A, nodes):
    attractors = A.diagonal().nonzero()[0]

    clusters = set()

    for attractor in attractors:
        cluster = tuple(A.getrow(attractor).nonzero()[1].tolist())
        if len(cluster) >= 5:
            named_cluster = tuple(nodes[n] for n in cluster)
            clusters.add(named_cluster)

    return sorted(list(clusters))


def preproc_graph(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.k_core(G, 3)
    for node in G.nodes():
        G.add_edge(node, node)
    return G


def get_graph(path):
    with open(path, 'r') as f:
        data = f.readlines()
    G = nx.Graph([line[:-1].split('\t') for line in data])
    return preproc_graph(G)
