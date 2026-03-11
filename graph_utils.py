"""
graph_utils.py
Generates graphs and computes ground-truth longest paths for training/evaluation.
Uses exact solvers for small graphs, approximations for larger ones.
"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from itertools import permutations
from typing import Optional
import random


# ─────────────────────────────────────────
# Ground truth: exact longest path (small graphs only)
# ─────────────────────────────────────────

def exact_longest_path(G: nx.Graph) -> tuple[int, list]:
    """
    Fast longest path approximation using repeated random DFS.
    Guaranteed to terminate quickly. Not exact but good enough for training labels.
    """
    best_len = 0
    best_path = []
    nodes = list(G.nodes())
    rng = random.Random(42)

    # Try 50 random starting nodes, DFS from each
    starts = rng.sample(nodes, min(50, len(nodes)))
    for src in starts:
        path = _random_dfs(G, src, rng)
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path

    return best_len, best_path


def _random_dfs(G: nx.Graph, start, rng: random.Random) -> list:
    """Random DFS — greedy path extension with random neighbor ordering."""
    visited = {start}
    path = [start]
    while True:
        neighbors = [n for n in G.neighbors(path[-1]) if n not in visited]
        if not neighbors:
            break
        nxt = rng.choice(neighbors)
        visited.add(nxt)
        path.append(nxt)
    return path


def dp_longest_path_dag(G: nx.DiGraph) -> tuple[int, list]:
    """Exact longest path on a DAG via topological DP. O(V+E)."""
    topo = list(nx.topological_sort(G))
    dist = {n: 0 for n in G.nodes()}
    prev = {n: None for n in G.nodes()}
    for u in topo:
        for v in G.successors(u):
            if dist[u] + 1 > dist[v]:
                dist[v] = dist[u] + 1
                prev[v] = u
    end = max(dist, key=dist.get)
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[end], path


# ─────────────────────────────────────────
# Graph generators — varied structure
# ─────────────────────────────────────────

def make_random_graph(n: int, p: float, seed: int = None) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=seed)

def make_grid_graph(rows: int, cols: int) -> nx.Graph:
    return nx.grid_2d_graph(rows, cols)

def make_barabasi_albert(n: int, m: int, seed: int = None) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)

def make_cycle_graph(n: int) -> nx.Graph:
    return nx.cycle_graph(n)

def make_complete_graph(n: int) -> nx.Graph:
    return nx.complete_graph(n)

def make_random_dag(n: int, p: float, seed: int = None) -> nx.DiGraph:
    """Random DAG: add edges only from lower to higher indexed nodes."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                G.add_edge(i, j)
    return G


GRAPH_FACTORIES = {
    "erdos_renyi":     lambda n, seed: make_random_graph(n, p=0.3, seed=seed),
    "barabasi_albert": lambda n, seed: make_barabasi_albert(n, m=2, seed=seed),
    "cycle":           lambda n, seed: make_cycle_graph(n),
    "grid":            lambda n, seed: make_grid_graph(int(n**0.5) or 3, int(n**0.5) or 3),
    "dag":             lambda n, seed: make_random_dag(n, p=0.3, seed=seed),
}


# ─────────────────────────────────────────
# Convert to PyG Data objects
# ─────────────────────────────────────────

def graph_to_pyg(G: nx.Graph, longest_path_len: int) -> Data:
    """
    Convert networkx graph to PyTorch Geometric Data.
    Node features: degree (normalized), clustering coeff, betweenness centrality.
    Edge features: weight (normalized).
    """
    n = G.number_of_nodes()
    G = nx.convert_node_labels_to_integers(G)

    # --- NORMALIZATION 1: Node Degrees ---
    raw_degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
    # Divide by the theoretical maximum degree (n - 1) to squish to [0.0, 1.0]
    normalized_degrees = raw_degrees / max(n - 1, 1)
    
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)
    try:
        betweenness = np.array(
            list(nx.betweenness_centrality(G, normalized=True).values()),
            dtype=np.float32
        )
    except Exception:
        betweenness = np.zeros(n, dtype=np.float32)

    # Stack the normalized features
    x = torch.tensor(
        np.stack([normalized_degrees, clustering, betweenness], axis=1),
        dtype=torch.float
    )

    # --- NORMALIZATION 2: Edge Weights ---
    edge_index_list = list(G.edges())
    weights = []
    for u, v in edge_index_list:
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = float(np.random.randint(1, 10))
            
        # Divide by 10.0 (the max possible weight) to squish to [0.1, 1.0]
        normalized_weight = G[u][v]['weight'] / 10.0
        weights.append(normalized_weight)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    y = torch.tensor([longest_path_len / max(n - 1, 1)], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=n)

# ─────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────

def build_dataset(
    num_graphs: int = 1000,
    min_nodes: int = 4,
    max_nodes: int = 10,   # kept small — exact solver is exponential
    graph_types: Optional[list] = None,
    seed: int = 42,
    verbose: bool = True,
) -> list[Data]:
    """
    Build a dataset of (graph, longest_path_length) pairs.
    Keeps graphs small enough for exact solving.
    """
    rng = random.Random(seed)
    if graph_types is None:
        graph_types = list(GRAPH_FACTORIES.keys())

    dataset = []
    skipped = 0

    for i in range(num_graphs):
        gtype = rng.choice(graph_types)
        n = rng.randint(min_nodes, max_nodes)
        gseed = rng.randint(0, 10_000)

        G = GRAPH_FACTORIES[gtype](n, gseed)

        # Skip disconnected or trivial graphs
        if G.number_of_edges() == 0:
            skipped += 1
            continue

        # Use fast DP for DAGs, brute force otherwise
        if isinstance(G, nx.DiGraph) and nx.is_directed_acyclic_graph(G):
            lp_len, _ = dp_longest_path_dag(G)
        else:
            if not isinstance(G, nx.Graph):
                skipped += 1
                continue
            # Convert DiGraph to undirected for brute force if needed
            if isinstance(G, nx.DiGraph):
                G = G.to_undirected()
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            if G.number_of_nodes() < 2:
                skipped += 1
                continue
            lp_len, _ = exact_longest_path(G)

        data = graph_to_pyg(G, lp_len)
        dataset.append(data)

        if verbose and (i + 1) % 100 == 0:
            print(f"  Built {i+1}/{num_graphs} graphs (skipped {skipped})")

    print(f"Dataset complete: {len(dataset)} graphs, {skipped} skipped.")
    return dataset


if __name__ == "__main__":
    print("Building small test dataset...")
    ds = build_dataset(num_graphs=50, verbose=True)
    print(f"Sample: nodes={ds[0].num_nodes}, y={ds[0].y.item():.3f}")