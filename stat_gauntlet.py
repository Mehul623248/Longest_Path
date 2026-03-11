"""
benchmark_gauntlet.py
Neuro-Symbolic Graph Routing: Final Evaluation Benchmark
"""

import torch
import networkx as nx
import numpy as np
import math
from scipy import stats

# ==========================================
# 1. ROUTING HEURISTICS
# ==========================================

def heuristic_pure_greedy(current_node, neighbors, graph):
    """Classical Baseline 1: Pure Greedy (Weight only)"""
    return {neighbor: float(graph[current_node][neighbor].get('weight', 1.0)) for neighbor in neighbors}

def heuristic_survival(current_node, neighbors, graph):
    """Classical Baseline 2: Warnsdorff/Survival (Weight + 0.5 * Degree)"""
    scores = {}
    for neighbor in neighbors:
        weight = float(graph[current_node][neighbor].get('weight', 1.0))
        degree = len(list(graph.neighbors(neighbor)))
        scores[neighbor] = weight + (0.5 * degree)
    return scores

def heuristic_eq1_mult(current_node, neighbors, graph):
    """Evolved Champion 1 (Sparse Erdos-Renyi): The Multiplicative Gate"""
    scores = {}
    n = graph.number_of_nodes()
    for neighbor in neighbors:
        raw_weight = float(graph[current_node][neighbor].get('weight', 1.0))
        raw_degree = float(len(list(graph.neighbors(neighbor))))
        
        x0 = raw_weight / 10.0
        x1 = raw_degree / max(n - 1, 1)
        
        # Evolved Math: (0.90 * Weight - 0.13) * Degree
        scores[neighbor] = (0.90 * x0 - 0.13) * x1
    return scores

def heuristic_eq2_blend(current_node, neighbors, graph):
    """Evolved Champion 2 (Dense Erdos-Renyi): The Blended Sigmoid"""
    scores = {}
    n = graph.number_of_nodes()
    for neighbor in neighbors:
        raw_weight = float(graph[current_node][neighbor].get('weight', 1.0))
        raw_degree = float(len(list(graph.neighbors(neighbor))))
        
        x0 = raw_weight / 10.0
        x1 = raw_degree / max(n - 1, 1)
        
        try:
            sig_x1 = 1.0 / (1.0 + math.exp(-x1))
        except OverflowError:
            sig_x1 = 0.0
            
        # Evolved Math: 0.52*(Weight * Degree) + 0.48*(Sigmoid(Degree))
        scores[neighbor] = 0.52 * (x0 * x1) + 0.48 * sig_x1
    return scores

# ==========================================
# 2. EVALUATION PROXY
# ==========================================

def beam_search(graph, start_node, scoring_func, beam_width=1):
    """Strict routing: Evaluates heuristics by forcing them to navigate the graph."""
    beam = [(0.0, 0.0, start_node, [start_node], {start_node})]
    best_true_score = 0.0

    while beam:
        next_beam = []
        for heur_score, true_score, current_node, path, visited in beam:
            neighbors = list(graph.neighbors(current_node))
            neighbor_scores = scoring_func(current_node, neighbors, graph)

            for neighbor in neighbors:
                if neighbor in visited:
                    neighbor_scores[neighbor] = float('-inf')

            valid_moves = False
            for neighbor, n_heur in neighbor_scores.items():
                if n_heur > float('-inf'):
                    valid_moves = True
                    raw_weight = float(graph[current_node][neighbor].get('weight', 1.0))
                    next_beam.append((
                        heur_score + n_heur,
                        true_score + raw_weight,
                        neighbor,
                        path + [neighbor],
                        visited | {neighbor}
                    ))

            if not valid_moves and true_score > best_true_score:
                best_true_score = true_score

        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    return best_true_score

# ==========================================
# 3. STATISTICAL HELPER
# ==========================================

def print_statistical_significance(baseline_name, baseline_scores, challenger_name, challenger_scores):
    """Runs a Wilcoxon Signed-Rank Test on paired graph routing scores."""
    statistic, p_value = stats.wilcoxon(challenger_scores, baseline_scores)
    
    print(f"  {challenger_name} vs {baseline_name}: p-value = {p_value:.5f}")
    if p_value < 0.05:
        if np.mean(challenger_scores) > np.mean(baseline_scores):
            print(f"    -> SIGNIFICANT WIN for {challenger_name}")
        else:
            print(f"    -> SIGNIFICANT LOSS for {challenger_name}")
    else:
        print("    -> NOT STATISTICALLY SIGNIFICANT (Tie)")

# ==========================================
# 4. EXPERIMENT EXECUTION
# ==========================================

def run_experiment(graph_generator_func, graph_name, num_graphs=100, beam_width=1):
    greedy, survival, eq1, eq2 = [], [], [], []
    
    for _ in range(num_graphs):
        G = graph_generator_func()
        
        # Apply random heavy/light edge weights
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = float(np.random.randint(1, 10))
            
        start_node = 0
        greedy.append(beam_search(G, start_node, heuristic_pure_greedy, beam_width))
        survival.append(beam_search(G, start_node, heuristic_survival, beam_width))
        eq1.append(beam_search(G, start_node, heuristic_eq1_mult, beam_width))
        eq2.append(beam_search(G, start_node, heuristic_eq2_blend, beam_width))
        
    print(f"\n" + "="*50)
    print(f" ENVIRONMENT: {graph_name.upper()}")
    print("="*50)
    print(f"Pure Greedy     : {np.mean(greedy):.2f}")
    print(f"Survival (0.5x) : {np.mean(survival):.2f}")
    print(f"Eq1 (Mult Gate) : {np.mean(eq1):.2f}")
    print(f"Eq2 (Blended)   : {np.mean(eq2):.2f}")

    print("\n--- Statistical Significance (vs Pure Greedy) ---")
    print_statistical_significance("Pure Greedy", greedy, "Eq1 (Mult Gate)", eq1)
    print_statistical_significance("Pure Greedy", greedy, "Eq2 (Blended)", eq2)

    print("\n--- Statistical Significance (vs Survival) ---")
    print_statistical_significance("Survival", survival, "Eq1 (Mult Gate)", eq1)
    print_statistical_significance("Survival", survival, "Eq2 (Blended)", eq2)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    N_NODES = 100
    
    # 1. Scale-Free (Barabasi-Albert) - Super-Hubs
    def gen_ba():
        return nx.barabasi_albert_graph(n=N_NODES, m=2)

    # 2. Small-World (Watts-Strogatz) - Clustered Neighborhoods
    def gen_ws():
        G = nx.watts_strogatz_graph(n=N_NODES, k=4, p=0.1)
        while not nx.is_connected(G):
            G = nx.watts_strogatz_graph(n=N_NODES, k=4, p=0.1)
        return G
        
    # 3. Random Sparse (Erdos-Renyi) - The Training Environment
    def gen_er():
        G = nx.erdos_renyi_graph(n=N_NODES, p=0.025)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n=N_NODES, p=0.025)
        return G

    print(f"Initiating 100-Node Graph Gauntlet...")
    run_experiment(gen_er, "Random Sparse Networks (Erdos-Renyi p=0.025)")
    run_experiment(gen_ba, "Scale-Free Networks (Barabasi-Albert m=2)")
    run_experiment(gen_ws, "Small-World Networks (Watts-Strogatz k=4, p=0.1)")