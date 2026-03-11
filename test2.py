import torch
import networkx as nx
import numpy as np
from scipy import stats

# --- Your Heuristic Functions ---


def heuristic_pure_greedy(current_node, neighbors, graph):
    """Classical Boss 1: Always prioritize the heaviest edge linearly."""
    scores = {}
    for neighbor in neighbors:
        weight = float(graph[current_node][neighbor].get('weight', 1.0))
        scores[neighbor] = weight  # f(x) = x
    return scores

def heuristic_survival(current_node, neighbors, graph):
    """Classical Boss 2: Balance heavy edges with the number of escape routes."""
    scores = {}
    for neighbor in neighbors:
        weight = float(graph[current_node][neighbor].get('weight', 1.0))
        # Count how many exit routes this neighbor has
        degree = len(list(graph.neighbors(neighbor)))
        
        # Reward heavy weights, but also reward having lots of escape routes
        scores[neighbor] = weight + (0.5 * degree) 
    return scores

def heuristic_dummy(current_node, neighbors, graph):
    """Baseline: completely random guesses"""
    return {neighbor: torch.rand(1).item() for neighbor in neighbors}

def heuristic_silu(current_node, neighbors, graph):
    """Your SiLU Discovery: x0 / (1 + exp(-x0))"""
    scores = {}
    for neighbor in neighbors:
        x0 = graph[current_node][neighbor].get('weight', 1.0)
        x0_tensor = torch.tensor(float(x0))
        scores[neighbor] = (x0_tensor / (1.0 + torch.exp(-x0_tensor))).item()
    return scores

def heuristic_sextic(current_node, neighbors, graph):
    """Your Sextic Polynomial (Simulated as x^6 for this test)"""
    scores = {}
    for neighbor in neighbors:
        x0 = graph[current_node][neighbor].get('weight', 1.0)
        # Using a simple x^6 to represent the greedy explosion
        scores[neighbor] = (float(x0) ** 6) 
    return scores

# --- The Beam Search Wrapper ---
# (Same as before, but accepts the scoring function as an argument)
def beam_search(graph, start_node, scoring_func, beam_width=5):
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
                    raw_weight = graph[current_node][neighbor].get('weight', 1.0)
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

# --- The Statistical Gauntlet ---
if __name__ == "__main__":
    NUM_GRAPHS = 100
    BEAM_WIDTH = 5
    
    dummy_scores = []
    heuristic_survival_scores = []
    heuristic_pure_greedy_scores = []
    silu_scores = []
    sextic_scores = []
    
    print(f"Generating {NUM_GRAPHS} random graphs and running gauntlet...")
    
    for i in range(NUM_GRAPHS):
        # Generate a random graph
        G = nx.erdos_renyi_graph(n=25, p=0.3)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = torch.randint(1, 10, (1,)).item()
            
        start_node = 0
        
        # Run all three algorithms on the EXACT SAME graph
        heuristic_survival_scores.append(beam_search(G, start_node, heuristic_survival, BEAM_WIDTH))
        heuristic_pure_greedy_scores.append(beam_search(G, start_node, heuristic_pure_greedy, BEAM_WIDTH))
        dummy_scores.append(beam_search(G, start_node, heuristic_dummy, BEAM_WIDTH))
        silu_scores.append(beam_search(G, start_node, heuristic_silu, BEAM_WIDTH))
        sextic_scores.append(beam_search(G, start_node, heuristic_sextic, BEAM_WIDTH))
        
    # --- Analysis & Statistics ---
    print("\n=== Average True Scores ===")
    print(f"Dummy Baseline : {np.mean(dummy_scores):.2f}")
    print(f"SiLU Equation  : {np.mean(silu_scores):.2f}")
    print(f"Sextic Equation: {np.mean(sextic_scores):.2f}")
    print(f"Survival Equation: {np.mean(heuristic_survival_scores):.2f}")
    print(f"Pure Greedy Equation: {np.mean(heuristic_pure_greedy_scores):.2f}")

    print("\n=== Statistical Significance (Wilcoxon Signed-Rank Test) ===")
    # We use Wilcoxon because path scores are often non-normally distributed
    
    # 1. Did SiLU beat Dummy?
    stat, p_silu_vs_dummy = stats.wilcoxon(silu_scores, dummy_scores)
    print(f"SiLU vs Dummy   -> p-value: {p_silu_vs_dummy:.5f}")
    
    # 2. Did Sextic beat Dummy?
    stat, p_sextic_vs_dummy = stats.wilcoxon(sextic_scores, dummy_scores)
    print(f"Sextic vs Dummy -> p-value: {p_sextic_vs_dummy:.5f}")
    
    # 3. Did SiLU beat Sextic?
    stat, p_silu_vs_sextic = stats.wilcoxon(silu_scores, sextic_scores)
    print(f"SiLU vs Sextic  -> p-value: {p_silu_vs_sextic:.5f}")

    #4. Did SiLu beat Pure Greedy?
    stat, p_silu_vs_pure_greedy = stats.wilcoxon(silu_scores, heuristic_pure_greedy_scores)
    print(f"SiLU vs Pure Greedy  -> p-value: {p_silu_vs_pure_greedy:.5f}")

    #5. Did Sextic beat Pure Greedy?
    stat, p_sextic_vs_pure_greedy = stats.wilcoxon(sextic_scores, heuristic_pure_greedy_scores)
    print(f"Sextic vs Pure Greedy  -> p-value: {p_sextic_vs_pure_greedy:.5f}")

    #6 Did SiLU beat Survival?
    stat, p_silu_vs_survival = stats.wilcoxon(silu_scores, heuristic_survival_scores)
    print(f"SiLU vs Survival  -> p-value: {p_silu_vs_survival:.5f}")

    #7 Did Sextic beat Survival?
    stat, p_sextic_vs_survival = stats.wilcoxon(sextic_scores, heuristic_survival_scores)
    print(f"Sextic vs Survival  -> p-value: {p_sextic_vs_survival:.5f}")
    
    