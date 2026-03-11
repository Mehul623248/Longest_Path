import torch
import networkx as nx

import torch
import math

def evaluate_neighbors_heuristic(current_node, neighbors, graph):
    """
    Evaluates neighbors using the exact SiLU-equivalent symbolic regression 
    expression discovered in pipeline_report (3).json: x0 / (1 + exp(-x0))
    """
    scores = {}
    for neighbor in neighbors:
        # x0 is our input feature. Here, we use the raw edge weight.
        # In a more complex model, this could be node degree, centrality, etc.
        x0 = graph[current_node][neighbor].get('weight', 1.0)
        
        # Ensure x0 is a tensor for PyTorch math operations
        x0_tensor = torch.tensor(float(x0))
        
        # --- YOUR SYMBOLIC REGRESSION EQUATION ---
        # Equation: (1.04 * x0)⁶ + (1.04 * x0)³ - 0.07
        gnn_score = (1.04 * x0_tensor)**6 + (1.04 * x0_tensor)**3 - 0.07
        
        
        scores[neighbor] = gnn_score.item()
        
    return scores

        


def beam_search_longest_path(graph, start_node, beam_width=3):
    """
    Executes a Beam Search to find the longest simple path.
    Tracks BOTH the heuristic score (for navigation) and the true score (actual edge weights).
    """
    # State tracking tuple: 
    # (heuristic_score, true_score, current_node, path_history, visited_set)
    beam = [(0.0, 0.0, start_node, [start_node], {start_node})]
    
    best_overall_path = []
    best_overall_true_score = 0.0
    best_overall_heuristic_score = 0.0

    while beam:
        next_beam_candidates = []
        
        for heur_score, true_score, current_node, path, visited in beam:
            # 1. Get neighbors and score them using your neural network equation
            neighbors = list(graph.neighbors(current_node))
            neighbor_scores = evaluate_neighbors_heuristic(current_node, neighbors, graph)
            
            # 2. LOGIT MASKING: Prevent cycles
            for neighbor in neighbors:
                if neighbor in visited:
                    neighbor_scores[neighbor] = float('-inf')

            # 3. Generate new valid states
            valid_moves_made = False
            for neighbor, n_heur_score in neighbor_scores.items():
                if n_heur_score > float('-inf'): 
                    valid_moves_made = True
                    
                    # --- THE TRUE SCORE TRACKER ---
                    # Pull the actual physical weight of the edge from the graph
                    raw_edge_weight = graph[current_node][neighbor].get('weight', 1.0)
                    
                    # Tally the true score alongside the heuristic score
                    new_true_score = true_score + raw_edge_weight
                    new_heur_score = heur_score + n_heur_score
                    
                    new_path = path + [neighbor]
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    
                    next_beam_candidates.append(
                        (new_heur_score, new_true_score, neighbor, new_path, new_visited)
                    )
            
            # 4. If we hit a dead end, log the final path
            # We save the path if its TRUE score is the best we have seen so far
            if not valid_moves_made:
                if true_score > best_overall_true_score:
                    best_overall_true_score = true_score
                    best_overall_heuristic_score = heur_score
                    best_overall_path = path

        # 5. Prune the search tree: Sort strictly by the HEURISTIC score
        # The algorithm navigates using the compass, not the true distance
        next_beam_candidates.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam_candidates[:beam_width]
        
    return best_overall_path, best_overall_true_score, best_overall_heuristic_score

# --- Execution Block ---
if __name__ == "__main__":
    # 1. Create a random complex micro-graph to test on
    G = nx.erdos_renyi_graph(n=20, p=0.3, seed=42)
    
    # Assign random weights to make the longest path non-obvious
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = torch.randint(1, 10, (1,)).item()

    # 2. Run the Beam Search
    START_NODE = 0
    BEAM_WIDTH = 5 # Try tweaking this! Higher = more exact but slower. Lower = faster but might miss the best path.
    
    print(f"Starting search from Node {START_NODE} with Beam Width {BEAM_WIDTH}...")
    longest_path, true_score, heur_score = beam_search_longest_path(G, start_node=0, beam_width=5)

    print(f"Path Length (Nodes): {len(longest_path)}")
    print(f"True Path Weight (Actual Length): {true_score}")
    print(f"Heuristic Score (The NN's Compass): {heur_score}")