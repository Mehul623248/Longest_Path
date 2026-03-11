import os
import json
import random
import numpy as np
import torch
import networkx as nx
from dataclasses import dataclass, field
from typing import Callable, Optional

# ─────────────────────────────────────────
# Transform representation
# ─────────────────────────────────────────

PRIMITIVES = [
    # --- The Basics ---
    ("relu",         lambda x0, x1=None: torch.relu(x0)),
    ("sigmoid",      lambda x0, x1=None: torch.sigmoid(x0)),
    ("tanh",         lambda x0, x1=None: torch.tanh(x0)),
    ("softplus",     lambda x0, x1=None: torch.log1p(torch.exp(x0))),
    ("cube",         lambda x0, x1=None: x0 ** 3),
    
    # --- The Heavy 1D Hybrids (Weight Gaters) ---
    ("silu",         lambda x0, x1=None: x0 * torch.sigmoid(x0)),
    ("silu_cube",    lambda x0, x1=None: x0 * torch.sigmoid(x0) + 0.5 * x0**3),
    ("sixth_degree", lambda x0, x1=None: x0**6 + x0**3),
    ("cube_sixth",   lambda x0, x1=None: torch.clamp(x0**3 + 0.5 * x0**6, -10, 10)),
    ("adaptive_silu",lambda x0, x1=None: x0 * torch.sigmoid(x0 * torch.tanh(torch.exp(x0 * (x0 - 0.315)).clamp(-10, 10)))),
    
    # --- The 2D Topology Operators ---
    ("degree",       lambda x0, x1=None: x1 if x1 is not None else torch.ones_like(x0)),
    ("add_2d",       lambda x0, x1=None: x0 + (x1 if x1 is not None else 0)),
    ("mult_2d",      lambda x0, x1=None: x0 * (x1 if x1 is not None else 1)),
]
PRIMITIVE_DICT = {name: fn for name, fn in PRIMITIVES}

@dataclass
class Transform:
    name:     str
    fn:       Callable
    fitness:  float = -np.inf
    parents:  list  = field(default_factory=list)
    gen:      int   = 0

    def __call__(self, x0: torch.Tensor, x1: torch.Tensor = None) -> torch.Tensor:
        try:
            out = self.fn(x0, x1)
            return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        except Exception:
            return x0  

# ─────────────────────────────────────────
# Genetic operators
# ─────────────────────────────────────────

def compose(t1: Transform, t2: Transform, blend: float = 0.5) -> Transform:
    def fn(x0, x1=None): return blend * t1(x0, x1) + (1 - blend) * t2(x0, x1)
    return Transform(name=f"blend({t1.name},{t2.name},{blend:.2f})", fn=fn, parents=[t1.name, t2.name], gen=max(t1.gen, t2.gen) + 1)

def chain(t1: Transform, t2: Transform) -> Transform:
    def fn(x0, x1=None): return t2(t1(x0, x1), x1) # pass x1 through!
    return Transform(name=f"chain({t1.name},{t2.name})", fn=fn, parents=[t1.name, t2.name], gen=max(t1.gen, t2.gen) + 1)

def perturb(t: Transform, noise_scale: float = 0.1) -> Transform:
    alpha = 1.0 + random.gauss(0, noise_scale)
    beta  = random.gauss(0, noise_scale)
    def fn(x0, x1=None): return t(alpha * x0 + beta, x1)
    return Transform(name=f"perturb({t.name},{alpha:.2f},{beta:.2f})", fn=fn, parents=[t.name], gen=t.gen + 1)

def residual(t: Transform, scale: float = None) -> Transform:
    if scale is None: scale = random.uniform(0.1, 1.0)
    def fn(x0, x1=None): return x0 + scale * t(x0, x1)
    return Transform(name=f"residual({t.name},{scale:.2f})", fn=fn, parents=[t.name], gen=t.gen + 1)

def mutate(t: Transform) -> Transform:
    ops = [
        lambda: perturb(t),
        lambda: residual(t),
        lambda: compose(t, random.choice([Transform(n, f) for n, f in PRIMITIVES])),
        lambda: chain(t,  random.choice([Transform(n, f) for n, f in PRIMITIVES])),
    ]
    return random.choice(ops)()

# ─────────────────────────────────────────
# Fitness evaluation (Beam Search Proxy)
# ─────────────────────────────────────────

def evaluate_transform_beam(transform: Transform, eval_graphs: list, beam_width: int = 3) -> float:
    total_true_score = 0.0
    
    for G in eval_graphs:
        start_node = 0
        n = G.number_of_nodes()
        beam = [(0.0, 0.0, start_node, {start_node})]
        best_true_score = 0.0

        while beam:
            next_beam = []
            for heur_score, true_score, current_node, visited in beam:
                neighbors = [nx_n for nx_n in G.neighbors(current_node) if nx_n not in visited]
                
                if not neighbors:
                    if true_score > best_true_score:
                        best_true_score = true_score
                    continue
                    
                for neighbor in neighbors:
                    raw_weight = float(G[current_node][neighbor].get('weight', 1.0))
                    raw_degree = float(len(list(G.neighbors(neighbor))))
                    
                    x0 = torch.tensor([raw_weight / 10.0])
                    x1 = torch.tensor([raw_degree / max(n - 1, 1)])
                    
                    try:
                        n_heur = transform(x0, x1).item()
                    except Exception:
                        n_heur = float('-inf')
                        
                    next_beam.append((
                        heur_score + n_heur,
                        true_score + raw_weight,
                        neighbor,
                        visited | {neighbor}
                    ))
            
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:beam_width]
            
        total_true_score += best_true_score
        
    return total_true_score

# ─────────────────────────────────────────
# Evolutionary loop
# ─────────────────────────────────────────

def run_evolution(
    population_size: int = 30,
    generations:     int = 20,
    elite_frac:      float = 0.2,
    mutation_rate:   float = 0.5,
    crossover_rate:  float = 0.3,
    eval_epochs:     int = 20,
    seed_expressions: Optional[list] = None,
    output_dir:      str = "evolution_results",
) -> list:
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print("Building Beam Search evaluation graphs...")
    eval_graphs = []
    for _ in range(50):
        # --- CHANGED: p=0.12 creates a sparse graph with dead-end traps ---
        G = nx.erdos_renyi_graph(n=100, p=0.025) 
        
        # Ensure the graph isn't completely fractured into tiny pieces
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n=100, p=0.025)
            
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = float(np.random.randint(1, 10))
        eval_graphs.append(G)

    print(f"\nInitializing population of {population_size}...")
    population = [Transform(name, fn) for name, fn in PRIMITIVES]

    if seed_expressions:
        for name, fn in seed_expressions:
            population.append(Transform(name, fn))

    while len(population) < population_size:
        base = random.choice(population[:len(PRIMITIVES)])
        population.append(mutate(base))

    population = population[:population_size]

    history = []
    n_elite = max(1, int(elite_frac * population_size))

    print(f"Evolving for {generations} generations...")
    for gen in range(generations):
        print(f"\nGeneration {gen+1}/{generations}")

        for i, t in enumerate(population):
            if t.fitness == -np.inf:
                t.fitness = evaluate_transform_beam(t, eval_graphs, beam_width=1)
                print(f"  [{i+1}/{len(population)}] {t.name[:50]:50s}  fitness={t.fitness:.4f}")

        population.sort(key=lambda t: t.fitness, reverse=True)

        best = population[0]
        avg  = np.mean([t.fitness for t in population if t.fitness > -np.inf])
        print(f"  Best: {best.name[:60]}  fitness={best.fitness:.4f}")
        print(f"  Avg fitness: {avg:.4f}")

        elites = population[:n_elite]
        new_pop = list(elites)

        while len(new_pop) < population_size:
            r = random.random()
            if r < crossover_rate and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                child = compose(p1, p2, blend=random.uniform(0.3, 0.7))
            elif r < crossover_rate + mutation_rate:
                parent = random.choice(elites)
                child = mutate(parent)
            else:
                name, fn = random.choice(PRIMITIVES)
                child = Transform(name, fn)
                child.fitness = -np.inf

            if child.name not in [t.name for t in elites]:
                child.fitness = -np.inf
            new_pop.append(child)

        population = new_pop[:population_size]

    population.sort(key=lambda t: t.fitness, reverse=True)

    results = [
        {"rank": i+1, "name": t.name, "fitness": t.fitness, "parents": t.parents, "gen": t.gen}
        for i, t in enumerate(population[:10])
    ]

    with open(os.path.join(output_dir, "top_transforms.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("TOP 5 TRANSFORMS FOUND:")
    for r in results[:5]:
        print(f"  {r['rank']}. {r['name'][:70]}  (fitness={r['fitness']:.4f})")
    print('='*50)

    return population

if __name__ == "__main__":
    run_evolution(population_size=20, generations=10)