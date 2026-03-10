"""
evolutionary_search.py

Phase 3: Evolutionary search over non-linear transforms.

Takes expressions discovered by symbolic regression and:
1. Mutates them (perturb parameters, swap operators)
2. Combines them (compose two activations, take weighted sum)
3. Evaluates fitness (how well does a GNN with this activation solve longest path?)
4. Selects best and repeats

This is the "outer loop" that searches the space of transforms.
Think of it as: symbolic regression finds candidate expressions,
evolution searches the space *between* those candidates.
"""

import os
import copy
import json
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataclasses import dataclass, field
from typing import Callable, Optional
import sympy as sp
from sympy import symbols, lambdify, sin, cos, exp, log, tanh, sqrt, Abs

from graph_utils import build_dataset
from kan_gnn import KANGNN
from train import evaluate


# ─────────────────────────────────────────
# Transform representation
# ─────────────────────────────────────────

x_sym = symbols('x')

# Primitive building blocks — these get composed and mutated
PRIMITIVES = [
    ("relu",      lambda x: torch.relu(x)),
    ("sigmoid",   lambda x: torch.sigmoid(x)),
    ("tanh",      lambda x: torch.tanh(x)),
    ("silu",      lambda x: x * torch.sigmoid(x)),
    ("sin",       lambda x: torch.sin(x)),
    ("cos",       lambda x: torch.cos(x)),
    ("exp_neg",   lambda x: torch.exp(-x.abs())),
    ("softplus",  lambda x: torch.log1p(torch.exp(x))),
    ("mish",      lambda x: x * torch.tanh(torch.log1p(torch.exp(x)))),
    ("square",    lambda x: x ** 2),
    ("cube",      lambda x: x ** 3),
    ("inv_sqrt",  lambda x: x / (1 + x.abs()).sqrt()),
    ("log_abs",   lambda x: torch.log(x.abs() + 1)),
    ("sinc",      lambda x: torch.sinc(x / np.pi)),
    ("gaussian",  lambda x: torch.exp(-x ** 2)),
    ("silu_cube",    lambda x: x * torch.sigmoid(x) + 0.5 * x**3),
("poly_sigmoid", lambda x: (x**3) / (1 + torch.exp(-x))),
("sixth_degree", lambda x: x**6 + x**3),
# Run 3 hybrids — directly from symbolic regression findings
    ("sixth_silu",    lambda x: torch.clamp(x**6 + x**3, -10, 10) * torch.sigmoid(x)),
    ("gated_sixth",   lambda x: torch.clamp(x**6, -10, 10) / (1 + torch.exp(-x))),
    ("cube_sixth",    lambda x: torch.clamp(x**3 + 0.5 * x**6, -10, 10)),
    ("adaptive_silu", lambda x: x * torch.sigmoid(x * torch.tanh(torch.exp(x * (x - 0.315)).clamp(-10, 10)))),
]

PRIMITIVE_DICT = {name: fn for name, fn in PRIMITIVES}


@dataclass
class Transform:
    """
    A candidate non-linear transform, represented both as
    a callable (for training) and a name (for tracking).
    """
    name:     str
    fn:       Callable
    fitness:  float = -np.inf
    parents:  list  = field(default_factory=list)
    gen:      int   = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.fn(x)
            # Safety: clamp to prevent explosion
            return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        except Exception:
            return x  # Fall back to identity


# ─────────────────────────────────────────
# Genetic operators
# ─────────────────────────────────────────

def compose(t1: Transform, t2: Transform, blend: float = 0.5) -> Transform:
    """Compose two transforms: f(x) = blend*t1(x) + (1-blend)*t2(x)."""
    def fn(x):
        return blend * t1(x) + (1 - blend) * t2(x)
    return Transform(
        name=f"blend({t1.name},{t2.name},{blend:.2f})",
        fn=fn,
        parents=[t1.name, t2.name],
        gen=max(t1.gen, t2.gen) + 1,
    )


def chain(t1: Transform, t2: Transform) -> Transform:
    """Chain two transforms: f(x) = t2(t1(x))."""
    def fn(x):
        return t2(t1(x))
    return Transform(
        name=f"chain({t1.name},{t2.name})",
        fn=fn,
        parents=[t1.name, t2.name],
        gen=max(t1.gen, t2.gen) + 1,
    )


def perturb(t: Transform, noise_scale: float = 0.1) -> Transform:
    """Add small parametric noise to a transform."""
    alpha = 1.0 + random.gauss(0, noise_scale)
    beta  = random.gauss(0, noise_scale)
    def fn(x):
        return t(alpha * x + beta)
    return Transform(
        name=f"perturb({t.name},{alpha:.2f},{beta:.2f})",
        fn=fn,
        parents=[t.name],
        gen=t.gen + 1,
    )


def residual(t: Transform, scale: float = None) -> Transform:
    """Add residual connection: f(x) = x + scale*t(x)."""
    if scale is None:
        scale = random.uniform(0.1, 1.0)
    def fn(x):
        return x + scale * t(x)
    return Transform(
        name=f"residual({t.name},{scale:.2f})",
        fn=fn,
        parents=[t.name],
        gen=t.gen + 1,
    )


def mutate(t: Transform) -> Transform:
    """Apply a random mutation to a transform."""
    ops = [
        lambda: perturb(t),
        lambda: residual(t),
        lambda: compose(t, random.choice([Transform(n, f) for n, f in PRIMITIVES])),
        lambda: chain(t,  random.choice([Transform(n, f) for n, f in PRIMITIVES])),
    ]
    return random.choice(ops)()


# ─────────────────────────────────────────
# Fitness evaluation
# ─────────────────────────────────────────

class TransformGNN(nn.Module):
    """
    Simplified GNN that uses a fixed (non-learned) transform
    as its activation function. Used for fast fitness evaluation.
    """
    def __init__(self, transform: Transform, hidden: int = 16):
        super().__init__()
        self.transform = transform
        self.lin1 = nn.Linear(3, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden * 2, 1)

    def forward(self, x, edge_index, batch):
        from torch_geometric.nn import global_mean_pool, global_max_pool, SimpleConv
        from torch_geometric.utils import add_self_loops

        h = self.transform(self.lin1(x))

        # Simple mean-aggregation message passing using PyG built-in
        row, col = edge_index
        num_nodes = x.size(0)
        # Aggregate neighbor features via mean pooling per node
        ones = torch.ones(col.size(0), 1, device=x.device)
        deg = torch.zeros(num_nodes, 1, device=x.device).scatter_add_(0, row.unsqueeze(1), ones).clamp(min=1)
        agg = torch.zeros(num_nodes, h.size(1), device=x.device).scatter_add_(0, row.unsqueeze(1).expand(-1, h.size(1)), h[col])
        agg = agg / deg

        h = self.transform(self.lin2(agg))

        h_mean = global_mean_pool(h, batch)
        h_max  = global_max_pool(h, batch)
        return torch.sigmoid(self.lin3(torch.cat([h_mean, h_max], dim=1))).squeeze(-1)


def evaluate_transform(
    transform: Transform,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    epochs:       int = 30,
) -> float:
    """
    Train a small GNN with this transform and return validation loss.
    Lower = better. Returns infinity on failure.
    """
    try:
        model = TransformGNN(transform).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss()

        model.train()
        for _ in range(epochs):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(pred, batch.y)
                if torch.isnan(loss):
                    return float("inf")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        val_loss, _ = evaluate(model, val_loader, device, criterion)
        return val_loss

    except Exception as e:
        return float("inf")


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
    device_str:      str = "cuda",
) -> list:
    """
    Main evolutionary search loop.
    
    seed_expressions: list of (name, callable) tuples from symbolic regression
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    random.seed(42)
    np.random.seed(42)

    # ── Dataset (small for fast eval) ──
    print("Building evaluation dataset...")
    dataset = build_dataset(num_graphs=400, min_nodes=5, max_nodes=12, verbose=False)
    split = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:split], batch_size=32, shuffle=True)
    val_loader   = DataLoader(dataset[split:], batch_size=32)

    # ── Initial population ──
    print(f"\nInitializing population of {population_size}...")
    population = [Transform(name, fn) for name, fn in PRIMITIVES]

    # Seed with expressions from symbolic regression
    if seed_expressions:
        for name, fn in seed_expressions:
            population.append(Transform(f"sr_{name}", fn))

    # Fill up with mutations of primitives
    while len(population) < population_size:
        base = random.choice(population[:len(PRIMITIVES)])
        population.append(mutate(base))

    population = population[:population_size]

    # ── Evolution ──
    history = []
    n_elite = max(1, int(elite_frac * population_size))

    print(f"Evolving for {generations} generations...")
    for gen in range(generations):
        print(f"\nGeneration {gen+1}/{generations}")

        # Evaluate fitness
        for i, t in enumerate(population):
            if t.fitness == -np.inf:  # Only evaluate if not already scored
                t.fitness = -evaluate_transform(t, train_loader, val_loader, device, eval_epochs)
                print(f"  [{i+1}/{len(population)}] {t.name[:50]:50s}  fitness={t.fitness:.4f}")

        # Sort by fitness (higher = better)
        population.sort(key=lambda t: t.fitness, reverse=True)

        best = population[0]
        avg  = np.mean([t.fitness for t in population if t.fitness > -np.inf])
        print(f"  Best: {best.name[:60]}  fitness={best.fitness:.4f}")
        print(f"  Avg fitness: {avg:.4f}")

        history.append({
            "gen":          gen + 1,
            "best_fitness": best.fitness,
            "best_name":    best.name,
            "avg_fitness":  float(avg),
        })

        # Save checkpoint
        with open(os.path.join(output_dir, "evolution_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if gen == generations - 1:
            break

        # ── Selection + reproduction ──
        elites = population[:n_elite]
        new_pop = list(elites)  # Elites survive unchanged

        while len(new_pop) < population_size:
            r = random.random()
            if r < crossover_rate and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                child = compose(p1, p2, blend=random.uniform(0.3, 0.7))
            elif r < crossover_rate + mutation_rate:
                parent = random.choice(elites)
                child = mutate(parent)
            else:
                # New random primitive
                name, fn = random.choice(PRIMITIVES)
                child = Transform(name, fn)
                child.fitness = -np.inf

            # Reset fitness for new individuals
            if child.name not in [t.name for t in elites]:
                child.fitness = -np.inf
            new_pop.append(child)

        population = new_pop[:population_size]

    # ── Final results ──
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
    run_evolution(
        population_size=20,
        generations=10,
        eval_epochs=15,
    )