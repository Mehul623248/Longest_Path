# Neuro-Symbolic Graph Routing

**Evolving Topology-Aware Heuristics via B-Spline Networks and Genetic Algorithms** 
Author: Mehul Murali

## 1. Project Overview

Graph routing algorithms (like Beam Search) traditionally rely on human-engineered heuristics—such as Pure Greedy (weight-only) or Warnsdorff/Survival rules (weight + degree)—to evaluate path quality. This project replaces human intuition with a **Neuro-Symbolic AI Pipeline** that organically learns, extracts, and optimizes mathematical routing heuristics directly from raw topological data.

By training a custom Graph Neural Network (KAN-GNN) on randomized mazes, distilling its knowledge via Symbolic Regression, and fine-tuning the resulting equations using a Genetic Algorithm, this pipeline successfully invented multi-variable heuristics that statistically outperform classical routing methods within their training distribution.

## 2. Theoretical Context

This pipeline operationalizes the hypothesis that there exists some transform $T$ such that:

T(graph structure)→low-dimensional space where longest path is "visible"

This is conceptually inspired by how Shor's algorithm found that the Quantum Fourier Transform was the "right" operation for factorization, collapsing an exponential search space by revealing periodic structure. While this pipeline cannot prove $P=NP$, it can:

1. Find better practical heuristics for typical graph instances.
2. Reveal mathematical structure worth investigating theoretically.
3. Narrow the search space for mathematical insight.

## 3. Architecture & Pipeline

This project operates across a three-stage pipeline to convert black-box neural weights into interpretable, closed-form mathematics:

### Phase 1: Topological Learning (KAN-GNN)
- Constructed a custom Message Passing Neural Network utilizing Kolmogorov-Arnold Network (KAN) B-Splines instead of standard linear layers.
- The model learned to predict the longest path in Erdos-Renyi graphs by independently analyzing Edge Weight and Node Degree.

### Phase 2: Mathematical Distillation (Symbolic Regression)
- Hooked the continuous activation surfaces of the GNN's message-passing layers.
- Utilized **PySR** to distill the multi-dimensional neural weights into explicit mathematical equations.

### Phase 3: Survival Evolution (Genetic Algorithm)
- Injected the PySR equations into a custom Genetic Algorithm.
- Evaluated mutations via a strict, no-lookahead Beam Search proxy on sparse, treacherous graphs, forcing the AI to optimize for actual routing survival rather than proxy loss functions.

## 4. The Evolved Heuristics

By forcing the AI to route through sparse networks where greedy decisions lead directly to dead ends, the Genetic Algorithm discarded linear/additive models and evolved two novel gating mechanisms:

### Equation 1: The Multiplicative Gate (Sparse Graph Champion)

f(Weight, Degree) = (0.90 * Weight - 0.13) * Degree

**Logic:** Dynamically turns low-weight edges into active penalties while completely nullifying heavy weights if they lead to dead ends (Degree= 0).

### Equation 2: The Blended Sigmoid (Dense Graph Champion)

f(Weight, Degree) = 0.52(Weight * Degree) + 0.48*(1/ (1 + e^(-Degree)))

**Logic:** Combines a multiplicative weight gate with a pure topological safety bias, using a sigmoid tie-breaker to favor highly-connected hubs when path weights are ambiguous.

## 5. Evaluation & Statistical Proof

To test out-of-distribution (OOD) generalization, the evolved heuristics were benchmarked against classical baselines (Pure Greedy and Survival) across three distinct 100-node graph topologies. Statistical significance was verified using the **Wilcoxon Signed-Rank Test** (a = 0.05).

### In-Distribution: Random Sparse Networks (Erdos-Renyi p=0.025)

The AI was trained and evolved strictly on Erdos-Renyi physics. In this environment, it became a perfect specialist, achieving a statistically significant blowout against all baselines.

| Method | Score |
|---|---|
| **Eq2 (Blended)** | **153.21** ✓ Winner |
| Eq1 (Mult Gate) | 134.95 |
| Survival Baseline | 97.81 |
| Pure Greedy | 69.44 |

*Significance: Eq1 vs Greedy (p = 0.00000), Eq2 vs Survival (p = 0.00000)*

### Out-of-Distribution: Scale-Free Networks (Barabasi-Albert m=2)

Scale-free networks contain massive "super-hubs." The AI's heavy reliance on Node Degree caused it to immediately gravitate toward these hubs, where it subsequently became trapped by the surrounding low-degree dead ends.

| Method | Score |
|---|---|
| **Pure Greedy** | **108.32** ✓ Winner |
| Eq1 (Mult Gate) | 102.11 |

*Significance: Eq1 vs Greedy (p = 0.428) → Statistical Tie*

### Out-of-Distribution: Small-World Networks (Watts-Strogatz k=4, p=0.1)

Small-world networks consist of tightly clustered, isolated neighborhoods. The AI's routing logic successfully navigated the clusters but did not achieve a statistically significant margin over the survival baseline.

| Method | Score |
|---|---|
| **Survival Baseline** | **193.18** ✓ Winner |
| Eq2 (Blended) | 182.84 |

*Significance: Eq2 vs Survival (p = 0.525) → Statistical Tie*

### Conclusion: The "No Free Lunch" Theorem

The statistical results perfectly demonstrate the *No Free Lunch Theorem* and the limits of OOD generalization. Because the pipeline relies on mechanistic interpretability (outputting explicit math rather than black-box matrices), we can definitively prove *why* the heuristic dominates random mazes and *why* it ties on scale-free super-hubs. It confirms that the neuro-symbolic architecture successfully reverse-engineered the exact mathematical physics of its specific training environment.

## 6. Setup and Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

PySR requires Julia for its symbolic regression engine. Install it via Python:

```bash
python -c "import pysr; pysr.install()"
```

## 7. Usage

Run the master orchestration script:

```bash
# Full pipeline (recommended first run)
python run_pipeline.py

# Run individual stages
python run_pipeline.py --stage train
python run_pipeline.py --stage symbolic
python run_pipeline.py --stage evolve

# Scale up for more thorough search
python run_pipeline.py \
  --epochs 300 \
  --num_graphs 5000 \
  --knots 16 \
  --pop_size 50 \
  --generations 30 \
  --wandb
```

## 8. Repository Structure

| File | Purpose |
|---|---|
| `graph_utils.py` | Graph generation + ground truth longest path solver |
| `kan_gnn.py` | KAN-GNN model with learnable B-spline activations |
| `train.py` | Training loop, multi-GPU support, activation extraction |
| `symbolic_regression.py` | PySR integration + novelty scoring |
| `evolutionary_search.py` | Genetic operators + fitness evaluation |
| `run_pipeline.py` | Master orchestration script |

> The pipeline outputs a `pipeline_report.json` and a `symbolic_results/` directory upon completion.