# Transform Discovery Pipeline

A research pipeline for discovering novel non-linear transforms that may reveal
hidden structure in the longest path problem. Inspired by how Shor's algorithm
found that the Quantum Fourier Transform was the "right" operation for factorization.

## Architecture

```
Phase 1: KAN-GNN Training
  ↓ Learns non-linearities from graph data using B-spline activations
  
Phase 2: Symbolic Regression (PySR)
  ↓ Distills learned activations into closed-form math expressions
  
Phase 3: Evolutionary Search
  ↓ Mutates and combines discovered expressions to search the transform space
  
Phase 4: Report
  → Summarizes novel transforms found for mathematical analysis
```

## Setup

```bash
pip install -r requirements.txt

# PySR requires Julia (for the symbolic regression engine)
python -c "import pysr; pysr.install()"
```

## Usage

```bash
# Full pipeline (recommended first run)
python run_pipeline.py

# Individual stages
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

## Key Files

| File | Purpose |
|---|---|
| `graph_utils.py` | Graph generation + ground truth longest path solver |
| `kan_gnn.py` | KAN-GNN model with learnable B-spline activations |
| `train.py` | Training loop, multi-GPU support, activation extraction |
| `symbolic_regression.py` | PySR integration + novelty scoring |
| `evolutionary_search.py` | Genetic operators + fitness evaluation |
| `run_pipeline.py` | Master orchestration script |

## What to look for in results

The pipeline outputs `pipeline_report.json` and a `symbolic_results/` directory.

**Interesting signals:**
- Activations with novelty R² < 0.5 (doesn't match any known activation)
- Symbolic expressions that are clean and interpretable (not just noise)
- Evolution finding the same composite transform repeatedly across runs
- Transforms that generalize — trained on small graphs, tested on larger ones

**What would be groundbreaking:**
If the symbolic regression consistently converges on the same unknown expression
across different random seeds and graph types, that's a strong signal there's
real mathematical structure being discovered — worth deeper analysis.

## Theoretical context

This pipeline operationalizes the hypothesis that there exists some transform T such that:

```
T(graph structure) → low-dimensional space
                   where longest path is "visible"
```

Analogous to how the Quantum Fourier Transform collapses factorization's 
exponential search space by revealing periodic structure.

This pipeline cannot prove P=NP. But it can:
1. Find better practical heuristics for typical graph instances
2. Reveal mathematical structure worth investigating theoretically  
3. Narrow the search space for mathematical insight

## Multi-GPU notes

Training automatically uses `nn.DataParallel` across all visible GPUs.
For distributed training across machines, modify `train.py` to use
`torch.distributed` with `DistributedDataParallel`.

Set visible GPUs with:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_pipeline.py
```
