"""
run_pipeline.py

Master script: runs the full transform discovery pipeline end-to-end.

Stages:
  1. Train KAN-GNN         → learns non-linearities from data
  2. Symbolic regression   → distills activations to math expressions
  3. Evolutionary search   → mutates/combines to find better transforms
  4. Report generation     → summarizes what was found

Usage:
  python run_pipeline.py                    # Full pipeline
  python run_pipeline.py --stage train      # Just train
  python run_pipeline.py --stage symbolic   # Just symbolic regression
  python run_pipeline.py --stage evolve     # Just evolution
"""

import os
import json
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────

def stage_train(args):
    from train import train, DEFAULT_CONFIG
    config = {
        **DEFAULT_CONFIG,
        "epochs":          args.epochs,
        "num_graphs":      args.num_graphs,
        "hidden_channels": args.hidden,
        "num_knots":       args.knots,
        "use_wandb":       args.wandb,
    }
    print("\n" + "="*60)
    print("STAGE 1: Training KAN-GNN")
    print("="*60)
    activations = train(config)
    return activations


def stage_symbolic(args):
    from symbolic_regression import run_symbolic_regression
    print("\n" + "="*60)
    print("STAGE 2: Symbolic Regression")
    print("="*60)
    results = run_symbolic_regression(
        activations_path=os.path.join(args.checkpoint_dir, "activations.pkl"),
        output_dir=args.symbolic_dir,
        max_activations=args.max_activations,
        pysr_iterations=args.pysr_iters,
    )
    return results


def stage_evolve(args, symbolic_results=None):
    from evolutionary_search import run_evolution, PRIMITIVE_DICT
    print("\n" + "="*60)
    print("STAGE 3: Evolutionary Search")
    print("="*60)

    # Seed evolution with symbolic results if available
    seed_expressions = []
    if symbolic_results:
        for r in symbolic_results:
            expr_str = r.get("expression", "")
            if expr_str and expr_str != "insufficient_data":
                try:
                    # Try to compile symbolic expression to a torch function
                    import sympy as sp
                    x = sp.Symbol('x')
                    expr = sp.sympify(expr_str)
                    # Convert to numpy lambda, then wrap in torch
                    fn_np = sp.lambdify(x, expr, modules="numpy")
                    def make_fn(f):
                        def torch_fn(t):
                            arr = t.detach().cpu().numpy()
                            out = f(arr)
                            return torch.tensor(out, dtype=t.dtype, device=t.device)
                        return torch_fn
                    seed_expressions.append((f"sr_{r['name']}", make_fn(fn_np)))
                except Exception:
                    pass

    population = run_evolution(
        population_size=args.pop_size,
        generations=args.generations,
        eval_epochs=args.eval_epochs,
        seed_expressions=seed_expressions,
        output_dir=args.evolve_dir,
    )
    return population


# ─────────────────────────────────────────
# Report
# ─────────────────────────────────────────

def generate_report(args):
    """Collect results from all stages and generate a summary report."""
    report = {"stages": {}}

    # Training history
    history_path = os.path.join(args.checkpoint_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        best_epoch = min(history, key=lambda h: h["val_loss"])
        report["stages"]["training"] = {
            "best_epoch": best_epoch["epoch"],
            "best_val_loss": best_epoch["val_loss"],
            "best_val_mae": best_epoch["val_mae"],
        }

    # Symbolic results
    sym_path = os.path.join(args.symbolic_dir, "symbolic_results.json")
    if os.path.exists(sym_path):
        with open(sym_path) as f:
            sym_data = json.load(f)
        novel = [
            r for r in sym_data.get("symbolic_results", [])
            if r.get("novelty", {}).get("is_novel", False)
        ]
        report["stages"]["symbolic_regression"] = {
            "novel_activations_found": len(novel),
            "novel_expressions": [
                {"name": r["name"], "expression": r["expression"], "r2": r["r2"]}
                for r in novel
            ],
        }

    # Evolution results
    evo_path = os.path.join(args.evolve_dir, "top_transforms.json")
    if os.path.exists(evo_path):
        with open(evo_path) as f:
            top = json.load(f)
        report["stages"]["evolution"] = {
            "top_transforms": top[:5]
        }

    # Save report
    report_path = "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print
    print("\n" + "="*60)
    print("PIPELINE REPORT")
    print("="*60)

    if "training" in report["stages"]:
        t = report["stages"]["training"]
        print(f"\nTraining:")
        print(f"  Best epoch: {t['best_epoch']}")
        print(f"  Val loss:   {t['best_val_loss']:.4f}")
        print(f"  Val MAE:    {t['best_val_mae']:.4f}")

    if "symbolic_regression" in report["stages"]:
        s = report["stages"]["symbolic_regression"]
        print(f"\nSymbolic Regression:")
        print(f"  Novel activations found: {s['novel_activations_found']}")
        for e in s.get("novel_expressions", []):
            print(f"  - {e['name']}: {e['expression']}  (R²={e['r2']:.3f})")

    if "evolution" in report["stages"]:
        ev = report["stages"]["evolution"]
        print(f"\nEvolution - Top transforms:")
        for t in ev.get("top_transforms", [])[:3]:
            print(f"  {t['rank']}. {t['name'][:70]}  fitness={t['fitness']:.4f}")

    print(f"\nFull report saved to: {report_path}")
    return report


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Transform Discovery Pipeline")

    p.add_argument("--stage", choices=["train", "symbolic", "evolve", "all"],
                   default="all", help="Which stage to run")

    # Directories
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--symbolic_dir",   default="symbolic_results")
    p.add_argument("--evolve_dir",     default="evolution_results")

    # Training
    p.add_argument("--epochs",      type=int, default=150)
    p.add_argument("--num_graphs",  type=int, default=2000)
    p.add_argument("--hidden",      type=int, default=32)
    p.add_argument("--knots",       type=int, default=12)
    p.add_argument("--wandb",       action="store_true")

    # Symbolic regression
    p.add_argument("--max_activations", type=int, default=15)
    p.add_argument("--pysr_iters",      type=int, default=40)

    # Evolution
    p.add_argument("--pop_size",    type=int, default=25)
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--eval_epochs", type=int, default=20)

    return p.parse_args()


def main():
    args = parse_args()

    print("Transform Discovery Pipeline")
    print(f"Stage: {args.stage}")
    print(f"GPUs available: {torch.cuda.device_count()}")

    symbolic_results = None

    if args.stage in ("train", "all"):
        stage_train(args)

    if args.stage in ("symbolic", "all"):
        symbolic_results = stage_symbolic(args)
    elif args.stage == "evolve":
        # --- NEW: Load the JSON manually if skipping the symbolic stage ---
        sym_path = os.path.join(args.symbolic_dir, "symbolic_results.json")
        if os.path.exists(sym_path):
            with open(sym_path, "r") as f:
                sym_data = json.load(f)
                symbolic_results = sym_data.get("symbolic_results", [])
                print(f"Loaded {len(symbolic_results)} expressions from previous symbolic run.")

    if args.stage in ("evolve", "all"):
        stage_evolve(args, symbolic_results)

    generate_report(args)


if __name__ == "__main__":
    main()
