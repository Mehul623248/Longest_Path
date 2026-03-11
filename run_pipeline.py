import os
import json
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    seed_expressions = []
    if symbolic_results:
        for r in symbolic_results:
            expr_str = r.get("expression", "")
            if expr_str and expr_str != "insufficient_data":
                try:
                    import sympy as sp
                    x, x0, x1 = sp.symbols('x x0 x1')
                    expr = sp.sympify(expr_str)
                    
                    free_syms = expr.free_symbols
                    if x0 in free_syms or x1 in free_syms:
                        fn_np = sp.lambdify((x0, x1), expr, modules="numpy")
                        def make_fn(f):
                            def torch_fn(t0, t1=None):
                                if t1 is None: t1 = t0 
                                arr0 = t0.detach().cpu().numpy()
                                arr1 = t1.detach().cpu().numpy()
                                out = f(arr0, arr1)
                                return torch.tensor(out, dtype=t0.dtype, device=t0.device)
                            return torch_fn
                        seed_expressions.append((f"sr_{r['name']}", make_fn(fn_np)))
                    else:
                        fn_np = sp.lambdify(x, expr, modules="numpy")
                        def make_fn(f):
                            def torch_fn(t0, t1=None):
                                arr = t0.detach().cpu().numpy()
                                out = f(arr)
                                return torch.tensor(out, dtype=t0.dtype, device=t0.device)
                            return torch_fn
                        seed_expressions.append((f"sr_{r['name']}", make_fn(fn_np)))
                except Exception as e:
                    print(f"Skipping {r['name']} due to parser error: {e}")
                    pass

    population = run_evolution(
        population_size=args.pop_size,
        generations=args.generations,
        eval_epochs=args.eval_epochs,
        seed_expressions=seed_expressions,
        output_dir=args.evolve_dir,
    )
    return population

def generate_report(args):
    report = {"stages": {}}

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

    evo_path = os.path.join(args.evolve_dir, "top_transforms.json")
    if os.path.exists(evo_path):
        with open(evo_path) as f:
            top = json.load(f)
        report["stages"]["evolution"] = {
            "top_transforms": top[:5]
        }

    report_path = "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print("PIPELINE REPORT")
    print("="*60)
    print(f"\nFull report saved to: {report_path}")
    return report

def parse_args():
    p = argparse.ArgumentParser(description="Transform Discovery Pipeline")
    p.add_argument("--stage", choices=["train", "symbolic", "evolve", "all"], default="all")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--symbolic_dir",   default="symbolic_results")
    p.add_argument("--evolve_dir",     default="evolution_results")
    p.add_argument("--epochs",      type=int, default=150)
    p.add_argument("--num_graphs",  type=int, default=2000)
    p.add_argument("--hidden",      type=int, default=32)
    p.add_argument("--knots",       type=int, default=12)
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--max_activations", type=int, default=15)
    p.add_argument("--pysr_iters",      type=int, default=40)
    p.add_argument("--pop_size",    type=int, default=25)
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--eval_epochs", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    print("Transform Discovery Pipeline")
    print(f"Stage: {args.stage}")
    
    symbolic_results = None

    if args.stage in ("train", "all"):
        stage_train(args)

    if args.stage in ("symbolic", "all"):
        symbolic_results = stage_symbolic(args)
    elif args.stage == "evolve":
        # --- NEW: LOAD JSON MANUALLY SO IT FINDS THE 2D EQUATION ---
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