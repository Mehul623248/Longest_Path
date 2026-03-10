"""
symbolic_regression.py

Phase 2: Take the learned activation functions from the KAN-GNN and
use PySR (symbolic regression) to find closed-form mathematical expressions.

This is the "distillation" step — converting neural network weights into
human-readable mathematics that can be analyzed and proved about.

PySR searches over expressions like:
  x^2, sin(x), log(|x|+1), x*exp(-x), ...
and finds the one that best fits the learned activation curve.
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("WARNING: PySR not installed. Run: pip install pysr")

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# ─────────────────────────────────────────
# PySR configuration
# ─────────────────────────────────────────

def make_pysr_model(
    complexity: int = 12,
    iterations: int = 50,
    populations: int = 20,
) -> "PySRRegressor":
    """
    Configure PySR for activation function discovery.
    
    The operator set includes both standard and exotic operations —
    we want to find genuinely new non-linearities, not just rediscover
    known ones.
    """
    return PySRRegressor(
        niterations=iterations,
        populations=populations,
        maxsize=complexity,

        # Standard operators
        binary_operators=["+", "*", "-", "/", "^"],

        # Non-linear unary operators — this is where new transforms emerge
        unary_operators=[
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
            "abs",
            "tanh",
            "sigmoid(x) = 1 / (1 + exp(-x))",          # Standard sigmoid
            "softplus(x) = log(1 + exp(x))",            # Smooth ReLU
            "mish(x) = x * tanh(log(1 + exp(x)))",      # Mish activation
            "inv(x) = 1 / x",
        ],

        # Parsimony: prefer simpler expressions
        parsimony=0.001,

        # Deterministic
        random_state=42,

        # Output options
        verbosity=0,
        progress=True,
        model_selection="best",
    )


# ─────────────────────────────────────────
# Activation analysis
# ─────────────────────────────────────────

def analyze_activation(
    name: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    model: "PySRRegressor",
    output_dir: str,
) -> dict:
    """
    Run symbolic regression on a single learned activation function.
    Returns the best symbolic expression found.
    """
    # Reshape for PySR
    X = x_vals.reshape(-1, 1)
    y = y_vals.reshape(-1)

    # Filter out NaN/inf
    mask = np.isfinite(X[:, 0]) & np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(X) < 10:
        return {"name": name, "expression": "insufficient_data", "r2": 0.0}

    # Fit
    model.fit(X, y)

    # Get best equation
    best_eq = model.sympy()
    r2 = float(model.score(X, y))

    result = {
        "name":       name,
        "expression": str(best_eq),
        "r2":         r2,
        "complexity": int(model.get_best()["complexity"]),
        "all_equations": [
            {
                "expr":       str(eq.sympy_format),
                "complexity": int(eq.complexity),
                "loss":       float(eq.loss),
            }
            for eq in model.equations_.itertuples()
        ] if hasattr(model, "equations_") else [],
    }

    # Plot
    _plot_activation(name, x_vals, y_vals, model, output_dir)

    return result


def _plot_activation(name, x_vals, y_vals, model, output_dir):
    """Plot learned vs symbolic approximation."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: learned activation
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='Learned')
    ax1.set_title(f'Learned: {name}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: symbolic approximation vs learned
    try:
        X = x_vals.reshape(-1, 1)
        y_pred = model.predict(X)
        ax2.plot(x_vals, y_vals,  'b-',  linewidth=2, label='Learned',  alpha=0.7)
        ax2.plot(x_vals, y_pred,  'r--', linewidth=2, label='Symbolic', alpha=0.7)
        ax2.set_title(f'Symbolic: {model.sympy()}')
        ax2.set_xlabel('x')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    except Exception as e:
        ax2.set_title(f'Symbolic fit failed: {e}')

    plt.tight_layout()
    safe_name = name.replace("/", "_")
    plt.savefig(os.path.join(output_dir, f"activation_{safe_name}.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────
# Novelty detector
# ─────────────────────────────────────────

KNOWN_ACTIVATIONS = {
    "relu":      lambda x: np.maximum(0, x),
    "sigmoid":   lambda x: 1 / (1 + np.exp(-x)),
    "tanh":      lambda x: np.tanh(x),
    "silu":      lambda x: x / (1 + np.exp(-x)),
    "gelu":      lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
    "softplus":  lambda x: np.log1p(np.exp(x)),
    "mish":      lambda x: x * np.tanh(np.log1p(np.exp(x))),
    "elu":       lambda x: np.where(x >= 0, x, np.exp(x) - 1),
}

def novelty_score(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    threshold: float = 0.95,
) -> dict:
    """
    Compare learned activation to known activations.
    Returns the best match and a novelty score (1 = completely novel).
    """
    best_r2 = -np.inf
    best_name = "unknown"

    y_norm = y_vals - y_vals.mean()
    y_std = y_vals.std()
    if y_std < 1e-8:
        return {"best_match": "constant", "r2": 1.0, "is_novel": False}

    y_norm = y_norm / y_std

    for name, fn in KNOWN_ACTIVATIONS.items():
        try:
            y_known = fn(x_vals)
            y_known_norm = y_known - y_known.mean()
            std = y_known_norm.std()
            if std < 1e-8:
                continue
            y_known_norm = y_known_norm / std

            # R² between normalized curves
            ss_res = np.sum((y_norm - y_known_norm) ** 2)
            ss_tot = np.sum(y_norm ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

            if r2 > best_r2:
                best_r2 = r2
                best_name = name
        except Exception:
            continue

    is_novel = best_r2 < threshold
    return {
        "best_match": best_name,
        "r2":         float(best_r2),
        "is_novel":   is_novel,
    }


# ─────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────

def run_symbolic_regression(
    activations_path: str = "checkpoints/activations.pkl",
    output_dir: str = "symbolic_results",
    max_activations: int = 20,  # Analyze top N most "interesting" ones
    pysr_iterations: int = 50,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load activations
    print(f"Loading activations from {activations_path}...")
    with open(activations_path, "rb") as f:
        activations = pickle.load(f)
    print(f"Found {len(activations)} activation functions")

    # Score novelty for all activations first (fast)
    print("\nScoring novelty...")
    novelty_scores = {}
    for name, (x_vals, y_vals) in activations.items():
        novelty_scores[name] = novelty_score(x_vals, y_vals)

    # Sort by novelty (most novel first)
    sorted_acts = sorted(
        activations.items(),
        key=lambda kv: novelty_scores[kv[0]]["r2"]  # lower r2 = more novel
    )

    # Print novelty summary
    print("\nNovelty ranking (most novel first):")
    for name, _ in sorted_acts[:10]:
        ns = novelty_scores[name]
        flag = "🆕 NOVEL" if ns["is_novel"] else f"≈ {ns['best_match']}"
        print(f"  {name:30s}  {flag}  (r²={ns['r2']:.3f})")

    # Run symbolic regression on most novel activations
    if not PYSR_AVAILABLE:
        print("\nPySR not available — skipping symbolic regression.")
        print("Install with: pip install pysr")
        results = []
    else:
        print(f"\nRunning symbolic regression on top {max_activations} novel activations...")
        pysr_model = make_pysr_model(iterations=pysr_iterations)
        results = []

        for i, (name, (x_vals, y_vals)) in enumerate(sorted_acts[:max_activations]):
            print(f"\n[{i+1}/{max_activations}] Analyzing: {name}")
            ns = novelty_scores[name]
            print(f"  Novelty: {ns}")

            result = analyze_activation(name, x_vals, y_vals, pysr_model, output_dir)
            result["novelty"] = ns
            results.append(result)

            print(f"  Best expression: {result['expression']}  (R²={result['r2']:.4f})")

    # Save results
    output_path = os.path.join(output_dir, "symbolic_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {"novelty_scores": novelty_scores, "symbolic_results": results},
            f, indent=2, default=str
        )
    print(f"\nResults saved to {output_path}")

    # Summary of novel expressions found
    novel = [r for r in results if r.get("novelty", {}).get("is_novel", False)]
    print(f"\n{'='*50}")
    print(f"NOVEL ACTIVATIONS FOUND: {len(novel)}")
    for r in novel:
        print(f"  {r['name']}: {r['expression']}  (R²={r['r2']:.4f})")
    print('='*50)

    return results


if __name__ == "__main__":
    run_symbolic_regression()
