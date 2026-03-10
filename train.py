"""
train.py

Training loop for the KAN-GNN model on longest path prediction.
Supports multi-GPU via DataParallel, WandB logging, and checkpoint saving.

After training, learned activations are extracted and saved for
symbolic regression in the next pipeline stage.
"""

import os
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from graph_utils import build_dataset
from kan_gnn import KANGNN

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "num_graphs":    2000,
    "min_nodes":     5,
    "max_nodes":     15,
    "train_split":   0.8,
    "batch_size":    64,

    # Model
    "hidden_channels": 32,
    "num_layers":      3,
    "num_knots":       12,   # More knots = more expressive activations
    "dropout":         0.1,

    # Training
    "epochs":          200,
    "lr":              3e-4,
    "weight_decay":    1e-5,
    "grad_clip":       1.0,

    # Infra
    "seed":            42,
    "checkpoint_dir":  "checkpoints",
    "use_wandb":       False,
}


# ─────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"Using {n} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n)]}")
        return torch.device("cuda")
    print("No GPU found, using CPU")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    total_mae  = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y)
        mae  = (pred - batch.y).abs().mean()
        total_loss += loss.item() * batch.num_graphs
        total_mae  += mae.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n, total_mae / n


# ─────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────

def train(config: dict):
    set_seed(config["seed"])
    device = get_device()
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # ── WandB ──
    if config["use_wandb"] and WANDB_AVAILABLE:
        wandb.init(project="longest-path-transforms", config=config)

    # ── Data ──
    print("\n[1/4] Building dataset...")
    dataset = build_dataset(
        num_graphs=config["num_graphs"],
        min_nodes=config["min_nodes"],
        max_nodes=config["max_nodes"],
        seed=config["seed"],
    )
    train_data, val_data = train_test_split(
        dataset, train_size=config["train_split"], random_state=config["seed"]
    )
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_data,   batch_size=config["batch_size"], shuffle=False, num_workers=4)
    print(f"   Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Model ──
    print("\n[2/4] Building model...")
    model = KANGNN(
        in_channels=3,
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        num_knots=config["num_knots"],
        dropout=config["dropout"],
    )

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    criterion = nn.HuberLoss(delta=0.1)  # Robust to outlier graphs

    # ── Training ──
    print("\n[3/4] Training...")
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0
        n = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs

        scheduler.step()
        train_loss = total_loss / n
        val_loss, val_mae = evaluate(model, val_loader, device, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
        })

        if config["use_wandb"] and WANDB_AVAILABLE:
            wandb.log(history[-1])

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            core_model = model.module if hasattr(model, "module") else model
            torch.save(core_model.state_dict(),
                       os.path.join(config["checkpoint_dir"], "best_model.pt"))

    # ── Extract activations ──
    print("\n[4/4] Extracting learned activations...")
    core_model = model.module if hasattr(model, "module") else model
    core_model.load_state_dict(
        torch.load(os.path.join(config["checkpoint_dir"], "best_model.pt"), map_location=device)
    )
    activations = core_model.extract_activations()

    with open(os.path.join(config["checkpoint_dir"], "activations.pkl"), "wb") as f:
        pickle.dump(activations, f)

    with open(os.path.join(config["checkpoint_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"   Saved {len(activations)} activation functions to checkpoints/activations.pkl")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print("\nDone. Run symbolic_regression.py next.")
    return activations


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--num_graphs",      type=int,   default=DEFAULT_CONFIG["num_graphs"])
    parser.add_argument("--hidden_channels", type=int,   default=DEFAULT_CONFIG["hidden_channels"])
    parser.add_argument("--num_knots",       type=int,   default=DEFAULT_CONFIG["num_knots"])
    parser.add_argument("--use_wandb",       action="store_true")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    train(config)
