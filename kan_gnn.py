"""
kan_gnn.py

KAN-GNN: Graph Neural Network where activation functions are learnable splines
(Kolmogorov-Arnold Network style). Instead of fixed ReLU/sigmoid, each
"neuron" learns its own 1D transform from data.

The learned activations are later extracted and fed to symbolic regression
to find closed-form mathematical expressions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np


# ─────────────────────────────────────────
# Learnable spline activation (KAN-style)
# ─────────────────────────────────────────

class SplineActivation(nn.Module):
    """
    Learnable 1D activation function parameterized as a B-spline.
    Each instance learns its own non-linearity from data.
    
    Architecture: residual form  f(x) = base(x) + spline(x)
    where base(x) = SiLU(x) (stable initialization)
    and   spline(x) = sum of B-spline basis functions with learned coefficients
    """

    def __init__(
        self,
        num_knots: int = 8,
        grid_range: tuple = (-3.0, 3.0),
        spline_order: int = 3,
    ):
        super().__init__()
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Learnable spline coefficients
        self.coefficients = nn.Parameter(
            torch.zeros(num_knots + spline_order - 1)
        )
        # Learnable blend weight between base and spline
        self.blend = nn.Parameter(torch.ones(1) * 0.1)

        # Fixed uniform grid
        grid = torch.linspace(grid_range[0], grid_range[1], num_knots + 2 * spline_order)
        self.register_buffer("grid", grid)

    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions via Cox-de Boor recursion."""
        x = x.unsqueeze(-1)  # (..., 1)
        grid = self.grid  # (G,)

        # Order 1: indicator functions
        basis = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Recurse up to desired order
        for k in range(1, self.spline_order):
            left_num  = x - grid[:-(k+1)]
            left_den  = grid[k:-1] - grid[:-(k+1)]
            right_num = grid[(k+1):] - x
            right_den = grid[(k+1):] - grid[1:-k]

            # Safe division
            left  = torch.where(left_den.abs() > 1e-8,
                                left_num / left_den, torch.zeros_like(left_num))
            right = torch.where(right_den.abs() > 1e-8,
                                right_num / right_den, torch.zeros_like(right_num))

            basis = left * basis[..., :-1] + right * basis[..., 1:]

        return basis  # (..., num_coefficients)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1)

        # Clamp to grid range for numerical stability
        x_clamped = x_flat.clamp(*self.grid_range)

        # Spline component
        basis = self.b_spline_basis(x_clamped)  # (N, num_coefficients)
        spline_out = basis @ self.coefficients    # (N,)

        # Base component (SiLU is smooth and well-behaved as initialization)
        base_out = F.silu(x_flat)

        # Blend: mostly base at init, spline learns corrections
        out = base_out + self.blend.abs() * spline_out
        return out.reshape(original_shape)

    def get_learned_values(self, num_points: int = 200) -> tuple:
        """Sample the learned activation for visualization / symbolic regression."""
        with torch.no_grad():
            x = torch.linspace(*self.grid_range, num_points).to(self.coefficients.device)
            y = self.forward(x)
        return x.cpu().numpy(), y.cpu().numpy()


# ─────────────────────────────────────────
# KAN Message Passing Layer
# ─────────────────────────────────────────

class KANConv(MessagePassing):
    """
    Graph convolution layer where the message, update, and aggregation
    transforms all use learnable SplineActivations.
    """

    def __init__(self, in_channels: int, out_channels: int, num_knots: int = 8):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear projections
        self.lin_msg  = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_out  = nn.Linear(out_channels, out_channels, bias=False)

        # Learnable activations (one per output channel — shared across nodes)
        self.act_msg  = nn.ModuleList([SplineActivation(num_knots) for _ in range(out_channels)])
        self.act_upd  = nn.ModuleList([SplineActivation(num_knots) for _ in range(out_channels)])

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # Add self-loops and compute degree normalization
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Project neighbor features
        msg = self.lin_msg(x_j)
        # Apply per-channel learnable activations
        msg = self._apply_channel_activations(msg, self.act_msg)
        return norm.view(-1, 1) * msg

    def update(self, aggr_out, x):
        # Combine aggregated messages with self features
        self_feat = self.lin_self(x)
        combined = aggr_out + self_feat
        combined = self._apply_channel_activations(combined, self.act_upd)
        out = self.lin_out(combined)
        return self.norm(out)

    def _apply_channel_activations(self, x: torch.Tensor, acts: nn.ModuleList) -> torch.Tensor:
        """Apply per-channel spline activations."""
        channels = x.unbind(dim=-1)
        activated = [act(ch) for act, ch in zip(acts, channels)]
        return torch.stack(activated, dim=-1)

    def get_all_activations(self) -> dict:
        """Extract learned activation functions for symbolic regression."""
        result = {}
        for i, act in enumerate(self.act_msg):
            x_vals, y_vals = act.get_learned_values()
            result[f"msg_ch{i}"] = (x_vals, y_vals)
        for i, act in enumerate(self.act_upd):
            x_vals, y_vals = act.get_learned_values()
            result[f"upd_ch{i}"] = (x_vals, y_vals)
        return result


# ─────────────────────────────────────────
# Full KAN-GNN Model
# ─────────────────────────────────────────

class KANGNN(nn.Module):
    """
    Multi-layer KAN-GNN for longest path prediction.
    
    Architecture:
        Input node features
        → 3x KANConv layers (message passing with learned activations)
        → Global pooling (mean + max concatenated)
        → MLP readout with learned activations
        → Scalar prediction (normalized longest path length)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_layers: int = 3,
        num_knots: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_act = SplineActivation(num_knots)

        # KAN convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(KANConv(hidden_channels, hidden_channels, num_knots))

        # Readout MLP with learned activations
        readout_in = hidden_channels * 2  # mean + max pooling concatenated
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden_channels),
            _ChannelwiseKAN(hidden_channels, num_knots),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            _ChannelwiseKAN(hidden_channels // 2, num_knots),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1] — normalized path length
        )

    def forward(self, x, edge_index, batch):
        # Input projection
        h = self.input_proj(x)
        h = torch.stack([self.input_act(h[:, i]) for i in range(h.size(1))], dim=1)

        # Message passing
        for conv in self.convs:
            h = h + conv(h, edge_index)  # Residual connections
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling
        h_mean = global_mean_pool(h, batch)
        h_max  = global_max_pool(h, batch)
        h_global = torch.cat([h_mean, h_max], dim=1)

        return self.readout(h_global).squeeze(-1)

    def extract_activations(self) -> dict:
        """
        Extract all learned activation functions from the model.
        Returns dict of {name: (x_values, y_values)} for symbolic regression.
        """
        activations = {}

        # Input activation
        x_v, y_v = self.input_act.get_learned_values()
        activations["input"] = (x_v, y_v)

        # Conv layer activations
        for layer_idx, conv in enumerate(self.convs):
            for name, (x_v, y_v) in conv.get_all_activations().items():
                activations[f"conv{layer_idx}_{name}"] = (x_v, y_v)

        return activations


class _ChannelwiseKAN(nn.Module):
    """Helper: apply independent SplineActivation to each channel."""
    def __init__(self, channels: int, num_knots: int):
        super().__init__()
        self.acts = nn.ModuleList([SplineActivation(num_knots) for _ in range(channels)])

    def forward(self, x):
        return torch.stack([act(x[:, i]) for i, act in enumerate(self.acts)], dim=1)


# ─────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Batch

    model = KANGNN(in_channels=3, hidden_channels=16, num_layers=2, num_knots=6)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fake batch of 2 graphs
    x = torch.randn(10, 3)
    edge_index = torch.randint(0, 10, (2, 20))
    batch = torch.tensor([0]*5 + [1]*5)

    out = model(x, edge_index, batch)
    print(f"Output shape: {out.shape}, values: {out.detach()}")

    acts = model.extract_activations()
    print(f"Extracted {len(acts)} activation functions")
