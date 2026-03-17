"""
Drifting model training script. Single-GPU/CPU, single-file.

Generative modeling via drifting with local Laplacian drift fields.
The generator f_θ maps noise ε to 2D samples. Training "drifts" generated
samples toward the data distribution using a local Laplacian vector field.

At inference: just one forward pass (1 NFE).

Usage: uv run train.py
"""

import gc
import math
import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, OUTPUT_DIM, DATASET, DRIFT_TYPE, get_sampler, evaluate_mmd

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Generator architecture
NOISE_DIM = 32            # input noise dimensionality
HIDDEN_DIM = 256          # hidden layer width
NUM_HIDDEN = 3            # number of hidden layers

# Drift computation
TEMP = 0.05               # temperature for kernel
LB_K = 16                 # k-nearest neighbors for Laplacian
LB_ALPHA = 1.0            # repulsion weight (V = V_pos - alpha * V_neg)

# Training
DATA_BATCH_SIZE = 2048    # number of target samples per step
GEN_BATCH_SIZE = 2048     # number of generated samples per step
LR = 1e-3                 # learning rate
SEED = 42                 # random seed

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP generator: noise -> 2D output."""
    def __init__(self, in_dim=NOISE_DIM, hidden=HIDDEN_DIM, num_hidden=NUM_HIDDEN,
                 out_dim=OUTPUT_DIM):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

# ---------------------------------------------------------------------------
# Drift computation: Local Laplacian
# ---------------------------------------------------------------------------

def laplace_beltrami(gen, k=16, temp=0.05):
    """
    Discrete Laplace-Beltrami on generated samples (self-repulsion term).
    For each point, find k-NN among other generated points and compute
    weighted mean displacement.
    """
    G = gen.shape[0]
    if G <= 1:
        return torch.zeros_like(gen)
    k = min(k, G - 1)
    if k <= 0:
        return torch.zeros_like(gen)
    dist_sq = torch.cdist(gen, gen).pow(2)
    dist_sq.fill_diagonal_(float("inf"))
    knn_dist_sq, knn_idx = dist_sq.topk(k, dim=-1, largest=False)
    weights = (-knn_dist_sq / (4.0 * temp * temp)).exp()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    neighbors = gen[knn_idx]
    diff = neighbors - gen.unsqueeze(1)
    return (weights.unsqueeze(-1) * diff).sum(dim=1)


def laplace_beltrami_interpolated(gen, pos, k=16, temp=0.05):
    """
    Laplace-Beltrami from real data, interpolated at generated points (attraction).
    For each generated point, find k-NN in real data and compute weighted
    mean displacement toward data.
    """
    G = gen.shape[0]
    P = pos.shape[0]
    if G == 0 or P == 0:
        return torch.zeros_like(gen)
    k = min(k, P)
    if k <= 0:
        return torch.zeros_like(gen)
    dist_sq = torch.cdist(gen, pos).pow(2)
    knn_dist_sq, knn_idx = dist_sq.topk(k, dim=-1, largest=False)
    weights = (-knn_dist_sq / (4.0 * temp * temp)).exp()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    neighbors = pos[knn_idx]
    diff = neighbors - gen.unsqueeze(1)
    return (weights.unsqueeze(-1) * diff).sum(dim=1)


def compute_drift_laplacian(gen, pos, temp=0.05, lb_k=16, lb_alpha=1.0):
    """
    Drift field V = V_pos - lb_alpha * V_neg.
    - Positive term (attraction): Laplacian from real data at generated points.
    - Negative term (repulsion): Laplacian on generated samples only.
    """
    pos_V = laplace_beltrami_interpolated(gen, pos, k=lb_k, temp=temp)
    neg_V = laplace_beltrami(gen, k=lb_k, temp=temp)
    return pos_V - lb_alpha * neg_V

# ---------------------------------------------------------------------------
# Drift computation: Mean-shift (attention-based kernel)
# ---------------------------------------------------------------------------

def compute_drift_mean_shift(gen, pos, temp=0.05, **kwargs):
    """
    Original mean-shift drift field with attention-based kernel.
    V(x) = V_p^+(x) - V_q^-(x) with batch-normalized exponential kernel.
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp()

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def drifting_loss(gen, pos, compute_drift_fn):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift_fn(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

sampler = get_sampler(DATASET)
print(f"Dataset: {DATASET}")

model = MLP(in_dim=NOISE_DIM, hidden=HIDDEN_DIM, num_hidden=NUM_HIDDEN,
            out_dim=OUTPUT_DIM).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Select drift function
if DRIFT_TYPE == "laplacian":
    drift_fn = partial(compute_drift_laplacian, temp=TEMP, lb_k=LB_K, lb_alpha=LB_ALPHA)
elif DRIFT_TYPE == "mean_shift":
    drift_fn = partial(compute_drift_mean_shift, temp=TEMP)
else:
    raise ValueError(f"Unknown DRIFT_TYPE: {DRIFT_TYPE}")

print(f"Drift type: {DRIFT_TYPE}")
print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
smooth_loss = None

while True:
    t0 = time.time()

    pos = sampler(DATA_BATCH_SIZE).to(device)
    noise = torch.randn(GEN_BATCH_SIZE, NOISE_DIM, device=device)
    gen = model(noise)
    loss = drifting_loss(gen, pos, drift_fn)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0

    loss_val = loss.item()

    # Fast fail
    if math.isnan(loss_val) or loss_val > 1e6:
        print("\nFAIL: loss exploded")
        exit(1)

    if step > 5:
        total_training_time += dt

    if smooth_loss is None:
        smooth_loss = loss_val
    else:
        smooth_loss = 0.96 * smooth_loss + 0.04 * loss_val

    if step % 100 == 0 or step < 5:
        remaining = max(0, TIME_BUDGET - total_training_time)
        pct_done = 100.0 * min(total_training_time / TIME_BUDGET, 1.0)
        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {smooth_loss:.6e} | "
              f"dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ",
              end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
val_mmd = evaluate_mmd(model, NOISE_DIM, DATASET, device)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

print("---")
print(f"val_mmd:          {val_mmd:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params}")
print(f"dataset:          {DATASET}")
print(f"drift_type:       {DRIFT_TYPE}")
