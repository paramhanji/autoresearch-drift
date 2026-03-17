"""
Fixed data preparation and evaluation for drifting model experiments.
DO NOT MODIFY — this is the fixed evaluation harness.

The drifting model is a generator f_θ(ε) that maps noise ε to 2D samples.
Evaluation measures how well generated samples match a target 2D distribution
using Maximum Mean Discrepancy (MMD).

Usage:
    python prepare.py   # verify setup (no data download needed)

Constants and evaluation are imported by train.py.
"""

import math
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 120         # training time budget in seconds (2 minutes)
EVAL_SAMPLES = 5000       # number of samples for evaluation
EVAL_SEED = 42            # seed for reference data in evaluation
EVAL_NOISE_SEED = 12345   # seed for model noise in evaluation
OUTPUT_DIM = 2            # dimensionality of generated samples
DATASET = "mog"            # fixed dataset
DRIFT_TYPE = "laplacian"   # fixed drift type

# Available datasets
DATASETS = ["swiss_roll", "checkerboard", "mog"]

# ---------------------------------------------------------------------------
# Toy Dataset Samplers
# ---------------------------------------------------------------------------

def sample_checkerboard(n, noise=0.05, seed=None):
    """4 diagonal squares in [-1, 1]^2."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    b = torch.randint(0, 2, (n,), generator=g)
    i = torch.randint(0, 2, (n,), generator=g) * 2 + b
    j = torch.randint(0, 2, (n,), generator=g) * 2 + b
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    pts = pts / 2.0
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_swiss_roll(n, noise=0.03, seed=None):
    """Spiral pattern normalized to [-1, 1]^2."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    u = torch.rand(n, generator=g)
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    pts = pts / (pts.abs().max() + 1e-8)
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_mog(n, num_components=8, std=0.08, radius=1.0, seed=None):
    """Mixture of Gaussians on a circle."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    angles = torch.linspace(0, 2 * math.pi, num_components + 1)[:-1]
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    idx = torch.randint(0, num_components, (n,), generator=g)
    pts = centers[idx] + std * torch.randn(n, 2, generator=g)
    return pts


def get_sampler(dataset_name):
    """Return sampler function for the given dataset name."""
    samplers = {
        "checkerboard": sample_checkerboard,
        "swiss_roll": sample_swiss_roll,
        "mog": sample_mog,
    }
    if dataset_name not in samplers:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(samplers.keys())}")
    return samplers[dataset_name]


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mmd(model, noise_dim, dataset_name, device):
    """
    Maximum Mean Discrepancy (MMD^2) evaluation.

    Generates EVAL_SAMPLES from the model and compares against a fixed
    reference set from the target distribution. Uses sum of Gaussian
    kernels across multiple bandwidths for a robust distribution distance.

    Returns MMD^2 (lower is better). Deterministic given a model.
    """
    model.eval()
    sampler = get_sampler(dataset_name)

    # Fixed reference data
    ref = sampler(EVAL_SAMPLES, seed=EVAL_SEED).to(device).float()

    # Generate model samples with fixed noise
    torch.manual_seed(EVAL_NOISE_SEED)
    noise = torch.randn(EVAL_SAMPLES, noise_dim, device=device)
    batch_size = min(EVAL_SAMPLES, 2048)
    gen_parts = []
    for i in range(0, EVAL_SAMPLES, batch_size):
        gen_parts.append(model(noise[i:i + batch_size]).float())
    gen = torch.cat(gen_parts, dim=0)

    # Compute pairwise squared distances once
    dxx = torch.cdist(ref, ref).pow(2)
    dyy = torch.cdist(gen, gen).pow(2)
    dxy = torch.cdist(ref, gen).pow(2)

    # Sum Gaussian kernels across multiple bandwidths
    sigmas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    mmd_sq = 0.0
    for s in sigmas:
        inv_s2 = 1.0 / (2.0 * s * s)
        kxx = (-dxx * inv_s2).exp().mean().item()
        kyy = (-dyy * inv_s2).exp().mean().item()
        kxy = (-dxy * inv_s2).exp().mean().item()
        mmd_sq += kxx + kyy - 2.0 * kxy

    return mmd_sq


# ---------------------------------------------------------------------------
# Main (verification only — no data to download for toy distributions)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Drifting model experiment setup")
    print(f"  TIME_BUDGET:  {TIME_BUDGET}s")
    print(f"  EVAL_SAMPLES: {EVAL_SAMPLES}")
    print(f"  OUTPUT_DIM:   {OUTPUT_DIM}")
    print(f"  DATASETS:     {DATASETS}")
    print()

    for name in DATASETS:
        sampler = get_sampler(name)
        pts = sampler(1000, seed=42)
        print(f"  {name:15s}: shape={tuple(pts.shape)}, "
              f"range=[{pts.min():.2f}, {pts.max():.2f}]")

    print()
    print("Ready to train. Run: uv run train.py")
