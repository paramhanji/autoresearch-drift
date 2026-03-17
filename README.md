# autoresearch-drift

Autonomous AI research on **generative drifting models** — a new class of generative models where iteration happens at training time, not inference time. Based on the paper "Generative Modeling via Drifting".

## What is this?

An experiment framework where an AI agent autonomously iterates on a drifting model implementation. The agent modifies `train.py`, runs 2-minute experiments, and keeps or discards changes based on the MMD evaluation metric.

- **One file to modify**: `train.py` — generator architecture, drift computation, optimizer, training loop
- **Fixed evaluation**: `prepare.py` — MMD metric on toy 2D distributions (swiss roll, checkerboard, mixture of Gaussians)
- **Fixed time budget**: 2 minutes per experiment → ~30 experiments per hour
- **One metric**: `val_mmd` (lower is better) — MMD² between generated and target samples

## How drifting models work

Traditional generative models (diffusion, flow matching) require many forward passes at inference time. Drifting models move the iteration to training time:

1. Generator `f_θ(ε)` maps noise to samples in a single forward pass
2. Each training step computes a drift field `V` that pushes generated samples toward the data
3. Loss: `||f_θ(ε) - stopgrad(f_θ(ε) + V)||²` — move predictions toward their drifted versions

The current implementation uses **local Laplacian drift fields** (k-NN based), with a **mean-shift** alternative also available.

## Quick start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Verify setup (no data download needed — distributions are sampled on the fly)
uv run prepare.py

# Run single experiment (~2 min)
uv run train.py
```

## Agent instructions

The agent program is defined in `program.md`. To run:

1. Point the agent at this repo
2. Tell it to read `program.md` and begin
3. The agent will establish a baseline and then iterate autonomously

## Files

| File | Role |
|------|------|
| `train.py` | **Modifiable** — generator, drift computation, training loop, hyperparameters |
| `prepare.py` | **Read-only** — toy data samplers, MMD evaluation, constants |
| `program.md` | Agent instructions — experiment protocol |
| `results.tsv` | Experiment log (untracked) |
| `analysis.ipynb` | Notebook for analyzing experiment results |
