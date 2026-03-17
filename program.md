# autoresearch-drift

This is an experiment to have an LLM agent autonomously research generative drifting models.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context and background on drifting models.
   - `prepare.py` — fixed constants, toy data samplers, MMD evaluation. Do not modify.
   - `train.py` — the file you modify. Generator architecture, drift computation, optimizer, training loop.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Background: Drifting Models

**Traditional generative models** (Diffusion, Flow Matching) iterate at **inference time** — many forward passes to generate one sample.

**Drifting Models** move the iteration to **training time**:
- Each training step "drifts" the generated samples toward the data distribution
- The drifting field V_{p,q} tells each generated sample where to go
- At inference: just one forward pass (1 NFE)

The training loss is: L = E_ε [||f_θ(ε) - stopgrad(f_θ(ε) + V(f_θ(ε)))||²]

The current implementation uses a **local Laplacian drift field**:
- V_pos: attraction toward real data (k-NN Laplacian from data, interpolated at generated points)
- V_neg: self-repulsion (k-NN Laplacian on generated samples)
- V = V_pos - α * V_neg

An alternative **mean-shift** drift field is also available, using batch-normalized exponential kernels.

Paper: "Generative Modeling via Drifting"

## Experimentation

Each experiment runs on a single GPU (or CPU for toy 2D). The training script runs for a **fixed time budget of 2 minutes** (wall clock training time, excluding startup). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - **Generator architecture**: MLP depth, width, activations, skip connections, normalization
  - **Drift computation**: Laplacian vs mean-shift, temperature, k-NN count, alpha, kernel choice
  - **Optimizer**: Adam, SGD, learning rate schedules
  - **Batch sizes**: data batch size, generated batch size
  - **Novel ideas**: adaptive temperature, curriculum, different loss formulations, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation (MMD), data samplers, and training constants (time budget).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_mmd` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_mmd** (MMD² between generated and reference samples). Since the time budget is fixed, focus on making each training step as effective as possible.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A simplification that gives equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Dataset**: The default dataset is `swiss_roll`. You may change it in train.py, but val_mmd values are only comparable within the same dataset. If you switch datasets, treat it as a new baseline. The available datasets are: `swiss_roll`, `checkerboard`, `mog`.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_mmd:          0.001234
training_seconds: 120.1
total_seconds:    125.5
peak_vram_mb:     123.4
num_steps:        5432
num_params:       123456
dataset:          swiss_roll
drift_type:       laplacian
```

You can extract the key metric from the log file:

```
grep "^val_mmd:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_mmd	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_mmd achieved (e.g. 0.001234) — use 0.000000 for crashes
3. peak memory in MB, round to .1f (e.g. 123.4) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_mmd	memory_mb	status	description
a1b2c3d	0.012345	50.0	keep	baseline swiss_roll
b2c3d4e	0.008200	55.0	keep	increase hidden_dim to 512
c3d4e5f	0.015000	50.0	discard	switch to mean_shift drift
d4e5f6g	0.000000	0.0	crash	batch_size too large (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_mmd:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_mmd improved (lower), you "advance" the branch, keeping the git commit
9. If val_mmd is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~2 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try different drift formulations, vary kernel functions, explore temperature schedules. The loop runs until the human interrupts you, period.
