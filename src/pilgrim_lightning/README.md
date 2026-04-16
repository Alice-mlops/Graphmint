# pilgrim_lightning

Lightning-first training + Kaggle inference library for models from `src/pilgrim`.

## AlGraphGPT Encoder Note

For permutation states (pancake states), use:
- `input_encoder_type: embedding_flatten` (recommended), or
- `input_encoder_type: onehot_linear`,
- `input_encoder_type: lehmer`, or
- `input_encoder_type: lehmer-breakpoints`.

`embedding_mean` support was removed because it is order-invariant for permutations.

## CLI

```bash
python -m pilgrim_lightning.cli run --config configs/pilgrim_lightning_starter.yaml
python -m pilgrim_lightning.cli train --config configs/pilgrim_lightning_starter.yaml
python -m pilgrim_lightning.cli infer --config configs/pilgrim_lightning_starter.yaml
python -m pilgrim_lightning.cli benchmark-algraphgpt --config configs/algraphgpt_timing.yaml
```

## Aim Logging

Central Aim service for this host:
- Tracking endpoint: `aim://127.0.0.1:53800`
- UI: `http://127.0.0.1:43800`
- Shared repo path: `/home/seregin/.local/share/aim/repo`

For local runs from this machine, set absolute repo path in YAML:

```yaml
train:
  aim:
    experiment: pilgrim-lightning-train
    repo: /home/seregin/.local/share/aim/repo
    tags: [pilgrim-lightning, stage:train]

inference:
  aim:
    experiment: pilgrim-lightning-infer
    repo: /home/seregin/.local/share/aim/repo
    tags: [pilgrim-lightning, stage:infer]
```

For Runpod or any remote worker:
- Create SSH tunnel to this host: `ssh -N -L 53800:127.0.0.1:53800 seregin@<aim-host>`
- Set remote env var: `export AIM_REPO=aim://127.0.0.1:53800`
- Keep `experiment`/`tags` in YAML but omit `repo`.

Why omit `repo` for remote URI:
- `aim.repo` is parsed as `Path`, and `aim://...` becomes a broken filesystem-like path.
- `AIM_REPO` env var preserves the URI correctly.

## API

```python
from pilgrim_lightning import (
    load_yaml_config,
    run_from_config,
    run_algraphgpt_timing_from_yaml,
)

cfg = load_yaml_config("configs/pilgrim_lightning_starter.yaml")
result = run_from_config(cfg, mode="run")

timing_result = run_algraphgpt_timing_from_yaml("configs/algraphgpt_timing.yaml")
print(timing_result.summary_path)
```

## AlGraphGPT Timing Benchmark

`benchmark-algraphgpt` runs:
- Lightning train/val micro-runs (per-batch timings + fit wall time).
- Direct forward latency benchmark.
- Beam inference timing on Kaggle `test.csv` subset.
- `compute_stats_by_n` summary for selected `n`.
- Optional `torch.profiler` traces/tables for train and inference bottlenecks.

Config file: `configs/algraphgpt_timing.yaml`

Key knobs:
- `model_sizes`: list of presets (`small/medium/large` etc).
- `train.limit_train_batches` and `train.limit_val_batches`.
- `inference.target_n` and `inference.max_rows_per_n`.
- `inference.direct_forward.*` for pure forward timing.
- `profiler.*` for op-level traces.

Outputs:
- `train_timings.csv`
- `inference_timings.csv`
- `beam_solve_times.csv`
- `by_n_all.csv`
- `operation_profile_all.csv`
- `SUMMARY.md`
- `operation_profile_<preset>_train_forward.csv`
- `operation_profile_<preset>_infer_forward.csv`
- `profiler/*_trace.json`, `profiler/*_profile_ops.csv`

All artifacts are written under `output_root/run_<timestamp>/`.

### Latest Run Snapshot (2026-03-09)

Run directory:
- `artifacts/pilgrim_lightning/algraphgpt_timing/run_20260309_000254`

Training (n=20):
- `small` (271,681 params): train batch mean `59.80 ms`, fit `0.85 s`
- `medium` (2,650,081 params): train batch mean `25.90 ms`, fit `0.65 s`
- `large` (10,675,841 params): train batch mean `76.34 ms`, fit `1.04 s`

Inference (`target_n=[5,12,15,16,20]`, 15 rows):
- `small`: forward mean `6.09 ms`, beam solve mean `270.36 ms`, beam wall `9.28 s`
- `medium`: forward mean `13.16 ms`, beam solve mean `436.28 ms`, beam wall `12.16 s`
- `large`: forward mean `20.77 ms`, beam solve mean `800.41 ms`, beam wall `17.36 s`

Semantic operation profiling highlights (`operation_profile_all.csv`):
- Attention per layer (`layer/*/attention`) increases strongly with size.
- Generator walk sampling (`walk/sample_tokens_total`, `walk/inverse_resample`) is a major inference cost.
- `forward/walk_token_encode_call` is consistently one of the largest model-internal blocks.

Evaluation (`compute_stats_by_n`) for all three sizes matched:
- `n=5`: score `10`, prob_step `11`, potential `-1`
- `n=12`: score `42`, prob_step `42`, potential `0`
- `n=15`: score `62`, prob_step `62`, potential `0`
- `n=16`: score `66`, prob_step `66`, potential `0`
- `n=20`: score `102`, prob_step `102`, potential `0`
