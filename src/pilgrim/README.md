# Pilgrim

## Lipschitz expansion regularizer (1-Lipschitz)

Training supports an optional 1-Lipschitz expansion penalty added to MSE loss.
This is computed on training batches only.

Config keys (all optional, defaults keep current behavior):
- `lip_weight` (float, default `0.0`): multiplier for the penalty. Set > 0 to enable.
  The training loss is `MSE + lip_weight * L_lip` (no averaging between terms).
- `lip_val_metric` (bool, default `False`): compute and print `LipVal` on the
  validation set (even if `lip_weight` is 0).
- `lip_max_states` (int | None, default `None`): cap on states per batch.
  Use `-1` to use all states in the batch.
- `lip_max_generators` (int | None, default `None`): cap on generators.
  Use `-1` to use all generators.
- `lip_generator_indices` (Sequence[int] | None, default `None`): explicit generator
  indices to use. If `None`, uses all.
- `lip_state_batch_size` (int | None, default `None`): internal batch size for the
  regularizer (for memory control).
- `lip_reduction` ("mean" | "sum", default `"mean"`): reduction over (u, v) pairs.
- `lip_seed` (int | None, default `None`): seed for subsampling. If `None`, uses
  global RNG.

Computation sizing (per call to the regularizer, i.e. per training batch and per
validation batch when `lip_val_metric` is enabled):
- Let `N_states` be the number of states after subsampling the batch:
  `N_states = min(batch_size, lip_max_states)` when `lip_max_states` is set, or
  `N_states = batch_size` when `lip_max_states` is `None`/`-1`.
- Let `N_generators` be the number of generators after selection:
  start from `len(graph.generators)` (or `len(lip_generator_indices)` if provided),
  then cap with `lip_max_generators` if set (or no cap for `None`/`-1`).
- Total (u, v) pairs evaluated is `N_states * N_generators` (this is what the
  reduction aggregates).
- Batching: states are processed in chunks of size `lip_state_batch_size`
  (or all states if `None`). For each chunk, all `N_generators` are applied.
  So the number of generator-application loops is
  `ceil(N_states / lip_state_batch_size) * N_generators`.

Example:
```python
CFG.update(
    lip_weight=0.05,
    lip_max_states=-1,
    lip_max_generators=8,
    lip_state_batch_size=256,
    lip_reduction="mean",
)
```

## Alice-in-Cayleyland model (random-walk attention)

`AliceInCayleyland` is a Pilgrim-family model variant that adds a small
random-walk neighborhood around each input node and aggregates it with
cross-attention in `forward()`.

Key idea:
- For each input state `z`, build a token set from either:
  - exact 1-hop neighbors plus one self token, or
  - `alice_num_walks` random walks of length `alice_walk_length`.
- Embed those tokens, then attend from the center embedding to them.

Important notes:
- In `alice_token_source="random_walk"` mode, forward cost scales roughly with
  `(1 + alice_num_walks * alice_walk_length)`.
- In `alice_token_source="one_hop"` mode, forward cost scales roughly with
  `(1 + n_selected_generators)`.
- If you also enable the Lipschitz expansion regularizer, training can become
  much more expensive because the regularizer calls the model many times.

Minimal config snippet:
```python
from pilgrim.model import AliceInCayleyland
from pilgrim.utils.pancake_utils import make_graph_for_n

graph = make_graph_for_n(n)

CFG.update(
    generator_moves=graph.definition.generators_permutations,
    alice_token_source="random_walk",
    alice_num_walks=8,
    alice_walk_length=5,
    alice_attention_heads=4,
    alice_backtrack_mode="inverse",
    alice_backtrack_memory=1,
)

model = AliceInCayleyland(CFG)
```

## Learning-rate scheduler

`train_model_one_n(...)` supports optional LR scheduling. You can either pass an
explicit `lr_scheduler_ctor=...`, or set `CFG["lr_scheduler"]` and let the
training loop build it.

Notes:
- Schedulers are **stepped once per epoch** (after validation).
- `ReduceLROnPlateau` is stepped as `scheduler.step(val_loss)`. All other
  schedulers are stepped as `scheduler.step()`.

Config forms:
- `lr_scheduler="plateau" | "cosine" | "cosine_restarts" | "none"`
- `lr_scheduler={"type": "...", ...params...}`

Examples:
```python
# Reduce on plateau (validation-driven)
CFG.update(lr_scheduler="plateau")

# Cosine annealing over the whole run (epoch-driven)
CFG.update(lr_scheduler={"type": "cosine", "t_max": CFG["num_epochs"], "eta_min": 5e-7})

# Cosine with warm restarts (SGDR-style)
CFG.update(
    lr_scheduler={"type": "cosine_restarts", "t0": 10, "t_mult": 2, "eta_min": 5e-7}
)
```

What is “cosine with restarts”?
- It runs a cosine LR decay for `t0` epochs, then **restarts** (jumps LR back up
  to the base LR), then repeats.
- After each restart, the cycle length becomes `t0 * (t_mult ** k)` for restart
  index \(k\) (so cycles can get longer over time).

## RL fitted value iteration and Aim metrics

The notebook `pancake_v_iteration_n19.ipynb` uses the RL utilities in
`src/pilgrim/rl/` together with the Aim tracker in
`src/pilgrim/aim_logging/rl_v_iteration.py`.

It trains a scalar value model `V(s)` with fitted value iteration:

- `V(center) = 0`
- `V(s) = 1 + min_a V(T(s, a))` for non-terminal states

The notebook logs to the shared Aim repo
`/home/seregin/.local/share/aim/repo` by default.

### What the tracked metrics mean

Optimization and stability:
- `train/bellman_loss`: MSE between the online model prediction and the
  one-step Bellman target built from the frozen target model.
- `train/total_loss`: Full optimized loss. In the current notebook this is the
  same as `train/bellman_loss` because the Lipschitz penalty is disabled.
- `grad/global_norm`: Global L2 norm of all gradients before optional clipping.
  Large transient spikes are not automatically a problem, but repeated growth or
  runaway values usually means unstable bootstrapping.
- `grad/max_abs`: Maximum absolute gradient entry. Useful for spotting rare but
  extreme outliers.
- `param/global_norm`: L2 norm of the current parameter vector.
- `param/max_abs`: Largest absolute parameter value.

Replay and batch composition:
- `replay/size`: Current replay-buffer size.
- `replay/fill_ratio`: Replay occupancy divided by capacity.
- `batch/size`: Number of replay states used in one optimizer step.
- `batch/center_fraction`: Fraction of batch states equal to the center. If this
  is too high, the training distribution is overly concentrated near the center.
- `batch/unique_ratio`: Fraction of unique rows in the sampled batch. Low values
  indicate heavy duplication and a narrow replay distribution.
- `batch/examples_per_s`: Effective training throughput for the replay batch.

Value diagnostics:
- `value/pred_*`: Statistics of online-model predictions on the sampled batch.
- `value/target_*`: Statistics of Bellman targets on the sampled batch.
- `value/residual_*`: Statistics of `prediction - target` on the sampled batch.
  A small residual only means Bellman consistency on sampled replay states; it
  does not prove good greedy-policy quality everywhere.
- `value/center_pred`: Predicted value for the central state. Ideally this
  stays near `0`.

Probe diagnostics:
- `probe/target_XX`: Reference distance returned by the graph sampler for one
  fixed probe state.
- `probe/value_XX`: Model prediction for that fixed probe state.
- `probe/reached_center_XX`: Whether a greedy rollout from that probe reaches
  the center within the configured rollout cap.
- `probe/rollout_len_XX`: Number of greedy actions taken for that probe rollout.
- `probe/success_rate`: Fraction of tracked probes that reached the center.
- `probe/rollout_len_mean` and `probe/rollout_len_max`: Aggregate greedy
  rollout lengths over the probe set.

### How to read the probe metrics

The probe metrics are diagnostics, not training objectives:

- The model is optimized only through Bellman loss on replay states.
- Probe values and greedy rollouts are periodic evaluation signals.
- `probe/success_rate` is not expected to improve monotonically.
- A low Bellman loss together with weak probe success means the model learned
  local Bellman consistency on the sampled distribution, but the induced greedy
  policy is still not reliable enough away from easy states.

### What is and is not evaluated during notebook training

During `trainer.fit(...)`, the notebook performs:

- Bellman-loss training on replay states.
- Periodic probe evaluation through the Aim tracker.
- Periodic greedy rollout checks on the fixed probe set.

The notebook does not perform:

- a separate train/validation split,
- a held-out evaluation dataset,
- checkpoint selection by validation metric,
- early stopping,
- full beam-search evaluation during training.

There is only an optional post-training notebook cell that prints the center
value and a few greedy rollouts from the probe set.
