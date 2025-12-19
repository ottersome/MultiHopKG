# Follow-ups and potential fixes

Context: seq2seq evaluation was running to max steps and RL metrics showed odd trends. Below are targeted fixes and checks.

## Seq2seq termination / masking
- The main masking bug was using `step_active` (last active subset) for encoder attention; fixed to mask `0..(2*step_counter+1)` per sample so padded steps are ignored and finished rollouts don’t influence decoding.
- Ensure env stop condition is reachable: `ReinforcedUnsupervisedEnv.step` only flags done when distance < `reached_destination_threshold`. If never triggered, consider raising the threshold or adding a STOP action whose selection sets `done=True`.

## RL target handling
- Q-target currently ignores terminal flags: `q_target = rewards + gamma * target_values`. Replace with `(1 - dones) * gamma * target_values` to avoid leaking future value past terminal transitions.
- Hydration done propagation is now merged with env success; confirm replay buffer stores `dones` correctly and sampling returns them unchanged.

## Replay hydration edge cases
- `initial_ids`/`answer_ids` in `hydrate_replay_buffer` are overwritten in the loop (only last path kept). Make them lists and stack tensors so resets/rest states map to each sample.
- When `done_flags` is true, synthetic `gt_action_id`/`gt_entity_id` are set to zero; ensure those placeholders aren’t used downstream (they are currently ignored).

## Buffer alignment metric drift
- `buffer_alignment_metric` measures MSE between stored actions and teacher relation embeddings. Persistent decline can stem from:
  - Teacher mapping mismatch: `get_teacher_action_embeddings` uses `step_counter`; if counters get reset incorrectly or exceed path length, relations default to the last one, biasing the metric.
  - Stale/incorrect steps: hydration resets may drop path progress if `done_flags` is mis-set; then alignment compares later-step actions to early-step teachers.
  - Default relation fallback: missing teacher entries use `default_relation_idx`, pulling alignment down as policy diverges.
  - Missing terminal bootstrapping fix (see Q-target note) can push actions away from teacher paths.
  - An action/state scale mismatch (e.g., if relation embeddings differ from policy output scale) will inflate MSE; verify normalization if needed.

## Additional sanity checks
- Verify encoder attention masks in evaluation/translation match path lengths (fixed).
- Consider logging env `done` rates during hydration/eval to spot unreachable thresholds.
- If over-indulgent navigation persists, hard-cap `max_transitions` with an early break and/or add a learned STOP action.
