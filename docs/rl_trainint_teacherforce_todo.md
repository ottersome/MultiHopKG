# rl_trainint_teacherforce.py TODOs

## Stop behavior and termination
- [ ] **Verify STOP actually fires in rollouts.** The environment defines a stop embedding + threshold, but the actor has no explicit incentive to hit it. We added stop reward shaping and step penalty; confirm the policy learns to end episodes early.
- [ ] **Tune step penalty and stop reward weight.** Start with `--step_penalty -0.01` and `--stop_reward_weight 1.0`, then adjust based on average steps and success rate.
- [ ] **Log stop rates.** Add metrics for fraction of stop actions taken and average stop distance so you can see if stopping is being used.

## Reward + path usage (known issues)
- [ ] **Hydration reward should use full path.** The reward call currently uses only the next state; it should use the full path tensor so the reward is path-aware.
- [ ] **Path mask length should be `2 * step + 1`.** Several masks use `step + 1`, which makes the model ignore action/state pairs.
- [ ] **Answer embeddings should be float.** Ensure `answer_bert_embs` stay `float32` in hydration/prepopulate to avoid MSE on integers.
- [ ] **Consistent prompt format.** Always append BOS to question tokens (or never) across prepopulate/hydrate/eval.
- [ ] **Use correct padding constants.** `PATH_PADDING_VALUE` should be used for path masks instead of `BART_PADDING_VALUE`.

## Reward model quality
- [ ] **Add contrastive path loss in pretraining.** (Already implemented in `pretraining.py`.) Re-run pretraining and re-measure reward separation.
- [ ] **Re-run `pretraining_analysis.py` after retraining.** Compare margin probe and corrupted/permuted gaps to see if the reward head became path-sensitive.

## Replay buffer integrity
- [ ] **Store real done flags in prepopulation.** This prevents training on transitions that should be terminal.
- [ ] **Check step counters vs max steps.** Off-by-one bugs can cause states to be advanced past max length.

## Evaluation alignment
- [ ] **Make evaluation use the same prompt and reward inputs as training.** `evaluate_seq2seq_outputs` currently mixes full Q+A prompts and question-only prompts in different steps.
- [ ] **Fix masking math in evaluation.** Make decoder pooling masks and denominator consistent.

