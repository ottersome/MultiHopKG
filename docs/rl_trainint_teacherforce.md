# rl_trainint_teacherforce.py notes

High-level role: trains a graph-navigation policy (continuous SAC variant) together with a pretrained graph-to-text model (HunchBart) using teacher forcing and replay-buffer hydration. The script is the experiment entrypoint for RL finetuning after graph/LLM pretraining.

## Startup and configuration
- `initial_setup()` pulls args from `run_configs.rl_alpha`, merges YAML overrides, seeds RNGs, and configures logging.
- `main()` wires everything: optional debugpy attach, wandb init, load pretrained HunchBart weights/tokenizer plus metadata paths, load KGE embeddings/checkpoint (`KGEModel`), and cached QA data via `data_utils.load_cached_pretraining_data`.
- Builds ANN indexers for entity/relation lookups, sets up `ReinforcedUnsupervisedEnv` with the KGE model, and instantiates the navigation policy (`ContinuousPolicyGradient`), critics (`GraphCriticQ` x2, `GraphCriticV`), and replay buffer (load from cache or prepopulate).
- Training hyperparams come from args (batch sizes, update counts, hydration cadence, teacherforce reg weight, etc.).

## Key helpers
- **AimWriter**: minimal Aim logging adapter exposing `add_scalar`.
- **QuestionCoverageSampler**: shuffles question ids to ensure uniform coverage when sampling minibatches.
- **Reward functions**:
  - `calculate_llm_reward_supasoft`: runs HunchBart encoder/decoder on graph paths and returns negative MSE between predicted answer embedding and reference BERT answer embedding (primary reward used).
  - `calculate_llm_reward_autoregressive`: teacher-forced CE-based reward (currently unused in the main loop).
- **Ground-truth processing**: `_prepare_question_prompts` builds decoder prompts from Q/A tokens; `get_ground_truth_paths` converts discrete triple ids into padded entity/action embeddings for supervision.

## Replay buffer lifecycle
- **Construction**: `QuestionReplayBuffer` stores per-question trajectories, actions, rewards, log-probs, masks, etc.
- **Prepopulation** (`prepopulate_replay_buffer`): seeds buffer with one-step random actions per question, computes LLM reward from the resulting path, and writes transitions keyed by question id.
- **Hydration** (`hydrate_replay_buffer`): periodically extends the oldest trajectories sampled per question. It:
  - Restarts finished paths, appends policy actions to paths, steps the environment, and recomputes rewards via `calculate_llm_reward_supasoft`.
  - Optionally fetches teacher targets (next relation) for alignment metrics.
  - Logs hydration reward stats and inserts refreshed transitions back into the buffer.

## SAC with teacher forcing (`train_multihopkg`)
- Sets up target value net, optimizers (policy, Q1/Q2, value, alpha), entropy target, and coverage sampler.
- **Teacher targets**: `get_teacher_action_embeddings` maps question id + step count to the ground-truth relation embedding for that step; used for policy regularization and buffer alignment metrics.
- **Update step** (`sac_update_step`):
  - Builds path masks for current/next states, computes target values, Q losses, value loss, and policy loss with entropy term; optionally adds teacher-forcing MSE on actions.
  - Updates alpha for entropy, softly updates target value net, and reports diagnostics (reward stats, log-prob stats, coverage, teacher alignment).
- **Training loop**:
  - Iterates epochs; each collection step samples questions via `QuestionCoverageSampler`, draws transitions from the replay buffer, and runs `sac_update_step`.
  - Hydrates replay buffer every `hydration_interval` updates.
  - Periodically evaluates on dev set with `evaluate_seq2seq_outputs`.

## Evaluation (`evaluate_seq2seq_outputs`)
- Runs the policy in the environment for up to `max_env_steps`, building a path trace.
- Feeds translated path embeddings into HunchBart to score token-level cross-entropy on answers, compute token accuracy/exact match, average steps, and generated lengths.
- Generates text answers with BART, logs sample questions/predictions, and compares predicted path embeddings to ANN nearest neighbors for interpretability (uses id→title maps).
- Logs metrics to Aim (and wandb if enabled).

## Script entrypoint
- After setup, `main()` calls `train_multihopkg` with all constructed modules and args.
- Side artifacts: replay buffer cache saved if newly built; logs go to `runs/rl_sac/<kg>/<timestamp>/` for Aim.
