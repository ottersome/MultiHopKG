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

## Reward choice: candid guidance
Short answer: the current supasoft reward (negative MSE between predicted answer embedding and BERT answer embedding) is a reasonable dense signal for exploration, but it is not strictly “better” than cross-entropy. Each has tradeoffs, and the best choice is often a hybrid.

### What you have now (embedding MSE reward)
Pros:
- Dense and smooth reward even when answers are wrong, which helps early exploration.
- Avoids sparse zero/one rewards; stabilizes SAC updates.
- Works even when tokenization or exact string match is brittle.

Cons:
- Encourages “semantic closeness” but can be satisfied by wrong answers that are embedding-near.
- Can drift if the answer embedding model and the generator diverge.
- MSE scale is uncalibrated across questions, so reward variance can be high.

### Cross-entropy reward (teacher-forced)
Pros:
- Tied directly to the actual tokens you want to generate.
- Stronger signal for exact answer correctness once the policy is near a good path.

Cons:
- Often too sparse/peaky early in RL; exploration suffers.
- Sensitive to tokenization quirks and exact phrasing.

### A better practical option: hybrid reward + curriculum
If I had to pick one direction, I would not “go back” to CE alone. I’d keep the embedding reward but add one of:
- **Hybrid**: `reward = -MSE(answer_emb, pred_emb) + lambda * (-CE)` with a schedule where `lambda` ramps up after N updates.
- **Contrastive**: reward margin between the correct answer embedding and a few hard negatives (corrupted/permuted paths already exist in pretraining).
- **Path-consistency shaping**: small stepwise reward for matching the teacher path relation at each hop (dense guidance), plus terminal reward on answer.

### My honest take
Use the current embedding reward as the main dense signal early on, then blend in CE (or teacher path alignment) once the navigator stops thrashing. This usually yields better learning dynamics than pure CE or pure embedding reward.

## Supasoft reward probe interpretation (from histogram plot)
The |Δ reward| histograms look heavily concentrated near zero with a long tail. That pattern usually means the reward model is not strongly path-sensitive on most samples.

What I think is happening:
- **Question-only collapse**: `calculate_llm_reward_supasoft` pools decoder hidden states over question tokens only, and the decoder is fed just the question tokens. If the model can predict the answer embedding from the question alone, it can ignore the path, making perfect/random/corrupted rewards similar.
- **Train–eval mismatch**: pretraining uses full `qna_tokens` (question + answer) but the reward evaluation uses question-only prompts. The alignment head may not generalize to question-only inference, flattening reward differences.
- **Weak contrastive pressure**: pretraining optimizes MSE to the answer embedding without an explicit “correct path should beat corrupted/random” constraint, so the head can regress toward an average embedding with low variance across paths.
- **Start-entity leakage**: “random” paths keep the same start entity and hop count; if the reward head latches onto the start entity or early steps, random vs perfect will look similar.

Quick sanity checks:
- Shuffle questions across samples and see if rewards change; if they do not, the path signal is being ignored.
- For a fixed question, sample many random paths and measure variance of predicted answer embeddings; low variance implies collapse.
- Compare `valid/alignment_loss_w_emb` vs `valid/alignment_loss_perm` in pretraining; if close, the head never became path-sensitive.

## Script entrypoint
- After setup, `main()` calls `train_multihopkg` with all constructed modules and args.
- Side artifacts: replay buffer cache saved if newly built; logs go to `runs/rl_sac/<kg>/<timestamp>/` for Aim.
