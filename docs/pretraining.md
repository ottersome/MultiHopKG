# pretraining.py notes

High-level role: pretrains the graph-to-text module (HunchBart) to map KG path embeddings into answer text, plus an answer-embedding alignment head. It produces the pretrained checkpoint and cache artifacts that `rl_trainint_teacherforce.py` expects for RL fine-tuning and reward computation.

## Data + model setup
- Loads MQuAKE QA data via `data_utils.load_qa_data`, tokenizes with the BART tokenizer, and computes BERT answer embeddings for alignment.
- Loads pretrained entity/relation embeddings from `path_graph_emb_data` (NumPy arrays + config metadata).
- Builds `HunchBart` with a pretrained BART backbone and a graph-embedding translator; **freezes the BART weights** so only the translator (and related heads) train.
- Expands the training set by concatenating train/val/test splits for maximum exposure.

## Training loop (supervised, graph-to-answer)
- Uses `GraphEmbeddingDataset` to produce:
  - tokenized question+answer (`qna_tokens`) and answer masks,
  - KG path embeddings and their attention mask,
  - BERT answer embeddings for alignment loss.
- Optimizes a composite loss:
  - token-level cross-entropy over answer tokens (teacher forcing),
  - MSE alignment between predicted answer embedding and BERT answer embedding.
- Validation includes counterfactual losses:
  - corrupts a random hop in the path (`_build_negative_graph_embeddings`),
  - permutes paths across samples,
  - compares CE and alignment losses for real vs corrupted paths.
- Optional generation probe: runs `BART.generate()` on question prompts to log exact-match and sample predictions.

## Outputs and artifacts
- Saves a checkpoint containing:
  - `gtllm_state_dict` (HunchBart weights),
  - tokenizer/model identifiers,
  - graph-embedding dimension,
  - paths to MQuAKE data and graph-embedding data,
  - the graph-embedding training metadata (`embedding_training_metaparam`).
- Writes/updates the QA cache under `args.path_cache_dir` (used later by RL training).

## How it fits with `rl_trainint_teacherforce.py`
- `rl_trainint_teacherforce.py` **loads the checkpoint produced here** via `args.pretrained_gtllm_path` and reconstructs HunchBart using:
  - base BART model id,
  - tokenizer id,
  - graph-embedding dimension,
  - `gtllm_state_dict`.
- It also **reuses the graph embedding metadata and files** (entity/relation embeddings + KGE checkpoint) referenced in the pretraining checkpoint to build the KG environment.
- The RL script **assumes the pretraining cache exists**, loading it via `load_cached_pretraining_data` (see comments around `args.pretraining_metadata_cache_path`).
- Conceptually: `pretraining.py` makes HunchBart a reliable graph-to-answer scorer; `rl_trainint_teacherforce.py` then treats HunchBart as a fixed (or lightly tuned) reward model for navigating the KG with a policy, using its outputs to shape RL updates and evaluate path quality.
