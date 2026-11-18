# Reproducible reviewer pipeline for the four supported QA settings.
#
#   make reproduce GPU=0
#   make reproduce GPU=0 WANDB=1
#
# Each family uses one fixed ConvE substrate and three independently seeded RL
# policies. Artifacts are staged with their config files for batch_eval.py.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Use the lockfile-managed environment. batch_eval.py launches one Python
# subprocess per checkpoint, so its own interpreter must already provide
# PyTorch and the project's optional runtime dependencies.
PYTHON ?= uv run python
GPU ?= 0
WANDB ?= 0
DETERMINISTIC ?= 1
RUN_NAME ?= reviewer_3seed
RL_SEEDS ?= 12 42 1337

FINAL_ROOT ?= model/final_evaluation/$(RUN_NAME)
BUILD_ROOT ?= model/.reproduce_runs/$(RUN_NAME)
LP_ROOT := $(FINAL_ROOT)/lp_embeddings_conve
EVAL_ROOT := $(FINAL_ROOT)/evaluation

KINSHIP_LP := $(LP_ROOT)/kinshiphinton_conve.tar
METAQA_LP := $(LP_ROOT)/metaqa_encoded_conve.tar
MQUAKE_SA_LP := $(LP_ROOT)/mquake_st_conve.tar

KINSHIP_STAGE := $(FINAL_ROOT)/kinship
METAQA_STAGE := $(FINAL_ROOT)/metaqa_encoded
MQUAKE_SA_STAGE := $(FINAL_ROOT)/mquake_sa
MQUAKE_MA_STAGE := $(FINAL_ROOT)/mquake_ma

ifeq ($(WANDB),1)
WANDB_FLAG := --wandb
else
WANDB_FLAG :=
endif

ifeq ($(DETERMINISTIC),1)
DETERMINISTIC_FLAG := --deterministic
else
DETERMINISTIC_FLAG :=
endif

.PHONY: help reproduce pipeline reproduce-kinship train-all train-kinship evaluate evaluate-kinship evaluate-additional-paraphrases clean-caches \
	process-data process-data-kinship process-data-metaqa process-data-mquake \
	prepare-lp-data prepare-lp-data-kinship prepare-lp-data-metaqa prepare-lp-data-mquake \
	linkpred-all rl-all \
	linkpred-kinship linkpred-metaqa linkpred-mquake-sa \
	rl-kinship rl-metaqa rl-mquake-sa rl-mquake-ma

help:
	@printf '%s\n' \
	  'make reproduce GPU=0          # full preprocessing, training, and evaluation' \
	  'make reproduce-kinship GPU=0  # full Kinship-only replication check' \
	  'make reproduce GPU=0 WANDB=1  # same pipeline with Weights & Biases logging' \
	  'Optional: RUN_NAME=my_run RL_SEEDS="12 42 1337"' \
	  'make train-all GPU=0          # stage three RL seeds per family, without evaluation' \
	  'make evaluate GPU=0           # evaluate an already staged run' \
	  'make evaluate-additional-paraphrases GPU=0  # MQuAKE external paraphrase suites' \
	  'make clean-caches             # remove only reproducible project caches.'

# Safe for a clean run: these directories contain only derived QA/tokenization,
# KG action-space, and local Python/test cache files.  This intentionally does
# not delete data/, processed KG indices, model checkpoints, final evaluation
# artifacts, or user-level Hugging Face/uv caches.
clean-caches:
	rm -rf -- .cache data/KinshipHinton/.kg_cache data/metaqa_encoded/.kg_cache data/mquake_st/.kg_cache
	find . -path './.venv' -prune -o \( -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf -- '{}' + \)

# These files define entity/relation IDs and traversal adjacency. MQuAKE SA and
# MA share one KG, so it is intentionally processed once.
process-data:
	+$(MAKE) process-data-kinship GPU="$(GPU)"
	+$(MAKE) process-data-metaqa GPU="$(GPU)"
	+$(MAKE) process-data-mquake GPU="$(GPU)"

process-data-kinship:
	./experiment-rs-nlp.sh configs/kinshiphinton-rs-nlp-conve.sh --process_data $(GPU)

process-data-metaqa:
	./experiment-rs-nlp.sh configs/metaqa_encoded-rs-nlp-conve.sh --process_data $(GPU)

process-data-mquake:
	./experiment-rs-nlp.sh configs/mquake_st-rs-nlp-conve.sh --process_data $(GPU)

# LP is trained on the complete observed KG, never a QA split.
prepare-lp-data: prepare-lp-data-kinship prepare-lp-data-metaqa prepare-lp-data-mquake

prepare-lp-data-kinship: process-data-kinship
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/KinshipHinton

prepare-lp-data-metaqa: process-data-metaqa
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/metaqa_encoded

prepare-lp-data-mquake: process-data-mquake
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/mquake_st

linkpred-all: linkpred-kinship linkpred-metaqa linkpred-mquake-sa

linkpred-kinship: prepare-lp-data-kinship
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/kinshiphinton-conve.sh --train $(GPU) $(WANDB_FLAG) $(DETERMINISTIC_FLAG) --model_root_dir "$(BUILD_ROOT)/kinship/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/kinship-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/kinship-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(KINSHIP_LP)"

linkpred-metaqa: prepare-lp-data-metaqa
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/metaqa_encoded_conve.sh --train $(GPU) $(WANDB_FLAG) $(DETERMINISTIC_FLAG) --model_root_dir "$(BUILD_ROOT)/metaqa_encoded/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/metaqa-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/metaqa-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(METAQA_LP)"

linkpred-mquake-sa: prepare-lp-data-mquake
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/mquake_st-conve.sh --train $(GPU) $(WANDB_FLAG) $(DETERMINISTIC_FLAG) --model_root_dir "$(BUILD_ROOT)/mquake_sa/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/mquake-sa-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/mquake-sa-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(MQUAKE_SA_LP)"

# $(1): staging directory; $(2): source config; $(3): LP checkpoint;
# $(4): build-directory/manifest label. Only the first seed refreshes the QA
# cache, fixing the data split for the remaining seeds.
define train_rl_family
	@mkdir -p "$(1)" "$(BUILD_ROOT)/manifests"
	@sed -e '/^checkpoint_path=/d' -e 's|^conve_state_dict_path=.*|conve_state_dict_path="$(3)"|' "$(2)" > "$(1)/$$(basename "$(2)")"
	@set -e; first_seed=1; for seed in $(RL_SEEDS); do \
		metadata="$(BUILD_ROOT)/manifests/$(4)-rl-s$$seed.json"; \
		refresh=""; \
		if [ "$$first_seed" -eq 1 ]; then refresh="--recompute_qadata_cache"; first_seed=0; fi; \
		./experiment-rs-nlp.sh "$(2)" --train $(GPU) $(WANDB_FLAG) $(DETERMINISTIC_FLAG) \
			--seed "$$seed" \
			--conve_state_dict_path "$(3)" \
			--model_root_dir "$(BUILD_ROOT)/$(4)/rl/s$$seed" \
			--run_metadata_output_path "$$metadata" \
			$$refresh; \
		checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$$metadata")"; \
		test -f "$$checkpoint"; \
		mv -f "$$checkpoint" "$(1)/s$${seed}_model.tar"; \
	done
endef

rl-all: rl-kinship rl-metaqa rl-mquake-sa rl-mquake-ma

rl-kinship: linkpred-kinship
	$(call train_rl_family,$(KINSHIP_STAGE),configs/kinshiphinton-rs-nlp-conve.sh,$(KINSHIP_LP),kinship)

rl-metaqa: linkpred-metaqa
	$(call train_rl_family,$(METAQA_STAGE),configs/metaqa_encoded-rs-nlp-conve.sh,$(METAQA_LP),metaqa_encoded)

rl-mquake-sa: linkpred-mquake-sa
	$(call train_rl_family,$(MQUAKE_SA_STAGE),configs/mquake_st-rs-nlp-conve.sh,$(MQUAKE_SA_LP),mquake_sa)

rl-mquake-ma: linkpred-mquake-sa
	$(call train_rl_family,$(MQUAKE_MA_STAGE),configs/mquake_st_ma-rs-nlp-conve.sh,$(MQUAKE_SA_LP),mquake_ma)

train-all: rl-all

# A compact end-to-end reproduction target. It still builds the KG, trains a
# fresh ConvE substrate, trains every requested RL seed, and evaluates those
# staged checkpoints; it simply avoids the other dataset families.
train-kinship: rl-kinship

# Evaluate each family separately so the reported mean +/- std is over its
# three seeds rather than over all twelve checkpoints. MQuAKE evidence labels
# are loaded directly from its bundled metadata CSVs during evaluation.
evaluate:
	@mkdir -p "$(EVAL_ROOT)"
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/kinship" --output_json "$(EVAL_ROOT)/kinship.json" --output_csv "$(EVAL_ROOT)/kinship.csv" "$(KINSHIP_STAGE)" -- $(DETERMINISTIC_FLAG)
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/metaqa_encoded" --output_json "$(EVAL_ROOT)/metaqa_encoded.json" --output_csv "$(EVAL_ROOT)/metaqa_encoded.csv" "$(METAQA_STAGE)" -- $(DETERMINISTIC_FLAG)
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_sa" --output_json "$(EVAL_ROOT)/mquake_sa.json" --output_csv "$(EVAL_ROOT)/mquake_sa.csv" "$(MQUAKE_SA_STAGE)" -- $(DETERMINISTIC_FLAG)
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_ma" --output_json "$(EVAL_ROOT)/mquake_ma.json" --output_csv "$(EVAL_ROOT)/mquake_ma.csv" "$(MQUAKE_MA_STAGE)" -- $(DETERMINISTIC_FLAG)

evaluate-kinship:
	@mkdir -p "$(EVAL_ROOT)"
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/kinship" --output_json "$(EVAL_ROOT)/kinship.json" --output_csv "$(EVAL_ROOT)/kinship.csv" "$(KINSHIP_STAGE)" -- $(DETERMINISTIC_FLAG)

# These are separately supplied, test-only question variants. They deliberately
# use isolated caches and never call the canonical QA splitter or rewrite its
# train/dev/test cache. Each family is evaluated against identical rows and
# gold paths with Question, Question-Paraphrased, and New-Paraphrase text.
evaluate-additional-paraphrases:
	@mkdir -p "$(EVAL_ROOT)"
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_sa_additional_question" --output_json "$(EVAL_ROOT)/mquake_sa_additional_question.json" --output_csv "$(EVAL_ROOT)/mquake_sa_additional_question.csv" "$(MQUAKE_SA_STAGE)" -- --external_eval_qa_path data/mquake_st/mquake_st_sa_nhop_additional.csv --external_eval_cache_path .cache/mquake_st/mquake_st_sa_additional_question_eval.json --external_eval_question_column Question --external_eval_setting sa
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_sa_additional_question_paraphrased" --output_json "$(EVAL_ROOT)/mquake_sa_additional_question_paraphrased.json" --output_csv "$(EVAL_ROOT)/mquake_sa_additional_question_paraphrased.csv" "$(MQUAKE_SA_STAGE)" -- --external_eval_qa_path data/mquake_st/mquake_st_sa_nhop_additional.csv --external_eval_cache_path .cache/mquake_st/mquake_st_sa_additional_question_paraphrased_eval.json --external_eval_question_column Question-Paraphrased --external_eval_setting sa
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_sa_additional_new_paraphrase" --output_json "$(EVAL_ROOT)/mquake_sa_additional_new_paraphrase.json" --output_csv "$(EVAL_ROOT)/mquake_sa_additional_new_paraphrase.csv" "$(MQUAKE_SA_STAGE)" -- --external_eval_qa_path data/mquake_st/mquake_st_sa_nhop_additional.csv --external_eval_cache_path .cache/mquake_st/mquake_st_sa_additional_new_paraphrase_eval.json --external_eval_question_column New-Paraphrase --external_eval_setting sa
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_ma_additional_question" --output_json "$(EVAL_ROOT)/mquake_ma_additional_question.json" --output_csv "$(EVAL_ROOT)/mquake_ma_additional_question.csv" "$(MQUAKE_MA_STAGE)" -- --external_eval_qa_path data/mquake_st/mquake_st_ma_nhop_additional.csv --external_eval_cache_path .cache/mquake_st/mquake_st_ma_additional_question_eval.json --external_eval_question_column Question --external_eval_setting ma
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_ma_additional_question_paraphrased" --output_json "$(EVAL_ROOT)/mquake_ma_additional_question_paraphrased.json" --output_csv "$(EVAL_ROOT)/mquake_ma_additional_question_paraphrased.csv" "$(MQUAKE_MA_STAGE)" -- --external_eval_qa_path data/mquake_st/mquake_st_ma_nhop_additional.csv --external_eval_cache_path .cache/mquake_st/mquake_st_ma_additional_question_paraphrased_eval.json --external_eval_question_column Question-Paraphrased --external_eval_setting ma
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_ma_additional_new_paraphrase" --output_json "$(EVAL_ROOT)/mquake_ma_additional_new_paraphrase.json" --output_csv "$(EVAL_ROOT)/mquake_ma_additional_new_paraphrase.csv" "$(MQUAKE_MA_STAGE)" -- --external_eval_qa_path data/mquake_st/mquake_st_ma_nhop_additional.csv --external_eval_cache_path .cache/mquake_st/mquake_st_ma_additional_new_paraphrase_eval.json --external_eval_question_column New-Paraphrase --external_eval_setting ma
	$(PYTHON) scripts/render_question_variant_comparison.py --output "$(EVAL_ROOT)/mquake_sa_additional_question_variants.md" --input "Question=$(EVAL_ROOT)/mquake_sa_additional_question.json" --input "Question-Paraphrased=$(EVAL_ROOT)/mquake_sa_additional_question_paraphrased.json" --input "New-Paraphrase=$(EVAL_ROOT)/mquake_sa_additional_new_paraphrase.json"
	$(PYTHON) scripts/render_question_variant_comparison.py --output "$(EVAL_ROOT)/mquake_ma_additional_question_variants.md" --input "Question=$(EVAL_ROOT)/mquake_ma_additional_question.json" --input "Question-Paraphrased=$(EVAL_ROOT)/mquake_ma_additional_question_paraphrased.json" --input "New-Paraphrase=$(EVAL_ROOT)/mquake_ma_additional_new_paraphrase.json"

# Explicit recursive calls preserve ordering even with make -j: evaluation
# must not begin before all staged checkpoints exist.
reproduce pipeline:
	+$(MAKE) train-all GPU="$(GPU)" WANDB="$(WANDB)" DETERMINISTIC="$(DETERMINISTIC)" RUN_NAME="$(RUN_NAME)" RL_SEEDS="$(RL_SEEDS)" FINAL_ROOT="$(FINAL_ROOT)" BUILD_ROOT="$(BUILD_ROOT)"
	+$(MAKE) evaluate GPU="$(GPU)" WANDB="$(WANDB)" DETERMINISTIC="$(DETERMINISTIC)" RUN_NAME="$(RUN_NAME)" RL_SEEDS="$(RL_SEEDS)" FINAL_ROOT="$(FINAL_ROOT)" BUILD_ROOT="$(BUILD_ROOT)"

reproduce-kinship:
	+$(MAKE) train-kinship GPU="$(GPU)" WANDB="$(WANDB)" DETERMINISTIC="$(DETERMINISTIC)" RUN_NAME="$(RUN_NAME)" RL_SEEDS="$(RL_SEEDS)" FINAL_ROOT="$(FINAL_ROOT)" BUILD_ROOT="$(BUILD_ROOT)"
	+$(MAKE) evaluate-kinship GPU="$(GPU)" WANDB="$(WANDB)" DETERMINISTIC="$(DETERMINISTIC)" RUN_NAME="$(RUN_NAME)" RL_SEEDS="$(RL_SEEDS)" FINAL_ROOT="$(FINAL_ROOT)" BUILD_ROOT="$(BUILD_ROOT)"
