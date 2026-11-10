# Reproducible reviewer pipeline for the four supported QA settings.
#
#   make reproduce GPU=0
#   make reproduce GPU=0 WANDB=1
#
# Each family uses one fixed ConvE substrate and three independently seeded RL
# policies. Artifacts are staged with their config files for batch_eval.py.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
GPU ?= 0
WANDB ?= 0
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

.PHONY: help reproduce pipeline train-all evaluate process-data prepare-lp-data \
	linkpred-all rl-all \
	linkpred-kinship linkpred-metaqa linkpred-mquake-sa \
	rl-kinship rl-metaqa rl-mquake-sa rl-mquake-ma

help:
	@printf '%s\n' \
	  'make reproduce GPU=0          # full preprocessing, training, and evaluation' \
	  'make reproduce GPU=0 WANDB=1  # same pipeline with Weights & Biases logging' \
	  'Optional: RUN_NAME=my_run RL_SEEDS="12 42 1337"' \
	  'make train-all GPU=0          # stage three RL seeds per family, without evaluation' \
	  'make evaluate GPU=0           # evaluate an already staged run.'

# These files define entity/relation IDs and traversal adjacency. MQuAKE SA and
# MA share one KG, so it is intentionally processed once.
process-data:
	./experiment-rs-nlp.sh configs/kinshiphinton-rs-nlp-conve.sh --process_data $(GPU)
	./experiment-rs-nlp.sh configs/metaqa_encoded-rs-nlp-conve.sh --process_data $(GPU)
	./experiment-rs-nlp.sh configs/mquake_st-rs-nlp-conve.sh --process_data $(GPU)

# LP is trained on the complete observed KG, never a QA split.
prepare-lp-data: process-data
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/KinshipHinton
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/metaqa_encoded
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/mquake_st

linkpred-all: linkpred-kinship linkpred-metaqa linkpred-mquake-sa

linkpred-kinship: prepare-lp-data
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/kinshiphinton-conve.sh --train $(GPU) $(WANDB_FLAG) --model_root_dir "$(BUILD_ROOT)/kinship/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/kinship-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/kinship-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(KINSHIP_LP)"

linkpred-metaqa: prepare-lp-data
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/metaqa_encoded_conve.sh --train $(GPU) $(WANDB_FLAG) --model_root_dir "$(BUILD_ROOT)/metaqa_encoded/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/metaqa-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/metaqa-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(METAQA_LP)"

linkpred-mquake-sa: prepare-lp-data
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/mquake_st-conve.sh --train $(GPU) $(WANDB_FLAG) --model_root_dir "$(BUILD_ROOT)/mquake_sa/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/mquake-sa-lp.json"
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
		./experiment-rs-nlp.sh "$(2)" --train $(GPU) $(WANDB_FLAG) \
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

rl-all: rl-kinship rl-mquake-sa rl-mquake-ma # rl-metaqa  #NOTE: Skip metaqa training  for now.

rl-kinship: linkpred-kinship
	$(call train_rl_family,$(KINSHIP_STAGE),configs/kinshiphinton-rs-nlp-conve.sh,$(KINSHIP_LP),kinship)

rl-metaqa: linkpred-metaqa
	$(call train_rl_family,$(METAQA_STAGE),configs/metaqa_encoded-rs-nlp-conve.sh,$(METAQA_LP),metaqa_encoded)

rl-mquake-sa: linkpred-mquake-sa
	$(call train_rl_family,$(MQUAKE_SA_STAGE),configs/mquake_st-rs-nlp-conve.sh,$(MQUAKE_SA_LP),mquake_sa)

rl-mquake-ma: linkpred-mquake-sa
	$(call train_rl_family,$(MQUAKE_MA_STAGE),configs/mquake_st_ma-rs-nlp-conve.sh,$(MQUAKE_SA_LP),mquake_ma)

train-all: rl-all

# Evaluate each family separately so the reported mean +/- std is over its
# three seeds rather than over all twelve checkpoints. MQuAKE evidence labels
# are loaded directly from its bundled metadata CSVs during evaluation.
evaluate:
	@mkdir -p "$(EVAL_ROOT)"
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/kinship" --output_json "$(EVAL_ROOT)/kinship.json" --output_csv "$(EVAL_ROOT)/kinship.csv" "$(KINSHIP_STAGE)"
	# MetaQA is intentionally omitted while its reviewer run is not staged.
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_sa" --output_json "$(EVAL_ROOT)/mquake_sa.json" --output_csv "$(EVAL_ROOT)/mquake_sa.csv" "$(MQUAKE_SA_STAGE)"
	$(PYTHON) batch_eval.py --gpu $(GPU) --log_dir "$(EVAL_ROOT)/mquake_ma" --output_json "$(EVAL_ROOT)/mquake_ma.json" --output_csv "$(EVAL_ROOT)/mquake_ma.csv" "$(MQUAKE_MA_STAGE)"

# Explicit recursive calls preserve ordering even with make -j: evaluation
# must not begin before all staged checkpoints exist.
reproduce pipeline:
	+$(MAKE) train-all GPU="$(GPU)" WANDB="$(WANDB)" RUN_NAME="$(RUN_NAME)" RL_SEEDS="$(RL_SEEDS)" FINAL_ROOT="$(FINAL_ROOT)" BUILD_ROOT="$(BUILD_ROOT)"
	+$(MAKE) evaluate GPU="$(GPU)" WANDB="$(WANDB)" RUN_NAME="$(RUN_NAME)" RL_SEEDS="$(RL_SEEDS)" FINAL_ROOT="$(FINAL_ROOT)" BUILD_ROOT="$(BUILD_ROOT)"
