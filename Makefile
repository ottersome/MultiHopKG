
# End-to-end training and evaluation for the supported QA configurations.
# Newly trained artifacts use *_make directories so released checkpoints remain
# untouched. Set GPU=1 (or another id) when needed.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
GPU ?= 0
WANDB ?= 0
FINAL_ROOT := model/final_evaluation
BUILD_ROOT := model/.make_runs
LP_ROOT := $(FINAL_ROOT)/lp_embeddings_conve/make

KINSHIP_SEED ?= 12
METAQA_SEED ?= 42
MQUAKE_SA_SEED ?= 12
MQUAKE_MA_SEED ?= 12

KINSHIP_LP := $(LP_ROOT)/kinshiphinton_conve.tar
METAQA_LP := $(LP_ROOT)/metaqa_encoded_conve.tar
MQUAKE_SA_LP := $(LP_ROOT)/mquake_st_conve.tar
MQUAKE_MA_LP := $(LP_ROOT)/mquake_st_ma_conve.tar

KINSHIP_DIR := $(FINAL_ROOT)/kinshiphinton_make
METAQA_DIR := $(FINAL_ROOT)/metaqa_encoded_rl_make
MQUAKE_SA_DIR := $(FINAL_ROOT)/mquake_sa_rl_make
MQUAKE_MA_DIR := $(FINAL_ROOT)/mquake_ma_rl_make

KINSHIP_RL := $(KINSHIP_DIR)/s$(KINSHIP_SEED)_model.tar
METAQA_RL := $(METAQA_DIR)/s$(METAQA_SEED)_model.tar
MQUAKE_SA_RL := $(MQUAKE_SA_DIR)/s$(MQUAKE_SA_SEED)_model.tar
MQUAKE_MA_RL := $(MQUAKE_MA_DIR)/s$(MQUAKE_MA_SEED)_model.tar

ifeq ($(WANDB),1)
WANDB_FLAG := --wandb
else
WANDB_FLAG :=
endif

.PHONY: help setup prepare-lp-data train-all linkpred-all rl-all pipeline evaluate \
	linkpred-kinship linkpred-metaqa-encoded linkpred-mquake-sa linkpred-mquake-ma \
	rl-kinship rl-metaqa-encoded rl-mquake-sa rl-mquake-ma

help:
	@printf '%s\n' \
	  'make train-all GPU=0  # train and stage all link-prediction and RL checkpoints' \
	  'make prepare-lp-data  # construct full-KG LP substrate training files' \
	  'make evaluate GPU=0   # batch-evaluate the staged *_make checkpoints' \
	  'make pipeline GPU=0 WANDB=1  # train, report to W&B, then evaluate' \
	  'Optional seed overrides: KINSHIP_SEED, METAQA_SEED, MQUAKE_SA_SEED, MQUAKE_MA_SEED.'

setup:
	pip3 install -r requirements.txt

train-all: linkpred-all rl-all
prepare-lp-data:
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/KinshipHinton
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/metaqa_encoded
	$(PYTHON) scripts/prepare_lp_training_data.py --data_dir data/mquake_st

linkpred-all: prepare-lp-data linkpred-kinship linkpred-metaqa-encoded linkpred-mquake-sa linkpred-mquake-ma
rl-all: rl-kinship rl-metaqa-encoded rl-mquake-sa rl-mquake-ma
pipeline: train-all evaluate

# Every training run writes a manifest containing the actual timestamped
# model_best.tar location. The checkpoint is then moved to a stable staging path.
linkpred-kinship: prepare-lp-data
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/kinshiphinton-conve.sh --train $(GPU) $(WANDB_FLAG) --model_root_dir "$(BUILD_ROOT)/kinship/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/kinship-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/kinship-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(KINSHIP_LP)"

linkpred-metaqa-encoded: prepare-lp-data
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/metaqa_encoded_conve.sh --train $(GPU) $(WANDB_FLAG) --model_root_dir "$(BUILD_ROOT)/metaqa_encoded/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/metaqa-encoded-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/metaqa-encoded-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(METAQA_LP)"

linkpred-mquake-sa: prepare-lp-data
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/mquake_st-conve.sh --train $(GPU) $(WANDB_FLAG) --model_root_dir "$(BUILD_ROOT)/mquake_sa/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/mquake-sa-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/mquake-sa-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(MQUAKE_SA_LP)"

linkpred-mquake-ma: prepare-lp-data
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(LP_ROOT)"
	./experiment-emb.sh configs/mquake_st_ma-conve.sh --train $(GPU) $(WANDB_FLAG) --model_root_dir "$(BUILD_ROOT)/mquake_ma/lp" --run_metadata_output_path "$(BUILD_ROOT)/manifests/mquake-ma-lp.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/mquake-ma-lp.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(MQUAKE_MA_LP)"

# Staged configs point at the LP artifact built above. Removing checkpoint_path
# prevents batch_eval from inheriting a stale source-config checkpoint.
rl-kinship: linkpred-kinship
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(KINSHIP_DIR)"
	./experiment-rs-nlp.sh configs/kinshiphinton-rs-nlp-conve.sh --train $(GPU) $(WANDB_FLAG) --conve_state_dict_path "$(KINSHIP_LP)" --model_root_dir "$(BUILD_ROOT)/kinship/rl" --run_metadata_output_path "$(BUILD_ROOT)/manifests/kinship-rl.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/kinship-rl.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(KINSHIP_RL)"
	@sed -e '/^checkpoint_path=/d' -e 's|^conve_state_dict_path=.*|conve_state_dict_path="$(KINSHIP_LP)"|' configs/kinshiphinton-rs-nlp-conve.sh > "$(KINSHIP_DIR)/kinshiphinton-rs-nlp-conve.sh"

rl-metaqa-encoded: linkpred-metaqa-encoded
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(METAQA_DIR)"
	./experiment-rs-nlp.sh configs/metaqa_encoded-rs-nlp-conve.sh --train $(GPU) $(WANDB_FLAG) --conve_state_dict_path "$(METAQA_LP)" --model_root_dir "$(BUILD_ROOT)/metaqa_encoded/rl" --run_metadata_output_path "$(BUILD_ROOT)/manifests/metaqa-encoded-rl.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/metaqa-encoded-rl.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(METAQA_RL)"
	@sed -e '/^checkpoint_path=/d' -e 's|^conve_state_dict_path=.*|conve_state_dict_path="$(METAQA_LP)"|' configs/metaqa_encoded-rs-nlp-conve.sh > "$(METAQA_DIR)/metaqa_encoded-rs-nlp-conve.sh"

rl-mquake-sa: linkpred-mquake-sa
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(MQUAKE_SA_DIR)"
	./experiment-rs-nlp.sh configs/mquake_st-rs-nlp-conve.sh --train $(GPU) $(WANDB_FLAG) --conve_state_dict_path "$(MQUAKE_SA_LP)" --model_root_dir "$(BUILD_ROOT)/mquake_sa/rl" --run_metadata_output_path "$(BUILD_ROOT)/manifests/mquake-sa-rl.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/mquake-sa-rl.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(MQUAKE_SA_RL)"
	@sed -e '/^checkpoint_path=/d' -e 's|^conve_state_dict_path=.*|conve_state_dict_path="$(MQUAKE_SA_LP)"|' configs/mquake_st-rs-nlp-conve.sh > "$(MQUAKE_SA_DIR)/mquake_st-rs-nlp-conve.sh"

rl-mquake-ma: linkpred-mquake-ma
	@mkdir -p "$(BUILD_ROOT)/manifests" "$(MQUAKE_MA_DIR)"
	./experiment-rs-nlp.sh configs/mquake_st_ma-rs-nlp-conve.sh --train $(GPU) $(WANDB_FLAG) --conve_state_dict_path "$(MQUAKE_MA_LP)" --model_root_dir "$(BUILD_ROOT)/mquake_ma/rl" --run_metadata_output_path "$(BUILD_ROOT)/manifests/mquake-ma-rl.json"
	@checkpoint="$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$(BUILD_ROOT)/manifests/mquake-ma-rl.json")"; test -f "$$checkpoint"; mv -f "$$checkpoint" "$(MQUAKE_MA_RL)"
	@sed -e '/^checkpoint_path=/d' -e 's|^conve_state_dict_path=.*|conve_state_dict_path="$(MQUAKE_MA_LP)"|' configs/mquake_st_ma-rs-nlp-conve.sh > "$(MQUAKE_MA_DIR)/mquake_st_ma-rs-nlp-conve.sh"

evaluate:
	$(PYTHON) batch_eval.py --gpu $(GPU) "$(KINSHIP_DIR)" "$(METAQA_DIR)" "$(MQUAKE_SA_DIR)" "$(MQUAKE_MA_DIR)"
