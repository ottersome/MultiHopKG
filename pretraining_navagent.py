import json
import os
from typing import Any, Callable, Dict, List, Tuple
import time

import debugpy
import numpy as np
import torch
from rich import traceback
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,  # type: ignore
)
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.bart import BartTokenizer
import wandb

from multihopkg.datasets import GraphEmbeddingDataset
from multihopkg import data_utils
from multihopkg.exogenous.sun_models import KGEModel
from multihopkg.logging import setup_logger
from multihopkg.models_language.classical import HunchBart
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.run_configs.pretraining import get_args
from multihopkg.data_utils import load_native_index
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds
from multihopkg.utils.vis import CustomProgress
from multihopkg.utils.schedulers import WarmupCosineScheduler
from multihopkg.vector_search import ANN_IndexMan, ANN_IndexMan_AbsClass, ANN_IndexMan_pRotatE

# import nice traceback from rich
traceback.install()

CACHED_DATA_COLUMNS = ["enc_questions", "enc_answer", "triples_ints"]

wandb_on = False

def collate_fn(batch, padding_value: int):
    batch_deconstructed = zip(*batch)
    qna: List[torch.Tensor] = list(next(batch_deconstructed))
    ans_masks: List[torch.Tensor] = list(next(batch_deconstructed))
    paths: List[torch.Tensor] = list(next(batch_deconstructed))

    qna_padded = torch.nn.utils.rnn.pad_sequence(
        qna, batch_first=True, padding_value=padding_value
    )
    ans_masks_padded = torch.nn.utils.rnn.pad_sequence(
        ans_masks, batch_first=True, padding_value=0
    )
    paths_padded = torch.nn.utils.rnn.pad_sequence(
        paths, batch_first=True, padding_value=padding_value
    )

    new_batch = (qna_padded, ans_masks_padded, paths_padded)

    return new_batch

def collate_wrapper(pad_value:int) -> Callable:
    def _collate_fn(batch):
        return collate_fn(batch, pad_value)
    return _collate_fn

def validation_loop(
    model: nn.Module,
    val_dataloader: DataLoader,
    tokenizer: BartTokenizer,
    verbose: bool,
) -> Dict[str, float]:
    # TODO: Implement some other more sophisticated validation metrics
    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int), "Expected the pad token to be an integer. Instead we get {pad_token_id}"
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    validation_metrics: Dict[str, List[float]] = {
        "loss" : [],
        "cf-loss" : []
    }
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Turn of all backprop
            qna_tokens, ans_masks, graph_embeddings = batch
            # Now we will round-robin graph_embeddings to get a negative sample. 
            negative_graph_embeddings = torch.roll(graph_embeddings, shifts=1, dims=0)

            padding_mask = qna_tokens == tokenizer.pad_token_id

            truth_answers = qna_tokens.clone()
            truth_answers[ans_masks == 0] = tokenizer.pad_token_id  # For the loss function.
            truth_answers = truth_answers[:, 1:].contiguous()

            # Compute the loss
            answers_inf_softmax_w_emb = model(
                graph_embeddings, qna_tokens[:,:-1], decoder_attention_mask=padding_mask[:,:-1]
            )
            answers_inf_softmax_wo_emb = model(
                negative_graph_embeddings, qna_tokens[:,:-1], decoder_attention_mask=padding_mask[:,:-1]
            )
            _, logits = answers_inf_softmax_w_emb.loss, answers_inf_softmax_w_emb.logits
            _, n_logits = answers_inf_softmax_wo_emb.loss, answers_inf_softmax_wo_emb.logits

            # Loss Calculation
            loss = loss_fn(logits.view(-1, logits.shape[-1]), truth_answers.view(-1)).mean()
            n_loss = loss_fn(n_logits.view(-1, n_logits.shape[-1]), truth_answers.view(-1)).mean()

            validation_metrics["loss"].append(loss.item())
            validation_metrics["cf-loss"].append(n_loss.item())
            if verbose and batch_idx == 0:
                # Take logits and covert them into idxs:
                qna_strs = tokenizer.batch_decode(qna_tokens)
                inference_ids = logits.argmax(dim=-1)
                ninference_ids = n_logits.argmax(dim=-1)
                inference_strs = [
                    tokenizer.decode(elem[ans_masks[idx, 1:] == 1])
                    for idx,elem in enumerate(inference_ids)
                ]
                ninference_strs = [
                    tokenizer.decode(elem[ans_masks[idx, 1:] == 1])
                    for idx,elem in enumerate(ninference_ids)
                ]
                true_strs = [
                    tokenizer.decode(elem[ans_masks[idx, 1:] == 1])
                    for idx,elem in enumerate(truth_answers)
                ]
                # inference_strs = tokenizer.batch_decode(inference_ids)
                logger.debug(f"For this batch ({batch_idx}) of validation. We end up with the metrics\n")
                for q, n, i,a in zip(qna_strs, ninference_strs, inference_strs, true_strs):
                    # logger.debug(f"\n\t- Q: {q}\n\t- I: {i}")
                    logger.debug(f"\n\t- Q: {q}\n\t - A:{a}\n\t - I: {i}\n\t - F: {n}\n")
                logger.debug(f"CounterFactual ration {loss/n_loss}")
                logger.debug("----------------------------------------\n\n")
    model.train()
    _validation_metrics = {}
    for k,v in  validation_metrics.items():
        _validation_metrics[k] = torch.mean(torch.tensor(v)).item() # eww
    return _validation_metrics
    

def train_loop(
    dataset_partitions: DataPartitions,
    word_tokenizer: BartTokenizer,
    model: nn.Module,
    entity_embeddings: nn.Embedding,
    relation_embeddings: nn.Embedding,
    # --- Training Parameters --- #
    batch_size: int,
    epochs: int,
    baseline_lr: float,
    minimum_lr: float,
    num_warmup_steps: int,
    # --- Validation Parameters -- #
    val_every_n_batches: int,
    verbose: bool,
) -> nn.Module:
    device = next(model.parameters()).device
    ########################################
    # Data Loading
    ########################################
    pad_token_id = word_tokenizer.pad_token_id
    assert isinstance(pad_token_id, int), "Expected the pad token to be an integer. Instead we get {pad_token_id}"
    train_dataset = GraphEmbeddingDataset(dataset_partitions.train, entity_embeddings, relation_embeddings, word_tokenizer, device)
    train_dataloader = DataLoader(train_dataset, batch_size, collate_fn=collate_wrapper(pad_token_id))
    # Validation
    val_dataset = GraphEmbeddingDataset(dataset_partitions.validation, entity_embeddings, relation_embeddings, word_tokenizer, device)
    val_dataloader = DataLoader(val_dataset, batch_size, collate_fn=collate_wrapper(pad_token_id))

    # DEBUG:: to check if the embeddings are being changed.
    ent_emb_backup = entity_embeddings.weight.clone()

    train_ds_size = len(train_dataset)
    logger.info(f"We are training with a dataset of size: {train_ds_size}")
    assert train_dataset is not None, "train_data empty in DataPartitions"
    logger.info(f"With a batch size of {batch_size} this will yield {len(train_dataloader)} batches")

    # Optimization parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=baseline_lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)

    total_steps = epochs * len(train_dataloader)
    logger.debug(f"Total steps: {total_steps}")
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=num_warmup_steps, total_steps=total_steps, min_lr=minimum_lr)

    loss_reports = []
    validation_reports: List[Tuple[int, Any]] = []
    cur_num_batches = 0

    with CustomProgress(column_names=["Train Loss", "Val  Loss", "lr_rate"],table_max_rows=10) as progress:
        task_epoch = progress.add_task("Epochs", total=epochs)
        for e in range(epochs):
            task_batch = progress.add_task("Batch", total=len(train_dataloader))
            for idx_batch,batch in enumerate(train_dataloader):

                # Validation
                if cur_num_batches % val_every_n_batches == 0:
                    val_report = validation_loop(model, val_dataloader, word_tokenizer, verbose)
                    validation_reports.append((
                        cur_num_batches,
                        val_report,
                    ))
                    if wandb_on:
                        wandb.log(val_report)

                cur_num_batches += 1

                # Actual Training
                qna_tokens, ans_masks, graph_embeddings = batch
                truth_answers = qna_tokens.clone()
                truth_answers[ans_masks == 0] = word_tokenizer.pad_token_id  # For the loss function.
                truth_answers = truth_answers[:, 1:].contiguous()
                # Compute the loss
                optimizer.zero_grad()
                # TODO: Watch out for offset*till
                padding_mask = qna_tokens != word_tokenizer.pad_token_id
                answers_inf_softmax = model(
                    graph_embeddings, qna_tokens[:,:-1], decoder_attention_mask=padding_mask[:,:-1]
                )
                _, logits = answers_inf_softmax.loss, answers_inf_softmax.logits

                loss = loss_fn(logits.view(-1, logits.shape[-1]), truth_answers.view(-1)).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_reports.append(loss.item())

                if wandb_on:
                    wandb.log({"loss_train": loss.item()})

                # Check for changes
                change_in_embeddings = torch.dist(ent_emb_backup, train_dataset.id2ent.weight).sum()
                logger.debug(f"Difference in embedding sizes: {change_in_embeddings}")
                grad = train_dataset.id2ent.weight.grad
                logger.debug(f"Repoerting on gradient of embedding: {grad}")

                table_reports = (f"{loss_reports[-1]}", f"{validation_reports[-1][-1]}", f"{scheduler.get_lr()}")
                progress.update_table(table_reports)
                progress.update(task_batch, advance=1)
                time.sleep(0.1)
            progress.update(task_epoch, advance=1)

    return model

def get_ann_index_managers(
    model_type: str,
    entity_embedding_tensor: torch.Tensor,
    relation_embedding_tensor: torch.Tensor,
    embedding_range: float
    ) -> Tuple[ANN_IndexMan_AbsClass, ANN_IndexMan_AbsClass]:

    if model_type == "pRotatE": # for rotational kge models
        ann_index_manager_ent = ANN_IndexMan_pRotatE(
            entity_embedding_tensor,
            embedding_range=embedding_range,
        )
        ann_index_manager_rel = ANN_IndexMan_pRotatE(
            relation_embedding_tensor,
            embedding_range=embedding_range,
        )
    else: # for non-rotational kge models
        ann_index_manager_ent = ANN_IndexMan(
            entity_embedding_tensor,
            exact_computation=True,
            nlist=100,
        )
        ann_index_manager_rel = ANN_IndexMan(
            entity_embedding_tensor,
            exact_computation=True,
            nlist=100,
        )
    return ann_index_manager_ent, ann_index_manager_rel

def main():
    args = get_args()
    set_seeds(args.seed)
    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()

    global wandb_on
    if args.wandb_on:
        timestamp = time.strftime("%m%d%Y_%H%M%S", time.localtime())
        wandb.init(
            project=f"{args.wandb_project}",
            config=vars(args),
            name=f"{args.wr_name}-{timestamp}",
            notes=args.wr_notes
        )
    wandb_on = args.wandb_on

    ########################################
    # Process the NLP components
    ########################################
    word_tokenizer = AutoTokenizer.from_pretrained(args.hunchbart_base_llm_tokenizer)
    question_embedding_module = AutoModel.from_pretrained(args.question_embedding_model).to(args.device)

    ########################################
    # Load Embedding Data
    ########################################
    path_entities_dict = os.path.join(args.path_mquake_data, "entities.dict")
    path_relations_dict = os.path.join(args.path_mquake_data, "relations.dict")
    id2ent, ent2id = load_native_index(path_entities_dict)
    id2rel, rel2id = load_native_index(path_relations_dict)
    logger.info(f"Loaded a total of :\n\t-{len(id2ent)} entities\n\t-{len(id2rel)} relations")

    ########################################
    # Process the Dataset
    ########################################
    raw_mquake_csv_data_path = os.path.join(args.path_mquake_data, "mquake_qna_ds.csv")
    meta_data_path = os.path.join(args.path_cache_dir, "mquake.json")
    logger.info(
        f"Loading the data from {meta_data_path}." + \
        str("\n\t Will be forcing recompute" if args.force_recompute_cache else "")
    )
    train_df, dev_df, test_df, _ = data_utils.load_qa_data(
        cached_metadata_path=meta_data_path,
        raw_QAData_path=raw_mquake_csv_data_path,
        question_tokenizer_name=args.hunchbart_base_llm_tokenizer,
        answer_tokenizer_name=args.hunchbart_base_llm_tokenizer,
        entity2id=ent2id,
        relation2id=rel2id,
        logger=logger,
        force_recompute=args.force_recompute_cache,
        supervised=False
    )
    dataset_partitions = DataPartitions(
        train_df,
        dev_df,
        test_df
    )

    ########################################
    # Load Pretrained Embeddings
    ########################################
    # TODO:
    embedds_dir = args.path_graph_emb_data
    entity_embeddings_np = np.load(os.path.join(embedds_dir, "entity_embedding.npy"))
    entity_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(entity_embeddings_np))
    relation_embeddings_np = np.load(os.path.join(embedds_dir, "relation_embedding.npy"))
    relation_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(relation_embeddings_np))
    checkpoint = torch.load(os.path.join(embedds_dir, "checkpoint"))
    config = json.load(open(os.path.join(embedds_dir, "config.json"), 'r'))
    graph_embed_geom = config["model"]

    with open(os.path.join(embedds_dir, "config.json"), 'r') as f:
        graph_embed_training_metadata = json.load(f)
    entity_dim = entity_embeddings.weight.shape[-1]
    relation_dim = relation_embeddings.weight.shape[-1]

    ########################################
    # Now load the KGE Model
    ########################################
    kge_model = KGEModel.from_pretrained(
        model_name=args.model,
        entity_embedding=entity_embeddings_np,
        relation_embedding=relation_embeddings_np,
        gamma=args.gamma,
        state_dict=checkpoint["model_state_dict"]
    )

    # Model Hyperparameters
    assert entity_embeddings.weight.shape[-1] == relation_embeddings.weight.shape[-1], "Relation and Embedding Dimensions are different. Assumption broken. Exiting"
    embeddings_size = entity_embeddings.weight.shape[-1]

    ########################################
    # Index Managers
    ########################################
    ann_index_manager_ent, ann_index_manager_rel = get_ann_index_managers(
        model_type=graph_embed_geom,
        entity_embedding_tensor=kge_model.get_all_entity_embeddings_wo_dropout(),
        relation_embedding_tensor=kge_model.get_all_relations_embeddings_wo_dropout(),
        embedding_range=kge_model.embedding_range.item()
    )

    ########################################
    # Environment

    environment = ITLGraphEnvironment(
        question_embedding_module = question_embedding_module,
        question_embedding_module_trainable = args.question_embedding_module_trainable,
        entity_dim = entity_dim,
        ff_dropout_rate = args.ff_dropout_rate,
        history_dim = 0, # No longer in use. Set to 0 to avoid errors
        history_num_layers = 0, # No longer in use. Set to 0 to avoid errors
        knowledge_graph = kge_model,
        relation_dim = relation_dim,
        nav_start_emb_type = args.nav_start_emb_type,
        node_data = None,
        node_data_key = None,
        rel_data = None,
        rel_data_key = None,
        # THe following four are also no longer being used
        id2entity = dict(),
        entity2id = dict(),
        id2relation = dict(),
        relation2id = dict(),
        ann_index_manager_ent = ann_index_manager_ent,
        ann_index_manager_rel = ann_index_manager_rel,
        steps_in_episode = args.num_rollout_steps,
        # Also stopped being used
        trained_pca = None,
        graph_pca = None,
        graph_annotation = "", # No longer in use. Set to "" to avoid errors
        num_rollouts = args.num_rollouts,
        use_kge_question_embedding = False, # This is only true for Eduin's supervised approach
        epsilon = args.nav_epsilon_error,
        add_transition_state = args.add_transition_state,
    )

    # Import the HunchBart Parameter
    navigator = ContinuousPolicyGradient(
        beta = args.beta,
        gamma = args.gamma,
        dim_action = dim_relation,
        dim_hidden = args.cpg_hidden,
        dim_observation = int,  # i.e input dim
        log_std_min = -20,
        log_std_max = 2,
    )
    nav_agent = ContinuousPolicyGradient(
        beta=args.beta,
        gamma=args.rl_gamma,
        dim_action=dim_relation,
        dim_hidden=args.rnn_hidden,
        dim_observation=2*dim_entity + dim_entity,  # observation will be into history
    ).to(args.device)
    
    # Freeze the BART model, keep embedding_translator trainable
    hunch_llm.freeze_bart()

    logger.info("Entering training loop")
    trained_model = train_loop(
        dataset_partitions,
        word_tokenizer,
        hunch_llm,
        entity_embeddings,
        relation_embeddings,
        args.batch_size,
        args.epochs,
        args.baseline_lr,
        args.minimum_lr,
        args.num_warmup_steps,
        args.val_every_n_batches,
        args.verbose,
    )

    logger.info("Training Finsihed")
    # Save the model under ./models/gtllm/date/
    timestamp = time.strftime("%m%d%Y_%H%M%S", time.localtime())
    run_name = "gtllm_"+args.wr_name if args.wr_name is not None else "gtllm"
    model_path = os.path.join(args.outPath_save_model, f"{run_name}_{timestamp}.pt")
    print(f"The model_path directory is {os.path.dirname(model_path)}")
    os.makedirs(os.path.dirname(model_path), exist_ok = True)

    save_info = {
        "gtllm_state_dict" :  trained_model.state_dict(),
        "hunchbart_base_llm_tokenizer" : args.hunchbart_base_llm_tokenizer,
        "hunchbart_base_llm_model" : args.hunchbart_base_llm_model,
        "hunchbart_hidden_dim": embeddings_size,  # Which is also the graph dim 
        # Data saves 
        "path_mquake_data": args.path_mquake_data,
        "path_graph_emb_data": args.path_graph_emb_data,
        "path_pretraining_cache": args.path_cache_dir,
        # Embedding Training Metaparam
        "embedding_training_metaparam": graph_embed_training_metadata,
    }
    torch.save(save_info, model_path)

    logger.info(f"Training Finsihed. Saved training info to {model_path}")

if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_MAIN__")
    main()
