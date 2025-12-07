import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import debugpy
import numpy as np
import torch
from aim import Run as AimRun, Text as AimText
from rich import traceback
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pandas as pd
from transformers import (
    AutoTokenizer,  # type: ignore
)
from transformers.models.bart import BartTokenizer
from transformers.models.bert import BertModel, BertTokenizer
import wandb

from multihopkg.datasets import GraphEmbeddingDataset
from multihopkg import data_utils
from multihopkg.logging import setup_logger
from multihopkg.models_language.classical import HunchBart
from multihopkg.run_configs.pretraining import get_args
from multihopkg.data_utils import load_native_index
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds
from multihopkg.utils.vis import CustomProgress
from multihopkg.utils.schedulers import WarmupCosineScheduler

# import nice traceback from rich
traceback.install()

CACHED_DATA_COLUMNS = ["enc_questions", "enc_answer", "triples_ints"]

wandb_on = False

def collate_fn(batch, padding_value: int):
    batch_deconstructed = zip(*batch)
    qna: List[torch.Tensor] = list(next(batch_deconstructed))
    ans_masks: List[torch.Tensor] = list(next(batch_deconstructed))
    paths: List[torch.Tensor] = list(next(batch_deconstructed))
    ans_bert_emb: List[torch.Tensor] = list(next(batch_deconstructed))

    qna_padded = torch.nn.utils.rnn.pad_sequence(
        qna, batch_first=True, padding_value=padding_value
    )
    ans_masks_padded = torch.nn.utils.rnn.pad_sequence(
        ans_masks, batch_first=True, padding_value=0
    )
    paths_padded = torch.nn.utils.rnn.pad_sequence(
        paths, batch_first=True, padding_value=padding_value
    )
    ans_bert_emb_final_tensor = torch.stack(ans_bert_emb)
    paths_attention_mask = ~(paths_padded == padding_value).all(dim=-1)

    new_batch = (qna_padded, ans_masks_padded, paths_padded, paths_attention_mask, ans_bert_emb_final_tensor)

    return new_batch

def collate_wrapper(pad_value:int) -> Callable:
    def _collate_fn(batch):
        return collate_fn(batch, pad_value)
    return _collate_fn


def _prepare_question_prompts(
    qna_tokens: torch.Tensor,
    ans_masks: torch.Tensor,
    pad_token_id: int,
    bos_token_id: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Extract only the question portion (prior to the first answer token) to use as decoder prompts.
    """
    prompts: List[torch.Tensor] = []
    prompt_lengths: List[int] = []
    device = qna_tokens.device
    seq_len = qna_tokens.shape[1]

    for seq, mask in zip(qna_tokens, ans_masks):
        mask_list = mask.tolist()
        seq_list = seq.tolist()
        try:
            answer_start = mask_list.index(1)
        except ValueError:
            answer_start = len(seq_list)
        answer_start = min(answer_start, seq_len)
        # prompt = [token for token in seq_list[:answer_start] if token != pad_token_id and token != bos_token_id and token != eos_token_id]
        prompt = [token for token in seq_list[:answer_start] if token != pad_token_id]
        if not prompt:
            raise ValueError("Was expecting a prompt in evaluation")
        prompt.append(bos_token_id)
        prompts.append(torch.tensor(prompt, dtype=torch.long, device=device))
        prompt_lengths.append(len(prompt))

    decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
        prompts, batch_first=True, padding_value=pad_token_id
    )
    decoder_attention_mask = (decoder_input_ids != pad_token_id).long()

    return decoder_input_ids, decoder_attention_mask, prompt_lengths


def _run_generation_evaluation(
    model: nn.Module,
    tokenizer: BartTokenizer,
    qna_tokens: torch.Tensor,
    ans_masks: torch.Tensor,
    graph_embeddings: torch.Tensor,
    graphemb_attn_mask: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    """
    Run BART.generate() to obtain free-form answers and compute simple sequence-level metrics.
    """
    if not hasattr(model, "bart") or not hasattr(model, "embedding_translator"):
        return None

    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int), "pad_token_id must be defined for generation evaluation"
    decoder_input_ids, decoder_attention_mask, prompt_lengths = _prepare_question_prompts(
        qna_tokens, ans_masks, pad_token_id, tokenizer.bos_token_id,
    )

    translated_embeddings = model.embedding_translator(graph_embeddings)
    encoder_attention_mask = graphemb_attn_mask.long()
    max_prompt_len = max(prompt_lengths) if prompt_lengths else 0
    max_generation_len = max(qna_tokens.shape[1], max_prompt_len + 1)

    generated_ids = model.bart.generate(  # type: ignore[attr-defined]
        inputs_embeds=translated_embeddings,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        min_length=0,
        # max_length=max_generation_len,
        max_new_tokens=20, # Out of lazyness. 
        num_beams=3,
    )

    eos_token_id = tokenizer.eos_token_id
    predictions: List[str] = []
    references: List[str] = []
    question_texts: List[str] = []
    generated_lengths: List[int] = []

    batch_size = generated_ids.size(0)
    for idx in range(batch_size):
        prompt_len = prompt_lengths[idx]
        generated_seq = generated_ids[idx]
        answer_tokens = generated_seq[prompt_len:]

        trimmed_tokens: List[int] = []
        for token_id in answer_tokens.tolist():
            if eos_token_id is not None and token_id == eos_token_id:
                break
            if token_id == pad_token_id:
                continue
            trimmed_tokens.append(token_id)

        generated_lengths.append(len(trimmed_tokens))
        predictions.append(tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip())
        ref_answer_tokens = qna_tokens[idx][ans_masks[idx] == 1]
        references.append(tokenizer.decode(ref_answer_tokens, skip_special_tokens=True).strip())
        question_prompt = decoder_input_ids[idx][:prompt_len]
        question_texts.append(tokenizer.decode(question_prompt, skip_special_tokens=True).strip())

    exact_match = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    avg_generated_len = (
        sum(generated_lengths) / len(generated_lengths) if generated_lengths else 0.0
    )

    samples = []
    for question, pred, ref in zip(question_texts, predictions, references):
        if len(samples) >= 3:
            break
        samples.append(
            {
                "question": question,
                "prediction": pred,
                "reference": ref,
            }
        )

    return {
        "count": batch_size,
        "exact_match": exact_match,
        "avg_generated_length": avg_generated_len,
        "samples": samples,
    }

def validation_loop(
    model: nn.Module,
    val_dataloader: DataLoader,
    tokenizer: BartTokenizer,
    verbose: bool,
    aim_run: Optional[AimRun],
    global_step: int,
) -> Dict[str, float]:
    # TODO: Implement some other more sophisticated validation metrics
    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int), "Expected the pad token to be an integer. Instead we get {pad_token_id}"
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    validation_metrics: Dict[str, List[float]] = {
        "valid/loss" : [],
        "valid/cf-loss" : [],
        "valid/alignment_loss_w_emb" : [],
        "valid/alignment_loss_wo_emb" : [],
        "valid/exact_match": [],
        "valid/avg_generated_length": [],
    }
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Turn of all backprop
            qna_tokens, ans_masks, graph_embeddings, graphemb_attn_mask, answer_bert_emb = batch
            # Now we will round-robin graph_embeddings to get a negative sample. 
            negative_graph_embeddings = torch.roll(graph_embeddings, shifts=1, dims=0)

            padding_mask = qna_tokens != tokenizer.pad_token_id

            truth_answers = qna_tokens.clone()
            truth_answers[ans_masks == 0] = tokenizer.pad_token_id  # For the loss function.
            truth_answers = truth_answers[:, 1:].contiguous()

            # Question Mask
            questions_masks = (ans_masks == 0) & (padding_mask)
            # questions_lens = torch.sum(question_mask, dim = 1)

            # Compute the loss
            answers_inf_softmax_w_emb, bert_alignment_inference_w_emb = model(
                graph_embeddings, graphemb_attn_mask, qna_tokens[:,:-1], decoder_attention_mask=padding_mask[:,:-1], questions_masks=questions_masks[:,:-1],
            )
            answers_inf_softmax_wo_emb, bert_alignment_inference_wo_emb = model(
                negative_graph_embeddings, graphemb_attn_mask, qna_tokens[:,:-1], decoder_attention_mask=padding_mask[:,:-1], questions_masks=questions_masks[:,:-1],
            )
            _, logits = answers_inf_softmax_w_emb.loss, answers_inf_softmax_w_emb.logits
            _, n_logits = answers_inf_softmax_wo_emb.loss, answers_inf_softmax_wo_emb.logits

            # Computer Bert Alignment Loss
            alignment_loss_w_emb = F.mse_loss(bert_alignment_inference_w_emb, answer_bert_emb)
            alignment_loss_wo_emb = F.mse_loss(bert_alignment_inference_wo_emb, answer_bert_emb)
            validation_metrics["valid/alignment_loss_w_emb"].append(alignment_loss_w_emb.item())
            validation_metrics["valid/alignment_loss_wo_emb"].append(alignment_loss_wo_emb.item())

            # Loss Calculation
            loss = loss_fn(logits.view(-1, logits.shape[-1]), truth_answers.view(-1)).mean()
            n_loss = loss_fn(n_logits.view(-1, n_logits.shape[-1]), truth_answers.view(-1)).mean()

            validation_metrics["valid/loss"].append(loss.item())
            validation_metrics["valid/cf-loss"].append(n_loss.item())
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

            generation_report = _run_generation_evaluation(
                model,
                tokenizer,
                qna_tokens,
                ans_masks,
                graph_embeddings,
                graphemb_attn_mask,
            )
            if generation_report is not None:
                count = max(generation_report["count"], 1)
                validation_metrics["valid/exact_match"].append(
                    generation_report["exact_match"] / count
                )
                validation_metrics["valid/avg_generated_length"].append(
                    generation_report["avg_generated_length"]
                )
                if verbose and generation_report["samples"]:
                    for sample in generation_report["samples"]:
                        logger.debug(
                            f"[GEN] Q: {sample['question']} | Pred: {sample['prediction']} | Ref: {sample['reference']}"
                        )
                if aim_run is not None and generation_report["samples"]:
                    for sample in generation_report["samples"]:
                        sample_text = (
                            f"Q: {sample['question']} | "
                            f"Pred: {sample['prediction']} | "
                            f"Ref: {sample['reference']}"
                        )
                        aim_run.track(
                            AimText(sample_text),
                            name="valid/generation_samples",
                            step=global_step,
                        )
    model.train()
    _validation_metrics = {}
    for k,v in  validation_metrics.items():
        _validation_metrics[k] = torch.mean(torch.tensor(v)).item() # eww
    return _validation_metrics
    

def train_loop(
    dataset_partitions: DataPartitions,
    word_tokenizer: BartTokenizer,
    bart_llm: nn.Module,
    entity_embeddings: nn.Embedding,
    relation_embeddings: nn.Embedding,
    bert_emb_size: int,
    # --- Training Parameters --- #
    batch_size: int,
    epochs: int,
    baseline_lr: float,
    minimum_lr: float,
    num_warmup_steps: int,
    # --- Validation Parameters -- #
    val_every_n_batches: int,
    verbose: bool,
    aim_run: Optional[AimRun],
) -> nn.Module:
    device = next(bart_llm.parameters()).device
    ########################################
    # Data Loading
    ########################################
    pad_token_id = word_tokenizer.pad_token_id
    assert isinstance(pad_token_id, int), "Expected the pad token to be an integer. Instead we get {pad_token_id}"
    train_dataset = GraphEmbeddingDataset(dataset_partitions.train, entity_embeddings, relation_embeddings, word_tokenizer, bert_emb_size,device)
    train_dataloader = DataLoader(train_dataset, batch_size, collate_fn=collate_wrapper(pad_token_id))
    # Validation
    val_dataset = GraphEmbeddingDataset(dataset_partitions.validation, entity_embeddings, relation_embeddings, word_tokenizer, bert_emb_size, device)
    val_dataloader = DataLoader(val_dataset, batch_size//2, collate_fn=collate_wrapper(pad_token_id))

    # DEBUG:: to check if the embeddings are being changed.
    ent_emb_backup = entity_embeddings.weight.clone()

    train_ds_size = len(train_dataset)
    logger.info(f"We are training with a dataset of size: {train_ds_size}")
    assert train_dataset is not None, "train_data empty in DataPartitions"
    logger.info(f"With a batch size of {batch_size} this will yield {len(train_dataloader)} batches")

    # Optimization parameters
    optimizer = torch.optim.Adam(bart_llm.parameters(), lr=baseline_lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)

    total_steps = epochs * len(train_dataloader)
    logger.debug(f"Total steps: {total_steps}")
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=num_warmup_steps, total_steps=total_steps, min_lr=minimum_lr)

    loss_reports = []
    validation_reports: List[Tuple[int, Any]] = []
    cur_num_batches = 0

    # with CustomProgress(column_names=["Train Loss", "Val  Loss", "lr_rate"],table_max_rows=10) as progress:
        # task_epoch = progress.add_task("Epochs", total=epochs)
    for e in range(epochs):
            # task_batch = progress.add_task("Batch", total=len(train_dataloader))
        for idx_batch,batch in enumerate(train_dataloader):

            # Validation
            if cur_num_batches % val_every_n_batches == 0:
                val_report = validation_loop(
                    bart_llm,
                    val_dataloader,
                    word_tokenizer,
                    verbose,
                    aim_run,
                    cur_num_batches,
                )
                validation_reports.append((
                    cur_num_batches,
                    val_report,
                ))
                if wandb_on:
                    wandb.log(val_report)
                if aim_run is not None:
                    for metric_name, metric_value in val_report.items():
                        aim_run.track(
                            metric_value,
                            name=metric_name,
                            step=cur_num_batches,
                        )

            cur_num_batches += 1

            # Actual Training
            qna_tokens, ans_masks, graph_embeddings, graphemb_attention_mask, ans_bert_embeddings = batch
            truth_answers = qna_tokens.clone()
            truth_answers[ans_masks == 0] = word_tokenizer.pad_token_id  # For the loss function.
            truth_answers = truth_answers[:, 1:].contiguous()

            # Compute the loss
            optimizer.zero_grad()
            # TODO: Watch out for offset*till
            padding_mask = qna_tokens != word_tokenizer.pad_token_id
            questions_masks = (ans_masks == 0) & (padding_mask)
            answers_inf_softmax, bert_output = bart_llm(
                graph_embeddings, 
                graphemb_attention_mask,
                qna_tokens[:,:-1],
                decoder_attention_mask=padding_mask[:,:-1],
                questions_masks=questions_masks[:,:-1],
            )
            _, logits = answers_inf_softmax.loss, answers_inf_softmax.logits


            # Graph Projector Backprop
            gtllm_loss = loss_fn(logits.view(-1, logits.shape[-1]), truth_answers.view(-1)).mean()
            bert_loss = F.mse_loss(bert_output, ans_bert_embeddings)
            final_loss = gtllm_loss + bert_loss
            final_loss.backward()
            optimizer.step()
            # scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            loss_reports.append(gtllm_loss.item())

            if wandb_on:
                wandb_payload = {
                    "loss_bart_train": gtllm_loss.item(),
                    "loss_bert_train": bert_loss.item(),
                    "final_loss_train": final_loss.item()
                }
                wandb.log(wandb_payload)
            if aim_run is not None:
                step_id = cur_num_batches
                aim_run.track(e, name="train/epoch", step=step_id)
                aim_run.track(gtllm_loss.item(), name="train/loss_bart_train", step=step_id)
                aim_run.track(bert_loss.item(), name="train/loss_bert_train", step=step_id)
                aim_run.track(final_loss.item(), name="train/final_loss_train", step=step_id)
                aim_run.track(current_lr, name="train/lr", step=step_id)

            # Check for changes
            change_in_embeddings = torch.dist(ent_emb_backup, train_dataset.id2ent.weight).sum()
            logger.debug(f"Difference in embedding sizes: {change_in_embeddings}")
            grad = train_dataset.id2ent.weight.grad
            logger.debug(f"Repoerting on gradient of embedding: {grad}")

            table_reports = (f"{loss_reports[-1]}", f"{validation_reports[-1][-1]}", f"{current_lr}")
            # progress.update_table(table_reports)
            # progress.update(task_batch, advance=1)
            # time.sleep(0.1)
        # progress.update(task_epoch, advance=1)

    return bart_llm

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
            name=f"{args.run_name}-{timestamp}",
            notes=args.wr_notes
        )
    wandb_on = args.wandb_on
    aim_run = AimRun(experiment=args.aim_experiment)
    run_name = args.run_name if args.run_name is not None else "gtllm_pretraining"
    aim_run.name = run_name
    aim_run["hparams"] = vars(args)

    ########################################
    # Process the NLP components
    ########################################
    word_tokenizer = AutoTokenizer.from_pretrained(args.hunchbart_base_llm_tokenizer)

    ########################################
    # Load Embedding Data
    ########################################
    path_entities_dict = os.path.join(args.path_mquake_data, "entities.dict")
    path_relations_dict = os.path.join(args.path_mquake_data, "relations.dict")
    id2ent, ent2id = load_native_index(path_entities_dict)
    id2rel, rel2id = load_native_index(path_relations_dict)
    logger.info(f"Loaded a total of :\n\t-{len(id2ent)} entities\n\t-{len(id2rel)} relations")
    
    ########################################
    # Load Bert
    ########################################
    # Bert Ground Truth For Alignment
    bert_model = BertModel.from_pretrained(
        args.bert_base_llm_model,
    ).to(args.device)
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_base_llm_tokenizer
    )
    bert_emb_size = bert_model.config.hidden_size
    

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
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
        force_recompute=args.force_recompute_cache,
        supervised=False
    )
    dataset_partitions = DataPartitions(
        train_df,
        dev_df,
        test_df
    )

    ########################################
    # Expand training exposure
    ########################################
    previous_train_size = len(dataset_partitions.train)
    dataset_partitions.train = pd.concat(
        [dataset_partitions.train, dataset_partitions.validation, dataset_partitions.test],
        ignore_index=True,
    )
    logger.info(
        "Expanded training split from %d to %d examples by mixing all partitions",
        previous_train_size,
        len(dataset_partitions.train),
    )

    ########################################
    # Load Pretrained Embeddings
    ########################################
    # TODO:
    embedds_dir = args.path_graph_emb_data
    entity_embeddings = nn.Embedding.from_pretrained(
        torch.from_numpy(np.load(os.path.join(embedds_dir, "entity_embedding.npy")))
    )
    relation_embeddings = nn.Embedding.from_pretrained(
        torch.from_numpy(np.load(os.path.join(embedds_dir, "relation_embedding.npy")))
    )
    with open(os.path.join(embedds_dir, "config.json"), 'r') as f:
        graph_embed_training_metadata = json.load(f)

    # Model Hyperparameters
    assert entity_embeddings.weight.shape[-1] == relation_embeddings.weight.shape[-1], "Relation and Embedding Dimensions are different. Assumption broken. Exiting"
    embeddings_size = entity_embeddings.weight.shape[-1]

    # Import the HunchBart Parameter
    hunch_llm = HunchBart(
        pretrained_bart_model_name=args.hunchbart_base_llm_model,
        graph_embedding_dim=embeddings_size,
        tokenizer=word_tokenizer
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
        bert_emb_size,
        args.batch_size,
        args.epochs,
        args.baseline_lr,
        args.minimum_lr,
        args.num_warmup_steps,
        args.val_every_n_batches,
        args.verbose,
        aim_run,
    )
    aim_run.close()

    logger.info("Training Finsihed")
    # Save the model under ./models/gtllm/date/
    timestamp = time.strftime("%m%d%Y_%H%M%S", time.localtime())
    run_name = "gtllm_"+args.run_name if args.run_name is not None else "gtllm"
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
