from collections import deque
import json
import os
import random
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union
import time

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import Progress
import torch
from rich import traceback
from rich.console import ConsoleRenderable, Group, RichCast
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,  # type: ignore
)
from transformers.models.bart import BartTokenizer

from multihopkg.datasets import GraphEmbeddingDataset
from multihopkg.logging import setup_logger
from multihopkg.models_language.classical import HunchBart
from multihopkg.run_configs.pretraining import get_args
from multihopkg.data_utils import load_native_index, translate_and_unroll_path
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds
from multihopkg.utils.data_structures import Triplet_Str

# import nice traceback from rich
traceback.install()

CACHED_DATA_COLUMNS = ["enc_questions", "enc_answer", "triples_ints"]


def process_qa_dataset(
    raw_location: str,
    cache_location: str,
    tvt_split: list[float],
    question_tokenizer: Any,
    answer_tokenizer: Any,
    existing_entities: Dict[str, int],
    existing_relations: Dict[str, int]
) -> DataPartitions:

    print(f"Raw location of raw data is {raw_location}")
    # This will load the dataset with questions and answers
    with open(raw_location, "r") as f:
        json_file = json.load(f)

    rows = []
    _d_num_invalid_samples = 0
    for sample in json_file:
        path: List[Triplet_Str] = sample["orig"]["triples"]
        int_path = translate_and_unroll_path(path, existing_entities, existing_relations)
        if len(int_path) ==  0: 
            _d_num_invalid_samples += 1
            continue
        # For now we only sample a single question
        a_question = random.choice(sample["questions"])
        answer = sample["answer"]

        encoded_question = question_tokenizer.encode(a_question)
        encoded_answer = answer_tokenizer.encode(answer)

        rows.append(
            [
                encoded_question,
                encoded_answer,
                int_path,
            ]
        )

    logger.info(f"⚠️Number of invalid samples was {_d_num_invalid_samples}")
    
    train_len = int(len(rows)*tvt_split[0])
    left_over_perc =  tvt_split[1] / (1 - tvt_split[0])
    # Train, Test, Validation Partitions
    random.shuffle(rows)
    valid_len = int((len(rows) - train_len) * left_over_perc)
    
    casted_columns = pd.Index(CACHED_DATA_COLUMNS)
    train_ds = pd.DataFrame(rows[:train_len], columns=casted_columns)
    valid_ds = pd.DataFrame(rows[train_len: train_len + valid_len], columns=casted_columns)
    test_ds = pd.DataFrame(rows[train_len + valid_len: ], columns=casted_columns)

    # Now we save it as well
    all_ds = {
        "train.pkl": train_ds,
        "valid.pkl": valid_ds,
        "test.pkl": test_ds,
    }

    for ds_file_name, df in all_ds.items():
        path_to_save = os.path.join(cache_location, ds_file_name)
        logger.info(f"Saving the split {ds_file_name}")
        df.to_pickle(path_to_save)


    return_data_partitions = DataPartitions(train_ds, valid_ds, test_ds)

    return return_data_partitions

def load_from_cache(cache_location: str) -> DataPartitions:
    files = ["train.csv", "valid.csv", "test.csv"]
    files_path = [os.path.join(cache_location, f) for f in files]

    data_partitions = []
    for f in files_path:
        df_partitions_list= pd.read_csv(f).values
        data_partitions.append(df_partitions_list)

    return DataPartitions(*data_partitions)

def check_cache_exists(cache_location: str):
    file_names = ["train.csv", "valid.csv", "test.csv"]
    for f in file_names:
        if not os.path.exists(os.path.join(cache_location, f)):
            return False

    return True


def collate_fn(batch, padding_value: int):
    batch_deconstructed = zip(*batch)
    qna: List[torch.Tensor] = list(next(batch_deconstructed))
    paths: List[torch.Tensor] = list(next(batch_deconstructed))

    qna_padded = torch.nn.utils.rnn.pad_sequence(
        qna, batch_first=True, padding_value=padding_value
    )
    paths_padded = torch.nn.utils.rnn.pad_sequence(
        paths, batch_first=True, padding_value=padding_value
    )

    new_batch = (qna_padded, paths_padded)

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
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    validation_metrics: Dict[str, List[float]] = {
        "loss" : [],
        "cf-loss" : []
    }
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Turn of all backprop
            qna_tokens, graph_embeddings = batch
            # Compute the loss
            answers_inf_softmax = model(graph_embeddings, qna_tokens)
            _, logits = answers_inf_softmax.loss, answers_inf_softmax.logits
            loss = loss_fn(logits.view(-1, logits.shape[-1]), qna_tokens.view(-1)).mean()

            validation_metrics["loss"].append(loss.item())
            if verbose and batch_idx == 0:
                # Take logits and covert them into idxs:
                qna_strs = tokenizer.batch_decode(qna_tokens)
                inference_ids = logits.argmax(dim=-1)
                inference_strs = tokenizer.batch_decode(inference_ids)
                logger.debug(
                    "----------------------------------------\n"
                    "For this batch ({batch_idx}) of validtion. We end up with the metrics\n"
                        f"\033[1;33m(Truth)\033[0m qna_tokens: \n\t{qna_strs}\n"
                        f"\033[1;32m(Inference)\033[0m answer_: \n\t{inference_strs}\n"
                    "----------------------------------------\n\n"
                )
    model.train()
    _validation_metrics = {}
    for k,v in  validation_metrics.items():
        _validation_metrics[k] = torch.mean(torch.tensor(v)).item() # eww
    return _validation_metrics
    

class CustomProgress(Progress):
    def __init__(self, column_names: Sequence[str], table_max_rows: int, *args, **kwargs) -> None:
        self.results: Deque[Sequence[str]] = deque(maxlen=table_max_rows)
        self.column_names = column_names
        self.update_table()
        super().__init__(*args, **kwargs)

    def update_table(self, result: Optional[Tuple[str,...]] = None):
        if result is not None:
            self.results.append(result)

        table = Table()
        for cn in self.column_names:
            table.add_column(cn)

        for row_cells in self.results:
            table.add_row(*row_cells)

        self.table = table

    def get_renderable(self) -> Union[ConsoleRenderable, RichCast, str]:
        renderable = Group(self.table, *self.get_renderables())
        return renderable


def train_loop(
    dataset_partitions: DataPartitions,
    word_tokenizer: BartTokenizer,
    model: nn.Module,
    entity_embeddings: nn.Embedding,
    relation_embeddings: nn.Embedding,
    # --- Training Parameters --- #
    batch_size: int,
    epochs: int,
    learning_rate: float,
    # --- Validation Parameters -- #
    val_every_n_batches: int,
    verbose: bool,
) -> nn.Module:
    device = next(model.parameters()).device
    ########################################
    # Data Loading
    ########################################
    train_dataset = GraphEmbeddingDataset(dataset_partitions.train, entity_embeddings, relation_embeddings, word_tokenizer, device)
    train_dataloader = DataLoader(train_dataset, batch_size, collate_fn=collate_wrapper(0))
    # Validation
    val_dataset = GraphEmbeddingDataset(dataset_partitions.validation, entity_embeddings, relation_embeddings, word_tokenizer, device)
    val_dataloader = DataLoader(val_dataset, batch_size, collate_fn=collate_wrapper(0))

    train_ds_size = len(train_dataset)
    logger.info(f"We are training with a dataset of size: {train_ds_size}")
    assert train_dataset is not None, "train_data empty in DataPartitions"
    logger.info(f"With a batch size of {batch_size} this will yield {len(train_dataloader)} batches")

    # Optimization parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_reports = []
    validation_reports: List[Tuple[int, Any]] = []
    cur_num_batches = 0

    with CustomProgress(column_names=["Train Loss", "Val  Loss"],table_max_rows=10) as progress:
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

                cur_num_batches += 1

                # Actual Training
                qna_tokens, graph_embeddings = batch
                # Compute the loss
                optimizer.zero_grad()
                answers_inf_softmax = model(graph_embeddings, qna_tokens)
                _, logits = answers_inf_softmax.loss, answers_inf_softmax.logits
                loss = loss_fn(logits.view(-1, logits.shape[-1]), qna_tokens.view(-1)).mean()
                loss.backward()
                optimizer.step()
                loss_reports.append(loss.item())

                table_reports = (f"{loss_reports[-1]}", f"{validation_reports[-1][-1][-1]}")
                progress.update_table(table_reports)
                progress.update(task_batch, advance=1)
                time.sleep(0.1)
            progress.update(task_epoch, advance=1)


    # Loss reporting
    _, ax = plt.subplots()
    ax.plot(loss_reports, label="Training")
    ax.set_xlabel("Batches")
    ax.set_ylabel("Loss")
    # Report validations
    validation_x_axis = [r[0] for r in validation_reports]
    validation_y_axis = [r[1] for r in validation_reports]
    ax.plot(validation_x_axis, validation_y_axis, label="Validation")
    plt.show()


    return model

def main():
    args = get_args()
    set_seeds(args.seed)
    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()


    ########################################
    # Process the NLP components
    ########################################
    question_tokenizer = AutoTokenizer.from_pretrained(args.question_tokenizer_name)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_tokenizer_name)

    ########################################
    # Load Embedding Data
    ########################################
    id2ent, ent2id = load_native_index(args.path_entities_dict)
    id2rel, rel2id = load_native_index(args.path_relations_dict)
    logger.info(f"Loaded a total of :\n\t-{len(id2ent)} entities\n\t-{len(id2rel)} relations")

    ########################################
    # Process the Dataset
    ########################################
    cache_exists = check_cache_exists(args.path_cache_dir)
    if cache_exists and not args.force_recompute_cache:
        logger.info(f"Loading data from cache {args.path_cache_dir}")
        dataset_partitions = load_from_cache(args.path_cache_dir)
    else:
        logger.info("Either cache does not exist or is being force to be recomuted... ")
        dataset_partitions = process_qa_dataset(
            args.path_dataraw,
            args.path_cache_dir,
            args.tvt_split,
            question_tokenizer,
            answer_tokenizer,
            ent2id,
            rel2id,
        )
    assert isinstance(dataset_partitions, DataPartitions)

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
        pretrained_bart_model=args.hunch_answer_model,
        answer_tokenizer=args.answer_tokenizer_name,
        graph_embedding_dim=embeddings_size,
    ).to(args.device)

    logger.info("Entering training loop")
    trained_model = train_loop(
        dataset_partitions,
        answer_tokenizer,
        hunch_llm,
        entity_embeddings,
        relation_embeddings,
        args.batch_size,
        args.epochs,
        args.lr,
        args.val_every_n_batches,
        args.verbose
    )

    logger.info("Training Finsihed")

if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_MAIN__")
    main()
