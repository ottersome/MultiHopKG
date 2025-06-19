from collections import deque
import json
import os
import random
from math import ceil
from symbol import pass_stmt
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union
import time
import threading

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import progress
import torch
from rich import traceback
from rich.live import Live
from rich.layout import Layout
from rich.console import Console, ConsoleRenderable, Group, RichCast
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from torch import batch_norm_stats, nn
from torch.types import Device
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,  # type: ignore
)
from transformers.models.bart import BartTokenizer
from transformers.utils.dummy_pt_objects import LayoutLMv2Model

from multihopkg.logging import setup_logger
from multihopkg.models_language.classical import HunchBart
from multihopkg.run_configs.pretraining import get_args
from multihopkg.utils import data_structures
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds

# import nice traceback from rich
traceback.install()

Triplet_Str = Tuple[str, str, str]
Triplet_Int = Tuple[int, int, int]
CACHED_DATA_COLUMNS = ["enc_questions", "enc_answer", "triples_ints"]

def translate_and_unroll_path(path: List[Triplet_Str], ent2id: Dict[str, int], rel2id: Dict[str,int]) -> List[int]:

    new_path: List[int] = []
    for head,rel,tail in path:
        try:
            if len(new_path) == 0:
                new_path.extend([
                    ent2id[head],
                    rel2id[rel],
                    ent2id[tail]
                ])
            else: 
                assert ent2id[head] == new_path[-1], \
                    "We assumed that steps in paths will share head and tail in intermediate paths. Assumption broken"
                new_path.extend([
                    rel2id[rel],
                    ent2id[tail],
                ])
        except KeyError:
            return []

    return new_path

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

        # TODO: Encode the Questions
        encoded_question = question_tokenizer.encode(a_question)
        encoded_answer = answer_tokenizer.encode(answer)

        rows.append(
            [
                encoded_question,
                encoded_answer,
                int_path,
                # For debugging purposes only.
                # sample["orig"]["triples"],
                # sample["orig"]["triples_labeled"],
            ]
        )

    logger.info(f"⚠️Number of invalid samples was {_d_num_invalid_samples}")
    
    train_len = int(len(rows)*tvt_split[0])
    left_over_perc =  tvt_split[1] / (1 - tvt_split[0])
    # Train, Test, Validation Partitions
    random.shuffle(rows)
    valid_len = int((len(rows) - train_len) * left_over_perc)
    
    train_ds = pd.DataFrame(rows[:train_len], columns=CACHED_DATA_COLUMNS)
    valid_ds = pd.DataFrame(rows[train_len: train_len + valid_len], columns=CACHED_DATA_COLUMNS)
    test_ds = pd.DataFrame(rows[train_len + valid_len: ], columns=CACHED_DATA_COLUMNS)

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
    files = ["train.pkl", "valid.pkl", "test.pkl"]
    files_path = [os.path.join(cache_location, f) for f in files]

    data_partitions = []
    for f in files_path:
        df_partitions_list= pd.read_pickle(f)
        data_partitions.append(df_partitions_list)

    return DataPartitions(*data_partitions)

def check_cache_exists(cache_location: str):
    file_names = ["train.csv", "valid.csv", "test.csv"]
    for f in file_names:
        if not os.path.exists(os.path.join(cache_location, f)):
            return False

    return True

def load_dictionaries(path_entities_dict: str, path_relations_dict):
    ent2id: Dict[str, int] = {}
    id2ent: Dict[int, str] = {}
    rel2id: Dict[str, int] = {}
    id2rel: Dict[int, str] = {}

    with open(path_entities_dict, 'r') as f: 
        for line in f: 
            idx, str_id  = line.strip().split()
            ent2id[str_id] = int(idx)
            id2ent[int(idx)] = str_id

    with open(path_relations_dict, 'r') as f: 
        for line in f: 
            idx, str_id  = line.strip().split()
            rel2id[str_id] = int(idx)
            id2rel[int(idx)] = str_id

    return ent2id, id2ent, rel2id, id2rel


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

class GraphEmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        id2ent: nn.Embedding,
        id2rel: nn.Embedding,
        word_tokenizer: BartTokenizer,
        device: Device,
    ):
        self.dataset: pd.DataFrame = dataset
        sep_token = word_tokenizer.sep_token
        assert isinstance(sep_token, str), "Expected the separator token to be a string. e.g. </s>"
        self.separator_token_id = word_tokenizer.convert_tokens_to_ids([sep_token])[0]
        assert isinstance(self.separator_token_id, int), f"Expected the separator token to be an integer. Instead we get {self.separator_token_id}"
        self.device = device

        # Get questions and answers a single string but separated by some token.
        self.ques_n_ans = self._merge_questions_and_answers(
            dataset.loc[:, DataPartitions.ASSUMED_COLUMNS[0]],
            dataset.loc[:, DataPartitions.ASSUMED_COLUMNS[1]],
            self.separator_token_id,
        )
        self.path = dataset.loc[:, DataPartitions.ASSUMED_COLUMNS[2]]
        # Embeddings
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.embeddings_dim = id2ent.embedding_dim

    def _merge_questions_and_answers(self, questions: List[List[int]], answers: List[List[int]], sep_token: int) -> List[List[int]]:
        """
        Merges questions and answers into a single string, separated by a token.
        """
        assert len(questions) == len(answers), "Expected questions and answers to have the same length"
        merged_questions_answers = []

        for question, answer in zip(questions, answers):
            qna = question + [sep_token] + answer
            merged_questions_answers.append(qna)

        return merged_questions_answers

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # All of these are ids
        qna_tokens = torch.tensor(self.ques_n_ans[idx], dtype=torch.long)
        path = self.path[idx]

        entities_ids = torch.tensor(path[::2], dtype=torch.long)
        relations_ids = torch.tensor(path[1::2], dtype=torch.long)

        # Obtain the embeddings
        entities_emb = self.id2ent(entities_ids)
        relations_emb = self.id2rel(relations_ids)

        # Recreate the paths
        path_embedding = torch.zeros((len(entities_ids) + len(relations_ids), self.embeddings_dim))
        path_embedding[0::2] = entities_emb
        path_embedding[1::2] = relations_emb

        # Dump the question and answer througth the normal embedding

        return qna_tokens.to(self.device), path_embedding.to(self.device)

def collate_wrapper(pad_value:int) -> Callable:
    def _collate_fn(batch):
        return collate_fn(batch, pad_value)
    return _collate_fn

def validation_loop(model:nn.Module, val_dataloader: DataLoader) -> List[float]:
    # TODO: Implement some other more sophisticated validation metrics
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    validation_metrics = [] 
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            # Turn of all backprop
            qna_tokens, graph_embeddings = batch
            # Compute the loss
            answers_inf_softmax = model(graph_embeddings, qna_tokens)
            _, logits = answers_inf_softmax.loss, answers_inf_softmax.logits
            loss = loss_fn(logits.view(-1, logits.shape[-1]), qna_tokens.view(-1)).mean()

            validation_metrics.append(loss.item())
    return validation_metrics
    

class CustomProgress(Progress):
    def __init__(self, table_max_rows: int, column_names: Sequence[str], *args, **kwargs) -> None:
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
):
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
    num_batches = ceil(train_ds_size / batch_size)
    logger.info(f"With a batch size of {batch_size} this will yield {num_batches} batches")

    # Optimization parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_reports = []
    validation_reports: List[Tuple[int, Any]] = []
    num_batches = 0
    train_live = TrainLive(epochs, len(train_dataloader))

    with CustomProgress(column_names=["Train Loss", "Val  Loss"],table_max_rows=10) as progress:
        task_epoch = progress.add_task("Epochs", total=epochs)
        for e in range(epochs):
            task_batch = progress.add_task("Batch", total=len(train_dataloader))
            for idx_batch,batch in enumerate(train_dataloader):

                # Validation
                if num_batches % val_every_n_batches == 0:
                    validation_reports.append((
                            num_batches,
                            validation_loop(model, val_dataloader),
                    ))

                num_batches += 1

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

                # logger.info(f"Batch loss-reports {loss_reports[-1]} val_reporst {validation_reports[-1][-1][-1]}")
                # logger.info("e, idx_batch", e, idx_batch)
                # train_live.update_batch([loss_reports[-1], validation_reports[-1][-1][-1]], e, idx_batch)
                table_reports = (f"{loss_reports[-1]}", f"{validation_reports[-1][-1][-1]}")
                progress.update_table(table_reports)
                progress.update(task_batch, advance=1)
                time.sleep(0.1)
            progress.update(task_epoch, advance=1)


    # Loss reporting
    fig, ax = plt.subplots()
    ax.plot(loss_reports, label="Training")
    ax.set_xlabel("Batches")
    ax.set_ylabel("Loss")
    # Report validations
    validation_x_axis = [r[0] for r in validation_reports]
    validation_y_axis = [r[1] for r in validation_reports]
    ax.plot(validation_x_axis, validation_y_axis, label="Validation")
    plt.show()



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
    word_tokenizer = AutoTokenizer.from_pretrained(args.answer_tokenizer_name)

    ########################################
    # Load Embedding Data
    ########################################
    ent2id, id2ent, rel2id, id2rel = load_dictionaries(args.path_entities_dict, args.path_relations_dict)
    # TODO: Figure out if we should use the one below
    # id2ent, ent2id = data_utils.load_inclusive_index(args.path_entities_dict)
    # id2rel, rel2id = data_utils.load_inclusive_index(args.path_relations_dict)
    logger.info(f"Loaded a total of :\n\t-{len(id2ent)} entities\n\t-{len(id2rel)} relations")

    ########################################
    # Process the Dataset
    ########################################
    cache_exists = check_cache_exists(args.path_cache_dir)
    if cache_exists and not args.recompute_cache:
        logger.info(f"Loading data from cache {args.path_cache_dir}")
        dataset_partitions = load_from_cache(args.path_cache_dir)
    else:
        logger.info(f"Either cache does not exist or is being force to be recomuted... ")
        dataset_partitions = process_qa_dataset(
            args.path_dataraw,
            args.path_cache_dir,
            args.tvt_split,
            question_tokenizer,
            word_tokenizer,
            ent2id,
            rel2id,
        )
    assert isinstance(dataset_partitions, DataPartitions)

    ########################################
    # Load Pretrained Embeddings
    ########################################
    entity_embeddings = torch.from_numpy(np.load(args.path_entities_embeddings))
    relation_embeddings = torch.from_numpy(np.load(args.path_relations_embeddings))
    embedding_training_metaparameters = json.load(open(args.path_embedding_training_config))

    # Model Hyperparameters
    # gamma = embedding_training_metaparameters["gamma"]
    # model_name = embedding_training_metaparameters["model"]
    checkpoint = torch.load(args.kge_checkpoint_path)
    # state_dict=checkpoint["model_state_dict"]
    assert entity_embeddings.shape[-1] == relation_embeddings.shape[-1], "Relation and Embedding Dimensions are different. Assumption broken. Exiting"
    embeddings_size = entity_embeddings.shape[-1]

    ########################################
    # Create Embedding Matrices
    ########################################
    entity_weights = torch.from_numpy(np.load(args.path_entities_embeddings))
    relation_weights = torch.from_numpy(np.load(args.path_relations_embeddings))
    entity_embeddings = nn.Embedding.from_pretrained(entity_weights)
    relation_embeddings = nn.Embedding.from_pretrained(relation_weights)
    embedding_training_metaparameters = json.load(open(args.path_embedding_training_config))

    # Import the HunchBart Parameter
    hunch_llm = HunchBart(
        pretrained_bart_model=args.hunch_answer_model,
        answer_tokenizer=word_tokenizer,
        graph_embedding_dim=embeddings_size,
    ).to(args.device)

    logger.info("Entering training loop")
    train_loop(
        dataset_partitions,
        word_tokenizer,
        hunch_llm,
        entity_embeddings,
        relation_embeddings,
        args.epochs,
        args.batch_size,
        args.lr,
        args.val_every_n_batches,
    )

    logger.info("Training Finsihed")
    # exit()
    # We need to Load the Bart Model.
    # We prepare our custom encoder for Bart Here

if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_MAIN__")
    main()
