import argparse
import json
import os
import random
from turtle import left
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BartModel,
    BertModel,
    PreTrainedTokenizer,
)
from transformers.models.cpmant.modeling_cpmant import CpmAntEncoder

from multihopkg import data_utils
from multihopkg.logging import setup_logger
from multihopkg.run_configs.pretraining import get_args
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds


def processe_qa_dataset(
    raw_location: str,
    cache_location: str,
    tvt_split: list[float],
    question_tokenizer: Any,
    answer_tokenizer: Any,
) -> DataPartitions:

    columns = []
    with open(raw_location, "r") as f:
        json_file = json.load(f)

    columns = ["enc_questions", "enc_answer", "triples", "triples_labeled"]
    rows = []
    for sample in json_file:
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
                sample["orig"]["triples"],
                sample["orig"]["triples_labeled"],
            ]
        )
    
    # Train, Test, Validation Partitions
    random.shuffle(rows)
    train_len = int(len(rows)*tvt_split[0])
    left_over_perc =  tvt_split[1] / (1 - tvt_split[0])
    valid_len = int((len(rows) - train_len) * left_over_perc)
    
    train_ds = rows[:train_len]
    valid_ds = rows[train_len: train_len + valid_len]
    test_ds = rows[train_len + valid_len: ]

    # Now we save it as well

    all_ds = {
        "train.csv": train_ds,
        "valid.csv": valid_ds,
        "test.csv": test_ds,
    }

    for ds_file_name, ds_list in all_ds.items():
        df = pd.DataFrame(ds_list, columns=columns)
        path_to_save = os.path.join(cache_location, ds_file_name)
        logger.info(f"Saving the split {ds_file_name}")
        df.to_csv(path_to_save, index=False)

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

def main():
    args = get_args()

    global logger
    logger = setup_logger("pretraining_py")
    set_seeds(args.seed)

    ########################################
    # Process the NLP components
    ########################################
    question_tokenizer = AutoTokenizer.from_pretrained(args.question_tokenizer_name)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_tokenizer_name)

    ########################################
    # Process the Dataset
    ########################################
    cache_exists = check_cache_exists(args.path_cache_dir)
    if cache_exists:
        dataset_partitions = load_from_cache(args.path_cache_dir)
    else:
        dataset_partitions = processe_qa_dataset(
            args.path_dataraw,
            args.path_cache_dir,
            args.tvt_split,
            question_tokenizer,
            answer_tokenizer,
        )
    assert isinstance(dataset_partitions, DataPartitions)

    ########################################
    # Load Pretrained Embeddings
    ########################################
    entity_embeddings = np.load(args.path_entities_embeddings)
    relation_embeddings = np.load(args.path_relations_embeddings)

    assert isinstance(relation_embeddings, np.ndarray)
    assert isinstance(entity_embeddings, np.ndarray)

    id2ent, ent2id = data_utils.load_inclusive_index(args.path_entities_dict)
    id2rel, rel2id = data_utils.load_inclusive_index(args.path_relations_dict)
    


    # TODO: We might want to use ITLKnowledgeGraph here
    # # Load the sun model
    # knowledge_graph = SunKnowledgeGraph.from(
    #     model=args.model,
    #     pretrained_sun_model_path=args.pretrained_sun_model_loc,
    #     data_path=args.data_dir,
    #     graph_embed_model_name=args.graph_embed_model_name,
    #     gamma=args.gamma,
    #     id2entity=id2ent,
    #     entity2id=ent2id,
    #     id2relation=id2rel,
    #     relation2id=rel2id,
    #     device=args.device,
    # )

    exit()

    # We need to Load the Bart Model.
    # We prepare our custom encoder for Bart Here
    hunch_llm = GraphBart(
        pretrained_bart_model=args.pretrained_llm_for_hunch,
        answer_tokenizer=answer_tokenizer,
        # We convert the graph embeddings to state embeddings obeying current state dimensions
        graph_embedding_dim=args.llm_model_dim,
    ).to(args.device)




if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_MAIN__")
    main()
