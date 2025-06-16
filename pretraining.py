import argparse
import json
import os
import random
import debugpy
from turtle import left
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, # type: ignore
) 
from transformers.models.cpmant.modeling_cpmant import CpmAntEncoder

from multihopkg import data_utils
from multihopkg.exogenous.sun_models import KGEModel
from multihopkg.logging import setup_logger
from multihopkg.run_configs.pretraining import get_args
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds

Triplet = Tuple[str, str,str]

def dataset_filtering(path: List[Triplet], existing_entities: Set[str], existing_rels: Set[str]) -> bool:
    ########################################
    # Will take dataset and remove samples for which triplets dont exist in our dictioniary
    ########################################
    for head,rel,tail in path:
        invalid_triplet_bool = any([
            head not in existing_entities,
            rel  not in existing_rels,
            tail not in existing_entities
        ])
        if invalid_triplet_bool:
            return False

    return True

def processe_qa_dataset(
    raw_location: str,
    cache_location: str,
    tvt_split: list[float],
    question_tokenizer: Any,
    answer_tokenizer: Any,
    existing_entities: Set[str],
    existing_relations: Set[str]
) -> DataPartitions:

    columns = []
    print(f"Raw location of raw data is {raw_location}")
    # This will load the dataset with questions and answers
    with open(raw_location, "r") as f:
        json_file = json.load(f)

    columns = ["enc_questions", "enc_answer", "triples", "triples_labeled"]
    rows = []
    _d_num_invalid_samples = 0
    for sample in json_file:
        path: List[Triplet] = sample["orig"]["triples"]
        valid_triplets = dataset_filtering(path, existing_entities, existing_relations)
        if not valid_triplets: 
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
                sample["orig"]["triples"],
                sample["orig"]["triples_labeled"],
            ]
        )

    logger.info(f"⚠️NUmber of invalid samples was {_d_num_invalid_samples}")
    
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
        dataset_partitions = processe_qa_dataset(
            args.path_dataraw,
            args.path_cache_dir,
            args.tvt_split,
            question_tokenizer,
            answer_tokenizer,
            set(id2ent.values()),
            set(id2rel.values()),
        )
    assert isinstance(dataset_partitions, DataPartitions)

    ########################################
    # Load Pretrained Embeddings
    ########################################
    entity_embeddings = np.load(args.path_entities_embeddings)
    relation_embeddings = np.load(args.path_relations_embeddings)
    embedding_training_metaparameters = json.load(open(args.path_embedding_training_config))

    # Model Hyperparameters
    gamma = embedding_training_metaparameters["gamma"]
    model_name = embedding_training_metaparameters["model"]
    checkpoint = torch.load(args.kge_checkpoint_path)
    state_dict=checkpoint["model_state_dict"]

    assert isinstance(relation_embeddings, np.ndarray)
    assert isinstance(entity_embeddings, np.ndarray)

    kge_model = KGEModel.from_pretrained(
        model_name=model_name,
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=gamma,
        state_dict=state_dict
    )

    logger.info(f"Loaded the kge model")
    exit()

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
