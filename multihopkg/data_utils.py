"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Data processing utilities.
"""

import collections
from functools import cmp_to_key
import json
import os
import pdb
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, List, Tuple, Union
import re
import ast
import logging

import numpy as np
import pandas as pd
from rich import traceback
from torch.nn import Embedding as nn_Embedding
from transformers import PreTrainedTokenizer, AutoTokenizer
from sklearn.model_selection import train_test_split

from multihopkg.utils.setup import get_git_root
from multihopkg.itl_typing import Triple
from multihopkg.itl_typing import DFSplit
from multihopkg.utils.metacode import stale_code

traceback.install()

START_RELATION = "START_RELATION"
NO_OP_RELATION = "NO_OP_RELATION"
NO_OP_ENTITY = "NO_OP_ENTITY"
DUMMY_RELATION = "DUMMY_RELATION"
DUMMY_ENTITY = "DUMMY_ENTITY"

DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1


def check_answer_ratio(examples):
    entity_dict = {}
    for e1, e2, r in examples:
        if not e1 in entity_dict:
            entity_dict[e1] = set()
        entity_dict[e1].add(e2)
    answer_ratio = 0
    for e1 in entity_dict:
        answer_ratio += len(entity_dict[e1])
    return answer_ratio / len(entity_dict)


def check_relation_answer_ratio(input_file, kg):
    example_dict = {}
    with open(input_file) as f:
        for line in f:
            e1, e2, r = line.strip().split()
            e1 = kg.entity2id[e1]
            e2 = kg.entity2id[e2]
            r = kg.relation2id[r]
            if not r in example_dict:
                example_dict[r] = []
            example_dict[r].append((e1, e2, r))
    r_answer_ratio = {}
    for r in example_dict:
        r_answer_ratio[r] = check_answer_ratio(example_dict[r])
    return r_answer_ratio


def change_to_test_model_path(dataset, model_path):
    model_dir = os.path.dirname(os.path.dirname(model_path))
    model_subdir = os.path.basename(os.path.dirname(model_path))
    file_name = os.path.basename(model_path)
    new_model_subdir = dataset + ".test" + model_subdir[len(dataset) :]
    new_model_subdir += "-test"
    new_model_path = os.path.join(model_dir, new_model_subdir, file_name)
    return new_model_path


def get_train_path(data_dir: str, test: bool, model: str):
    if "NELL" in data_dir:
        if not model.startswith("point"):
            if test:
                train_path = os.path.join(data_dir, "train.dev.large.triples")
            else:
                train_path = os.path.join(data_dir, "train.large.triples")
        else:
            if test:
                train_path = os.path.join(data_dir, "train.dev.triples")
            else:
                train_path = os.path.join(data_dir, "train.triples")
    else:
        train_path = os.path.join(data_dir, "train.triples")

    return train_path


def load_seen_entities(adj_list_path, entity_index_path):
    _, id2entity = load_index(entity_index_path)
    with open(adj_list_path, "rb") as f:
        adj_list = pickle.load(f)
    seen_entities = set()
    for e1 in adj_list:
        seen_entities.add(id2entity[e1])
        for r in adj_list[e1]:
            for e2 in adj_list[e1][r]:
                seen_entities.add(id2entity[e2])
    print("{} seen entities loaded...".format(len(seen_entities)))
    return seen_entities


def load_triples_with_label(
    data_path,
    r,
    entity_index_path,
    relation_index_path,
    seen_entities=None,
    verbose=False,
):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples, labels = [], []
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            pair, label = line.strip().split(": ")
            e1, e2 = pair.strip().split(",")
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print(
                        "Skip triple ({}) with unseen entity: {}".format(
                            num_skipped, line.strip()
                        )
                    )
                continue
            triples.append(triple2ids(e1, e2, r))
            labels.append(label.strip())
    return triples, labels


def load_triples(
    data_path: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    add_reverse_relations: bool = False,
    group_examples_by_query: bool = False,
) -> List[Triple]:
    triples = []
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            e1, e2, r = line.strip().split()

            # TODO: Stuff was cut from the original funciton like seen_entities and group_examples_by_query
            # Think if you want to reincoporate this
            triples.append(triple2ids(e1, e2, r, entity2id, relation2id))
            if add_reverse_relations:
                triples.append(triple2ids(e2, e1, r + "_inv", entity2id, relation2id))
            
    # TODO: Again
    # if group_examples_by_query:
    #     for e1_id in triple_dict:
    #         for r_id in triple_dict[e1_id]:
    #             triples.append((e1_id, list(triple_dict[e1_id][r_id]), r_id))
    triples = []
    if group_examples_by_query:
        triple_dict = {}
    return triples


def triple2ids(
    e1, e2, r, entity2id: Dict[str, int], relation2id: Dict[str, int]
) -> Triple:
    return entity2id[e1], entity2id[e2], relation2id[r]
    

def sun_load_triples_and_dict(
    data_path: str,
) -> Dict[str, List[Triple]]:
    config = json.load(open(os.path.join(data_path, "config.json")))
    print(config)
    return config

def load_triples_and_dict(
    data_paths: Sequence[str],
    entity_index_path: str,
    relation_index_path: str,
    group_examples_by_query: bool=False,
    add_reverse_relations: bool = False,
    seen_entities: Optional[Any] = None, # TODO: Replace Any 
    verbose: bool = False,
) -> Tuple[Dict[str, List[Triple]], Dict[int, str], Dict[int, str]]:
    """
    Convert triples stored on disc into indices.
    Args:
        - data_path (Sequence[str]): Paths to the triples file (e.g. [**/train.triples, **/dev.triples])
        - entity_index_path (str): Path to the entity index file (e.g. entity2id.txt)
        - relation_index_path (str): Path to the relation index file (e.g. relation2id.txt)
        - group_examples_by_query (bool): If set, group examples by query relation
        - add_reverse_relations (bool): If set, add reverse relations to the KB environment
        - seen_entities (Optional[Any]): If set, only include triples where the entities are in seen_entities
        - verbose (bool): If set, print the number of triples loaded
    Returns:
        - triplets (Dict[str, Triples]): A dictionary of triples, keyed by the data_path. Correspondance is usually based on splits.
    """
    ########################################
    # Load up dictionaries between str and int idxs
    ########################################
    entity2id, id2entity = load_index(entity_index_path)
    relation2id, id2relation = load_index(relation_index_path)
    
    ########################################
    # Load up the Triplets
    ########################################
    triplets_dict = { k: load_triples(k, entity2id, relation2id, add_reverse_relations, group_examples_by_query) for k in data_paths}
    for k,triplets in triplets_dict.items():
        print("{} triples loaded from {}".format(len(triplets), k))

    return triplets_dict, id2entity, id2relation


def load_entity_hist(input_path):
    entity_hist = {}
    with open(input_path) as f:
        for line in f.readlines():
            v, f = line.strip().split()
            entity_hist[v] = int(f)
    return entity_hist


def load_index(input_path):
    """
    Loads dictionaries to map int-index to str-index and vice-versa
    This specific implementation takes row number as int-index
    Use `load_index_column_wise` for one with int-index as the second column
    """
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index

def load_inclusive_index(input_path: str):
    """
    Loads dictionaries to map int-index to str-index and vice-versa
    This specific implementation takese the first column as embedding matrix int
         and the second column as the RDF to a specific dataset
    Use `load_index_column_wise` for one with int-index as the second column
    """
    index, rev_index = {}, {}
    with open(input_path) as f:
        for line in f.readlines():
            i, v = line.strip().split()
            index[i] = v
            rev_index[v] = i
    return index, rev_index

def prepare_triple_dicts(
    df_split: DFSplit,
):
    """
    Similar to prepare_kb_envrioment but for our ITL approach.
    """
    # Merge the splits
    raw_kb_triples = pd.concat([df_split.train, df_split.dev, df_split.test])
    pdb.set_trace()

    # Create entity and relation indices
    entity_hist = collections.defaultdict(int)
    relation_hist = collections.defaultdict(int)
    type_hist = collections.defaultdict(int)

    # # Index entities and relations
    # # TODO: Previous code considered 'types'. Maybe need?
    # for line in set(raw_kb_triples + keep_triples + removed_triples):
    #     e1, e2, r = line.strip().split()
    #     ########################################
    #     # Count them frequencies for later sorting
    #     ########################################
    #     entity_hist[e1] += 1
    #     entity_hist[e2] += 1
    #     relation_hist[r] += 1
    #     if add_reverse_relations:
    #         inv_r = r + "_inv"
    #         relation_hist[inv_r] += 1
    
  
def prepare_kb_envrioment(
    raw_kb_path, train_path, dev_path, test_path, test_mode, add_reverse_relations=True
):
    """
    Process KB data which was saved as a set of triples.
        (a) Remove train and test triples from the KB envrionment.
        (b) Add reverse triples on demand.
        (c) Index unique entities and relations appeared in the KB.

    :param raw_kb_path: Path to the raw KB triples.
    :param train_path: Path to the train set KB triples.
    :param dev_path: Path to the dev set KB triples.
    :param test_path: Path to the test set KB triples.
    :param add_reverse_relations: If set, add reverse triples to the KB environment.
    """
    data_dir = os.path.dirname(raw_kb_path)

    def hist_to_vocab(_dict):
        return sorted(
            sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True
        )

    pdb.set_trace()

    # Create entity and relation indices
    entity_hist = collections.defaultdict(int)
    relation_hist = collections.defaultdict(int)
    type_hist = collections.defaultdict(int)
    with open(raw_kb_path) as f:
        raw_kb_triples = [l.strip() for l in f.readlines()]
    with open(train_path) as f:
        train_triples = [l.strip() for l in f.readlines()]
    with open(dev_path) as f:
        dev_triples = [l.strip() for l in f.readlines()]
    with open(test_path) as f:
        test_triples = [l.strip() for l in f.readlines()]

    pdb.set_trace()

    if test_mode:
        keep_triples = train_triples + dev_triples
        removed_triples = test_triples
    else:
        keep_triples = train_triples
        removed_triples = dev_triples + test_triples

    # Index entities and relations
    # TODO: Previous code considered 'types'. Maybe need?
    for line in set(raw_kb_triples + keep_triples + removed_triples):
        e1, e2, r = line.strip().split()
        ########################################
        # Count them frequencies for later sorting
        ########################################
        entity_hist[e1] += 1
        entity_hist[e2] += 1
        relation_hist[r] += 1

        if add_reverse_relations:
            inv_r = r + "_inv"
            relation_hist[inv_r] += 1

    pdb.set_trace()
    ########################################
    # Dump the collected frequencies.
    # id's are row numbers
    ########################################
    # Save the entity and relation indices sorted by decreasing frequency
    with open(os.path.join(data_dir, "entity2id.txt"), "w") as o_f:
        for e, freq in hist_to_vocab(entity_hist):
            o_f.write("{}\t{}\n".format(e, freq))
    with open(os.path.join(data_dir, "relation2id.txt"), "w") as o_f:
        for r, freq in hist_to_vocab(relation_hist):
            o_f.write("{}\t{}\n".format(r, freq))
    with open(os.path.join(data_dir, "type2id.txt"), "w") as o_f:
        for t, freq in hist_to_vocab(type_hist):
            o_f.write("{}\t{}\n".format(t, freq))
    pdb.set_trace()

    print("{} entities indexed".format(len(entity_hist)))
    print("{} relations indexed".format(len(relation_hist)))
    print("{} types indexed".format(len(type_hist)))

    entity2id, id2entity = load_index(os.path.join(data_dir, "entity2id.txt"))
    relation2id, id2relation = load_index(os.path.join(data_dir, "relation2id.txt"))
    # TODO: I am not sure if I need types. 
    # type2id, id2type = load_index(os.path.join(data_dir, "type2id.txt"))

    pdb.set_trace()

    ########################################
    # TF is this doing ?
    # * Creating Adjacency list
    # * Creating type2id
    ########################################
    removed_triples : bool =  set(removed_triples)
    adj_list = collections.defaultdict(collections.defaultdict)
    entity2typeid = [0 for i in range(len(entity2id))]
    num_facts = 0

    for line in set(raw_kb_triples + keep_triples):
        e1, e2, r = line.strip().split()
        triple_signature = "{}\t{}\t{}".format(e1, e2, r)
        e1_id = entity2id[e1]
        e2_id = entity2id[e2]
        t1 = get_type(e1)
        t2 = get_type(e2)
        t1_id = type2id[t1]
        t2_id = type2id[t2]
        entity2typeid[e1_id] = t1_id
        entity2typeid[e2_id] = t2_id

        ########################################
        # Only add triplets that are not deemed "removed"
        # Create adjacency list
        ########################################
        if not triple_signature in removed_triples:
            r_id = relation2id[r]

            if not r_id in adj_list[e1_id]:
                adj_list[e1_id][r_id] = set()

            # if e2_id in adj_list[e1_id][r_id]:
            #     print(
            #         "Duplicate fact: {} ({}, {}, {})!".format(
            #             line.strip(),
            #             id2entity[e1_id],
            #             id2relation[r_id],
            #             id2entity[e2_id],
            #         )
            #     )

            adj_list[e1_id][r_id].add(e2_id)
            num_facts += 1

            ########################################
            # In case of reverse relationships
            ########################################
            if add_reverse_relations:
                inv_r = r + "_inv"
                inv_r_id = relation2id[inv_r]
                if not inv_r_id in adj_list[e2_id]:
                    adj_list[e2_id][inv_r_id] = set([])
                if e1_id in adj_list[e2_id][inv_r_id]:
                    print(
                        "Duplicate fact: {} ({}, {}, {})!".format(
                            line.strip(),
                            id2entity[e2_id],
                            id2relation[inv_r_id],
                            id2entity[e1_id],
                        )
                    )
                adj_list[e2_id][inv_r_id].add(e1_id)
                num_facts += 1

    print("{} facts processed".format(num_facts))
    # Save adjacency list
    adj_list_path = os.path.join(data_dir, "adj_list.pkl")
    with open(adj_list_path, "wb") as o_f:
        pickle.dump(dict(adj_list), o_f)
    with open(os.path.join(data_dir, "entity2typeid.pkl"), "wb") as o_f:
        pickle.dump(entity2typeid, o_f)


def get_seen_queries(data_dir, entity_index_path, relation_index_path):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)
    seen_queries = set()
    with open(os.path.join(data_dir, "train.triples")) as f:
        for line in f:
            e1, e2, r = line.strip().split("\t")
            e1_id = entity2id[e1]
            r_id = relation2id[r]
            seen_queries.add((e1_id, r_id))

    seen_exps = []
    unseen_exps = []
    num_exps = 0
    with open(os.path.join(data_dir, "dev.triples")) as f:
        for line in f:
            num_exps += 1
            e1, e2, r = line.strip().split("\t")
            e1_id = entity2id[e1]
            r_id = relation2id[r]
            if (e1_id, r_id) in seen_queries:
                seen_exps.append(line)
            else:
                unseen_exps.append(line)
    num_seen_exps = len(seen_exps) + 0.0
    num_unseen_exps = len(unseen_exps) + 0.0
    seen_ratio = num_seen_exps / num_exps
    unseen_ratio = num_unseen_exps / num_exps
    print("Seen examples: {}/{} {}".format(num_seen_exps, num_exps, seen_ratio))
    print("Unseen examples: {}/{} {}".format(num_unseen_exps, num_exps, unseen_ratio))

    return seen_queries, (seen_ratio, unseen_ratio)


def get_relations_by_type(data_dir, relation_index_path):
    with open(os.path.join(data_dir, "raw.kb")) as f:
        triples = list(f.readlines())
    with open(os.path.join(data_dir, "train.triples")) as f:
        triples += list(f.readlines())
    triples = list(set(triples))

    query_answers = dict()

    theta_1_to_M = 1.5

    for triple_str in triples:
        e1, e2, r = triple_str.strip().split("\t")
        if not r in query_answers:
            query_answers[r] = dict()
        if not e1 in query_answers[r]:
            query_answers[r][e1] = set()
        query_answers[r][e1].add(e2)

    to_M_rels = set()
    to_1_rels = set()

    dev_rels = set()
    with open(os.path.join(data_dir, "dev.triples")) as f:
        for line in f:
            e1, e2, r = line.strip().split("\t")
            dev_rels.add(r)

    relation2id, _ = load_index(relation_index_path)
    num_rels = len(dev_rels)
    print("{} relations in dev dataset in total".format(num_rels))
    for r in dev_rels:
        ratio = np.mean([len(x) for x in query_answers[r].values()])
        if ratio > theta_1_to_M:
            to_M_rels.add(relation2id[r])
        else:
            to_1_rels.add(relation2id[r])
    num_to_M = len(to_M_rels) + 0.0
    num_to_1 = len(to_1_rels) + 0.0

    print("to-M relations: {}/{} ({})".format(num_to_M, num_rels, num_to_M / num_rels))
    print("to-1 relations: {}/{} ({})".format(num_to_1, num_rels, num_to_1 / num_rels))

    to_M_examples = []
    to_1_examples = []
    num_exps = 0
    with open(os.path.join(data_dir, "dev.triples")) as f:
        for line in f:
            num_exps += 1
            e1, e2, r = line.strip().split("\t")
            if relation2id[r] in to_M_rels:
                to_M_examples.append(line)
            elif relation2id[r] in to_1_rels:
                to_1_examples.append(line)
    num_to_M_exps = len(to_M_examples) + 0.0
    num_to_1_exps = len(to_1_examples) + 0.0
    to_M_ratio = num_to_M_exps / num_exps
    to_1_ratio = num_to_1_exps / num_exps
    print("to-M examples: {}/{} ({})".format(num_to_M_exps, num_exps, to_M_ratio))
    print("to-1 examples: {}/{} ({})".format(num_to_1_exps, num_exps, to_1_ratio))

    return to_M_rels, to_1_rels, (to_M_ratio, to_1_ratio)


def load_configs(args, config_path):
    with open(config_path) as f:
        print("loading configuration file {}".format(config_path))
        for line in f:
            if not "=" in line:
                continue
            arg_name, arg_value = line.strip().split("=")
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print("{} = {}".format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == "True":
                        setattr(args, arg_name, True)
                    elif arg_value == "False":
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError(
                            "Unrecognized boolean value description: {}".format(
                                arg_value
                            )
                        )
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError(
                        "Unrecognized attribute type: {}: {}".format(
                            arg_name, type(arg_value2)
                        )
                    )
            else:
                raise ValueError("Unrecognized argument: {}".format(arg_name))
    return args

def extract_literals(column: Union[str, pd.Series], flatten: bool = False) -> Union[pd.Series, List[str]]:
    """
    Extracts the list of string literals from each entry in the provided column (Pandas Series or string)
    using ast.literal_eval. Optionally flattens the extracted lists into a single list if 'flatten' is set to True.

    Args:
        column (Union[str, pd.Series]): The column containing string representations of lists. Can be a
                                        Pandas Series or a string representation of a list.
        flatten (bool): If True, flattens the lists into a single list. Default is False.

    Returns:
        Union[pd.Series, List[str]]: A Pandas Series of lists if flatten is False, otherwise a single flattened list of strings.
    """

    # Convert the input to a Pandas Series if it's a string
    if isinstance(column, str):
        column = pd.Series([column])

    # Convert string representations of lists into actual Python lists
    column = column.apply(ast.literal_eval)

    # Flatten the lists if the flatten argument is True
    if flatten: column = [item for sublist in column for item in sublist]
    return column

def process_and_cache_triviaqa_data(
    raw_QAData_path: str,
    cached_toked_qatriples_metadata_path: str,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    override_split: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[DFSplit, Dict] :
    """
    Args:
        raw_triples_loc (str) : Place where the unprocessed triples are
        cached_toked_qatriples_path (str) : Place where processed triples are meante to go. You must format them.
        idx_2_graphEnc (Dict[str, np.array]) : The encoding of the tripleshttps://www.youtube.com/watch?v=f-sRcVkZ9yg
        text_tokenizer (AutoTokenizer) : The tokenizer for the text
    Returns:

    Data Assumptions:
        - csv file consists of 1..N columns.
          N-1 is Question, N is Answer
        - 1..N-2 represent the path
          These columns are organized as Entity, Relation, Entity,...
    LG: We might change this assumption to a whole graph later on:
    """

    ## NOTE:  ---
    ## Old Data Loading has been moved elsewhere
    ## ----------
    ## Processing
    csv_df = pd.read_csv(raw_QAData_path)
    assert (
        len(csv_df.columns) > 2
    ), "The CSV file should have at least 2 columns. One triplet and one QA pair"
    
    # FIX: The harcoding of things like "Question" and "Answer" is not good.
    # !TODO: Make this more flexible and relavant entities and relations be optional features
    questions = csv_df["Question"]
    answers = csv_df["Answer"]
    query_ent = csv_df["Query-Entity"]
    query_rel = csv_df["Query-Relation"]
    answer_ent = csv_df["Answer-Entity"]
    paths = extract_literals(csv_df["Paths"]) if 'Paths' in csv_df.columns else None
    splitLabel = csv_df["SplitLabel"] if 'SplitLabel' in csv_df.columns else None
    hops = csv_df["Hops"] if 'Hops' in csv_df.columns else None

    # Ensure directory exists
    dir_name = os.path.dirname(cached_toked_qatriples_metadata_path)
    os.makedirs(dir_name, exist_ok=True)

    ## Prepare the language data
    questions = questions.map(lambda x: question_tokenizer.encode(x, add_special_tokens=False))
    answers = answers.map(
        lambda x: [answer_tokenizer.bos_token_id]
        + answer_tokenizer.encode(x, add_special_tokens=False)
        + [answer_tokenizer.eos_token_id]
    )

    # Preparing the KG data by converting text to indices
    query_ent = query_ent.map(lambda ent: entity2id[ent])
    query_rel = query_rel.map(lambda rel: relation2id[rel])
    answer_ent = answer_ent.map(lambda ent: entity2id[ent])
    if paths is not None:
        paths = paths.map(lambda path: [[entity2id[head], relation2id[rel], entity2id[tail]] for head, rel, tail in path])

    # timestamp without nanoseconds
    timestamp = str(int(datetime.now().timestamp()))
    cached_split_locations: Dict[str, str] = {
        name: cached_toked_qatriples_metadata_path.replace(".json", "") + f"_Split-{name}" + f"_date-{timestamp}" + ".parquet"
        for name in ["train", "dev", "test"]
    }

    repo_root = get_git_root()
    if repo_root is None:
        raise ValueError("Cannot get the git root path. Please make sure you are running a clone of the repo")

    cached_split_locations = {key : val.replace(repo_root + "/", "") for key,val in cached_split_locations.items()}

    # Start amalgamating the data into its final form
    # TODO: test set
    new_df = pd.concat([questions, answers, query_ent, query_rel, answer_ent, paths, hops, splitLabel], axis=1)
    new_df = new_df.sample(frac=1).reset_index(drop=True) # Shuffle before splitting by label

    # Check if splitLabel column has meaningful values to guide the split
    if override_split and 'SplitLabel' in new_df.columns and new_df['SplitLabel'].notna().any() and not new_df['SplitLabel'].eq('').all():
        train_df = new_df[new_df['SplitLabel'] == 'train'].reset_index(drop=True)
        test_df = new_df[new_df['SplitLabel'] != 'train'].reset_index(drop=True)
        if logger: logger.info(f"Using splitLabel column to split the data into train and test sets.")
    else:
        new_df = new_df.sample(frac=1).reset_index(drop=True)
        train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

     # If the test set is too small, use it as dev
    if len(test_df) < 100:
        dev_df = test_df
        if logger: logger.warning("Test set is too small, using it as dev set!!")
    else:
        dev_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    if not isinstance(train_df, pd.DataFrame) or not isinstance(dev_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise RuntimeError("The data was not loaded properly. Please check the data loading code.")

    for name,df in {"train": train_df, "dev": dev_df, "test": test_df}.items():
        df.to_parquet(cached_split_locations[name], index=False)

    ## Prepare metadata for export
    # Tokenize the text by applying a pandas map function
    # Store the metadata
    metadata = {
        "question_tokenizer": question_tokenizer.name_or_path,
        "answer_tokenizer": answer_tokenizer.name_or_path,
        "question_column": "Question",
        "answer_column": "Answer",
        "query_entities_column": "Query-Entity",
        "query_relations_column": "Query-Relation",
        "answer_entity_column": "Answer-Entity",
        "paths_column": "Paths",
        "hops_column": "Hops",
        "splitLabel_column": "SplitLabel",
        "0-index_column": True,
        "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "saved_paths": cached_split_locations,
        "timestamp": timestamp,
    }

    with open(cached_toked_qatriples_metadata_path, "w") as f:
        json.dump(metadata, f)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Save the triplets to a file for later use with other algorithms
    for name,df in {"train": train_df, "dev": dev_df, "test": test_df}.items():
        triplets = []
        for i, row in df.iterrows():
            triplets.append((id2entity[row['Query-Entity']], id2relation[row['Query-Relation']], id2entity[row['Answer-Entity']]))
        save_triplets = pd.DataFrame(triplets, columns=["head", "relation", "tail"])
        save_triplets.to_csv(cached_split_locations[name].replace(".parquet", f"_{name}_triplets.txt"), sep='\t', index=False, header=False)

    return DFSplit(train=train_df, dev=dev_df, test=test_df), metadata


def load_qa_data(
    cached_metadata_path: str,
    raw_QAData_path,
    question_tokenizer_name: str,
    answer_tokenizer_name: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int], 
    logger: logging.Logger,
    force_recompute: bool = False,
    override_split: bool = True,
):

    if os.path.exists(cached_metadata_path) and not force_recompute:
        logger.info(
            f"\033[93m Found cache for the QA data {cached_metadata_path} will load it instead of working on {raw_QAData_path}. \033[0m"
        )
        # Read the first line of the raw csv to count the number of columns
        train_metadata = json.load(open(cached_metadata_path.format(question_tokenizer_name, answer_tokenizer_name)))
        saved_paths: Dict[str, str] = train_metadata["saved_paths"]

        train_df = pd.read_parquet(saved_paths["train"])
        # TODO: Eventually use this to avoid data leakage
        dev_df = pd.read_parquet(saved_paths["dev"])
        test_df = pd.read_parquet(saved_paths["test"])

        # Ensure that we are not reading them integers as strings, but also not as floats
        logger.info(
            f"Loaded cached data from \033[93m\033[4m{json.dumps(cached_metadata_path,indent=4)} \033[0m"
        )
    else:
        ########################################
        # Actually compute the data.
        ########################################
        logger.info(
            f"\033[93m Did not find cache for the QA data {cached_metadata_path}. Will now process it from {raw_QAData_path} \033[0m"
        )
        question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
        answer_tokenzier   = AutoTokenizer.from_pretrained(answer_tokenizer_name)
        df_split, train_metadata = ( # Includes shuffling
            process_and_cache_triviaqa_data(  # TOREM: Same here, might want to remove if not really used
                raw_QAData_path,
                cached_metadata_path,
                question_tokenizer,
                answer_tokenzier,
                entity2id,
                relation2id,
                override_split=override_split,
                logger=logger,
            )
        )
        train_df, dev_df, test_df = df_split.train, df_split.dev, df_split.test
        logger.info(
            f"Done. Result dumped at : \n\033[93m\033[4m{train_metadata['saved_paths']}\033[0m"
        )

    return train_df, dev_df, test_df, train_metadata


@stale_code
def data_loading_router(
    raw_QAPathData_path: str,
    cached_toked_qatriples_path: str,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
):
    if "freebaseqa" in raw_QAPathData_path:
        return process_freebaseqa_data(
            raw_QAPathData_path,
            cached_toked_qatriples_path,
            question_tokenizer,
        )
    elif "triviaqa" in raw_QAPathData_path:
        return process_and_cache_triviaqa_data(
            raw_QAPathData_path,
            cached_toked_qatriples_path,
            question_tokenizer,
            answer_tokenizer,
        )
    else:
        raise ValueError("The data loading router could not find a matching data loader for the data path")


def process_freebaseqa_data(
    raw_QAPathData_path: str,
    cached_toked_qatriples_path: str,
    text_tokenizer: PreTrainedTokenizer,
):
    # Start Loading the data  s
    files = ["train.parquet","dev.parquet","test.parquet"]
    for file in files:
        if not os.path.exists(os.path.join(raw_QAPathData_path, file)):
            raise ValueError(f"The file {file} does not exist in the raw data path {raw_QAPathData_path}")
        pdb.set_trace()

def load_index_column_wise(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2entity = {}
    entity2id = {}
    with open(path) as f:
        # File is a two column tsv file
        for line in f:
            id, idx = re.split(r'\s+', line.strip())
            id2entity[int(idx)] = id
            entity2id[id] = int(idx)  # Yeah I know how this looks.

    return id2entity, entity2id

def load_dictionaries(raw_data_path: str) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str], Dict[str, int]]:
    # Load the dictionaries
    entity2id_path      = os.path.join(raw_data_path, "entity2id.txt")
    relation2id_path    = os.path.join(raw_data_path, "relation2id.txt")
    id2entity, entity2id        = load_index_column_wise(entity2id_path)
    id2relation, relation2id    = load_index_column_wise(relation2id_path)

    return  id2entity, entity2id, id2relation, relation2id

