"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Data processing utilities.
"""

import json
import logging
import ast
import collections
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.itl_typing import DFSplit
from src.setup import get_git_root

START_RELATION = 'START_RELATION'
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

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
    new_model_subdir = dataset + '.test' + model_subdir[len(dataset):]
    new_model_subdir += '-test'
    new_model_path = os.path.join(model_dir, new_model_subdir, file_name)
    return new_model_path

def get_train_path(args):
    if 'NELL' in args.data_dir:
        if not args.model.startswith('point'):
            if args.test:
                train_path = os.path.join(args.data_dir, 'train.dev.large.triples')
            else:
                train_path = os.path.join(args.data_dir, 'train.large.triples')
        else:
            if args.test:
                train_path = os.path.join(args.data_dir, 'train.dev.triples')
            else:
                train_path = os.path.join(args.data_dir, 'train.triples')
    else:
        train_path = os.path.join(args.data_dir, 'train.triples')

    return train_path

def load_seen_entities(adj_list_path, entity_index_path):
    _, id2entity = load_index(entity_index_path)
    with open(adj_list_path, 'rb') as f:
        adj_list = pickle.load(f)
    seen_entities = set()
    for e1 in adj_list:
        seen_entities.add(id2entity[e1])
        for r in adj_list[e1]:
            for e2 in adj_list[e1][r]:
                seen_entities.add(id2entity[e2])
    print('{} seen entities loaded...'.format(len(seen_entities)))
    return seen_entities
 
def load_triples_with_label(data_path, r, entity_index_path, relation_index_path, seen_entities=None, verbose=False):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples, labels = [], []
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            pair, label = line.strip().split(': ')
            e1, e2 = pair.strip().split(',')
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip())) 
                continue
            triples.append(triple2ids(e1, e2, r))
            labels.append(label.strip())
    return triples, labels

def load_triples(data_path, entity_index_path, relation_index_path, group_examples_by_query=False,
                 add_reverse_relations=False, seen_entities=None, verbose=False):
    """
    Convert triples stored on disc into indices.
    """
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples = []
    if group_examples_by_query:
        triple_dict = {}
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            e1, e2, r = line.strip().split()
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip())) 
                continue
            # if r in ['concept:agentbelongstoorganization', 'concept:teamplaysinleague']:
            #     continue
            if group_examples_by_query:
                e1_id, e2_id, r_id = triple2ids(e1, e2, r)
                if e1_id not in triple_dict:
                    triple_dict[e1_id] = {}
                if r_id not in triple_dict[e1_id]:
                    triple_dict[e1_id][r_id] = set()
                triple_dict[e1_id][r_id].add(e2_id)
                if add_reverse_relations:
                    r_inv = r + '_inv'
                    e2_id, e1_id, r_inv_id = triple2ids(e2, e1, r_inv)
                    if e2_id not in triple_dict:
                        triple_dict[e2_id] = {}
                    if r_inv_id not in triple_dict[e2_id]:
                        triple_dict[e2_id][r_inv_id] = set()
                    triple_dict[e2_id][r_inv_id].add(e1_id)
            else:
                triples.append(triple2ids(e1, e2, r))
                if add_reverse_relations:
                    triples.append(triple2ids(e2, e1, r + '_inv'))
    if group_examples_by_query:
        for e1_id in triple_dict:
            for r_id in triple_dict[e1_id]:
                triples.append((e1_id, list(triple_dict[e1_id][r_id]), r_id))
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def load_entity_hist(input_path):
    entity_hist = {}
    with open(input_path) as f:
        for line in f.readlines():
            v, f = line.strip().split()
            entity_hist[v] = int(f)
    return entity_hist

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index

def load_explicit_index(input_path):
    """
    Assumes that the second column are the predetermined ids for an embedding matrix.
    """
    str2int, int2str = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            str_idx, int_idx = line.strip().split()
            _int_idx = int(int_idx)
            str2int[str_idx] = _int_idx
            int2str[_int_idx] = str_idx
    return str2int, int2str 

def prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, test_mode, add_reverse_relations=True):
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

    def get_type(e_name):
        if e_name == DUMMY_ENTITY:
            return DUMMY_ENTITY
        if 'nell-995' in data_dir.lower():
            if '_' in e_name:
                return e_name.split('_')[1]
            else:
                return 'numerical'
        else:
            return 'entity'

    def hist_to_vocab(_dict):
        # Just sort them, first by frequency and then key  ?
        return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

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

    if test_mode:
        keep_triples = train_triples + dev_triples
        removed_triples = test_triples
    else:
        keep_triples = train_triples
        removed_triples = dev_triples + test_triples

    # Index entities and relations
    for line in set(raw_kb_triples + keep_triples + removed_triples):
        e1, e2, r = line.strip().split()
        entity_hist[e1] += 1
        entity_hist[e2] += 1
        if 'nell-995' in data_dir.lower():
            t1 = e1.split('_')[1] if '_' in e1 else 'numerical'
            t2 = e2.split('_')[1] if '_' in e2 else 'numerical'
        else:
            t1 = get_type(e1)
            t2 = get_type(e2)
        type_hist[t1] += 1
        type_hist[t2] += 1
        relation_hist[r] += 1
        if add_reverse_relations:
            inv_r = r + '_inv'
            relation_hist[inv_r] += 1
    # Save the entity and relation indices sorted by decreasing frequency
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for e, freq in hist_to_vocab(entity_hist):
            if e.lower() == "friðrik_þór_friðriksson":
                exit
            o_f.write('{}\t{}\n'.format(e, freq))
    with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_RELATION, DUMMY_RELATION_ID))
        o_f.write('{}\t{}\n'.format(START_RELATION, START_RELATION_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_RELATION, NO_OP_RELATION_ID))
        for r, freq in hist_to_vocab(relation_hist):
            o_f.write('{}\t{}\n'.format(r, freq))
    with open(os.path.join(data_dir, 'type2id.txt'), 'w') as o_f:
        for t, freq in hist_to_vocab(type_hist):
            o_f.write('{}\t{}\n'.format(t, freq))
    print('{} entities indexed'.format(len(entity_hist)))
    print('{} relations indexed'.format(len(relation_hist)))
    print('{} types indexed'.format(len(type_hist)))
    entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
    type2id, id2type = load_index(os.path.join(data_dir, 'type2id.txt'))

    removed_triples = set(removed_triples)
    adj_list = collections.defaultdict(collections.defaultdict)
    entity2typeid = [0 for i in range(len(entity2id))]
    num_facts = 0
    for line in set(raw_kb_triples + keep_triples):
        e1, e2, r = line.strip().split()
        triple_signature = '{}\t{}\t{}'.format(e1, e2, r)
        e1_id = entity2id[e1]
        e2_id = entity2id[e2]
        t1 = get_type(e1)
        t2 = get_type(e2)
        t1_id = type2id[t1]
        t2_id = type2id[t2]
        entity2typeid[e1_id] = t1_id
        entity2typeid[e2_id] = t2_id
        if not triple_signature in removed_triples:
            r_id = relation2id[r]
            if not r_id in adj_list[e1_id]:
                adj_list[e1_id][r_id] = set()
            if e2_id in adj_list[e1_id][r_id]:
                print('Duplicate fact: {} ({}, {}, {})!'.format(
                    line.strip(), id2entity[e1_id], id2relation[r_id], id2entity[e2_id]))
            adj_list[e1_id][r_id].add(e2_id)
            num_facts += 1
            if add_reverse_relations:
                inv_r = r + '_inv'
                inv_r_id = relation2id[inv_r]
                if not inv_r_id in adj_list[e2_id]:
                    adj_list[e2_id][inv_r_id] = set([])
                if e1_id in adj_list[e2_id][inv_r_id]:
                    print('Duplicate fact: {} ({}, {}, {})!'.format(
                        line.strip(), id2entity[e2_id], id2relation[inv_r_id], id2entity[e1_id]))
                adj_list[e2_id][inv_r_id].add(e1_id)
                num_facts += 1
    print('{} facts processed'.format(num_facts))
    # Save adjacency list
    adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
    with open(adj_list_path, 'wb') as o_f:
        pickle.dump(dict(adj_list), o_f)
    with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'wb') as o_f:
        pickle.dump(entity2typeid, o_f)

def get_seen_queries(data_dir, entity_index_path, relation_index_path):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)
    seen_queries = set()
    with open(os.path.join(data_dir, 'train.triples')) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            e1_id = entity2id[e1]
            r_id = relation2id[r]
            seen_queries.add((e1_id, r_id))

    seen_exps = []
    unseen_exps = []
    num_exps = 0
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            num_exps += 1
            e1, e2, r = line.strip().split('\t')
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
    print('Seen examples: {}/{} {}'.format(num_seen_exps, num_exps, seen_ratio))
    print('Unseen examples: {}/{} {}'.format(num_unseen_exps, num_exps, unseen_ratio))

    return seen_queries, (seen_ratio, unseen_ratio)

def get_relations_by_type(data_dir, relation_index_path):
    with open(os.path.join(data_dir, 'raw.kb')) as f:
        triples = list(f.readlines())
    with open(os.path.join(data_dir, 'train.triples')) as f:
        triples += list(f.readlines())
    triples = list(set(triples))

    query_answers = dict()

    theta_1_to_M = 1.5

    for triple_str in triples:
        e1, e2, r = triple_str.strip().split('\t')
        if not r in query_answers:
            query_answers[r] = dict()
        if not e1 in query_answers[r]:
            query_answers[r][e1] = set()
        query_answers[r][e1].add(e2)

    to_M_rels = set()
    to_1_rels = set()

    dev_rels = set()
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            dev_rels.add(r)

    relation2id, _ = load_index(relation_index_path)
    num_rels = len(dev_rels)
    print('{} relations in dev dataset in total'.format(num_rels))
    for r in dev_rels:
        ratio = np.mean([len(x) for x in query_answers[r].values()])
        if ratio > theta_1_to_M:
            to_M_rels.add(relation2id[r])
        else:
            to_1_rels.add(relation2id[r])
    num_to_M = len(to_M_rels) + 0.0
    num_to_1 = len(to_1_rels) + 0.0

    print('to-M relations: {}/{} ({})'.format(num_to_M, num_rels, num_to_M / num_rels))
    print('to-1 relations: {}/{} ({})'.format(num_to_1, num_rels, num_to_1 / num_rels))

    to_M_examples = []
    to_1_examples = []
    num_exps = 0
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            num_exps += 1
            e1, e2, r = line.strip().split('\t')
            if relation2id[r] in to_M_rels:
                to_M_examples.append(line)
            elif relation2id[r] in to_1_rels:
                to_1_examples.append(line)
    num_to_M_exps = len(to_M_examples) + 0.0
    num_to_1_exps = len(to_1_examples) + 0.0
    to_M_ratio = num_to_M_exps / num_exps
    to_1_ratio = num_to_1_exps / num_exps
    print('to-M examples: {}/{} ({})'.format(num_to_M_exps, num_exps, to_M_ratio))
    print('to-1 examples: {}/{} ({})'.format(num_to_1_exps, num_exps, to_1_ratio))

    return to_M_rels, to_1_rels, (to_M_ratio, to_1_ratio)

def load_configs(args, config_path):
    with open(config_path) as f:
        print('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args

def extract_literals(column: Union[str, pd.Series], flatten: bool = False) -> Union[pd.Series, List[str]]:
    """
    Extract Python literals from string representations in pandas columns.
    
    Safely evaluates string representations of Python literals (lists, dicts, etc.)
    using ast.literal_eval. Optionally flattens nested lists into a single flat list.
    This is commonly used for processing path data stored as string representations
    of lists in CSV files.
    
    Args:
        column: Pandas Series containing string representations of Python literals,
               or a single string representation
        flatten: If True, flattens all extracted lists into a single list.
                If False, returns a Series of individual lists
                
    Returns:
        If flatten=False: Pandas Series where each element is the evaluated literal
        If flatten=True: Single flattened list containing all elements from all lists
        
    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['[1, 2, 3]', '[4, 5]', '[6]'])
        >>> result = extract_literals(data, flatten=False)
        >>> print(result.tolist())  # [[1, 2, 3], [4, 5], [6]]
        >>> 
        >>> flat_result = extract_literals(data, flatten=True)
        >>> print(flat_result)  # [1, 2, 3, 4, 5, 6]
        
    Raises:
        ValueError: If any string cannot be safely evaluated as a Python literal
        SyntaxError: If any string contains invalid Python syntax
    """
    # Convert single string input to pandas Series for uniform processing
    if isinstance(column, str):
        column = pd.Series([column])

    # Safely evaluate string representations of Python literals
    evaluated_column = column.apply(ast.literal_eval)

    # DEBUG TODO:  Need to fix the lsp problem here so we need to debug till here and disambiguiate
    # Flatten all lists into a single list if requested
    if flatten:
        flattened_result = [item for sublist in evaluated_column for item in sublist]
        return flattened_result
        
    return evaluated_column


def process_and_cache_triviaqa_data(
    raw_QAData_path: str,
    cached_toked_qatriples_metadata_path: str,
    question_tokenizer: PreTrainedTokenizer,
    entity2id_path: str,
    relation2id_path: str,
    seed: Optional[int] = None,
    override_split: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[DFSplit, Dict[str, Any]]:
    """
    Process and cache question-answer dataset with entity/relation mapping.
    
    Loads raw QA data from CSV, tokenizes questions, maps entities and relations
    to their integer IDs, creates train/dev/test splits, and caches the processed
    data for future use. Supports both automatic splitting and label-guided splitting.
    
    The function expects CSV data with specific column structure:
    - Question: Natural language questions
    - Source-Entity: Starting entity for reasoning
    - Answer-Entity: Target answer entity
    - Paths: (Optional) Reasoning paths as string representations of lists
    - Hops: (Optional) Number of reasoning hops
    - SplitLabel: (Optional) Predefined split labels ('train', 'dev', 'test')
    
    Args:
        raw_QAData_path: Path to the raw CSV file containing QA data
        cached_toked_qatriples_metadata_path: Path where processed metadata will be saved
        question_tokenizer: HuggingFace tokenizer for question text processing
        entity2id: Path to mapping from entity names to integer IDs
        relation2id: Path to mapping from relation names to integer IDs
        seed: Optional seed for random number generation
        override_split: If True, use SplitLabel column for splitting when available
        logger: Optional logger for progress tracking and warnings
        
    Returns:
        Tuple containing:
            - DFSplit: Object with train/dev/test DataFrames
            - Dict: Metadata including tokenizer info, column mappings, and file paths
            
    Raises:
        AssertionError: If CSV file has fewer than 3 columns
        ValueError: If git root cannot be determined
        RuntimeError: If data loading fails or DataFrames are invalid
        KeyError: If required entities/relations are missing from vocabularies
        
    Note:
        - Questions are tokenized without special tokens ([CLS], [SEP])
        - Entity and relation names are mapped to integer IDs
        - Paths are converted from string representations to lists of [head, rel, tail] triples
        - Automatic splitting uses 80/10/10 train/dev/test if no SplitLabel column
        - Small test sets (<100 samples) are used as dev sets with 50/50 dev/test split
    """

    # Load and validate CSV data
    csv_df = pd.read_csv(raw_QAData_path)
    assert len(csv_df.columns) > 2, \
        "CSV file must have at least 3 columns (Question, Source-Entity, Answer-Entity)"
    
    # Extract required columns
    questions = csv_df["Question"]
    source_ent = csv_df["Source-Entity"] 
    answer_ent = csv_df["Answer-Entity"]
    
    # Extract optional columns
    if "Paths" in csv_df.columns:
        paths = extract_literals(csv_df["Paths"]) 
        assert isinstance(paths, pd.Series) # FOr us to use .map a few lines below.
    else:
        paths = None
    split_label = csv_df["SplitLabel"] if 'SplitLabel' in csv_df.columns else None
    hops = csv_df["Hops"] if 'Hops' in csv_df.columns else None

    # Ensure output directory exists
    dir_name = os.path.dirname(cached_toked_qatriples_metadata_path)
    os.makedirs(dir_name, exist_ok=True)

    # Tokenize questions (without special tokens for later processing)
    tokenized_questions = questions.map(
        lambda x: question_tokenizer.encode(x, add_special_tokens=False)
    )
     
    entity2id, _ = load_index(entity2id_path)
    relation2id, _ = load_index(relation2id_path)

    # Map entities and relations to integer IDs
    mapped_source_ent = source_ent.map(lambda ent: entity2id[ent])
    mapped_answer_ent = answer_ent.map(lambda ent: entity2id[ent])
    if paths is not None:
        mapped_paths = paths.map(
            lambda path: [
                [entity2id[head], relation2id[rel], entity2id[tail]] 
                for head, rel, tail in path
            ]
        )

    # Generate unique timestamp for file naming
    timestamp = str(int(datetime.now().timestamp()))
    cached_split_locations: Dict[str, str] = {
        name: cached_toked_qatriples_metadata_path.replace(".json", "") + 
              f"_Split-{name}_date-{timestamp}.parquet"
        for name in ["train", "dev", "test"]
    }

    # Get repository root for relative path generation
    repo_root = get_git_root()
    if repo_root is None:
        raise ValueError("Cannot determine git root path. Ensure you're in a git repository.")

    # Convert to relative paths
    cached_split_locations = {
        key: val.replace(repo_root + "/", "") 
        for key, val in cached_split_locations.items()
    }

    # Combine all processed data into final DataFrame
    data_columns = [tokenized_questions, mapped_source_ent, mapped_answer_ent]
    if paths is not None:
        data_columns.append(mapped_paths)
    if hops is not None:
        data_columns.append(hops)
    if split_label is not None:
        data_columns.append(split_label)
        
    new_df = pd.concat(data_columns, axis=1)
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle data with fixed seed

    # Create train/dev/test splits
    dev_splitted = False
    if (override_split and 'SplitLabel' in new_df.columns and 
        new_df['SplitLabel'].notna().any() and not new_df['SplitLabel'].eq('').all()):
        # Use predefined split labels
        train_df = new_df[new_df['SplitLabel'] == 'train'].reset_index(drop=True)

        if 'test' in new_df["SplitLabel"].values and 'dev' in new_df["SplitLabel"].values:
            test_df = new_df[new_df['SplitLabel'] == 'test'].reset_index(drop=True)
            dev_df = new_df[new_df['SplitLabel'] == 'dev'].reset_index(drop=True)
            dev_splitted = True
            if logger: logger.info("Using SplitLabel column for dev/test splitting")
        else:
            test_df = new_df[new_df['SplitLabel'] != 'train'].reset_index(drop=True)

        if logger: 
            logger.info("Using SplitLabel column for data splitting")
    else:
        # Automatic splitting
        assert False, "I dont believe this should be running"
        train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=seed)

    # Handle dev set creation
    if len(test_df) < 100:
        # Use entire test set as dev set for small datasets
        dev_df = test_df
        if logger: 
            logger.warning("Test set too small (<100 samples), using as dev set")
    elif not dev_splitted:
        # Automatic splitting
        assert False, "I dont believe this should be running"
        dev_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)
        if logger: logger.info("Automatically splitting test set into dev/test")

    # Validate DataFrame creation
    if not all(isinstance(df, pd.DataFrame) for df in [train_df, dev_df, test_df]):
        raise RuntimeError("Data loading failed - invalid DataFrames created")

    # Save processed data to parquet files
    for name, df in {"train": train_df, "dev": dev_df, "test": test_df}.items():
        df.to_parquet(cached_split_locations[name], index=False)

    # Create metadata for reproducibility and documentation
    metadata: Dict[str, Any] = {
        "question_tokenizer": question_tokenizer.name_or_path,
        "question_column": "Question",
        "source_entities_column": "Source-Entity",
        "answer_entity_column": "Answer-Entity",
        "paths_column": "Paths",
        "hops_column": "Hops",
        "splitLabel_column": "SplitLabel",
        "zero_indexed_columns": True,
        "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "saved_paths": cached_split_locations,
        "timestamp": timestamp,
    }

    # Save metadata to JSON file
    with open(cached_toked_qatriples_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return DFSplit(train=train_df, dev=dev_df, test=test_df), metadata


def load_qa_data(
    cached_metadata_path: str,
    raw_QAData_path: str,
    question_tokenizer_name: str,
    entity2id_path: str,
    relation2id_path: str, 
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    force_recompute: bool = False,
    override_split: bool = True,
) -> Tuple[List, List, List, Dict[str, Any]]:
    """
    Load QA dataset with intelligent caching and fallback processing.
    
    Attempts to load preprocessed data from cache first. If cache is missing
    or force_recompute is True, processes raw data and creates new cache.
    This function provides a unified interface for QA data loading with
    automatic preprocessing and caching management.
    
    Args:
        cached_metadata_path: Path to cached metadata JSON file
        raw_QAData_path: Path to raw CSV data file (used if cache missing)
        question_tokenizer_name: HuggingFace tokenizer identifier
        entity2id: Entity name to integer ID mapping
        relation2id: Relation name to integer ID mapping
        seed: Optional seed for random number generation
        logger: Optional logger for progress tracking
        force_recompute: If True, ignore cache and reprocess data
        override_split: If True, use SplitLabel column when available
        
    Returns:
        Tuple containing:
            - train_df: Training DataFrame
            - dev_df: Development DataFrame  
            - test_df: Test DataFrame
            - train_metadata: Metadata dictionary with processing information
            
    Raises:
        FileNotFoundError: If raw data file doesn't exist when cache is missing
        json.JSONDecodeError: If cached metadata is corrupted
        KeyError: If required entities/relations missing from vocabularies
        
    Note:
        - Cached data is loaded from parquet files for efficiency
        - Metadata tracks tokenizer, column mappings, and file locations
        - Automatic fallback to raw processing if cache is invalid
    """

    if os.path.exists(cached_metadata_path) and not force_recompute:
        # Load from cache
        print(f"\033[93mFound cached QA data at {cached_metadata_path}, loading instead of "
              f"processing {raw_QAData_path}\033[0m")
              
        # Load metadata and extract file paths
        with open(cached_metadata_path, 'r') as f:
            train_metadata = json.load(f)
        saved_paths: Dict[str, str] = train_metadata["saved_paths"]

        # Load preprocessed DataFrames
        train_df = pd.read_parquet(saved_paths["train"])
        dev_df = pd.read_parquet(saved_paths["dev"])
        test_df = pd.read_parquet(saved_paths["test"])

        print(f"Loaded cached data from \033[93m\033[4m{cached_metadata_path}\033[0m")
        
    else:
        # Process raw data
        print(f"\033[93mCache not found or force_recompute=True. "
              f"Processing raw data from {raw_QAData_path}\033[0m")
              
        # Load tokenizer and process data
        question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
        df_split, train_metadata = process_and_cache_triviaqa_data(
            raw_QAData_path,
            cached_metadata_path,
            question_tokenizer,
            entity2id_path,
            relation2id_path,
            seed=seed,
            override_split=override_split,
            logger=logger,
        )
        
        # Extract DataFrames from split object
        train_df, dev_df, test_df = df_split.train, df_split.dev, df_split.test
        print(f"Processing complete. Data saved to:\n"
              f"\033[93m\033[4m{train_metadata['saved_paths']}\033[0m")

    # At this point we need to make it more compatible w/ sales force

    # Convert token containers into plain Python lists. Freshly processed data
    # may already be list-valued; parquet-loaded data often comes back as ndarrays.
    def normalize_question_tokens(tokens):
        return tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)

    train_df['Question'] = train_df['Question'].apply(normalize_question_tokens)
    dev_df['Question'] = dev_df['Question'].apply(normalize_question_tokens)
    test_df['Question'] = test_df['Question'].apply(normalize_question_tokens)

    output_columns = ['Source-Entity', 'Answer-Entity', 'Question']
    if 'Paths' in train_df.columns:
        output_columns.append('Paths')
    if 'Hops' in train_df.columns:
        output_columns.append('Hops')

    train_list = train_df[output_columns].values.tolist()
    dev_list = dev_df[output_columns].values.tolist()
    test_list = test_df[output_columns].values.tolist()

    return train_list, dev_list, test_list, train_metadata
