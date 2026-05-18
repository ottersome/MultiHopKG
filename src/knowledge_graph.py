"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Knowledge Graph Environment.
"""
from tqdm import tqdm

import collections
import copy
import hashlib
import os
import pickle

import torch
import torch.nn as nn

from src.data_utils import load_index
from src.data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from src.data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from src.data_utils import START_RELATION_ID
import src.utils.ops as ops
from src.utils.ops import int_var_cuda, var_cuda


class KnowledgeGraph(nn.Module):
    """
    The discrete knowledge graph is stored with an adjacency list.
    """
    CACHE_VERSION = 1
    _GRAPH_DATA_CACHE = {}
    _ACTION_SPACE_CACHE = {}
    _ANSWER_CACHE = {}

    def __init__(self, args):
        super(KnowledgeGraph, self).__init__()
        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}
        self.type2id, self.id2type = {}, {}
        self.entity2typeid = {}
        self.adj_list = None
        self.bandwidth = args.bandwidth
        self.args = args

        self.action_space = None
        self.action_space_buckets = None
        self.unique_r_space = None

        self.train_subjects = None
        self.train_objects = None
        self.dev_subjects = None
        self.dev_objects = None
        self.all_subjects = None
        self.all_objects = None
        self.train_subject_vectors = None
        self.train_object_vectors = None
        self.dev_subject_vectors = None
        self.dev_object_vectors = None
        self.all_subject_vectors = None
        self.all_object_vectors = None

        print('** Create {} knowledge graph **'.format(args.model))
        self.load_graph_data(args.data_dir)
        self.load_all_answers(args.data_dir)

        # Define NN Modules
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate
        self.num_graph_convolution_layers = args.num_graph_convolution_layers
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_img_embeddings = None
        self.relation_img_embeddings = None
        self.EDropout = None
        self.RDropout = None
        
        self.define_modules()
        self.initialize_modules()

    def _cache_dir(self, data_dir):
        cache_dir = os.path.join(data_dir, '.kg_cache')
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _cache_path(self, data_dir, prefix, payload):
        cache_key = hashlib.md5(repr(payload).encode('utf-8')).hexdigest()
        file_name = '{}_v{}_{}.pt'.format(prefix, self.CACHE_VERSION, cache_key)
        return os.path.join(self._cache_dir(data_dir), file_name)

    def _tensor_to_cpu(self, tensor):
        if hasattr(tensor, 'detach'):
            return tensor.detach().cpu()
        return tensor.cpu()

    def _serialize_action_space(self, action_space):
        (r_space, e_space), action_mask = action_space
        return {
            'r_space': self._tensor_to_cpu(r_space),
            'e_space': self._tensor_to_cpu(e_space),
            'action_mask': self._tensor_to_cpu(action_mask),
        }

    def _deserialize_action_space(self, payload):
        return (
            (
                int_var_cuda(payload['r_space']),
                int_var_cuda(payload['e_space']),
            ),
            var_cuda(payload['action_mask'])
        )

    def _serialize_nested_vector_dict(self, nested_dict):
        payload = {}
        for x in nested_dict:
            payload[x] = {}
            for y in nested_dict[x]:
                payload[x][y] = self._tensor_to_cpu(nested_dict[x][y])
        return payload

    def _deserialize_nested_vector_dict(self, nested_dict):
        payload = collections.defaultdict(collections.defaultdict)
        for x in nested_dict:
            payload[x] = {}
            for y in nested_dict[x]:
                payload[x][y] = int_var_cuda(nested_dict[x][y])
        return payload

    def _graph_data_cache_key(self, data_dir):
        return os.path.abspath(data_dir)

    def _action_space_cache_spec(self, data_dir, cache_namespace):
        return {
            'data_dir': os.path.abspath(data_dir),
            'namespace': cache_namespace,
            'model': self.args.model,
            'bandwidth': self.bandwidth,
            'use_action_space_bucketing': getattr(self.args, 'use_action_space_bucketing', False),
            'bucket_interval': getattr(self.args, 'bucket_interval', None),
        }

    def _answer_cache_spec(self, data_dir, add_reversed_edges):
        return {
            'data_dir': os.path.abspath(data_dir),
            'model': self.args.model,
            'test': bool(getattr(self.args, 'test', False)),
            'add_reversed_edges': bool(add_reversed_edges),
        }

    def load_graph_data(self, data_dir):
        graph_cache_key = self._graph_data_cache_key(data_dir)
        cached_graph_data = self._GRAPH_DATA_CACHE.get(graph_cache_key)
        if cached_graph_data is not None:
            self.entity2id = cached_graph_data['entity2id']
            self.id2entity = cached_graph_data['id2entity']
            self.type2id = cached_graph_data['type2id']
            self.id2type = cached_graph_data['id2type']
            self.entity2typeid = cached_graph_data['entity2typeid']
            self.relation2id = cached_graph_data['relation2id']
            self.id2relation = cached_graph_data['id2relation']
            self.adj_list = cached_graph_data['adj_list']
            if self.args.model.startswith('point'):
                self.vectorize_action_space(data_dir)
            return

        # Load indices
        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        self.type2id, self.id2type = load_index(os.path.join(data_dir, 'type2id.txt'))
        with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'rb') as f:
            self.entity2typeid = pickle.load(f)
        self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
       
        # Load graph structures
        print(self.args.model)
        if self.args.model.startswith('point'): 
            # Base graph structure used for training and test
            adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
            with open(adj_list_path, 'rb') as f:
                self.adj_list = pickle.load(f)
            self.vectorize_action_space(data_dir)
        self._GRAPH_DATA_CACHE[graph_cache_key] = {
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'type2id': self.type2id,
            'id2type': self.id2type,
            'entity2typeid': self.entity2typeid,
            'relation2id': self.relation2id,
            'id2relation': self.id2relation,
            'adj_list': self.adj_list,
        }

    def vectorize_action_space(self, data_dir, cache_namespace='base'):
        """
        Pre-process and numericalize the knowledge graph structure.
        """
        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    e_id = self.entity2id[e.strip()]
                    score = float(score)
                    pgrk_scores[e_id] = score
            return pgrk_scores
                    
        # Sanity check
        num_facts = 0
        out_degrees = collections.defaultdict(int)
        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
        print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
        print('Sanity check: {} facts in knowledge graph'.format(num_facts))

        # load page rank scores
        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))
        
        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))
                if len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1))
            return action_space

        def get_unique_r_space(e1):
            if e1 in self.adj_list:
                return list(self.adj_list[e1].keys())
            else:
                return []

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    action_mask[i, j] = 1
            return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        action_space_cache_spec = self._action_space_cache_spec(data_dir, cache_namespace)
        action_space_cache_key = repr(action_space_cache_spec)
        cached_action_space = self._ACTION_SPACE_CACHE.get(action_space_cache_key)
        if cached_action_space is None:
            action_space_cache_path = self._cache_path(data_dir, 'action_space', action_space_cache_spec)
            if os.path.exists(action_space_cache_path):
                cached_action_space = torch.load(action_space_cache_path, map_location='cpu')
                self._ACTION_SPACE_CACHE[action_space_cache_key] = cached_action_space
                print('Loaded action space cache from {}'.format(action_space_cache_path))
        if cached_action_space is not None:
            if cached_action_space['use_action_space_bucketing']:
                self.entity2bucketid = cached_action_space['entity2bucketid'].long()
                self.action_space_buckets = {}
                print("Loading action space to buckets")
                for key, bucket_payload in cached_action_space['action_space_buckets'].items():
                    self.action_space_buckets[key] = self._deserialize_action_space(bucket_payload)
                self.action_space = None
            else:
                self.action_space = self._deserialize_action_space(cached_action_space['action_space'])
                self.action_space_buckets = None
                self.entity2bucketid = None
            unique_r_space = cached_action_space.get('unique_r_space')
            self.unique_r_space = int_var_cuda(unique_r_space) if unique_r_space is not None else None
            return

        if self.args.use_action_space_bucketing:
            """
            Store action spaces in buckets.
            """
            self.action_space_buckets = {}
            action_space_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            num_facts_saved_in_action_table = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                key = int(len(action_space) / self.args.bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key
                self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
                action_space_buckets_discrete[key].append(action_space)
                num_facts_saved_in_action_table += len(action_space)
            print('Sanity check: {} facts saved in action table'.format(
                num_facts_saved_in_action_table - self.num_entities))
            for key in action_space_buckets_discrete:
                print('Vectorizing action spaces bucket {}...'.format(key))
                self.action_space_buckets[key] = vectorize_action_space(
                    action_space_buckets_discrete[key], key * self.args.bucket_interval)
        else:
            action_space_list = []
            max_num_actions = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                action_space_list.append(action_space)
                if len(action_space) > max_num_actions:
                    max_num_actions = len(action_space)
            print('Vectorizing action spaces...')
            self.action_space = vectorize_action_space(action_space_list, max_num_actions)
            
            if self.args.model.startswith('rule'):
                unique_r_space_list = []
                max_num_unique_rs = 0
                for e1 in sorted(self.adj_list.keys()):
                    unique_r_space = get_unique_r_space(e1)
                    unique_r_space_list.append(unique_r_space)
                    if len(unique_r_space) > max_num_unique_rs:
                        max_num_unique_rs = len(unique_r_space)
                self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)
        action_space_cache_payload = {
            'use_action_space_bucketing': bool(self.args.use_action_space_bucketing),
            'unique_r_space': self._tensor_to_cpu(self.unique_r_space) if self.unique_r_space is not None else None,
        }
        if self.args.use_action_space_bucketing:
            serialized_buckets = {}
            for key, action_space in self.action_space_buckets.items():
                serialized_buckets[key] = self._serialize_action_space(action_space)
            action_space_cache_payload['entity2bucketid'] = self._tensor_to_cpu(self.entity2bucketid)
            action_space_cache_payload['action_space_buckets'] = serialized_buckets
        else:
            action_space_cache_payload['action_space'] = self._serialize_action_space(self.action_space)
        self._ACTION_SPACE_CACHE[action_space_cache_key] = action_space_cache_payload
        action_space_cache_path = self._cache_path(data_dir, 'action_space', action_space_cache_spec)
        torch.save(action_space_cache_payload, action_space_cache_path)
        print('Saved action space cache to {}'.format(action_space_cache_path))

    def load_all_answers(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        dev_subjects, dev_objects = {}, {}
        all_subjects, all_objects = {}, {}
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
        for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
            if 'NELL' in self.args.data_dir and self.args.test and file_name == 'train.triples':
                continue
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    if file_name in ['raw.kb', 'train.triples']:
                        add_subject(e1, e2, r, train_subjects)
                        add_object(e1, e2, r, train_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                    if file_name in ['raw.kb', 'train.triples', 'dev.triples']:
                        add_subject(e1, e2, r, dev_subjects)
                        add_object(e1, e2, r, dev_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                    add_subject(e1, e2, r, all_subjects)
                    add_object(e1, e2, r, all_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        self.train_subjects = train_subjects
        self.train_objects = train_objects
        self.dev_subjects = dev_subjects
        self.dev_objects = dev_objects
        self.all_subjects = all_subjects
        self.all_objects = all_objects
       
        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v
        
        self.train_subject_vectors = answers_to_var(train_subjects)
        self.train_object_vectors = answers_to_var(train_objects)
        self.dev_subject_vectors = answers_to_var(dev_subjects)
        self.dev_object_vectors = answers_to_var(dev_objects)
        self.all_subject_vectors = answers_to_var(all_subjects)
        self.all_object_vectors = answers_to_var(all_objects)

    def load_fuzzy_facts(self):
        # extend current adjacency list with fuzzy facts
        dev_path = os.path.join(self.args.data_dir, 'dev.triples')
        test_path = os.path.join(self.args.data_dir, 'test.triples')
        with open(dev_path) as f:
            dev_triples = [l.strip() for l in f.readlines()]
        with open(test_path) as f:
            test_triples = [l.strip() for l in f.readlines()]
        removed_triples = set(dev_triples + test_triples)
        theta = 0.5
        fuzzy_fact_path = os.path.join(self.args.data_dir, 'train.fuzzy.triples')
        count = 0
        with open(fuzzy_fact_path) as f:
            for line in f:
                e1, e2, r, score = line.strip().split()
                score = float(score)
                if score < theta:
                    continue
                print(line)
                if '{}\t{}\t{}'.format(e1, e2, r) in removed_triples:
                    continue
                e1_id = self.entity2id[e1]
                e2_id = self.entity2id[e2]
                r_id = self.relation2id[r]
                if not r_id in self.adj_list[e1_id]:
                    self.adj_list[e1_id][r_id] = set()
                if not e2_id in self.adj_list[e1_id][r_id]:
                    self.adj_list[e1_id][r_id].add(e2_id)
                    count += 1
                    if count > 0 and count % 1000 == 0:
                        print('{} fuzzy facts added'.format(count))

        self.vectorize_action_space(self.args.data_dir, cache_namespace='fuzzy')

    def get_inv_relation_id(self, r_id):
        return r_id + 1

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings.weight)

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_img_embeddings(self):
        return self.EDropout(self.entity_img_embeddings.weight)

    def get_entity_img_embeddings(self, e):
        return self.EDropout(self.entity_img_embeddings(e))

    def get_relation_img_embeddings(self, r):
        return self.RDropout(self.relation_img_embeddings(r))

    def virtual_step(self, e_set, r):
        """
        Given a set of entities (e_set), find the set of entities (e_set_out) which has at least one incoming edge
        labeled r and the source entity is in e_set.
        """
        batch_size = len(e_set)
        e_set_1D = e_set.view(-1)
        r_space = self.action_space[0][0][e_set_1D]
        e_space = self.action_space[0][1][e_set_1D]
        e_space = (r_space.view(batch_size, -1) == r.unsqueeze(1)).long() * e_space.view(batch_size, -1)
        e_set_out = []
        for i in range(len(e_space)):
            e_set_out_b = var_cuda(unique(e_space[i].data))
            e_set_out.append(e_set_out_b.unsqueeze(0))
        e_set_out = ops.pad_and_cat(e_set_out, padding_value=self.dummy_e)
        return e_set_out

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    def define_modules(self):
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            if self.args.model == 'complex':
                self.entity_img_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        if self.args.model == 'complex':
            self.relation_img_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        if not self.args.relation_only:
            nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID        

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID
