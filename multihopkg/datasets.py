from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple, List

import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.types import Device
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bart import BartTokenizer

from multihopkg.utils.data_structures import DataPartitions

class TestDataset(Dataset):
    __test__ = False # To avoid pytest confusion

    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, candidate_head) if (candidate_head, relation, tail) not in self.triple_set
                   else (-1, head) for candidate_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, candidate_tail) if (head, relation, candidate_tail) not in self.triple_set
                   else (-1, tail) for candidate_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

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
        self.separator_token_id = word_tokenizer.convert_tokens_to_ids([sep_token])
        if isinstance(self.separator_token_id, list):
            self.separator_token_id = self.separator_token_id[0] # Appeases LSP
        assert isinstance(self.separator_token_id, int), f"Expected the separator token to be an integer. Instead we get {self.separator_token_id}"
        self.device = device

        # Get questions and answers a single string but separated by some token.
        self.ques_n_ans, self.answer_masks = self._merge_questions_and_answers(
            dataset.loc[:, DataPartitions.ASSUMED_COLUMNS[0]].tolist(),
            dataset.loc[:, DataPartitions.ASSUMED_COLUMNS[1]].tolist(),
            self.separator_token_id,
        )
        self.path = dataset.loc[:, DataPartitions.ASSUMED_COLUMNS[2]].tolist()
        # Embeddings
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.embeddings_dim = id2ent.embedding_dim

    def _merge_questions_and_answers(
        self, questions: List[List[int]], answers: List[List[int]], sep_token: int
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Merges questions and answers into a single string, separated by a token.
        """
        assert len(questions) == len(answers), "Expected questions and answers to have the same length"
        merged_questions_answers = []
        answer_masks = []

        for question, answer in zip(questions, answers):
            qna = question + answer + [sep_token] # sep_token is both separator and eos
            # <s> question_nonspecial_tokens </s> ans_nonspecial_tokens </s>
            mask = [0] * (len(question)) + [1] * len(answer) + [0]
            merged_questions_answers.append(qna)
            answer_masks.append(mask)

        return merged_questions_answers, answer_masks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # All of these are ids
        qna_tokens = torch.tensor(self.ques_n_ans[idx], dtype=torch.long)
        ans_masks = torch.tensor(self.answer_masks[idx], dtype=torch.long)
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

        return qna_tokens.to(self.device), ans_masks.to(self.device), path_embedding.to(self.device)
