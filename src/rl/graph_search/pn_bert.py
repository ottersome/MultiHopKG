"""
BERT-conditioned Graph Search Policy.

Extends the base GraphSearchPolicy by concatenating a BERT-encoded
natural-language question to the state vector X2 right before the
policy scoring function. This keeps the base MLP sizes unchanged
and makes the policy aware of the question.

HuggingFace `bert-base-uncased` is used directly; if the transformers
dependency or weights are unavailable, a zero vector of the configured
size is used as a fallback to keep the pipeline runnable.
"""
import os
import json
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel  # type: ignore

from src.rl.graph_search.pn import GraphSearchPolicy
import src.utils.ops as ops


class GraphSearchPolicyBert(GraphSearchPolicy):
    def __init__(self, args, pretrained_question_model):
        # Question encoder config
        self.question_repr_dim = int(getattr(args, 'bert_hidden_size', 768))
        self.bert_model_name = pretrained_question_model
        self.max_question_len = int(getattr(args, 'max_question_len', 64))
        self.question_texts_path = getattr(args, 'question_texts_path', '') or ''
        self._texts: Dict[Tuple[int, int], str] = {}

        # Initialize base modules first
        super(GraphSearchPolicyBert, self).__init__(args)

        # Fusion layer maps [X2; Q_txt] -> action_dim so we can reuse action embeddings
        self.fusion = nn.Linear(self.action_dim + self.question_repr_dim, self.action_dim)

        # Lazy-load HF encoder; keep model eval/frozen
        self._tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self._bert = AutoModel.from_pretrained(self.bert_model_name)
        self._bert.eval()
        for p in self._bert.parameters():
            p.requires_grad = False

        # Optional mapping from (e_s, relation) to question text
        if self.question_texts_path and os.path.exists(self.question_texts_path):
            self._load_question_texts(self.question_texts_path)

    def _load_question_texts(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.tsv':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        continue
                    try:
                        e_s = int(parts[0]); r = int(parts[1]); q = parts[2]
                    except Exception:
                        continue
                    self._texts[(e_s, r)] = q
        elif ext in ('.jsonl', '.json'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        e_s = int(obj['e_s']); r = int(obj['r']); q = str(obj['question'])
                    except Exception:
                        continue
                    self._texts[(e_s, r)] = q

    def _encode_questions(self, e_s: torch.Tensor, q_rel: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Return [B, question_repr_dim] question embeddings.
        If no texts or HF unavailable, return zeros on the right device.
        """
        batch = e_s.size(0)
        if not self._texts:
            return torch.zeros((batch, self.question_repr_dim), device=device)
        # Form texts batch
        e_s_list = e_s.detach().flatten().tolist()
        q_list = q_rel.detach().flatten().tolist()
        texts: List[str] = [self._texts.get((int(es), int(qr)), '') for es, qr in zip(e_s_list, q_list)]
        if all(t == '' for t in texts):
            return torch.zeros((batch, self.question_repr_dim), device=device)
        if self._tokenizer is None or self._bert is None:
            return torch.zeros((batch, self.question_repr_dim), device=device)
        with torch.no_grad():
            # Ensure model on the right device
            self._bert.to(device)
            toks = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_question_len,
                return_tensors='pt'
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out = self._bert(**toks)
            pooled = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None \
                else out.last_hidden_state[:, 0, :]
            # If hidden size differs, project to desired size on the fly
            if pooled.size(-1) != self.question_repr_dim:
                W = torch.randn(pooled.size(-1), self.question_repr_dim, device=device)
                pooled = pooled @ W
            return pooled

    def transit(self, e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        # Unpack and compute base features as in parent
        src_entity, nlp_question, target_entity, last_step, last_r, seen_nodes = obs

        H = self.path[-1][0][-1, :, :]
        # Q Was removed here: SO below wont work
        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_embeddings(src_entity)
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, E_s, Q], dim=-1)
        else:
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, Q], dim=-1)

        # Base MLP (unchanged)
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        # Concatenate question representation to X2 and fuse to action_dim
        Q_txt = self._encode_questions(question_relation, X2.device)
        X2q = torch.cat([X2, Q_txt], dim=-1)
        X2_fused = self.fusion(X2q)

        def policy_nn_fun(X2_in, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2_in, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            return action_dist, ops.entropy(action_dist)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        if use_action_space_bucketing:
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_action_space_in_buckets(e, obs, kg)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X2_fused[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            return db_outcomes, inv_offset, entropy
        else:
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2_fused, action_space)
            return [(action_space, action_dist)], None, entropy
