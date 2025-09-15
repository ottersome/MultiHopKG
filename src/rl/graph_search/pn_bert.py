"""
BERT-conditioned Graph Search Policy.

Extends the base GraphSearchPolicy by concatenating a question representation
to the state vector X before the MLP. The question embedding is produced by
src.nlp.question_encoder.QuestionEncoder.

This class is drop-in compatible with PolicyGradient/RewardShapingPolicyGradient
since it preserves method signatures and computes the question embedding inside
transit() using (e_s, q, kg) from obs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.graph_search.pn import GraphSearchPolicy
from src.nlp.question_encoder import QuestionEncoder


class GraphSearchPolicyBert(GraphSearchPolicy):
    def __init__(self, args):
        self.question_repr_dim = int(getattr(args, 'bert_hidden_size', 768))
        super(GraphSearchPolicyBert, self).__init__(args)
        # Override modules now that base constructor called define_modules() — redefine with extra dim
        self.question_encoder = QuestionEncoder(args)
        self._redefine_with_question()

    def _redefine_with_question(self):
        # Re-define the MLP and path encoder with augmented input size
        if self.relation_only:
            input_dim = self.history_dim + self.relation_dim + self.question_repr_dim
        elif self.relation_only_in_path:
            input_dim = self.history_dim + self.entity_dim * 2 + self.relation_dim + self.question_repr_dim
        else:
            input_dim = self.history_dim + self.entity_dim + self.relation_dim + self.question_repr_dim

        # Replace MLP layers to match new input size
        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        # Dropouts keep rate from args
        # path_encoder remains identical (input is previous action embedding), so no change

    def transit(self, e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        # Unpack and compute base features as in parent, then append question embedding
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        Q = kg.get_relation_embeddings(q)
        H = self.path[-1][0][-1, :, :]
        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_embeddings(e_s)
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, E_s, Q], dim=-1)
        else:
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, Q], dim=-1)

        # Append question representation
        Q_txt = self.question_encoder.encode_batch(e_s, q, kg)
        if Q_txt.device != X.device:
            Q_txt = Q_txt.to(X.device)
        X = torch.cat([X, Q_txt], dim=-1)

        # MLP as in base class
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * 1e20, dim=-1)
            return action_dist, (-(action_dist * torch.log(action_dist + 1e-20)).sum(dim=-1))

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = torch.nn.functional.pad(torch.cat(db_r_space, dim=0), (0, 0))  # no-op, keep structure
            e_space = torch.nn.functional.pad(torch.cat(db_e_space, dim=0), (0, 0))
            action_mask = torch.nn.functional.pad(torch.cat(db_action_mask, dim=0), (0, 0))
            # Restore original order for current batch slice
            r_space = r_space[inv_offset]
            e_space = e_space[inv_offset]
            action_mask = action_mask[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        if use_action_space_bucketing:
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_action_space_in_buckets(e, obs, kg)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            return db_outcomes, inv_offset, entropy
        else:
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2, action_space)
            return [(action_space, action_dist)], None, entropy
