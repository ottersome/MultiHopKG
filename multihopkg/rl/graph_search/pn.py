"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Graph Search Policy Network.
"""

import numpy as np
import pandas

import torch
import torch.nn as nn
import torch.nn.functional as F

import multihopkg.utils.ops as ops
from multihopkg.utils.ops import var_cuda, zeros_var_cuda
from multihopkg.knowledge_graph import ITLKnowledgeGraph, SunKnowledgeGraph
from multihopkg.vector_search import ANN_IndexMan
from multihopkg.environments import Environment, Observation
from typing import Tuple, List, Optional
import pdb


class GraphSearchPolicy(nn.Module):
    def __init__(
        self,
        relation_only: bool,
        history_dim: int,
        history_num_layers: int,
        entity_dim: int,
        relation_dim: int,
        ff_dropout_rate: float,
        xavier_initialization: bool,
        relation_only_in_path: bool,
        reward_module: nn.Module,
    ):
        super(GraphSearchPolicy, self).__init__()
        # WARN: I am erasing self.model because I cannot see it being used anywhere here
        # self.model = model
        self.relation_only = relation_only

        self.history_dim = history_dim
        self.history_num_layers = history_num_layers
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        if self.relation_only:
            self.action_dim = relation_dim
        else:
            self.action_dim = entity_dim + relation_dim
        self.ff_dropout_rate = ff_dropout_rate
        # WARN: Same here. NOt seemingy used anywheres
        # self.rnn_dropout_rate = rnn_dropout_rate
        # self.action_dropout_rate = action_dropout_rate

        self.xavier_initialization = xavier_initialization

        self.relation_only_in_path = relation_only_in_path
        self.path = None

        # Set policy network modules
        self.define_modules()
        self.initialize_modules()

        # Fact network modules
        self.fn = None
        self.fn_kg = None

    # * Added
    def policy_nn_fun(self, X2):
        """
        Input comes from:
            X = self.W1(X)
            X = F.relu(X)
            X = self.W1Dropout(X)
            X = self.W2(X)
            X2 = self.W2Dropout(X)
        """

        mu = self.mu_layer(X2)
        log_sigma = self.sigma_layer(X2)
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)
        sigma = torch.exp(log_sigma)

        # Create a normal distribution using the mean and standard deviation
        dist = torch.distributions.Normal(mu, sigma)
        entropy = dist.entropy().sum(dim=-1)
        return dist, entropy

    # * Added

    def transit(
        self,
        e,
        obs,
        kg,
        use_action_space_bucketing=True,
        merge_aspace_batching_outcome=False,
    ):
        """
        Compute the next action distribution based onsample_action
            current node (entity) in KG and the query relation
            (b) action history representation
        :param e: agent location (node) at step t.
        :param obs: agent observation at step t.
            e_s: source node
            q: query relation
            e_t: target node
            last_step: If set, the agent is carrying out the last step.
            last_r: label of edge traversed in the previous step
            seen_nodes: notes seen on the paths
        :param kg: Knowledge graph environment.
        :param use_action_space_bucketing: If set, group the action space of different nodes
            into buckets by their sizes.
        :param merge_aspace_batch_outcome: If set, merge the transition probability distribution
            generated of different action space bucket into a single batch.
        :return
            With aspace batching and without merging the outcomes:
                db_outcomes: (Dynamic Batch) (action_space, action_dist)
                    action_space: (Batch) padded possible action indices
                    action_dist: (Batch) distribution over actions.
                inv_offset: Indices to set the dynamic batching output back to the original order.
                entropy: (Batch) entropy of action distribution.
            Else:
                action_dist: (Batch) distribution over actions.
                entropy: (Batch) entropy of action distribution.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Representation of the current state (current node and other observations)
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

        # MLP
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2)
                - (1 - action_mask) * ops.HUGE_INT,
                dim=-1,
            )
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
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
            """ """
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_action_space_in_buckets(
                e, obs, kg
            )
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X2[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            inv_offset = [
                i for i, _ in sorted(enumerate(references), key=lambda x: x[1])
            ]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            if merge_aspace_batching_outcome:
                db_action_dist = []
                for _, action_dist in db_outcomes:
                    db_action_dist.append(action_dist)
                action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
                action_dist = ops.pad_and_cat(db_action_dist, padding_value=0)[
                    inv_offset
                ]
                db_outcomes = [(action_space, action_dist)]
                inv_offset = None
        else:
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2)
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        #! entropy should not be returned
        return db_outcomes, inv_offset, entropy

    def initialize_path(self, init_action, kg):
        # [batch_size, action_dim]
        if self.relation_only_in_path:
            init_action_embedding = kg.get_relation_embeddings(init_action[0])
        else:
            init_action_embedding = self.get_action_embedding(init_action, kg)
        init_action_embedding.unsqueeze_(1)
        # [num_layers, batch_size, dim]
        init_h = zeros_var_cuda(
            [self.history_num_layers, len(init_action_embedding), self.history_dim]
        )
        init_c = zeros_var_cuda(
            [self.history_num_layers, len(init_action_embedding), self.history_dim]
        )
        self.path = [self.path_encoder(init_action_embedding, (init_h, init_c))[1]]

    def update_path(self, action, kg, offset=None):
        """
        Once an action was selected, update the action history.
        :param action (r, e): (Variable:batch) indices of the most recent action
            - r is the most recently traversed edge;
            - e is the destination entity.
        :param offset: (Variable:batch) if None, adjust path history with the given offset, used for search
        :param KG: Knowledge graph environment.
        """

        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        # update action history
        if self.relation_only_in_path:
            action_embedding = kg.get_relation_embeddings(action[0])
        else:
            action_embedding = self.get_action_embedding(action, kg)
        if offset is not None:
            offset_path_history(self.path, offset)

        self.path.append(
            self.path_encoder(action_embedding.unsqueeze(1), self.path[-1])[1]
        )

    def get_action_space_in_buckets(self, e, obs, kg, collapse_entities=False):
        """
        To compute the search operation in batch, we group the action spaces of different states
        (i.e. the set of outgoing edges of different nodes) into buckets based on their sizes to
        save the memory consumption of paddings.

        For example, in large knowledge graphs, certain nodes may have thousands of outgoing
        edges while a long tail of nodes only have a small amount of outgoing edges. If a batch
        contains a node with 1000 outgoing edges while the rest of the nodes have a maximum of
        5 outgoing edges, we need to pad the action spaces of all nodes to 1000, which consumes
        lots of memory.

        With the bucketing approach, each bucket is padded separately. In this case the node
        with 1000 outgoing edges will be in its own bucket and the rest of the nodes will suffer
        little from padding the action space to 5.

        Once we grouped the action spaces in buckets, the policy network computation is carried
        out for every bucket iteratively. Once all the computation is done, we concatenate the
        results of all buckets and restore their original order in the batch. The computation
        outside the policy network module is thus unaffected.

        :return db_action_spaces:
            [((r_space_b0, r_space_b0), action_mask_b0),
             ((r_space_b1, r_space_b1), action_mask_b1),
             ...
             ((r_space_bn, r_space_bn), action_mask_bn)]

            A list of action space tensor representations grouped in n buckets, s.t.
            r_space_b0.size(0) + r_space_b1.size(0) + ... + r_space_bn.size(0) = e.size(0)

        :return db_references:
            [l_batch_refs0, l_batch_refs1, ..., l_batch_refsn]
            l_batch_refsi stores the indices of the examples in bucket i in the current batch,
            which is used later to restore the output results to the original order.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        assert len(e) == len(last_r)
        assert len(e) == len(e_s)
        assert len(e) == len(q)
        assert len(e) == len(e_t)
        db_action_spaces, db_references = [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entity2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            batch_ref = {}
            for i in range(len(e)):
                key = int(key1[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                batch_ref[key].append(i)
            for key in batch_ref:
                action_space = kg.action_space_buckets[key]
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
                l_batch_refs = batch_ref[key]
                g_bucket_ids = key2[l_batch_refs].tolist()
                r_space_b = action_space[0][0][g_bucket_ids]
                e_space_b = action_space[0][1][g_bucket_ids]
                action_mask_b = action_space[1][g_bucket_ids]
                e_b = e[l_batch_refs]
                last_r_b = last_r[l_batch_refs]
                e_s_b = e_s[l_batch_refs]
                q_b = q[l_batch_refs]
                e_t_b = e_t[l_batch_refs]
                seen_nodes_b = seen_nodes[l_batch_refs]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)
                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)

        return db_action_spaces, db_references

    def get_action_space(self, e, obs, kg):
        r_space, e_space = kg.action_space[0][0][e], kg.action_space[0][1][e]
        action_mask = kg.action_space[1][e]
        action_space = ((r_space, e_space), action_mask)
        return self.apply_action_masks(action_space, e, obs, kg)

    def apply_action_masks(self, action_space, e, obs, kg):
        (r_space, e_space), action_mask = action_space
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Prevent the agent from selecting the ground truth edge
        ground_truth_edge_mask = self.get_ground_truth_edge_mask(
            e, r_space, e_space, e_s, q, e_t, kg
        )
        action_mask -= ground_truth_edge_mask
        self.validate_action_mask(action_mask)

        # Mask out false negatives in the final step
        if last_step:
            false_negative_mask = self.get_false_negative_mask(e_space, e_s, q, e_t, kg)
            action_mask *= 1 - false_negative_mask
            self.validate_action_mask(action_mask)

        # Prevent the agent from stopping in the middle of a path
        # stop_mask = (last_r == NO_OP_RELATION_ID).unsqueeze(1).float()
        # action_mask = (1 - stop_mask) * action_mask + stop_mask * (r_space == NO_OP_RELATION_ID).float()
        # Prevent loops
        # Note: avoid duplicate removal of self-loops
        # seen_nodes_b = seen_nodes[l_batch_refs]
        # loop_mask_b = (((seen_nodes_b.unsqueeze(1) == e_space.unsqueeze(2)).sum(2) > 0) *
        #      (r_space != NO_OP_RELATION_ID)).float()
        # action_mask *= (1 - loop_mask_b)
        return (r_space, e_space), action_mask

    def get_ground_truth_edge_mask(self, e, r_space, e_space, e_s, q, e_t, kg):
        ground_truth_edge_mask = (
            (e == e_s).unsqueeze(1)
            * (r_space == q.unsqueeze(1))
            * (e_space == e_t.unsqueeze(1))
        )
        inv_q = kg.get_inv_relation_id(q)
        inv_ground_truth_edge_mask = (
            (e == e_t).unsqueeze(1)
            * (r_space == inv_q.unsqueeze(1))
            * (e_space == e_s.unsqueeze(1))
        )
        return (
            (ground_truth_edge_mask + inv_ground_truth_edge_mask)
            * (e_s.unsqueeze(1) != kg.dummy_e)
        ).float()

    def get_answer_mask(self, e_space, e_s, q, kg):
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_object_vectors
        else:
            answer_vectors = kg.train_object_vectors
        answer_masks = []
        for i in range(len(e_space)):
            _e_s, _q = int(e_s[i]), int(q[i])
            if not _e_s in answer_vectors or not _q in answer_vectors[_e_s]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e_s][_q]
            answer_mask = torch.sum(
                e_space[i].unsqueeze(0) == answer_vector, dim=0
            ).long()
            answer_masks.append(answer_mask)
        answer_mask = torch.cat(answer_masks).view(len(e_space), -1)
        return answer_mask

    def get_false_negative_mask(self, e_space, e_s, q, e_t, kg):
        answer_mask = self.get_answer_mask(e_space, e_s, q, kg)
        # This is a trick applied during training where we convert a multi-answer predction problem into several
        # single-answer prediction problems. By masking out the other answers in the training set, we are forcing
        # the agent to walk towards a particular answer.
        # This trick does not affect inference on the test set: at inference time the ground truth answer will not
        # appear in the answer mask. This can be checked by uncommenting the following assertion statement.
        # Note that the assertion statement can trigger in the last batch if you're using a batch_size > 1 since
        # we append dummy examples to the last batch to make it the required batch size.
        # The assertion statement will also trigger in the dev set inference of NELL-995 since we randomly
        # sampled the dev set from the training data.
        # assert(float((answer_mask * (e_space == e_t.unsqueeze(1)).long()).sum()) == 0)
        false_negative_mask = (
            answer_mask * (e_space != e_t.unsqueeze(1)).long()
        ).float()
        return false_negative_mask

    def validate_action_mask(self, action_mask):
        action_mask_min = action_mask.min()
        action_mask_max = action_mask.max()
        assert action_mask_min == 0 or action_mask_min == 1
        assert action_mask_max == 0 or action_mask_max == 1

    def get_action_embedding(self, action, kg):
        """
        Return (batch) action embedding which is the concatenation of the embeddings of
        the traversed edge and the target node.

        :param action (r, e):
            (Variable:batch) indices of the most recent action
                - r is the most recently traversed edge
                - e is the destination entity.
        :param kg: Knowledge graph enviroment.
        """
        r, e = action
        relation_embedding = kg.get_relation_embeddings(r)
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = kg.get_entity_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def define_modules(self):
        if self.relation_only:
            input_dim = self.history_dim + self.relation_dim
        elif self.relation_only_in_path:
            input_dim = self.history_dim + self.entity_dim * 2 + self.relation_dim
        else:
            input_dim = self.history_dim + self.entity_dim + self.relation_dim

        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        if self.relation_only_in_path:
            self.path_encoder = nn.LSTM(
                input_size=self.relation_dim,
                hidden_size=self.history_dim,
                num_layers=self.history_num_layers,
                batch_first=True,
            )
        else:
            self.path_encoder = nn.LSTM(
                input_size=self.action_dim,
                hidden_size=self.history_dim,
                num_layers=self.history_num_layers,
                batch_first=True,
            )

    def initialize_modules(self):
        if self.xavier_initialization:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            for name, param in self.path_encoder.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.xavier_normal_(param)


class ITLGraphEnvironment(Environment, nn.Module):

    def __init__(
        self,
        question_embedding_module: nn.Module,  # Generally a BertModel
        question_embedding_module_trainable: bool,
        entity_dim: int,
        ff_dropout_rate: float,
        history_dim: int,
        history_num_layers: int,
        knowledge_graph: SunKnowledgeGraph,
        relation_dim: int,

        node_data: str,
        node_data_key: str,
        rel_data: str,
        rel_data_key: str,					  
        ann_index_manager_ent: ANN_IndexMan,
        ann_index_manager_rel: ANN_IndexMan,
        steps_in_episode: int,
    ):
        super(ITLGraphEnvironment, self).__init__()
        # Should be injected via information extracted from Knowledge Grap
        self.action_dim = relation_dim  # TODO: Ensure this is a solid default
        self.question_embedding_module_trainable = question_embedding_module_trainable
        self.entity_dim = entity_dim
        self.ff_dropout_rate = ff_dropout_rate
        self.history_dim = history_dim  # History is STATE
        self.history_encoder_num_layers = history_num_layers
        self.knowledge_graph = knowledge_graph
        self.padding_value = (
            question_embedding_module.config.pad_token_id
        )  # TODO (mega): Confirm this is correct to get the padding value
        self.path = None
        self.relation_dim = relation_dim
        self.ann_index_manager_ent = ann_index_manager_ent
        self.ann_index_manager_rel = ann_index_manager_rel
        self.steps_in_episode = steps_in_episode

        self.entity2title = {}
        self.relation2title = {}

        if node_data: # Enters if node_data is neither a NoneType or an empty string
            # Extracts the dataframe containing the special encoding name (i.e., MID) and proper title (i.e., Title)
            node_df = pandas.read_csv(node_data).fillna('')
            self.entity2title = node_df.set_index(node_data_key)['Title'].to_dict()

        if rel_data: # Enters if rel_data is neither a NoneType or an empty string
            # Extracts the dataframe containing the special encoding name (i.e., MID) and proper title (i.e., Title)
            rel_df = pandas.read_csv(rel_data).fillna('')
            self.rel2title = rel_df.set_index(rel_data_key)['Title'].to_dict()
        ########################################
        # Core States (3/5)
        ########################################
        self.current_questions_emb: Optional[torch.Tensor] = None
        self.current_position: Optional[torch.Tensor] = None
        self.current_step_no = (
            self.steps_in_episode
        )  # This value denotes being at "reset" state. As in, when episode is done

        ########################################
        # Get the actual torch modules defined
        # Of most importance is self.path_encoder
        ########################################
        assert isinstance(
            question_embedding_module, nn.Module
        ), "The question embedding module must be a torch.nn.Module, otherwis no computation graph. You passed a {}".format(
            type(question_embedding_module)
        )
        self.question_embedding_module = question_embedding_module
        self.question_dim = self.question_embedding_module.config.hidden_size
        # (self.W1, self.W2, self.W1Dropout, self.W2Dropout, self.path_encoder, self.concat_projector) = (
        (self.concat_projector, self.W2, self.W1Dropout, self.W2Dropout, _) = (
            self._define_modules(
                self.entity_dim,
                self.ff_dropout_rate,
                self.history_dim,
                self.history_encoder_num_layers,
                self.relation_dim,
                self.question_dim,
            )
        )

    def get_llm_embeddings(self, questions: List[np.ndarray]) -> torch.Tensor:
        """
        Will take a list of list of token ids, pad them and then pass them to the embedding module to get single embeddings for each question
        Args:
            - questions (List[List[int]]): The tensor denoting the questions for this batch.
        Return:
            - questions_embeddings (torch.Tensor): The embeddings of the questions.
        """
        # Format the input for the legacy funciton inside
        tensorized_questions = [
            torch.tensor(q).to(torch.int32).view(1, -1) for q in questions
        ]
        # We should conver them to embeddinggs before sending them over

        padded_tokens, attention_mask = ops.pad_and_cat(
            tensorized_questions, padding_value=self.padding_value, padding_dim=1
        )
        embedding_output = self.question_embedding_module(
            input_ids=padded_tokens, attention_mask=attention_mask
        )
        last_hidden_state = embedding_output.last_hidden_state
        # TODO: Figure out if we want to grab a single one of the embeddings or just aggregaate them through mean.
        final_embedding = last_hidden_state.mean(dim=1)

        return final_embedding

    # TOREM: We need to test if this can replace forward for now.
    def step(self, actions: torch.Tensor) -> Observation:
        """
        This one will simply find the closes emebdding in our class and dump it here as an observation.
        Args:
            - actions (torch.Tensor): Shall be of shape (batch_size, action_dimension)
        Return:
            - observations (torch.Tensor): The observations at the current state. Shape: (batch_size, observation_dim)
        """
        assert isinstance(
            self.current_position, torch.Tensor
        ), f"invalid self.current_position, type: {type(self.current_position)}. Please make sure to run ITLKnowledgeGraph::rest() before running get_observations."

        self.current_step_no += 1

        # Make sure action and current position are detached from computation graph
        detached_actions = actions.detach()
        detached_curpos = self.current_position.detach()

        assert isinstance(
            self.current_questions_emb, torch.Tensor
        ), f"self.current_questions_emb (type: {type(self.current_questions_emb)}) must be set via `reset` before calling this."

        ########################################
        # ANN mostly for debugging for now
        ########################################
        # new_pos = detached_curpos + detached_actions
        new_pos = self.knowledge_graph.sun_model.flexible_forward_rotate(
            detached_curpos, detached_actions
        )

        print(f"new_pos.type: {type(new_pos)}")
        print(f"new_pos.shape: {new_pos.shape}")
        matched_entity_embeddings, corresponding_ent_idxs = self.ann_index_manager_ent.search(
            new_pos, topk=1
        )
        # self.current_position = ann_matches # Either
        self.current_position = new_pos  # Or

        ########################################
        # Projections
        ########################################
        concatenations = torch.cat(
            [self.current_questions_emb, self.current_position], dim=-1
        )
        projected_state = self.concat_projector(concatenations)

        # TODO: Think about this, we have a transformer so I dont think we need to do this.
        # concatenations_w_sequence = concatenations.unsqueeze(1)
        # cur_state, _ = self.path_encoder(concatenations)

        # Corresponding indices is a list of indices of the matched embeddings (batch_size, topk=1)
        observation = Observation(
            position=matched_entity_embeddings,
            position_id=corresponding_ent_idxs,
            state=projected_state,
        )

        return observation

    def _define_modules(
        self,
        entity_dim: int,
        ff_dropout_rate: float,
        history_dim: int,
        history_num_layers: int,
        relation_dim: int,
        question_dim: int,
    ) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        # We assume both relationships and entityes have mbeddings
        print(f"entity_dim: {entity_dim}, relation_dim: {relation_dim}")
        # input_dim = history_dim + entity_dim + relation_dim
        # We assume action_dim is relation_dim
        action_dim = relation_dim
        # input_dim = action_dim + question_dim
        # input_dim = action_dim + question_dim
        input_dim = entity_dim + relation_dim + question_dim

        # W1 = nn.Linear(input_dim, action_dim)
        # W2 = nn.Linear(action_dim, action_dim)
        W1 = nn.Linear(input_dim, history_dim)
        W2 = nn.Linear(
            history_dim, action_dim
        )  # We ignore this for now, leave it so that file runs
        W1Dropout = nn.Dropout(p=ff_dropout_rate)
        W2Dropout = nn.Dropout(p=ff_dropout_rate)  # Same ignore here

        # # TODO: Check if we actually want to use lstm, we have a tranformer with positional encoding so I dont think we need this.
        path_encoder = nn.LSTM(
            input_size=action_dim + question_dim,
            hidden_size=history_dim,  # AFAIK equiv this output size
            num_layers=history_num_layers,
            batch_first=True,
        )

        # State Variables for holding rollout information
        # I might regret this
        self.current_position = None

        return W1, W2, W1Dropout, W2Dropout, path_encoder

    def reset(self, initial_states_info: torch.Tensor) -> Observation:
        """
        Will reset the episode to the initial position
        This will happen by grabbign the initial_states_info embeddings, concatenating them with the centroid and then passing them to the environment
        Args:
            - initial_state_info (torch.Tensor): In this implemntation sit is the initial_states_info
        Returnd:
            - postion (torch.Tensor): Position in the graph
            - state (torch.Tensor): Aggregation of states visited so far summarized in a single vector per batch element.
        """

        # Sanity Check: Make sure we finilized previos epsiode correclty
        if self.current_step_no != self.steps_in_episode:
            raise RuntimeError(
                "Mis-use of the environment. Episode step must've been set back to 0 before end."
                " Maybe you did not end your episode correctly"
            )
        ## Values
        # Local Alias: initial_states_info is just a name we stick to in order to comply with inheritance of Environment.
        self.current_questions_emb = initial_states_info  # (batch_size, emb_dim)
        self.current_step_no = 0

        # Create more complete representation of state
        centroid = self.knowledge_graph.get_centroid()
        tiled_centroids = centroid.unsqueeze(0).repeat(len(initial_states_info), 1)
        # concatenations = torch.cat([self.current_questions_emb, tiled_centroids], dim=-1)
        concatenations = torch.cat(
            [tiled_centroids, self.current_questions_emb], dim=-1
        )
        projected_concat = self.concat_projector(concatenations)
        # NOTE: We were using path encoder here before and are not sure if removing is good idea.

        # This was for when we were receiving sequences. I mean I gues we still are.
        projected_state = projected_concat
        self.current_step = 0

        self.current_position = centroid.unsqueeze(0).repeat(
            len(initial_states_info), 1
        )

        observation = Observation(
            position=self.current_position.detach().numpy(),
            position_id=np.zeros((self.current_position.shape[0])),
            state=projected_state,
        )

        return observation

    def get_centroid(self) -> torch.Tensor:
        if not self.centroid:
            self.centroid = self.knowledge_graph.get_centroid()
        return self.centroid

    # * This is Nura's code. Might not really bee kj
    def get_action_space(self, e, obs, kg):
        r_space, e_space = kg.action_space[0][0][e], kg.action_space[0][1][e]
        action_mask = kg.action_space[1][e]
        action_space = ((r_space, e_space), action_mask)
        return action_space


# TODO: Implementn ann
def run_ann(approximate_states: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
