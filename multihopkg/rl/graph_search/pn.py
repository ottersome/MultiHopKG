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

from multihopkg.exogenous.sun_models import KGEModel, get_embeddings_from_indices
import multihopkg.utils.ops as ops
from multihopkg.utils.ops import var_cuda, zeros_var_cuda
from multihopkg.vector_search import ANN_IndexMan_AbsClass
from multihopkg.environments import Environment, Observation
from typing import Tuple, List, Dict, Optional
import pdb

import sys
import random

class ITLGraphEnvironment(Environment, nn.Module):

    def __init__(
        self,
        question_embedding_module: nn.Module,  # Generally a BertModel
        question_embedding_module_trainable: bool,
        entity_dim: int,
        ff_dropout_rate: float,
        history_dim: int,
        history_num_layers: int,
        knowledge_graph: KGEModel,
        relation_dim: int,
        nav_start_emb_type: str,
        node_data: Optional[str],
        node_data_key: Optional[str],
        rel_data: Optional[str],
        rel_data_key: Optional[str],
        id2entity: Dict[int, str],
        entity2id: Dict[str, int],
        id2relation: Dict[int, str],
        relation2id: Dict[str, int],				  
        ann_index_manager_ent: ANN_IndexMan_AbsClass,
        ann_index_manager_rel: ANN_IndexMan_AbsClass,
        steps_in_episode: int,
        trained_pca,
        graph_pca,
        graph_annotation: str,
        num_rollouts: int = 0, # Number of trajectories to be used in the environment per question, 0 means 1 trajectory
        use_kge_question_embedding: bool = False,
        epsilon: float = 0.1, # For error margin in the distance, TODO: Must find a better value
        add_transition_state: bool = False, # If True, will include the transition state in the observation
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
        self._num_rollouts = num_rollouts  # Number of trajectories to be used in the environment per question
        self.steps_in_episode = steps_in_episode
        self.trained_pca = trained_pca
        self.graph_pca = graph_pca
        self.graph_annotation = graph_annotation

        self.id2entity = id2entity
        self.entity2id = entity2id
        self.id2relation = id2relation
        self.relation2id = relation2id
        
        self.entity2title = {}
        self.relation2title = {}

        if node_data: # Enters if node_data is neither a NoneType or an empty string
            # Extracts the dataframe containing the special encoding name (i.e., MID) and proper title (i.e., Title)
            node_df = pandas.read_csv(node_data).fillna('')
            self.entity2title = node_df.set_index(node_data_key)['Title'].to_dict()

        if rel_data: # Enters if rel_data is neither a NoneType or an empty string
            # Extracts the dataframe containing the special encoding name (i.e., MID) and proper title (i.e., Title)
            rel_df = pandas.read_csv(rel_data).fillna('')
            self.relation2title = rel_df.set_index(rel_data_key)['Title'].to_dict()
        ########################################
        # Core States (3/5)
        ########################################
        self.current_questions_emb: Optional[torch.Tensor] = None
        self.current_position: Optional[torch.Tensor] = None
        self.current_step_no = (
            self.steps_in_episode
        )  # This value denotes being at "reset" state. As in, when episode is done

        assert nav_start_emb_type in ['centroid', 'random', 'relevant'], f"Invalid start_embedding_type: {nav_start_emb_type}"
        self.nav_start_emb_type = nav_start_emb_type
        self.start_emb_func = {
            'centroid': self.get_centroid_embedding,
            'random': self.get_random_embedding,
            'relevant': self.get_relevant_embedding
        }

        ########################################
        # Get the actual torch modules defined
        # Of most importance is self.path_encoder
        ########################################
        assert isinstance(
            question_embedding_module, nn.Module
        ), "The question embedding module must be a torch.nn.Module, otherwis no computation graph. You passed a {}".format(
            type(question_embedding_module)
        )
        self.use_kge_question_embedding = use_kge_question_embedding

        self.question_embedding_module = question_embedding_module # TODO: Consider moving to if condition if unused
        if self.use_kge_question_embedding: # use the entity and relation embeddings as the question embedding
            self.question_dim = self.entity_dim + self.relation_dim
        else:
            self.question_dim = self.question_embedding_module.config.hidden_size

        self.answer_embeddings = None  # This is the embeddings of the answer (batch_size, entity_dim)
        self.answer_found = None       # This is a flag to denote if the answer has been already been found (batch_size, 1)
        self.epsilon = epsilon                 # This is the error margin in the distance for finding the answer
        self.add_transition_state = add_transition_state # If True, will include the observation triplet in the state

        # (self.W1, self.W2, self.W1Dropout, self.W2Dropout, self.path_encoder, self.concat_projector) = (
        # (self.concat_projector, self.W2, self.W1Dropout, self.W2Dropout, _) = (
        # LG: Changes for clarity sake
        ( _, _, _, _, self.path_encoder, self.concat_projector) = (
            self._define_modules(
                self.entity_dim,
                self.ff_dropout_rate,
                self.history_dim,
                self.history_encoder_num_layers,
                self.relation_dim,
                self.question_dim,
            )
        )

    def get_kge_question_embedding(self, entities: List[np.ndarray], relations: List[np.ndarray], device: torch.device) -> torch.Tensor:
        # Under the assumption that there is only one relevant entity per question
        relevant_rels_temp = [rels[0] for rels in relations]
        rel_tensor = get_embeddings_from_indices(
            self.knowledge_graph.relation_embedding,
            torch.tensor(relevant_rels_temp, dtype=torch.int),
        )

        # Under the assumption that there is only one relevant entity per question
        relevant_entities_temp = [ents[0] for ents in entities]
        ent_tensor = get_embeddings_from_indices(
            self.knowledge_graph.entity_embedding,
            torch.tensor(relevant_entities_temp, dtype=torch.int),
        )

        return torch.cat([ent_tensor, rel_tensor], dim=-1).to(device) # Shape: (batch, 2*embedding_dim)

    def get_llm_embeddings(self, questions: List[np.ndarray], device: torch.device) -> torch.Tensor:
        """
        Will take a list of list of token ids, pad them and then pass them to the embedding module to get single embeddings for each question
        Args:
            - questions (List[List[int]]): The tensor denoting the questions for this batch.
        Return:
            - questions_embeddings (torch.Tensor): The embeddings of the questions.
        """
        # Format the input for the legacy funciton inside
        tensorized_questions = [
            torch.tensor(q).to(torch.int32).to(device).view(1, -1) for q in questions
        ]
        # We should conver them to embeddinggs before sending them over

        padded_tokens, attention_mask = ops.pad_and_cat(
            tensorized_questions, padding_value=self.padding_value, padding_dim=1
        )
        attention_mask = attention_mask.to(device)
        embedding_output = self.question_embedding_module( input_ids=padded_tokens, attention_mask=attention_mask)
        last_hidden_state = embedding_output.last_hidden_state
        # TODO: Figure out if we want to grab a single one of the embeddings or just aggregaate them through mean.
        final_embedding = last_hidden_state.mean(dim=1)

        return final_embedding

    def reset(self, initial_states_info: torch.Tensor, answer_ent: List[int], query_ent: List[int] = None, warmup: bool = True) -> Observation:
        """
        Will reset the episode to the initial position
        This will happen by grabbing the initial_states_info embeddings, concatenating them with the centroid and then passing them to the environment
        Args:
            - initial_state_info (torch.Tensor): In this implementation sit is the initial_states_info
            - answer_ent (List[int]): The answer entity for the current batch
            - query_ent (List[int]): The relevant entities for the current batch
        Returns:
            - observation (Observation): Observation object containing the state and the KGE embeddings
        """

        # Sanity Check: Make sure we finilized previos epsiode correclty
        if self.current_step_no != self.steps_in_episode and not(warmup):
            raise RuntimeError(
                "Mis-use of the environment. Episode step must've been set back to 0 before end."
                " Maybe you did not end your episode correctly"
            )
        
        device = self.path_encoder.parameters().__next__().device

        if self.training: self.num_rollouts = self._num_rollouts
        else: self.num_rollouts = 0

        with torch.no_grad():
            ## Values
            # Local Alias: initial_states_info is just a name we stick to in order to comply with inheritance of Environment.
            self.current_questions_emb = initial_states_info                                                                    # (batch_size, text_dim)
            self.current_step_no = 0

            # get the embeddings of the answer entities
            self.answer_embeddings = self.knowledge_graph.get_starting_embedding('relevant', answer_ent).detach()               # (batch_size, entity_dim)
            self.answer_found = torch.zeros((len(answer_ent),1), dtype=torch.bool).to(self.answer_embeddings.device).detach()   # (batch_size, 1)

            init_emb = self.start_emb_func[self.nav_start_emb_type](len(initial_states_info), query_ent).to(device)             # (batch_size, entity_dim)
            self.current_position = init_emb.clone()                                                                            # (batch_size, entity_dim)


        # Initialize Hidden State
        # self.hidden_state = torch.zeros(self.path_encoder.num_layers, len(answer_ent), self.path_encoder.hidden_size).to(device)
        # self.cell_state = torch.zeros(self.path_encoder.num_layers, len(answer_ent), self.path_encoder.hidden_size).to(device)

        # dummy_action = torch.zeros((len(answer_ent), self.action_dim)).to(device)

        # ! Inspecting projections (gradients variance is too high from the start)

        self.q_projected = self.concat_projector(self.current_questions_emb)                                        # (batch_size, emb_dim)

        if self.num_rollouts > 0:
            # Expand the states to the number of rollouts
            # (batch_size, emb_dim) -> (batch_size, num_rollouts, emb_dim)
            self.q_projected = self.q_projected.unsqueeze(1).expand(-1, self.num_rollouts, -1)                      # (batch_size, num_rollouts, entity_dim + relation_dim)
            self.current_questions_emb = self.current_questions_emb.unsqueeze(1).expand(-1, self.num_rollouts, -1)  # (batch_size, num_rollouts, text_dim)
            self.current_position = self.current_position.unsqueeze(1).expand(-1, self.num_rollouts, -1)            # (batch_size, num_rollouts, entity_dim)Z
            self.answer_embeddings = self.answer_embeddings.unsqueeze(1).expand(-1, self.num_rollouts, -1)          # (batch_size, num_rollouts, entity_dim)
            self.answer_found = self.answer_found.unsqueeze(1).expand(-1, self.num_rollouts, -1)                    # (batch_size, num_rollouts, 1)
            init_emb = init_emb.unsqueeze(1).expand(-1, self.num_rollouts, -1)                                      # (batch_size, num_rollouts, entity_dim)

        if self.add_transition_state:
            actions = torch.zeros_like(init_emb).to(device)  # (batch_size, num_rollouts, entity_dim)
            projected_state = torch.cat(
                [self.q_projected, init_emb, actions, init_emb], dim=-1
            ) # (batch_size, emb_dim + 2*entity_dim + action_dim) or (batch_size, num_rollouts, emb_dim + 2*entity_dim + action_dim)
        else:
            projected_state = torch.cat(
                [self.q_projected, init_emb], dim=-1
            ) # (batch_size, emb_dim + entity_dim) or (batch_size, num_rollouts, emb_dim + entity_dim)

        # projected_state = torch.cat(
        #     [self.q_projected, dummy_action], dim=-1
        # )

        observation = Observation(
            state=projected_state,
            kge_cur_pos=self.current_position,
            kge_prev_pos=torch.zeros_like(self.current_position.detach()),
            kge_action=torch.zeros(self.action_dim),
        )

        return observation

    # TOREM: We need to test if this can replace forward for now.
    def step(self, actions: torch.Tensor) -> Tuple[Observation, torch.Tensor, torch.Tensor]:
        """
        This one will simply find the closes emebdding in our class and dump it here as an observation.
        Args:
            - actions (torch.Tensor): Shall be of shape (batch_size, action_dimension)
        Return:
            - observations (torch.Tensor): The observations at the current state. Shape: (batch_size, observation_dim)
            - rewards (torch.Tensor) (float): The rewards at the current state. Shape: (batch_size, 1)
            - dones (torch.Tensor) (bool): The dones at the current state. Shape: (batch_size, 1)
        """
        assert isinstance(
            self.current_position, torch.Tensor
        ), f"invalid self.current_position, type: {type(self.current_position)}. Please make sure to run ITLKnowledgeGraph::rest() before running get_observations."

        self.current_step_no += 1

        # Make sure action and current position are detached from computation graph
        detached_actions = actions.detach()                 # (batch_size, action_dim) or (batch_size, num_rollouts, action_dim)
        detached_curpos = self.current_position.detach()    # (batch_size, entity_dim) or (batch_size, num_rollouts, entity_dim)

        assert isinstance(
            self.current_questions_emb, torch.Tensor
        ), f"self.current_questions_emb (type: {type(self.current_questions_emb)}) must be set via `reset` before calling this."

        ########################################
        # ANN mostly for debugging for now
        ########################################

        # ! Restraining the movement to the neighborhood
        prev_position = self.current_position.clone() # (batch_size, entity_dim) or (batch_size, num_rollouts, entity_dim)

        self.current_position = self.knowledge_graph.flexible_forward(
            self.current_position, actions, 
        ) # (batch_size, entity_dim) or (batch_size, num_rollouts, entity_dim)

        # No gradients are calculated here
        with torch.no_grad():
            diff = self.knowledge_graph.absolute_difference(self.answer_embeddings, self.current_position) # (batch_size, entity_dim) or (batch_size, num_rollouts, entity_dim)
            
            found_ans = torch.norm(diff, dim=-1, keepdim=True) < self.epsilon   # (batch_size, 1) or (batch_size, num_rollouts, 1))
            self.answer_found = torch.logical_or(self.answer_found, found_ans)  # (batch_size, 1) or (batch_size, num_rollouts, 1)
            extrinsic_reward = found_ans.float()                                # (batch_size, 1) or (batch_size, num_rollouts, 1)


        ########################################
        # Projections
        ########################################
        # ! Inspecting projections (gradients variance is too high from the start)
        if self.add_transition_state:
            projected_state = torch.cat(
                [self.q_projected, prev_position, actions, self.current_position], dim=-1 # query,
            ) # (batch_size, emb_dim + 2*entity_dim + action_dim) or (batch_size, num_rollouts, emb_dim + 2*entity_dim + action_dim)
        else:
            projected_state = torch.cat(
                [self.q_projected, self.current_position], dim=-1 # query,
            ) # (batch_size, emb_dim + entity_dim) or (batch_size, num_rollouts, emb_dim + entity_dim)

        # Corresponding indices is a list of indices of the matched embeddings (batch_size, topk=1)
        observation = Observation(
            state=projected_state,
            kge_cur_pos=self.current_position, #.detach(), # TODO: Check if we need to detach this for reward calculation
            kge_prev_pos=detached_curpos,
            kge_action=detached_actions,
        )
        
        return observation, extrinsic_reward, self.answer_found

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
        input_dim = entity_dim + question_dim

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

        residual_adapter = ResidualAdapter(question_dim, entity_dim + relation_dim)

        # State Variables for holding rollout information
        # I might regret this
        self.current_position = None

        # W1 = nn.LSTM(
        #     input_size=entity_dim + question_dim,
        #     hidden_size=history_dim,  # AFAIK equiv this output size
        #     num_layers=history_num_layers,
        #     batch_first=True,
        # )

        # W1 = AttentionFusion(
        #     semantic_dim=entity_dim + relation_dim,
        #     text_dim=question_dim,
        #     fusion_dim=history_dim,
        # )

        return W1, W2, W1Dropout, W2Dropout, path_encoder, residual_adapter
    
    def get_starting_embedding(self, start_emb_type: str, size: int) -> torch.Tensor:
        node_emb = self.knowledge_graph.get_starting_embedding(start_emb_type)

        init_emb = node_emb.unsqueeze(0).repeat(size, 1)
        return init_emb
    
    def get_centroid_embedding(self, size: int, relevant_ent: List[int] = None) -> torch.Tensor:
        return self.get_starting_embedding('centroid', size)

    def get_random_embedding(self, size: int, relevant_ent: List[int] = None) -> torch.Tensor:
        return self.get_starting_embedding('random', size)
    
    def get_relevant_embedding(self, size: int, query_entity: List[int] = None) -> torch.Tensor:
        # relevant_ent = torch.tensor([random.choice(sublist) for sublist in relevant_ent], dtype=torch.int)
        query_entity = torch.tensor(query_entity, dtype=torch.int)
    
        # Create more complete representation of state
        init_emb = self.knowledge_graph.get_starting_embedding(self.nav_start_emb_type, query_entity)

        if init_emb.dim() == 1: init_emb = init_emb.unsqueeze(0)
        assert init_emb.shape[0] == size, "Error! Initial states info and relevant embeddings must have the same batch size."
        return init_emb

    # * This is Nura's code. Might not really bee kj
    def get_action_space(self, e, obs, kg):
        r_space, e_space = kg.action_space[0][0][e], kg.action_space[0][1][e]
        action_mask = kg.action_space[1][e]
        action_space = ((r_space, e_space), action_mask)
        return action_space

# Eduin's code, not sure if it works
class AttentionFusion(nn.Module):
    def __init__(self, text_dim, semantic_dim, fusion_dim):
        super(AttentionFusion, self).__init__()
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.semantic_projection = nn.Linear(semantic_dim, fusion_dim)
        self.query = nn.Linear(fusion_dim, fusion_dim)
        self.key = nn.Linear(fusion_dim, fusion_dim)
        self.value = nn.Linear(fusion_dim, fusion_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_embedding, semantic_embedding):
        # Project embeddings to the common dimensionality
        text_proj = self.text_projection(text_embedding)
        semantic_proj = self.semantic_projection(semantic_embedding)

        # Compute query, key, and value for attention
        query = self.query(text_proj)
        key = self.key(semantic_proj)
        value = self.value(semantic_proj)

        # Calculate attention scores
        attention_scores = torch.bmm(query.unsqueeze(1), key.unsqueeze(2)).squeeze(-1)
        attention_weights = self.softmax(attention_scores)

        # Weighted sum of values
        fused_embedding = attention_weights * value
        return fused_embedding

class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x) + self.residual(x)


