"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient (REINFORCE algorithm) training and inference.
"""

from typing import Dict, Optional

import numpy as np
import torch

from src.eval import FaithfulnessEvaluator, RolloutEvaluator, get_example_hops, get_example_weight
from src.learn_framework import LFramework
import src.rl.graph_search.beam_search as search
import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        self.relation_only = args.relation_only
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval

        # Inference hyperparameters
        self.beam_size = args.beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

    def reward_fun(self, e1, r, e2, pred_e2):
        return self.binary_reward(e2, pred_e2)

    def binary_reward(self, e2, pred_e2):
        gold_answers = getattr(self, '_batch_gold_answers', None)
        gold_answer_mask = getattr(self, '_batch_gold_answer_mask', None)
        if gold_answers is None or gold_answer_mask is None or gold_answers.size(0) != pred_e2.size(0):
            return (pred_e2 == e2).float()
        hits = (pred_e2.unsqueeze(1) == gold_answers) & gold_answer_mask
        return hits.any(dim=1).float()

    def loss(self, mini_batch):
        
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r
    
        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg) # Uses dummy relation and the source entity to encode a path

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']
        if verbose:
            # print inference paths
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    print('beam {}: score = {} \n<PATH> {}'.format(
                        j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores

    def supports_rollout_evaluation(self) -> bool:
        if getattr(self.args, 'disable_rollout_eval', False):
            return False
        return True

    def evaluate_with_rollouts(self, data, split_name: str = 'dev',
                               num_rollouts: Optional[int] = None) -> Optional[Dict[str, float]]:
        if getattr(self.args, 'disable_rollout_eval', False):
            return None
        if not data:
            return None

        pool_mode = getattr(self.args, 'rollout_eval_pool', 'max')
        eval_rollouts = num_rollouts if num_rollouts is not None else getattr(self.args, 'rollout_eval_num_rollouts', 0)
        if not eval_rollouts or eval_rollouts <= 0:
            eval_rollouts = getattr(self, 'beam_size', self.num_rollouts)
        eval_rollouts = max(1, int(eval_rollouts))
        eval_batch_size = getattr(self.args, 'rollout_eval_batch_size', 0)
        if not eval_batch_size or eval_batch_size <= 0:
            eval_batch_size = self.dev_batch_size if getattr(self, 'dev_batch_size', None) else self.batch_size
        eval_batch_size = max(1, int(eval_batch_size))

        evaluator = RolloutEvaluator(positive_reward=1.0, pool=pool_mode)
        hop_evaluators: Dict[int, RolloutEvaluator] = {}
        faithfulness_evaluator = FaithfulnessEvaluator(self.kg)

        disable_dropout = not getattr(self.args, 'keep_rollout_eval_dropout', False)
        prev_dropout = getattr(self, 'action_dropout_rate', 0.0)
        if disable_dropout:
            self.action_dropout_rate = 0.0

        training_state = self.training
        try:
            self.eval()
            total_examples = len(data)
            with torch.no_grad():
                for start in range(0, total_examples, eval_batch_size):
                    mini_batch = data[start:start + eval_batch_size]
                    if not mini_batch:
                        continue

                    e1, e2, r = self.format_batch(mini_batch)
                    beam_output = search.beam_search(
                        self.mdl,
                        e1,
                        r,
                        e2,
                        self.kg,
                        self.num_rollout_steps,
                        eval_rollouts,
                        return_search_traces=True
                    )

                    pred_entities = beam_output['pred_e2s']
                    pred_scores = beam_output['pred_e2_scores']

                    pred_entities_np = pred_entities.detach().cpu().numpy()
                    pred_scores_np = pred_scores.detach().cpu().numpy()

                    gold_answers = getattr(self, '_batch_gold_answers', None)
                    gold_answer_mask = getattr(self, '_batch_gold_answer_mask', None)
                    if gold_answers is not None and gold_answer_mask is not None:
                        gold_answers_np = gold_answers.detach().cpu().numpy()
                        gold_answer_mask_np = gold_answer_mask.detach().cpu().numpy().astype(bool)
                        rewards_np = np.zeros_like(pred_entities_np, dtype=np.float64)
                        for row in range(pred_entities_np.shape[0]):
                            row_answers = gold_answers_np[row][gold_answer_mask_np[row]]
                            rewards_np[row] = np.isin(pred_entities_np[row], row_answers).astype(np.float64)
                    else:
                        target_entities_np = e2.detach().cpu().numpy().reshape(-1, 1)
                        rewards_np = (pred_entities_np == target_entities_np).astype(np.float64)

                    dummy_entity = int(self.kg.dummy_e)
                    invalid_mask = (pred_entities_np == dummy_entity)
                    if np.any(invalid_mask):
                        pred_scores_np = np.where(invalid_mask, -1e10, pred_scores_np)
                        rewards_np[invalid_mask] = 0.0

                    example_weights = np.asarray(
                        [get_example_weight(example) for example in mini_batch],
                        dtype=np.float64
                    )
                    evaluator.update(pred_scores_np, rewards_np, pred_entities_np, weights=example_weights)
                    for row, example in enumerate(mini_batch):
                        hop = get_example_hops(example)
                        if hop is None:
                            continue
                        hop = int(hop)
                        hop_evaluator = hop_evaluators.setdefault(
                            hop,
                            RolloutEvaluator(positive_reward=1.0, pool=pool_mode)
                        )
                        hop_evaluator.update(
                            pred_scores_np[row:row + 1],
                            rewards_np[row:row + 1],
                            pred_entities_np[row:row + 1],
                            weights=example_weights[row:row + 1]
                        )
                    if 'search_traces' in beam_output:
                        search_traces = beam_output['search_traces']
                        output_beam_size = pred_entities.size(1)
                        for row, example in enumerate(mini_batch):
                            top_ind = row * output_beam_size
                            pred_path = []
                            for step in range(self.num_rollout_steps):
                                h = int(search_traces[step][1][top_ind])
                                rel = int(search_traces[step + 1][0][top_ind])
                                t = int(search_traces[step + 1][1][top_ind])
                                pred_path.append((h, rel, t))
                            faithfulness_evaluator.update(
                                example,
                                pred_path,
                                pred_entities_np[row].tolist(),
                                weight=float(example_weights[row])
                            )
        finally:
            if disable_dropout:
                self.action_dropout_rate = prev_dropout
            if training_state:
                self.train()

        if evaluator.num_examples == 0:
            return None

        metrics = evaluator.compute()
        for hop, hop_evaluator in sorted(hop_evaluators.items()):
            hop_metrics = hop_evaluator.compute()
            metrics[f'per_hop/{hop}hop_examples'] = hop_metrics['examples']
            metrics[f'per_hop/{hop}hop_hits@1'] = hop_metrics.get('hits@1', 0.0)
            metrics[f'per_hop/{hop}hop_hits@3'] = hop_metrics.get('hits@3', 0.0)
            metrics[f'per_hop/{hop}hop_hits@5'] = hop_metrics.get('hits@5', 0.0)
            metrics[f'per_hop/{hop}hop_hits@10'] = hop_metrics.get('hits@10', 0.0)
            metrics[f'per_hop/{hop}hop_hits@20'] = hop_metrics.get('hits@20', 0.0)
            metrics[f'per_hop/{hop}hop_mrr'] = hop_metrics['mrr']
        metrics.update(faithfulness_evaluator.compute())
        metrics['num_rollouts'] = eval_rollouts
        metrics['pool'] = pool_mode
        metrics['split'] = split_name
        return metrics

    def record_path_trace(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]
