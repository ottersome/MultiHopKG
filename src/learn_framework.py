"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Base learning framework.
"""

import os
import random
import shutil
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops
from typing import Dict, List, Optional

from transformers import AutoTokenizer, AutoModel  # type: ignore


class LFramework(nn.Module):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None

        self.inference = not args.train
        self.run_analysis = args.run_analysis

        self.kg = kg
        self.mdl = mdl
        print('{} module created'.format(self.model))

        # Optional question encoder (BERT) to replace relation ids with NL vectors
        self.use_question_encoder = bool(getattr(args, 'use_question_encoder', False))
        self._q_tokenizer = None
        self._q_encoder = None
        self._q_hidden = None
        # Project question repr to relation_dim used throughout the policy
        self._q_proj: Optional[nn.Linear] = None
        if self.use_question_encoder:
            if AutoTokenizer is not None and AutoModel is not None:
                self._q_tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
                self._q_encoder = AutoModel.from_pretrained(args.bert_model_name)
                self._q_encoder.eval()
                for p in self._q_encoder.parameters():
                    p.requires_grad = False
                # Hidden size from config; default to 768 if unavailable
                self._q_hidden = getattr(getattr(self._q_encoder, 'config', object()), 'hidden_size', None)
                assert self._q_hidden != None
                self._q_hidden = int(self._q_hidden)
            else:
                # transformers not available
                self._q_hidden = int(getattr(args, 'relation_dim', 200))
            # Always set a projection to match relation_dim expected downstream
            self._q_proj = nn.Linear(self._q_hidden, args.relation_dim)

    def supports_rollout_evaluation(self) -> bool:
        return False

    def evaluate_with_rollouts(self, data, split_name: str = 'dev') -> Optional[Dict[str, float]]:
        _ = split_name  # Unused in base implementation
        return None

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def run_train(self, train_data, dev_data):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []

        # Setup wandb model watching if enabled
        _wandb_enabled = getattr(self.args, 'wandb_enabled', False)
        _wandb = None
        if _wandb_enabled:
            try:
                import wandb as _wandb  # type: ignore
                # Only watch once per run
                if not hasattr(self, '_wandb_watching'):
                    _wandb.watch(self, log='gradients', log_freq=500)
                    setattr(self, '_wandb_watching', True)
            except Exception:
                _wandb_enabled = False

        # Global step counter for inter-batch logging
        if not hasattr(self, '_global_step'):
            self._global_step = 0
        log_interval = int(getattr(self.args, 'wandb_log_interval', 0) or 0)

        for epoch_id in range(self.start_epoch, self.num_epochs):
            print('Epoch {}'.format(epoch_id))
            if self.rl_variation_tag.startswith('rs'):
                # Reward shaping module sanity check:
                #   Make sure the reward shaping module output value is in the correct range
                train_scores = self.test_fn(train_data)
                dev_scores = self.test_fn(dev_data)
                print('Train set average fact score: {}'.format(float(train_scores.mean())))
                print('Dev set average fact score: {}'.format(float(dev_scores.mean())))

            # Update model parameters
            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            entropies = []
            if self.run_analysis:
                rewards = None
                fns = None
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):

                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue
                # 🎯 Pay close attention here. 
                loss = self.loss(mini_batch)
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()

                batch_losses.append(loss['print_loss'])
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                if self.run_analysis:
                    if rewards is None:
                        rewards = loss['reward']
                    else:
                        rewards = torch.cat([rewards, loss['reward']])
                    if fns is None:
                        fns = loss['fn']
                    else:
                        fns = torch.cat([fns, loss['fn']])

                # Inter-batch wandb logging
                self._global_step += 1
                if _wandb_enabled and log_interval > 0 and (self._global_step % log_interval == 0):
                    log_dict = {
                        'step': int(self._global_step),
                        'train_step/loss': float(loss['print_loss']) if isinstance(loss['print_loss'], (int, float)) else float(loss['print_loss']),
                        'train_step/epoch': int(epoch_id),
                    }
                    if 'entropy' in loss:
                        try:
                            log_dict['train_step/entropy'] = float(loss['entropy'])
                        except Exception:
                            pass
                    if self.optim and self.optim.param_groups:
                        log_dict['lr'] = float(self.optim.param_groups[0]['lr'])
                    _wandb.log(log_dict)
            # Check training statistics
            avg_train_loss = np.mean(batch_losses).item() if batch_losses else 0.0
            stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, avg_train_loss)
            if entropies:
                avg_entropy = float(np.mean(entropies))
                stdout_msg += ' entropy = {}'.format(avg_entropy)
            else:
                avg_entropy = None
            print(stdout_msg)
            self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            # wandb: log training metrics
            if _wandb_enabled:
                log_dict = {
                    'epoch': epoch_id,
                    'train/loss': avg_train_loss,
                    'lr': float(self.optim.param_groups[0]['lr']) if self.optim and self.optim.param_groups else None,
                }
                if avg_entropy is not None:
                    log_dict['train/entropy'] = avg_entropy
                if hasattr(self, 'action_dropout_rate'):
                    log_dict['train/action_dropout'] = float(self.action_dropout_rate)
                _wandb.log(log_dict)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                hit_ratio = num_hits / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                num_fns = float(fns.sum())
                fn_ratio = num_fns / len(fns)
                print('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            if self.run_analysis or epoch_id % self.num_peek_epochs == 0:
                self.eval()
                self.batch_size = self.dev_batch_size
                with torch.no_grad():
                    dev_scores = self.forward(dev_data, verbose=False)
                print('Dev set performance: (correct evaluation)')
                h1, h3, h5, h10, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
                metrics = mrr
                print('Dev set performance: (include test set labels)')
                src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)

                rollout_metrics = None
                if self.supports_rollout_evaluation():
                    rollout_metrics = self.evaluate_with_rollouts(dev_data, split_name='dev')
                    if rollout_metrics:
                        hits_keys = sorted(
                            [k for k in rollout_metrics.keys() if k.startswith('hits@')],
                            key=lambda item: int(item.split('@')[1]) if item.count('@') == 1 else item
                        )
                        metrics_summary = ' '.join(
                            f"{k}={rollout_metrics[k]:.4f}" for k in hits_keys
                        )
                        num_rollouts_used = rollout_metrics.get('num_rollouts', getattr(self, 'num_rollouts', None))
                        pool_mode = rollout_metrics.get('pool', getattr(self.args, 'rollout_eval_pool', 'max'))
                        print(
                            f"Dev rollout evaluation (num_rollouts={num_rollouts_used}, pool={pool_mode}): "
                            f"{metrics_summary} mrr={rollout_metrics['mrr']:.4f}"
                        )
                # wandb: log dev metrics
                if _wandb_enabled:
                    log_dict = {
                        'epoch': epoch_id,
                        'dev/mrr': float(mrr),
                        'dev/hits@1': float(h1),
                        'dev/hits@3': float(h3),
                        'dev/hits@5': float(h5),
                        'dev/hits@10': float(h10),
                    }
                    if rollout_metrics:
                        for k, v in rollout_metrics.items():
                            if k.startswith('hits@') or k == 'mrr':
                                log_dict[f'dev_rollout/{k}'] = float(v)
                        if 'examples' in rollout_metrics:
                            log_dict['dev_rollout/examples'] = float(rollout_metrics['examples'])
                        if 'num_rollouts' in rollout_metrics:
                            log_dict['dev_rollout/num_rollouts'] = float(rollout_metrics['num_rollouts'])
                        log_dict['dev_rollout/pool'] = rollout_metrics.get('pool', getattr(self.args, 'rollout_eval_pool', 'max'))
                    _wandb.log(log_dict)
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor 
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)
                if self.run_analysis:
                    num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                    dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                    hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                    fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                    if epoch_id == 0:
                        with open(num_path_types_file, 'w') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'w') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                    else:
                        with open(num_path_types_file, 'a') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'a') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))

    def forward(self, examples, verbose=False):
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            pred_score = self.predict(mini_batch, verbose=verbose)
            pred_scores.append(pred_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        return scores

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        # Will fill an array with (batch_size, num_labels) with 1's where the label is 1
        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label
        batch_e1, batch_e2 = [], []
        # q_inputs collects either relation ids or token id lists depending on mode
        q_inputs: List = []
        for i in range(len(batch_data)):
            e1, e2, q = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            q_inputs.append(q)
        # TODO: Why is e2 not receiving the same treatment
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)

        # Prepare query/question tensor
        if self.use_question_encoder and any(isinstance(x, list) for x in q_inputs):
            # q_inputs is a list of token id lists (already tokenized without specials)
            # Build input_ids with [CLS] and [SEP] if tokenizer is available
            #TODO: Why do we use max_question_len here ?... I dont think we do 
            # max_len_cfg = int(getattr(self.args, 'max_question_len', 64))

            assert self._q_tokenizer is not None, "q_tokenizer expected in format_batch for nlp mode "

            # TODO: Is this a correct use of tokens with this specific tokenizer?
            cls_id = int(self._q_tokenizer.cls_token_id)
            sep_id = int(self._q_tokenizer.sep_token_id)

            proc_ids: List[List[int]] = []
            for toks in q_inputs:
                assert isinstance(toks,list), "Your input is empty, remove it from the dataset"
                assert len(toks) < self.args.max_question_len, "Your question is pretty long"
                seq = [cls_id] + toks + [sep_id]
                proc_ids.append(seq)

            # Pad to batch max length
            max_len = max(len(x) for x in proc_ids) 
            # TODO: Ensure this padding value is the correct one. This might be the trigger.
            input_ids = torch.full((len(proc_ids), max_len), fill_value=self._q_tokenizer.pad_token_id, dtype=torch.long) 
            attention_mask = torch.zeros((len(proc_ids), max_len), dtype=torch.long)
            for i, seq in enumerate(proc_ids):
                L = len(seq)
                input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :L] = 1

            input_ids = var_cuda(input_ids, requires_grad=False)
            attention_mask = var_cuda(attention_mask, requires_grad=False)

            with torch.no_grad():
                # Ensure encoder on the same device
                self._q_encoder.to(input_ids.device)
                out = self._q_encoder(input_ids=input_ids, attention_mask=attention_mask)
                # Prefer pooler_output; else take [CLS] token representation
                pooled = out.pooler_output# if hasattr(out, 'pooler_output') and out.pooler_output is not None \
                    #else out.last_hidden_state[:, 0, :]

            # Project to relation_dim expected by policy network
            assert self._q_proj is not None, "You also need q_proj for format_batch"
            batch_r = self._q_proj(pooled)
        else:
            # Legacy path: q_inputs is a list of relation ids
            batch_r = var_cuda(torch.LongTensor(q_inputs), requires_grad=False)

        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        # Try to mirror the "q" type of existing samples to avoid mixing modes
        use_token_list = False
        if len(mini_batch) > 0:
            try:
                use_token_list = isinstance(mini_batch[0][2], list)
            except Exception:
                use_token_list = False
        if multi_answers:
            dummy_q = [] if (self.use_question_encoder and use_token_list) else dummy_r
            dummy_example = (dummy_e, [dummy_e], dummy_q)
        else:
            dummy_q = [] if (self.use_question_encoder and use_token_list) else dummy_r
            dummy_example = (dummy_e, dummy_e, dummy_q)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(self.model_dir, 'model_best.tar')
            shutil.copyfile(out_tar, best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            print('=> saving checkpoint to \'{}\''.format(out_tar))

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            print(r, '{}'.format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
