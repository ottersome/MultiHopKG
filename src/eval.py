"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Compute Evaluation Metrics.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/evaluation.py
"""

import numpy as np
import pickle
from numbers import Integral
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

from src.parse_args import args
from src.data_utils import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID
from src.data_utils import DUMMY_RELATION_ID, START_RELATION_ID, NO_OP_RELATION_ID


def _get_answer_mask(all_answers, e1, query):
    """Return list of answer ids to mask for a given (e1, query)."""
    if all_answers is None:
        return []

    # Normalize query to int relation id when possible.
    if isinstance(query, torch.Tensor):
        if query.dim() == 0:
            query_id = int(query.item())
        else:
            return []
    elif isinstance(query, Integral):
        query_id = int(query)
    else:
        return []

    answers_for_e1 = all_answers.get(e1)
    if answers_for_e1 is None:
        return []
    candidates = answers_for_e1.get(query_id)
    if not candidates:
        return []
    return list(candidates)


def _core_example(example):
    return example[:3]


def get_gold_path(example):
    if len(example) < 4 or example[3] is None or isinstance(example[3], str):
        return None
    raw_path = example[3].tolist() if hasattr(example[3], 'tolist') else example[3]
    if not isinstance(raw_path, (list, tuple)):
        return None
    return [tuple(int(x) for x in edge) for edge in raw_path]


def get_gold_answers(example):
    _, answers, _ = _core_example(example)
    if hasattr(answers, 'tolist'):
        answers = answers.tolist()
    if isinstance(answers, (list, tuple, set)):
        return [int(answer) for answer in answers]
    return [int(answers)]


def get_example_hops(example):
    if len(example) >= 5:
        try:
            return int(example[4])
        except Exception:
            return None
    gold_path = get_gold_path(example)
    return len(gold_path) if gold_path is not None else None


def hits_and_ranks(examples, scores, all_answers, verbose=False):
    """
    Compute ranking based metrics.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, query = _core_example(example)
        gold_answers = get_gold_answers(example)
        answer_mask = _get_answer_mask(all_answers, e1, query)
        e2_multi = list(dict.fromkeys(dummy_mask + answer_mask + gold_answers))
        # save the relevant prediction
        target_scores = scores[i, gold_answers].clone()
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, gold_answers] = target_scores
    
    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    for i, example in enumerate(examples):
        gold_answers = set(get_gold_answers(example))
        pos = np.where(np.isin(top_k_targets[i], list(gold_answers)))[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += 1.0 / (pos + 1)

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)
    mrr = float(mrr) / len(examples)

    if verbose:
        print('Hits@1 = {:.3f}'.format(hits_at_1))
        print('Hits@3 = {:.3f}'.format(hits_at_3))
        print('Hits@5 = {:.3f}'.format(hits_at_5))
        print('Hits@10 = {:.3f}'.format(hits_at_10))
        print('MRR = {:.3f}'.format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr


def hits_and_ranks_counts(examples, scores, all_answers):
    """
    Compute unnormalized ranking counts for a score batch.

    This mirrors hits_and_ranks but returns counts so callers can stream large
    evaluations without concatenating every batch's full entity-score matrix.
    """
    assert (len(examples) == scores.shape[0])
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, query = _core_example(example)
        gold_answers = get_gold_answers(example)
        answer_mask = _get_answer_mask(all_answers, e1, query)
        e2_multi = list(dict.fromkeys(dummy_mask + answer_mask + gold_answers))
        target_scores = scores[i, gold_answers].clone()
        scores[i, e2_multi] = 0
        scores[i, gold_answers] = target_scores

    _, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    for i, example in enumerate(examples):
        gold_answers = set(get_gold_answers(example))
        pos = np.where(np.isin(top_k_targets[i], list(gold_answers)))[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += 1.0 / (pos + 1)

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr, len(examples)


def format_hits_and_ranks_counts(counts, verbose=False):
    hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr, total = counts
    hits_at_1 = float(hits_at_1) / total
    hits_at_3 = float(hits_at_3) / total
    hits_at_5 = float(hits_at_5) / total
    hits_at_10 = float(hits_at_10) / total
    mrr = float(mrr) / total

    if verbose:
        print('Hits@1 = {:.3f}'.format(hits_at_1))
        print('Hits@3 = {:.3f}'.format(hits_at_3))
        print('Hits@5 = {:.3f}'.format(hits_at_5))
        print('Hits@10 = {:.3f}'.format(hits_at_10))
        print('MRR = {:.3f}'.format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr

def hits_at_k(examples, scores, all_answers, verbose=False):
    """
    Hits at k metrics.
    :param examples: List of triples and labels (+/-).
    :param pred_targets:
    :param scores:
    :param all_answers:
    :param verbose:
    """
    assert(len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, query = _core_example(example)
        gold_answers = get_gold_answers(example)
        answer_mask = _get_answer_mask(all_answers, e1, query)
        e2_multi = list(dict.fromkeys(answer_mask + dummy_mask + gold_answers))
        # save the relevant prediction
        target_scores = scores[i, gold_answers].clone()
        # mask all false negatives
        scores[i][e2_multi] = 0
        scores[i][dummy_mask] = 0
        # write back the save prediction
        scores[i][gold_answers] = target_scores
        
    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    for i, example in enumerate(examples):
        gold_answers = set(get_gold_answers(example))
        pos = np.where(np.isin(top_k_targets[i], list(gold_answers)))[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)

    if verbose:
        print('Hits@1 = {:.3f}'.format(hits_at_1))
        print('Hits@3 = {:.3f}'.format(hits_at_3))
        print('Hits@5 = {:.3f}'.format(hits_at_5))
        print('Hits@10 = {:.3f}'.format(hits_at_10))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10

def hits_and_ranks_by_seen_queries(examples, scores, all_answers, seen_queries, verbose=False):
    seen_exps, unseen_exps = [], []
    seen_ids, unseen_ids = [], []
    for i, example in enumerate(examples):
        e1, e2, r = _core_example(example)
        if (e1, r) in seen_queries:
            seen_exps.append(example)
            seen_ids.append(i)
        else:
            unseen_exps.append(example)
            unseen_ids.append(i)

    _, _, _, _, seen_mrr = hits_and_ranks(seen_exps, scores[seen_ids], all_answers, verbose=False)
    _, _, _, _, unseen_mrr = hits_and_ranks(unseen_exps, scores[unseen_ids], all_answers, verbose=False)
    if verbose:
        print('MRR on seen queries: {:.3f}'.format(seen_mrr))
        print('MRR on unseen queries: {:.3f}'.format(unseen_mrr))
    return seen_mrr, unseen_mrr

def hits_and_ranks_by_relation_type(examples, scores, all_answers, relation_by_types, verbose=False):
    to_M_rels, to_1_rels = relation_by_types
    to_M_exps, to_1_exps = [], []
    to_M_ids, to_1_ids = [], []
    for i, example in enumerate(examples):
        e1, e2, r = _core_example(example)
        if r in to_M_rels:
            to_M_exps.append(example)
            to_M_ids.append(i)
        else:
            to_1_exps.append(example)
            to_1_ids.append(i)

    _, _, _, _, to_m_mrr = hits_and_ranks(to_M_exps, scores[to_M_ids], all_answers, verbose=False)
    _, _, _, _, to_1_mrr = hits_and_ranks(to_1_exps, scores[to_1_ids], all_answers, verbose=False)
    if verbose:
        print('MRR on to-M relations: {:.3f}'.format(to_m_mrr))
        print('MRR on to-1 relations: {:.3f}'.format(to_1_mrr))
    return to_m_mrr, to_1_mrr

def link_MAP(examples, scores, labels, all_answers, verbose=False):
    """
    Per-query mean average precision.
    """
    assert (len(examples) == len(scores))
    queries = {}
    for i, example in enumerate(examples):
        e1, e2, r = _core_example(example)
        gold_answers = get_gold_answers(example)
        if not e1 in queries:
            queries[e1] = []
        queries[e1].append((examples[i], labels[i], scores[i][gold_answers].max()))

    aps = []
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]

    for e1 in queries:
        ranked_examples = sorted(queries[e1], key=lambda x:x[2], reverse=True)
        acc_precision, offset, num_pos = 0, 0, 0
        for i in range(len(ranked_examples)):
            triple, label, score = ranked_examples[i]
            e1, e2, r = _core_example(triple)
            if label == '+':
                num_pos += 1
                acc_precision += float(num_pos) / (i + 1 - offset)
            else:
                answer_set = {}
                if e1 in all_answers and r in all_answers[e1]:
                    answer_set = all_answers[e1][r]
                if e2 in answer_set or e2 in dummy_mask:
                    print('False negative found: {}'.format(triple))
                    offset += 1 
        if num_pos > 0:
            ap = acc_precision / num_pos
            aps.append(ap)
    map = np.mean(aps)
    if verbose:
        print('MAP = {:.3f}'.format(map))
    return map

def export_error_cases(examples, scores, all_answers, output_path):
    """
    Export indices of examples to which the top-1 prediction is incorrect.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = _core_example(example)
        gold_answers = get_gold_answers(example)
        e2_multi = list(dict.fromkeys(dummy_mask + list(all_answers[e1][r]) + gold_answers))
        # save the relevant prediction
        target_scores = scores[i, gold_answers].clone()
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, gold_answers] = target_scores

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    top_1_errors, top_10_errors = [], []
    for i, example in enumerate(examples):
        gold_answers = set(get_gold_answers(example))
        pos = np.where(np.isin(top_k_targets[i], list(gold_answers)))[0]
        if len(pos) <= 0 or pos[0] > 0:
            top_1_errors.append(i)
        if len(pos) <= 0 or pos[0] > 9:
            top_10_errors.append(i)
    with open(output_path, 'wb') as o_f:
        pickle.dump([top_1_errors, top_10_errors], o_f)        
                 
    print('{}/{} top-1 error cases written to {}'.format(len(top_1_errors), len(examples), output_path))
    print('{}/{} top-10 error cases written to {}'.format(len(top_10_errors), len(examples), output_path))


def compute_precision_recall_f1(pred: Set, gt: Set, eps: float = 1e-8) -> Tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def edit_distance(seq1: Sequence, seq2: Sequence) -> Tuple[int, int, int]:
    m = len(seq1)
    n = len(seq2)
    if m == 0 and n == 0:
        return 0, m, n
    if m == 0 or n == 0:
        return max(m, n), m, n
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    return dp[m][n], m, n


def build_inverse_relation_mapping(kg) -> Dict[int, int]:
    relation2id = getattr(kg, 'relation2id', {})
    id2relation = getattr(kg, 'id2relation', {})
    inverse_mapping = {}
    for r_id, rel_name in id2relation.items():
        if isinstance(rel_name, str) and rel_name.endswith('_inv'):
            base_name = rel_name[:-4]
            if base_name in relation2id:
                inverse_mapping[int(r_id)] = int(relation2id[base_name])
    return inverse_mapping


def canon_edge(h: int, r: int, t: int, inverse_mapping: Dict[int, int]) -> Tuple[int, int, int]:
    if r in inverse_mapping:
        return int(t), int(inverse_mapping[r]), int(h)
    return int(h), int(r), int(t)


def canon_rel(r: int, inverse_mapping: Dict[int, int]) -> int:
    return int(inverse_mapping.get(int(r), int(r)))


def clean_path_edges(path: Sequence[Tuple[int, int, int]],
                     special_tokens: Set[int],
                     inverse_mapping: Dict[int, int]) -> List[Tuple[int, int, int]]:
    return [
        canon_edge(h, r, t, inverse_mapping)
        for h, r, t in path
        if int(r) not in special_tokens
    ]


def gt_edge_overlap_f1(pred_path: Sequence[Tuple[int, int, int]],
                       gt_path: Sequence[Tuple[int, int, int]],
                       special_tokens: Set[int],
                       inverse_mapping: Dict[int, int]) -> Tuple[float, float, float]:
    pred_edges = set(clean_path_edges(pred_path, special_tokens, inverse_mapping))
    gt_edges = {tuple(int(x) for x in edge) for edge in gt_path}
    return compute_precision_recall_f1(pred_edges, gt_edges)


def relation_edit_distance_norm(pred_relations: Sequence[int],
                                gt_relations: Sequence[int],
                                special_tokens: Set[int],
                                inverse_mapping: Dict[int, int],
                                eps: float = 1e-8) -> float:
    pred_rels = [canon_rel(r, inverse_mapping) for r in pred_relations if int(r) not in special_tokens]
    gt_rels = [int(r) for r in gt_relations]
    dist, m, n = edit_distance(pred_rels, gt_rels)
    return dist / (max(m, n) + eps)


def path_edit_distance_norm(pred_path: Sequence[Tuple[int, int, int]],
                            gt_path: Sequence[Tuple[int, int, int]],
                            special_tokens: Set[int],
                            inverse_mapping: Dict[int, int],
                            eps: float = 1e-8) -> float:
    pred_edges = clean_path_edges(pred_path, special_tokens, inverse_mapping)
    gt_edges = [tuple(int(x) for x in edge) for edge in gt_path]
    dist, m, n = edit_distance(pred_edges, gt_edges)
    return dist / (max(m, n) + eps)


def answer_set_f1(predicted_endpoints: Iterable[int],
                  gold_answers: Iterable[int],
                  eps: float = 1e-8) -> Tuple[float, float, float]:
    pred_set = {int(x) for x in predicted_endpoints}
    gold_set = {int(x) for x in gold_answers}
    tp = len(pred_set & gold_set)
    precision = tp / (len(pred_set) + eps)
    recall = tp / (len(gold_set) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


class FaithfulnessEvaluator:
    """Aggregates path-faithfulness metrics for top rollout/beam paths."""

    def __init__(self, kg) -> None:
        self.inverse_mapping = build_inverse_relation_mapping(kg)
        self.special_tokens = {DUMMY_RELATION_ID, START_RELATION_ID, NO_OP_RELATION_ID}
        self.edge_precision = 0.0
        self.edge_recall = 0.0
        self.edge_f1 = 0.0
        self.rel_edit = 0.0
        self.path_edit = 0.0
        self.answer_precision = 0.0
        self.answer_recall = 0.0
        self.answer_f1 = 0.0
        self.num_examples = 0
        self.by_hop: Dict[int, Dict[str, float]] = {}

    def update(self,
               example,
               pred_path: Sequence[Tuple[int, int, int]],
               predicted_endpoints: Iterable[int]) -> None:
        gt_path = get_gold_path(example)
        if not gt_path:
            return
        _, gold_answer, _ = _core_example(example)
        gold_answers = get_gold_answers(example)
        hop = get_example_hops(example) or len(gt_path)
        gt_relations = [int(edge[1]) for edge in gt_path]
        pred_relations = [int(edge[1]) for edge in pred_path]

        edge_p, edge_r, edge_f = gt_edge_overlap_f1(
            pred_path, gt_path, self.special_tokens, self.inverse_mapping)
        rel_dist = relation_edit_distance_norm(
            pred_relations, gt_relations, self.special_tokens, self.inverse_mapping)
        path_dist = path_edit_distance_norm(
            pred_path, gt_path, self.special_tokens, self.inverse_mapping)
        ans_p, ans_r, ans_f = answer_set_f1(predicted_endpoints, gold_answers)

        self.edge_precision += edge_p
        self.edge_recall += edge_r
        self.edge_f1 += edge_f
        self.rel_edit += rel_dist
        self.path_edit += path_dist
        self.answer_precision += ans_p
        self.answer_recall += ans_r
        self.answer_f1 += ans_f
        self.num_examples += 1

        entry = self.by_hop.setdefault(int(hop), {
            'examples': 0,
            'edge_f1': 0.0,
            'relation_edit_distance': 0.0,
            'path_edit_distance': 0.0,
            'answer_set_f1': 0.0,
        })
        entry['examples'] += 1
        entry['edge_f1'] += edge_f
        entry['relation_edit_distance'] += rel_dist
        entry['path_edit_distance'] += path_dist
        entry['answer_set_f1'] += ans_f

    def compute(self) -> Dict[str, float]:
        if self.num_examples == 0:
            return {}
        n = float(self.num_examples)
        metrics = {
            'faithfulness/examples': self.num_examples,
            'faithfulness/edge_precision': self.edge_precision / n,
            'faithfulness/edge_recall': self.edge_recall / n,
            'faithfulness/edge_f1': self.edge_f1 / n,
            'faithfulness/relation_edit_distance': self.rel_edit / n,
            'faithfulness/path_edit_distance': self.path_edit / n,
            'faithfulness/answer_set_precision': self.answer_precision / n,
            'faithfulness/answer_set_recall': self.answer_recall / n,
            'faithfulness/answer_set_f1': self.answer_f1 / n,
        }
        for hop, values in sorted(self.by_hop.items()):
            count = float(values['examples'])
            metrics[f'faithfulness/{hop}hop_examples'] = values['examples']
            for key in ['edge_f1', 'relation_edit_distance', 'path_edit_distance', 'answer_set_f1']:
                metrics[f'faithfulness/{hop}hop_{key}'] = values[key] / count
        return metrics


def _stable_logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp for a small sequence of scores."""
    if not values:
        return float('-inf')
    arr = np.asarray(values, dtype=np.float64)
    max_val = np.max(arr)
    return max_val + np.log(np.sum(np.exp(arr - max_val)))


def _normalize_hits(hits_ks: Iterable[int]) -> Tuple[int, ...]:
    normalized = tuple(sorted({int(k) for k in hits_ks if k > 0}))
    if not normalized:
        raise ValueError('hits_ks must contain at least one positive integer')
    return normalized


def _find_answer_position_max(sorted_indices: np.ndarray,
                              rewards_row: np.ndarray,
                              entities_row: np.ndarray,
                              positive_reward: float) -> int:
    seen = set()
    position = 0
    for rollout_idx in sorted_indices:
        if np.isclose(rewards_row[rollout_idx], positive_reward):
            return position
        entity_id = int(entities_row[rollout_idx])
        if entity_id not in seen:
            seen.add(entity_id)
            position += 1
    return -1


def _find_answer_position_sum(sorted_indices: np.ndarray,
                              log_probs_row: np.ndarray,
                              rewards_row: np.ndarray,
                              entities_row: np.ndarray,
                              positive_reward: float) -> int:
    scores_by_entity: Dict[int, list] = {}
    correct_entity = None
    for rollout_idx in sorted_indices:
        entity_id = int(entities_row[rollout_idx])
        scores_by_entity.setdefault(entity_id, []).append(log_probs_row[rollout_idx])
        if correct_entity is None and np.isclose(rewards_row[rollout_idx], positive_reward):
            correct_entity = entity_id
    if correct_entity is None:
        return -1
    aggregated = {entity: _stable_logsumexp(scores) for entity, scores in scores_by_entity.items()}
    ranked_entities = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    for position, (entity_id, _) in enumerate(ranked_entities):
        if entity_id == correct_entity:
            return position
    return -1


def _rollout_totals(log_probs_arr: np.ndarray,
                    rewards_arr: np.ndarray,
                    entities_arr: np.ndarray,
                    positive_reward: float,
                    pool: str,
                    hits_ks: Tuple[int, ...]) -> Tuple[Dict[int, int], float, int]:
    hits_counts = {k: 0 for k in hits_ks}
    mrr_total = 0.0
    batch_size = log_probs_arr.shape[0]
    sorted_indices = np.argsort(-log_probs_arr, axis=1)

    for row in range(batch_size):
        if pool == 'max':
            answer_pos = _find_answer_position_max(sorted_indices[row], rewards_arr[row], entities_arr[row], positive_reward)
        else:
            answer_pos = _find_answer_position_sum(sorted_indices[row], log_probs_arr[row], rewards_arr[row], entities_arr[row], positive_reward)

        if answer_pos < 0:
            continue

        for k in hits_ks:
            if answer_pos < k:
                hits_counts[k] += 1
        mrr_total += 1.0 / (answer_pos + 1)

    return hits_counts, mrr_total, batch_size


def evaluate_rollout_batch(log_probs: Sequence[Sequence[float]],
                           rewards: Sequence[Sequence[float]],
                           final_entities: Sequence[Sequence[int]],
                           positive_reward: float = 1.0,
                           pool: str = 'max',
                           hits_ks: Iterable[int] = (1, 3, 5, 10, 20)) -> Dict[str, float]:
    """
    Compute Hits@K and MRR metrics for rollout-based reasoning batches.

    This mirrors the MINERVA evaluation logic by ranking rollouts either by the
    highest scoring unique entity (``pool='max'``) or by aggregating path scores per
    entity with log-sum-exp (``pool='sum'``).
    """
    log_probs_arr = np.asarray(log_probs, dtype=np.float64)
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    entities_arr = np.asarray(final_entities)

    if log_probs_arr.ndim != 2:
        raise ValueError('log_probs must be a 2-D array-like structure')
    if rewards_arr.shape != log_probs_arr.shape:
        raise ValueError('rewards must have the same shape as log_probs')
    if entities_arr.shape != log_probs_arr.shape:
        raise ValueError('final_entities must have the same shape as log_probs')
    if log_probs_arr.shape[0] == 0:
        raise ValueError('log_probs must contain at least one example')

    pool = pool.lower()
    if pool not in {'max', 'sum'}:
        raise ValueError("pool must be either 'max' or 'sum'")

    hits_ks = _normalize_hits(hits_ks)
    hits_counts, mrr_total, batch_size = _rollout_totals(
        log_probs_arr,
        rewards_arr,
        entities_arr,
        positive_reward,
        pool,
        hits_ks
    )

    metrics = {f'hits@{k}': hits_counts[k] / batch_size for k in hits_ks}
    metrics['mrr'] = mrr_total / batch_size if batch_size else 0.0
    return metrics


class RolloutEvaluator:
    """Incrementally aggregates rollout-based evaluation metrics."""

    def __init__(self,
                 positive_reward: float = 1.0,
                 pool: str = 'max',
                 hits_ks: Iterable[int] = (1, 3, 5, 10, 20)) -> None:
        self.positive_reward = positive_reward
        self.pool = pool.lower()
        if self.pool not in {'max', 'sum'}:
            raise ValueError("pool must be either 'max' or 'sum'")
        self.hits_ks = _normalize_hits(hits_ks)
        self._hits_counts = {k: 0 for k in self.hits_ks}
        self._mrr_total = 0.0
        self._num_examples = 0

    def update(self,
               log_probs: Sequence[Sequence[float]],
               rewards: Sequence[Sequence[float]],
               final_entities: Sequence[Sequence[int]]) -> None:
        log_probs_arr = np.asarray(log_probs, dtype=np.float64)
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        entities_arr = np.asarray(final_entities)

        if log_probs_arr.ndim != 2:
            raise ValueError('log_probs must be a 2-D array-like structure')
        if rewards_arr.shape != log_probs_arr.shape:
            raise ValueError('rewards must have the same shape as log_probs')
        if entities_arr.shape != log_probs_arr.shape:
            raise ValueError('final_entities must have the same shape as log_probs')
        if log_probs_arr.shape[0] == 0:
            raise ValueError('log_probs must contain at least one example')

        hits_counts, mrr_total, batch_size = _rollout_totals(
            log_probs_arr,
            rewards_arr,
            entities_arr,
            self.positive_reward,
            self.pool,
            self.hits_ks
        )

        for k in self.hits_ks:
            self._hits_counts[k] += hits_counts[k]
        self._mrr_total += mrr_total
        self._num_examples += batch_size

    def merge(self, other: 'RolloutEvaluator') -> None:
        if not isinstance(other, RolloutEvaluator):
            raise TypeError('Can only merge with another RolloutEvaluator')
        if (self.positive_reward != other.positive_reward or
                self.pool != other.pool or
                self.hits_ks != other.hits_ks):
            raise ValueError('Evaluators must share the same configuration to merge')
        for k in self.hits_ks:
            self._hits_counts[k] += other._hits_counts[k]
        self._mrr_total += other._mrr_total
        self._num_examples += other._num_examples

    def compute(self) -> Dict[str, float]:
        if self._num_examples == 0:
            raise ValueError('No examples have been added to the evaluator')
        metrics = {f'hits@{k}': self._hits_counts[k] / self._num_examples for k in self.hits_ks}
        metrics['mrr'] = self._mrr_total / self._num_examples
        metrics['examples'] = self._num_examples
        return metrics

    @property
    def num_examples(self) -> int:
        return self._num_examples
