#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""

from datetime import datetime
from argparse import Namespace
import copy
import itertools
import json
import numpy as np
import os, sys
import random
import debugpy
import platform

from typing import Dict, Iterable, List, Optional, Set, TypeVar, Union

import torch

from src.parse_args import parse_args
import src.data_utils as data_utils
from src.data_utils import HopFilter, parse_hop_filter
import src.eval
from src.hyperparameter_range import hp_range
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ComplEx, ConvE, DistMult, TransE
from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.utils.experiment_io import write_run_manifest
from src.emb.emb import EmbeddingBasedMethod
from src.itl_typing import QAExample
from src.learn_framework import LFramework
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from src.utils.ops import flatten

args: Optional[Namespace] = None

ExampleT = TypeVar('ExampleT')


def configure_runtime(runtime_args: Namespace) -> None:
    """Apply process-wide runtime settings after CLI arguments are explicitly parsed."""
    torch.cuda.set_device(runtime_args.gpu)
    torch.manual_seed(runtime_args.seed)
    torch.cuda.manual_seed_all(runtime_args.seed)
    random.seed(runtime_args.seed)
    np.random.seed(runtime_args.seed)

def setup_wandb(args, job_type='train'):
    """Initialize Weights & Biases run if enabled via args.wandb.
    Sets args.wandb_enabled to True/False based on initialization success.
    """
    # Default to disabled
    setattr(args, 'wandb_enabled', False)
    if not getattr(args, 'wandb', False):
        return
    # Allow disabling via mode param
    if getattr(args, 'wandb_mode', '') == 'disabled':
        return
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"wandb not available ({e}); disabling wandb logging.")
        return
    # Respect chosen mode
    if getattr(args, 'wandb_mode', None):
        os.environ.setdefault('WANDB_MODE', args.wandb_mode)
    run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.model_dir))
    tags = [t.strip() for t in getattr(args, 'wandb_tags', '').split(',') if t.strip()]
    init_kwargs = dict(
        project=getattr(args, 'wandb_project', 'salesforce-multihopkg'),
        entity=(args.wandb_entity or None),
        name=run_name,
        dir=args.model_dir,
        reinit=True,
        job_type=job_type,
        config=vars(args)
    )
    if getattr(args, 'wandb_group', ''):
        init_kwargs['group'] = args.wandb_group
    if getattr(args, 'wandb_notes', ''):
        init_kwargs['notes'] = args.wandb_notes
    if tags:
        init_kwargs['tags'] = tags
    try:
        wandb.init(**init_kwargs)
        if getattr(wandb.run, 'sweep_id', None) or os.environ.get('WANDB_SWEEP_ID'):
            setattr(args, 'disable_checkpoint_saving', True)
            print('W&B sweep detected; checkpoint saving disabled for this run.')
        # Define a common step metric for nice charts
        wandb.define_metric('epoch')
        wandb.define_metric('train/*', step_metric='epoch')
        wandb.define_metric('dev/*', step_metric='epoch')
        wandb.define_metric('dev_rollout/*', step_metric='epoch')
        wandb.define_metric('dev_faithfulness/*', step_metric='epoch')
        wandb.define_metric('dev_per_hop/*', step_metric='epoch')
        # Optional inter-batch step metric
        wandb.define_metric('step')
        wandb.define_metric('train_step/*', step_metric='step')
        setattr(args, 'wandb_enabled', True)
        print(f"wandb run initialized: {run_name}")
    except Exception as e:
        print(f"Failed to initialize wandb ({e}); continuing without logging.")
        setattr(args, 'wandb_enabled', False)

def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.add_reverse_relations)

def initialize_model_directory(args: Namespace, random_seed: Optional[int] = None) -> None:
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    entire_graph_tag = '-EG' if args.train_entire_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    # Hyperparameter signature
    if args.model in ['rule']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.history_num_layers,
            args.learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta
        )
    elif args.model.startswith('point'):
        if args.baseline == 'avg_reward':
            print('* Policy Gradient Baseline: average reward')
        elif args.baseline == 'avg_reward_normalized':
            print('* Policy Gradient Baseline: average reward baseline plus normalization')
        else:
            print('* Policy Gradient Baseline: None')
        if args.action_dropout_anneal_interval < 1000:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.action_dropout_anneal_factor,
                args.action_dropout_anneal_interval,
                args.bandwidth,
                args.beta
            )
            if args.mu != 1.0:
                hyperparam_sig += '-{}'.format(args.mu)
        else:
            hyperparam_sig = 's{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.seed,
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta
            )
        if args.reward_shaping_threshold > 0:
            hyperparam_sig += '-{}'.format(args.reward_shaping_threshold)
    elif args.model == 'distmult':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model == 'complex':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model == 'transe':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model in ['conve', 'hypere', 'triplee']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.num_out_channels,
            args.kernel_size,
            args.emb_dropout_rate,
            args.hidden_dropout_rate,
            args.feat_dropout_rate,
            args.label_smoothing_epsilon
        )
    else:
        raise NotImplementedError

    datetimestr = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig
    )
    model_sub_dir = model_sub_dir + '_d+' + datetimestr
    if getattr(args, 'train_hop', 0):
        model_sub_dir += '-train{}hop'.format(args.train_hop)
    if args.model == 'set':
        model_sub_dir += '-{}'.format(args.beam_size)
        model_sub_dir += '-{}'.format(args.num_paths_per_entity)
    if args.relation_only:
        model_sub_dir += '-ro'
    elif args.relation_only_in_path:
        model_sub_dir += '-rpo'
    elif args.type_only:
        model_sub_dir += '-to'

    if args.test:
        model_sub_dir += '-test'

    if random_seed:
        model_sub_dir += '.{}'.format(random_seed)

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir

def summarize_model_parameters(model: Optional[torch.nn.Module]) -> Optional[Dict]:
    """Return parameter counts and estimated sizes (in MB) broken down by top-level modules."""
    if model is None:
        return None
    module_stats: Dict[str, Dict[str, float]] = {}
    total_params = 0
    trainable_params = 0
    total_bytes = 0
    trainable_bytes = 0
    mb = 1024 * 1024

    for name, param in model.named_parameters():
        numel = int(param.numel())
        elem_bytes = int(param.element_size()) if hasattr(param, 'element_size') else 4
        prefix = name.split('.')[0] if '.' in name else name
        entry = module_stats.setdefault(prefix, {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'total_size_bytes': 0,
            'trainable_size_bytes': 0,
        })
        entry['total_parameters'] += numel
        entry['total_size_bytes'] += numel * elem_bytes
        total_params += numel
        total_bytes += numel * elem_bytes
        if param.requires_grad:
            entry['trainable_parameters'] += numel
            entry['trainable_size_bytes'] += numel * elem_bytes
            trainable_params += numel
            trainable_bytes += numel * elem_bytes

    breakdown = {}
    for prefix, entry in module_stats.items():
        frozen_params = entry['total_parameters'] - entry['trainable_parameters']
        frozen_bytes = entry['total_size_bytes'] - entry['trainable_size_bytes']
        breakdown[prefix] = {
            'total_parameters': entry['total_parameters'],
            'trainable_parameters': entry['trainable_parameters'],
            'frozen_parameters': frozen_params,
            'total_size_mb': round(entry['total_size_bytes'] / mb, 4),
            'trainable_size_mb': round(entry['trainable_size_bytes'] / mb, 4),
            'frozen_size_mb': round(frozen_bytes / mb, 4),
        }

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'total_size_mb': round(total_bytes / mb, 4),
        'trainable_size_mb': round(trainable_bytes / mb, 4),
        'frozen_size_mb': round((total_bytes - trainable_bytes) / mb, 4),
        'parameter_breakdown': breakdown
    }


def dump_hyperparameters(args, model: Optional[torch.nn.Module] = None):
    """Serialize resolved hyperparameters (plus model stats) to stdout and a JSON file."""
    hparams = {k: v for k, v in sorted(vars(args).items())}
    model_stats = summarize_model_parameters(model)
    if model_stats is not None:
        hparams['model_parameter_stats'] = model_stats
    payload = json.dumps(hparams, indent=2, sort_keys=True)
    print('Resolved hyperparameters:\n{}'.format(payload))
    output_path = args.hparams_output_path or os.path.join(args.model_dir, 'hparams.json')
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            f.write(payload)
        print('Hyperparameters saved to {}'.format(output_path))
    except Exception as exc:
        print('Failed to write hyperparameters to {}: {}'.format(output_path, exc))
    return output_path


def dump_metrics(args, metrics: Optional[Dict]) -> Optional[str]:
    """Serialize evaluation metrics to JSON when requested."""
    if metrics is None or not getattr(args, 'metrics_output_path', ''):
        return None
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    output_path = args.metrics_output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            f.write(payload)
        print('Metrics saved to {}'.format(output_path))
        return output_path
    except Exception as exc:
        print('Failed to write metrics to {}: {}'.format(output_path, exc))
        return None


def dump_run_metadata(args: Namespace, operation: str) -> Optional[str]:
    """Record the checkpoint produced or consumed by a completed run."""
    output_path = getattr(args, 'run_metadata_output_path', '')
    if not output_path:
        return None

    if operation == 'train':
        checkpoint_path = os.path.join(args.model_dir, 'model_best.tar')
    else:
        checkpoint_path = get_checkpoint_path(args)
    try:
        manifest_path = write_run_manifest(
            output_path,
            fingerprint=getattr(args, 'run_fingerprint', ''),
            operation=operation,
            model_dir=args.model_dir,
            checkpoint_path=checkpoint_path,
            seed=args.seed,
            train_hop=getattr(args, 'train_hop', 0),
        )
        print('Run metadata saved to {}'.format(manifest_path))
        return str(manifest_path)
    except Exception as exc:
        print('Failed to write run metadata to {}: {}'.format(output_path, exc))
        return None


def construct_model(args: Namespace) -> LFramework:
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    assert not args.model.endswith(".gc"), \
        "As far as I know this sort of logic is way legacy.\n"\
        "If not, some major assumptions have been broke."
    if args.model.endswith('.gc'):
        kg.load_fuzzy_facts()

    # NOTE: Policy Gradient is a child class to LFramework
    if args.model in ['point', 'point.gc']:
        pn = GraphSearchPolicy(args)
        lf = PolicyGradient(args, kg, pn)
    elif args.model.startswith('point.rs'):
        pn = GraphSearchPolicy(args)
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'complex':
            fn = ComplEx(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'distmult':
            fn = DistMult(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'transe':
            fn = TransE(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
    elif args.model == 'complex':
        fn = ComplEx(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'distmult':
        fn = DistMult(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'transe':
        fn = TransE(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf

def subsample_examples(
    examples: list[ExampleT],
    fraction: float = 1.0,
    max_examples: int = 0,
    seed: int = 0,
    split_name: str = 'data',
) -> List[ExampleT]:
    fraction = float(fraction)
    max_examples = int(max_examples or 0)
    if fraction >= 1.0 and max_examples <= 0:
        return examples
    if fraction <= 0:
        raise ValueError('{} fraction must be > 0, got {}'.format(split_name, fraction))
    original_size = len(examples)
    target_size = original_size
    if fraction < 1.0:
        target_size = max(1, int(round(original_size * fraction)))
    if max_examples > 0:
        target_size = min(target_size, max_examples)
    if target_size >= original_size:
        return examples
    rng = random.Random(seed)
    sampled_examples = rng.sample(examples, target_size)
    print('Using {}/{} {} examples (fraction={}, max_examples={})'.format(
        len(sampled_examples), original_size, split_name, fraction, max_examples))
    return sampled_examples

def filter_examples_by_hop(
    examples: list[QAExample],
    whitelisted_hops: set[int],
    split_name: str,
    required: bool = False,
) -> list[QAExample]:
    """Keep examples whose Hops metadata matches the whitelisted hop counts."""
    if not whitelisted_hops:
        return examples
    filtered = [
        example for example in examples
        if example.Hops in whitelisted_hops
    ]
    _hop_label = ','.join(str(hop) for hop in sorted(whitelisted_hops))
    print(f'Using {len(filtered)}/{len(examples)} {split_name} examples for hop(s) {_hop_label}.')
    if required and not filtered:
        raise ValueError( f'No {split_name} examples matched hop(s) {_hop_label}. Ensure the QA data contains a Hops column.')
    return filtered

def train(lf: LFramework) -> None:
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')

    if args.use_question_encoder:
        # Load QA-style train/dev with question tokens
        train_data, dev_data, _, _ = data_utils.load_qa_data(
            args.cached_qa_metadata_path,
            args.raw_QAData_path,
            args.bert_model_name,
            entity_index_path,
            relation_index_path,
            force_recompute=args.recompute_qadata_cache,
            evaluate_paraphrases=False,
            filter_original_paraphrases=args.filter_original_paraphrases
        )
        assert 'NELL' not in args.data_dir,  "We have not accounted for NELL yet"
        seen_entities = set()
        # dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    else:
        train_data = data_utils.load_triples(
            train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
            add_reverse_relations=args.add_reversed_training_edges)

        # Only construct dev_data from triples when not using the question encoder
        if 'NELL' in args.data_dir:
            adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
            seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
        else:
            seen_entities = set()
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    if args.train_hop:
        if not args.use_question_encoder:
            raise ValueError('--train_hop requires --use_question_encoder and QA data with Hops metadata.')
        train_data = filter_examples_by_hop(train_data, {args.train_hop}, 'train', required=True)
        dev_data = filter_examples_by_hop(dev_data, {args.train_hop}, 'dev', required=True)
        if args.num_rollout_steps != args.train_hop:
            print(
                f'Warning: --train_hop={args.train_hop} but --num_rollout_steps={args.num_rollout_steps}. '
                'Set both to the same value for an exact n-hop experiment.'
            )
    train_data = subsample_examples(
        train_data, args.train_data_fraction, args.max_train_examples, args.seed, 'train')
    dev_data = subsample_examples(
        dev_data, args.dev_data_fraction, args.max_dev_examples, args.seed + 1, 'dev')
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    # Ensure wandb is initialized before training if requested
    if not getattr(args, 'wandb_enabled', False):
        setup_wandb(args, job_type='train')
    # Train with QA questions and evaluate on standard triples dev set
    lf.run_train(train_data, dev_data)

def inference(lf: LFramework) -> dict[str, dict[str, object]]:
    lf.batch_size = args.dev_batch_size
    lf.eval()
    _wandb_enabled = getattr(args, 'wandb_enabled', False)
    _wandb = None
    if getattr(args, 'wandb', False) and not _wandb_enabled:
        setup_wandb(args, job_type='inference')
        _wandb_enabled = getattr(args, 'wandb_enabled', False)
    if _wandb_enabled:
        try:
            import wandb as _wandb  # type: ignore
        except Exception:
            _wandb_enabled = False
            _wandb = None
    if args.model == 'hypere':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        secondary_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(secondary_kg_state_dict)
    elif args.model == 'triplee':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        complex_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(complex_kg_state_dict)
        distmult_kg_state_dict = get_distmult_kg_state_dict(torch.load(args.distmult_state_dict_path))
        lf.tertiary_kg.load_state_dict(distmult_kg_state_dict)
    else:
        lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    eval_metrics: dict[str, dict[str, object]] = {
        'dev': {},
        'test': {}
    }

    def _print_rollout_metrics(split_name: str, metrics: Dict[str, float]) -> None:
        hits_keys = sorted(
            [k for k in metrics.keys() if k.startswith('hits@')],  # noqa: SIM118
            key=lambda item: int(item.split('@')[1]) if item.count('@') == 1 else item
        )
        summary = ' '.join(f"{k}={metrics[k]:.4f}" for k in hits_keys)
        num_rollouts_used = metrics.get('num_rollouts')
        pool_mode = metrics.get('pool', args.rollout_eval_pool)
        print(
            f"{split_name} rollout performance (num_rollouts={num_rollouts_used}, pool={pool_mode}): "
            f"{summary} mrr={metrics['mrr']:.4f}"
        )
        if metrics.get('faithfulness/examples', 0):
            print(
                "{} faithfulness: f1_rel={:.4f} red={:.4f} f1_sg={:.4f} ped={:.4f} answer_set_f1={:.4f}".format(
                    split_name,
                    metrics.get('faithfulness/relation_f1', 0.0),
                    metrics.get('faithfulness/relation_edit_distance', 0.0),
                    metrics.get('faithfulness/subgraph_f1', metrics.get('faithfulness/edge_f1', 0.0)),
                    metrics.get('faithfulness/ped', 0.0),
                    metrics.get('faithfulness/answer_set_f1', 0.0)
                )
            )
            if metrics.get('faithfulness/semantic_path_attempts', 0):
                print(
                    '{} semantic path coverage: {:.4f} ({:.4f}/{:.4f})'.format(
                        split_name,
                        metrics.get('faithfulness/semantic_path_coverage', 0.0),
                        metrics.get('faithfulness/semantic_path_examples', 0.0),
                        metrics.get('faithfulness/semantic_path_attempts', 0.0),
                    )
                )
        per_hop = []
        for key in sorted(k for k in metrics if k.startswith('per_hop/') and k.endswith('hits@1')):
            hop = key.split('/')[1].replace('_hits@1', '')
            per_hop.append('{}={:.4f}'.format(hop, metrics[key]))
        if per_hop:
            print("{} per-hop Hits@1: {}".format(split_name, ' '.join(per_hop)))

    def _attach_per_hop_metrics(
        eval_bucket: dict[str, object],
        per_hop_metrics: dict[int, dict[str, float]],
    ) -> None:
        for hop, hop_metrics in sorted(per_hop_metrics.items()):
            prefix = f'per_hop/{hop}hop'
            eval_bucket[f'{prefix}_examples'] = hop_metrics['examples']
            eval_bucket[f'{prefix}_hits@1'] = hop_metrics['hits@1']
            eval_bucket[f'{prefix}_hits@3'] = hop_metrics['hits@3']
            eval_bucket[f'{prefix}_hits@5'] = hop_metrics['hits@5']
            eval_bucket[f'{prefix}_hits@10'] = hop_metrics['hits@10']
            eval_bucket[f'{prefix}_mrr'] = hop_metrics['mrr']

    def _print_per_hop_ranking(split_name: str, per_hop_metrics: Dict[int, Dict[str, float]]) -> None:
        if not per_hop_metrics:
            return
        summary = ' '.join(
            '{}hop={:.4f}'.format(hop, hop_metrics['hits@1'])
            for hop, hop_metrics in sorted(per_hop_metrics.items())
        )
        print('{} per-hop Hits@1: {}'.format(split_name, summary))

    def _log_rollout_metrics_to_wandb(prefix: str, metrics: Dict[str, float]) -> None:
        if not _wandb_enabled or _wandb is None:
            return
        rollout_log = {}
        for k, v in metrics.items():
            if k.startswith('hits@') or k == 'mrr':
                rollout_log[f'{prefix}_rollout/{k}'] = float(v)
            elif k.startswith('faithfulness/'):
                rollout_log[f'{prefix}_{k}'] = float(v)
            elif k.startswith('per_hop/'):
                rollout_log[f'{prefix}_{k}'] = float(v)
        rollout_log[f'{prefix}_rollout/examples'] = float(metrics.get('examples', 0))
        if 'num_rollouts' in metrics:
            rollout_log[f'{prefix}_rollout/num_rollouts'] = float(metrics['num_rollouts'])
        if 'pool' in metrics:
            rollout_log[f'{prefix}_rollout/pool'] = metrics['pool']
        _wandb.log(rollout_log)

    if args.compute_map:
        relation_sets = [
            'concept:athletehomestadium',
            'concept:athleteplaysforteam',
            'concept:athleteplaysinleague',
            'concept:athleteplayssport',
            'concept:organizationheadquarteredincity',
            'concept:organizationhiredperson',
            'concept:personborninlocation',
            'concept:teamplayssport',
            'concept:worksfor'
        ]
        mps = []
        for r in relation_sets:
            print('* relation: {}'.format(r))
            test_path = os.path.join(args.data_dir, 'tasks', r, 'test.pairs')
            test_data, labels = data_utils.load_triples_with_label(
                test_path, r, entity_index_path, relation_index_path, seen_entities=seen_entities)
            pred_scores = lf.forward(test_data, verbose=False)
            mp = src.eval.link_MAP(test_data, pred_scores, labels, lf.kg.all_objects, verbose=True)
            mps.append(mp)
        map_ = np.mean(mps)
        print('Overall MAP = {}'.format(map_))
        eval_metrics['test']['avg_map'] = map
    elif args.eval_by_relation_type:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        to_m_rels, to_1_rels, _ = data_utils.get_relations_by_type(args.data_dir, relation_index_path)
        relation_by_types = (to_m_rels, to_1_rels)
        print('Dev set evaluation by relation type (partial graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.dev_objects, relation_by_types,
            verbose=True, beam_size=args.beam_size)
        print('Dev set evaluation by relation type (full graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.all_objects, relation_by_types,
            verbose=True, beam_size=args.beam_size)
    elif args.eval_by_seen_queries:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        seen_queries = data_utils.get_seen_queries(args.data_dir, entity_index_path, relation_index_path)
        print('Dev set evaluation by seen queries (partial graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries,
            verbose=True, beam_size=args.beam_size)
        print('Dev set evaluation by seen queries (full graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries,
            verbose=True, beam_size=args.beam_size)
    else:
        if args.use_question_encoder:
            _, dev_data, test_data, _ = data_utils.load_qa_data(
                args.cached_qa_metadata_path,
                args.raw_QAData_path,
                args.bert_model_name,
                entity_index_path,
                relation_index_path,
                force_recompute=args.recompute_qadata_cache,
                evaluate_paraphrases=args.evaluate_paraphrases,
                filter_original_paraphrases=args.filter_original_paraphrases
            )
            requested_eval_hops = parse_hop_filter(args.eval_hops)
            if requested_eval_hops:
                dev_data = filter_examples_by_hop(
                    dev_data, requested_eval_hops, 'dev evaluation', required=True)
                test_data = filter_examples_by_hop(
                    test_data, requested_eval_hops, 'test evaluation', required=True)
            for split_name, split_data in [('Dev', dev_data), ('Test', test_data)]:
                if hasattr(lf, 'supports_rollout_evaluation') and lf.supports_rollout_evaluation():
                    rollout_metrics = lf.evaluate_with_rollouts(split_data, split_name=split_name.lower())
                    if rollout_metrics:
                        _print_rollout_metrics(split_name, rollout_metrics)
                        eval_split = split_name.lower()
                        eval_metrics[eval_split] = {
                            f'rollout_{k}': v
                            for k, v in rollout_metrics.items()
                            if k.startswith('hits@')
                        }
                        for k, v in rollout_metrics.items():
                            if k.startswith('per_hop/') or k.startswith('faithfulness/'):
                                eval_metrics[eval_split][k] = v
                        eval_metrics[eval_split]['rollout_mrr'] = rollout_metrics['mrr']
                        _log_rollout_metrics_to_wandb(f'inference/{eval_split}', rollout_metrics)
            return eval_metrics

        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        print('Dev set performance:')
        pred_scores = lf.forward(dev_data, verbose=args.save_beam_search_paths)
        dev_metrics = src.eval.hits_and_ranks(
            dev_data, pred_scores, lf.kg.dev_objects, verbose=True, beam_size=args.beam_size)
        eval_metrics['dev'] = {}
        eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
        eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
        eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
        eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
        eval_metrics['dev']['mrr'] = dev_metrics[4]
        src.eval.hits_and_ranks(
            dev_data, pred_scores, lf.kg.all_objects, verbose=True, beam_size=args.beam_size)
        if hasattr(lf, 'supports_rollout_evaluation') and lf.supports_rollout_evaluation():
            rollout_dev_metrics = lf.evaluate_with_rollouts(dev_data, split_name='dev')
            if rollout_dev_metrics:
                _print_rollout_metrics('Dev', rollout_dev_metrics)
                for k, v in rollout_dev_metrics.items():
                    if k.startswith('hits@'):
                        eval_metrics['dev'][f'rollout_{k}'] = v
                eval_metrics['dev']['rollout_mrr'] = rollout_dev_metrics['mrr']
                _log_rollout_metrics_to_wandb('inference/dev', rollout_dev_metrics)
        if _wandb_enabled and _wandb is not None:
            dev_log = {
                'inference/dev/hits@1': float(dev_metrics[0]),
                'inference/dev/hits@3': float(dev_metrics[1]),
                'inference/dev/hits@5': float(dev_metrics[2]),
                'inference/dev/hits@10': float(dev_metrics[3]),
                'inference/dev/mrr': float(dev_metrics[4]),
                'inference/dev/examples': float(len(dev_data))
            }
            _wandb.log(dev_log)
        print('Test set performance:')
        pred_scores = lf.forward(test_data, verbose=False)
        test_metrics = src.eval.hits_and_ranks(
            test_data, pred_scores, lf.kg.all_objects, verbose=True, beam_size=args.beam_size)
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]
        if hasattr(lf, 'supports_rollout_evaluation') and lf.supports_rollout_evaluation():
            rollout_test_metrics = lf.evaluate_with_rollouts(test_data, split_name='test')
            if rollout_test_metrics:
                _print_rollout_metrics('Test', rollout_test_metrics)
                for k, v in rollout_test_metrics.items():
                    if k.startswith('hits@'):
                        eval_metrics['test'][f'rollout_{k}'] = v
                    elif k.startswith('per_hop/'):
                        eval_metrics['test'][k] = v
                eval_metrics['test']['rollout_mrr'] = rollout_test_metrics['mrr']
                _log_rollout_metrics_to_wandb('inference/test', rollout_test_metrics)
        if _wandb_enabled and _wandb is not None:
            test_log = {
                'inference/test/hits@1': float(test_metrics[0]),
                'inference/test/hits@3': float(test_metrics[1]),
                'inference/test/hits@5': float(test_metrics[2]),
                'inference/test/hits@10': float(test_metrics[3]),
                'inference/test/mrr': float(test_metrics[4]),
                'inference/test/examples': float(len(test_data))
            }
            _wandb.log(test_log)

    return eval_metrics

def run_ablation_studies(args):
    """
    Run the ablation study experiments reported in the paper.
    """
    def set_up_lf_for_inference(args):
        initialize_model_directory(args)
        lf = construct_model(args)
        lf.cuda()
        lf.batch_size = args.dev_batch_size
        lf.load_checkpoint(get_checkpoint_path(args))
        lf.eval()
        return lf

    def rel_change(metrics, ab_system, kg_portion):
        ab_system_metrics = metrics[ab_system][kg_portion]
        base_metrics = metrics['ours'][kg_portion]
        return int(np.round((ab_system_metrics - base_metrics) / base_metrics * 100))

    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dataset = os.path.basename(args.data_dir)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(
        dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    to_m_rels, to_1_rels, (to_m_ratio, to_1_ratio) = data_utils.get_relations_by_type(args.data_dir, relation_index_path)
    relation_by_types = (to_m_rels, to_1_rels)
    to_m_ratio *= 100
    to_1_ratio *= 100
    seen_queries, (seen_ratio, unseen_ratio) = data_utils.get_seen_queries(args.data_dir, entity_index_path, relation_index_path)
    seen_ratio *= 100
    unseen_ratio *= 100

    systems = ['ours', '-ad', '-rs']
    mrrs, to_m_mrrs, to_1_mrrs, seen_mrrs, unseen_mrrs = {}, {}, {}, {}, {}
    for system in systems:
        print('** Evaluating {} system **'.format(system))
        if system == '-ad':
            args.action_dropout_rate = 0.0
            if dataset == 'umls':
                # adjust dropout hyperparameters
                args.emb_dropout_rate = 0.3
                args.ff_dropout_rate = 0.1
        elif system == '-rs':
            config_path = os.path.join('configs', '{}.sh'.format(dataset.lower()))
            args = parse_args()
            args = data_utils.load_configs(args, config_path)
        
        lf = set_up_lf_for_inference(args)
        pred_scores = lf.forward(dev_data, verbose=False)
        _, _, _, _, mrr = src.eval.hits_and_ranks(
            dev_data, pred_scores, lf.kg.dev_objects, verbose=True, beam_size=args.beam_size)
        if to_1_ratio == 0:
            to_m_mrr = mrr
            to_1_mrr = -1
        else:
            to_m_mrr, to_1_mrr = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.dev_objects, relation_by_types,
                verbose=True, beam_size=args.beam_size)
        seen_mrr, unseen_mrr = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries,
            verbose=True, beam_size=args.beam_size)
        mrrs[system] = {'': mrr * 100}
        to_m_mrrs[system] = {'': to_m_mrr * 100}
        to_1_mrrs[system] = {'': to_1_mrr  * 100}
        seen_mrrs[system] = {'': seen_mrr * 100}
        unseen_mrrs[system] = {'': unseen_mrr * 100}
        _, _, _, _, mrr_full_kg = src.eval.hits_and_ranks(
            dev_data, pred_scores, lf.kg.all_objects, verbose=True, beam_size=args.beam_size)
        if to_1_ratio == 0:
            to_m_mrr_full_kg = mrr_full_kg
            to_1_mrr_full_kg = -1
        else:
            to_m_mrr_full_kg, to_1_mrr_full_kg = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.all_objects, relation_by_types,
                verbose=True, beam_size=args.beam_size)
        seen_mrr_full_kg, unseen_mrr_full_kg = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries,
            verbose=True, beam_size=args.beam_size)
        mrrs[system]['full_kg'] = mrr_full_kg * 100
        to_m_mrrs[system]['full_kg'] = to_m_mrr_full_kg * 100
        to_1_mrrs[system]['full_kg'] = to_1_mrr_full_kg * 100
        seen_mrrs[system]['full_kg'] = seen_mrr_full_kg * 100
        unseen_mrrs[system]['full_kg'] = unseen_mrr_full_kg * 100

    # overall system comparison (table 3)
    print('Partial graph evaluation')
    print('--------------------------')
    print('Overall system performance')
    print('Ours(ConvE)\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours'][''], mrrs['-rs'][''], mrrs['-ad']['']))
    print('--------------------------')
    # performance w.r.t. relation types (table 4, 6)
    print('Performance w.r.t. relation types')
    print('\tTo-many\t\t\t\tTo-one\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours'][''], to_m_mrrs['-rs'][''], rel_change(to_m_mrrs, '-rs', ''), to_m_mrrs['-ad'][''], rel_change(to_m_mrrs, '-ad', ''),
        to_1_ratio, to_1_mrrs['ours'][''], to_1_mrrs['-rs'][''], rel_change(to_1_mrrs, '-rs', ''), to_1_mrrs['-ad'][''], rel_change(to_1_mrrs, '-ad', '')))
    print('--------------------------')
    # performance w.r.t. seen queries (table 5, 7)
    print('Performance w.r.t. seen/unseen queries')
    print('\tSeen\t\t\t\tUnseen\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours'][''], seen_mrrs['-rs'][''], rel_change(seen_mrrs, '-rs', ''), seen_mrrs['-ad'][''], rel_change(seen_mrrs, '-ad', ''),
        unseen_ratio, unseen_mrrs['ours'][''], unseen_mrrs['-rs'][''], rel_change(unseen_mrrs, '-rs', ''), unseen_mrrs['-ad'][''], rel_change(unseen_mrrs, '-ad', '')))
    print()
    print('Full graph evaluation')
    print('--------------------------')
    print('Overall system performance')
    print('Ours(ConvE)\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours']['full_kg'], mrrs['-rs']['full_kg'], mrrs['-ad']['full_kg']))
    print('--------------------------')
    print('Performance w.r.t. relation types')
    print('\tTo-many\t\t\t\tTo-one\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours']['full_kg'], to_m_mrrs['-rs']['full_kg'], rel_change(to_m_mrrs, '-rs', 'full_kg'), to_m_mrrs['-ad']['full_kg'], rel_change(to_m_mrrs, '-ad', 'full_kg'),
        to_1_ratio, to_1_mrrs['ours']['full_kg'], to_1_mrrs['-rs']['full_kg'], rel_change(to_1_mrrs, '-rs', 'full_kg'), to_1_mrrs['-ad']['full_kg'], rel_change(to_1_mrrs, '-ad', 'full_kg')))
    print('--------------------------')
    print('Performance w.r.t. seen/unseen queries')
    print('\tSeen\t\t\t\tUnseen\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours']['full_kg'], seen_mrrs['-rs']['full_kg'], rel_change(seen_mrrs, '-rs', 'full_kg'), seen_mrrs['-ad']['full_kg'], rel_change(seen_mrrs, '-ad', 'full_kg'),
        unseen_ratio, unseen_mrrs['ours']['full_kg'], unseen_mrrs['-rs']['full_kg'], rel_change(unseen_mrrs, '-rs', 'full_kg'), unseen_mrrs['-ad']['full_kg'], rel_change(unseen_mrrs, '-ad', 'full_kg')))

def export_to_embedding_projector(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_to_embedding_projector()

def export_reward_shaping_parameters(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_reward_shaping_parameters()

def export_fuzzy_facts(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_fuzzy_facts()

def export_error_cases(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.batch_size = args.dev_batch_size
    lf.eval()
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    lf.load_checkpoint(get_checkpoint_path(args))
    print('Dev set performance:')
    pred_scores = lf.forward(dev_data, verbose=False)
    src.eval.hits_and_ranks(
        dev_data, pred_scores, lf.kg.dev_objects, verbose=True, beam_size=args.beam_size)
    src.eval.export_error_cases(
        dev_data, pred_scores, lf.kg.dev_objects,
        os.path.join(lf.model_dir, 'error_cases.pkl'), beam_size=args.beam_size)

def compute_fact_scores(lf):
    data_dir = args.data_dir
    train_path = os.path.join(data_dir, 'train.triples')
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(train_path, entity_index_path, relation_index_path)
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    test_data = data_utils.load_triples(test_path, entity_index_path, relation_index_path)
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    train_scores = lf.forward_fact(train_data)
    dev_scores = lf.forward_fact(dev_data)
    test_scores = lf.forward_fact(test_data)

    print('Train set average fact score: {}'.format(float(train_scores.mean())))
    print('Dev set average fact score: {}'.format(float(dev_scores.mean())))
    print('Test set average fact score: {}'.format(float(test_scores.mean())))

def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path

def load_configs(config_path):
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

def run_experiment(run_args):
    global args
    args = run_args
    configure_runtime(args)

    if args.test:
        if 'NELL' in args.data_dir:
            dataset = os.path.basename(args.data_dir)
            args.distmult_state_dict_path = data_utils.change_to_test_model_path(dataset, args.distmult_state_dict_path)
            args.complex_state_dict_path = data_utils.change_to_test_model_path(dataset, args.complex_state_dict_path)
            args.conve_state_dict_path = data_utils.change_to_test_model_path(dataset, args.conve_state_dict_path)
        args.data_dir += '.test'

    if args.process_data:

        # Process knowledge graph data

        process_data()
    else:
        with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
            if args.search_random_seed:

                # Search for best random seed

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.rss'.format(task, args.model)
                o_f = open(out_log, 'w')

                print('** Search Random Seed **')
                o_f.write('** Search Random Seed **\n')
                o_f.close()
                num_runs = 5

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                mrrs_search = {}
                for i in range(num_runs):

                    o_f = open(out_log, 'a')

                    random_seed = random.randint(0, 1e16)
                    print("\nRandom seed = {}\n".format(random_seed))
                    o_f.write("\nRandom seed = {}\n\n".format(random_seed))
                    torch.manual_seed(random_seed)
                    torch.cuda.manual_seed_all(args, random_seed)
                    initialize_model_directory(args, random_seed)
                    setup_wandb(args, job_type='sweep')
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[random_seed] = metrics['test']['hits_at_1']
                    hits_at_10s[random_seed] = metrics['test']['hits_at_10']
                    mrrs[random_seed] = metrics['test']['mrr']
                    mrrs_search[random_seed] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Random Seed\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Random Seed\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')

                    # compute result variance
                    import numpy as np
                    hits_at_1s_ = list(hits_at_1s.values())
                    hits_at_10s_ = list(hits_at_10s.values())
                    mrrs_ = list(mrrs.values())
                    print('Hits@1 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    print('Hits@10 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    print('MRR mean: {:.3f}\tstd: {:.6f}'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.write('Hits@1 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    o_f.write('Hits@10 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    o_f.write('MRR mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.close()
                    
                # find best random seed
                best_random_seed, best_mrr = sorted(mrrs_search.items(), key=lambda x: x[1], reverse=True)[0]
                print('* Best Random Seed = {}'.format(best_random_seed))
                print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                    hits_at_1s[best_random_seed],
                    hits_at_10s[best_random_seed],
                    mrrs[best_random_seed]))
                with open(out_log, 'a'):
                    o_f.write('* Best Random Seed = {}\n'.format(best_random_seed))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\n'.format(
                        hits_at_1s[best_random_seed],
                        hits_at_10s[best_random_seed],
                        mrrs[best_random_seed])
                    )
                    o_f.close()

            elif args.grid_search:

                # Grid search

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.gs'.format(task, args.model)
                o_f = open(out_log, 'w')

                print("** Grid Search **")
                o_f.write("** Grid Search **\n")
                hyperparameters = args.tune.split(',')

                if args.tune == '' or len(hyperparameters) < 1:
                    print("No hyperparameter specified.")
                    sys.exit(0)

                grid = hp_range[hyperparameters[0]]
                for hp in hyperparameters[1:]:
                    grid = itertools.product(grid, hp_range[hp])

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                grid = list(grid)
                print('* {} hyperparameter combinations to try'.format(len(grid)))
                o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
                o_f.close()

                for i, grid_entry in enumerate(list(grid)):

                    o_f = open(out_log, 'a')

                    if not (type(grid_entry) is list or type(grid_entry) is list):
                        grid_entry = [grid_entry]
                    grid_entry = flatten(grid_entry)
                    print('* Hyperparameter Set {}:'.format(i))
                    o_f.write('* Hyperparameter Set {}:\n'.format(i))
                    signature = ''
                    for j in range(len(grid_entry)):
                        hp = hyperparameters[j]
                        value = grid_entry[j]
                        if hp == 'bandwidth':
                            setattr(args, hp, int(value))
                        else:
                            setattr(args, hp, float(value))
                        signature += ':{}'.format(value)
                        print('* {}: {}'.format(hp, value))
                    initialize_model_directory(args)
                    setup_wandb(args, job_type='grid-search')
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[signature] = metrics['dev']['hits_at_1']
                    hits_at_10s[signature] = metrics['dev']['hits_at_10']
                    mrrs[signature] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Signature\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------\n')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Signature\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')
                    # find best hyperparameter set
                    best_signature, best_mrr = sorted(mrrs.items(), key=lambda x:x[1], reverse=True)[0]
                    print('* best hyperparameter set')
                    o_f.write('* best hyperparameter set\n')
                    best_hp_values = best_signature.split(':')[1:]
                    for i, value in enumerate(best_hp_values):
                        hp_name = hyperparameters[i]
                        hp_value = best_hp_values[i]
                        print('* {}: {}'.format(hp_name, hp_value))
                    print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\ns'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))

                    o_f.close()

            elif args.run_ablation_studies:
                run_ablation_studies(args)
            else:
                if args.checkpoint_path and not args.train:
                    args.model_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
                    print('Reusing model directory: {}'.format(args.model_dir))
                else:
                    initialize_model_directory(args)
                lf = None
                if args.dump_hparams or args.dump_hparams_only:
                    lf = construct_model(args)
                    dump_hyperparameters(args, lf)
                    if args.dump_hparams_only:
                        return
                setup_wandb(args, job_type='run')
                if lf is None:
                    lf = construct_model(args)
                lf.cuda()

                if args.train:
                    train(lf)
                    if args.evaluate_per_hop:
                        if not args.use_question_encoder:
                            raise ValueError('--evaluate_per_hop requires --use_question_encoder.')
                        metrics = inference(lf)
                        dump_metrics(args, metrics)
                    dump_run_metadata(args, 'train')
                elif args.inference:
                    metrics = inference(lf)
                    dump_metrics(args, metrics)
                    dump_run_metadata(args, 'inference')
                elif args.eval_by_relation_type:
                    metrics = inference(lf)
                    dump_metrics(args, metrics)
                elif args.eval_by_seen_queries:
                    metrics = inference(lf)
                    dump_metrics(args, metrics)
                elif args.export_to_embedding_projector:
                    export_to_embedding_projector(lf)
                elif args.export_reward_shaping_parameters:
                    export_reward_shaping_parameters(lf)
                elif args.compute_fact_scores:
                    compute_fact_scores(lf)
                elif args.export_fuzzy_facts:
                    export_fuzzy_facts(lf)
                elif args.export_error_cases:
                    export_error_cases(lf)

if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        debugpy.listen(args.debug_port)
        print(f"debugpy listening on {args.debug_port}", flush=True)
        debugpy.wait_for_client()
    
    run_experiment(args)
