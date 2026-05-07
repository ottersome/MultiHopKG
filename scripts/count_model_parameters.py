#!/usr/bin/env python3
"""
Count model parameters from a repo config.

Reports total parameters, BERT/question-encoder parameters, and the model size
excluding BERT. This constructs the model architecture but does not train or run
inference.
"""

import argparse
import copy
import os
import sys
from typing import Dict, Iterable, Tuple


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_shell_config(config_path: str) -> Dict[str, str]:
    values = {}
    with open(config_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("#", 1)[0].strip()
            if not key or any(ch.isspace() for ch in key):
                continue
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            values[key] = value
    return values


def parse_bool(value: str) -> bool:
    if value in ("True", "true", "1", "yes", "y"):
        return True
    if value in ("False", "false", "0", "no", "n"):
        return False
    raise ValueError(f"Unrecognized boolean value: {value}")


def apply_config(args, config_values: Dict[str, str], repo_parser=None):
    action_by_dest = {}
    if repo_parser is not None:
        action_by_dest = {
            action.dest: action
            for action in getattr(repo_parser, "_actions", [])
            if getattr(action, "dest", None)
        }
    for name, value in config_values.items():
        if not hasattr(args, name):
            continue
        current = getattr(args, name)
        action = action_by_dest.get(name)
        declared_type = getattr(action, "type", None) if action is not None else None
        action_class = action.__class__.__name__ if action is not None else ""
        if action_class in ("_StoreTrueAction", "_StoreFalseAction") or isinstance(current, bool):
            setattr(args, name, parse_bool(value))
        elif declared_type is int:
            setattr(args, name, int(value))
        elif declared_type is float:
            setattr(args, name, float(value))
        elif declared_type is bool:
            setattr(args, name, parse_bool(value))
        elif declared_type is str:
            setattr(args, name, value)
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(args, name, int(value))
        elif isinstance(current, float):
            setattr(args, name, float(value))
        else:
            setattr(args, name, value)
    return args


def parameter_count(parameters: Iterable) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in parameters:
        numel = int(param.numel())
        total += numel
        if param.requires_grad:
            trainable += numel
    return total, trainable


def format_count(value: int) -> str:
    return f"{value:,} ({value / 1_000_000:.3f}M)"


def bytes_to_mb(num_params: int, bytes_per_param: int = 4) -> float:
    return num_params * bytes_per_param / (1024 * 1024)


def construct_model(args):
    from src.knowledge_graph import KnowledgeGraph
    from src.emb.fact_network import ComplEx, ConvE, DistMult, TransE
    from src.emb.emb import EmbeddingBasedMethod
    from src.rl.graph_search.pn import GraphSearchPolicy
    from src.rl.graph_search.pg import PolicyGradient
    from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient

    kg = KnowledgeGraph(args)

    if args.model in ["point", "point.gc"]:
        pn = GraphSearchPolicy(args)
        return PolicyGradient(args, kg, pn)

    if args.model.startswith("point.rs"):
        pn = GraphSearchPolicy(args)
        fn_model = args.model.split(".")[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == "complex":
            fn = ComplEx(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == "distmult":
            fn = DistMult(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == "conve":
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == "transe":
            fn = TransE(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        else:
            raise NotImplementedError(fn_model)
        return RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)

    if args.model == "complex":
        return EmbeddingBasedMethod(args, kg, ComplEx(args))
    if args.model == "distmult":
        return EmbeddingBasedMethod(args, kg, DistMult(args))
    if args.model == "conve":
        return EmbeddingBasedMethod(args, kg, ConvE(args, kg.num_entities))
    if args.model == "transe":
        return EmbeddingBasedMethod(args, kg, TransE(args))
    raise NotImplementedError(args.model)


def main() -> None:
    cli = argparse.ArgumentParser(description="Count model parameters for a config.")
    cli.add_argument("config", help="Path to config .sh file")
    cli.add_argument("--bert-name-fragment", default="_q_encoder",
                     help="Parameter-name fragment used to identify BERT params")
    cli.add_argument("--model-root-dir", default=None,
                     help="Optional override for args.model_root_dir")
    known, extra = cli.parse_known_args()

    # src.parse_args parses sys.argv at import time. Hide this script's args from it,
    # then build a normal default namespace manually from the parser.
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    from src.parse_args import parser as repo_parser
    sys.argv = original_argv

    args = repo_parser.parse_args([])
    args = apply_config(args, load_shell_config(known.config), repo_parser)
    if known.model_root_dir is not None:
        args.model_root_dir = known.model_root_dir
    if extra:
        args = repo_parser.parse_args(extra, namespace=args)

    model = construct_model(args)

    total, trainable = parameter_count(model.parameters())
    bert_named = [
        (name, param)
        for name, param in model.named_parameters()
        if known.bert_name_fragment in name
    ]
    bert_total, bert_trainable = parameter_count(param for _, param in bert_named)
    non_bert_total = total - bert_total
    non_bert_trainable = trainable - bert_trainable

    print(f"Config: {known.config}")
    print(f"Model: {args.model}")
    print(f"Data dir: {args.data_dir}")
    print("")
    print("Parameter counts")
    print(f"  Total with BERT:     {format_count(total)}")
    print(f"  Trainable with BERT: {format_count(trainable)}")
    print(f"  BERT only:           {format_count(bert_total)}")
    print(f"  BERT trainable:      {format_count(bert_trainable)}")
    print(f"  Without BERT:        {format_count(non_bert_total)}")
    print(f"  Trainable no BERT:   {format_count(non_bert_trainable)}")
    print("")
    print("Approx fp32 size")
    print(f"  Total with BERT: {bytes_to_mb(total):.2f} MB")
    print(f"  BERT only:       {bytes_to_mb(bert_total):.2f} MB")
    print(f"  Without BERT:    {bytes_to_mb(non_bert_total):.2f} MB")
    print("")
    print("Top-level modules")
    module_counts: Dict[str, int] = {}
    for name, param in model.named_parameters():
        prefix = name.split(".", 1)[0]
        module_counts[prefix] = module_counts.get(prefix, 0) + int(param.numel())
    for prefix, count in sorted(module_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {prefix}: {format_count(count)}")


if __name__ == "__main__":
    main()
