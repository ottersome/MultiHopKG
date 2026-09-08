"""Shared configuration and artifact helpers for experiment command-line tools."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Set as AbstractSet
from pathlib import Path


PathLike = str | os.PathLike[str]
RUN_MANIFEST_SCHEMA_VERSION = 1


EXPERIMENT_STORE_TRUE_ARGS = frozenset({
    'process_data',
    'train',
    'inference',
    'search_random_seed',
    'eval',
    'eval_by_relation_type',
    'eval_by_seen_queries',
    'run_ablation_studies',
    'run_analysis',
    'dump_hparams',
    'dump_hparams_only',
    'test',
    'group_examples_by_query',
    'add_reversed_training_edges',
    'use_action_space_bucketing',
    'type_only',
    'relation_only',
    'relation_only_in_path',
    'use_question_encoder',
    'allow_direct_answer_edges',
    'recompute_qadata_cache',
    'evaluate_paraphrases',
    'filter_original_paraphrases',
    'disable_rollout_eval',
    'keep_rollout_eval_dropout',
    'visualize_paths',
    'save_beam_search_paths',
    'export_to_embedding_projector',
    'export_reward_shaping_parameters',
    'compute_fact_scores',
    'export_fuzzy_facts',
    'export_error_cases',
    'compute_map',
    'grid_search',
    'debug',
    'wandb',
    'disable_checkpoint_saving',
    'disable_early_stopping',
    'evaluate_per_hop',
})


def strip_inline_comment(line: str) -> str:
    """Strip an unquoted shell comment from a config assignment."""
    in_single = False
    in_double = False
    escaped = False
    characters: list[str] = []
    for character in line:
        if escaped:
            characters.append(character)
            escaped = False
            continue
        if character == '\\':
            characters.append(character)
            escaped = True
            continue
        if character == "'" and not in_double:
            in_single = not in_single
        elif character == '"' and not in_single:
            in_double = not in_double
        elif character == '#' and not in_single and not in_double:
            break
        characters.append(character)
    return ''.join(characters).strip()


def load_shell_config(config_path: PathLike) -> dict[str, str]:
    """Load simple shell-style ``key=value`` assignments without executing them."""
    path = Path(config_path)
    config: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = strip_inline_comment(raw_line)
        if not line or line.startswith(('#', '#!')):
            continue
        assignment = line.split('=', 1)
        if len(assignment) != 2:
            raise OSError(
                f'config file does not include a key-value pair in line:\n{raw_line}'
            )
        key = assignment[0].strip()
        value = assignment[1].strip()
        if not key:
            continue
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        config[key] = value
    return config


def config_to_cli_args(
    config: Mapping[str, str],
    store_true_args: AbstractSet[str] = EXPERIMENT_STORE_TRUE_ARGS,
) -> list[str]:
    """Convert config assignments to argparse-compatible command arguments."""
    cli_args: list[str] = []
    for key, value in config.items():
        flag = f'--{key}'
        if key in store_true_args:
            if value == 'True':
                cli_args.append(flag)
            elif value != 'False':
                raise ValueError(f'Unsupported boolean value for {key}: {value}')
        else:
            cli_args.extend([flag, value])
    return cli_args


def summarize_error(stderr: str, stdout: str, max_characters: int = 400) -> str:
    """Return the tail of subprocess output as a compact failure description."""
    for output in (stderr, stdout):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return ' | '.join(lines[-4:])[:max_characters]
    return 'No output captured'


def ensure_parent(path: PathLike) -> Path:
    """Create an artifact's parent directory and return its normalized Path."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return artifact_path


def write_run_manifest(
    output_path: PathLike,
    *,
    fingerprint: str,
    operation: str,
    model_dir: PathLike,
    checkpoint_path: PathLike,
    seed: int,
    train_hop: int,
) -> Path:
    """Atomically record a completed experiment and its reusable checkpoint."""
    manifest_path = ensure_parent(output_path)
    resolved_model_dir = Path(model_dir).resolve()
    resolved_checkpoint = Path(checkpoint_path).resolve()
    payload: dict[str, object] = {
        'schema_version': RUN_MANIFEST_SCHEMA_VERSION,
        'fingerprint': fingerprint,
        'completed': resolved_checkpoint.is_file(),
        'operation': operation,
        'model_dir': str(resolved_model_dir),
        'checkpoint_path': str(resolved_checkpoint),
        'seed': seed,
        'train_hop': train_hop,
    }
    temporary_path = manifest_path.with_name(
        f'.{manifest_path.name}.{os.getpid()}.tmp'
    )
    try:
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary_path, manifest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return manifest_path
