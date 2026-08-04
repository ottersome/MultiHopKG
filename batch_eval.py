#!/usr/bin/env python3

"""Batch evaluator for model checkpoints with compact summaries."""
from collections import defaultdict

from dataclasses import dataclass
from pprint import pprint
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union


STORE_TRUE_ARGS = {
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
}

@dataclass
class CheckpointConfigs:
    config: Dict[str, str]
    checkpoint_paths: List[Path]

DEFAULT_PATTERNS = 's*_model.tar'


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    extra_args: List[str] = []
    if '--' in raw_argv:
        separator = raw_argv.index('--')
        extra_args = raw_argv[separator + 1:]
        raw_argv = raw_argv[:separator]

    parser = argparse.ArgumentParser(
        description='Evaluate many checkpoints with an existing experiment config.'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU id passed to src.experiments')
    parser.add_argument('--split', default='test', choices=['dev', 'test'],
                        help='Metric split to display in the compact summary')
    parser.add_argument('--pattern', action='append', default=[],
                        help='Checkpoint glob for directory targets; may be repeated')
    parser.add_argument('--no_recursive', action='store_true',
                        help='Disable recursive checkpoint discovery inside directory targets')
    parser.add_argument('--output_json', default='',
                        help='Optional path for the aggregated evaluation JSON')
    parser.add_argument('--output_csv', default='',
                        help='Optional path for the aggregated evaluation CSV')
    parser.add_argument('--log_dir', default='',
                        help='Directory for per-checkpoint logs and metrics artifacts')
    parser.add_argument('--fail_fast', action='store_true',
                        help='Stop after the first failed evaluation')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the commands that would run without executing them')
    parser.add_argument('target',
                        help='Checkpoint files or directories containing checkpoints')
    parsed = parser.parse_args(raw_argv)
    parsed.extra_args = extra_args
    return parsed


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    chars: List[str] = []
    for char in line:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == '\\':
            chars.append(char)
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == '#' and not in_single and not in_double:
            break
        chars.append(char)
    return ''.join(chars).strip()


def load_shell_config(config_path: Path) -> Dict[str, str]:
    config: Dict[str, str] = {}
    for raw_line in config_path.read_text().splitlines():
        line = strip_inline_comment(raw_line)
        if not line or line.startswith('#') or line.startswith('#!'):
            continue
        split = line.split('=', 1)
        if len(split) != 2:
            raise IOError(f"config file does not include a key-value pair in line:\n{raw_line}")
        key = split[0].strip()
        value = split[1].strip()
        if not key:
            continue
        if ((value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))):
            value = value[1:-1]
        config[key] = value
    return config


def config_to_cli_args(config: Dict[str, str]) -> List[str]:
    cli_args: List[str] = []
    for key, value in config.items():
        flag = f'--{key}'
        if key in STORE_TRUE_ARGS:
            if value == 'True':
                cli_args.append(flag)
            elif value != 'False':
                raise ValueError(f'Unsupported boolean value for {key}: {value}')
        else:
            cli_args.extend([flag, value])
    return cli_args


def natural_checkpoint_key(path: Path) -> Tuple[str, int, str]:
    file_name = path.name
    for pattern in (r'^s(\d+)_model\.tar$', r'^checkpoint-(\d+)\.tar$'):
        match = re.match(pattern, file_name)
        if match:
            return (str(path.parent), int(match.group(1)), file_name)
    return (str(path.parent), sys.maxsize, file_name)



def discover_checkpoints(target_path: str, checkpoint_pattern: str) -> Dict[str,CheckpointConfigs]:
    # TODO: Get the configs from here
    target = Path(target_path)
    if not target.is_dir():
        raise IOError("Given path is not a directory")

    all_available_checkpoints = list(target.rglob(checkpoint_pattern))
    final_results_dict = defaultdict()
    for aac in all_available_checkpoints:
        parent_path = aac.parent.resolve()
        if parent_path not in final_results_dict:
            try:
                checkpoint_configs_path = next(parent_path.rglob("*.sh"))
            except StopIteration:
                continue
            checkpoint_configs = load_shell_config(checkpoint_configs_path)
            final_results_dict[parent_path] = CheckpointConfigs(
                checkpoint_configs,
                [aac]
            )
        else:
            final_results_dict[parent_path].checkpoint_paths.append(aac)
    return final_results_dict


def sanitize_label(path: Path) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', f'{path.parent.name}_{path.stem}')


def flatten_metrics(prefix: str, value, out: Dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f'{prefix}.{key}' if prefix else key
            flatten_metrics(next_prefix, nested, out)
        return
    out[prefix] = value


def summarize_error(stderr: str, stdout: str) -> str:
    for text in (stderr, stdout):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return ' | '.join(lines[-4:])[:400]
    return 'No output captured'


def format_metric(value) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def format_per_hop_hits(split_metrics) -> str:
    if not isinstance(split_metrics, dict):
        return '-'
    per_hop = []
    for key in sorted(k for k in split_metrics if k.startswith('per_hop/') and k.endswith('hits@1')):
        hop = key.split('/')[1].replace('_hits@1', '')
        value = split_metrics.get(key)
        if value is None:
            continue
        per_hop.append(f'{hop}={value:.4f}')
    return ' '.join(per_hop) if per_hop else '-'


def print_summary(rows: Sequence[Dict[str, object]], split: str) -> None:
    metric_names = ['hits@1', 'hits@3', 'hits@5', 'hits@10', 'hits@20', 'mrr']
    headers = [
        'label',
        'status',
        *(f'{split}.{metric}' for metric in metric_names),
        f'{split}.per_hop_h@1',
    ]
    table: List[List[str]] = [headers]
    for row in rows:
        metrics = row.get('metrics', {}) or {}
        split_metrics = metrics.get(split, {}) if isinstance(metrics, dict) else {}
        if not isinstance(split_metrics, dict):
            split_metrics = {}
        table.append([
            str(row['label']),
            str(row['status']),
            *(format_metric(split_metrics.get(f'rollout_{metric}')) for metric in metric_names),
            format_per_hop_hits(split_metrics),
        ])
    widths = [max(len(entry[i]) for entry in table) for i in range(len(headers))]
    for index, entries in enumerate(table):
        line = '  '.join(entry.ljust(widths[i]) for i, entry in enumerate(entries))
        print(line)
        if index == 0:
            print('  '.join('-' * width for width in widths))


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    # config = load_shell_config(config_path)
    # base_cli = config_to_cli_args(config)
    pattern = args.pattern or DEFAULT_PATTERNS
    checkpoint_configs_and_paths = discover_checkpoints(args.target, pattern)
    # extra_args = list(args.extra_args)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = Path(args.log_dir) if args.log_dir else repo_root / 'batch_eval_logs' / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(repo_root) if not existing_pythonpath else f'{repo_root}:{existing_pythonpath}'

    num_experiments = 0
    for i, v in checkpoint_configs_and_paths.items():
        num_experiments += len(v.checkpoint_paths)
    cur_run_num_experiments = 0
    print(f"About two run these {num_experiments} experiments")
    for checkpoints_parent_path in checkpoint_configs_and_paths.keys():
        print("")
        
    for checkpoints_parent_path in checkpoint_configs_and_paths.keys():
        checkpoints_config: Dict[str, str] = checkpoint_configs_and_paths[checkpoints_parent_path].config
        all_config_paths: List[Path] = checkpoint_configs_and_paths[checkpoints_parent_path].checkpoint_paths

        for checkpoint in all_config_paths:
            extra_args = list(args.extra_args)

            base_cli = config_to_cli_args(checkpoints_config)

            label = sanitize_label(checkpoint)
            metrics_path = log_dir / f'{label}.metrics.json'
            stdout_path = log_dir / f'{label}.stdout.log'
            stderr_path = log_dir / f'{label}.stderr.log'
            cmd = [
                sys.executable,
                '-m',
                'src.experiments',
                '--inference',
                '--gpu',
                str(args.gpu),
                '--checkpoint_path',
                str(checkpoint),
                '--metrics_output_path',
                str(metrics_path),
                *base_cli,
                *extra_args,
            ]
            print(f'[{cur_run_num_experiments+1}/{num_experiments}] {checkpoint}')
            if args.dry_run:
                print('  ' + shlex.join(cmd))
                rows.append({
                    'label': label,
                    'checkpoint': str(checkpoint),
                    'status': 'dry-run',
                    'command': cmd,
                })
                continue

            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)

            row: Dict[str, object] = {
                'label': label,
                'checkpoint': str(checkpoint),
                'command': cmd,
                'returncode': completed.returncode,
                'stdout_log': str(stdout_path),
                'stderr_log': str(stderr_path),
                'metrics_path': str(metrics_path),
            }
            if completed.returncode == 0 and metrics_path.is_file():
                row['status'] = 'ok'
                row['metrics'] = json.loads(metrics_path.read_text())
            else:
                row['status'] = 'failed'
                row['error'] = summarize_error(completed.stderr, completed.stdout)
                print(f"  failed: {row['error']}")
                if args.fail_fast:
                    rows.append(row)
                    break
            rows.append(row)
            cur_run_num_experiments += 1

    print_summary(rows, args.split)

    failed_rows = [row for row in rows if row.get('status') == 'failed']
    if failed_rows:
        print()
        print(f'Failures: {len(failed_rows)}')
        for row in failed_rows:
            print(f"- {row['label']}: {row.get('error')}")

    if args.output_json:
        output_json = Path(args.output_json)
        ensure_parent(output_json)
        output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f'JSON summary saved to {output_json}')

    if args.output_csv:
        output_csv = Path(args.output_csv)
        ensure_parent(output_csv)
        flat_rows: List[Dict[str, object]] = []
        header_names = {
            'label', 'checkpoint', 'status', 'returncode',
            'stdout_log', 'stderr_log', 'metrics_path', 'error'
        }
        for row in rows:
            flat_row = {key: row.get(key, '') for key in header_names}
            metrics = row.get('metrics')
            if isinstance(metrics, dict):
                flattened: Dict[str, object] = {}
                flatten_metrics('', metrics, flattened)
                flat_row.update(flattened)
                header_names.update(flattened.keys())
            flat_rows.append(flat_row)
        ordered_headers = sorted(header_names)
        with output_csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered_headers)
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f'CSV summary saved to {output_csv}')

    print(f'Artifacts saved under {log_dir}')
    return 1 if failed_rows else 0


if __name__ == '__main__':
    raise SystemExit(main())
