#!/usr/bin/env python3

"""Train and evaluate n-hop configs, then render a Table-6-style Hits@1 report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from batch_eval import ensure_parent, load_shell_config, summarize_error


ColumnKey = Tuple[str, str, int]
AggregateKey = Tuple[str, str, str, str, int]


@dataclass(frozen=True)
class ReportIdentity:
    """Labels used to place one config in the final report."""

    dataset: str
    answer_type: str
    model: str
    model_group: str
    hop: int


@dataclass
class EvaluationResult:
    """Outcome and artifacts for one config/seed training run."""

    config: str
    seed: int
    identity: ReportIdentity
    status: str
    command: List[str]
    metrics_path: str
    stdout_log: str
    stderr_log: str
    hits_at_1: Optional[float] = None
    returncode: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class AggregateCell:
    """Mean/std summary for one model and report column."""

    mean: float
    std: float
    count: int


def parse_seed_list(value: str) -> List[int]:
    """Parse comma-separated non-negative seeds while preserving declared order."""
    values: List[int] = []
    for raw_value in value.split(','):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        parsed = int(raw_value)
        if parsed < 0:
            raise argparse.ArgumentTypeError('seeds must be non-negative integers')
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError('at least one seed is required')
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    extra_args: List[str] = []
    if '--' in raw_argv:
        separator = raw_argv.index('--')
        extra_args = raw_argv[separator + 1:]
        raw_argv = raw_argv[:separator]

    parser = argparse.ArgumentParser(
        description=(
            'Run per-hop training configs and aggregate test Hits@1 into a '
            'dataset/answer-type/hop table.'
        ),
        epilog=(
            'Optional report-only config variables:\n'
            '  report_dataset="Kinship"\n'
            '  report_answer_type="Single"\n'
            '  report_model="MINERVA"\n'
            '  report_model_group="Adapted Path-Based Models"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'config_target',
        help='A per-hop .sh config or a directory searched recursively for .sh configs.',
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU id passed to the launcher.')
    parser.add_argument(
        '--seeds',
        type=parse_seed_list,
        default=None,
        help='Comma-separated seed override. By default, each config uses its own seed.',
    )
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument(
        '--launcher',
        default='experiment-rs-nlp.sh',
        help='Training launcher, relative to the repository root by default.',
    )
    parser.add_argument(
        '--log_dir',
        default='metrics_logs/',
        help='Artifact directory (default: per_hop_eval_logs/<timestamp>).',
    )
    parser.add_argument('--output_json', default='', help='Optional result JSON path.')
    parser.add_argument('--output_csv', default='', help='Optional aggregate CSV path.')
    parser.add_argument('--output_table', default='', help='Optional Markdown table path.')
    parser.add_argument('--precision', type=int, default=3)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--fail_fast', action='store_true')
    args = parser.parse_args(raw_argv)
    args.extra_args = extra_args
    if args.precision < 0:
        parser.error('--precision must be non-negative')
    return args


def discover_configs(target: Path) -> List[Path]:
    """Return deterministic per-hop config paths from a file or directory."""
    if target.is_file():
        if target.suffix != '.sh':
            raise ValueError('Config file must have a .sh suffix: {}'.format(target))
        return [target.resolve()]
    if not target.is_dir():
        raise FileNotFoundError('Config target does not exist: {}'.format(target))
    configs = sorted(path.resolve() for path in target.rglob('*.sh'))
    if not configs:
        raise FileNotFoundError('No .sh configs found under {}'.format(target))
    return configs


def normalize_dataset_label(data_dir: str) -> str:
    """Provide readable defaults while allowing explicit report_dataset overrides."""
    name = Path(data_dir.rstrip('/')).name
    normalized = re.sub(r'[^a-z0-9]+', '', name.lower())
    if normalized in {'kinship', 'kinshiphinton', 'kinshiphintonlatest'}:
        return 'Kinship'
    if normalized.startswith('mquakest'):
        return 'MQuAKE-ST'
    if normalized.startswith('metaqa'):
        return 'MetaQA'
    return name


def infer_answer_type(config: Mapping[str, str]) -> str:
    """Infer a display label when report_answer_type is not declared."""
    raw = '{} {}'.format(config.get('data_dir', ''), config.get('raw_QAData_path', '')).lower()
    multi_markers = ('multi', '_ma', '-ma')
    return 'Multi' if any(marker in raw for marker in multi_markers) else 'Single'


def config_identity(config_path: Path, config: Mapping[str, str]) -> ReportIdentity:
    """Read required hop metadata and optional report labels from one config."""
    try:
        hop = int(config['train_hop'])
    except KeyError as exc:
        raise ValueError('{} is missing required train_hop'.format(config_path)) from exc
    except ValueError as exc:
        raise ValueError('{} has an invalid train_hop'.format(config_path)) from exc
    if hop <= 0:
        raise ValueError('{} must set train_hop to a positive integer'.format(config_path))

    dataset = config.get('report_dataset') or normalize_dataset_label(config.get('data_dir', ''))
    answer_type = config.get('report_answer_type') or infer_answer_type(config)
    model = config.get('report_model') or config.get('model') or config_path.stem
    model_group = config.get('report_model_group') or 'Models'
    return ReportIdentity(dataset, answer_type, model, model_group, hop)


def config_seeds(config_path: Path, config: Mapping[str, str], override: Optional[List[int]]) -> List[int]:
    if override is not None:
        return override
    try:
        seed = int(config['seed'])
    except KeyError as exc:
        raise ValueError(f'{config_path} is missing seed; provide it or use --seeds') from exc
    except ValueError as exc:
        raise ValueError(f'{config_path} has an invalid seed') from exc
    if seed < 0:
        raise ValueError(f'{config_path} must use a non-negative seed')
    return [seed]


def sanitize_label(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('_')


def metric_key(hop: int) -> str:
    return 'per_hop/{}hop_hits@1'.format(hop)


def extract_hits_at_1(metrics: Mapping[str, object], split: str, hop: int) -> float:
    split_metrics = metrics.get(split)
    if not isinstance(split_metrics, dict):
        raise KeyError("Metrics do not contain split '{}'".format(split))
    value = split_metrics.get(metric_key(hop))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise KeyError("Metrics do not contain numeric '{}.{}'".format(split, metric_key(hop)))
    return float(value)


def build_command(
    launcher: Path,
    config_path: Path,
    gpu: int,
    seed: int,
    hop: int,
    metrics_path: Path,
    extra_args: Sequence[str],
) -> List[str]:
    return [
        'bash',
        str(launcher),
        str(config_path),
        '--train',
        str(gpu),
        '--seed',
        str(seed),
        '--train_hop',
        str(hop),
        '--num_rollout_steps',
        str(hop),
        '--evaluate_per_hop',
        '--eval_hops',
        str(hop),
        '--metrics_output_path',
        str(metrics_path),
        *extra_args,
    ]


def run_config(
    config_path: Path,
    identity: ReportIdentity,
    seed: int,
    args: argparse.Namespace,
    repo_root: Path,
    launcher: Path,
    log_dir: Path,
) -> EvaluationResult:
    label = sanitize_label(f'{config_path.stem}-{identity.hop}hop-seed{seed}')
    metrics_path = log_dir / f'{label}.metrics.json'
    stdout_path = log_dir / f'{label}.stdout.log'
    stderr_path = log_dir / f'{label}.stderr.log'
    command = build_command(
        launcher, config_path, args.gpu, seed, identity.hop, metrics_path, args.extra_args)
    result = EvaluationResult(
        config=str(config_path),
        seed=seed,
        identity=identity,
        status='dry-run' if args.dry_run else 'pending',
        command=command,
        metrics_path=str(metrics_path),
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
    )
    print('  {}'.format(shlex.join(command)))
    if args.dry_run:
        return result

    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (
        str(repo_root) if not existing_pythonpath else '{}:{}'.format(repo_root, existing_pythonpath)
    )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    result.returncode = completed.returncode
    if completed.returncode != 0:
        result.status = 'failed'
        result.error = summarize_error(completed.stderr, completed.stdout)
        return result
    if not metrics_path.is_file():
        result.status = 'failed'
        result.error = 'Training completed without writing {}'.format(metrics_path)
        return result

    try:
        metrics = json.loads(metrics_path.read_text())
        result.hits_at_1 = extract_hits_at_1(metrics, args.split, identity.hop)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        result.status = 'failed'
        result.error = str(exc)
        return result
    result.status = 'ok'
    return result


def aggregate_results(results: Iterable[EvaluationResult]) -> Dict[AggregateKey, AggregateCell]:
    values: DefaultDict[AggregateKey, List[float]] = defaultdict(list)
    for result in results:
        if result.status != 'ok' or result.hits_at_1 is None:
            continue
        identity = result.identity
        key = (
            identity.model_group,
            identity.model,
            identity.dataset,
            identity.answer_type,
            identity.hop,
        )
        values[key].append(result.hits_at_1)

    aggregates: Dict[AggregateKey, AggregateCell] = {}
    for key, cell_values in values.items():
        aggregates[key] = AggregateCell(
            mean=statistics.fmean(cell_values),
            std=statistics.stdev(cell_values) if len(cell_values) > 1 else 0.0,
            count=len(cell_values),
        )
    return aggregates


def report_columns(results: Iterable[EvaluationResult]) -> List[ColumnKey]:
    columns = {
        (result.identity.dataset, result.identity.answer_type, result.identity.hop)
        for result in results
    }
    answer_order = {'Single': 0, 'Multi': 1}
    return sorted(columns, key=lambda item: (item[0], answer_order.get(item[1], 99), item[1], item[2]))


def format_cell(cell: AggregateCell, precision: int, bold: bool) -> str:
    value = '{mean:.{p}f} ± {std:.{p}f}'.format(mean=cell.mean, std=cell.std, p=precision)
    return '**{}**'.format(value) if bold else value


def render_markdown_table(
    results: Sequence[EvaluationResult],
    aggregates: Mapping[AggregateKey, AggregateCell],
    precision: int,
) -> str:
    columns = report_columns(results)
    model_keys = sorted({
        (result.identity.model_group, result.identity.model)
        for result in results
    })
    if not columns or not model_keys:
        return 'No per-hop results available.'

    best_by_group_column: Dict[Tuple[str, ColumnKey], float] = {}
    for group, _model in model_keys:
        for column in columns:
            dataset, answer_type, hop = column
            aggregate_key = (group, _model, dataset, answer_type, hop)
            cell = aggregates.get(aggregate_key)
            if cell is None:
                continue
            best_key = (group, column)
            best_by_group_column[best_key] = max(
                cell.mean, best_by_group_column.get(best_key, float('-inf')))

    headers = ['Model group', 'Model'] + [
        '{} / {} / {}-hop'.format(dataset, answer_type, hop)
        for dataset, answer_type, hop in columns
    ]
    lines = [
        '| {} |'.format(' | '.join(headers)),
        '| {} |'.format(' | '.join(['---'] * len(headers))),
    ]
    for group, model in model_keys:
        row = [group, model]
        for column in columns:
            dataset, answer_type, hop = column
            key = (group, model, dataset, answer_type, hop)
            cell = aggregates.get(key)
            if cell is None:
                row.append('—')
                continue
            is_best = cell.mean == best_by_group_column[(group, column)]
            row.append(format_cell(cell, precision, is_best))
        lines.append('| {} |'.format(' | '.join(row)))
    return '\n'.join(lines)


def write_aggregate_csv(
    path: Path,
    aggregates: Mapping[AggregateKey, AggregateCell],
) -> None:
    ensure_parent(path)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['model_group', 'model', 'dataset', 'answer_type', 'hop', 'mean', 'std', 'runs'],
        )
        writer.writeheader()
        for (group, model, dataset, answer_type, hop), cell in sorted(aggregates.items()):
            writer.writerow({
                'model_group': group,
                'model': model,
                'dataset': dataset,
                'answer_type': answer_type,
                'hop': hop,
                'mean': cell.mean,
                'std': cell.std,
                'runs': cell.count,
            })


def serializable_result(result: EvaluationResult) -> Dict[str, object]:
    payload = asdict(result)
    payload['identity'] = asdict(result.identity)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    config_target = Path(args.config_target)
    if not config_target.is_absolute():
        config_target = repo_root / config_target
    launcher = Path(args.launcher)
    if not launcher.is_absolute():
        launcher = repo_root / launcher
    if not launcher.is_file():
        raise FileNotFoundError('Launcher does not exist: {}'.format(launcher))

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = Path(args.log_dir) if args.log_dir else repo_root / 'per_hop_eval_logs' / timestamp
    if not log_dir.is_absolute():
        log_dir = repo_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, ReportIdentity, int]] = []
    for config_path in discover_configs(config_target):
        config = load_shell_config(config_path)
        identity = config_identity(config_path, config)
        for seed in config_seeds(config_path, config, args.seeds):
            jobs.append((config_path, identity, seed))

    print('Running {} per-hop training/evaluation job(s).'.format(len(jobs)))
    results: List[EvaluationResult] = []
    for index, (config_path, identity, seed) in enumerate(jobs, start=1):
        print('[{}/{}] {} ({}-hop, seed={})'.format(
            index, len(jobs), config_path, identity.hop, seed))
        result = run_config(config_path, identity, seed, args, repo_root, launcher, log_dir)
        results.append(result)
        if result.status == 'failed':
            print('  failed: {}'.format(result.error))
            if args.fail_fast:
                break

    aggregates = aggregate_results(results)
    table = render_markdown_table(results, aggregates, args.precision)
    print('\n{}'.format(table))

    table_path = Path(args.output_table) if args.output_table else log_dir / '{}_hits_at_1.md'.format(args.split)
    ensure_parent(table_path)
    table_path.write_text('{}\n'.format(table))

    json_path = Path(args.output_json) if args.output_json else log_dir / 'results.json'
    ensure_parent(json_path)
    json_path.write_text(json.dumps(
        [serializable_result(result) for result in results], indent=2, sort_keys=True))

    csv_path = Path(args.output_csv) if args.output_csv else log_dir / 'aggregates.csv'
    write_aggregate_csv(csv_path, aggregates)

    print('Table saved to {}'.format(table_path))
    print('Run details saved to {}'.format(json_path))
    print('Aggregate CSV saved to {}'.format(csv_path))
    failed = [result for result in results if result.status == 'failed']
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
