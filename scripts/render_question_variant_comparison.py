#!/usr/bin/env python3
"""Render side-by-side aggregate metrics for externally evaluated question text."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    ('Hits@1', 'rollout_hits@1'),
    ('Hits@3', 'rollout_hits@3'),
    ('Hits@10', 'rollout_hits@10'),
    ('MRR', 'rollout_mrr'),
    ('Relation F1', 'faithfulness/relation_f1'),
    ('Relation edit distance', 'faithfulness/relation_edit_distance'),
    ('Subgraph F1', 'faithfulness/subgraph_f1'),
    ('PED', 'faithfulness/ped'),
    ('Answer-correct Subgraph F1', 'faithfulness/answer_correct_subgraph_f1'),
    ('Answer-correct PED', 'faithfulness/answer_correct_ped'),
)


def parse_input(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split('=', 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected LABEL=PATH') from error
    if not label:
        raise argparse.ArgumentTypeError('comparison label must not be empty')
    return label, Path(path)


def metric_values(path: Path, metric_key: str) -> list[float]:
    rows = json.loads(path.read_text())
    values = []
    for row in rows:
        if row.get('status') != 'ok':
            continue
        metrics = row.get('metrics', {})
        test_metrics = metrics.get('test', {}) if isinstance(metrics, dict) else {}
        value = test_metrics.get(metric_key) if isinstance(test_metrics, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def format_values(values: list[float]) -> str:
    if not values:
        return '-'
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f'{mean:.4f} ± {std:.4f} (n={len(values)})'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Create a question-variant comparison table from batch_eval JSON summaries.')
    parser.add_argument('--output', required=True, type=Path, help='Markdown output path')
    parser.add_argument('--input', action='append', type=parse_input, required=True,
                        help='variant label and batch_eval JSON, as LABEL=PATH; may be repeated')
    args = parser.parse_args()

    variants: list[tuple[str, Path]] = args.input
    missing = [str(path) for _, path in variants if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing batch evaluation summary: {}'.format(', '.join(missing)))

    lines = [
        '# External question-variant comparison',
        '',
        'Each column evaluates the same annotated external test rows with a different question-text field.',
        '',
        '| Metric | {} |'.format(' | '.join(label for label, _ in variants)),
        '|---|{}|'.format('|'.join('---:' for _ in variants)),
    ]
    for metric_label, metric_key in METRICS:
        values = [format_values(metric_values(path, metric_key)) for _, path in variants]
        lines.append('| {} | {} |'.format(metric_label, ' | '.join(values)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n')
    print('Question-variant comparison saved to {}'.format(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
