"""Shared raw-KG preparation utilities for MQuAKE-ST SA and MA.

The QA variants share one observed KG. ``raw.kb`` represents that complete
KG, while the named train/dev/test files retain their source partition
semantics. Link-prediction training explicitly builds its transductive input
from their union in ``prepare_lp_training_data.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def parse_literal_list(value: object, column_name: str) -> list[object]:
    """Parse a list-valued CSV cell and reject malformed multi-answer data."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f'{column_name} must be a list literal or list, got {type(value)}')
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f'{column_name} must parse to a list, got {type(parsed)}')
    return parsed


def source_question_numbers(qa_df: pd.DataFrame) -> pd.Series:
    """Preserve source IDs, with a one-indexed CSV-row fallback."""
    if 'Question-Number' in qa_df.columns:
        question_numbers = qa_df['Question-Number'].copy()
        missing = question_numbers.isna()
        question_numbers.loc[missing] = qa_df.index[missing] + 1
        return question_numbers
    return pd.Series(qa_df.index + 1, index=qa_df.index, name='Question-Number')


def require_columns(qa_df: pd.DataFrame, required: list[str], qa_path: Path) -> None:
    missing = [column for column in required if column not in qa_df.columns]
    if missing:
        raise ValueError(f'Missing required columns from {qa_path}: {missing}')


def build_mquake_st_kg_files(triplets_path: str | Path, output_raw_kb_path: str | Path) -> Path:
    """Write one canonical MQuAKE-ST KG layout shared by SA and MA.

    The raw source files are Tail-Relation-Head. Salesforce consumes
    Head-Tail-Relation, represented below as A, B, r.
    """
    triplets_path = Path(triplets_path)
    output_raw_kb_path = Path(output_raw_kb_path)
    kg_path = triplets_path.resolve().parent
    output_path = output_raw_kb_path.resolve().parent
    output_path.mkdir(parents=True, exist_ok=True)

    def read_partition(name: str) -> pd.DataFrame:
        return pd.read_csv(
            kg_path / name, names=['A', 'r', 'B'], header=None, sep=r'\s+')

    train_triplets = read_partition('train.txt')
    valid_triplets = read_partition('valid.txt')
    test_triplets = read_partition('test.txt')
    train = train_triplets[['A', 'B', 'r']]
    valid = valid_triplets[['A', 'B', 'r']]
    test = test_triplets[['A', 'B', 'r']]
    full_kg = pd.concat([train, valid, test], ignore_index=True).drop_duplicates()

    full_kg.to_csv(output_raw_kb_path, sep='\t', index=False, header=False)
    train.to_csv(output_path / 'train.triples', sep='\t', index=False, header=False)
    valid.to_csv(output_path / 'dev.triples', sep='\t', index=False, header=False)
    test.to_csv(output_path / 'test.triples', sep='\t', index=False, header=False)

    all_edges = pd.concat([train_triplets, valid_triplets, test_triplets], ignore_index=True)
    all_edges = all_edges.drop(columns=['r']).drop_duplicates().reset_index(drop=True)
    zero_columns = pd.DataFrame(
        np.zeros((len(all_edges), 2), dtype=int), columns=['A_zero', 'B_zero'])
    pagerank_input = pd.concat(
        [all_edges['A'], zero_columns['A_zero'], all_edges['B'], zero_columns['B_zero']],
        axis=1,
    )
    pagerank_input.to_csv(
        output_path / 'pgrk_input_mquake_st.csv', sep=',', index=False, header=False)

    print(f'Wrote full raw KG ({len(full_kg)} unique triples) to {output_raw_kb_path}')
    print(f'Wrote split KG files to {output_path}')
    return output_path
