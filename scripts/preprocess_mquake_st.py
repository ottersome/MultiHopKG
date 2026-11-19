"""Prepare the single-answer MQuAKE-ST QA CSV and its shared KG files.

Run with ``python -m scripts.preprocess_mquake_st``.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from scripts.preprocess_mquake_st_common import (
    build_mquake_st_kg_files,
    require_columns,
    source_question_numbers,
)


def build_single_answer_qa_file(
    qa_ds_path: str | Path, output_path: str | Path, output_qa_filename: str,
) -> None:
    qa_path = Path(qa_ds_path)
    if not qa_path.is_file():
        raise FileNotFoundError(f'Single-answer Q&A data not found: {qa_path}')
    qa_df = pd.read_csv(qa_path)
    require_columns(
        qa_df,
        ['Question', 'Answer', 'Hops', 'Source-Entity', 'Answer-Entity', 'Paths', 'SplitLabel'],
        qa_path,
    )

    output_df = pd.DataFrame({
        'Question-Number': source_question_numbers(qa_df),
        'Question': qa_df['Question'],
        'Answer': qa_df['Answer'],
        'Hops': qa_df['Hops'],
        'Source-Entity': qa_df['Source-Entity'],
        'Answer-Entity': qa_df['Answer-Entity'],
        'Paths': qa_df['Paths'],
    })
    for column in ('Path-Key', 'Question-Paraphrased', 'Question-Disambiguated'):
        if column in qa_df.columns:
            output_df[column] = qa_df[column]
    output_df['SplitLabel'] = qa_df['SplitLabel']

    output_file = Path(output_path) / output_qa_filename
    output_df.to_csv(output_file, index=False)
    print(f'Saved {len(output_df)} single-answer QA rows to {output_file}')


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--triplets_path', default='./raw_data/mquake_st/kg/triplets.txt')
    parser.add_argument('--output_raw_kb_path', default='./data/mquake_st/raw.kb')
    parser.add_argument('--qa_ds_path', default='./raw_data/mquake_st/qa/single_answers/qa_nhop.csv')
    parser.add_argument('--output_qa_filename', default='mquake_st-qa_nhop.csv')
    args = parser.parse_args()

    output_path = build_mquake_st_kg_files(args.triplets_path, args.output_raw_kb_path)
    build_single_answer_qa_file(args.qa_ds_path, output_path, args.output_qa_filename)


if __name__ == '__main__':
    main()
