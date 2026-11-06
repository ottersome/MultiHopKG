"""Build the transductive full-KG triple file used by LP substrate training.

The QA agents navigate the observed KG, not a link-prediction benchmark split.
ConvE is therefore trained on the union of all locally supplied KG triples.
This deliberately includes the conventional train/dev/test triple partitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TRIPLE_FILES = ('raw.kb', 'train.triples', 'dev.triples', 'test.triples')


def read_triples(path: Path) -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f'{path}:{line_number}: expected three fields, found {len(fields)}')
        triples.add(tuple(fields))
    return triples


def build_full_graph(data_dir: Path, output_path: Path) -> int:
    triples: set[tuple[str, str, str]] = set()
    for file_name in TRIPLE_FILES:
        input_path = data_dir / file_name
        if not input_path.is_file():
            raise FileNotFoundError(f'Missing KG triple file: {input_path}')
        triples.update(read_triples(input_path))

    ordered_triples = sorted(triples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(f'{head}\t{tail}\t{relation}\n' for head, tail, relation in ordered_triples))
    return len(ordered_triples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, type=Path)
    parser.add_argument('--output_path', type=Path, default=None)
    args = parser.parse_args()
    output_path = args.output_path or args.data_dir / 'lp_train.triples'
    count = build_full_graph(args.data_dir, output_path)
    print(f'Wrote {count} unique full-KG triples to {output_path}')


if __name__ == '__main__':
    main()
