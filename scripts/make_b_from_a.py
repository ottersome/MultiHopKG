#!/usr/bin/env python3
"""
Create file B from file A by reordering columns under fixed assumptions.

ASSUMPTIONS (matching compare script):
- The input XSV has NO header row.
- File A column order is a-b-c (indices [0,1,2]).
- We output file B in order a-c-b (indices [0,2,1]).

Notes:
- Enforces exactly 3 columns per row.
- Supports delimiter auto-detection or forcing via --delimiter.

Exit codes:
  0: success
  2: usage or runtime error
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterator, List, Optional


def iter_rows(
    path: str,
    *,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    sample_bytes: int = 4096,
) -> tuple[Iterator[List[str]], str]:
    """Yield data rows from a delimited file (no header expected) and return delimiter used.

    If delimiter is None, attempts to auto-detect using csv.Sniffer.
    Returns (iterator, delimiter_used).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    f = open(path, "r", encoding=encoding, newline="")
    # We'll close f in a generator finalizer inside the iterator
    sample = f.read(sample_bytes)
    f.seek(0)

    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(sample) if sample else csv.get_dialect("excel")
            used_delim = getattr(dialect, "delimiter", ",")
        except csv.Error:
            dialect = csv.get_dialect("excel")
            used_delim = getattr(dialect, "delimiter", ",")
        reader = csv.reader(f, dialect=dialect)
    else:
        used_delim = delimiter
        reader = csv.reader(f, delimiter=delimiter)

    def _iter():
        try:
            for row in reader:
                yield row
        finally:
            f.close()

    return _iter(), used_delim


def write_rows(
    path: str,
    rows: Iterator[List[str]],
    *,
    delimiter: str,
    encoding: str = "utf-8",
) -> None:
    with open(path, "w", encoding=encoding, newline="") as out:
        writer = csv.writer(out, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read A (abc) and write B (acb), no headers."
    )
    p.add_argument("file_a", help="Path to input file A (abc)")
    p.add_argument("file_b", help="Path to output file B (acb)")
    p.add_argument(
        "-d", "--delimiter", help="Field delimiter (auto-detect if omitted)"
    )
    p.add_argument(
        "-e", "--encoding", default="utf-8", help="File encoding (default: utf-8)"
    )
    p.add_argument(
        "-s",
        "--strip",
        action="store_true",
        help="Strip leading/trailing whitespace from fields before writing",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        rows_a, used_delim = iter_rows(
            args.file_a, delimiter=args.delimiter, encoding=args.encoding
        )

        def mapped_rows() -> Iterator[List[str]]:
            for idx, row in enumerate(rows_a, start=1):
                if len(row) != 3:
                    raise ValueError(
                        f"Row {idx}: expected exactly 3 columns; got {len(row)}"
                    )
                a, b, c = row[0], row[1], row[2]
                if args.strip:
                    a, b, c = a.strip(), b.strip(), c.strip()
                # Output order a-c-b
                yield [a, c, b]

        write_rows(args.file_b, mapped_rows(), delimiter=used_delim, encoding=args.encoding)
        print(
            f"Wrote B (acb) to {args.file_b} using delimiter '{used_delim}' from A."
        )
        return 0
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

