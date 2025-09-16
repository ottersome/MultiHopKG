#!/usr/bin/env python3
"""
Compare two delimited text files for equivalence w.r.t. columns under a fixed
assumption about column order in each file.

IMPORTANT ASSUMPTIONS (as requested):
- The input XSV files have NO header row.
- File A is in the column order a-b-c (i.e., columns [0,1,2]).
- File B is in the column order a-c-b (i.e., columns [0,2,1]).
- We consider the files equivalent if, after reordering columns in File B from
  a-c-b to a-b-c, every row in File B matches the corresponding row in File A.

Notes:
- This script enforces exactly 3 columns per row in both files, matching the
  a/b/c assumption above.
- It compares rows as sets (order-insensitive). Duplicate rows are ignored for
  equality purposes because sets are used, as requested.

Exit codes:
  0: files are equivalent (per above assumptions)
  1: files are NOT equivalent
  2: usage or runtime error (e.g., file not found, wrong number of columns)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable, Iterator, List, Optional, Tuple


def iter_rows(
    path: str,
    *,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    sample_bytes: int = 4096,
) -> Iterator[List[str]]:
    """Yield data rows from a delimited file (no header expected).

    Attempts to auto-detect the delimiter if not provided using csv.Sniffer.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    f = open(path, "r", encoding=encoding, newline="")
    try:
        sample = f.read(sample_bytes)
        f.seek(0)

        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(sample) if sample else csv.get_dialect("excel")
            except csv.Error:
                dialect = csv.get_dialect("excel")
            reader = csv.reader(f, dialect=dialect)
        else:
            reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            yield row
    finally:
        f.close()


def normalize_fields(fields: Iterable[str], *, strip: bool) -> List[str]:
    out: List[str] = []
    for val in fields:
        out.append(val.strip() if strip else val)
    return out


def compare_files_abc_acb(
    file_a: str,
    file_b: str,
    *,
    delimiter: Optional[str],
    encoding: str,
    strip: bool,
) -> Tuple[bool, str]:
    """Compare two files assuming:
    - file A columns: a-b-c => indices [0,1,2]
    - file B columns: a-c-b => indices [0,2,1]
    Rows must match after reordering B as [0,2,1].
    """
    iter_a = iter_rows(file_a, delimiter=delimiter, encoding=encoding)
    iter_b = iter_rows(file_b, delimiter=delimiter, encoding=encoding)

    set_a = set()
    set_b = set()

    # Collect all rows from A
    for idx_a, row_a in enumerate(iter_a, start=1):
        if len(row_a) != 3:
            return (
                False,
                f"File A, row {idx_a}: expected exactly 3 columns; got {len(row_a)}",
            )
        row_a_n = tuple(normalize_fields(row_a, strip=strip))
        set_a.add(row_a_n)

    # Collect all rows from B (after mapping a-c-b -> a-b-c)
    for idx_b, row_b in enumerate(iter_b, start=1):
        if len(row_b) != 3:
            return (
                False,
                f"File B, row {idx_b}: expected exactly 3 columns; got {len(row_b)}",
            )
        row_b_n = tuple(normalize_fields([row_b[0], row_b[2], row_b[1]], strip=strip))
        set_b.add(row_b_n)

    if set_a == set_b:
        return True, "Files are equivalent under A=abc and B=acb mapping (row order ignored)."

    only_a = set_a - set_b
    only_b = set_b - set_a

    # Prepare a concise diff
    def sample_rows(s, n=10):
        out = []
        for i, r in enumerate(s):
            if i >= n:
                break
            out.append(str(list(r)))
        return out

    lines = [
        "Files differ under A=abc and B=acb mapping (row order ignored):",
        f"  Unique rows in A (showing up to 10):",
    ]
    if only_a:
        lines.extend(["    " + x for x in sample_rows(only_a)])
    else:
        lines.append("    (none)")
    lines.append("  Unique rows in B (showing up to 10):")
    if only_b:
        lines.extend(["    " + x for x in sample_rows(only_b)])
    else:
        lines.append("    (none)")

    return False, "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare two XSV files assuming A=abc and B=acb (no headers)."
        )
    )
    p.add_argument("file_a", help="Path to first file")
    p.add_argument("file_b", help="Path to second file")
    p.add_argument(
        "-d",
        "--delimiter",
        help="Field delimiter (auto-detect if omitted)",
    )
    p.add_argument(
        "-e",
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)",
    )
    p.add_argument(
        "-s",
        "--strip",
        action="store_true",
        help="Strip leading/trailing whitespace from fields before comparing",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        equal, details = compare_files_abc_acb(
            args.file_a,
            args.file_b,
            delimiter=args.delimiter,
            encoding=args.encoding,
            strip=args.strip,
        )
        print(details)
        return 0 if equal else 1
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
