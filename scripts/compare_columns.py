#!/usr/bin/env python3
"""
Compare two delimited text files (e.g., CSV/TSV) for column equivalence.

By default, checks that both files contain the same set of column names,
ignoring order and case/whitespace if flags are provided. Optionally, the
comparison can respect duplicate column names (multiset comparison).

Exit codes:
  0: columns are equivalent per the chosen mode
  1: columns are NOT equivalent
  2: usage or runtime error (e.g., file not found)
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from collections import Counter
from typing import Iterable, List, Optional, Tuple


def read_header(
    path: str,
    *,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    sample_bytes: int = 4096,
) -> List[str]:
    """Read the first row as header from a delimited file.

    Attempts to auto-detect the delimiter if not provided using csv.Sniffer.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding=encoding, newline="") as f:
        # Read a sample for sniffer if needed, then reset
        sample = f.read(sample_bytes)
        f.seek(0)

        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(sample) if sample else csv.get_dialect("excel")
            except csv.Error:
                # Fallback to comma if sniffing fails
                dialect = csv.get_dialect("excel")
            reader = csv.reader(f, dialect=dialect)
        else:
            reader = csv.reader(f, delimiter=delimiter)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"File appears to be empty: {path}")

        return header


def normalize_columns(
    columns: Iterable[str], *, case_insensitive: bool, strip: bool
) -> List[str]:
    out: List[str] = []
    for col in columns:
        if strip:
            col = col.strip()
        if case_insensitive:
            col = col.lower()
        out.append(col)
    return out


def compare_columns(
    cols_a: Iterable[str],
    cols_b: Iterable[str],
    *,
    respect_duplicates: bool = False,
) -> Tuple[bool, str]:
    """Compare two column name lists.

    Returns (equal, details).
    """
    if respect_duplicates:
        ca, cb = Counter(cols_a), Counter(cols_b)
        if ca == cb:
            return True, "Columns are equivalent (including duplicate counts)."
        # Build a helpful diff
        lines = ["Columns differ (considering duplicate counts):"]
        all_keys = sorted(set(ca) | set(cb))
        for k in all_keys:
            a_n, b_n = ca.get(k, 0), cb.get(k, 0)
            if a_n != b_n:
                lines.append(f"  {k!r}: fileA={a_n}, fileB={b_n}")
        return False, "\n".join(lines)
    else:
        sa, sb = set(cols_a), set(cols_b)
        if sa == sb:
            return True, "Columns are equivalent (ignoring order)."
        only_a = sorted(sa - sb)
        only_b = sorted(sb - sa)
        lines = ["Columns differ (set comparison):"]
        if only_a:
            lines.append("  Only in fileA: " + ", ".join(repr(x) for x in only_a))
        if only_b:
            lines.append("  Only in fileB: " + ", ".join(repr(x) for x in only_b))
        return False, "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Test if two delimited files have equivalent columns (ignoring order by default)."
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
        "-i",
        "--case-insensitive",
        action="store_true",
        help="Compare column names case-insensitively",
    )
    p.add_argument(
        "-s",
        "--strip",
        action="store_true",
        help="Strip leading/trailing whitespace from column names before comparing",
    )
    p.add_argument(
        "--respect-duplicates",
        action="store_true",
        help="Respect duplicate column names (multiset comparison)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        header_a = read_header(args.file_a, delimiter=args.delimiter, encoding=args.encoding)
        header_b = read_header(args.file_b, delimiter=args.delimiter, encoding=args.encoding)

        norm_a = normalize_columns(
            header_a, case_insensitive=args.case_insensitive, strip=args.strip
        )
        norm_b = normalize_columns(
            header_b, case_insensitive=args.case_insensitive, strip=args.strip
        )

        equal, details = compare_columns(
            norm_a, norm_b, respect_duplicates=args.respect_duplicates
        )
        print(details)
        if equal:
            # Show a quick summary of the normalized columns for visibility
            try:
                example = ", ".join(sorted(set(norm_a)))
                print(f"Columns: {example}")
            except Exception:
                pass
        return 0 if equal else 1
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

