#!/usr/bin/env python3
"""
Check bidirectional (double) edges in a 4-column CSV.

Assumptions:
- Columns are: col1, col2, col3, col4. Only col1 and col3 matter (nodes).
- For every edge a->b, there must be a corresponding b->a somewhere in the file.

Usage:
  python scripts/check_double_edges.py path/to/file.csv

Options:
  --delimiter, -d       CSV delimiter (default: ",")
  --has-header          Skip the first row as a header

Output:
- Prints "TRUE" if all edges are double (symmetric), otherwise "FALSE".
- Exits with code 0 on TRUE, 1 on FALSE, 2 on read/parse errors.
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check if a CSV has double edges: for every a->b there is b->a.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the 4-column CSV file",
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        default=",",
        help="CSV delimiter",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat first row as header and skip it",
    )
    return parser.parse_args()


def load_edges(csv_path: Path, delimiter: str, has_header: bool) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)

        if has_header:
            next(reader, None)

        for idx, row in enumerate(reader, start=1 if not has_header else 2):
            if not row:
                continue  # skip empty lines
            if len(row) < 3:
                # Not enough columns to form an edge
                print(
                    f"ERROR: Row {idx} has fewer than 3 columns: {row}",
                    file=sys.stderr,
                )
                raise ValueError("Invalid row: fewer than 3 columns")
            u = str(row[0]).strip()
            v = str(row[2]).strip()
            if u == "" and v == "":
                continue
            edges.add((u, v))

    return edges


def is_symmetric(edges: set[tuple[str, str]]) -> tuple[bool, tuple[str, str] | None]:
    for (u, v) in edges:
        if (v, u) not in edges:
            return False, (u, v)
    return True, None


def main() -> int:
    args = parse_args()

    if not args.csv_file.exists():
        print(f"ERROR: File not found: {args.csv_file}", file=sys.stderr)
        return 2

    try:
        edges = load_edges(args.csv_file, args.delimiter, args.has_header)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}", file=sys.stderr)
        return 2

    ok, offending = is_symmetric(edges)
    if ok:
        print("TRUE")
        return 0
    else:
        # Print FALSE and first offending edge for quick debugging
        if offending is not None:
            u, v = offending
            print("FALSE")
            print(f"Missing reverse edge: {v}->{u}", file=sys.stderr)
        else:
            print("FALSE")
        return 1


if __name__ == "__main__":
    sys.exit(main())

