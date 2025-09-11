#!/usr/bin/env python3
"""
Convert an xSV (something-separated values) file of triples into a
4-column edge list with symmetric edges and no relation labels.

Input assumptions:
- Each row contains at least 3 columns: subject, relation, object, ...
- We ignore the relation (2nd column) and any columns after the 3rd.

Output:
- A 4-column CSV-like file (configurable delimiter) where columns 1 and 3 are
  the node IDs, and columns 2 and 4 are empty (placeholders).
- For each triple (s, r, o), we output two rows: s,,o, and o,,s (symmetric).
- Duplicate edges are deduplicated by default.

Usage:
  python scripts/xsv_to_symmetric_edges.py input.xsv -o output.csv \
      --in-delimiter $'\t' --out-delimiter ',' --has-header

Exit codes:
- 0 on success
- 2 on I/O or parse errors
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert triples to symmetric 4-column edge list (cols 2 & 4 empty)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Path to input xSV file (triples)")
    p.add_argument(
        "-o", "--output", type=Path, default=None, help="Output file path (default: stdout)"
    )
    p.add_argument(
        "--in-delimiter",
        default=",",
        help="Input delimiter (xSV). For tab, use $'\t' in most shells.",
    )
    p.add_argument(
        "--out-delimiter",
        default=",",
        help="Output delimiter for the 4-column file.",
    )
    p.add_argument(
        "--has-header",
        action="store_true",
        help="Skip the first row of the input as a header",
    )
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate edges (by default edges are deduplicated)",
    )
    return p.parse_args()


def load_pairs(
    path: Path, in_delim: str, has_header: bool, dedupe: bool
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=in_delim, skipinitialspace=True)
        if has_header:
            next(reader, None)

        for idx, row in enumerate(reader, start=1 if not has_header else 2):
            if not row:
                continue
            if len(row) < 3:
                print(
                    f"ERROR: Row {idx} has fewer than 3 columns: {row}",
                    file=sys.stderr,
                )
                raise ValueError("Invalid row: fewer than 3 columns")

            s = str(row[0]).strip()
            o = str(row[2]).strip()
            if s == "" or o == "":
                # Skip incomplete nodes
                continue

            # Add (s,o) and (o,s) to ensure symmetry
            for a, b in ((s, o), (o, s)):
                if dedupe:
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                pairs.append((a, b))

    return pairs


def write_edges(
    pairs: list[tuple[str, str]], out_path: Path | None, out_delim: str
) -> None:
    out_f = None
    try:
        if out_path is None:
            out = sys.stdout
        else:
            out_f = out_path.open("w", encoding="utf-8", newline="")
            out = out_f

        writer = csv.writer(out, delimiter=out_delim, lineterminator="\n")
        empty = ""
        for u, v in pairs:
            writer.writerow([u, empty, v, empty])
    finally:
        if out_f is not None:
            out_f.close()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        return 2

    try:
        pairs = load_pairs(
            args.input, args.in_delimiter, args.has_header, dedupe=not args.no_dedupe
        )
    except Exception as e:
        print(f"ERROR: Failed to read input: {e}", file=sys.stderr)
        return 2

    try:
        write_edges(pairs, args.output, args.out_delimiter)
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

