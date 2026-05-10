#!/usr/bin/env python3
"""
Render a compact per-hop Hits@1 table from a CSV of run metrics.

Expected CSV columns:
  model,dataset,answer_type,kinship_2hop,kinship_3hop,mquake_single_2hop,...

More commonly, export W&B summaries and provide columns:
  model,dataset,answer_type,hop,mean,std

This script accepts the second normalized format and prints a LaTeX table body
matching the Kinship / MQuAKE-ST layout used in the paper draft.
"""

import argparse
import math
from typing import Dict, Tuple

import pandas as pd


ColumnKey = Tuple[str, str, int]


DISPLAY_COLUMNS = [
    ("Kinship", "Single", 2),
    ("Kinship", "Single", 3),
    ("MQuAKE-ST", "Single", 2),
    ("MQuAKE-ST", "Single", 3),
    ("MQuAKE-ST", "Single", 4),
    ("MQuAKE-ST", "Multi", 2),
    ("MQuAKE-ST", "Multi", 3),
    ("MQuAKE-ST", "Multi", 4),
]


def normalize_dataset(value: str) -> str:
    lowered = str(value).strip().lower()
    if "kinship" in lowered:
        return "Kinship"
    if "mquake" in lowered:
        return "MQuAKE-ST"
    return str(value).strip()


def normalize_answer_type(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered.startswith("multi"):
        return "Multi"
    return "Single"


def format_cell(mean, std=None) -> str:
    if pd.isna(mean):
        return "--"
    if std is None or pd.isna(std):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} $\\pm$ {float(std):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-hop Hits@1 table cells.")
    parser.add_argument("csv_path", help="CSV with columns model,dataset,answer_type,hop,mean[,std]")
    parser.add_argument("--model-order", default="EmbedKGQA,TransferNet,ReaRev,MINERVA,MultiHopKG,SQUIRE",
                        help="comma-separated row order")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    required = {"model", "dataset", "answer_type", "hop", "mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].map(normalize_dataset)
    df["answer_type"] = df["answer_type"].map(normalize_answer_type)
    df["hop"] = df["hop"].astype(int)

    values: Dict[Tuple[str, ColumnKey], Tuple[float, float]] = {}
    for _, row in df.iterrows():
        key = (
            str(row["model"]),
            (str(row["dataset"]), str(row["answer_type"]), int(row["hop"])),
        )
        values[key] = (row["mean"], row["std"] if "std" in row else math.nan)

    for model in [item.strip() for item in args.model_order.split(",") if item.strip()]:
        cells = []
        for col_key in DISPLAY_COLUMNS:
            mean, std = values.get((model, col_key), (math.nan, math.nan))
            cells.append(format_cell(mean, std))
        print(f"{model} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
