# How to call:
# python -m scripts.preprocess_mquake_st_ma

from argparse import ArgumentParser
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def parse_literal_list(value, column_name):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{column_name} must be a list literal or list, got {type(value)}")
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{column_name} must parse to a list, got {type(parsed)}")
    return parsed


def build_kg_files(triplets_path, output_raw_kb_path):
    kg_path = Path(triplets_path).resolve().parent
    output_path = Path(output_raw_kb_path).resolve().parent
    print(f"Will look for train.txt and similar files in {kg_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = kg_path.joinpath("train.txt")
    valid_path = kg_path.joinpath("valid.txt")
    test_path = kg_path.joinpath("test.txt")

    # In MQuAKE-ST the source files are Tail-Relation-Head. The Salesforce code
    # expects Head Tail Relation columns, named A B r here.
    train_triplets = pd.read_csv(train_path, names=["A", "r", "B"], header=None, sep=r"\s+")
    valid_triplets = pd.read_csv(valid_path, names=["A", "r", "B"], header=None, sep=r"\s+")
    test_triplets = pd.read_csv(test_path, names=["A", "r", "B"], header=None, sep=r"\s+")

    all_triples_reorg = pd.concat(
        [train_triplets[["A", "B", "r"]], valid_triplets[["A", "B", "r"]], test_triplets[["A", "B", "r"]]],
        axis=0,
        ignore_index=True,
    ).drop_duplicates()
    raw_csv = all_triples_reorg
    train_triples_reorg = all_triples_reorg
    valid_triples_reorg = valid_triplets[["A", "B", "r"]]
    test_triples_reorg = test_triplets[["A", "B", "r"]]

    raw_csv.to_csv(output_raw_kb_path, sep="\t", index=False, header=False)
    train_triples_reorg.to_csv(output_path.joinpath("train.triples"), sep="\t", index=False, header=False)
    valid_triples_reorg.to_csv(output_path.joinpath("dev.triples"), sep="\t", index=False, header=False)
    test_triples_reorg.to_csv(output_path.joinpath("test.triples"), sep="\t", index=False, header=False)
    print("Dump raw.kb into: ", Path(output_raw_kb_path).resolve())

    all_edges = pd.concat([train_triplets, valid_triplets, test_triplets], axis=0, ignore_index=True)
    all_edges = all_edges.drop(columns=["r"]).drop_duplicates().reset_index(drop=True)
    zero_df = pd.DataFrame(np.zeros((len(all_edges), 2), dtype=int), columns=pd.Index(["A_zero", "B_zero"]))
    pagerank_df = pd.concat(
        [
            all_edges.loc[:, "A"],
            zero_df.loc[:, "A_zero"],
            all_edges.loc[:, "B"],
            zero_df.loc[:, "B_zero"],
        ],
        axis=1,
    )
    pagerank_df.to_csv(
        output_path.joinpath("pgrk_input_mquake_st.csv"),
        sep=",",
        index=False,
        header=False,
    )

    return output_path


def build_multi_answer_qa_file(qa_ds_path, output_path, output_qa_filename):
    qa_ds_path = Path(qa_ds_path)
    if not qa_ds_path.exists():
        raise RuntimeError(f"Could not find the Q&A data at {qa_ds_path}")

    qa_df = pd.read_csv(qa_ds_path)
    required_columns = ["Question", "Answer", "Hops", "Source-Entity", "Answer-Entity", "SplitLabel"]
    missing_columns = [col for col in required_columns if col not in qa_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns from {qa_ds_path}: {missing_columns}")

    qna_df = pd.DataFrame()
    if "Question-Number" in qa_df.columns:
        qna_df["Question-Number"] = qa_df["Question-Number"]
    qna_df["Question"] = qa_df["Question"]
    qna_df["Answer"] = qa_df["Answer"].apply(lambda value: repr(parse_literal_list(value, "Answer")))
    qna_df["Hops"] = qa_df["Hops"]
    qna_df["Source-Entity"] = qa_df["Source-Entity"]
    qna_df["Answer-Entity"] = qa_df["Answer-Entity"].apply(
        lambda value: repr(parse_literal_list(value, "Answer-Entity"))
    )

    # The multi-answer MQuAKE-ST files provide Path-Key relation chains rather
    # than concrete evidence-edge Paths. Keep Path-Key for traceability, but do
    # not fabricate a Paths column because the QA loader expects concrete
    # [head, relation, tail] triples there.
    if "Path-Key" in qa_df.columns:
        qna_df["Path-Key"] = qa_df["Path-Key"]
    if "Question-Paraphrased" in qa_df.columns:
        qna_df["Question-Paraphrased"] = qa_df["Question-Paraphrased"]
    if "Question-Disambiguated" in qa_df.columns:
        qna_df["Question-Disambiguated"] = qa_df["Question-Disambiguated"]
    qna_df["SplitLabel"] = qa_df["SplitLabel"]

    output_path_qna_df = Path(output_path).joinpath(output_qa_filename)
    qna_df.to_csv(output_path_qna_df, sep=",", index=False)
    print(f"Saved multi-answer qna dataset to {output_path_qna_df}")
    print(f"Rows: {len(qna_df)}")
    print(f"Max answers per row: {qna_df['Answer-Entity'].apply(lambda x: len(ast.literal_eval(x))).max()}")


def main():
    ap = ArgumentParser()
    ap.add_argument("--triplets_path", default="./raw_data/mquake_st/kg/triplets.txt", help="Location of KG data")
    ap.add_argument("--output_raw_kb_path", default="./data/mquake_st/raw.kb", help="Location of raw.kb output")
    ap.add_argument(
        "--qa_ds_path",
        default="./raw_data/mquake_st/qa/multi_answers/qa_nhop.csv",
        help="Location of multi-answer Q&A CSV",
    )
    ap.add_argument(
        "--output_qa_filename",
        default="mquake_st-ma-qa_nhop.csv",
        help="Output Q&A CSV filename under the data directory",
    )
    args = ap.parse_args()

    output_path = build_kg_files(args.triplets_path, args.output_raw_kb_path)
    build_multi_answer_qa_file(args.qa_ds_path, output_path, args.output_qa_filename)


if __name__ == "__main__":
    main()
