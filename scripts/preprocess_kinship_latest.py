# How to call 
# python -m scripts.preprocess_kinship_latest

from networkx import pagerank
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import numpy as np

# TODO
# - [ ]  Get: 
    # - [ ] `train.triples`
    # - [ ] `dev.triples`
    # - [ ] `entity2id.txt`
    # - [ ] `relation2id.txt`
    # - [ ] `./datasets/data_preprocessed/mquake/mquake_qa_2hop.csv`
    #

def main():
    ap = ArgumentParser()
    ap.add_argument("--triplets_path", default="./raw_data/KinshipHintonLatest/kg/orig/triplets.txt", help="Location of data")
    ap.add_argument("--output_raw_kb_path", default="./data/KinshipHintonLatest/raw.kb", help="Location of data")
    ap.add_argument("--qa_ds_path", default="./raw_data/KinshipHintonLatest/qa/kinship_hinton_qa_nhop.csv", help="Location of data")

    args = ap.parse_args()

    triplets_path = args.triplets_path
    kg_path = Path(triplets_path).resolve().parent 
    output_path = Path(args.output_raw_kb_path).resolve().parent 
    print(f"Will look for train.txt and similar files in {kg_path}")
    
    train_path = kg_path.joinpath("train.txt")
    valid_path = kg_path.joinpath("valid.txt")
    test_path = kg_path.joinpath("test.txt")

    # Create raw.kb from all triplets
    # In kinship the format is Tail-Relation-Head
    train_triplets = pd.read_csv(train_path, names=["A", "r", "B",], header=None, sep=r"\s+")
    valid_triplets = pd.read_csv(valid_path, names=["A", "r", "B",], header=None, sep=r"\s+")
    test_triplets = pd.read_csv(test_path, names=["A", "r", "B",], header=None, sep=r"\s+")

    raw_csv = train_triplets[["A","B", "r"]]
    train_triples_reorg = train_triplets[["A", "B", "r"]]
    valid_triples_reorg = valid_triplets[["A", "B", "r"]]
    test_triples_reorg = test_triplets[["A", "B", "r"]]

    path_output_train = output_path.joinpath("train.triples")
    path_output_dev = output_path.joinpath("dev.triples") # They use 'dev' language in this algorithm
    path_output_test = output_path.joinpath("test.triples")

    raw_csv.to_csv(args.output_raw_kb_path, sep="\t", index=False, header=False)
    train_triples_reorg.to_csv(path_output_train, sep="\t", index=False, header=False)
    valid_triples_reorg.to_csv(path_output_dev, sep="\t", index=False, header=False)
    test_triples_reorg.to_csv(path_output_test, sep="\t", index=False, header=False)
    print("Dump raw.kb into: ", Path(args.output_raw_kb_path).resolve())

    print(f"train_triplets head: {train_triplets.head()}")
    # Prepping page rank
    all_edges = pd.concat([
        train_triplets,
        valid_triplets,
        test_triplets,
        ], axis=0, ignore_index=True)
    all_edges = all_edges\
        .drop(columns=["r"])\
        .drop_duplicates()\
        .reset_index(drop=True)
    print(f"Data dataframe is of {len(all_edges.columns)} column and with {len(all_edges)} rows")
    zero_df = np.zeros((len(all_edges), len(all_edges.columns)), dtype=int)
    full_zero = pd.DataFrame(zero_df, columns=pd.Index(["A", "B"]))
    print(f"Edges within Zero_frames: {full_zero.columns} and num of rows: {len(full_zero)}")
    print(f"Edges within all_edges: {all_edges.columns} and num of rows: {len(all_edges)}")
    pagerank_df = pd.concat(
        [
            all_edges.loc[:, "A"],
            full_zero.loc[:, "A"],
            all_edges.loc[:, "B"],
            full_zero.loc[:, "B"],
        ],
        axis=1,
    )
    print(f"Resulting page rank csv has {len(pagerank_df.columns)} columns and {len(pagerank_df)} rows")
    pagerank_df.to_csv(
        output_path.joinpath("pgrk_input_kinshiphinton_latest.csv"),
        sep=",",
        index=False,
        header=False,
    )

    ## NOw we focus on the QA data so that it has the following columsn
    # Question-Number, Question, Answer, Hops, Source-Entity, Answer-Entity, Paths, SplitLabel
    qa_ds_path = Path(args.qa_ds_path)
    if not qa_ds_path.exists():
        raise RuntimeError("Could not find the Q&A data")
    qa_df = pd.read_csv(qa_ds_path)
    qna_df = pd.DataFrame()
    qna_df["Question"] = qa_df["Question"]
    qna_df["Answer"] = qa_df["Answer"]
    qna_df["Hops"] = qa_df["Hops"]
    qna_df["Source-Entity"] = qa_df["Source-Entity"]
    qna_df["Answer-Entity"] = qa_df["Answer-Entity"]
    qna_df["Paths"] = qa_df["Paths"]
    qna_df["SplitLabel"] = qa_df["SplitLabel"]
    # TODO Add paraphrased questionshere
    # Seems like it was split specifically for salesforce input pipeline. 

    output_path_qna_df = output_path.joinpath("kinship_qa_nhop.csv")
    qna_df.to_csv(
        output_path_qna_df,
        sep=",",
        index=True,
    )
    print(f"Saved qna datset to {output_path_qna_df}")


if __name__ == "__main__":

    REPO_ROOT = Path(__file__).resolve().parent.parent
    # os.chdir(REPO_ROOT)
    main()
