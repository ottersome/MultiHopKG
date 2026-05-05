# How to call 
# python -m scripts.preprocess_metaqa_st

import ast
from typing import List
from networkx import pagerank
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# TODO
# - [ ]  Get: 
    # - [ ] `train.triples`
    # - [ ] `dev.triples`
    # - [ ] `entity2id.txt`
    # - [ ] `relation2id.txt`
    # - [ ] `./datasets/data_preprocessed/metaqa/metaqa_qa_2hop.csv`
    #

def normalize_pd_element(element: str):
    if "[" in element: 
        element = ast.literal_eval(element)
    if isinstance(element, list):
        new_element = []
        for elem in element:
            new_element.append(elem.strip().replace(" ", "_"))
        return new_element
    elif isinstance(element, str):
        new_element = element.strip().replace(" ", "_")
        return new_element
    else:
        raise ValueError("Element is not a list or string")

            

def main():
    ap = ArgumentParser()
    ap.add_argument("--triplets_path", default="./raw_data/MetaQA/kb/kb.txt", help="Location of data")
    ap.add_argument("--output_raw_kb_path", default="./data/MetaQA/raw.kb", help="Location of data")
    ap.add_argument("--qa_ds_path", default="./raw_data/MetaQA/metaqa_nhop.csv", help="Location of data")
    # TODO: MultiAnswer

    args = ap.parse_args()

    triplets_path = args.triplets_path
    kg_path = Path(triplets_path).resolve().parent 
    output_path = Path(args.output_raw_kb_path).resolve().parent 
    print(f"Will look for train.txt and similar files in {kg_path}")
    Path.mkdir(output_path, parents=True, exist_ok=True)
    

    # Create raw.kb from all triplets
    # In MetaQA the format is Tail-Relation-Head
    all_triplets = pd.read_csv(triplets_path, names=["A", "r", "B",], header=None, sep=r"|")
    all_triplets["A"] = all_triplets["A"].apply(lambda x: x.strip().replace(" ", "_"))
    all_triplets["A"] = all_triplets["r"].apply(lambda x: x.strip().replace(" ", "_"))
    all_triplets["B"] = all_triplets["B"].apply(lambda x: x.strip().replace(" ", "_"))
    train_triplets, train_leftover = train_test_split(all_triplets, test_size=0.2, random_state=42)
    valid_triplets, test_triplets = train_test_split(train_leftover, test_size=0.5, random_state=42)

    # Separate them into their appropriate splits.


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
        output_path.joinpath("pgrk_input_metaqa.csv"),
        sep=",",
        index=False,
        header=False,
    )

    # Grab the QnA data
    data_path = Path("raw_data/MetaQA")
    hops = ["1hop", "2hop", "3hop"]
    qna_df = []
    for hop in hops:
        file_path = data_path.joinpath(f"metaqa_{hop}.csv")
        df = pd.read_csv(file_path, sep=",")
        print(f"Read {file_path} with columns:\n{df.columns}")
        qna_df.append(df)
    qna_df = (
        pd.concat(qna_df, axis=0, ignore_index=True)
        .reset_index(drop=True)
        .drop(columns=["Question-Number"])
    )
    qna_df["Source-Entity"] = qna_df["Source-Entity"].apply(normalize_pd_element)
    qna_df["Answer"] = qna_df["Answer"].apply(normalize_pd_element)
    qna_df["Answer-Entity"] = qna_df["Answer-Entity"].apply(normalize_pd_element)
    qna_df.to_csv(output_path.joinpath("metaqa_qa_nhop.csv"), sep=",", index=True)

if __name__ == "__main__":

    REPO_ROOT = Path(__file__).resolve().parent.parent
    # os.chdir(REPO_ROOT)
    main()
