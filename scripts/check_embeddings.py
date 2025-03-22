import argparse
import torch
import numpy as np

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc_entities_embs", type=str, default="./models/protatE_FBWikiV4/entity_embedding.npy")
    ap.add_argument("--loc_relations_embs", type=str, default="./models/protatE_FBWikiV4/relation_embedding.npy")
    ap.add_argument("--loc_entities_dict", type=str, default="./data/FBWikiV4/entities.dict")
    ap.add_argument("--loc_relations_dict", type=str, default="./data/FBWikiV4/relations.dict")

    return ap.parse_args()

def main(args: argparse.Namespace):
    loc_entities_embs: str = args.loc_entities_embs
    loc_relations_embs: str = args.loc_relations_embs
    loc_entities_dict: str = args.loc_entities_dict
    loc_relations_dict: str = args.loc_relations_dict

    entities_embs = np.load(loc_entities_embs)
    relations_embs = np.load(loc_relations_embs)

    # Check if we have the same amount of embeddings as we do in the dicts
    with open(loc_entities_dict, "r") as file:
        # Read each line
        entities_list = file.readlines()
        entities_list = [line.strip().split("\t")[1] for line in entities_list]
        # In reality this list acts as a dict, indices being keys to both embedding and RDF values

    # Check if we have the same amount of embeddings as we do in the dicts
    with open(loc_relations_dict, "r") as file:
        # Read each line
        relations_list = file.readlines()
        relations_list = [line.strip().split("\t")[1] for line in relations_list]
        # In reality this list acts as a dict, indices being keys to both embedding and RDF values

    # Assert that the embeddings are of same length as the dicts
    assert (
        len(entities_list) == entities_embs.shape[0]
    ), f"We have {len(entities_list)} entities in the dict, but {entities_embs.shape[0]} embeddings"
    assert (
        len(relations_list) == relations_embs.shape[0]
    ), f"We have {len(relations_list)} relations in the dict, but {relations_embs.shape[0]} embeddings"

    print(f"Num of entity embeddings = {len(entities_list)} and {entities_embs.shape[0]}")
    print(f"Num of relation embeddings = {len(relations_list)} and {relations_embs.shape[0]}")


if __name__ == "__main__":
    args = argsies()
    main(args)
