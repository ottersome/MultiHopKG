import argparse
import debugpy
import os
import pandas as pd

import torch
from multihopkg import data_utils
from multihopkg.graph.classical import (
    find_heads_with_connections,
    generate_csr_representation,
    sample_paths_given_csr,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_mquake_data", type=str, default="./data/mquake/")
    parser.add_argument("--path_batch_size", default=4)
    parser.add_argument("--path_max_len", default=5)
    parser.add_argument("--path_entity_info", default="./data/mquake/entities_info.csv")
    parser.add_argument("--path_relation_info", default="./data/mquake/relations_info.csv")
    parser.add_argument("--path_output_dump", default="./temp/stringified_paths.csv")
    parser.add_argument("--path_num_beams", default=4)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--num_beams", default=3)

    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()

def main():
    args = arguments()
    if args.debug:
        print("🫷 Waiting for a debugpy connection")
        debugpy.listen(("0.0.0.0", 42023))
        print("Received debugpy connection. Continuing with execution.")

    # Load mquake data
    path_triplets = os.path.join(args.path_mquake_data, "expNpruned_triplets.txt")
    triplets_str = []
    triplets_int = []
    id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(args.path_mquake_data)
    with open(path_triplets, "r") as f:
        next(f)  # Skip the header row
        for line in f:
            head, rel, tail = line.strip().split()
            triplets_str.append((head, rel, tail))
            try:
                triplets_int.append(
                    (
                        ent2id[head],
                        rel2id[rel],
                        ent2id[tail],
                    )
                )
            except KeyError:
                breakpoint()

    # Read Entity and Relation backup info
    entity_info = pd.read_csv(args.path_entity_info, index_col=0)
    relation_info = pd.read_csv(args.path_relation_info, index_col=0)
    
    # Generate the csr representation
    num_nodes = len(ent2id)
    indptr, tail_indices, rel_ids = generate_csr_representation(triplets_int, num_nodes)

    # Get list of one degree heads
    degree_heads = list(find_heads_with_connections(triplets_int, tail_indices))
    random_idx = torch.randint(0, len(degree_heads), (args.batch_size,))
    random_starting_heads = torch.tensor(degree_heads)[random_idx]

    paths_ents, paths_rels = sample_paths_given_csr(
        torch.tensor(indptr),
        torch.tensor(tail_indices),
        torch.tensor(rel_ids),
        random_starting_heads,
        args.path_max_len,
        args.num_beams,
    )
    print(f"Done with sampling path_ents. We got shapes {paths_ents.shape[-1]} and {paths_rels.shape[-1]}")
    assert paths_ents.shape[-1] -1 == paths_rels.shape[-1], "Paths for entities and relations dont math in the amount of steps"
    num_elems = paths_rels.shape[0]
    num_steps = paths_rels.shape[-1]
    final_num_elems_per_row = num_steps*2 + 1

    steps = []

    for i in range(1, num_elems):
        row_ents = paths_ents[i,:].tolist()
        row_rels = paths_rels[i,:].tolist()

        new_row = [""] * final_num_elems_per_row
        new_row[0::2] = [
            entity_info.loc[id2ent[ent]]["Title"]
            if ent != -1 else "PAD" for ent in row_ents
        ]
        new_row[1::2] = [
            relation_info.loc[id2rel[rel]]["Title"]
            if rel != -1 else "PAD" for rel in row_rels
        ]
        steps.append(new_row)
    
    print(f"Saving the new data into {args.path_output_dump}")
    pd.DataFrame(steps).to_csv(args.path_output_dump)


if __name__ == "__main__":
    main()
