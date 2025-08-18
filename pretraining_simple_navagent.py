import argparse
import math
import os
import pickle
import time
from doctest import debug
from math import log
from typing import Any, List

import debugpy
import numpy as np
import pandas as pd
import torch
from pyarrow import dictionary
from sympy import degree, num_digits
from torch.utils.data import DataLoader, Dataset

from multihopkg import data_utils
from multihopkg.graph.classical import (
    are_nodes_connected,
    find_heads_with_connections,
    generate_csr_representation,
    random_walk,
    sample_paths_given_csr,
)
from multihopkg.logging import setup_logger
from multihopkg.utils.data_structures import Triplet_Int, Triplet_Str


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_mquake_data", type=str, default="./data/mquake/")
    parser.add_argument("--amount_of_paths", type=int, default=1000)
    parser.add_argument(
        "--path_generation_cache", type=str, default="./cache/random_walks.pkl"
    )
    parser.add_argument("--path_batch_size", default=4)
    parser.add_argument("--path_max_len", default=5)
    parser.add_argument("--path_num_beams", default=4)


    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()


class RandomWalkDataset(Dataset):
    PATH_SAMPLING_BATCH_SIZE = 128

    def __init__(
        self,
        triplets_str: List[Triplet_Str],
        triplets_int: List[Triplet_Int],
        num_nodes: int,
        num_edges: int,
        amount_of_paths: int,
        path_generation_cache: str,
        batch_size: int,
        max_path_len: int,
        num_beams: int,
    ):
        self.amount_of_paths = amount_of_paths
        self.paths = []
        self.triplets_str = triplets_str
        self.triplets_int = triplets_int
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.max_path_len = max_path_len
        self.num_beams = num_beams

        logger.info(f"Loaded {len(self.triplets_str)} triplets_str")

        # Check if we go the paths generated already
        if os.path.exists(path_generation_cache):
            self.paths = self._load_cached_paths(path_generation_cache)
        else:
            self.paths = self._generate_and_cache_paths(
                triplets_int = triplets_int,
                num_nodes = num_nodes,
                path_generation_cache = path_generation_cache,
                batch_size = self.batch_size,
            )

    # TODO: Change this return type to be more specific
    def _load_cached_paths(self, path_generation_cache: str) -> List[Any]:
        with open(path_generation_cache, "rb") as f:
            return pickle.load(f)

    # TODO: Change this return type to be more specific
    def _generate_and_cache_paths(
        self,
        triplets_int: List[Triplet_Int],
        num_nodes: int,
        path_generation_cache: str,
        batch_size: int,
    ) -> List[Any]:
        os.makedirs(os.path.dirname(path_generation_cache), exist_ok=True)

        # Generate the CSR representation
        indptr, tail_indices, rel_ids = generate_csr_representation(triplets_int, num_nodes)

        # Get list of one degree heads
        degree_heads = list(find_heads_with_connections(triplets_int, tail_indices))

        num_batches = math.ceil(self.amount_of_paths / self.PATH_SAMPLING_BATCH_SIZE )

        generated_paths = []

        for b in range(num_batches):
            random_idx = torch.randint(0, len(degree_heads), (batch_size,))
            random_starting_heads = torch.tensor(degree_heads)[random_idx]

            paths = sample_paths_given_csr(
                torch.tensor(indptr),
                torch.tensor(tail_indices),
                torch.tensor(rel_ids),
                random_starting_heads,
                self.max_path_len,
                self.num_beams,
            )
            generated_paths.extend(paths)

        # Now we do caching
        with open(path_generation_cache, "wb") as f:
            pickle.dump(generated_paths, f)

        return generated_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


def _triplet_str_to_int(triplet: pd.Series, ent2id: dict, rel2id: dict) -> Triplet_Int:
    return (
        ent2id[triplet.iloc[0]],
        rel2id[triplet.iloc[1]],
        ent2id[triplet.iloc[2]],
    )


def main():
    args = arguments()
    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()

    # Load mquake data
    path_triplets = os.path.join(args.path_mquake_data, "expNpruned_triplets.txt")
    triplets_str = []
    triplets_int = []
    id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(args.path_mquake_data)
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    logger.info(
        f"Converted {len(triplets_int)} triplets to ints in {t1-t0:.4f} seconds"
    )
    num_nodes = len(id2ent)
    num_edges = len(triplets_int)
    logger.info(f"Read {num_nodes} nodes and {num_edges} edges")

    # Now we simply create a dataset of random walks
    # So that we can later ask our model to from point A to point B
    dataset = RandomWalkDataset(
        triplets_str = triplets_str,
        triplets_int = triplets_int,
        num_nodes = num_nodes,
        num_edges = num_edges,
        amount_of_paths = args.amount_of_paths,
        path_generation_cache = args.path_generation_cache,
        batch_size = args.path_batch_size,
        max_path_len = args.path_max_len,
        num_beams = args.path_num_beams,
    )

    # Create DataLoader for training
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # print(f"Created dataset with {len(dataset)} random walk paths")
    # print(f"DataLoader ready with {len(dataloader)} batches")


if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_SIMPLE_MAIN__")
    main()
