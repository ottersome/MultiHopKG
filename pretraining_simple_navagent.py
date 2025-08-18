import os
import time
from typing import List, Tuple

import debugpy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
import pickle
from sklearn.model_selection import train_test_split

from multihopkg import data_utils
from multihopkg.logging import setup_logger
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.utils.data_structures import Triplet_Int, Triplet_Str
from multihopkg.run_configs.pretraining_simple_navagent import arguments

class RandomWalkDataset(Dataset):
    PATH_SAMPLING_BATCH_SIZE = 128

    def __init__(
        self,
        entity_embeddings: nn.Embedding,
        relation_embeddings: nn.Embedding,
        paths: List[List[int]],
    ):
        self.paths = paths
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]

def load_path_data(
    path_mquake_data: str,
    path_cache_dir: str,
    amount_of_paths: int,
    path_generation_batch_size: int,
    n_hops: int,
    path_num_beams: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Load the path data from the mquake data.
    Args:
        path_mquake_data (str): Path to the mquake data directory.
        path_cache_dir (str): Path to the cache directory.
    Returns:
        Tuple[List[List[int]], List[List[int]], List[List[int]]]: A tuple of three lists, each containing a list of paths.
    """

    train_cache_path = os.path.join(path_cache_dir, "train_paths.pkl")
    dev_cache_path = os.path.join(path_cache_dir, "dev_paths.pkl")
    test_cache_path = os.path.join(path_cache_dir, "test_paths.pkl")

    cache_complete = all([
        os.path.exists(train_cache_path),
        os.path.exists(dev_cache_path),
        os.path.exists(test_cache_path),
    ])
    
    if cache_complete:
        with open(train_cache_path, "rb") as f:
            train_paths = pickle.load(f)
        with open(dev_cache_path, "rb") as f:
            dev_paths = pickle.load(f)
        with open(test_cache_path, "rb") as f:
            test_paths = pickle.load(f)

        return train_paths, dev_paths, test_paths
    else:
        # Load mquake data
        path_triplets = os.path.join(path_mquake_data, "expNpruned_triplets.txt")
        id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(path_mquake_data)
        triplets_int = data_utils.load_triples_hrt(path_triplets, ent2id, rel2id, has_headers=True)

        generated_paths = data_utils.generate_paths_for_nav_training(
            triplets_ints = triplets_int,
            amount_of_paths = amount_of_paths,
            generation_batch_size = path_generation_batch_size,
            num_hops = n_hops,
            num_beams = path_num_beams,
        )

        # Train-Dev-Test Split
        temp_paths, test_paths = train_test_split(generated_paths, test_size=0.1)
        train_paths, dev_paths = train_test_split(temp_paths, test_size=0.11111)

        # Ensure that the dir is created
        os.makedirs(path_cache_dir, exist_ok=True)

        # Cache the data
        with open(train_cache_path, "wb") as f:
            pickle.dump(train_paths, f)
        with open(dev_cache_path, "wb") as f:
            pickle.dump(dev_paths, f)
        with open(test_cache_path, "wb") as f:
            pickle.dump(test_paths, f)

        return train_paths, dev_paths, test_paths

def train_loop(
    nav_agent: ContinuousPolicyGradient,
    train_paths: List[List[int]],
    entity_embeddings: nn.Embedding,
    relation_embeddings: nn.Embedding,
    # -- Training Params -- #
    epochs: int, 
    # steps_in_episode: int,
    batch_size: int,
) -> nn.Module:
    # So that we can later ask our model to from point A to point B
    train_dataset = RandomWalkDataset(
        entity_embeddings = entity_embeddings,
        relation_embeddings = relation_embeddings,
        paths = train_paths,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training Loop
    nav_agent.train()
    for epoch in range(epochs):
        logger.debug(f"Starting epoch {epoch}")
        for batch_idx, batch in enumerate(train_dataloader):
            logger.debug(f"Going through batch {batch_idx}")


            # TODO: Implement the training loop
            pass

    return nav_agent

def main():
    args = arguments()
    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()

    # Load the Embeddings
    entity_embeddings = nn.Embedding.from_pretrained(
        torch.from_numpy(np.load(os.path.join(args.path_embeddings_dir, "entity_embedding.npy")))
    )
    relation_embeddings = nn.Embedding.from_pretrained(
        torch.from_numpy(np.load(os.path.join(args.path_embeddings_dir, "relation_embedding.npy")))
    )

    dim_action_relation = relation_embeddings.embedding_dim

    # Load mquake data
    train_paths, dev_paths, test_paths = load_path_data(
        path_mquake_data = args.path_mquake_data,
        path_cache_dir = args.path_generation_cache,
        amount_of_paths = args.amount_of_paths,
        path_generation_batch_size = args.path_batch_size,
        n_hops = args.path_n_hops,
        path_num_beams = args.path_num_beams,
    )

    # Create the Navigator
    nav_agent = ContinuousPolicyGradient(
        beta = args.rl_beta,
        gamma = args.rl_gamma,
        dim_action = dim_action_relation,
        dim_hidden = args.rl_dim_hidden,
        dim_observation = args.env_dim_observation,
    )

    ########################################
    # Training
    ########################################

    trained_model = train_loop(
        nav_agent,
        train_paths,
        entity_embeddings,
        relation_embeddings,
        epochs = args.epochs,
        # steps_in_episode = args.num_rollout_steps,
        batch_size = args.batch_size,
    )

if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_SIMPLE_MAIN__")
    main()
