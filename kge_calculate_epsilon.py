"""
Calculates the ideal epsilon separation to distiguish embedding space between entities.
"""
import argparse
import os

import pandas
import numpy as np
import torch

from multihopkg.run_configs import alpha
from multihopkg.run_configs.common import overload_parse_defaults_with_yaml
from multihopkg.utils.setup import set_seeds
import multihopkg.data_utils as data_utils

from multihopkg.exogenous.sun_models import KGEModel

def initial_setup() -> argparse.Namespace:
    args = alpha.get_args()
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    set_seeds(args.seed)

    assert isinstance(args, argparse.Namespace)

    return args

def main():
    args = initial_setup()
    ########################################
    # Get the data
    ########################################

    # Load the dictionaries
    id2ent, ent2id, id2rel, rel2id =  data_utils.load_dictionaries(args.data_dir)

    entity_embeddings = np.load(os.path.join(args.trained_model_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(args.trained_model_path, "relation_embedding.npy"))
    checkpoint = torch.load(os.path.join(args.trained_model_path , "checkpoint"))
    kge_model = KGEModel.from_pretrained(
        model_name=args.model,
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=args.gamma,
        state_dict=checkpoint["model_state_dict"]
    ).to(args.device)

    if args.node_data_path:
        # Load the node data
        node_df = pandas.read_csv(args.node_data_path).fillna('')
        entity2title = node_df.set_index(args.node_data_key)['Title'].to_dict()

    # --------------------------------------
    # Evaluate Epsilon value through comparison between all entities embeddings

    if args.nav_epsilon_metric == 'l2':
        p = 2
    elif args.nav_epsilon_metric == 'l1':
        p = 1
    elif args.nav_epsilon_metric == 'deg' and args.model == 'pRotatE':
        p = -1
        conversion_constant = 180/torch.pi # convert radians to degrees
    else:
        raise ValueError(f"Unknown epsilon metric: {args.nav_epsilon_metric}")
    
    max_val = 0
    min_val = 1e8
    head_max_id = -1
    tail_max_id = -1
    head_min_id = -1
    tail_min_id = -1
    for head_idx in range(kge_model.nentity - 1):
        tail_idx = torch.arange(head_idx+1, kge_model.nentity).int().cuda()

        head = torch.index_select(
            kge_model.entity_embedding, 
            dim=0, 
            index=torch.Tensor([head_idx]).int().cuda()
        ).unsqueeze(1)
        
        tail = torch.index_select(
            kge_model.entity_embedding, 
            dim=0, 
            index=tail_idx
        ).unsqueeze(1)

        diff = kge_model.absolute_difference(
            head,
            tail,
        )

        if p == -1:
            diff_avg = (conversion_constant*diff).mean(dim=-1).squeeze(1)
        else:
            diff_avg = (diff).norm(p=p, dim=-1).squeeze(1)
        
        # Update max and min values
        if diff_avg.numel() > 0:  # Ensure diff_avg is not empty
            if diff_avg.max() > max_val:
                max_val = diff_avg.max().item()
                head_max_id = head_idx
                tail_max_id = tail_idx[diff_avg.argmax()].item()
            if diff_avg.min() < min_val:
                min_val = diff_avg.min().item()
                head_min_id = head_idx
                tail_min_id = tail_idx[diff_avg.argmin()].item()

    print(f"Episilon Value: {min_val}")

    print(f"Max Epsilon Value: {max_val}")
    print(f"Min Epsilon Value: {min_val}")

    if args.node_data_path:
        print(f"Max Epsilon Pair: {entity2title[id2ent[head_max_id]]} - {entity2title[id2ent[tail_max_id]]}")
        print(f"Min Epsilon Pair: {entity2title[id2ent[head_min_id]]} - {entity2title[id2ent[tail_min_id]]}")
    else:
        print(f"Max Epsilon Pair: {id2ent[head_max_id]} - {id2ent[tail_max_id]}")
        print(f"Min Epsilon Pair: {id2ent[head_min_id]} - {id2ent[tail_min_id]}")

if __name__ == "__main__":
    main(),