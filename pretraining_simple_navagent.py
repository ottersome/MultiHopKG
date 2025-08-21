import os
import time
from tracemalloc import start
from typing import Any, List, Tuple, Dict, Optional

import debugpy
from sympy import use
from sympy.printing.pycode import k
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.optim as optim
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from multihopkg import data_utils
from multihopkg.logging import setup_logger
from multihopkg.rl.graph_search.cpg import AttentionContinuousPolicyGradient, ContinuousPolicyGradient
from multihopkg.utils.data_structures import Triplet_Int, Triplet_Str
from multihopkg.run_configs.pretraining_simple_navagent import arguments

import wandb

from multihopkg.utils.setup import set_seeds

class RandomWalkDataset(Dataset):
    PATH_SAMPLING_BATCH_SIZE = 128

    def __init__(
        self,
        entity_embeddings: nn.Embedding,
        relation_embeddings: nn.Embedding,
        paths: List[List[int]],
        max_path_length: int = 10,
    ):
        tensor_paths = torch.tensor(paths, dtype=torch.int, device=entity_embeddings.weight.device)
        self.pad_id = entity_embeddings.weight.shape[0] - 1  # Replace -1 with the last entity
        logger.debug(f"Got Padding {torch.sum(tensor_paths == -1).item()} out of {tensor_paths.numel()}")
        tensor_paths[tensor_paths == -1] = self.pad_id

        self.paths = tensor_paths
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.max_path_length = max_path_length
        self.device = entity_embeddings.weight.device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        
        # Get start and target embeddings
        start_embeddings = self.entity_embeddings(path[0])
        # target_embeddings = self.entity_embeddings(path[-1])
        # Target_embedding should be the last non_padding, we should retrieve that
        path_mask_entities = path != self.pad_id
        last_non_pad_id = path_mask_entities.sum() - 1
        target_embeddings = self.entity_embeddings(path[last_non_pad_id])



        # start_embedding = entity_embeddings[0]
        # target_embedding = entity_embeddings[-1]
        
        # TODO: think if this is necessary
        # It all requires a bit more relations anyways

        # Get intermediate nodes and relations for training
        # intermediate_nodes = []
        # for i in range(len(path) - 1):
        #     # For each step in the path, we need:
        #     # 1. Current node embedding
        #     # 2. Target node embedding
        #     # 3. Relation embedding (if available)
        #     current_node = entity_embeddings[i]
        #     next_node = entity_embeddings[i + 1]
        #     
        #     # In a real scenario, you would have the relation between nodes
        #     # Here we'll use a placeholder or compute it
        #     # TODO: Change this for the acutal one ?
        #     # relation_vector = next_node - current_node  # Simple approximation
        #     
        #     intermediate_nodes.append((current_node, next_node, relation_vector))
        
        return {
            'start': start_embeddings,
            'target': target_embeddings,
            # 'path': entity_embeddings,
            # 'steps': intermediate_nodes,
            # 'path_length': len(path)
        }

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
        logger.info(f"Found cache {path_cache_dir}, loading from it.")
        with open(train_cache_path, "rb") as f:
            train_paths = pickle.load(f)
        with open(dev_cache_path, "rb") as f:
            dev_paths = pickle.load(f)
        with open(test_cache_path, "rb") as f:
            test_paths = pickle.load(f)

        return train_paths, dev_paths, test_paths
    else:
        # Load mquake data
        logger.info(f"No cache found on{path_cache_dir}. Generating...")
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

def compute_returns(rewards: List[torch.Tensor], gamma: float, masks: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    """
    Compute discounted returns for each timestep.
    
    Args:
        rewards: List of reward tensors for each timestep
        gamma: Discount factor
        masks: Optional list of boolean masks (1 for not done, 0 for done)
        
    Returns:
        List of return tensors for each timestep
    """
    returns = []
    R = torch.zeros_like(rewards[-1])
    
    for i in reversed(range(len(rewards))):
        if masks is not None:
            R = rewards[i] + gamma * R * masks[i]
        else:
            R = rewards[i] + gamma * R
        returns.insert(0, R)
        
    return returns

def wandb_report(metrics: Dict[str, Any], epoch: int, batch_idx: int):
    if wandb.run is None:
        return

    # Log metrics
    for k, v in metrics.items():
        wandb.log({f"train/{k}": v})


def train_loop(
    nav_agent: AttentionContinuousPolicyGradient,
    train_paths: List[List[int]],
    entity_embeddings: nn.Embedding,
    relation_embeddings: nn.Embedding,
    # -- Training Params -- #
    epochs: int, 
    steps_in_episode: int,
    batch_size: int,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "./checkpoints/navigator/",
    save_interval: int = 5,
    use_entropy_loss: bool = True,
) -> nn.Module:
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    nav_agent = nav_agent.to(device)
    entity_embeddings = entity_embeddings.to(device)
    relation_embeddings = relation_embeddings.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(nav_agent.parameters(), lr=learning_rate)
    
    # Create dataset
    train_dataset = RandomWalkDataset(
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        paths=train_paths,
        max_path_length=steps_in_episode + 1,  # +1 to include target
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: {
            'start': torch.stack([item['start'] for item in batch]).to(device),
            'target': torch.stack([item['target'] for item in batch]).to(device),
        }
    )

    # Training Loop
    nav_agent.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_reward = 0
        epoch_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            logger.debug(f"At epoch-batch {epoch}-{batch_idx}")
            start_states = batch["start"]
            target_states = batch["target"]
            batch_size = start_states.size(0)
            
            # Initialize episode variables
            actions = []
            log_probs = []
            rewards = []
            masks = []
            entropies = []
            
            # Current state starts at the beginning of the path
            current_states = start_states
            
            # Run episode for steps_in_episode steps
            step_distances = []
            mus, sigmas = [], []
            # step_i_distances = {f"step_{i}_distance": None for i in range(steps_in_episode) }
            action_magnitude = []
            for step in range(steps_in_episode):
                actions_step, log_probs_step, entropy_step, mu, sigma = nav_agent(
                    current_states, target_states
                )
                
                # Store action, log_prob, and entropy
                actions.append(actions_step)
                log_probs.append(log_probs_step)
                entropies.append(entropy_step)
                
                # Apply action to get next state (simulate movement in embedding space)
                next_states = current_states + actions_step
                # next_states = actions_step
                
                # Calculate reward based on distance to target
                # distances = torch.norm(next_states - target_states, dim=-1)
                distances = torch.linalg.vector_norm(next_states - target_states, dim=-1)
                _debug_avg_distance = distances.mean().item()
                _action_magnitude = torch.linalg.vector_norm(actions_step, dim=-1).mean()
                action_magnitude.append(_action_magnitude)
                # step_i_distances[f"step_{step}_distance"] = _debug_avg_distance
                step_distances.append(_debug_avg_distance)
                step_rewards = 1.0 / (1.0 + distances)  # Higher reward for closer distance
                
                # Add bonus reward for reaching target
                target_reached = distances < 0.06 # TODO: Find something better than this. This is too arbitrary.
                if epoch == 1:
                    debugpy.breakpoint()
                # step_rewards = step_rewards + 2.0 * target_reached.float()
                
                # Store rewards
                rewards.append(step_rewards)
                
                # Check if done (reached target or max steps)
                dones = target_reached | (step == steps_in_episode - 1)
                masks.append(~dones)  # Store inverted dones as masks (1 = not done, 0 = done)
                
                # Update current state
                current_states = next_states
                
                # Break if all episodes in batch are done
                if dones.all():
                    break
                
                # For episodes that are done, reset their states to the final state
                # to avoid meaningless gradient updates
                mus.append(mu.mean().item())
                sigmas.append(sigma.mean().item())
                if dones.any():
                    # import pdb
                    print(f"Wow. We got here. At step {step} with a distances of {distances}")
                    # pdb.set_trace()
                    # current_states = torch.where(
                    #     dones.unsqueeze(1),
                    #     current_states,  # Keep the same state if done
                    #     current_states   # Otherwise use the next state
                    # )

            # Report on step distances
            logger.debug(f"Step distances ({epoch}-{batch_idx}): {step_distances}")
            
            # Convert episode data to tensors
            actions = torch.stack(actions)
            _action_magnitude = torch.mean(actions)
            _debug_action = torch.linalg.vector_norm(actions, dim=-1).mean(dim=-1)
            log_probs = torch.stack(log_probs)
            rewards = torch.stack(rewards)
            masks = torch.stack(masks)
            entropies = torch.stack(entropies)

            _step_rewards = torch.mean(rewards, dim=1)
            
            # Calculate returns (discounted rewards)
            # returns = compute_returns(rewards, nav_agent.get_gamma(), masks)
            returns = compute_returns(rewards, nav_agent.get_gamma(), None) # DEBUG: Do we need masks?
            returns = torch.stack(returns)
            
            # Normalize returns for stability
            # returns = torch.sum(returns, dim=0)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            
            # Calculate policy loss
            policy_loss = -(log_probs * returns).mean()
            
            # Add entropy regularization
            entropy_loss = -nav_agent.get_beta() * entropies.mean()
            
            # Total loss
            if use_entropy_loss:
                loss = policy_loss + entropy_loss
            else:
                loss = policy_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(nav_agent.parameters(), 0.5)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_reward += rewards.mean().item()
            epoch_steps += 1

            step_wise_rewards = { f"step_{i}_rewards": _step_rewards[i] for i in range(len(_step_rewards)) }
            step_wise_distances = { f"step_{i}_distance": step_distances[i] for i in range(len(step_distances)) }
            step_wise_actions = { f"step_{i}_action": _debug_action[i] for i in range(len(_debug_action)) }
            training_metrics = {
                'train/loss': loss.item(),
                'train/reward': rewards.mean().item(),
                # 'train/distance': torch.Tensor(step_distances).mean().item(),
                'train/mu': torch.Tensor(mus).mean().item(),
                'train/sigma': torch.Tensor(sigmas).mean().item(),
                # 'train/action_magnitude': _action_magnitude.item(),
                'train/policy_loss': policy_loss.item(),
                'train/entropy_loss': entropy_loss.item(),
                # **step_i_distances,
                # **(step_wise_distances.update(step_wise_rewards)),
                **{
                    **step_wise_distances,
                    **step_wise_rewards,
                    **step_wise_actions,
                }
            }
            loss_metrics_str = { k: f"{v:.4f}" for k, v in training_metrics.items() }
            
            # Update progress bar
            progress_bar.set_postfix(loss_metrics_str)
            wandb_report(training_metrics, epoch, batch_idx)
        
        # Log epoch results
        avg_loss = epoch_loss / epoch_steps
        avg_reward = epoch_reward / epoch_steps
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"navigator_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': nav_agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'reward': avg_reward,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "navigator_final.pt")
    torch.save({
        'model_state_dict': nav_agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    return nav_agent

def main():
    args = arguments()
    set_seeds(args.seed)
    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()

    if args.wandb_on:
        args.wr_name = f"{args.wr_name}_{args.seed}"
        logger.info(
            f"🪄 Initializing Weights and Biases. Under project name {args.wr_project_name} and run name {args.wr_name}"
        )
        wandb_run = wandb.init(
            project=args.wr_project_name,
            name=args.wr_name,
            config=vars(args),
            notes=args.wr_notes,
        )
        for k, v in wandb.config.items():
            setattr(args, k, v)

        print(f"Run URL: {wandb_run.url}")
        print(f"Args: {args}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the Embeddings
    entities_embeddings_tensor = (
        torch.from_numpy(np.load(os.path.join(args.path_embeddings_dir, "entity_embedding.npy")))
    )
    relations_tensor = (
        torch.from_numpy(np.load(os.path.join(args.path_embeddings_dir, "relation_embedding.npy")))
    )
    num_special_embeddings = 1 # Mostly for padding
    num_entities = entities_embeddings_tensor.shape[0] 
    entities_dim = entities_embeddings_tensor.shape[1]
    num_relations = relations_tensor.shape[0]
    relations_dim = relations_tensor.shape[1]
    entity_embeddings = nn.Embedding(num_entities + num_special_embeddings, entities_dim)
    relation_embeddings = nn.Embedding(num_relations + num_special_embeddings, relations_dim)

    # Load pre-trained embeddings and zero out the special embeddings
    entity_embeddings.weight.data[:num_entities] = entities_embeddings_tensor
    entity_embeddings.weight.data[num_entities:] = 0
    relation_embeddings.weight.data[:num_relations] = relations_tensor
    relation_embeddings.weight.data[num_relations:] = 0

    dim_action_relation = relation_embeddings.embedding_dim
    dim_entity = entity_embeddings.embedding_dim

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
    # For the observation dimension, we concatenate the current state and target state
    dim_observation = dim_entity * 2  # current_state + target_state
    
    nav_agent = AttentionContinuousPolicyGradient(
        beta = args.rl_beta,
        gamma = args.rl_gamma,
        dim_action = dim_action_relation,
        dim_hidden = args.rl_dim_hidden,
        dim_observation = dim_observation,
        use_attention = args.use_attention,
        use_tanh_squashing=not args.dont_use_tanh_squashing,
    )

    ########################################
    # Training
    ########################################

    trained_model = train_loop(
        nav_agent,
        train_paths,
        entity_embeddings,
        relation_embeddings,
        epochs = args.epoch,
        steps_in_episode = args.steps_in_episode,
        batch_size = args.training_batch_size,
        learning_rate = args.learning_rate,
        device = device,
        checkpoint_dir = args.checkpoint_dir,
        save_interval = args.save_interval,
        use_entropy_loss = not args.dont_use_entropy_loss,
    )

if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_SIMPLE_MAIN__")
    main()
