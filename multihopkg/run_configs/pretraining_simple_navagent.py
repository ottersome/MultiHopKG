import argparse

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # -- General Training Parameters -- #
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--training_batch_size", type=int, default=128)
    parser.add_argument("--steps_in_episode", type=int, default=4, help="Number of steps in each episode")
    parser.add_argument("--learning_rate", '-lr', type=float, default=5e-2, help="Learning rate for optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/navigator/")
    parser.add_argument("--save_interval", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--random_emebeddings", action="store_true")
    parser.add_argument("--dont_use_tanh_squashing", action="store_true")
    parser.add_argument("--dont_use_entropy_loss", action="store_true")

    # -- General Data Parameters -- #
    parser.add_argument("--path_mquake_data", type=str, default="./data/mquake/")
    parser.add_argument("--amount_of_paths", type=int, default=10000)
    parser.add_argument(
        "--path_generation_cache", type=str, default=".cache/random_walks/"
    )
    parser.add_argument("--path_embeddings_dir", type=str, default="./models/graph_embeddings/transE_mquake_dim500/")

    # Paths and in multi-step
    parser.add_argument("--path_batch_size", type=int, default=256)
    parser.add_argument("--path_n_hops", type=int, default=4)
    parser.add_argument("--path_num_beams", type=int, default=4)

    # --- Reinforcement Learning Parameters --- #
    parser.add_argument("--rl_beta", default=0.01, type=float, help="For entropy regularization")
    parser.add_argument("--rl_gamma", default=0.93, type=float, help="Discount factor for future rewards")
    parser.add_argument("--rl_dim_hidden", default=400, type=int, help="Representation size for policy network")
    parser.add_argument("--env_dim_observation", type=int, default=1000, help="Dimension of observation space (will be overridden)")
    parser.add_argument("--use_navigator", action="store_true", help="Use the navigator model with state-target attention")

    parser.add_argument("--debug", "-d", action="store_true")

    # -- Wandb Parameters -- #
    parser.add_argument("--wandb_on", "-w", action="store_true")
    parser.add_argument("--wr_name", type=str, default="test")
    parser.add_argument("--wr_notes", type=str, default="Still just trying to make it learn")
    parser.add_argument("--wr_project_name", type=str, default="simple_navagent")
    return parser.parse_args()
