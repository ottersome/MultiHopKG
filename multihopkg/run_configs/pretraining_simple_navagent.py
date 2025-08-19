import argparse

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # -- General Training Parameters -- #
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--training_batch_size", type=int, default=32)
    parser.add_argument("--steps_in_episode", type=int, default=8, help="Number of steps in each episode")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/navigator/")
    parser.add_argument("--save_interval", type=int, default=5, help="Save model every N epochs")

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
    parser.add_argument("--rl_gamma", default=0.99, type=float, help="Discount factor for future rewards")
    parser.add_argument("--rl_dim_hidden", default=400, type=int, help="Representation size for policy network")
    parser.add_argument("--env_dim_observation", type=int, default=1000, help="Dimension of observation space (will be overridden)")
    parser.add_argument("--use_navigator", action="store_true", help="Use the navigator model with state-target attention")

    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()
