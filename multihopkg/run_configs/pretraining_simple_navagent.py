import argparse

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # -- General Training Parameters -- #
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--training_batch_size", type=int, default=32)
    # parser.add_argument("--steps_in_episode", type="")

    # -- General Data Parameters -- #
    parser.add_argument("--path_mquake_data", type=str, default="./data/mquake/")
    parser.add_argument("--amount_of_paths", type=int, default=10000)
    parser.add_argument(
        "--path_generation_cache", type=str, default="./cache/random_walks/"
    )
    parser.add_argument("--path_embeddings_dir", type=str, default="./models/graph_embeddings/transE_mquake_dim500/")

    # Paths and in multi-step
    parser.add_argument("--path_batch_size", default=256)
    parser.add_argument("--path_n_hops", default=4)
    parser.add_argument("--path_num_beams", default=4)

    # --- Reinforcement Learning Parameters --- #
    parser.add_argument("--rl_beta", default=0.0, type=float, help="For entropy regularization. Hasnt been used in a while.")
    parser.add_argument("--rl_gamma", default=0.9, type=float, help="For calculating the sum for reinforcement learning")
    parser.add_argument("--rl_dim_hidden", default=400, type=int, help="Reprsentation size fo continuous policy gradient network")
    # TODO: We might seriously refactor the CPG
    # parser.add_argument("--env_dim_observation", default=<++>, type=<++>, help=<++>)

    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()
